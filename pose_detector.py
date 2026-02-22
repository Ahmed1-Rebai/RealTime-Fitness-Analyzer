import argparse
import os
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import pyttsx3

# ----------------- Constants ----------------- #
COLORS = {
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'black': (0, 0, 0)
}

NUMBER_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", 
                "Six", "Seven", "Eight", "Nine", "Ten"]

# ----------------- Setup MediaPipe Pose ----------------- #
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Custom drawing specs
custom_drawing_spec = mp_drawing.DrawingSpec(
    color=COLORS['green'], thickness=2, circle_radius=2)
custom_connections_spec = mp_drawing.DrawingSpec(
    color=COLORS['white'], thickness=2)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Slower speech rate for clarity

# ----------------- Voice Feedback System ----------------- #
class VoiceFeedback:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.last_feedback_time = 0
        self.cooldown = 1.5  # seconds between voice feedbacks
        self.last_message = ""
        self.last_rep_count = 0
        self.correction_cooldown = 2  # seconds between correction feedbacks
        
    def give_feedback(self, message, is_rep_count=False):
        if not self.enabled:
            return
        current_time = time.time()
        
        if is_rep_count:
            # Always announce rep counts immediately
            try:
                engine.say(message)
                engine.runAndWait()
                self.last_rep_count = int(message.split()[-1]) if message.split()[-1].isdigit() else 0
            except Exception as e:
                print(f"Voice feedback error: {e}")
        else:
            # For correction feedback, use cooldown
            if (current_time - self.last_feedback_time > self.cooldown and 
                message != self.last_message and
                current_time - self.last_feedback_time > self.correction_cooldown):
                try:
                    engine.say(message)
                    engine.runAndWait()
                    self.last_feedback_time = current_time
                    self.last_message = message
                except Exception as e:
                    print(f"Voice feedback error: {e}")

voice_feedback = VoiceFeedback()


def detect_exercise(landmarks, image_shape):
    """Heuristic to distinguish between squat and push-up using landmark bounding box and shoulder/hip relation.
    Returns 'squat', 'pushup', or 'unknown'.
    """
    try:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox_h = max_y - min_y
        bbox_w = max_x - min_x
        aspect = bbox_h / (bbox_w + 1e-6)

        left_sh_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_sh_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        shoulder_y = (left_sh_y + right_sh_y) / 2.0
        hip_y = (left_hip_y + right_hip_y) / 2.0
        y_diff = abs(shoulder_y - hip_y)

        # Heuristics:
        # - Squat: tall vertical bbox and shoulder-hip vertical separation
        # - Push-up: flatter bbox (horizontal) and shoulder~hip y positions
        if aspect > 1.05 and y_diff > 0.05:
            return 'squat'
        if aspect < 0.95 or y_diff < 0.05:
            return 'pushup'
        return 'unknown'
    except Exception:
        return 'unknown'

# ----------------- Angle Calculation Function ----------------- #
def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# ----------------- Posture Analysis Functions ----------------- #
def check_back_angle(shoulder, hip, knee, threshold=150):
    """Check if back is straight enough"""
    angle = calculate_angle(shoulder, hip, knee)
    return angle >= threshold, angle

def check_knee_alignment(hip, knee, ankle, threshold=10):
    """Check if knees are tracking properly over toes"""
    # Convert to numpy arrays
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)
    
    # Calculate vectors
    hip_knee = knee - hip
    knee_ankle = ankle - knee
    
    # Calculate angle between the two vectors in degrees
    angle = np.degrees(np.arccos(np.dot(hip_knee, knee_ankle) / 
                  (np.linalg.norm(hip_knee) * np.linalg.norm(knee_ankle))))
    
    return angle <= threshold, angle

# ----------------- Visualization Functions ----------------- #
def draw_angle(image, angle, position, color=COLORS['white']):
    """Draw angle text on image"""
    cv2.putText(image, f"{int(angle)}°", 
               (int(position[0]), int(position[1])),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_feedback(image, text, position, color=COLORS['green'], size=1):
    """Draw feedback text on image"""
    cv2.putText(image, text, 
               (int(position[0]), int(position[1])),
               cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

def draw_progress_bar(image, progress, position, size=(200, 20), color=COLORS['green']):
    """Draw a progress bar showing squat depth"""
    width, height = size
    x, y = int(position[0]), int(position[1])
    
    # Background
    cv2.rectangle(image, (x, y), (x + width, y + height), COLORS['white'], 1)
    
    # Progress
    fill_width = int(width * progress)
    cv2.rectangle(image, (x, y), (x + fill_width, y + height), color, -1)
    
    # Text
    cv2.putText(image, "Depth", (x, y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)

def draw_tip_box(image, tips, position, width=300):
    """Draw a box with multiple tips"""
    x, y = int(position[0]), int(position[1])
    line_height = 30
    box_height = len(tips) * line_height + 20
    
    # Draw box background
    cv2.rectangle(image, (x, y), (x + width, y + box_height), COLORS['black'], -1)
    cv2.rectangle(image, (x, y), (x + width, y + box_height), COLORS['yellow'], 2)
    
    # Draw title
    cv2.putText(image, "FORM TIPS", (x + 10, y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['yellow'], 2)
    
    # Draw each tip
    for i, tip in enumerate(tips):
        cv2.putText(image, f"• {tip}", (x + 10, y + 25 + (i+1)*line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)

# ----------------- Main Program ----------------- #
def main():
    parser = argparse.ArgumentParser(description="RealTime Fitness Analyzer")
    parser.add_argument('--source', default='0', help='Camera index (0) or video file path')
    parser.add_argument('--no-tts', action='store_true', help='Disable text-to-speech')
    parser.add_argument('--output', help='Path to save annotated output video (mp4)')
    parser.add_argument('--exercise', choices=['auto', 'squat', 'pushup'], default='auto', help='Force exercise type or auto-detect')
    args = parser.parse_args()

    # Initialize variables
    counter = 0
    stage = "up"  # Start in the "up" position
    last_rep_time = time.time()
    rep_times = deque(maxlen=5)  # Store last 5 rep times for pace calculation
    feedback = ""
    feedback_color = COLORS['green']
    last_rep_end_time = 0
    show_tips = False
    tips_duration = 3  # seconds to show tips after rep
    last_voice_feedback = ""
    rep_count_announced = False

    # Configure voice feedback per CLI
    voice_feedback.enabled = not args.no_tts

    # Open source (camera index or video file)
    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)

    # Set camera resolution for camera sources
    try:
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass

    # Prepare optional video writer (will match the combined frame size)
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = cap.get(cv2.CAP_PROP_FPS)
        if out_fps == 0 or np.isnan(out_fps):
            out_fps = 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            width, height = 1280, 720
        writer = cv2.VideoWriter(args.output, fourcc, out_fps, (width + 300, height))
    
    # General squat tips (shown between reps)
    general_tips = [
        "Keep chest up and core engaged",
        "Push through your heels when standing",
        "Keep knees aligned with toes",
        "Go down until thighs parallel to floor",
        "Maintain neutral spine position"
    ]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process image
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create a black background for stats
        stats_bg = np.zeros((image.shape[0], 300, 3), dtype=np.uint8)
        
        if results.pose_landmarks:
            # Draw pose landmarks with custom style
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=custom_drawing_spec,
                connection_drawing_spec=custom_connections_spec)
            
            landmarks = results.pose_landmarks.landmark
            
            try:
                # Get left side landmarks
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Get right side landmarks
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angles
                back_angle = calculate_angle(l_shoulder, l_hip, l_knee)
                left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

                # Determine exercise type
                detected_ex = args.exercise if args.exercise != 'auto' else detect_exercise(landmarks, image.shape)

                # Depth progress mapping per exercise
                if detected_ex == 'pushup':
                    depth_progress = min(1.0, max(0.0, 1 - (avg_elbow_angle / 180)))
                else:
                    depth_progress = min(1.0, max(0.0, 1 - (avg_knee_angle / 180)))

                # Check posture
                back_straight, _ = check_back_angle(l_shoulder, l_hip, l_knee)
                knee_aligned_left, _ = check_knee_alignment(l_hip, l_knee, l_ankle)
                knee_aligned_right, _ = check_knee_alignment(r_hip, r_knee, r_ankle)

                # REP COUNTER LOGIC (branch by exercise)
                if detected_ex == 'pushup':
                    # push-up: use elbow angle
                    if avg_elbow_angle > 160:
                        if stage == 'down':
                            counter += 1
                            rep_times.append(time.time() - last_rep_time)
                            last_rep_time = time.time()
                            last_rep_end_time = time.time()
                            show_tips = True
                            if counter < len(NUMBER_WORDS):
                                rep_word = NUMBER_WORDS[counter]
                            else:
                                rep_word = str(counter)
                            voice_feedback.give_feedback(f"Rep {rep_word}", is_rep_count=True)
                            rep_count_announced = True
                        stage = 'up'
                    elif avg_elbow_angle < 90 and stage == 'up':
                        stage = 'down'
                        rep_count_announced = False
                else:
                    # default: squat logic (knee angle)
                    if avg_knee_angle > 160:
                        if stage == 'down':
                            counter += 1
                            rep_times.append(time.time() - last_rep_time)
                            last_rep_time = time.time()
                            last_rep_end_time = time.time()
                            show_tips = True
                            if counter < len(NUMBER_WORDS):
                                rep_word = NUMBER_WORDS[counter]
                            else:
                                rep_word = str(counter)
                            voice_feedback.give_feedback(f"Rep {rep_word}", is_rep_count=True)
                            rep_count_announced = True
                        stage = 'up'
                    elif avg_knee_angle < 100 and stage == 'up':
                        stage = 'down'
                        rep_count_announced = False

                # Calculate pace (reps per minute)
                if len(rep_times) > 1:
                    avg_rep_time = np.mean(rep_times)
                    pace = int(60 / avg_rep_time)
                else:
                    pace = 0

                # FORM FEEDBACK
                feedback = "✅ Good form"
                feedback_color = COLORS['green']
                specific_tips = []
                voice_message = ""

                # Back / posture feedback
                if not back_straight and stage == "down":
                    feedback = "⚠️ Keep back straight"
                    feedback_color = COLORS['red']
                    specific_tips.append("Chest up - don't lean forward")
                    specific_tips.append("Engage your core muscles")
                    if not rep_count_announced:
                        voice_message = "Keep your chest up and back straight"

                # Knee alignment feedback (mostly for squat)
                if (not knee_aligned_left or not knee_aligned_right) and stage == "down" and detected_ex != 'pushup':
                    feedback = "⚠️ Watch knee alignment"
                    feedback_color = COLORS['red']
                    specific_tips.append("Keep knees over toes")
                    specific_tips.append("Don't let knees cave inward")
                    if not voice_message and not rep_count_announced:
                        voice_message = "Keep your knees aligned with your toes"

                # Depth feedback
                if detected_ex != 'pushup' and avg_knee_angle < 70 and stage == "down":
                    feedback = "⚠️ Reduce depth slightly"
                    feedback_color = COLORS['yellow']
                    specific_tips.append("Don't go below parallel")
                    specific_tips.append("Maintain control throughout")
                    if not voice_message and not rep_count_announced:
                        voice_message = "Don't squat too deep, stop at parallel"

                # Push-up specific feedback (torso too low/high)
                if detected_ex == 'pushup' and stage == 'down':
                    # ensure torso stays relatively flat
                    # reuse detect_exercise heuristics for simple check
                    if detect_exercise(landmarks, image.shape) != 'pushup':
                        feedback = "⚠️ Body alignment"
                        feedback_color = COLORS['yellow']
                        specific_tips.append("Keep your body in a straight line")
                        if not voice_message and not rep_count_announced:
                            voice_message = "Keep your body straight during push-ups"

                # Give correction feedback if needed (and not during rep count announcement)
                if voice_message and not rep_count_announced:
                    voice_feedback.give_feedback(voice_message)
                    last_voice_feedback = voice_message

                # VISUALIZATION
                # Draw angles on joints
                if detected_ex == 'pushup':
                    draw_angle(image, left_elbow_angle, (l_elbow[0] * image.shape[1], l_elbow[1] * image.shape[0]))
                    draw_angle(image, right_elbow_angle, (r_elbow[0] * image.shape[1], r_elbow[1] * image.shape[0]))
                else:
                    draw_angle(image, left_knee_angle, (l_knee[0] * image.shape[1], l_knee[1] * image.shape[0]))
                    draw_angle(image, right_knee_angle, (r_knee[0] * image.shape[1], r_knee[1] * image.shape[0]))
                draw_angle(image, back_angle, (l_hip[0] * image.shape[1], l_hip[1] * image.shape[0]))

                # Draw depth progress bar
                draw_progress_bar(image, depth_progress, (50, 100))

                # Draw feedback
                draw_feedback(image, feedback, (50, 50), feedback_color, 1.2)

                # Draw stats on black background
                cv2.putText(stats_bg, f"Exercise: {detected_ex}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 1)
                cv2.putText(stats_bg, f"Reps: {counter}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['green'], 2)
                cv2.putText(stats_bg, f"Pace: {pace}/min", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['yellow'], 2)
                cv2.putText(stats_bg, f"Left Knee: {int(left_knee_angle)}°", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 1)
                cv2.putText(stats_bg, f"Right Knee: {int(right_knee_angle)}°", (20, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 1)
                cv2.putText(stats_bg, f"Back Angle: {int(back_angle)}°", (20, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 1)

                # Show tips after each rep or when form is poor
                if show_tips or len(specific_tips) > 0:
                    tips_to_show = specific_tips if len(specific_tips) > 0 else general_tips
                    draw_tip_box(image, tips_to_show, (image.shape[1] - 350, 50))
                    # Reset tips after duration
                    if time.time() - last_rep_end_time > tips_duration:
                        show_tips = False

            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Combine main image with stats panel
        stats_bg_resized = cv2.resize(stats_bg, (300, image.shape[0]))
        combined = np.hstack((stats_bg_resized, image))
        if writer is not None:
            try:
                writer.write(combined)
            except Exception:
                pass
        
        # Show FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(combined, f"FPS: {int(fps)}", (10, combined.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        
        cv2.imshow('Squat Analysis with Voice Feedback', combined)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()