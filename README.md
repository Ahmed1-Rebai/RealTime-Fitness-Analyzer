# RealTime-Fitness-Analyzer

>A lightweight real-time exercise analyzer using MediaPipe Pose and OpenCV. Tracks repetitions, gives form feedback (visual + optional voice), supports squats and push-ups, and can process live camera or video files with annotated output.

---

## Features

- Real-time pose estimation using MediaPipe Pose.
- Rep counting for squats and push-ups (heuristic-based).
- Form feedback displayed on-screen and optional voice guidance (pyttsx3).
- CLI to select camera or video input, disable TTS, force exercise type, and save annotated output.
- Simple, readable visual overlays: angles, progress bar, tips, and stats.

## Quick Start

Requirements

- Python 3.8+ (3.10+ recommended)
- A working webcam (for live mode) or a recorded video file
- Install dependencies from `requirements.txt` (see below)

Create a virtual environment and install:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Run the analyzer (default: camera 0 with TTS enabled):

```bash
python pose_detector.py
```

Process a video file, disable TTS and save annotated output:

```bash
python pose_detector.py --source path/to/video.mp4 --no-tts --output annotated.mp4
```

Force the exercise type (skip auto-detection):

```bash
python pose_detector.py --exercise pushup
```

CLI options

- `--source` : Camera index (default `0`) or path to a video file.
- `--no-tts` : Disable text-to-speech prompts.
- `--output` : Path to write an annotated video (MP4).
- `--exercise` : `auto`, `squat`, or `pushup` to force or auto-detect.

## How it works (short)

1. MediaPipe detects landmarks every frame.
2. A small heuristic determines whether the user is performing a squat or a push-up.
3. Angles (knee/elbow/back) are computed for rep detection and form checks.
4. Visual overlays and optional voice guidance provide immediate feedback.

## Tips for Best Results

- Ensure the camera captures the whole body (for squats) or the full torso (for push-ups).
- Use good lighting and avoid cluttered backgrounds for more stable pose detection.
- For push-ups, place the camera on the side or slightly elevated to see elbow flexion.

## Development Notes

- Main script: `pose_detector.py`
- You can adjust thresholds and heuristics in the script (angle cutoffs, cooldowns).
- The project includes a simple heuristic classifier; for production use, consider training a small classifier on labeled clips.

## Next Improvements (ideas)

- Add logging/verbosity flags and structured logs.
- Improve classifier accuracy (ML-based) and add more exercises.
- Add unit tests and CI checks; update `requirements.txt` pinning versions.

## License

MIT License — feel free to fork and adapt for your own projects.

---

If you want, I can also update `requirements.txt` with pinned versions, add a short CONTRIBUTING section, or create a small demo GIF for the README. Which would you like next?
# RealTime-Fitness-Analyzer

Petit outil de détection de posture et d'analyse de squats en temps réel basé sur MediaPipe et OpenCV.

## Description

Ce script capture la webcam, détecte les landmarks de posture via `mediapipe`, calcule des angles (genoux, dos), affiche des indications visuelles et fournit un retour vocal avec `pyttsx3`.

Fichier principal: [pose_detector.py](pose_detector.py)

## Prérequis

- Python 3.8+
- Webcam fonctionnelle et permissions d'accès

## Installation

1. Créez et activez un environnement virtuel (recommandé):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Installez les dépendances:

```bash
pip install -r requirements.txt
```

## Exécution

Lancez le script depuis la racine du projet:

```bash
python pose_detector.py
```

Appuyez sur `q` pour quitter la fenêtre.

## Notes et dépannage

- Sur Windows, `mediapipe` peut nécessiter une version spécifique de `protobuf`; suivez les erreurs affichées pour installer les versions recommandées.
- Si la synthèse vocale ne fonctionne pas, vérifiez que `pyttsx3` est correctement installé et que votre système a un moteur TTS (SAPI5 sur Windows).
- Si la caméra ne s'ouvre pas, testez d'autres indices de caméra (par ex. `cv2.VideoCapture(1)`) ou vérifiez les permissions de l'application.

## Licence

Usage personnel / éducatif. Adapter selon besoin.
