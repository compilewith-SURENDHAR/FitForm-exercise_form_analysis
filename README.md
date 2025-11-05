# FITFORM: Posture Tracking and Machine Learning for corrections to improve posture and form while exercising. ( squat only)


Our project is an AI-based Personalised Exercise Feedback Assistant: an algorithm that views your exercise posture in real time and tells what you're getting right, and what you're getting wrong! 


# to run

To run the app, first run `pip install -r requirements.txt` and then run `python app_squat.py`

The app will be available at http://localhost:8050

# overall logic behind

             ┌───────────────────────┐
             │   Raw Videos (.mp4)   │
             │   ./data/processed    │
             └───────────┬───────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │   generate_input_vector.py               │
    │ - Uses Mediapipe Pose (sp.get_params)    │
    │ - Extracts 5 squat parameters per frame  │
    │ - Writes → input_vectors.csv             │
    └─────────────────────────────────────────┘

                         │
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │   labels.csv (manual annotations)        │
    │ - Trainer marks errors (c, k, h, r, x,i) │
    └───────────────────┬─────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │   generate_output_vector.py              │
    │ - Reads labels.csv                       │
    │ - Expands to frame-wise one-hot labels   │
    │ - Writes → output_vectors.csv            │
    └─────────────────────────────────────────┘

                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │   dataprocessing/create_data_matrices.py │
    │ - get_data(): loads input/output CSVs    │
    │ - Returns numpy arrays (X, y)            │
    └─────────────────────────────────────────┘

                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │   tfmodel.py                             │
    │ - Calls get_data()                       │
    │ - Splits train/test sets                 │
    │ - Defines + compiles model               │
    │ - Trains (.fit → hist, hist.history)     │
    │ - Saves model → working_model.keras      │
    │ - Plots accuracy/loss curves             │
    └─────────────────────────────────────────┘

                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │   app_squat.py                           │
    │ - Loads working_model.keras              │
    │ - Captures webcam feed (OpenCV)          │
    │ - Uses sp.get_params() on each frame     │
    │ - Model predicts squat correctness       │
    │ - Displays live feedback in browser      │
    └─────────────────────────────────────────┘


