import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from modules.session_logger import get_all_sessions_df, EXPORT_DIR


def evaluate_emotion_accuracy(df: pd.DataFrame = None) -> dict:
    """
    Computes classification metrics comparing detected_emotion against self_reported_emotion.
    
    Returns:
        dict: Summary statistics including overall accuracy, precision, recall, F1, and detailed report text.
    """
    if df is None:
        df = get_all_sessions_df()
        
    if df.empty:
        return {
            "status": "empty",
            "message": "No session data available in database yet.",
            "sample_size": 0,
            "overall_accuracy": 0.0,
            "report_string": "No sessions recorded yet."
        }
        
    # Filter rows with both detected and self-reported emotions
    valid_df = df.dropna(subset=["detected_emotion", "self_emotion"]).copy()
    
    if len(valid_df) == 0:
        return {
            "status": "partial",
            "message": "No session records with self-reported emotions found yet.",
            "sample_size": 0,
            "overall_accuracy": 0.0,
            "report_string": "No self-reported emotion records yet."
        }
        
    y_true = valid_df["self_emotion"].str.lower().str.strip()
    y_pred = valid_df["detected_emotion"].str.lower().str.strip()
    
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, zero_division=0)
    
    # Save metrics CSV for research paper
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(EXPORT_DIR, "emotion_accuracy_report.csv")
    report_df.to_csv(report_path)
    
    return {
        "status": "success",
        "sample_size": len(valid_df),
        "overall_accuracy": round(accuracy * 100, 2),
        "report_string": report_str,
        "report_csv_path": report_path
    }


if __name__ == "__main__":
    res = evaluate_emotion_accuracy()
    print("=== EMOTION DETECTION ACCURACY REPORT ===")
    if res.get("status") == "success":
        print(f"Sample Size: {res['sample_size']}")
        print(f"Overall Accuracy: {res['overall_accuracy']}%")
        print("\nClassification Report:\n" + res["report_string"])
    else:
        print(res.get("message"))
