import os
import pandas as pd
import numpy as np
from scipy import stats
from modules.session_logger import get_all_sessions_df, EXPORT_DIR


def analyze_stress_reduction(df: pd.DataFrame = None) -> dict:
    """
    Performs paired t-test and computes Cohen's d effect size for Pre vs Post PSS-4 stress scores.
    """
    if df is None:
        df = get_all_sessions_df()
        
    if df.empty:
        return {
            "status": "empty",
            "message": "No database entries found yet. Complete a chat or PSS-4 test to begin!",
            "sample_size": 0,
            "mean_change": 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "pre_mean": 0.0,
            "post_mean": 0.0
        }
        
    # Drop rows missing pre_pss or post_pss
    valid_df = df.dropna(subset=["pre_pss", "post_pss"]).copy()
    
    # Select unique session_ids (using latest row per session_id)
    unique_sessions = valid_df.groupby("session_id").last().reset_index()
    
    n = len(unique_sessions)
    if n < 2:
        return {
            "status": "partial",
            "message": f"Recorded N={n} complete sessions. Minimum 2 required for statistical paired t-test.",
            "sample_size": n,
            "mean_change": 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "pre_mean": 0.0,
            "post_mean": 0.0
        }
        
    pre_scores = unique_sessions["pre_pss"].values
    post_scores = unique_sessions["post_pss"].values
    diffs = post_scores - pre_scores  # Negative value indicates stress reduction
    
    pre_mean, pre_sd = float(np.mean(pre_scores)), float(np.std(pre_scores, ddof=1))
    post_mean, post_sd = float(np.mean(post_scores)), float(np.std(post_scores, ddof=1))
    diff_mean, diff_sd = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(pre_scores, post_scores)
    
    # Cohen's d effect size
    cohen_d = abs(diff_mean) / diff_sd if diff_sd != 0 else 0.0
    
    results = {
        "status": "success",
        "sample_size": n,
        "pre_mean": round(pre_mean, 2),
        "pre_sd": round(pre_sd, 2),
        "post_mean": round(post_mean, 2),
        "post_sd": round(post_sd, 2),
        "mean_change": round(diff_mean, 2),
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_value), 5),
        "is_statistically_significant": bool(p_value < 0.05),
        "cohens_d": round(float(cohen_d), 3)
    }
    
    # Export results CSV for paper
    res_df = pd.DataFrame([results])
    res_path = os.path.join(EXPORT_DIR, "stress_reduction_analysis.csv")
    res_df.to_csv(res_path, index=False)
    results["csv_path"] = res_path
    
    return results


if __name__ == "__main__":
    res = analyze_stress_reduction()
    print("=== PRE-POST PSS-4 STRESS REDUCTION ANALYSIS ===")
    if res.get("status") == "success":
        print(f"Sample Size (N): {res['sample_size']}")
        print(f"Pre-Session PSS-4 Mean: {res['pre_mean']} (SD = {res['pre_sd']})")
        print(f"Post-Session PSS-4 Mean: {res['post_mean']} (SD = {res['post_sd']})")
        print(f"Mean Change: {res['mean_change']}")
        print(f"Paired t-statistic: {res['t_statistic']}")
        print(f"p-value: {res['p_value']} (Significant: {res['is_statistically_significant']})")
        print(f"Cohen's d Effect Size: {res['cohens_d']}")
    else:
        print(res.get("message"))
