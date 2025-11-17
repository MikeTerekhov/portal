import pandas as pd
import numpy as np

def verify_merge_integrity(final_csv, sym_csv, meta_csv, verbose=True):
    """Verify that the merged CSV contains all data from symmetry and metadata sources."""

    df_final = pd.read_csv(final_csv)
    df_sym   = pd.read_csv(sym_csv)
    df_meta  = pd.read_csv(meta_csv)

    # --- Normalize IDs like in the main pipeline ---
    def normalize_id(s):
        return s.astype(str).str.strip().str.split("_", n=1).str[0].str.lstrip("0")

    # Find the participant ID column in meta
    meta_id_col = next((c for c in df_meta.columns if c.startswith("Participant ID")), None)
    if meta_id_col is None:
        raise KeyError("Could not find 'Participant ID' column in metadata file.")

    df_meta["subject_id"] = normalize_id(df_meta[meta_id_col])
    df_sym["subject_id"]  = normalize_id(df_sym["subject_id"])
    df_final["subject_id"] = normalize_id(df_final["subject_id"])

    results = {}

    # --------------------------------------------------
    # 1. Subject coverage check
    # --------------------------------------------------
    meta_ids = set(df_meta["subject_id"])
    final_ids = set(df_final["subject_id"])
    sym_ids = set(df_sym["subject_id"])

    missing_from_final = meta_ids - final_ids
    missing_from_sym   = sym_ids - final_ids

    results["missing_subjects_from_final_vs_meta"] = missing_from_final
    results["missing_subjects_from_final_vs_sym"]  = missing_from_sym

    # --------------------------------------------------
    # 2. Event coverage check (Event Name)
    # --------------------------------------------------
    if "Event Name" in df_meta.columns and "Event Name" in df_final.columns:
        meta_events = df_meta[["subject_id", "Event Name"]].drop_duplicates()
        final_events = df_final[["subject_id", "Event Name"]].drop_duplicates()

        merged_events = pd.merge(meta_events, final_events, on=["subject_id", "Event Name"], how="left", indicator=True)
        missing_events = merged_events.loc[merged_events["_merge"] == "left_only"]
        results["missing_events"] = missing_events

    # --------------------------------------------------
    # 3. Key field completeness check
    # --------------------------------------------------
    # Select a few fields that should exist and have minimal NaN if merged correctly
    important_fields = [
        "dominant_leg", "injured_leg_acl", "right_force", "left_force",
        "injury_symmetry", "injury_deficit"
    ]

    existing_fields = [f for f in important_fields if f in df_final.columns]
    completeness = df_final[existing_fields].notna().mean().round(3).to_dict()
    results["field_completeness_fraction"] = completeness

    # --------------------------------------------------
    # 4. Sanity check: left/right/injury forces preserved
    # --------------------------------------------------
    missing_forces = df_final.loc[df_final[["left_force", "right_force"]].isna().any(axis=1)]
    results["rows_missing_force_data"] = len(missing_forces)

    # --------------------------------------------------
    # 5. Cross-reference spot check: 
    #    (verify forces match between symmetry CSV and merged file)
    # --------------------------------------------------
    sym_subset = df_sym[["subject_id", "left_force", "right_force"]].dropna()
    merged_check = pd.merge(
        sym_subset, df_final[["subject_id", "left_force", "right_force"]],
        on="subject_id", suffixes=("_sym", "_final")
    )
    force_mismatch = (
        (np.abs(merged_check["left_force_sym"] - merged_check["left_force_final"]) > 1e-6) |
        (np.abs(merged_check["right_force_sym"] - merged_check["right_force_final"]) > 1e-6)
    )
    results["force_value_mismatch_count"] = int(force_mismatch.sum())

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------
    if verbose:
        print("\n=== MERGE INTEGRITY REPORT ===")
        print(f"Subjects in metadata: {len(meta_ids)}")
        print(f"Subjects in symmetry: {len(sym_ids)}")
        print(f"Subjects in final:    {len(final_ids)}")
        print(f"Missing from final vs metadata: {missing_from_final}")
        print(f"Missing from final vs symmetry: {missing_from_sym}")
        print(f"\nEvent coverage: {len(results.get('missing_events', []))} missing event rows")
        print("\nField completeness:")
        for k, v in completeness.items():
            print(f"  {k:20s}: {v*100:.1f}% non-null")
        print(f"\nRows missing any force data: {results['rows_missing_force_data']}")
        print(f"Force mismatch count (should be 0): {results['force_value_mismatch_count']}")
        print("==============================\n")

    return results


if __name__ == "__main__":
    # Example usage (replace with your file paths)
    results = verify_merge_integrity(
        final_csv="./merged_subject_rightjoin_with_metrics_FILTERED.csv",
        sym_csv="./300_natalie_symmetry_report_2025-10-20.csv",
        meta_csv="./AssessmentOfNeuromus_DATA_LABELS_2025-11-08_1518-1.csv"
    )
