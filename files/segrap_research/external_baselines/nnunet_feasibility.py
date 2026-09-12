"""
External Baseline Study — TRUE nnU-Net (nnunetv2) framework feasibility report.

This script does NOT run any training. It only checks whether nnunetv2 is
installed and writes a text report assessing whether/how a true nnU-Net
comparison could be added later, distinct from the MONAI-UNet "3D U-Net
baseline" used elsewhere in this study (see README.md's naming section —
MONAI UNet != true nnU-Net).

Usage:
    python external_baselines/nnunet_feasibility.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_nnunetv2_installed():
    try:
        import nnunetv2
        return True, getattr(nnunetv2, '__version__', 'unknown')
    except ImportError as e:
        return False, str(e)


def build_report() -> str:
    installed, info = check_nnunetv2_installed()

    lines = []
    lines.append(f"nnunetv2_installed: {'yes' if installed else 'no'}")
    lines.append(f"nnunetv2_version: {info if installed else 'N/A'}")
    lines.append("")
    lines.append("recommended_setting: binary_crop_nnunet")
    lines.append("")
    lines.append("notes_on_recommendation:")
    lines.append(
        "  Two ways exist to add a 'true' nnU-Net number, and they are NOT"
        " interchangeable:"
    )
    lines.append("")
    lines.append(
        "  Option A — binary_crop_nnunet (RECOMMENDED if a true-nnU-Net"
        " number is wanted at all):"
    )
    lines.append(
        "    Feed nnU-Net the EXACT SAME pre-cropped, pre-resized thin-wall"
        " OAR binary crops this study already uses (same case_id/oar_name"
        " split as data/segrap_dataset.py, same target_size, same"
        " normalisation). Task stays binary fg/bg per OAR crop, directly"
        " comparable to Stage5C / the other external baselines. Trade-off:"
        " this bypasses most of nnU-Net's own value proposition (its"
        " automated spacing/intensity/patch-size planning), so what's"
        " actually being measured is closer to 'nnU-Net's network +"
        " trainer loop on pre-made crops' rather than a full nnU-Net run."
    )
    lines.append("")
    lines.append("  Option B — whole_volume_reference (qualitative only):")
    lines.append(
        "    Let nnU-Net run its OWN standard pipeline on whole CT volumes"
        " with all 10 thin-wall OARs as an 11-class (bg + 10) segmentation"
        " task. This is what nnU-Net is actually designed for and gives the"
        " most representative 'true nnU-Net' number, but the TASK no longer"
        " matches Stage5C's per-OAR binary-crop setting (different label"
        " cardinality, different effective training-step content, different"
        " loss balancing across classes of very different size). Usable"
        " only as a qualitative external reference, never as a like-for-like"
        " ablation entry in the same table as Stage5C/3D-U-Net/SegResNet."
    )
    lines.append("")
    lines.append("required_conversion_scripts:")
    lines.append(
        "  - export_to_nnunet_format.py: write each cached (image, label)"
        " pair from data/segrap_dataset.py's existing train/val/test split"
        " into nnU-Net's raw-data folder layout (imagesTr/labelsTr/imagesTs"
        " + dataset.json), one nnU-Net 'case' per OAR-crop sample (NOT per"
        " patient) to match Option A's framing."
    )
    lines.append(
        "  - a dataset.json generator matching nnU-Net v2's schema"
        " (channel_names, labels={'background':0,'oar':1}, numTraining,"
        " file_ending)."
    )
    lines.append(
        "  - an nnUNetv2_plan_and_preprocess + nnUNetv2_train invocation"
        " wrapper, OR a custom nnUNetTrainer subclass if the existing fixed"
        " crop/split must be enforced exactly -- nnU-Net's own preprocessor"
        " resamples/crops by its own heuristics by default, which would"
        " silently violate the 'do not change crop/resize/preprocessing'"
        " constraint unless explicitly disabled."
    )
    lines.append("")
    lines.append("estimated_risk:")
    lines.append(
        "  Medium-high for Option A specifically on the FAIRNESS dimension"
        " (not the tooling dimension). nnU-Net's standard workflow assumes"
        " it owns the full preprocessing pipeline; forcing it to consume"
        " already-cropped/resized/normalised 128^3 arrays as-is means most"
        " of nnU-Net's automation is bypassed, so the comparison would be"
        " 'nnU-Net's architecture choices' rather than 'nnU-Net the"
        " framework' in the way most readers would assume from the name."
    )
    lines.append(
        "  Tooling/installation risk is LOW: nnunetv2 installed cleanly in"
        " this environment via `pip install nnunetv2` with no build errors"
        " (pulled in batchgenerators, batchgeneratorsv2, acvl-utils,"
        " dynamic-network-architectures as dependencies, all pure-Python/"
        " no compilation required)."
    )
    lines.append("")
    lines.append("notes:")
    lines.append(
        "  - Do NOT conflate the MONAI UNet '3D U-Net baseline' used"
        " elsewhere in this study with true nnU-Net. They are different"
        " things -- see external_baselines/README.md's naming section."
    )
    lines.append(
        "  - This report runs no training and starts no nnU-Net job. The"
        " decision of whether to pursue Option A, Option B, or neither is"
        " left to the user; this file only lays out what each would"
        " require and what it would (and would not) prove."
    )

    return '\n'.join(lines)


def main():
    report = build_report()
    print(report)

    out_dir = os.path.join('.', 'results', 'external_baselines')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'nnunet_feasibility_report.txt')
    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    main()
