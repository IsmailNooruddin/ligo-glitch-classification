"""
Experiment: Cross-Detector Testing (Option B)
Main extension required by instructor:
1. Train on Hanford (H1) -> Test on Livingston (L1)
2. Train on Livingston (L1) -> Test on Hanford (H1)
3. Within-detector controls: H1 -> H1, L1 -> L1
Repeated over 3-5 random seeds to report Macro-F1 (mean ± std dev).
"""

def main():
    print("Running Cross-Detector Transfer Experiment (Option B)...")
    # TODO: Implement cross-detector pipeline across seeds

if __name__ == "__main__":
    main()
