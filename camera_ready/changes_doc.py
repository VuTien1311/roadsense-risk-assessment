"""Write CAMERA_READY_CHANGES.md: reviewer comment -> change -> where, with numbers.

Numbers are read from the generated LaTeX macros (paper/numbers.tex), so the
document and the manuscript can never disagree.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "camera_ready"


def load_macros():
    text = (OUT / "paper" / "numbers.tex").read_text(encoding="utf-8")
    return {name: value for name, value in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text)}


def main():
    m = load_macros()

    def v(name):
        return m.get(name, "??").replace("\\textbf{??}", "??").replace("{,}", ",")

    extra = []
    if "TestTFourImages" in m:
        line = (f"* Hardware: {v('TestTFourImages')} of the 9,500 test images ({v('TestTFourPct')} %) and every 1k "
                "perturbation run were processed on an NVIDIA T4 with the same pinned library versions")
        if "CrossHWFlipFull" in m:
            changed = ("none" if float(v("CrossHWFlipFull")) == 0 else f"{v('CrossHWFlipFull')} %")
            line += f"; rerunning the clean 1k subset on the T4 changed {changed} of its {v('CrossHWFlipPairsFull')} advisories"
        if "CrossHWValFlips" in m:
            line += (f", and {v('CrossHWValFlips')} of the {v('CrossHWValPairs')} val_rest images processed on both GPUs "
                     f"got a different advisory ({v('CrossHWValFlip')} %)")
        extra.append(line + ".")
    if "OpValopChosenUnsafe" in m:
        extra.append(f"* Operating point x{v('OpChosenScale')} (the largest dev scale without unsafe CLEAR) on the "
                     f"{v('OpValopImages')} held-out val_rest images: unsafe CLEAR {v('OpValopBaseUnsafe')} -> "
                     f"{v('OpValopChosenUnsafe')} of {v('OpValopOccupied')} occupied images ({v('OpValopBaseUnsafePct')} % -> "
                     f"{v('OpValopChosenUnsafePct')} %), over-caution {v('OpValopBaseOverPct')} % -> "
                     f"{v('OpValopChosenOverPct')} %. The dev margin does not transfer, so the published cut-offs are kept.")
    if "TrainImages" in m:
        line = (f"* Supplementary BDD100K train split ({v('TrainImages')} images; HybridNets was trained on them): "
                f"unsafe CLEAR {v('TrainUnsafe')}/{v('TrainOccupied')}, over-caution {v('TrainOverPct')} %")
        if "TrainNoHybUnsafe" in m:
            line += f"; with HybridNets replaced by the dev road prior {v('TrainNoHybUnsafe')} and {v('TrainNoHybOverPct')} %"
        extra.append(line + ". Labels from public copies of the official det_20 release, accepted only because they "
                     "reproduce the official val labels and masks for all 10,000 val images.")
    if "LatTFourTotal" in m:
        extra.append(f"* Median latency on an NVIDIA T4: {v('LatTFourTotal')} s/image ({v('LatTFourImages')} images).")
    extra_text = ("\n## Additional checks added for the camera-ready version\n\n" + "\n".join(extra) + "\n") if extra else ""

    doc = f"""# Camera-ready changes: "Conservative Road-Scene Risk Assessment via Fail-Aware Multi-Model Fusion" (m17641)

All numbers below are generated from the released evaluation code (commit `{v('CodeCommit')}`) and
match the manuscript macros exactly.

## Note for the program chairs

The submitted version reported results on a 100-image hard set whose image list was not preserved.
For the camera-ready version the hard set was regenerated with the published deterministic
selection script, its image IDs are released (SHA-256 prefix `{v('HardIDsHash')}`), and the
evaluation was extended to 9,500 BDD100K validation images with ground-truth annotations.
The headline internal statistic is essentially unchanged (stable ratio {v('HardStablePct')} % on the
regenerated hard set vs. 86.1 % submitted); all other numbers are new. The method (models,
thresholds, fusion rules) is unchanged.

## Reviewer 1

| Comment | Change | Where |
|---|---|---|
| Justify conservative advisory thresholds with ablations | Operating curve over the advisory cut-offs (published point on dev: {v('OpDevBaseOverPct')} % over-caution; the cut-offs can be relaxed up to x{v('OpDevSafeScale')} before the first unsafe CLEAR, lowering over-caution to {v('OpDevSafeOverPct')} %); +/-20 % sensitivity of every reliability constant (stable ratio moves at most {v('SensRelStableDelta')} pp); cumulative B0-B3 build-up and leave-one-out ablations on 9,500 test images | Sec. V-C, V-D, V-F, Fig. 4, Table III |
| Uncertainty propagation rationale | Asymmetric handling stated explicitly (weak evidence can add caution at the low end, e.g. CLEAR to MONITOR, but cannot justify STOP/BRAKE, which the gates cap); c_d defined; the evidence-confidence equation (c_e) corrected to the implemented form | Sec. III-A, III-C |
| Detector-empty scenes | Fallback fired on {v('TestFallback')} of {v('TestEmpty')} detector-empty test images; unsafe CLEAR {v('TestUnsafe')} with vs. {v('TestUnsafeNoFB')} without the fallback | Sec. V-D |
| Severe weather / deployment limits | Results by weather, compared within each time of day because most night images are tagged clear (daytime relevant recall: clear {v('TestWClearDayRecRel')} %, rain {v('TestWRainyDayRecRel')} %, snow {v('TestWSnowyDayRecRel')} %; fog too rare: {v('TestWFoggyImages')} images); latency {v('LatTotal')} s/image on RTX 3050 | Sec. V-E, V-F, VI |
| Multi-region checks and advisory stability | New Fig. 2 (Stable vs. Check example); single-region ablation (stable ratio {v('TestSingleStablePct')} % vs. {v('TestStablePct')} %); perturbation stability (advisory flips {v('StdFlipFull')} % on 1k images; with the center region alone {v('StdFlipSingle')} %, state changes {v('StdStateFlipSingle')} % vs. {v('StdStateFlipFull')} %: the check stabilizes the states, not the advisories) | Sec. III-B, V-D, V-F |
| HybridNets vs. Depth Anything at night | Replacing Depth Anything changes {v('TestNoDepthChanged')} advisories, HybridNets {v('TestNoHybChanged')}; on night images {v('TestNoDepthNightChangedPct')} % vs. {v('TestNoHybNightChangedPct')} % (daytime {v('TestNoDepthDayChangedPct')} % vs. {v('TestNoHybDayChangedPct')} %); only {v('TestLaneGuidedPct')} % of corridor rows are lane-guided; road IoU {v('TestNightRoadIoU')} at night vs. {v('TestDayRoadIoU')} by day | Sec. V-D, V-E |
| Reproducibility (hard-set selection, audit, CIs) | Selection formulas, released ID lists and hashes, ground-truth protocol, Wilson + image-cluster bootstrap + exact McNemar | Sec. IV |

## Reviewer 2

| Comment | Change | Where |
|---|---|---|
| No ground-truth validation | Detection precision {v('TestPrec')} %, relevant recall {v('TestRecRel')} %; PATH precision {v('TestPathPrec')} % / recall {v('TestPathRec')} %; unsafe CLEAR {v('TestUnsafe')}/{v('TestOccupied')}; over-caution {v('TestOver')}/{v('TestFree')}; unsupported BRAKE/STOP {v('TestUnsupportedPct')} % | Sec. V-A to V-C, Table I |
| Non-reproducible custom set | 9,500-image test split (val minus dev) + deterministic hard set with released IDs; byte-identical cached replays | Sec. IV-B, IV-D |
| Detection correctness / path-occupancy accuracy | Reliability states validated: non-STABLE estimates are {v('TestFPRatio')}x more often false positives; Check concentrates on truncated objects ({v('TestTruncCheck')} % vs. {v('TestTruncStable')} %) | Sec. V-B, Table II |

## Reviewer 3

| Comment | Change | Where |
|---|---|---|
| Presentation (2/5) | Full rewrite; fixed broken figure reference and caption; consistent terminology; u renamed to a relative depth index (smaller = nearer) | whole paper |
| Novelty (2/5) | Positioned as fusion-level selective prediction; related work on selective prediction, conformal prediction and run-time monitoring added; operating-curve analysis | Sec. I, II, V-C |
| Minor grammar | Language pass | whole paper |

## Honest negative findings now reported

* Over-caution: CLEAR on only {v('TestClearPct')} % of test images; BRAKE/STOP on {v('TestSeverePct')} %.
* Over-inclusive perspective corridor: PATH precision {v('TestPathPrec')} %; HybridNets under-used.
* Exploratory fix (not part of the method): corridor intersected with the HybridNets drivable mask
  raises PATH precision to {v('TestExpPathPrec')} % with {v('TestExpUnsafe')} unsafe CLEAR.
* The {v('TestUnsafe')} unsafe-CLEAR test images, from the evaluation CSVs: in {v('TestUnsafeDetected')} the in-lane road
  user was detected but related as SIDE by the corridor (advisory capped), in {v('TestUnsafeMissed')} the detector
  missed it ({v('TestUnsafeMissedNight')} at night); train split {v('TrainUnsafeDetected')} and {v('TrainUnsafeMissed')} of {v('TrainUnsafe')}.
{extra_text}
## Format

`main.pdf` is six pages in the IEEE conference A4 format, as the submitted version and as the
conference requires (4-6 pages including tables, figures and references; A4 Word template on the
conference site). To fit, the hard-set category rows of Table IV, two of the six qualitative cases,
the note on the submitted version's hard set and one adverse-conditions reference moved to the
extended version (`main_extended.pdf`, not for submission), short formulas are set in the text, and
the prose was tightened. No headline result was removed. Secondary numbers that left the six-page
text: the x2.5 dev operating point, the hard-set category rows and the test-split no-HybridNets pair
of the train sentence (all in the extended version), and the 0.02 % behind "1 of the 5,130".

## Still to do by the authors

1. Replace the repository placeholder `\\RepoURL` with the public URL (after team approval); the
   last page has room for about five more reference-size lines.
2. Check the final-manuscript deadline in EDAS (the conference site lists 20 November 2026 for
   batch 2; conference day 21 November 2026) and the presentation file.
3. Double-check the newly added references.
4. IEEE PDF eXpress (conference ID 63724X on the site) and copyright form; inform the chairs about
   the regenerated numbers.
5. Look at the unsafe-CLEAR cases in `audit/unsafe_clear_test.jpg` yourselves: the paper states only
   what the CSVs show (detected but SIDE, or missed), not why (curved lane, camera offset, wide
   annotated lane); add a cause only if you agree on it.
6. Check the author kit for a copyright notice on page 1 and for the author-block layout.
"""
    (OUT / "CAMERA_READY_CHANGES.md").write_text(doc, encoding="utf-8")
    print(f"[CHANGES] wrote {OUT / 'CAMERA_READY_CHANGES.md'}")


if __name__ == "__main__":
    main()
