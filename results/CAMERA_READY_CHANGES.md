# Camera-ready changes: "Conservative Road-Scene Risk Assessment via Fail-Aware Multi-Model Fusion" (m17641)

All numbers below are generated from the released evaluation code (commit `8c65b8a`) and
match the manuscript macros exactly.

## Note for the program chairs

The submitted version reported results on a 100-image hard set whose image list was not preserved.
For the camera-ready version the hard set was regenerated with the published deterministic
selection script, its image IDs are released (SHA-256 prefix `5d6c11f5bdd1198a`), and the
evaluation was extended to 9,500 BDD100K validation images with ground-truth annotations.
The headline internal statistic is essentially unchanged (stable ratio 86.5 % on the
regenerated hard set vs. 86.1 % submitted); all other numbers are new. The method (models,
thresholds, fusion rules) is unchanged.

## Reviewer 1

| Comment | Change | Where |
|---|---|---|
| Justify conservative advisory thresholds with ablations | Operating curve over the advisory cut-offs (published point on dev: 85.0 % over-caution; the cut-offs can be relaxed up to x1.4 before the first unsafe CLEAR, lowering over-caution to 72.4 %); +/-20 % sensitivity of every reliability constant (stable ratio moves at most 34.0 pp); cumulative B0-B3 build-up and leave-one-out ablations on 9,500 test images | Sec. V-C, V-D, V-F, Fig. 4, Table III |
| Uncertainty propagation rationale | Asymmetric handling stated explicitly (weak evidence can add caution at the low end, e.g. CLEAR to MONITOR, but cannot justify STOP/BRAKE, which the gates cap); c_d defined; the evidence-confidence equation (c_e) corrected to the implemented form | Sec. III-A, III-C |
| Detector-empty scenes | Fallback fired on 236 of 295 detector-empty test images; unsafe CLEAR 11 with vs. 15 without the fallback | Sec. V-D |
| Severe weather / deployment limits | Results by weather, compared within each time of day because most night images are tagged clear (daytime relevant recall: clear 83.9 %, rain 80.7 %, snow 82.3 %; fog too rare: 13 images); latency 2.3 s/image on RTX 3050 | Sec. V-E, V-F, VI |
| Multi-region checks and advisory stability | New Fig. 2 (Stable vs. Check example); single-region ablation (stable ratio 80.3 % vs. 84.9 %); perturbation stability (advisory flips 5.2 % on 1k images; with the center region alone 5.1 %, state changes 3.4 % vs. 3.0 %: the check stabilizes the states, not the advisories) | Sec. III-B, V-D, V-F |
| HybridNets vs. Depth Anything at night | Replacing Depth Anything changes 3,631 advisories, HybridNets 364; on night images 38.8 % vs. 5.5 % (daytime 38.2 % vs. 2.7 %); only 9.8 % of corridor rows are lane-guided; road IoU 0.83 at night vs. 0.83 by day | Sec. V-D, V-E |
| Reproducibility (hard-set selection, audit, CIs) | Selection formulas, released ID lists and hashes, ground-truth protocol, Wilson + image-cluster bootstrap + exact McNemar | Sec. IV |

## Reviewer 2

| Comment | Change | Where |
|---|---|---|
| No ground-truth validation | Detection precision 89.1 %, relevant recall 78.9 %; PATH precision 9.2 % / recall 53.8 %; unsafe CLEAR 11/3,250; over-caution 4,799/5,844; unsupported BRAKE/STOP 67.5 % | Sec. V-A to V-C, Table I |
| Non-reproducible custom set | 9,500-image test split (val minus dev) + deterministic hard set with released IDs; byte-identical cached replays | Sec. IV-B, IV-D |
| Detection correctness / path-occupancy accuracy | Reliability states validated: non-STABLE estimates are 1.8x more often false positives; Check concentrates on truncated objects (69.2 % vs. 2.5 %) | Sec. V-B, Table II |

## Reviewer 3

| Comment | Change | Where |
|---|---|---|
| Presentation (2/5) | Full rewrite; fixed broken figure reference and caption; consistent terminology; u renamed to a relative depth index (smaller = nearer) | whole paper |
| Novelty (2/5) | Positioned as fusion-level selective prediction; related work on selective prediction, conformal prediction and run-time monitoring added; operating-curve analysis | Sec. I, II, V-C |
| Minor grammar | Language pass | whole paper |

## Honest negative findings now reported

* Over-caution: CLEAR on only 2.1 % of test images; BRAKE/STOP on 65.1 %.
* Over-inclusive perspective corridor: PATH precision 9.2 %; HybridNets under-used.
* Exploratory fix (not part of the method): corridor intersected with the HybridNets drivable mask
  raises PATH precision to 13.5 % with 11 unsafe CLEAR.
* The 11 unsafe-CLEAR test images, from the evaluation CSVs: in 6 the in-lane road
  user was detected but related as SIDE by the corridor (advisory capped), in 5 the detector
  missed it (4 at night); train split 30 and 31 of 61.

## Additional checks added for the camera-ready version

* Hardware: 3,270 of the 9,500 test images (34 %) and every 1k perturbation run were processed on an NVIDIA T4 with the same pinned library versions; rerunning the clean 1k subset on the T4 changed none of its 1,000 advisories, and 1 of the 5,130 val_rest images processed on both GPUs got a different advisory (0.02 %).
* Operating point x1.4 (the largest dev scale without unsafe CLEAR) on the 8,400 held-out val_rest images: unsafe CLEAR 9 -> 28 of 2,867 occupied images (0.31 % -> 0.98 %), over-caution 82.0 % -> 70.2 %. The dev margin does not transfer, so the published cut-offs are kept.
* Supplementary BDD100K train split (69,863 images; HybridNets was trained on them): unsafe CLEAR 61/23,604, over-caution 80.8 %; with HybridNets replaced by the dev road prior 94 and 79.1 %. Labels from public copies of the official det_20 release, accepted only because they reproduce the official val labels and masks for all 10,000 val images.
* Median latency on an NVIDIA T4: 1.5 s/image (119 images).

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

1. Replace the repository placeholder `\RepoURL` with the public URL (after team approval); the
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
