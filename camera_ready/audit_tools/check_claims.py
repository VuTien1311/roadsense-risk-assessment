"""Check the wording of the train sentences in main.tex against the numbers in numbers.tex.

Usage: python check_claims.py <paper folder> [results.json]
Each claim the text makes about the train split is tested; FAIL means the sentence must be reworded.
"""
import json
import re
import sys
from pathlib import Path

paper = Path(sys.argv[1])
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (paper / "numbers.tex").read_text(encoding="utf-8")))


def num(name):
    value = macros[name].replace("{,}", "").replace("\\textbf{", "").rstrip("}")
    return float(value)


checks = []


def claim(text, ok, detail):
    checks.append(ok)
    print(("PASS" if ok else "FAIL"), text, "|", detail)


show = ["TrainImages", "TrainOccupied", "TrainUnsafe", "TrainOverPct", "TrainNoHybUnsafe", "TrainNoHybOverPct",
        "TrainRoadIoU", "TestRoadIoU", "TestNoHybUnsafe", "TestNoHybOverPct", "OpChosenScale", "OpTrainImages",
        "OpTrainBaseUnsafePct", "OpTrainChosenUnsafePct", "OpTrainBaseOverPct", "OpTrainChosenOverPct",
        "OpTrainChosenUnsafe", "OpTrainBaseUnsafe", "OpValopBaseUnsafePct", "OpValopChosenUnsafePct",
        "TrainWFoggyImages", "TestWFoggyImages", "TrainWSnowyImages", "TestWSnowyImages", "TrainWFoggyDayImages",
        "TrainWFoggyDayRecRel", "TrainWSnowyDayRecRel", "TrainWClearDayRecRel"]
for name in show:
    print(f"  {name} = {macros.get(name, 'MISSING')}")

claim("the dev-chosen relaxation raises unsafe CLEAR on train too ('does not transfer')",
      num("OpTrainChosenUnsafePct") > num("OpTrainBaseUnsafePct"),
      f"{macros['OpTrainBaseUnsafePct']} -> {macros['OpTrainChosenUnsafePct']} %")
claim("over-caution falls with the relaxation on train ('over-caution from .. to ..')",
      num("OpTrainChosenOverPct") < num("OpTrainBaseOverPct"),
      f"{macros['OpTrainBaseOverPct']} -> {macros['OpTrainChosenOverPct']} %")
for split in ("Test", "Train"):
    for tod in ("Day", "Night"):
        clear, rainy, snowy = (num(f"{split}W{w}{tod}RecRel") for w in ("Clear", "Rainy", "Snowy"))
        claim(f"{split} {tod}: rain lowers relevant recall", rainy < clear, f"{clear} -> {rainy}")
        claim(f"{split} {tod}: snow hardly changes / 'leaves it at' (within 3 points of clear)",
              abs(snowy - clear) <= 3.0, f"{clear} vs {snowy}")
claim("train weather sentence covers more images than the test one (rainy, snowy)",
      num("TrainWRainyImages") > num("TestWRainyImages") and num("TrainWSnowyImages") > num("TestWSnowyImages"),
      f"rainy {macros['TrainWRainyImages']} vs {macros['TestWRainyImages']}, snowy {macros['TrainWSnowyImages']} vs {macros['TestWSnowyImages']}")
claim("fog 'remains too rare' on train (< 0.5 % of the images)",
      num("TrainWFoggyImages") < 0.005 * num("TrainImages"), f"{macros['TrainWFoggyImages']} of {macros['TrainImages']}")
claim("HybridNets was trained on train: its road IoU is higher there (the parenthesis implies optimism)",
      num("TrainRoadIoU") > num("TestRoadIoU"), f"{macros['TrainRoadIoU']} vs {macros['TestRoadIoU']}")
claim("train images counted = 69,863", num("TrainImages") == 69863 and num("OpTrainImages") == 69863,
      f"{macros['TrainImages']} / {macros['OpTrainImages']}")
for split in ("Test", "Train"):
    unsafe, detected, side, missed = (num(f"{split}{name}") for name in
                                      ("Unsafe", "UnsafeDetected", "UnsafeDetectedSide", "UnsafeMissed"))
    claim(f"{split}: unsafe CLEAR = detected + missed", detected + missed == unsafe, f"{detected} + {missed} = {unsafe}")
    claim(f"{split}: 'the corridor related it as SIDE' in every detected case", side == detected, f"{side} of {detected}")
    claim(f"{split}: 'about half of the unsafe CLEAR cases' are detected-but-SIDE (40-60 %)",
          0.4 <= detected / unsafe <= 0.6, f"{detected} of {unsafe}")
claim("B0 baseline: full system 'less over-cautious' (p < 0.05 as printed) 'with no more unsafe CLEAR'",
      num("TestOverPct") < num("TestBZeroOverPct") and macros["TestBZeroOverP"].startswith("<0.0")
      and num("TestUnsafe") <= num("TestBZeroUnsafe"),
      f"over-caution {macros['TestOverPct']} vs {macros['TestBZeroOverPct']} (p{macros['TestBZeroOverP']}), "
      f"unsafe {macros['TestUnsafe']} vs {macros['TestBZeroUnsafe']}")
claim("fallback: unsafe CLEAR rises without it, p printed as not significant",
      num("TestUnsafeNoFB") > num("TestUnsafe") and float(macros["TestNoFBUnsafeP"].lstrip("=<")) > 0.05,
      f"{macros['TestUnsafe']} -> {macros['TestUnsafeNoFB']}, p{macros['TestNoFBUnsafeP']}")
claim("night: depth 'matters equally' (within 2 points), road cues 'about twice as much' at night (1.6x-2.4x)",
      abs(num("TestNoDepthNightChangedPct") - num("TestNoDepthDayChangedPct")) <= 2.0
      and 1.6 <= num("TestNoHybNightChangedPct") / num("TestNoHybDayChangedPct") <= 2.4
      and num("TestNoHybNightChangedPct") < num("TestNoDepthNightChangedPct") / 3,
      f"depth {macros['TestNoDepthNightChangedPct']} vs {macros['TestNoDepthDayChangedPct']}, "
      f"HybridNets {macros['TestNoHybNightChangedPct']} vs {macros['TestNoHybDayChangedPct']}")
claim("center region alone: flips 'essentially unchanged' (within 1 point), state changes 'slightly' higher",
      abs(num("StdFlipSingle") - num("StdFlipFull")) <= 1.0 and abs(num("FlipSingle") - num("FlipFull")) <= 1.0
      and 0 < num("StdStateFlipSingle") - num("StdStateFlipFull") <= 1.0,
      f"1k {macros['StdFlipSingle']} vs {macros['StdFlipFull']}, hard {macros['FlipSingle']} vs {macros['FlipFull']}, "
      f"states {macros['StdStateFlipSingle']} vs {macros['StdStateFlipFull']}")
if len(sys.argv) > 2:
    results = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    scales = sorted(p["scale"] for p in results["operating"].get("train", []))
    claim("Fig. 4 caption: train curve spans x0.8 to x2.0", bool(scales) and scales[0] == 0.8 and scales[-1] == 2.0,
          f"scales {scales}")
print("all claims hold" if all(checks) else "SOME CLAIMS FAIL - reword main.tex")
