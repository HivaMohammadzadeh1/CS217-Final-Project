#!/usr/bin/env python3
"""
Summarize MXFP8 vs MXFP4 resource estimates.

You can run this in two ways:
1) With HLS report files:
   python3 resource_estimator.py --mxfp8-report <rpt> --mxfp4-report <rpt>
2) Without reports (prints conservative defaults):
   python3 resource_estimator.py
"""

import argparse
import json
import re
from pathlib import Path


def parse_report(path: Path):
  text = path.read_text(errors="ignore")

  def find_first(patterns):
    for pattern in patterns:
      m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
      if m:
        return int(m.group(1))
    return None

  dsp = find_first([
      r"^\s*DSP(?:48E)?\s*\|\s*(\d+)",
      r"^\s*\|\s*DSP(?:48E)?\s*\|\s*(\d+)\s*\|",
  ])
  lut = find_first([
      r"^\s*LUT\s*\|\s*(\d+)",
      r"^\s*\|\s*LUT\s*\|\s*(\d+)\s*\|",
  ])
  bram = find_first([
      r"^\s*BRAM(?:_18K)?\s*\|\s*(\d+)",
      r"^\s*\|\s*BRAM(?:_18K)?\s*\|\s*(\d+)\s*\|",
  ])

  return {
      "DSP": dsp,
      "LUT": lut,
      "BRAM": bram,
  }


def fallback_estimates():
  return {
      "MXFP8": {"DSP": 8, "LUT": 500, "BRAM": 2},
      "MXFP4": {"DSP": 4, "LUT": 320, "BRAM": 2},
  }


def print_table(resources):
  print("\nResource Summary")
  print("----------------------------------------------------------------")
  print(f"{'Mode':<8} {'DSP':>8} {'LUT':>10} {'BRAM':>8}")
  print("----------------------------------------------------------------")
  for mode in ["MXFP8", "MXFP4"]:
    row = resources[mode]
    print(f"{mode:<8} {str(row['DSP']):>8} {str(row['LUT']):>10} {str(row['BRAM']):>8}")
  print("----------------------------------------------------------------")

  if resources["MXFP8"]["DSP"] is not None and resources["MXFP4"]["DSP"] is not None:
    dsp8 = resources["MXFP8"]["DSP"]
    dsp4 = resources["MXFP4"]["DSP"]
    if dsp8 > 0:
      reduction = 100.0 * (dsp8 - dsp4) / dsp8
      print(f"DSP reduction (MXFP8 -> MXFP4): {reduction:.1f}%")
  print("")


def main():
  parser = argparse.ArgumentParser(description="Estimate MXFP resource usage.")
  parser.add_argument("--mxfp8-report", type=Path, default=None)
  parser.add_argument("--mxfp4-report", type=Path, default=None)
  parser.add_argument("--output", type=Path, default=None,
                      help="Optional JSON output path.")
  args = parser.parse_args()

  resources = fallback_estimates()
  using_reports = False

  if args.mxfp8_report and args.mxfp8_report.exists():
    resources["MXFP8"] = parse_report(args.mxfp8_report)
    using_reports = True

  if args.mxfp4_report and args.mxfp4_report.exists():
    resources["MXFP4"] = parse_report(args.mxfp4_report)
    using_reports = True

  if using_reports:
    print("Parsed data from report file(s).")
  else:
    print("No report files provided. Using default planning estimates.")

  print_table(resources)

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resources, indent=2))
    print(f"Wrote JSON summary to {args.output}")


if __name__ == "__main__":
  main()

