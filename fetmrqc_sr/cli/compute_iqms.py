# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Command line interface for the extraction of IQMs from fetal brain images
using FetMRQC
"""
import os
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from fetal_brain_utils import csv_to_list, print_title
from fetmrqc_sr.metrics import SRMetrics, BOUNTI_LABELS, FETA_LABELS, DHCP_LABELS
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys

EXCEPTION_STATUS = None


def process_subject(idx, run, verbose, robust, correct_bias, metrics=None, seg_method="bounti"):
    global EXCEPTION_STATUS
    while not EXCEPTION_STATUS:
        try:
            name = Path(run["im"]).name
            sub = run["sub"]
            print(f"Processing subject {sub} ({name}) (PID: {os.getpid()})")
            if seg_method == "bounti":
                map_seg = BOUNTI_LABELS
            elif seg_method == "feta":
                map_seg = FETA_LABELS
            elif seg_method == "dhcp":
                map_seg = DHCP_LABELS
            
            sr_metrics = SRMetrics(
                verbose=verbose,
                robust_preprocessing=robust,
                correct_bias=correct_bias,
                map_seg=map_seg,
                counter=idx,
            )
            if metrics is not None:
                sr_metrics.set_metrics(metrics)
            res = sr_metrics.evaluate_metrics(
                run["im"], run["mask"], run["seg"]
            )
            return idx, res
        except Exception as e:

            EXCEPTION_STATUS = e
            return e, run, os.getpid()


def main(argv=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    p = argparse.ArgumentParser(
        description=("Computes quality metrics from given images."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--bids_csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )

    p.add_argument(
        "--out_csv",
        help="Path where the IQA results will be stored.",
        required=True,
    )

    p.add_argument(
        "--continue_run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether QC run should re-use existing results if a metrics.csv file at "
            "`out_path`/metrics.csv."
        ),
    )

    p.add_argument(
        "--metrics",
        nargs="*",
        help="List of metrics to compute. If not provided, all metrics are computed.",
        default=None,
    )

    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Enable verbose."),
    )
    p.add_argument(
        "--nworkers", type=int, default=5, help="Number of workers."
    )

    p.add_argument(
        "--robust_prepro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable robust preprocessing prior to IQMs computation.",
    )
    p.add_argument(
        "--correct_bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable bias field correction.",
    )

    p.add_argument(
        "--seg_label_scheme",
        choices=["bounti", "feta", "dhcp"],
        default="bounti",
        help="Segmentation label scheme used for segmentation.",
    )
    args = p.parse_args(argv)
    df_base = pd.read_csv(args.bids_csv)
    df_base = df_base.set_index("name")
    print_title("Computing IQMs")

    metrics_dict = {}

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    # If a file is found, continue.

    if os.path.isfile(args.out_csv) and args.continue_run:
        print("\tCONTINUING FROM A PREVIOUSLY FOUND RUN.")
        df = pd.read_csv(args.out_csv).set_index("name")
        metrics_dict = df.to_dict(orient="index")
        # Remove duplicate keys
        metrics_dict = {
            k: {k2: v2 for k2, v2 in v.items() if k2 not in df_base.columns}
            for k, v in metrics_dict.items()
        }
    df_run = df_base[~df_base.index.isin(metrics_dict.keys())]

    start = time.time()
    # for idx, run in df_run.iterrows():
    # res = process_subject(idx, run, args.verbose)
    with ProcessPoolExecutor(args.nworkers) as executor:
        # submit tasks and collect futures
        futures = []
        for idx, run in df_run.iterrows():
            futures.append(
                executor.submit(process_subject, idx, run, args.verbose, args.robust_prepro, args.correct_bias, args.metrics, args.seg_label_scheme)
            )
        # process task results as they are available
        for i, future in enumerate(as_completed(futures)):
        #for i, (idx, run) in enumerate(df_run.iterrows()):
            #out = process_subject(idx, run, args.verbose, args.metrics)
            out = future.result()
            if len(out) == 2:
                idx, res = out
            elif len(out) == 3:
                error, run, pid = out

                print(
                    f"ERROR in thread {pid}. Here are the inputs that led to the error:"
                )
                missing = []
                for k, v in run.items():
                    if k in ["im", "mask", "seg"]:
                        # Check if v is nan
                        if v is None or pd.isna(v) or v == "nan" or v == "":
                            missing.append(k)
                    print(f"\t{k}: {v}")
                if len(missing) > 0:
                    print(
                        f"WARNING: Found some missing inputs: {', '.join(missing)}. This likely caused the error."
                    )
                raise type(error).with_traceback(error, error.__traceback__)
            metrics_dict[idx] = res
            df = pd.DataFrame.from_dict(metrics_dict, orient="index")
            df = pd.concat([df_base, df], axis=1, join="inner")
            df.index.name = "name"
            df.to_csv(args.out_csv)
    print(f"Finished in {time.time() - start:.2f} seconds")
    return 0


if __name__ == "__main__":
    main()
