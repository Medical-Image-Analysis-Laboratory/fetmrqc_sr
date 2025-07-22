# FetMRQC_SR: Quality control for fetal brain MRI
#
# Copyright 2025 Medical Image Analysis Laboratory (MIAL)
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

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from datetime import datetime
import json
from fetmrqc_sr import ROOT_DIR, IQMS

DATASET = os.path.join(ROOT_DIR, "data", "QC_IQMs.csv")

OUT_DIR = os.path.join(ROOT_DIR, "data", "models")


def load_dataset(dataset, first_iqm):
    df = pd.read_csv(dataset)
    xy_index = df.columns.tolist().index(first_iqm)

    train_x = df[df.columns[xy_index:]].copy()
    train_y = df[df.columns[:xy_index]].copy()

    return train_x, train_y


def get_rating(rating, class_threshold=1.0):
    """Format the rating: if it is a classification task,
    binarize the rating at the class_threshold
    """
    if isinstance(rating, list):
        return [int(r > class_threshold) for r in rating]
    elif isinstance(rating, pd.DataFrame):
        return (rating > class_threshold).astype(int)
    else:
        return rating > class_threshold


# Root of package fetmrqc_sr
# data_folder = os.path.dirname(os.path.abspath(__file__))


def main():
    # Parser version of the code below
    import argparse

    parser = argparse.ArgumentParser(
        "Train a FetMRQC_SR classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        help="Path to the csv file dataset.",
        default=DATASET,
    )
    parser.add_argument(
        "--first_iqm",
        help="First IQM in the csv of the dataset.",
        default="centroid",
    )
    parser.add_argument(
        "--target",
        help="Target rating to use as ground truth.",
        default="qcglobal",
        choices=[
            "qcglobal",
            "is_reconstructed",
            "geom_artefact",
            "recon_artefact",
            "noise",
            "intensity_gm",
            "intensity_dgm",
        ],
    )

    parser.add_argument(
        "--model_path",
        help="Path to the model to use.",
        default=os.path.join(OUT_DIR, "fetmrqc_SR_full_qcglobal.joblib"),
    )

    parser.add_argument(
        "--iqms_list",
        help="Custom list of IQMs to use. By default, all IQMs are used.",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--threshold",
        help="Threshold for classification as a positive example.",
        default=0.7,
        type=float,
    )

    parser.add_argument(
        "--out_csv",
        help=f"Where to save the results (default: {OUT_DIR}/<model_name>_pred.csv)",
        default=None,
    )

    args = parser.parse_args()

    iqms = IQMS
    if args.iqms_list is not None:
        iqms = args.iqms_list
        print(f"Using custom IQMs: {iqms}")

    out_csv = (
        os.path.abspath(args.out_csv)
        if args.out_csv
        else os.path.join(
            OUT_DIR, f"{args.model_path.split('/')[-1].split('.')[0]}_pred.csv"
        )
    )
    out_dir = os.path.dirname(out_csv)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.dataset)
    train_x, train_y = load_dataset(args.dataset, args.first_iqm)
    # Load model weights
    model = joblib.load(args.model_path)

    predictions = model.predict_proba(train_x[iqms])
    # Add the predictions to the first column
    # Prob_good
    df.insert(1, "fetmrqc_prob_good", predictions[:, 1])
    df.insert(2, "fetmrqc_pred_good", predictions[:, 1] > args.threshold)

    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
