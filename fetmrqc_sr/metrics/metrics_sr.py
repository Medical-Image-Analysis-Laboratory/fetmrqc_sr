# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
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
import pdb
import numpy as np
import nibabel as ni
import skimage
import traceback
from .utils import (
    normalized_cross_correlation,
    shannon_entropy,
    joint_entropy,
    mutual_information,
    normalized_mutual_information,
    psnr,
    rmse,
    mae,
    nmae,
    nrmse,
    ssim,
    centroid,
    rank_error,
    mask_volume,
    compute_topological_features
)
from skimage.filters import sobel, laplace
from inspect import getmembers, isfunction
from fetal_brain_qc.utils import squeeze_dim
from scipy.stats import kurtosis, variation
import pandas as pd
from .mriqc_metrics import (
    summary_stats,
    volume_fraction,
    snr,
    cnr,
    cjv,
    wm2max,
)
import sys
from functools import partial
from fetal_brain_utils import get_cropped_stack_based_on_mask
import warnings

SKIMAGE_FCT = [fct for _, fct in getmembers(skimage.filters, isfunction)]
SEGM = {"CSF": 1, "GM": 2, "WM": 3, "BS": 4, "CBM": 5}
# Re-mapping to do for FeTA labels: ventricles as CSF, dGM as GM.
FETA_LABELS = [None, 1, 2, 3, 1, 5, 2, 4]
segm_names = list(SEGM.keys())

BOUNTI_LABELS = [
    None,
    1,
    1,
    2,
    2,
    3,
    3,
    1,
    1,
    1,
    4,
    5,
    5,
    5,
    2,
    2,
    2,
    2,
    1,
    1
]


DHCP_LABELS = [
    None,
    1,
    2,
    3,
    None,
    1,
    5,
    2,
    4,
    2
]


class SRMetrics:
    """Contains a battery of metrics that can be evaluated on individual
    pairs of super-resolution stacks and segmentations.
    """

    def __init__(
        self,
        verbose=False,
        robust_preprocessing=False,
        correct_bias=False,
        map_seg=BOUNTI_LABELS,
        counter=0,
    ):
        default_params = dict(
            compute_on_mask=True,
            mask_intersection=True,
            use_window=True,
            reduction="mean",
        )

        self.verbose = verbose
        self.robust_prepro = robust_preprocessing
        self.correct_bias = correct_bias
        self.metrics_func = {
            "centroid": partial(centroid),
            "rank_error": partial(
                rank_error,
                threshold=0.99,
                relative_rank=False,
            ),
            "rank_error_relative": partial(
                rank_error,
                threshold=0.99,
                relative_rank=True,
            ),
            "mask_volume": mask_volume,
            "ncc_window": self.process_metric(
                metric=normalized_cross_correlation, **default_params
            ),
            "ncc_median": self.process_metric(
                metric=normalized_cross_correlation,
                mask_intersection=True,
                reduction="median",
            ),
            "joint_entropy_window": self.process_metric(
                metric=joint_entropy, **default_params
            ),
            "joint_entropy_median": self.process_metric(
                metric=joint_entropy,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="median",
            ),
            "mi_window": self.process_metric(
                metric=mutual_information, **default_params
            ),
            "mi_median": self.process_metric(
                metric=mutual_information,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="median",
            ),
            "nmi_window": self.process_metric(
                metric=normalized_mutual_information, **default_params
            ),
            "nmi_median": self.process_metric(
                metric=normalized_mutual_information,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="median",
            ),
            "shannon_entropy": self.process_metric(
                shannon_entropy,
                type="noref",
                compute_on_mask=True,
            ),
            "psnr_window": self.process_metric(
                psnr,
                use_datarange=True,
                **default_params,
            ),
            "nrmse_window": self.process_metric(nrmse, **default_params),
            "rmse_window": self.process_metric(rmse, **default_params),
            "nmae_window": self.process_metric(nmae, **default_params),
            "mae_window": self.process_metric(mae, **default_params),
            "ssim_window": partial(self._ssim, **default_params),
            "mean": self.process_metric(
                np.mean,
                type="noref",
                compute_on_mask=True,
            ),
            "std": self.process_metric(
                np.std,
                type="noref",
                compute_on_mask=True,
            ),
            "median": self.process_metric(
                np.median,
                type="noref",
                compute_on_mask=True,
            ),
            "percentile_5": self.process_metric(
                partial(np.percentile, q=5),
                type="noref",
                compute_on_mask=True,
            ),
            "percentile_95": self.process_metric(
                partial(np.percentile, q=95),
                type="noref",
                compute_on_mask=True,
            ),
            "kurtosis": self.process_metric(
                kurtosis,
                type="noref",
                compute_on_mask=True,
            ),
            "variation": self.process_metric(
                variation,
                type="noref",
                compute_on_mask=True,
            ),
            ## Filter-based metrics
            "filter_laplace": partial(self._metric_filter, filter=laplace),
            "filter_sobel": partial(self._metric_filter, filter=sobel),
            "seg_sstats": self.process_metric(self._seg_sstats, type="seg"),
            "seg_volume": self.process_metric(self._seg_volume, type="seg"),
            "seg_snr": self.process_metric(self._seg_snr, type="seg"),
            "seg_cnr": self.process_metric(self._seg_cnr, type="seg"),
            "seg_cjv": self.process_metric(self._seg_cjv, type="seg"),
            "seg_wm2max": self.process_metric(self._seg_wm2max, type="seg"),
            "seg_topology": self.process_metric(self._seg_topology, type="seg"),
        }
        self._metrics = self.get_all_metrics()
        self._check_metrics()
        self.map_seg = map_seg
        # Summary statistics from the segmentation, used for computing a bunch of metrics
        # besides being a metric itself
        self._sstats = None
        self.counter=counter

    def get_all_metrics(self):
        return list(self.metrics_func.keys())

    def set_metrics(self, metrics):
        self._metrics = metrics

    def _valid_mask(self, mask_path):
        mask = ni.load(mask_path).get_fdata()
        if mask.sum() == 0:
            return False
        else:
            return True

    def get_nan_output(self, metric):
        sstats_keys = [
            "mean",
            "median",
            "median",
            "p95",
            "p05",
            "k",
            "stdv",
            "mad",
            "n",
        ]

        topo_keys = [
            "b1",
            "b2",
            "b3",
            "ec",
        ]
        if "seg_" in metric:
            metrics = segm_names
            if "seg_sstats" in metric:
                metrics = [f"{n}_{k}" for n in segm_names for k in sstats_keys]
                return {m: np.nan for m in metrics}
            elif "seg_topology" in metric:
                metrics = [f"{n}_{k}" for n in segm_names + ["mask"] for k in topo_keys]
                return {m: np.nan for m in metrics}
            return {m: np.nan for m in metrics}
        else:
            return [np.nan]

    def get_default_output(self, metric):
        """Return the default output for a given metric when the mask is invalid and metrics cannot be computed."""
        METRIC_DEFAULT = {"cjv": 0}
        if metric not in METRIC_DEFAULT.keys():
            return [0.0, False]
        else:
            return METRIC_DEFAULT[metric]

    def _flatten_dict(self, d):
        """Flatten a nested dictionary by concatenating the keys with '_'."""
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(
                    {
                        k + "_" + kk: vv
                        for kk, vv in self._flatten_dict(v).items()
                    }
                )
            else:
                out[k] = v
        return out

    def eval_metrics_and_update_results(
        self,
        results,
        metric,
        args_dict,
    ):
        """Evaluate a metric and update the results dictionary."""
        
        try:
            out = self.metrics_func[metric](**args_dict)
        except Exception:
            if self.verbose:

                print(
                    f"EXCEPTION with {metric}\n" + traceback.format_exc(),
                    file=sys.stderr,
                )
            out = self.get_nan_output(metric)
        # Checking once more that if the metric is nan, we replace it with 0
        if isinstance(out, dict):
            out = self._flatten_dict(out)
            for k, v in out.items():
                results[metric + "_" + k] = v if not np.isnan(v) else 0.0
                results[metric + "_" + k + "_nan"] = np.isnan(v)
        else:
            if np.isnan(out[0]):
                out = (0, True)
            results[metric], results[metric + "_nan"] = out
        return results

    def evaluate_metrics(self, sr_path, mask_path, seg_path):
        """Evaluate the metrics for a given LR image and mask.

        Args:
            sr_path (str): Path to the LR image.
            seg_path (str, optional): Path to the segmentation. Defaults to None.

        Returns:
            dict: Dictionary containing the results of the metrics.
        """

        # Remark: Could do something better with a class here: giving flexible
        # features as input.
        # Reset the summary statistics
        self._sstats = None

        resample_to = 0.8
        
        imagec, maskc, seg_dict = self._load_and_prep_nifti(
            sr_path, mask_path, seg_path, resample_to
        )
        
        args_dict = {
            "image": imagec,
            "mask": maskc,
            "seg_dict": seg_dict,
            "vx_size": [resample_to] * 3,
        }
        if any(["seg_" in m for m in self._metrics]):
            assert seg_path is not None, (
                "Segmentation path should be provided "
                "when evaluating segmentation metrics."
            )
        results = {}
        for m in self._metrics:
            if self.verbose:
                print("\tRunning", m)
            results = self.eval_metrics_and_update_results(
                results, m, args_dict
            )
        return results

    def _check_metrics(self):
        """Check that the metrics are valid."""

        for m in self._metrics:
            if m not in self.metrics_func.keys():
                raise RuntimeError(
                    f"Metric {m} is not part of the available metrics."
                )

    def process_metric(
        self,
        metric,
        type="ref",
        **kwargs,
    ):
        """Wrapper to process the different categories of metrics.

        Args:
            metric (str): Name of the metric (in the list of available metrics).
            type (str, optional): Type of metric. Defaults to "ref". Available types are:
                - "ref": metric that is computed by comparing neighbouring slices.
                - "noref": metric that relies on individual slices
                - "seg": metric that make use of a segmentation.
            **kwargs: Additional processing to be done before evaluating the metric, detailed in the docstring of the corresponding function.
        """

        if type == "ref":
            return partial(
                self.preprocess_and_evaluate_metric, metric=metric, **kwargs
            )
        elif type == "noref":
            return partial(
                self.preprocess_and_evaluate_noref_metric,
                noref_metric=metric,
                **kwargs,
            )
        elif type == "seg":
            return partial(
                self.preprocess_and_evaluate_seg_metric,
                seg_metric=metric,
                **kwargs,
            )
        else:
            raise RuntimeError(
                f"Unknown metric type {type}. Please choose among ['ref', 'noref', 'seg']"
            )

    def load_and_format_seg(self, seg_path):
        """Load segmentation and format it to be used by the metrics"""

        seg_path = str(seg_path).strip()
        if seg_path.endswith(".nii.gz"):
            seg_ni = ni.load(seg_path)
            seg = squeeze_dim(seg_ni.get_fdata(), -1).astype(np.uint8)
            seg_remapped = np.zeros_like(seg)
            for label, target in enumerate(self.map_seg):
                if target is not None:
                    seg_remapped[seg == label] = target
            seg_ni = ni.Nifti1Image(
                seg_remapped,
                seg_ni.affine,
            )
            
        else:
            raise ValueError(
                f"Unknown file format for segmentation file {seg_path}"
            )
        # We cannot return a nifti object as seg_path might be .npz
        return seg_ni

    def _scale_intensity_percentiles(
        self, im, q_low, q_up, to_low, to_up, clip=True
    ):
        from warnings import warn

        a_min: float = np.percentile(im, q_low)  # type: ignore
        a_max: float = np.percentile(im, q_up)  # type: ignore
        b_min = to_low
        b_max = to_up

        if a_max - a_min == 0.0:
            warn("Divide by zero (a_min == a_max)", Warning)
            if b_min is None:
                return im - a_min
            elif a_max == 0.0:
                a_max = np.max(im)
            else:
                return im - a_min + b_min

        im = (im - a_min) / (a_max - a_min)
        if (b_min is not None) and (b_max is not None):
            im = im * (b_max - b_min) + b_min
        if clip:
            im = np.clip(im, b_min, b_max)

        return im

    def _preprocess_nifti(
        self, im_ni, mask_ni, seg_path, resample_to=0.8, mask_im=True, robust=True, bias_corr=False
    ):
        from nilearn.image import resample_img
        # ni.save(im_ni, f"test_{self.counter}.nii.gz")
        seg = self.load_and_format_seg(seg_path)
        if bias_corr:
            import SimpleITK as sitk
            def ni2sitk(im):
                im_sitk = sitk.GetImageFromArray(im.get_fdata().transpose(2,1,0))
                x,y,z = im.header.get_zooms()
                im_sitk.SetSpacing((float(x), float(y), float(z)))
                # get qform
                x,y,z = im.affine[:3,3]
                im_sitk.SetOrigin((float(x), float(y), float(z)))
                return im_sitk

            im_sitk = ni2sitk(im_ni)
            mask_sitk = ni2sitk(mask_ni)
            mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
            im_max, im_min = sitk.GetArrayFromImage(im_sitk).max(), sitk.GetArrayFromImage(im_sitk).min()
            # Normalize im with sitk
            rescaler = sitk.RescaleIntensityImageFilter()
            rescaler.SetOutputMaximum(1)
            rescaler.SetOutputMinimum(0)
            im_sitk = rescaler.Execute(im_sitk)

            # Correct bias field
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetBiasFieldFullWidthAtHalfMaximum(0.25)
            corrector.SetSplineOrder(3)
            corrected_im = corrector.Execute(im_sitk, mask_sitk)
            corrected_im = rescaler.Execute(corrected_im)
            img  = sitk.GetArrayFromImage(corrected_im)
            #Map back to original range
            img = (img  * (im_max - im_min)) + im_min
            im_ni = ni.Nifti1Image(img.transpose(2,1,0), im_ni.affine, im_ni.header)
        # ni.save(im_ni, f"test_{self.counter}_after_bias.nii.gz")

        img = (
            im_ni.get_fdata() * mask_ni.get_fdata()
            if mask_im
            else im_ni.get_fdata()
        )
        
        
        if robust:
            imgv = img[mask_ni.get_fdata()>0]
            img = self._scale_intensity_percentiles(
                img, 0.5, 99.5, 0, 1, clip=True
            )

            new_affine = np.diag([resample_to] * 3)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image_ni = resample_img(
                    ni.Nifti1Image(
                        img,
                        im_ni.affine,
                        im_ni.header,
                    ),
                    target_affine=new_affine,
                )
                mask_ni = ni.Nifti1Image(
                    mask_ni.get_fdata(), im_ni.affine, im_ni.header
                )
                mask_ni = resample_img(
                    mask_ni, target_affine=new_affine, interpolation="nearest"
                )
                seg = ni.Nifti1Image(
                    seg.get_fdata(), im_ni.affine, im_ni.header
                )
                seg = resample_img(
                    seg, target_affine=new_affine, interpolation="nearest"
                )
        else:
            image_ni = ni.Nifti1Image(
                        img,
                        im_ni.affine,
                        im_ni.header,
                    )
            

        def crop_stack(x, y):
            return get_cropped_stack_based_on_mask(
                x,
                y,
                boundary_i=5,
                boundary_j=5,
                boundary_k=5,
            )

        # seg_dict = {k: (seg == l).astype(np.uint8) for k, l in SEGM.items()}

        seg_dict = {
            k: crop_stack(
                ni.Nifti1Image(
                    (seg.get_fdata() == l).astype(np.uint8),
                    image_ni.affine,
                    image_ni.header,
                ),
                mask_ni,
            )
            for k, l in SEGM.items()
        }
        imagec = crop_stack(image_ni, mask_ni)
        #ni.save(imagec, f"test_{self.counter}_postproc.nii.gz")
        maskc = crop_stack(mask_ni, mask_ni)
        return imagec, maskc, seg_dict

    def _load_and_prep_nifti(self, sr_path, mask_path, seg_path, resample_to):
        image_ni = ni.load(sr_path)
        
        # zero_fill the Nan values
        image_ni = ni.Nifti1Image(
            np.nan_to_num(image_ni.get_fdata()),
            image_ni.affine,
            image_ni.header,
        )
        mask_ni = ni.load(mask_path)
        seg_ni = ni.load(seg_path)

        mask = np.clip(
            mask_ni.get_fdata() + (seg_ni.get_fdata() > 0).astype(int), 0, 1
        )
        mask = squeeze_dim(mask, -1)
        mask_ni = ni.Nifti1Image(mask, mask_ni.affine, mask_ni.header)
        imagec, maskc, seg_dict = self._preprocess_nifti(
            image_ni, mask_ni, seg_path, resample_to, robust=self.robust_prepro, bias_corr=self.correct_bias
        )

        def squeeze_flip_tr(x):
            """Squeeze_dim returns a numpy array"""
            return squeeze_dim(x, -1)[::-1, ::-1, ::-1].transpose(2, 1, 0)

        if imagec is not None:
            imagec = squeeze_flip_tr(imagec.get_fdata())
            maskc = squeeze_flip_tr(maskc.get_fdata())
            seg_dict = {
                k: squeeze_flip_tr(v.get_fdata()) for k, v in seg_dict.items()
            }
        return imagec, maskc, seg_dict

    def _remove_empty_slices(self, image, mask):
        s = np.flatnonzero(
            mask.sum(
                axis=(
                    1,
                    2,
                )
            )
            > 50  # Remove negligible amounts
        )
        imin, imax = s[0], s[-1]
        return image[imin:imax], mask[imin:imax]

    def _ssim(
        self,
        image,
        mask,
        seg_path=None,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
        use_window=True,
        window_size=3,
        **kwargs,
    ):

        if (
            image is None
            or mask is None
            or any([s < 7 for s in image.shape[1:]])
        ):
            # image is None when the mask is empty: nothing is computed.
            # Similarly, return None when the image is of size smaller than 7
            return np.nan, True
        metric_out = []
        isnan = False

        image, mask = self._remove_empty_slices(image, mask)

        datarange = image[mask > 0].max() - image[mask > 0].min()
        for i, img_i in enumerate(image):
            if use_window:
                l, r = window_size // 2, window_size - window_size // 2
                range_j = range(max(0, i - l), min(image.shape[0], i + r))
            else:
                range_j = range(0, image.shape[0])
            for j in range_j:
                im_i = img_i
                im_j = image[j]
                mask_curr = (
                    mask[i] * mask[j]
                    if mask_intersection
                    else ((mask[i] + mask[j]) > 0).astype(int)
                )
                m = (
                    ssim(im_i, im_j, mask_curr, datarange)
                    if compute_on_mask
                    else ssim(im_i, im_j, datarange)
                )
                # Try to not consider self-correlation
                if not np.isnan(m) and i != j:
                    metric_out.append(m)
                if np.isnan(m):
                    isnan = True

        if reduction == "mean":
            return np.mean(metric_out), isnan
        elif reduction == "median":
            return np.median(metric_out), isnan

    def _seg_sstats(self, image, segmentation):
        self._sstats = summary_stats(image, segmentation)
        return self._sstats

    def _seg_volume(self, image, segmentation):
        return volume_fraction(segmentation)

    def _seg_snr(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        snr_dict = {}
        for tlabel in segmentation.keys():
            snr_dict[tlabel] = snr(
                self._sstats[tlabel]["median"],
                self._sstats[tlabel]["stdv"],
                self._sstats[tlabel]["n"],
            )
        snr_dict["total"] = float(np.mean(list(snr_dict.values())))
        return snr_dict

    def _seg_cnr(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        out = cnr(
            self._sstats["WM"]["median"],
            self._sstats["GM"]["median"],
            self._sstats["WM"]["stdv"],
            self._sstats["GM"]["stdv"],
        )
        is_nan = np.isnan(out)
        return 0.0 if is_nan else out, is_nan

    def _seg_cjv(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        out = cjv(
            # mu_wm, mu_gm, sigma_wm, sigma_gm
            self._sstats["WM"]["median"],
            self._sstats["GM"]["median"],
            self._sstats["WM"]["mad"],
            self._sstats["GM"]["mad"],
        )
        is_nan = np.isnan(out)
        return 1000 if is_nan else out, is_nan

    def _seg_wm2max(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        out = wm2max(image, self._sstats["WM"]["median"])
        is_nan = np.isnan(out)
        return 0.0 if is_nan else out, is_nan

    def _seg_topology(self, image, segmentation):
        topo_dict = {}
        
        for tlabel in segmentation.keys():
            betti_numbers, ec = compute_topological_features(
                segmentation[tlabel]
            )
            b1, b2, b3 = betti_numbers
            topo_dict[f"{tlabel}_b1"] = b1
            topo_dict[f"{tlabel}_b2"] = b2
            topo_dict[f"{tlabel}_b3"] = b3
            topo_dict[f"{tlabel}_ec"] = ec
        mask = np.zeros_like(image)
        for tlabel in segmentation.keys():
            mask += segmentation[tlabel]

        betti_numbers, ec = compute_topological_features(mask)
        topo_dict["mask_b1"] = betti_numbers[0]
        topo_dict["mask_b2"] = betti_numbers[1]
        topo_dict["mask_b3"] = betti_numbers[2]
        topo_dict["mask_ec"] = ec
        return topo_dict

    def preprocess_and_evaluate_metric(
        self,
        metric,
        image,
        mask,
        *,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
        use_window=True,
        window_size=3,
        use_datarange=False,
        **kwargs,
    ):
        VALID_REDUCTIONS = ["mean", "median"]
        assert reduction in VALID_REDUCTIONS, (
            f"Unknown reduction function {reduction}."
            f"Choose from {VALID_REDUCTIONS}"
        )

        if image is None or mask is None:
            # image is None when the mask is empty: nothing is computed.
            return np.nan, True

        image, mask = self._remove_empty_slices(image, mask)
        metric_out = []
        isnan = False
        if use_datarange:
            if compute_on_mask:
                datarange = image[mask > 0].max() - image[mask > 0].min()
            else:
                datarange = image.max() - image.min()
        for i, img_i in enumerate(image):
            if use_window:
                l, r = window_size // 2, window_size - window_size // 2
                range_j = range(max(0, i - l), min(image.shape[0], i + r))
            else:
                range_j = range(0, image.shape[0])
            for j in range_j:
                im_i = img_i
                im_j = image[j]
                if compute_on_mask:
                    idx = (
                        np.where(mask[i] * mask[j])
                        if mask_intersection
                        else np.where(mask[i] + mask[j])
                    )
                    im_i, im_j = im_i[idx], im_j[idx]
                if use_datarange:
                    m = metric(im_i, im_j, datarange)
                else:
                    m = metric(im_i, im_j)

                if not np.isnan(m) and i != j:
                    metric_out.append(m)
                if np.isnan(m):
                    isnan = True
        if reduction == "mean":
            return np.mean(metric_out), isnan
        elif reduction == "median":
            return np.median(metric_out), isnan

    def preprocess_and_evaluate_noref_metric(
        self,
        noref_metric,
        image,
        mask,
        seg_path=None,
        *,
        compute_on_mask=True,
        flatten=True,
        **kwargs,
    ):
        if compute_on_mask:
            image = image[np.where(mask)]
        if flatten:
            metric = noref_metric(image.flatten())
        return metric, np.isnan(metric)

    def preprocess_and_evaluate_seg_metric(
        self, seg_metric, image, seg_dict, **kwargs
    ):
        return seg_metric(image, seg_dict)

    ### Filter-based metrics

    def _metric_filter(
        self,
        image,
        filter=None,
        **kwargs,
    ) -> np.ndarray:
        """Given a path to a LR image and its corresponding image,
        loads and processes the LR image, filters it with a `filter` from
        skimage.filters and returns the mean of the absolute value.

        Inputs
        ------
        filter:
            A filter from skimage.filters to be applied to the mask

        Output
        ------
        """

        assert (
            filter in SKIMAGE_FCT
        ), f"ERROR: {filter} is not a function from `skimage.filters`"

        filtered = filter(image)
        res = np.mean(abs(filtered - image))
        return res, np.isnan(res)
