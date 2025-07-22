# FetMRQC SR

[FetMRQC SR](https://arxiv.org/abs/2503.10156) is the super-resolution extension of FetMRQC [[paper1](https://arxiv.org/pdf/2304.05879.pdf),[paper2](https://arxiv.org/pdf/2311.04780.pdf)] is a tool for quality assessment (QA) and quality control (QC) of T2-weighted (T2w) fetal brain MR images. 

It builds on top of the utilities developed in the [FetMRQC repository](https://github.com/Medical-Image-Analysis-Laboratory/fetmrqc), a tool for the QC of low-resolution T2w scans.

It contains the tools needed

It consists of two parts.
1. A **rating interface** (visual report) to standardize and facilitate quality annotations of T2w fetal brain MRI images, by creating interactive HTML-based visual reports from fetal brain scans. It uses a pair of low-resolution (LR) T2w images with corresponding brain masks to provide snapshots of the brain in the three orientations of the acquisition in the subject-space. 
2. A **QA/QC model** that can predict the quality of given super-resolution reconstructed volumes. 

Given a list of SRR images listed using `qc_list_bids`, it then uses `srqc_segmentation` to compute the segmentations using BOUNTI [1] and extracts image quality metrics (IQMs) using `srqc_compute_iqms`. These IQMs can then be transformed in FetMRQC SR predictions using `srqc_inference`.

If you have found this useful in your research, please cite 
> Thomas Sanchez, Vladyslav Zalevskyi, Angeline Mihailov, Gerard Martí-Juan, Elisenda Eixarch, Andras Jakab, Vincent Dunet, Mériam Koob, Guillaume Auzias, Meritxell Bach Cuadra. (2025) **Automatic quality control in multi-centric fetal brain MRI super-resolution reconstruction.** [arXiv preprint arXiv:2503.10156](https://www.arxiv.org/abs/2503.10156)

## Installing FetMRQC_SR
To install FetMRQC SR, just create a new `conda` environment with python 3.9.0

```
conda create --name fetmrqc_sr python=3.9.0
```

Then, simply activate the environment and install `fetmrqc_sr` and its dependencies by running `pip install -e .`

## Generating reports for manual QC
After installing fetmrqc_sr, you will need to follow these steps to generate manual QC reports.
1. Given a [BIDS-formatted](https://bids.neuroimaging.io/index.html) dataset, get a CSV list of the data with `qc_list_bids` (use `--help` to see the detail). You will need to use the option `--skip_masks`.
2. Once you have your csv file, you can generate the visual reports for manual annotations using  
```
qc_generate_reports --bids_csv <csv_path> --out_dir <output_directory> --sr
```
3. You can then run `qc_generate_index` to generate an index file to easily navigate the reports.
After intalling fetmrqc_sr, you will need to generate a csv file with a `name` and an `im` column listing the path to the SRR volumes for which you want to generate the reports. If you do not have such a CSV, you can generate it using `qc_list_bids` (use `--help` to see the detail). If you do, please use the `--skip_masks` argument.

Once you have your csv file, you can generate the reports using  `qc_generate_reports --bids_csv <csv_path> --out_dir <output_directory> --sr`

Finally, you can run `qc_generate_index` to generate an index file to easily navigate the reports.

## Custom model training using FetMRQC SR
Once your manual ratings are done, you then train a custom QC model as follows.
1. Get back a CSV file using `qc_ratings_to_csv` in the folder where your ratings are stored.
2. Compute brain segmentations using `srqc_segmentation` and IQMs using `srqc_compute_iqms`. 
3. Train your custom models using the manual ratings with automatically extracted IQMs using `srqc_train_model`.

## References
[1] Uus, Alena U., et al. "BOUNTI: Brain vOlumetry and aUtomated parcellatioN for 3D feTal MRI." bioRxiv (2023).

## License
Copyright 2025 Medical Image Analysis Laboratory. 

## Acknowledgements
This project was supported by the ERA-net Neuron MULTIFACT – SNSF grant [31NE30_203977](https://data.snf.ch/grants/grant/203977).

