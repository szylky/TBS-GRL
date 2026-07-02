# Audit-Budget-Aware Provider Risk Ranking

This repository contains the companion code for the paper **Audit-Budget-Aware Provider Risk Ranking in Big Healthcare Data via Relationally Enhanced Label-Scarce Temporal Representation Learning**.

The code implements a provider-level risk ranking pipeline for large-scale healthcare claims data. The workflow includes data preprocessing, temporal representation learning, relational graph construction, risk classification, and audit-budget-aware provider ranking evaluation.

## Repository Structure

- `SZYLKY_DMEPOS/`: code for the DMEPOS dataset.
- `SZYLKY_Part_B/`: code for the Medicare Part B dataset.
- `SZYLKY_Part_D/`: code for the Medicare Part D dataset.
- `batch_run_5datasets.py`: script for running five dataset splits in batch.
- `requirements.txt`: Python dependency list.

Each dataset folder contains the following main files and directories in numeric order:

- `00_Global_Config/`: configuration files and hyperparameters.
- `01_Dataset/`: default location for input datasets.
- `02_Data_Preprocessing/`: provider-level feature construction.
- `03_Temporal_Modeling/`: temporal representation learning and relational graph construction.
- `04_Group_Classification/`: provider risk classification, ranking, and audit-budget evaluation.
- `05_Outputs/`: default output location for training and testing results.
- `Train.py`: runs data preprocessing, temporal modeling, and risk classification training.
- `Test.py`: loads the trained model and performs inference and evaluation.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later is recommended.

## Running a Single Dataset

Example for DMEPOS:

```bash
python SZYLKY_DMEPOS/Train.py
python SZYLKY_DMEPOS/Test.py
```

The same pattern can be used for `SZYLKY_Part_B` and `SZYLKY_Part_D`.

## Running Five Dataset Splits

```bash
python batch_run_5datasets.py --project DMEPOS
python batch_run_5datasets.py --project Part_B
python batch_run_5datasets.py --project Part_D
```

By default, the batch runner uses dataset indices `0` through `4`.

## Data

Input data should be placed under the `01_Dataset/` directory in each dataset folder. Large raw healthcare datasets are not included in this repository.
