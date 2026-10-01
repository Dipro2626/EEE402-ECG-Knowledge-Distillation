# Datasets

All databases are public. They are **not stored in this repository** (each training CSV is 0.5–1.5 GB,
above GitHub's file limit). Download them from the links below and place them under `data/` exactly as shown.

## 1. Training data — Chapman-Shaoxing and Ningbo (single lead + 17 HRV features)

We used the redistributed single-lead CSV files:

- **Kaggle — CHAP_NING_PTB:** https://www.kaggle.com/datasets/tushartalukder11/chap-ning-ptb
  Files used: `chapman_data_without_others.csv`, `ningbo_data_without_others.csv`

Layout of each CSV row: columns 0–4999 = one ECG lead (500 Hz, 10 s); 5000–5016 = 17 HRV features
(bpm, mean_nn, SDNN, SDSD, RMSSD, CVNN, CVSD, median_nn, MAD_nn, MCV_nn, IQR_nn, SDRMSSD, prc20_nn, prc80_nn,
min_nn, max_nn, HTI); remaining columns = one-hot rhythm labels.

Original 12-lead source of both databases (Chapman University, Shaoxing People's Hospital, Ningbo First Hospital):
- **PhysioNet — A large scale 12-lead ECG database for arrhythmia study:** https://physionet.org/content/ecg-arrhythmia/1.0.0/ (DOI 10.13026/wgex-er52)

## 2. External test data — CPSC-2018 and Georgia (12-lead, used zero-shot)

- **PhysioNet/CinC Challenge 2021 ("Will Two Do?") v1.0.3:** https://physionet.org/content/challenge-2021/1.0.3/
  Folders used: `training/cpsc_2018/` and `training/georgia/`

Convert them to the parquet layout that `data.py` reads:

```
python scripts/convert_cinc2021_to_parquet.py --src <challenge-2021>/training/georgia   --out data/GEORGIA/tweleve_lead_georgia_wo_filters.parquet
python scripts/convert_cinc2021_to_parquet.py --src <challenge-2021>/training/cpsc_2018 --out data/CPSC18/cpsc_data_wo_filter.parquet
```

`evaluate_external.py --target cpsc` reads `data/CPSC18/cpsc_data.parquet` (a band-pass-filtered version we
received pre-processed); copy the unfiltered file to that name, or use `--target cpsc_nofilter`. In our runs
the two versions differed by less than 1 AF-F1 point. The converter does no filtering, so numbers may differ
slightly from ours.

## 3. External test data — MIT-BIH Arrhythmia Database

- **PhysioNet — MIT-BIH Arrhythmia Database v1.0.0:** https://physionet.org/content/mitdb/1.0.0/

Unzip so that the `.hea/.dat/.atr` files are in `data/mit-bih-arrhythmia-database-1.0.0/`.
`data.py` resamples 360 → 500 Hz and cuts 10-s windows inside single rhythm annotations.

## Expected layout

```
data/
├── chapman_data_without_others.csv
├── ningbo_data_without_others.csv
├── CPSC18/
│   ├── cpsc_data.parquet
│   └── cpsc_data_wo_filter.parquet
├── GEORGIA/
│   └── tweleve_lead_georgia_wo_filters.parquet
└── mit-bih-arrhythmia-database-1.0.0/
    ├── 100.hea, 100.dat, 100.atr, ...
```

A different location can be used with the environment variables `EXGNET_DATA_DIR` (CSV/parquet folder) and
`EXGNET_MITBIH_DIR` (MIT-BIH folder).

## Licences and citation

Use each database under its own licence and cite its creators:

- J. Zheng et al., "A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients," *Scientific Data*, 7, 48, 2020.
- M. A. Reyna et al., "Will two do? Varying dimensions in electrocardiography: the PhysioNet/Computing in Cardiology Challenge 2021," *Computing in Cardiology*, 2021.
- F. Liu et al., "An open access database for evaluating the algorithms of ECG rhythm and morphology abnormality detection," *J. Med. Imaging Health Inform.*, 8(7), 2018.
- G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Eng. Med. Biol. Mag.*, 20(3), 2001.
- A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, 101(23), 2000.
