⚠️ **NOTE**: As mentioned in the paper, we authors **do not have permission to share data**. We are sorry for unconvinence caused by this. Insteadly, we provide noise data with the same shape of the original dataset to make everything runable and easy to understand. You can easily replace them with your own dataset to run the model. 

### 🗂️ Introduction abotu dataset structure

- `toc_processed` contains the data for the *Coagulant dosage determination using deep learning-based graph attention multivariate time series forecasting model* paper. Where the data is by collected by hour. This dataset contains total 8 features.
- `ppm_original` cotians the data collected by minutes in original Excel format.
    - `ppm_original/ppm.csv` the original datafile.
- `ppm_processed` contains the data collected by minutes after pre-processing by the `notebook/00_data_preprocessing.ipynb` notebook.
    - `ppm_processed/ppm.npy` the pre-processed datafile in Numpy format, which contains all 11 features: 'Raw_Flow', 'Raw_Temp', 'Raw_EC', 'Raw_TB', 'Raw_Alk', 'Raw_pH', 'Raw_TOC', 'Chl_Pre', 'Sed_Chl', 'Sed_TB', 'Chl_Mid'.
    - `ppm_processed/datetime.npy` the pre-processed datetime file in Numpy format. We seperately store the datetime file because the datetime file is not used in the model training.

```
.
└── data/
    ├── ppm_original/ - not included 
    │   └── ppm.csv - not included 
    ├── ppm_processed/ - not included 
    │   └── ppm.npy
    ├── toc_processed - included as noise data
    │   ├── scale_2016-2020.npy - 2016~2020 features + coagulant dosage
    │   ├── scale_2016-2020_sed.npy - 2016~2020 features + sedimentation
    │   ├── scale_2021.npy - 2021 features + coagulant dosage
    │   └── scale_2021_sed.npy - 2021 features + sedimentation
    └── README.md
```
