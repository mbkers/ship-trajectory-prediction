# ship-trajectory-prediction

**What the project does**:

This repository contains MATLAB files to carry out ship trajectory prediction on AIS data using Recurrent Neural Networks (RNNs).

The ship trajectory prediction model forms a core component of a larger real-time (or operationally relevant) anomaly detection workflow as part of the [Nereus project](https://oceaninnovationchallenge.org/oceaninnovators-cohort2#cbp=/ocean-innovations/space-based-maritime-surveillance).

In the context of the anomaly detection workflow, the ship trajectory prediction model is designed to be applied to anomalous events such as AIS "shut-off" events (i.e., no AIS transmission for a specified duration).

The ship's trajectory before the "shut-off" is input to the model and a trajectory prediction is output for a user-defined duration (e.g., 2.5 hours).

An assessment is made of the predicted trajectory based on a contextual analysis and, if certain risk thresholds are met, satellite "tip and cue" actions are triggered.

*Note that this repository covers only the ship trajectory prediction model and does not cover the other mentioned steps of the anomaly detection workflow*.

<!-- It is important to note that the ship trajectory prediction model can also be used more generally; for example, in interpolation (where AIS data is available) ... -->

**Why the project is useful**:

The model's predictive capability enables an operational association of ship observations from self-reporting systems such as AIS and imaging satellite technologies such as SAR and optical which are unsynchronised and need to be scheduled in advance.

In other words, if a vessel of interest is in motion a trajectory prediction is required to define the acquisition area around the probable location of the vessel when the satellite will pass over it.

The goal is to enable end users to easily identify the anomalies that require their attention by virtue of the vast amount of satellite data made available to them, all while considering their limited time and resources.

<!--
Unlike the majority of the literature, We can contribute in the following two ways: generic vessel trajectories and generic ship type.
-->

## Getting started

### Requirements

- [MATLAB R2023b](https://uk.mathworks.com/help/matlab/release-notes.html)
- [Deep Learning Toolbox](https://uk.mathworks.com/help/deeplearning/release-notes.html)
- [Statistics and Machine Learning Toolbox](https://uk.mathworks.com/help/stats/release-notes.html)
- [Fuzzy Logic Toolbox](https://uk.mathworks.com/help/fuzzy/release-notes.html)

To visualise the results, the following toolbox is recommended:
- [Mapping Toolbox](https://uk.mathworks.com/help/map/release-notes.html)

To accelerate training in the script [s_net_stacked_bilstm.m](s_net_stacked_bilstm.m), the following toolbox is recommended:
- [Parallel Computing Toolbox](https://uk.mathworks.com/help/parallel-computing/release-notes.html)

Download or clone this repository to your machine and open it in MATLAB.

### File description

The script [s_data_preprocessing.m](s_data_preprocessing.m) carries out data preprocessing which is ordered as follows:

1. Import data:
  - AIS data is downloaded from [Marine Cadastre](https://marinecadastre.gov/) with the following parameters:
    - From = 2021-04-30
    - To = 2021-05-30
    - X Min = -78
    - Y Min = 31.79999999999997
    - X Max = -74.3
    - Y Max = 37.300000000000026
    - File Size = 1003.62 mb
  - The study area is similar to the one defined in [Chen et al., 2020](https://doi.org/10.3390/ijgi9020116) (North Carolina, USA).
2. Missing and invalid data
3. Aggregate data into sequences:
  - The data is first aggregated into sequences based on the MMSI number.
  - Next, the sequences are segmented into subsequences based on a time interval threshold.
  - Additionally, the implied speed is calculated from the latitude and longitude data and included as a new feature.
4. Resample subsequences:
   - The subsequences are resampled to regular time intervals by using interpolation.
5. Feature transformation:
  - A feature transformation is made to detrend the data, similar to [Chen et al., 2020](https://doi.org/10.3390/ijgi9020116).
6. Filter subsequences by motion pattern:
  - A type of clustering is implemented similar to [Capobianco et al., 2021](https://doi.org/10.1109/TAES.2021.3096873).
7. Sliding window:
  - A sliding window is implemented which is common in the literature.
8. Prepare training, validation and test data

The script [s_net_encoder_decoder.m](s_net_encoder_decoder.m) creates, trains and tests a recurrent sequence-to-sequence encoder-decoder model with attention (by using functions rather than a MATLAB layer array, layerGraph or dlnetwork object) for ship trajectory prediction.

The architecture of this model is inspired by [Capobianco et al., 2021](https://doi.org/10.1109/TAES.2021.3096873).

The script [s_net_stacked_bilstm.m](s_net_stacked_bilstm.m) is an early model that is defined as a `dlnetwork` object as opposed to the model in script [s_net_encoder_decoder.m](s_net_encoder_decoder.m) which is defined as a Model Function.

This model is inspired by [Chen et al., 2020](https://doi.org/10.3390/ijgi9020116).

For more details on their difference see [here](https://uk.mathworks.com/help/deeplearning/ug/define-custom-training-loops-loss-functions-and-networks.html#mw_7173ce81-4cb6-4221-ac2e-5688aa0fa950).

## Model details

The model provided in this repository...

A recurrent sequence-to-sequence encoder-decoder model with attention is implemented.

The encoder-decoder model architecture is organised as follows:

1. Encoder: BiLSTM
2. Aggregate function: Attention mechanism
3. Decoder: LSTM

The advantage of this model is that it accepts variable-length input and output sequences.

## Metrics and evaluation

The model is trained using the Mean Absolute Error (MAE) loss and evaluated using the mean great circle distance between predicted and target sequences on the test set (MAE<sub>gc</sub>).

## Runtime

## Limitations

Limitations.

## Resources

MATLAB documentation used as part of this project:
- A
- B
- C

## License

The license is currently being assessed and will be updated shortly.

<!--
The license is available in the [LICENSE file](LICENSE) in this repository.
-->

Where users can get help with your project
Who maintains and contributes to the project
