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

## Getting started

### Requirements

- [MATLAB R2023b](https://uk.mathworks.com/help/matlab/release-notes.html)
- [Deep Learning Toolbox](https://uk.mathworks.com/help/deeplearning/release-notes.html)
- [Statistics and Machine Learning Toolbox](https://uk.mathworks.com/help/stats/release-notes.html)
- [Fuzzy Logic Toolbox](https://uk.mathworks.com/help/fuzzy/release-notes.html)

To visualise the results, the following toolbox is recommended:
- [Mapping Toolbox](https://uk.mathworks.com/help/map/release-notes.html)

<!--
To accelerate training, the following toolbox is recommended:
- [Parallel Computing Toolbox](https://uk.mathworks.com/help/parallel-computing/release-notes.html)
-->

Download or clone this repository to your machine and open it in MATLAB.

### File description

The script [x] performs data preprocessing.

The script [y] ...

## Model details

The model provided in this repository...

A recurrent sequence-to-sequence encoder-decoder model with attention is implemented.

The advantage of this type of model is that it can accept variable input and output sequence lengths.

The encoder-decoder model architecture is organised as follows:

1. Encoder: BiLSTM
2. Aggregate function: Attention mechanism
3. Decoder: LSTM

## Metrics and evaluation

The model is trained using the Mean Absolute Error (MAE) loss and evaluated using the mean great circle distance between predicted and target sequences on the test set (MAE<sub>gc</sub>).

## License

The license is available in the [LICENSE file](LICENSE) in this repository.

Where users can get help with your project
Who maintains and contributes to the project
