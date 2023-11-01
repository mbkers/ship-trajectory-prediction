# ship-trajectory-prediction

**What the project does**:

This repository contains MATLAB files to carry out ship trajectory prediction on AIS data using Recurrent Neural Networks (RNNs).

The ship trajectory prediction model forms a core component of a larger, real-time (or operationally relevant) anomaly detection workflow as part of the [Nereus project](https://oceaninnovationchallenge.org/oceaninnovators-cohort2#cbp=/ocean-innovations/space-based-maritime-surveillance).

In the context of the anomaly detection workflow, the ship trajectory prediction model is designed to be applied to anomalous events such as AIS "shut-off" events (i.e. no AIS transmission for a specified duration).

The ship's trajectory before the "shut-off" is input to the model and a trajectory prediction is output for a user-defined duration (e.g. 2.5 hours).

An assessment is made of the predicted trajectory based on a contextual analysis and, if certain risk thresholds are met, satellite "tip and cue" actions are triggered.

*Note that this repository covers only the ship trajectory prediction model and does not cover the other steps of the anomaly detection workflow*.

<!-- It is important to note that the ship trajectory prediction model can also be used more generally; for example, in interpolation (where AIS data is available) ... -->

**Why the project is useful**:

The model's predictive capability enables an operational association of ship observations from self-reporting systems such as AIS and imaging satellite technologies such as SAR and optical which are unsynchronised and need to be scheduled in advance.

In other words, if a vessel of interest is in motion a trajectory prediction is required to define the acquisition area around the probable location of the vessel when the satellite will pass over it.

The goal is to enable end users to easily identify the anomalies that require their attention by virtue of the vast amount of satellite data made available to them, all while considering their limited time and resources.

<!-- Unlike the majority of the literature, We can contribute in the following two ways: generic vessel trajectories and generic ship type. -->

## Getting started

### Requirements

- [MATLAB R2023b](https://uk.mathworks.com/help/matlab/release-notes.html)
- [Deep Learning Toolbox](https://uk.mathworks.com/help/deeplearning/release-notes.html)
- [Statistics and Machine Learning Toolbox](https://uk.mathworks.com/help/stats/release-notes.html)
- [Fuzzy Logic Toolbox](https://uk.mathworks.com/help/fuzzy/release-notes.html)
- [Mapping Toolbox](https://uk.mathworks.com/help/map/release-notes.html) <!-- To visualise the results, the following toolbox is recommended: -->
- [Parallel Computing Toolbox](https://uk.mathworks.com/help/parallel-computing/release-notes.html)

Download or clone this repository to your machine and open it in MATLAB.

### File description

Firstly, run the script [s_data_preprocessing.m](s_data_preprocessing.m). This script performs the following data preprocessing steps:

1. Import data:
   - AIS data is downloaded from [Marine Cadastre](https://marinecadastre.gov/) with the following parameters:
     - Date: 2021-04-30 to 2021-05-30
     - Longitude limits (min to max): -78 to -74.3
     - Latitude limits (min to max): 31.8 to 37.3
     - File size: 1003.62 MB
   <!-- - The study area is similar to the one defined in [Chen et al., 2020](https://doi.org/10.3390/ijgi9020116) (North Carolina, USA). -->
2. Missing and invalid data
3. Aggregate data into sequences:
   - The data is aggregated into sequences/trajectories based on the MMSI number.
   - At the same time, the implied speed and implied bearing features are calculated from the latitude and longitude data. This is because the latitude and longitude data availability is greater than the Speed Over Ground (SOG) and Course Over Ground (COG) data. 
   - Next, the sequences are segmented into subsequences/subtrajectories based on a time interval threshold. In other words, if an AIS sequence contains gaps in transmission for longer than a specified time threshold then it is further split into subsequences.
4. Resample subsequences:
   - The subsequences are resampled to regularly spaced time intervals by interpolating the data values.
5. Feature transformation:
   - A feature transformation is done to detrend the data. Specifically, the difference between consecutive observations for all features is done. <!-- (similar to [Chen et al., 2020](https://doi.org/10.3390/ijgi9020116)) -->
6. Filter subsequences by motion pattern:
   - The subsequences are filtered according to if they intersect a set of Polygonal Geographical Areas (PGAs) (similar to [Capobianco et al., 2021](https://doi.org/10.1109/TAES.2021.3096873)), which can be thought of as a type of clustering.
7. Sliding window:
   - A sliding window is applied to the subsequences. Specifically, for each subsequence an input and response window of equal size is created. The windows are then shifted along by a specified time step. An example of this process is given below:

     ![Sliding window example.](/assets/images/sliding_window.png)

8. Prepare training, validation and test data splits:
   - The input and response features are selected.
   - The data is split into training (80%), validation (10%) and test (10%) sets.
   - The data is also rescaled. <!-- to the range [0,1]. -->
9. Save data

Secondly, run the script [s_net_encoder_decoder.m](s_net_encoder_decoder.m) which creates, trains and tests a recurrent sequence-to-sequence encoder-decoder model with attention. The encoder-decoder network architecture is shown in the [Model details](#model-details) section.

The model is defined as a Model Function as opposed to a typical MATLAB layer array, layerGraph or `dlnetwork` object. For more details on their differences see [this documentation](https://uk.mathworks.com/help/deeplearning/ug/define-custom-training-loops-loss-functions-and-networks.html#mw_7173ce81-4cb6-4221-ac2e-5688aa0fa950).

The advantage of this model is that it accepts variable-length input and output sequences. Originally, a stacked BiLSTM model defined as a `dlnetwork` object was implemented at the beginning of the project, but this required fixed-length input and output sequences.

Moreover, the [s_net_encoder_decoder.m](s_net_encoder_decoder.m) script includes the following steps:

1. Load data
2. Preprocess data
3. Initialise model parameters
4. Define model function(s)
5. Define model loss function
6. Specify training options
7. Train model
8. Test model
9. Make predictions (example)

## Model details

The recurrent sequence-to-sequence encoder-decoder model with attention is shown in the schematic below:

![Encoder-decoder model.](/assets/images/net_encoder_decoder.png)

The encoder uses a bidirectional LSTM (BiLSTM) operation and the decoder uses an LSTM operation followed by an attention mechanism.

<!-- The architecture of this model is inspired by [Capobianco et al., 2021](https://doi.org/10.1109/TAES.2021.3096873). -->

## Metrics and evaluation

The model is trained using the Mean Absolute Error (MAE) loss and evaluated using the mean great circle distance between predicted and target sequences on the test set (MAE<sub>gc</sub>).

Quantitative results:

X

Qualitative results:

X

## Runtime

The total training time was X running on an NVIDIA GeForce RTX 3080 with approximately 10 GB of memory.

## Limitations

Known limitations include:

- The performance of the model may be sensitive to the training dataset. <!-- (Capobianco et al., 2021) -->
- A simplification has been made which includes training the model exclusively on cargo vessel types from a specific geographic region.

## Next steps

Aside from code improvements and additions, some next steps include:

- Hyperparameter optimisation, scaling up training data as well as training on university HPC clusters.
- Generalise the model to work on various vessel types from different geographic regions.
- To investigate other architectures such as a probabilistic RNN as opposed to a deterministic RNN (by way of including an MDN layer) with a KDE applied to the multiple outputs. This has the advantage of incorporating model uncertainty. <!-- (e.g. Encoder-Decoder + MDN RNN) -->
- Additionally, transformers.

## Resources

MATLAB documentation used as part of this project:
- A
- B
- C

## License

The license is currently being assessed and will be updated shortly.

<!-- The license is available in the [LICENSE file](LICENSE) in this repository. -->

## Contact

The primary contact for this project is X.
