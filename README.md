# ship-trajectory-prediction

<div align="center">
    <img src="/assets/images/summary_figure.webp" height="320" alt="Summary figure">
</div>

**What the project does**:

This repository hosts MATLAB files designed to carry out ship trajectory prediction on AIS data using Recurrent Neural Networks (RNNs).

The ship trajectory prediction model forms a core component of an operationally relevant (or "real-time") anomaly detection workflow, as part of the [Nereus project](https://oceaninnovationchallenge.org/oceaninnovators-cohort2#cbp=/ocean-innovations/space-based-maritime-surveillance).

Within the context of this anomaly detection workflow, the ship trajectory prediction model is designed to respond to anomalous events such as AIS "shut-off" events, signifying the absence of AIS transmissions for a predefined duration.

The model takes the ship's trajectory leading up to the "shut-off" event as input and generates a trajectory prediction for a user-defined period (e.g. 2.5 hours).

An assessment of the predicted trajectory is then conducted through a contextual analysis. If certain risk thresholds are exceeded, this triggers satellite "tip and cue" actions.

*Note that this repository covers exclusively the ship trajectory prediction model and does not cover the other components of the anomaly detection workflow*.

**Why the project is useful**:

The model's predictive capability enables the operational integration of ship observations from asynchronous self-reporting systems, like AIS, with imaging satellite technologies such as SAR and optical sensors, which require scheduling in advance.

In practical terms, if a target vessel is in motion, a trajectory prediction becomes essential for determining the acquisition area surrounding the likely position of the vessel when a satellite is scheduled to pass overhead.

The end goal is to enable end-users to efficiently identify anomalies demanding their attention amidst the (fortunately) abundant satellite data available to them, all within the constraints of their limited time and resources.

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
   - The AIS data is originally downloaded from [Marine Cadastre](https://marinecadastre.gov/) with the following parameters:
     - Date: 2021-04-30 to 2021-05-30
     - Longitude limits (min to max): -78 to -74.3
     - Latitude limits (min to max): 31.8 to 37.3
     - The study area is the North Carolina coastline, USA. (This region is similar to the one defined in Chen et al., [2020](https://doi.org/10.3390/ijgi9020116).)
     - For access to the original CSV file (file size: 1003.62 MB) or for any inquiries, please feel free to raise an issue in this GitHub repository.
2. Manage missing and invalid data
3. Aggregate data into sequences:
   - The data is aggregated into sequences or trajectories based on the Maritime Mobile Service Identity (MMSI) number, which uniquely identifies a vessel.
   - Simultaneously, the implied speed and implied bearing features are calculated based on the latitude and longitude data. This is because of the higher availability of latitude and longitude data as compared to the Speed Over Ground (SOG) and Course Over Ground (COG) data.
   - Next, the sequences are segmented into subsequences or subtrajectories using a predefined time interval threshold. In other words, if an AIS sequence has transmission gaps exceeding a specified time threshold, it is further split into smaller subsequences.
4. Resample subsequences:
   - The subsequences are resampled to regularly spaced time intervals by interpolating the data values.
5. Transform features:
   - A feature transformation is done to detrend the data (Chen et al., [2020](https://doi.org/10.3390/ijgi9020116)). Specifically, the difference between consecutive observations is calculated for each feature. The transformed features are named similarly to the original ones, but with a delta symbol (Δ) or suffix "_diff" added to indicate the difference calculation, for example, the transformation of 'lat' (latitude) becomes 'Δlat' or 'lat_diff'.
6. Filter subsequences by motion pattern:
   - The subsequences are filtered according to whether they intersect a set of Polygonal Geographical Areas (PGAs) (Capobianco et al., [2021](https://doi.org/10.1109/TAES.2021.3096873)), which can be considered as a type of clustering method.
7. Apply a sliding window:
   - A sliding window technique is applied to the subsequences, producing extra sequences from each one (these could be termed as "subsubsequences"). These generated sequences then serve as the input and response data for creating the model. Specifically, for each subsequence an input window and a response window of equal size are created. The windows are then progressively shifted by a specified time step. An illustrative example of this process is provided below:

     <img src="/assets/images/sliding_window.png" width="500"> <!-- ![Sliding window example.](/assets/images/sliding_window.png) -->

8. Prepare training, validation and test data splits:
   - The input and response features are selected. Currently, `lat_diff` and `lon_diff` are selected from the available features. <!-- which includes `lat`, `lon`, `speed_implied`, `bearing_implied`, `lat_diff`, `lon_diff`, `speed_implied_diff` and `bearing_implied_diff`. -->
   - The data is partitioned into training (80%), validation (10%) and test (10%) sets.
   - Additionally, the data is rescaled to the range [-1,1].
9. Save data variables

Secondly, run the script [s_net_encoder_decoder.m](s_net_encoder_decoder.m) which creates, trains and tests a recurrent sequence-to-sequence encoder-decoder model with attention. The encoder-decoder network architecture is detailed in the [Model details](#model-details) section.

(Note that the model is defined as a Model Function rather than a conventional MATLAB layer array, layerGraph or `dlnetwork` object. For details on their differences, refer to [this documentation](https://uk.mathworks.com/help/deeplearning/ug/define-custom-training-loops-loss-functions-and-networks.html#mw_7173ce81-4cb6-4221-ac2e-5688aa0fa950).)

The notable advantage of this model is its capacity to handle input and output sequences of varying lengths. Initially, a stacked BiLSTM model was implemented but this required fixed-length input and output sequences. <!-- (defined as a `dlnetwork` object) -->

Furthermore, the [s_net_encoder_decoder.m](s_net_encoder_decoder.m) script includes the following steps:

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

The recurrent sequence-to-sequence encoder-decoder model with attention is shown in the diagram below:

![Encoder-decoder model.](/assets/images/net_encoder_decoder.png)

The input sequence is represented as a sequence of points $x_1, x_2, \ldots, x_t$, where $x_i = (\Delta \text{lat}_i, \Delta \text{lon}_i)$ for $i = 1, 2, \ldots, t$. The same notation is used for the prediction sequence $y_1, y_2, \ldots, y_t$.

Moreover, the input sequence is passed through the encoder, which produces an encoded representation of the input sequence as well as a hidden state that is used to initialise the decoder's hidden state.

The encoder consists of a bidirectional LSTM (BiLSTM) layer. <!-- operation --> The decoder makes predictions at each time step, using the previous prediction as input (for inference), and outputs an updated hidden state and context values. <!-- For training: TF; for inference: autoregressive decoding. -->

The decoder passes the input data concatenated with the input context through an LSTM layer, and takes the updated hidden state and the encoder output and passes it through an attention mechanism to determine the context vector.

The LSTM output follows a dropout layer before being concatenated with the context vector and passed through a fully connected layer for regression.

<!-- The network architecture is similar to the one presented in Capobianco et al., [2021](https://doi.org/10.1109/TAES.2021.3096873). -->

## Metrics and evaluation

The model is trained using the [Huber loss](https://uk.mathworks.com/help/deeplearning/ref/dlarray.huber.html) between predicted and target sequences from the training set. The model is then evaluated using the mean and max great circle distance between predicted and target sequences from the test set. Using the Huber loss during training and a physical distance (like the great circle distance) for evaluation combines the benefits of a robust training process with an evaluation metric that provides a direct real-world interpretation of the model’s performance.  <!-- Mean Absolute Error (MAE) loss -->

Moreover, a prediction time of 2.5 hours is specified in the following results. The training settings can be found in the script [s_net_encoder_decoder.m](s_net_encoder_decoder.m).

### Quantitative results:

<img src="/assets/images/plot_gc_dist.png" width="500"> <!-- ![Plot of mean great circle distance.](/assets/images/plot_gc_dist_mean.png) -->

The histograms show the mean and max great circle distance between predicted and target sequences on the test set (1802 (sub)sequences in total). The mode error is 1-2 km and the overall mean error, often referred to as the grand mean or pooled mean error, is 3.59 km and 7.22 km for the mean and max great circle distance, respectively.

### Qualitative results:

<img src="/assets/images/fig_qual_1.png" width="225"> <img src="/assets/images/fig_qual_2.png" width="225"> <img src="/assets/images/fig_qual_3.png" width="225"> <img src="/assets/images/fig_qual_4.png" width="225">

<!-- ![Straight line trajectory.](/assets/images/fig_qual_1.png) ![Straight line trajectory.](/assets/images/fig_qual_2.png) ![Curved trajectory.](/assets/images/fig_qual_3.png) ![Curved trajectory.](/assets/images/fig_qual_4.png) -->

### Qualitative results (high error examples):

<img src="/assets/images/fig_qual_high_error_1.png" width="225"> <img src="/assets/images/fig_qual_high_error_2.png" width="225">

In general, the model demonstrates satisfactory performance for common types of motion; however, there are a small number of instances where the model fails to accurately predict the true trajectory. These instances are currently being scrutinised, and it is suspected that they may be attributed to inaccuracies in the data preprocessing.

## Runtime

The total training time for 100 epochs was approximately 1 hour after running on an NVIDIA GeForce RTX 3080 with 10 GB of memory.

## Limitations

Known limitations include:

- Sensitivity to the training dataset: The model's performance may be influenced by the composition and quality of the training data.
- Geographic and vessel type specificity: The model has been trained solely on cargo vessel types from a particular geographic region, which may restrict its generalisability to other vessel types and regions.

These limitations are acknowledged and should be taken into consideration when applying the model in different contexts.

## Next steps

### Short term:

To enhance the model's capabilities and to address its limitations, several avenues for improvement are being explored, including:

1. Increase training data: Significantly expand the training dataset.
2. Utilise HPC clusters at the university to facilitate efficient processing of large datasets.
3. Hyperparameter optimisation: Fine tune the model by systematically optimising hyperparameters, such as learning rates, batch sizes, and network architecture. This can lead to better performance and generalisability.
4. Input features: Investigate different combinations of input features to identify the most informative ones.
5. Generalisation to various vessel types and regions: Modify the model to work effectively with different vessel types and geographic regions.
6. Code improvements and additions: Continuously improve and expand the codebase, addressing issues and adding new features to enhance the model's flexibility and usability.

### Medium term:

1. Probabilistic RNNs: Investigate the use of probabilistic RNNs by incorporating a Mixture Density Network (MDN) layer and applying Kernel Density Estimation (KDE) to the multiple outputs. This approach provides the advantage of capturing model uncertainty. <!-- Implementing a model that combines an Encoder-Decoder architecture with MDN RNN is one option to explore. Additionally, consider implementing techniques like beam search to improve sequence generation. -->
2. Transformer models: Investigate the applicability of transformer-based architectures in the context of ship trajectory prediction.

### Long term:

1. Convert MATLAB code to PyTorch.

## References

MATLAB documentation:
- [Custom Training Loops](https://uk.mathworks.com/help/deeplearning/deep-learning-custom-training-loops.html)
- [Train Deep Learning Model in MATLAB](https://uk.mathworks.com/help/deeplearning/ug/training-deep-learning-models-in-matlab.html)
- [Define Custom Training Loops, Loss Functions, and Networks](https://uk.mathworks.com/help/deeplearning/ug/define-custom-training-loops-loss-functions-and-networks.html)
- [Sequence-to-Sequence Translation Using Attention](https://uk.mathworks.com/help/deeplearning/ug/sequence-to-sequence-translation-using-attention.html)

Journal articles:
- Capobianco et al., [2021](https://doi.org/10.1109/TAES.2021.3096873).
- Chen et al., [2020](https://doi.org/10.3390/ijgi9020116).
- Sørensen et al., [2022](https://doi.org/10.3390/s22052058).
- More to follow (including a literature review).

## License

The license is available in the [LICENSE file](LICENSE.txt) in this repository.

<!-- The license is under evaluation and will be updated in the near future. -->
