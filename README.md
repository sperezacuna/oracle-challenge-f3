## About The Project

This is an EUR/USD pair markey dynamics prediction model based on deep neural networks built under the specification for the third challenge of the contest [Reto Enseña Oracle](https://nuwe.io/dev/competitions/reto-ensena-oracle-espana/modelo-predictivo-reto-3), hence the generated model will be capable identifying a 3-day-ahead market evolution indicator given a set of historic (strictly previous) data.

The main functionality of the project can be summarized as follows:

* Clean the [provided training dataset](https://challenges-asset-files.s3.us-east-2.amazonaws.com/0-challenges_data/2023_04/Oracle_3rd_challenge/training_set.csv), particularly:
  * Delete outliers, implementing [interquartile range](https://en.wikipedia.org/wiki/Interquartile_range) as base indicator.
  * Fill missing open (close) data transitively with following (previous) day close (open) value.
  * Fill missing or deleted high and low prices by inference using bayesian regression along each tuple (to no fall into [look-ahead bias](https://analyzingalpha.com/look-ahead-bias)).
  * Assert data coherence; if data is not consistent, adjust to min-max values.
  * Recalculate all incorrect labels according to APEX observatory analysis findings (further insights on this analysis can be found in _presentation.pdf_).
* Merge the [training](https://challenges-asset-files.s3.us-east-2.amazonaws.com/0-challenges_data/2023_04/Oracle_3rd_challenge/training_set.csv) and [testing](https://challenges-asset-files.s3.us-east-2.amazonaws.com/0-challenges_data/2023_04/Oracle_3rd_challenge/testing.csv) datasets, and calculate all missing labels, which will serve the purpose of ground truth and **never be included in the training data** at any point in time **prior to its inference** with a margin of 3 days (as found by APEX insights on the data).
* Create a basic (but deep) model as per the topology of a [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron).
* Initially train the model with all data _(but three days)_ previous to the first testing sample.
* For each following testing sample, use the current model to infer its label, giving it to the model along with a certain number of unlabeled contiguous previous samples. After that, go on with model training for a lesser amount of epochs with a lesser amount of data corresponding to the last N+1-3 days (which will serve the purposes of a sliding window of time).
* Save the results as a _json_ file.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built Using

Base technologies:

* [APEX](https://apex.oracle.com/es/)
* [Python](https://www.python.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)

Additional dependencies:

* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Sklearn](https://scikit-learn.org/stable/)

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

Given that [Python 3.9+](https://www.python.org/downloads/) and [conda](https://docs.conda.io/) are installed and correctly configured in the system, and that you have [CUDA-capable hardware](https://developer.nvidia.com/cuda-gpus) installed, you may follow these steps.

### Prerequisites

* [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 11.0 or above is correctly installed.
* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) version 7 or above is correctly installed.

### Installation

1. Clone this repository locally.

```bash
git clone git@github.com:sperezacuna/oracle-challenge-f3.git
```
2. Create a new [conda environment](https://docs.conda.io/projects/conda/en/latest/commands/create.html), with all dependecies installed.

```bash
conda create --name <env> --file requirements.txt
```

3. Activate it.

```bash
conda activate <env>
```

## Execution

Train a new model for continuous inference based on the [provided training dataset](https://challenges-asset-files.s3.us-east-2.amazonaws.com/0-challenges_data/2023_04/Oracle_3rd_challenge/training_set.csv) to infer the [provided testing dataset](https://challenges-asset-files.s3.us-east-2.amazonaws.com/0-challenges_data/2023_04/Oracle_3rd_challenge/testing.csv) using `main.py` script. You may specify the following parameters:
    
`-m MODELTYPE`, to establish the base binary-classification model type, either:

  - `v1`, basic (but deep) model as per the topology of a [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron). Concrete specification of the model layers and parametrization can be found at _model/v1.py_
  
`--help`, to show the help message for the script.

The model script will generate the results.json file containing the inferred labels for the testing dataset, which will be saved at _results/`MODELTYPE`_, along with
  - The ground truth for the testing dataset, according to insight on how the labels are calculated, which have prroved to be truthy, saved at _data/processed/full_set.csv_.
  - The full dataset, including testing and training data, along with its labels, saved at _results/ground-truth.json_.

(along with a graph of training statistics) will be saved at _models/`MODELTYPE`_

Example:
```bash
python main.py -m v1
```

> As the model creations is directly dependent on the testing dataset used for inference (although only to data prior to each inference sample, in any given moment), we cannot this time provide a model pretrained by us.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contributing

This project is being developed during the course of a competition, so PRs from people outside the competition will **not** be **allowed**. Feel free to fork this repo and follow up the development as you see fit.

Don't forget to give the project a star!

<p align="right">(<a href="#top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

Iago Barreiro Río - i.barreiro.rio@gmail.com

Santiago Pérez Acuña - santiago@perezacuna.com

Victor Figueroa Maceira - victorfigma@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>
