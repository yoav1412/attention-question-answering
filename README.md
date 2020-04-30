# Natural Language Comprehension With Transformers
## Extractive question answering with the [SQUAD dataset](https://rajpurkar.github.io/SQuAD-explorer/)

* This project is a Keras (tensorflow backend) implementation of a deep neural network that combines self-attention architecture with convolutional layers, utilizing pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings.
* The model architecture generally follows the one proposed [here](https://arxiv.org/pdf/1804.09541.pdf) by Yu et. al. (QANet, 2018). The most significant change is that in this implementation we skip all character-level representations. 
* Some of the layer implementations are inspired by [this tutorial](https://www.tensorflow.org/tutorials/text/transformer#top_of_page) and [this repo](https://github.com/nptdat/qanet).

### The Dataset
We solve the problem of supervised question-answering, where each exmaple in the data is composed of a context paragraph, and a question. By definition, the answer to the question is found within the context, and the model's output is therefore a set of two pointers - to the start- and end-positions of the answer in the context paragraph.
For further reading about the problem formulation, the dataset and the performance metrics refer to the [SQUAD paper](https://arxiv.org/abs/1606.05250).

### Model Architecture
The general architecture, proposed in the QANet paper, is as follows:
![qanet-arch](https://github.com/yoav1412/attention-question-answering/blob/master/images/qanet-arch.jpg)

As stated above, in the original paper the context and question are represented as a concatenation of their word- and character-level embeddings. Following the results in section 4 of [Seo et. al. (2018)](https://arxiv.org/pdf/1611.01603.pdf) we drop the character level-embeddings as they provide little value.

### Training
Two versions of the model were trained, both with 1 embedding-encoder block and 7 model-encoder blocks (as proposed in the QANet paper), but with different hidden layer sizes of 96 and 128 (original paper uses just 128).
The training takes ~30 minutes per epoch on a `p3.2xlarge` gpu instance on aws.

### Results
* The 96-d model was trained for 16 epochs and achieved **73.4 / 60.8 f1 / em** scores (see the SQUAD paper for more information about the evaluation metrics).
* The 128-d model was trained for 13 epochs and achieved **74.1 / 61.3 f1 / em** scores.
The two best trained models can be found [here](https://github.com/yoav1412/attention-question-answering/tree/master/trained_models).

![validation metrics](https://github.com/yoav1412/attention-question-answering/blob/master/images/validation_metrics_plot.png)

### Pipeline
To run this code:
1. Download the SQUAD data form [here](https://rajpurkar.github.io/SQuAD-explorer/) and place the two files `train-v1.1.json` and `dev-v1.1.json` in `data/squad`.
2. Download the pre-trained 300-d GloVe word embeddings from [here](https://nlp.stanford.edu/projects/glove/) and place the file `glove.840B.300d.txt` in `data/glove` (note that any other pre-trained embedding can be used).
3. `pip install -r requirements.txt`
4 Run `python -m parse_squad_data.py` to build a processed version of the dataset (takes ~5 minutes).
5. Configure the desired training parameters in `conf.py` (such as number of epochs, batch size, etc.)
6. Run `python -m main.py`

### Interactive question answering with a trained model
The file `inference.py` contains a script that enables providing a trained model with a new context and question, and recieve an asnwer.
To use it:
1. Extract the trained model weights form the zip file in [trained_models](https://github.com/yoav1412/attention-question-answering/tree/master/trained_models) and place it in the same directory.
2. Run `python inference.py --weights={PATH_TO_WEIGHTS_FILE}`

An example interactive session using the 128-d model:
![example](https://github.com/yoav1412/attention-question-answering/blob/master/images/interactive_session_example.jpg)
