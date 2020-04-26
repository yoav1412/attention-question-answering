# Natural Language Comprehension With Transformers
## Extractive question answering with the [SQUAD dataset](https://rajpurkar.github.io/SQuAD-explorer/)

* This project is a Keras (tensorflow) implementation of a deep neural network that combines self-attention (transformer) architecture with convolutional layers.
* The model architecture generally follows the one proposed [here](https://arxiv.org/pdf/1804.09541.pdf) by Yu et. al. (QANet, 2018). The most significant change is that in this implementation we skip all character-level representation.

### The Dataset
We solve the problem of supervised question-answering, where each exmaple in the data is composed of a context paragraph, and a question. The answer to the question is found within the context, and the model's output is a set of two pointers - to the start and end positions of the answer in the context paragraph.
For further reading about the problem formulation refer to the [SQUAD paper](https://arxiv.org/abs/1606.05250).

### Model Architecture
The general architecture, proposed in the QANet paper, is a follows:
![qanet-arch](https://github.com/yoav1412/attention-question-answering/blob/master/images/qanet-arch.jpg)

### Training
I trained two versions of the model, both with 1 embedding-encoder block and 7 model-encoder blocks (as proposed in the QANet paper), but with different hidden layer sizes of 96 and 128 (original mpaper uses just 128).
The training takes ~30 minutes per epoch on a `p3.2xlarge` machine on aws. 


### Results
* The 96-d model was trained for 16 epochs and achieved **73.4 / 60.8 f1 / em** scores (see the SQUAD paper for more information about the evaluation metrics).
* The 128-d model was trained for 13 epochs and achieved **74.1 / 61.3 f1 / em** scores.
The two best trained models can be found [here]().

### Pipeline
To run this code:
1. Download the data form [here](https://rajpurkar.github.io/SQuAD-explorer/) and place the two files `train-v1.1.json` and `dev-v1.1.json` in `data/squad`.
2. Download the pre-trained 300-d GloVe word embeddings from [here](https://nlp.stanford.edu/projects/glove/) and place the file `glove.840B.300d.txt` in `data/glove`
3. `pip install -r requirements.txt`
4 Run `python -m parse_squad_data.py` to build a processed version of the dataset (~5 minutes).
5. Configure the desired training parameters in `conf.py`
6. Run `python -m main.py`
