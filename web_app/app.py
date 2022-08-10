from flask import Flask, request, jsonify, render_template
import torch
from torch import nn
import nltk
from nltk.corpus import stopwords
from torchtext.data.utils import get_tokenizer

nltk.download('stopwords')

# use a standard english tokenizer
tokenizer = get_tokenizer('basic_english')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# SIMPLE BOW CLASSIFIERF

class BoWClassifier(nn.Module):
# I referred to this tutorial for help: https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
    def forward(self, bow_vec):
        # pass through linear layer
        return self.linear(bow_vec)

# SIMPLE LSTM MODEL

class LSTM(nn.Module):
# I referred to this tutorial for help: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
    def __init__(self, vocab_size, emb_dim, hid_dim, output_dim, dropout = 0.5):
        super().__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # RNN layer
        self.rnn = nn.LSTM(emb_dim, hid_dim) # default batch_first is False
        # dropout
        self.dropout = nn.Dropout(dropout)
        # linear layer 
        self.linear1 = nn.Linear(hid_dim, output_dim)
    def forward(self, text):
        # input text is dimension [seq_len, batch_size]
        # apply embeddings to the words
        embedded = self.embedding(text)
        #print('embedded shape')
        #print(embedded.shape)
        # embedded is dimension [seq_len, batch_size, emb_dim] because batch_first = False
        # run through RNN
        output, hidden = self.rnn(embedded)
        # hidden[0] is dimension [1, batch_size, hid_dim]
        # get the predictions
        last_output = output[-1]
        last_output = self.dropout(last_output)
        scores = self.linear1(last_output) 
        # scores should have the dimension [batch_size, output_dim]
        return scores

# SIMPLE GRU MODEL

class GRU(nn.Module):
# I referred to this tutorial for help: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
    def __init__(self, vocab_size, emb_dim, hid_dim, output_dim, dropout = 0.5):
        super().__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # RNN layer
        self.rnn = nn.GRU(emb_dim, hid_dim) # default batch_first is False
        # dropout
        self.dropout = nn.Dropout(dropout)
        # linear layer 
        self.linear1 = nn.Linear(hid_dim, output_dim)
    def forward(self, text):
        # input text is dimension [seq_len, batch_size]
        # apply embeddings to the words
        seq_len = text.shape[0]
        batch_size = text.shape[1]
        embedded = self.embedding(text)
        #print('embedded shape')
        #print(embedded.shape)
        # embedded is dimension [seq_len, batch_size, emb_dim] because batch_first = False
        # run through RNN
        output, hidden = self.rnn(embedded)
        # hidden[0] is dimension [1, batch_size, hid_dim]
        # get the predictions
        last_output = output[-1]
        last_output = self.dropout(last_output)
        scores = self.linear1(last_output) 
        # scores should have the dimension [batch_size, output_dim]
        return scores

def predict_label_BOW(text, model, vocab):
    labels = ['Negative', 'Positive']

    # put model in eval mode
    model.eval()

    # process input
    stop_words = set(stopwords.words('english'))
    tokens = tokenizer(text)
    tokens_without_stopwords = [token for token in tokens if token not in stop_words]
    bow = torch.zeros(len(vocab))
    c = 0 
    for token in tokens_without_stopwords:
        bow[vocab[token]] += 1
        c += 1
    bow = bow/c
    # add batch of 1 (dimension)
    bow = bow.unsqueeze(0)
    #print('bow shape')
    #print(bow.shape)

    with torch.no_grad():
        scores = model(bow)

    predicted_label = scores.argmax(1)

    return labels[predicted_label.item()], scores

def predict_label_RNN(text, model, vocab):
  labels = ['Negative', 'Positive']

  # put model in eval mode
  model.eval()

  # process input
  stop_words = set(stopwords.words('english'))
  tokens = tokenizer(text)
  tokens_without_stopwords = [token for token in tokens if token not in stop_words]
  idxs = [vocab[token] for token in tokens_without_stopwords]
  idxs = torch.tensor(idxs)
  # add batch of 1 (dimension)
  idxs = idxs.unsqueeze(1)

  with torch.no_grad():
    scores = model(idxs)

  print(scores)  
  predicted_label = scores.argmax(1)

  return labels[predicted_label.item()], scores

################# MODEL INITIALIZATION ##################

BOW_vocab = torch.load('final_models/BOW_vocab.pth')

INPUT_DIM = len(BOW_vocab)
OUTPUT_DIM = 2 # there are only 2 labels - 0 (neg) or 1 (pos)

BOW_model = BoWClassifier(OUTPUT_DIM, INPUT_DIM)

BOW_model.load_state_dict(torch.load('final_models/BOW-model.pt'))

LSTM_vocab = torch.load('final_models/LSTM_vocab.pth')

INPUT_DIM = len(LSTM_vocab)
EMBEDDING_DIM = 32
HIDDEN_DIM = 128
OUTPUT_DIM = 2 # there are only 2 labels - 0 (neg) or 1 (pos)
DROPOUT = 0.10

LSTM_model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

LSTM_model.load_state_dict(torch.load('final_models/LSTM-model.pt', map_location=torch.device('cpu')))

GRU_vocab = torch.load('final_models/GRU_vocab.pth')

INPUT_DIM = len(GRU_vocab)
EMBEDDING_DIM = 32
HIDDEN_DIM = 128
OUTPUT_DIM = 2 # there are only 2 labels - 0 (neg) or 1 (pos)
DROPOUT = 0.25

GRU_model = GRU(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

GRU_model.load_state_dict(torch.load('final_models/GRU-model.pt', map_location=torch.device('cpu')))

################# FLASK ROUTES ##########################

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        return app.send_static_file('index.html')
    else: 
        review_text = request.form['content']
        BOW_prediction, BOW_scores = predict_label_BOW(review_text, BOW_model, BOW_vocab)
        LSTM_prediction, LSTM_scores = predict_label_RNN(review_text, LSTM_model, LSTM_vocab)
        GRU_prediction, GRU_scores = predict_label_RNN(review_text, GRU_model, GRU_vocab)

        BOW_scores = BOW_scores[0].tolist()
        LSTM_scores = LSTM_scores[0].tolist()
        GRU_scores = GRU_scores[0].tolist()

        return render_template("prediction.html", text = review_text, \
            BOW_prediction = BOW_prediction, LSTM_prediction = LSTM_prediction, GRU_prediction = GRU_prediction, \
                BOW_scores = BOW_scores, LSTM_scores = LSTM_scores, GRU_scores = GRU_scores)
