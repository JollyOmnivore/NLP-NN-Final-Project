import torch
from torch import nn
import numpy as np
import pandas as pd
import gensim.downloader
from gensim.models import Word2Vec, FastText
import pickle
from gensim.models import KeyedVectors
import gensim.downloader
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
from torch.utils.data import DataLoader
from collections import Counter
from nltk import bigrams
from nltk.util import ngrams
import tkinter as tk

'''
    HEY OVER HERE if you have an Nvidia graphics card you need to go to this page and get your pip install command
    Link: https://pytorch.org/get-started/locally/
'''
torch.set_default_device('cuda') # Comment this out if you are using a device without and Nvidia gpu



global WordEmbeddingType #This was used in order to keep track of the curent word embedding model 





"""
Create Neural Network model (architecture and forward pass)
"""
class RNNModel(nn.Module):
    
    
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        
        # Defining some parameters
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        """
        Check for GPU
        """
        
        is_cuda = torch.cuda.is_available()
        print(is_cuda)
        # set device to use GPU if available
        if is_cuda:
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, CPU used")

        '''
        RNN Setup from W9 HW with a few modifications 
        '''
        #Defining the layers
        # Word embeddings layer
        self.word_embeddings = nn.Embedding(input_size, embedding_dim)
        # RNN or LSTM Layer hm why not both?? carter later both is eh ;-;
        #self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True) #previously commented out
        #It works with both but not seeing any improvment will revisit later 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers) #wooooooo yeah LSTM LAYERS
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # For classification you would need an additional layer with the sigmoid activation fnction for example
        # self.sig = nn.Sigmoid() dont remove comment since this is not a classification problem 
        
        # "tie weights" : make the embedding layer share weights with the output layer --> reduces parameterization
        if embedding_dim == hidden_dim:
            self.word_embeddings.weight = self.fc.weight
    
    def forward(self, x, hidden):
        # Passing in the input and hidden state into the model and obtaining outputs
        embeds = self.word_embeddings(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        #fully connected layer
        out = self.fc(lstm_out)
        
        return out, hidden
    
    def init_hidden(self, size):
        # generate layer of zeros to be used by LSTM layer:
        # we need two because we have a hidden state and a cell state
        hidden = torch.zeros(self.n_layers, size, self.hidden_dim).to(self.device)
        cell = torch.zeros(self.n_layers, size, self.hidden_dim).to(self.device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        # hidden states due to different sequences are independent -> RESET
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell



class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, sequence_length, text_columns = '', sep=','):
        """
        Initialize dataset object
        """
        
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.words = self.load_dataset(text_columns, sep)
        self.vocab = self.get_vocab()
        
        # dictionaries to map indeces to words and vice versa
        self.index_to_word = {index: word for index, word in enumerate(self.vocab)}
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}

        # list of indeces
        self.words_indices = [self.word_to_index[w] for w in self.words]

        
    def load_dataset(self, text_column, sep=","):
        
        """
        Load in data set and tokenize
        """
        train_df = pd.read_csv(self.dataset_path, sep=sep).head(1000) #NOOOO LIMITSSS
        print(train_df.shape)
        
        train_df[text_column] = train_df[text_column].apply(clean_text)   #uses my regex function on data
        text = train_df[text_column].str.cat(sep=' ')
   
        return text.split(' ')

    def get_vocab(self):
        """
        Get vocabulary = unique words in data set
        """
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        """
        lets the model know when to stop
        -> need to subtract sequence length otherwise we'd have inconsistent tensor sizes!
        """
        return len(self.words_indices) - self.sequence_length

    def __getitem__(self, index):
        """
        create correct sized tensors
        -> i.e. size of the sequence being sent through
        Why two tensors? HIDDEN AND CELL STATE
        """
        return (
            torch.tensor(self.words_indices[index:index+self.sequence_length]),
            torch.tensor(self.words_indices[index+1:index+self.sequence_length+1]),
        )
    
"""
    Function created in order to clean up words in the dataframe before they are used for training
    
    input string: I, am" happy)
    output string: I am happy
"""
    
def clean_text(text):
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    text = re.sub(r"(?<!\()\)", "", text)
    text = re.sub(r"(?<!\[)\]", "", text)
    text = re.sub(r"\.(?=\s|$)", "", text)
    text = re.sub(r"[:,](?=\s|$)", "", text)
    text = re.sub(r"['’]", "", text)    
    text = text.lower() #This fixes vocab having things like The and the. Unfortunatly names and landmarks will be lowercased 
    return text


def train(dataset, model, batch_size, n_epochs, criterion, optimizer):
    """
    Training the model
    """
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(n_epochs):
        # Initializing hidden state for first input using method defined below
        LossAverage = []
        hidden = model.init_hidden(5)
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred, hidden = model.forward(x, hidden)
            #print(y_pred)
            loss = criterion(y_pred.transpose(1,2), y)

            hidden = model.detach_hidden(hidden)

            loss.backward()
            optimizer.step()
            
            LossAverage.append(float(loss.item()))
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
        epochAverage = np.mean(LossAverage)
        print("Average loss for epoch "+ str(epoch) +":",str(epochAverage)) #A lil diagnostic tool fro tuning
        
    torch.save(model.state_dict(),'RNNModel')
    
def predict(dataset, model, text, next_words=1):#Next word one mean that it will only guess one word 
    """
    Making predictions for new data after training
    """
    words = text.split(' ')
    hidden = model.init_hidden(len(words))

    for i in range(0, next_words):
        indices = []
        for w in words[i:]:
            indices.append(dataset.word_to_index[w])
        x = torch.tensor([indices])
        y_pred, hidden = model(x, hidden)

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu()# Added.cpu to pull data from V-Ram
        
        """
        Either choose words based on probability or get the word with highest probability
        You may quickly see why a random choice may be better
        """
        
        word_index = np.random.choice(len(last_word_logits), p=np.array(p)) 
        #word_index = np.argmax(np.array(np.array(p))) # I tried messing with this since im only prediting one word at a time but it still gets overpowered by The
        
        """
            Added something called TopK from Pytorch library to see if I can get better predictions
            unfortunly I get results like "the economy of the netherlands the  a and  the"
            (Note to revisit Carter)
        """
        
        """
        top_k = 10
        top_probs, top_indices = torch.topk(p, k=top_k)
        top_probs = top_probs.numpy()
        top_indices = top_indices.numpy()
        top_probs = top_probs / np.sum(top_probs)  # Normalize
        word_index = np.random.choice(top_indices, p=top_probs)
        """
        
        
        words.append(dataset.index_to_word[word_index])

    return words

"""
Instantiating the model
"""

# read in data, CHANGE THIS TO YOUR FILEPATH
path_to_data = "./WikiQACorpus/WikiQA.tsv"
data = Dataset(path_to_data, 5, sep="\t", text_columns='Sentence')
#removed after debugging
#print("printing data val")
#print(data)
#print("printing data.vocab val")
#print(data.vocab)
vocab_size = len(data.vocab)

# Define hyperparameters, TUNE SOME OF THESE, you can do this manually if you want
#WILLOW DO SHT HERE!!!!!
n_epochs = 6 # how many epochs: one epoch uses all the data ONCE, an epoch is split into BATCHES
lr=0.01 # learning rate
GlobalEmbedddim = 300
embedding_dim = GlobalEmbedddim # user chosen value for now, will change to actual embedding size
hidden_dim = 10
n_layers = 5
batch_size = 500 # data points per batch

# initialize the model
model = RNNModel(input_size=vocab_size, output_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers)

criterion = nn.CrossEntropyLoss() #Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optimizer = how to update network weights
"""
Set word embeddings:
This will have to be changed for the homework!
"""




"""
------------------------------ HEY OVER HERE !!!!!!!!!!!!!!!!
"""


data = Dataset(path_to_data, 5, sep="\t", text_columns='Sentence')
#word2vec_model = gensim.downloader.load('word2vec-google-news-300') #Dont forget to uncomment me 
print("printing data.words")
#print(data.words)


'''
w2vec code from slides ---------------------------------------------------
'''
WordEmbeddingType = "W2V"
df = pd.read_csv("./WikiQACorpus/WikiQA.tsv", sep="\t")

documents = df["Sentence"].apply(clean_text)#uses my regex function on data

tokened_Docs = [doc.split(" ") for doc in documents]

#Word2Vec model parameters
vector_size = GlobalEmbedddim  
window = 5         # Context window size
min_count = 1      # Minimum word frequency
sg = 1             # Training algorithm: 1 for skip-gram, 0 for CBOW
workers=6          # Number of threads for training 

# Initialize Word2Vec model
# Look into workers
w2vmodel = Word2Vec(min_count=min_count, vector_size=vector_size, window=window, sg=sg,workers=workers)

# Build vocabulary
w2vmodel.build_vocab(tokened_Docs)

print("Training W2Vec model")

# Train model
w2vmodel.train(tokened_Docs, total_examples=w2vmodel.corpus_count, epochs=w2vmodel.epochs)
#model.save("word2vec.model")
#w2vmodel = Word2Vec.load("word2vec.model")
print("Finished training W2Vec model")
doc_embeddings_matrix = np.array([w2vmodel.wv[word] for word in data.vocab if word in w2vmodel.wv])



'''
End of w2vec code from slides ---------------------------------------------------
'''

'''
---------------------------FastText Word embeddings 
# This was used in order to test the differences between word embedding models
#from my research this was worse and run much slower than W2V so I ultimaly went back to W2V

data = Dataset(path_to_data, 5, sep="\t", text_columns='Sentence')

# Loading a pre-trained FastText model instead of Word2Vec
print("Starting fast text download")
fasttext_model = gensim.downloader.load('fasttext-wiki-news-subwords-300')  # Example FastText model
print("Finished fast text download")
print("printing data.words")
#print(data.words)


WordEmbeddingType = "FastText"
df = pd.read_csv("./WikiQACorpus/WikiQA.tsv", sep="\t")

documents = df["Sentence"].apply(clean_text)  # uses my regex function on data

tokened_docs = [doc.split(" ") for doc in documents]

# FastText model parameters (similar to Word2Vec)
vector_size = GlobalEmbedddim
window = 7         # Context window size
min_count = 1      # Minimum word frequency
sg = 1             # Training algorithm: 1 for skip-gram, 0 for CBOW
workers = 6        # Number of threads for training 

# Initialize FastText model
ft_model = FastText(vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=workers)

# Build vocabulary
ft_model.build_vocab(tokened_docs)

print("Training FastText model")

# Train model
ft_model.train(tokened_docs, total_examples=ft_model.corpus_count, epochs=ft_model.epochs)

print("Finished training FastText model")
doc_embeddings_matrix = np.array([ft_model.wv[word] for word in data.vocab if word in ft_model.wv])


------------------------End of Fast test
'''



""" N-greams test RuntimeError: The size of tensor a (35261) must match the size of tensor b (378330) at non-singleton dimension 0

"""

'''
Bert implementation for word embedds-------------------------- from geeks for geeks 
'''

"""Did not work to to bert model size being to large for GPU Memory even with the small version 
print("Starting bert")



# Load data
df = pd.read_csv("./WikiQACorpus/WikiQA.tsv", sep="\t")
documents = df["Sentence"]  # Assuming the text cleaning is handled if needed

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small')
model = BertModel.from_pretrained('prajjwal1/bert-small')

# Tokenize documents
tokenized = tokenizer(documents.tolist(), padding=True, truncation=True, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**tokenized)
    embeddings = outputs.last_hidden_state  # You can also use outputs.pooler_output for pooled outputs

# Example of using embeddings: calculating mean of token embeddings for each document
doc_embeddings = embeddings.mean(dim=1).numpy()

# Optionally, you can convert these embeddings into a format similar to your original w2v output
doc_embeddings_matrix = np.vstack(doc_embeddings)




"""

'''
End of Bert--------------------------------------------- MODEL was far to large for GPU VRam
'''


'''
store model code from slides -----------------------------------------
#Unfortunte this went Unused to dude to the curent class structure

# Store just the words + their trained embeddings.
word_vectors = w2vmodel.wv
word_vectors.save("word2vec.wordvectors")

# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')


store moddle code from slides -----------------------------------------
'''


# LOAD IN ACTUAL WORD EMBEDDINGS HERE

# Then attempt the copy if sizes match


model.word_embeddings.weight.data.copy_(torch.from_numpy((doc_embeddings_matrix)))

# Make sure that the weights in the embedding layer are not updated
model.word_embeddings.weight.requires_grad=True #I made it so the weights are updated here 

"""
Training the model
"""
train(data, model, batch_size, n_epochs, criterion, optimizer)

"""
Output for testing the "best" model.
Try to make the follow-up words (i.e. the entire sentence) make sense!
"""

"""
Next word prediction function

This function is called in my GUI and is used to interate over the input words and confirm all words are inside of the vocab
Next it will attempt to predict one word using predict() function
It will convert the reponse back to a string and feed the reponse back into the predict 
"""

print(data.vocab)

def sentencePredictor(userInput, nextWords): #we adding fancy next word prediction
    x = 0
    userInput = userInput.lower()
    inputWords = re.findall(r'\b\w+\b',userInput)
    print(inputWords)
    for word in inputWords:
        if word not in data.vocab:
            print("Error you entered a word not included in our library")
            return "Error you entered a word not included in our library"
    while x < nextWords:
        userInput = (predict(data, model, text=userInput))
        userInput= " ".join(userInput)
        print(userInput)
        x += 1
    storeResponses(userInput, nextWords)
    return userInput


"""
Response storage

Gets called by sentencePredictor() and is passed the final output and how many words the model added 
"""

def storeResponses(response, numWords):
    with open('Responses.txt', 'a') as file:
        message = (
            f"({response}) Number of words predicted {numWords} "
            f"Word Embedding Model: {WordEmbeddingType} "
            f"Embedding Settings- Vector size {GlobalEmbedddim} Window Size {window} "
            f"Model HyperParameters- Epochs: {n_epochs} Hidden Dims {hidden_dim} "
            f"NN Layers {n_layers} Batch Size {batch_size} \n"
        )
        file.write(message)


""" ------------------------------------------------ used for testing in terminal 
#sentencePredictor("the economy of the Netherlands", 4)

while True:
    userInput = input("Text: ")
    sentencePredictor(userInput, 6)
"""

"""
    Function to manage user inputs and passing to model in order to predict 
"""
def process_input():
    # Grabs input from box and dropdown
    input_text = input_entry.get()
    selected_number = number_var.get()
    
    #runs my sentencePredictor funcion 
    response = sentencePredictor(input_text, int(selected_number))

    output_text.delete("1.0", tk.END)  # Clears output box
    output_text.insert("1.0", response)  # Inserts result




"""
    Tkinter Code to create style and Format GUI 
"""

# Creates GUI
root = tk.Tk()
root.title("Not GPT")

# Creates a frame
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Input entry box, width increased by 50%
input_entry = tk.Entry(frame, width=30)
input_entry.grid(row=0, column=0, padx=(0, 5), pady=(10, 10))

# Dropdown for selecting a number
number_var = tk.StringVar(root)
number_var.set(1) 
numbers = [str(i) for i in range(1, 11)]
number_menu = tk.OptionMenu(frame, number_var, *numbers)
number_menu.config(width=8)
number_menu.grid(row=0, column=1, padx=(0, 5), pady=(10, 10))
process_button = tk.Button(frame, text="Send", command=process_input)
process_button.grid(row=0, column=2, padx=(0, 5), pady=(10, 10))
output_text = tk.Text(frame, width=45, height=3)  # Increased width and added height
output_text.grid(row=1, column=0, columnspan=3, pady=(0, 10))  # Span across all columns

# opens GUI
root.mainloop() 
 
