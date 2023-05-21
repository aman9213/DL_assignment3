## seq2seq_without_Attention file
In this file I have done hyperparameter tuning using WANDB. /
To run this file first upload hindi dataset folder (rename your folder hin) on your google drive in My drive section then mount your drive in google colab using below code --> \
```
from google.colab import files 
from google.colab import drive 
drive.mount('/content/drive')
```

After that you each can run the cell one by one .

### Important function or class
### Methods:
### methode name: `tokenize()` 
     
    discription: generate tokenized data for every word
    Arguments:(source,target,path) 
    source: column heading of input column in csv file
    target: column heading of target column in csv file
    path: path of your file
     
    Return: it will return tokrnized data 
  ```
    train_data_numerical=tokenize('shastragaar','शस्त्रागार',train_path)
    valid_data_numerical=tokenize('jaisawal','जयसवाल',valid_path,)
    test_data_numerical=tokenize('thermax','थरमैक्स',test_path)
  ```
 ### `class AksharantarDataset(Dataset):` 
 
 ```
 class AksharantarDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
  
    return len(self.data)

  def __getitem__(self, index):
      
    source, target = self.data[index]
    source_tensor = torch.tensor(source, dtype=torch.long)
    target_tensor = torch.tensor(target, dtype=torch.long)
      
    return source_tensor, target_tensor

 ```
 discription: above class give iteratable tokenized  data
 
 ### `class Encoder(nn.Module):`
   ### Methods:
   ### methode name:` def __init__(self,input_size,embedding_size,hidden_size,num_encoder_layer,dropout,input_vocab_size,bidirection,cell_type):`
        discription: it will build the encoder model.
        Arguments: input_size=maximum length of input words(which is 28 in hindi dataset including "sos" and "EOS" token.
                    embedding_size= size of embedding layers (each characterin a word get represented in this dimention)
                    hidden_size= size of hidden state of encoder layers
                    num_encoder_layer=number of encoder layers
                    dropout=dropout probability in fully connected part
                    input_vocab_size= vocabulary size of input words (which 29 in case of hindi dataset including "SOS","EOS","PAD")
                    bidirection=[True,False]
                    cell_type= type of sequential model(RNN,GRU,LSTM)
   ###  methode name:`def forward(self,inputs):`
         discription: it will do forward  prop in encoder
         Arguments: inputs= input to encoder(embedded input)
          Returns: hidden,cell
 
### `class Decoder(nn.Module):`
  ### Methods:
  ### methode name: `def __init__(self,input_size,embedding_size,hidden_size,output_vocab_size,num_decoder_layer,dropout,bidirection,cell_type):`
      discription: it will build the encoder model
      Arguments: input_size= maximum length of targrt words(which is 20 including "SOS" and "EOS".
                 embedding_size= size of embedding layer in decoder
                 hidden_size = size of hidden states in decoder layers
                 output_vocab_size= vocabulary size of target word (which is 58 including "SOS","EOS" and "PAD".
                 num_decoder_layer= number of decoder layers
                 bidirection=[True,False]
                 cell_type=type of sequential model (RNN,GRU,LSTM)
  ### methode name: `def forward(self,x,hidden,cell):`
       discription: it will do forward prop in decoder
       Arguments: x = input to decoder at every time step
                      hidden= privious hidden state output
                      cell = previous cell output
        Returns: List of prediction ,hidden ,cell
### `class seq2seq(nn.Module):` 
     dicription :it will connect encoder and decoder
### Methods: 
   ### methode name: `__init__(self,encoder,decoder):`
         discription :take the encoder and decoder
   ### methode name:`def forward(self,source,target,teacher_forceing=0.5):`
       discription: it will do forward prop through encoder and decoder while training
       Arguments: source= inputs word
                  target= target word 
                  teacher_forceing= for proper training of the model
       Returns: list of outputs of all time steps
   ### methode name:`def prediction(self,sources):`
       discription: his will forward prop while testin or validating
       Arguments: source= inputs words
       Returns:predicted outputs
     
     
### Methode:
  ###  methode name: `def train():`
        description: it will swepping of hyperparameters (we can find best hyper parameters)
        Arguments:None
        Return :None
   
  ### methode name: `def train_best_model():`
  
        description : it will train the model with best hyperparameters
        Argunments=None
        Returns=None
        
 **last two or three cell is for predicting the outputs saving in csv file.**\
***you can use this ReadMe file for other ipynb files also***
    
                  
***For runing the model please use(_____final.ipynb)file only for Attention and Without Attention***
       
         
                
        
                    
 
 
 
           
 
