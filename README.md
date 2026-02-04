# LLMs_Tutorials

Sentence ---> break sentence --> token -->assign Number ---> LLM wont understand Numbers
embedding_vector --->DIMENTION (we will decide based on which embedding word model we are using)



The below python code  
from evey sentence its created a token from
```
words = list(set(text.split()))
print(words)
```


Dictonary is created 
for every word we assigned a number  
```
word2idx = {w: i for i, w in enumerate(words)}
print(word2idx)
```

This we have done through the python code .....but generally in LLMs 
Both this 2 works are done by TOKENIZATION 

# In LLM there are Tokenizers 
eg: GPT Models --> Tokenizer used here is ==> BPE [Byte Pair Encoding] --> 
Firt converts the words into Token
Second assigns a number to every Token


# Tokenization #
Breaks Sentences into words i.e Token 
For every Token assigning a Unique ID     

Eg:- my name is sagar and i love coding 
["my", "name", "is", "sagar", "and", "i", "love", "coding"]

Token Id: [101,223,31,987,45,12,34,556]

my ----> 101 ----> [................]512
 
first token: my   --->101 --->convert this into a Embedding vector --->[..........]512
second token: name --->223 --->convert this into a Embedding vector --->[..........]512 
third token: is --->31 --->convert this into a Embedding vector --->[..........]512 

Note:-  The size of embedding vector for all the token will be the same i.e 512  -->  [d_model]

**sentence:**
[
e_my,
e_name,
e_is,
e_sagar,
e_and,
e_i,
e_love,
e_coding
]


# what is the length of an embedding vector?#

How many values will have in an embedding vector ?
d_model = 256 , 512 

is --->31 --->convert this into a Embedding vector --->[..........]256 
is --->31 --->convert this into a Embedding vector --->[..........]512 


| Model Name |    n_params    |   n_layers    |   d_model |     
| GPT-3 6.7B |    6.7B        |   32          |   4096    |
| GP-3 13B   |    13.0B       |   40          |   5140    |
| GPT-3 Small|    125M        |   12          |   768     |
| GPT-3 Medim|    350M        |   24          |   1024    |      
| GPT-3 Large|    760M        |   24          |   1536    |

Every Token gets the exact lenght of embedding vector
every token will have 512 values in it.....if **d_model=512**

"my" = [0.2,-0.7,0.1.....]512
"name" = [0.5,0.8,-0.3....]512

training will shape them into meanful vectors.

Embeddings are trainable parameters just like weights in a neural network 
    we update the weight 
    we train the model 
    we update the weight

# Vocabulary size # 

Vocabulary size = Total number of unique tokens your tokenizer can produce.

    Tokens are not words 

    Tokens : words , subwords, characters, punctuation, special token


suppose we have provided our dataset, raw text  --> it created ==> 50,000 [Tokens] -->This is Vocabulary  [unique words]

For every Token => 1D Vector of length 512    [Bcoz we have defined d_model=512]

Embedding Layer 
Vocabulary = 50,000
d_model = 512

matriz ---> 50000 x 512 ---> table [rows & columns]
50000 rows 
each row wil have 512 values 

# How does vocabulary size effect the model? # 

Bigger Vocabulary    --> 50,000 token

Small Vocabulary     --> 10,000 token   

|   GPT-1   | 40,000    | BPE Tokens|
|   GPT-2   | 50,257    | Tokens    |
|   BERT    | 30,522    | Tokens    |
|   LLaMa   | 32,000    | Tokens    |


# How does the embedding matrix actually get learned during training? #
    How training happens ? 
    How embedding layers learn the embedding vectors ?

For each Token ---> We have a Embedding Vector 
This embedding vectors have random values 

shape of matrix: (vocab_size x embedding_dim)



# How these parameters of embedding matrix are updated #

forward 
backward propagation  --> 
In Backward pass --> Bases on the Loss Model will see which weight caused this loss and by how much 
This is done by chain rule of calculus.


## Positional Encoding in Transformers ##

Input - my name is sagar and i love coding 

["my", "name", "is", "sagar", "and", "i", "love", "coding"]

Token Id: [101,223,31,987,45,12,34,556]

"my"    = [0.2,-0.7,0.1,.....]      lenght of this embedding vector depends on -->d_model
"name"  = [0.5,0.8,-0.67,......]
"is"    = [0.05,0.08,-0.04,....]
"sagar" = [0.35,0.38,-0.33,...]
"and"   = []
"i"     = []
"love"  = []
"coding"= [0.25,0.28,-0.32,.....]

we created this Endedding vectors --->Because Transformer can understand the meaning of each word.
Transformer will understand the meaning of each Token each Word  using this Embedding Vectors

Transformer excepts inputs in form of Embedding Vectors 
so we converted our Token into Embedding Vectors 

# PositionalEncoding#  --> To understand the position of each token.

Why we are using this Positional Encoding !!!

all the token embedding will go together at once to the Transformer 
Tranformer will not know which token is at the First Postion  or which Token is at the Second Position.
What is the exact sequence of the token in the data set....
It will not understand the postion of Each token in sequence.

So to tell the Transformer --> which Token is at which Postion ---> we add Posititonal Encoding.

