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


# Stage2:-  we want only vectors i.e Numbers # 

