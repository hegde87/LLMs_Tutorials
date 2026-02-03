
# Tokenizer --> Divides Text into Token & then assigns ID to all the Token
# 1) SentencePiece Tokenizer : pip install sentencepiece
# 2)BPE tokenizer: pip install tokenizers     
# 3)load a pre-trained word-embedding model like -->word2vec / FastText: pip install gensim 
# 4)Hugging Face: pip install transformers 

----------------------------------------------------------


# Tokenizer --> Divides Text into Token & then assigns ID to all the Token
# 1) SentencePiece Tokenizer : pip install sentencepiece
# 2)BPE tokenizer: pip install tokenizers     
# 3)load a pre-trained word-embedding model like -->word2vec / FastText: pip install gensim 
# 4)Hugging Face: pip install transformers 

# First we will use:- 1) SentencePiece Tokenizer ---> pip install sentencepiece 

when ever we are creating a LLM from Scratch 
Choice of Tokenizer , this thing matters a lot 
we have 2 Options 

# Option 1:- we create our own tokenizer --> we train it with our dataset , teach how to creates tokens. 

# Option 2:- Use a Pre-Trained Tokenizer 

Example: GPT-4-O-Mini, lama 
the tokenizer which is already created and which is already trained --> during the training of GPT-2 OR GPT-4-O-Mini 

# Note: Any pre-trained tokenizer split text in a different way  into tokens.

eg: 
---
GP2-2:- ##unbelievable##  ---> ["un", "believ", "able"]  --> breaking it into 3 Tokens. 
---
BERT:-  #unbelievable# ---> ["un", "believable"] --> breaking it into 2 Tokens.
---

Lets go with OPTION 1 :- Training  our own tokenzier
question: why should we train the own tokenizer?
answer: Our dataset will be different from any LLM 

# For Example: a LLM can answer Legal queries properly / a LLM can answer DR question

Example: we want to create a LLM which can reply with Hindi & English Mix Responses.
DataSet contains:
- Indian city names 
- hindi/english mix sentences
- festivals
- local slangs 
- local food names

when we know that the dataset what we are providing to LLM is unique and pre-trained tokenizer are not trained on this dataset, they wont be able to create good tokens.
eg: I LOVE SAMOSA.    ---> ["sa","mos", "a"]  --> this makes no sense.  -->if we are going with pre-trained tokenizer.

Note: if in our Dataset we are using normal English then we can have Pre-Trained Tokenizer.

# Lets create our own Tokenizer !!! 

dataset
```
hello friends how are you
the tea is very hot
my name is Sagar Hegde
the roads of Pune are busy
it is raining in pune
the train is late again
i love eating vada pav and drinking tea
holi is my favorite festival
diwali vrings lights and sweets
india won the cricket match
```
```
step1:- pip install sentencepiece
```
```
step2:- import module in our code 
```
```
step3:- train it with our dataset --> so that it generates tokens 
```
```
step4:- then wil give that tokens to our LLM for training 
```