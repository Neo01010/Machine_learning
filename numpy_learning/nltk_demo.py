import nltk

nltk.download()

sentence = """ Executing task: /home/ren/machine-learning/bin/python3.5 
/home/ren/Desktop/machine_learning/numpy_learning/numpy_demo.py """
tokens = nltk.word_tokenize(sentence)

print(tokens)

# tagged = nltk.pos_tag(tokens)
# print(tagged[0:6])
