import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

# we're still going to use a Tokenizer here, but we don't need to fit it
tokenizer = Tokenizer(num_words=3000)
# for human-friendly printing
labels = ['negative', 'positive']

# read in our saved dictionary
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices

# read in your saved model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('model.h5')

# model.save('FullModel.hdf5')
# okay here's the interactive part
evaluatedData = open('evaluatedData.txt', 'w')
evaluatedDictionary = {"data": []}
count = 0
toEval = open('.\dataToEval.txt')
file_content = toEval.readlines()
for line in file_content:
    line = line.rstrip()
    print(line)

    # format your input for the neural net
    testArr = convert_text_to_index_array(line)
    print(testArr)
    inputToken = tokenizer.sequences_to_matrix([testArr], mode='binary')
    print(inputToken)
    # predict which bucket your input belongs in
    pred = model.predict(inputToken)
    # and print it for the humons
    print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
    evalSentence = ""
    evaluatedData.write(labels[np.argmax(pred)] + " sentiment with " +  str(pred[0][np.argmax(pred)] * 100) + " confidence for: " + line)
    evaluatedDictionary["data"].append({"text": line, "sentiment": labels[np.argmax(pred)], "confidence": pred[0][np.argmax(pred)] * 100, "message_num": count})
    count +=1

# print(evaluatedDictionary)
with open('evaluatedDict.json', 'w') as outfile:
    json.dump(evaluatedDictionary, outfile)
toEval.close()
evaluatedData.close()
