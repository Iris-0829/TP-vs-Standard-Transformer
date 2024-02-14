
import numpy as np

sentences = []
for _ in range(10000):
    length = np.random.geometric(0.33)

    sentence = ["0" for _ in range(length)] + ["1" for _ in range(length)]
    sentence = " ".join(sentence)

    sentences.append(sentence)

train_set = sentences[:8000]
valid_set = sentences[8000:9000]
test_set = sentences[9000:]

fo_train = open("standard_dataset/train.txt", "w")
fo_valid = open("standard_dataset/valid.txt", "w")
fo_test = open("standard_dataset/test.txt", "w")

for sentence in train_set:
    fo_train.write(sentence + "\n")

for sentence in valid_set:
    fo_valid.write(sentence + "\n")

for sentence in test_set:
    fo_test.write(sentence + "\n")







