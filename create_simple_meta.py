
import numpy as np
import random

datasets = []
for _ in range(1000):
    word1 = random.choice(list(range(100)))
    word2 = random.choice(list(range(100)))

    sentences = []
    for index in range(200):
        length = np.random.geometric(0.33)

        sentence = ["0" for _ in range(length)] + ["1" for _ in range(length)]
        sentence = " ".join(sentence)

        sentences.append(sentence)

    train = " ".join(sentences[:100])
    test = " ".join(sentences[100:])
    datasets.append((train, test))

train_set = datasets[:800]
valid_set = datasets[800:900]
test_set = datasets[900:]

fo_train = open("meta_dataset/train.txt", "w")
fo_valid = open("meta_dataset/valid.txt", "w")
fo_test = open("meta_dataset/test.txt", "w")

for train, test in train_set:
    fo_train.write(train + "\t" + test + "\n")

for train, test in valid_set:
    fo_valid.write(train + "\t" + test + "\n")

for train, test in test_set:
    fo_test.write(train + "\t" + test + "\n")







