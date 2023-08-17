from transformers import pipeline
import numpy as np

classifier = pipeline(task="text-classification")

import time

allm = []

start = time.time()
print(classifier("I like you. I love you."))
elapsed = time.time() - start
print("Elapsed {:.4f} seconds".format(elapsed))


for i in range(0, 100):
    start = time.time()
    print(classifier("I like you. I love you."))
    elapsed = time.time() - start
    print("Elapsed {:.4f} seconds".format(elapsed))
    allm.append(elapsed)

allm = np.array(allm)
m = allm.mean()
s = allm.std()
q = np.quantile(allm, 0.95)

print("{:.4f} {:.4f} {:.4f}".format(m, s, q))

import pdb
pdb.set_trace()
