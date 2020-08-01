import numpy as np 
import matplotlib.pyplot as plt

no_nonlinearity = np.load("ftf_accs.npy")
fig = plt.figure()
plt.title("No Backwards Nonlinearity")
plt.xlabel("Batch")
plt.ylabel("Training Accuracy")
plt.plot(no_nonlinearity)
fig.savefig("no_nonlinearity.jpg")
plt.show()

learnt_back = np.load("tft_accs.npy")
fig = plt.figure()
print(learnt_back)
plt.title("Learning_backwards_weights")
plt.xlabel("Batch")
plt.ylabel("Training Accuracy")
plt.plot(learnt_back)
fig.savefig("learning_backwards_weights.jpg")
plt.show()

FA_accs = np.load("ttf_accs.npy")
fig = plt.figure()
print(FA_accs)
plt.title("Feedback Alignmnt")
plt.xlabel("Batch")
plt.ylabel("Training Accuracy")
plt.plot(FA_accs)
fig.savefig("feedback_alignment.jpg")
plt.show()

FA_accs_no_nonlinearity = np.load("ttf_accs.npy")
fig = plt.figure()
print(FA_accs)
plt.title("Feedback Alignmnt Without Nonlinearity")
plt.xlabel("Batch")
plt.ylabel("Training Accuracy")
plt.plot(FA_accs_no_nonlinearity)
fig.savefig("feedback_alignment_no_nonlinearity.jpg")
plt.show()