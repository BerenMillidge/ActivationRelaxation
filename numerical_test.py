#Numerical check, code to reproduce figures 1 and 2
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_batches= 10
num_train_batches=20
batch_size = 64
norm_factor = 1 # 255

transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

# get some random training images
trainset = list(iter(trainloader))
for i,(img, label) in enumerate(trainset):
  trainset[i] = (img.reshape(len(img),784) /norm_factor ,label)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

testset = list(iter(testloader))
for i,(img, label) in enumerate(testset):
  testset[i] = (img.reshape(len(img),784) /norm_factor ,label)



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def onehot(x):
    z = torch.zeros([len(x),10])
    for i in range(len(x)):
      z[i,x[i]] = 1
    return z.float().to(DEVICE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### functions ###
def set_tensor(xs):
  return xs.float().to(DEVICE)

def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return set_tensor(torch.ones_like((x)))

def relu(xs):
  return torch.clamp(xs,min=0)

def relu_deriv(xs):
  rel = relu(xs)
  rel[rel>0] = 1
  return rel 

def softmax(xs):
  return torch.nn.softmax(xs)

def sigmoid(xs):
  return F.sigmoid(xs)

def sigmoid_deriv(xs):
  return F.sigmoid(xs) * (torch.ones_like(xs) - F.sigmoid(xs))

def accuracy(out, L):
  B,l = out.shape
  total = 0
  for i in range(B):
    if torch.argmax(out[i,:]) == torch.argmax(L[i,:]):
      total +=1
  return total/ B


def register_hook(tensor, msg):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))

def get_printer(msg):
    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(f"{msg} shape: {tensor.shape}"
                  f" max: {tensor.max()} min: {tensor.min()}"
                  f" mean: {tensor.mean()}")
    return printer

def numerical_test(lr,n_iters=100):
    w1 = nn.Parameter(torch.empty((784,300)).normal_(mean=0, std=0.05))
    w2 = nn.Parameter(torch.empty((300,100)).normal_(mean=0, std=0.05))
    w3 = nn.Parameter(torch.empty((100,100)).normal_(mean=0, std=0.05))
    w4 = nn.Parameter(torch.empty((100,10)).normal_(mean=0, std=0.05))
    images, labels = trainset[0] 
    img = nn.Parameter(images)
    labels = onehot(labels)
    # check backprop

    x1 = tanh(img @ w1)
    x2 = tanh(x1 @ w2)
    x3 = tanh(x2 @ w3)
    x4 = linear(x3 @ w4)
    register_hook(x1, "x1")
    register_hook(x2, "x2")
    register_hook(x3,"x3")
    register_hook(x4,"x4")
    print(x4.shape)
    print(labels.shape)
    L = torch.sum((x4 - labels)**2)
    L.backward()
    bp_w1grad = w1.grad.clone()
    bp_w2grad = w2.grad.clone()
    bp_w3grad = w3.grad.clone()
    bp_w4grad = w4.grad.clone()
    bp_x1grad = x1.grad.clone()
    bp_x2grad = x2.grad.clone()
    bp_x3grad = x3.grad.clone()
    bp_x4grad = x4.grad.clone()
    bp_inpgrad = img.grad.clone()
    # beginning AR alg
    #lr = 0.1
    n_epochs = n_iters
    with torch.no_grad():
        e_L = 2 * ( x4 - labels)
        w4grad = x3.T @ (bp_x4grad * linear_deriv(x3 @ w4))
        print("w4 grad")
        print(w4grad[0:10,0])
        print(bp_w4grad[0:10,0])
        print(torch.sum(w4grad - bp_w4grad)) 
        old_x3_deriv = linear_deriv(x3 @ w4)
        old_x2_deriv = tanh_deriv(x2 @ w3)
        old_x1_deriv = tanh_deriv(x1 @ w2)
        old_x0_deriv = tanh_deriv(img @ w1)
        old_x3 = x3.clone()
        old_x2 = x2.clone()
        old_x1 = x1.clone()
        x3s = []
        x2s = []
        x1s = []
        e3s =  []
        e2s = []
        e1s = []
        for i in range(n_epochs):
            x3grad =x3 - ((e_L * old_x3_deriv) @ w4.T)
            x2grad = x2 - ((x3 * old_x2_deriv) @ w3.T)
            x1grad = x1 - ((x2 * old_x1_deriv) @ w2.T)
            x3 -= (lr * x3grad)
            x2 -= (lr * x2grad)
            x1 -=( lr * x1grad)
            x3s.append(torch.sum(x3.clone()).item())
            x2s.append(torch.sum(x2.clone()).item())
            x1s.append(torch.sum(x1.clone()).item())
            e3s.append(torch.sum((x3 - bp_x3grad)**2).item())
            e2s.append(torch.sum((x2 - bp_x2grad)**2).item())
            e1s.append(torch.sum((x1 - bp_x1grad)**2).item())
        print("x3:")
        print(x3[0:10,0])
        print(bp_x3grad[0:10,0])
        print("x2:")
        print(x2[0:10,0])
        print(bp_x2grad[0:10,0])
        print("x1:")
        print(x1[0:10,0])
        print(bp_x1grad[0:10,0])
        w3grad = old_x2.T @ (x3 * tanh_deriv(old_x2 @ w3))
        print(w3grad.shape)
        print(bp_w3grad.shape)
        w2grad = old_x1.T @ (x2 * tanh_deriv(old_x1 @ w2))
        w1grad = img.T @ (x1 * tanh_deriv(img @ w1))
        #plt.plot(x3s)
        #plt.show()
        #plt.plot(x2s)
        #plt.show()
        #plt.plot(x1s)
        #plt.show()
        print("W3")
        print(w3grad[0:10,0])
        print(bp_w3grad[0:10,0])
        print("W2")
        print(w2grad[0:10,0])
        print(bp_w2grad[0:10,0])
        print("W1")
        print(w1grad[0:10,0])
        print(bp_w1grad[0:10,0])
        
        print(torch.sum(w3grad - bp_w3grad))
        print(torch.sum(w2grad - bp_w2grad))
        print(torch.sum(w1grad - bp_w1grad))
    return e3s,e2s,e1s


def plot_numerical_divergence(errors,i):
    fig,ax = plt.subplots(1,1,figsize=(9,7))
    plt.title("Divergence from True Gradient",fontsize=20,fontweight="bold",pad=25)
    ax.plot(errors)
    plt.xlabel("Iteration",fontsize=20,style="oblique",labelpad=10)
    plt.ylabel("MSE between predicted and true gradient",fontsize=20,style="oblique",labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = plt.legend(prop={"size":8})
    legend.fontsize=18
    legend.style="oblique"
    frame  = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")
    ax.tick_params(axis='both',which='major',labelsize=20)
    ax.tick_params(axis='both',which='minor',labelsize=18)
    fig.tight_layout()
    fig.savefig("figures/numerics_proper_divergence_layer_" + str(i) +".jpg")
    plt.show()

def plot_numerical_divergence_layers(errors):
    fig,ax = plt.subplots(1,1,figsize=(9,7))
    plt.title("Layerwise Divergences from True Gradient",fontsize=20,fontweight="bold",pad=25)
    for (i,es) in enumerate(errors):
        ax.plot(es,label="Layer " + str(i+1))
        plt.xlabel("Iteration",fontsize=20,style="oblique",labelpad=10)
    plt.ylabel("MSE between predicted and true gradient",fontsize=20,style="oblique",labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = plt.legend(prop={"size":16})
    legend.fontsize=18
    legend.style="oblique"
    frame  = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")
    ax.tick_params(axis='both',which='major',labelsize=20)
    ax.tick_params(axis='both',which='minor',labelsize=18)
    fig.tight_layout()
    fig.savefig("figures/lnumerics_proper_divergence_layers" +".jpg")
    plt.show()
    


def plot_learning_rate_comparison(errors_list, lrs,i,log_scale=False):
    fig,ax = plt.subplots(1,1,figsize=(9,7))
    plt.title("Learning Rate Comparison Layer " + str(i),fontsize=20,fontweight="bold",pad=25)
    for (ey0,lr) in zip(errors_list,lrs):
        labelstr = "Learning Rate " + str(lr)
        ax.plot(ey0,label=labelstr)
    plt.xlabel("Iteration",fontsize=20,style="oblique",labelpad=10)
    #plt.ylabel("MSE between predicted and true gradient",fontsize=20,style="oblique",labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    log = ""
    if log_scale:
        plt.yscale("log")
        log = "_log"
    legend = plt.legend(prop={"size":16})
    legend.fontsize=16
    legend.style="oblique"
    frame  = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")
    ax.tick_params(axis='both',which='major',labelsize=20)
    ax.tick_params(axis='both',which='minor',labelsize=18)
    fig.tight_layout()
    fig.savefig("figures/numerics_proper_learning_rate_comparison_layer_" +str(i)+".jpg")
    plt.show()

def learning_rate_comparison(lrs,num_iterations=100):
    l3_errs = []
    l2_errs = []
    l1_errs = []
    for lr in lrs:
        e3s,e2s,e1s = numerical_test(lr, num_iterations)
        l3_errs.append(e3s)
        l2_errs.append(e2s)
        l1_errs.append(e1s)
    for (i,errs) in enumerate([l3_errs,l2_errs,l1_errs]):
        plot_learning_rate_comparison(errs,lrs,i+1)

if __name__ == '__main__':
    #baseline result
    base_lr = 0.1
    e3s,e2s,e1s = numerical_test(base_lr,100)
    #plot_numerical_divergence(e3s,3)
    #plot_numerical_divergence(e2s,2)
    #plot_numerical_divergence(e1s,1)
    plot_numerical_divergence_layers([e3s,e2s,e1s])
    #learning rate comparison
    lrs = [0.5,0.2,0.1,0.05,0.02,0.01,0.005]
    learning_rate_comparison(lrs,200)


