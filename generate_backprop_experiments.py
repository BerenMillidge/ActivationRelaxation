import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
base_call = "python main.py"
output_file = open(generated_name, "w")
seeds = 5

"""condition="mnist_AR"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) +" --dataset mnist"
    print(final_call)
    print(final_call, file=output_file)

condition="mnist_BP"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --dataset mnist" + " --network_type bp"
    print(final_call)
    print(final_call, file=output_file)"""

condition="svhn_AR"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) +" --dataset svhn"
    print(final_call)
    print(final_call, file=output_file)

condition="svhn_BP"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --dataset svhn" + " --network_type bp"
    print(final_call)
    print(final_call, file=output_file)

"""condition="fashion_AR"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) +" --dataset fashion"
    print(final_call)
    print(final_call, file=output_file)

condition="fashion_BP"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --dataset fashion" + " --network_type bp"
    print(final_call)
    print(final_call, file=output_file)"""