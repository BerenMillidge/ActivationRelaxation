import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
base_call = "python main.py"
output_file = open(generated_name, "w")
seeds = 5
datasets = ["svhn","fashion"]
for dataset in datasets:
    base_call = base_call + " --dataset " + str(dataset)
    condition="default"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath)
        print(final_call)
        print(final_call, file=output_file)

    condition="backwards_weights_with_update"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --update_backwards_weights True"
        print(final_call)
        print(final_call, file=output_file)
        
    condition="feedback_alignment"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --update_backwards_weights False"
        print(final_call)
        print(final_call, file=output_file)
        
    condition="feedback_alignment_no_nonlinearity"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --update_backwards_weights False --use_backward_nonlinearity False"
        print(final_call)
        print(final_call, file=output_file)

    condition="no_nonlinearities"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backward_nonlinearity False"
        print(final_call)
        print(final_call, file=output_file)

    condition="full_construct"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --use_backward_nonlinearity False --update_backwards_weights True"
        print(final_call)
        print(final_call, file=output_file)
