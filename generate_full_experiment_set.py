import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
bcall = "python main.py --store_gradient_angle True"
output_file = open(generated_name, "w")
seeds = 5
datasets = ["mnist","fashion"]
for dataset in datasets:
    base_call = bcall + " --dataset " + str(dataset)
    for s in range(seeds):
        condition=dataset+"_default"
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath)
        print(final_call)
        print(final_call, file=output_file)

    condition=dataset+"_backwards_weights_with_update"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --update_backwards_weights True"
        print(final_call)
        print(final_call, file=output_file)
        
    condition=dataset+"_feedback_alignment"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --update_backwards_weights False"
        print(final_call)
        print(final_call, file=output_file)
        
    condition=dataset+"_feedback_alignment_no_nonlinearity"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --update_backwards_weights False --use_backward_nonlinearity False"
        print(final_call)
        print(final_call, file=output_file)

    condition=dataset+"_no_nonlinearities"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backward_nonlinearity False"
        print(final_call)
        print(final_call, file=output_file)

    condition=dataset+"_full_construct"
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --use_backwards_weights True --use_backward_nonlinearity False --update_backwards_weights True"
        print(final_call)
        print(final_call, file=output_file)

        condition=dataset+"_bp"
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --network_type bp"
        print(final_call)
        print(final_call, file=output_file)
