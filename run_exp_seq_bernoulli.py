import argparse
import os

from algorithm import ConUCB, RelativeConUCB, LinUCB
from env import BernoulliEnv
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main experiment.")
    parser.add_argument("--in_folder", dest="in_folder",
                        help="input the folder containing input files")
    parser.add_argument("--out_folder", dest="out_folder",
                        help="input the folder to output")
    parser.add_argument("--arm_pool_size", dest="arm_pool_size",
                        type=int, help="pool_size of each iteration")
    parser.add_argument("--num_repeat", dest="num_repeat",
                        type=int, help="# of repeat")
    args = parser.parse_args()

    # Preprocess.
    datasetname = args.in_folder.strip("/").split("/")[-1]
    print("Using dataset {}".format(datasetname))
    print("Results will be save at '{}'".format(args.out_folder))
    X, tilde_X, arm_affinity_matrix, arm_to_suparms, suparm_to_arms = utils.prepare_data(args.in_folder)
    assert X.shape[1] == tilde_X.shape[1]
    dim = X.shape[1]
    num_user = len(arm_affinity_matrix)

    # Initialize the experiment.
    assert os.path.exists(args.out_folder)
    simulate_exp = BernoulliEnv(
        X,
        tilde_X,
        arm_affinity_matrix,
        arm_to_suparms,
        suparm_to_arms,
        out_folder=os.path.join(args.out_folder, datasetname),
        device=utils.DEVICE,
        arm_pool_size=50,
        relative_noise=0.1,
        budget_func=utils.BUDGET_FUNCTION,
        is_early_register=False,
        num_iter=num_user * 400
    )

    algorithms = {
        "LinUCB": LinUCB(dim, utils.DEVICE),
        "ConUCB": ConUCB(dim, utils.DEVICE),
        "ConUCB_share-attribute": ConUCB(dim, utils.DEVICE, is_update_all_attribute=True),
    }

    # pos.
    for select_pair_mechanism in ["doublebestrelated2"]:
        algorithm_name = "RelativeConUCB_{}_pos".format(select_pair_mechanism)
        algorithms[algorithm_name] = RelativeConUCB(dim, utils.DEVICE, select_pair_mechanism, "pos")
        algorithm_name = "RelativeConUCB_{}_pos_share-attribute".format(select_pair_mechanism)
        algorithms[algorithm_name] = RelativeConUCB(dim, utils.DEVICE, select_pair_mechanism, "pos", is_update_attribute_all=True)

    # pos&neg.
    algorithm_name = "RelativeConUCB_{}_pos&neg".format("doublebestrelated2")
    algorithms[algorithm_name] = RelativeConUCB(dim, utils.DEVICE, "doublebestrelated2", "pos&neg")
    algorithm_name = "RelativeConUCB_{}_pos&neg_share-attribute".format("doublebestrelated2")
    algorithms[algorithm_name] = RelativeConUCB(dim, utils.DEVICE, "doublebestrelated2", "pos&neg", is_update_attribute_all=True)

    # difference.
    for select_pair_mechanism in ["bestdiffrelated2", "bestthendiffrelated2"]:
        algorithm_name = "RelativeConUCB_{}_difference".format(select_pair_mechanism)
        algorithms[algorithm_name] = RelativeConUCB(dim, utils.DEVICE, select_pair_mechanism, "difference")
        algorithm_name = "RelativeConUCB_{}_difference_share-attribute".format(select_pair_mechanism)
        algorithms[algorithm_name] = RelativeConUCB(dim, utils.DEVICE, select_pair_mechanism, "difference", is_update_attribute_all=True)

    simulate_exp.run_algorithms(algorithms, num_repeat=args.num_repeat)
    print("Results was saved at '{}'".format(args.out_folder))