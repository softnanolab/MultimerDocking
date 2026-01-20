import pandas as pd
import numpy as np
import fire

def print_metrics(csv_path: str):
    df = pd.read_csv(csv_path)
    data_dict = df.to_dict(orient="list")
    print("Loaded the following columns: ", data_dict.keys())

    print("Dimer RMSD: ", np.mean(data_dict["dimer_rmsd"]))
    print("Chain RMSD: ", np.mean(data_dict["chainA_rmsd"] + data_dict["chainB_rmsd"]))
    print("DockQ: ", np.mean(data_dict["dockq"]))
    print("fnat: ", np.mean(data_dict["fnat"]))
    print("iRMSD: ", np.mean(data_dict["irmsd"]))
    print("LRMSD: ", np.mean(data_dict["lrmsd"]))
    print("DockQ Failed (count): ", np.sum(data_dict["dockq_failed"]))


def main(csv_path: str):
    print_metrics(csv_path)

if __name__ == "__main__":
    fire.Fire(main)