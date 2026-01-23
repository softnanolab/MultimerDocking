import pandas as pd
import numpy as np
import fire
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

def plot_metric_histograms(csv_path: str):
    df = pd.read_csv(csv_path)
    dockq_failed_count = np.sum(df["dockq_failed"])
    df = df[df['dockq_failed'] == 0]  # Keep only successful dockq calculations

    df_sorted = df.sort_values('dockq', ascending=False)  # Sort by dockq from high to low

    # Export to CSV
    output_path = str(Path(csv_path).parent / "test_metrics_sorted.csv")
    df_sorted.to_csv(output_path, index=False)

    # Copy top 3 and bottom 3 sample files
    samples_dir = Path(csv_path).parent / "samples"
    edge_samples_dir = Path(csv_path).parent / "edge_samples"
    edge_samples_dir.mkdir(parents=True, exist_ok=True)
    
    for protein_id in df_sorted.head(3)["protein_id"].tolist() + df_sorted.tail(3)["protein_id"].tolist():
        for suffix in ["_pred.cif", "_true.cif"]:
            source = samples_dir / f"{protein_id}{suffix}"
            if source.exists():
                shutil.copy2(source, edge_samples_dir / f"{protein_id}{suffix}")

    data_dict = df.to_dict(orient="list")
    print("Loaded the following columns: ", data_dict.keys())

    save_dir = Path(csv_path).parent / "metric_histograms"
    save_dir.mkdir(parents=True, exist_ok=True)

    fraction_correct_dimers = np.sum(np.array(data_dict["dockq"]) >= 0.23) / len(data_dict["dockq"])
    fraction_incorrect_dimers = np.sum(np.array(data_dict["dockq"]) < 0.23) / len(data_dict["dockq"])

    for key in data_dict.keys():
        if key == "dockq_failed" or key == "protein_id":
            continue
        plt.hist(data_dict[key], bins=50)
        plt.ylabel("Frequency")
        plt.title(f"{key} distribution")
        if "dockq" in key:
            plt.title(f"{key} distribution (fraction correct: {fraction_correct_dimers:.2f}, fraction incorrect: {fraction_incorrect_dimers:.2f})")
            plt.vlines(0.23, 0, plt.ylim()[1], colors="red", linestyles="dashed")
        if "rmsd" in key:
            plt.xlabel(f"{key} (Å)")
        else:
            plt.xlabel(key)
        plt.savefig(str(save_dir / f"{key}_distribution.png"))
        plt.close()
        print(f"{key} histogram saved.")

    print("Dimer RMSD (mean): ", np.mean(data_dict["dimer_rmsd"]))
    print("Monomer Chain RMSD (mean): ", np.mean(data_dict["monomer_chainA_rmsd"] + data_dict["monomer_chainB_rmsd"]))
    print("Cross Chain RMSD (mean): ", np.mean(data_dict["cross_chain_rmsd_A"] + data_dict["cross_chain_rmsd_B"]))
    print("DockQ (mean): ", np.mean(data_dict["dockq"]), f"({fraction_correct_dimers:.2f} correct, {fraction_incorrect_dimers:.2f} incorrect)")
    print("fnat (mean): ", np.mean(data_dict["fnat"]))
    print("iRMSD (mean): ", np.mean(data_dict["irmsd"]))
    print("LRMSD (mean): ", np.mean(data_dict["lrmsd"]))
    print("DockQ Failed (count): ", dockq_failed_count)

def main(csv_path: str):
    plot_metric_histograms(csv_path)

if __name__ == "__main__":
    fire.Fire(main)