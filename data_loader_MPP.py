
import pandas as pd
import pickle
import statistics

# Define the file path structure
file_path_template = "MPP_FE/{datasetname}/{datasetname}{filtration}.pkl"
file_lavel_template = "MPP_FE/{datasetname}/{datasetname}_labels.csv"
file_path_FP = "FP/{datasetname}_fp.pkl"


# Function to load dataset by name
def load_dataset(datasetname):
    # Construct the file path
    file_path = file_path_template.format(datasetname=datasetname, filtration='ricci')
    file_path2 = file_path_template.format(datasetname=datasetname, filtration='atom')
    file_path_label = file_lavel_template.format(datasetname=datasetname)
    file_path_fp = file_path_FP.format(datasetname=datasetname)

    try:
        # Load the CSV file into a DataFrame
        with open(file_path, 'rb') as g:
            features_ricci = pickle.load(g)

        with open(file_path2, 'rb') as g:
            features_atom = pickle.load(g)
        with open(file_path_fp, 'rb') as g:
            features_fp = pickle.load(g)

        number_graph = len(features_ricci[0])
        number_slices_r = len(features_ricci[0][0])
        number_slices_a = len(features_atom[0][0])

        feat_ricci = [
            [[features_ricci[k][i][j] for k in range(4)] for j in range(number_slices_r)]
            for i in range(number_graph)
        ]
        feat_atom = [
            [[features_atom[k][i][j] for k in range(4)] for j in range(number_slices_a)]
            for i in range(number_graph)
        ]

        print(f"Successfully loaded dataset: {datasetname} features")
        La = pd.read_csv(file_path_label)
        Label = La.values.tolist()
        # Label=La['y'].to_numpy()
        # Label=Label[:, np.newaxis]
        return feat_ricci, feat_atom, features_fp, Label
    except FileNotFoundError:
        print(f"Error: File for dataset '{datasetname}' not found.")
        return None


def stat(acc_list, metrics):
    mean = statistics.mean(acc_list)
    stdev = statistics.stdev(acc_list)
    print(f'Final {metrics}  using 10 run: {mean:.4f} \u00B1 {stdev:.4f}%')