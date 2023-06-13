from umap import UMAP
import numpy as np
import h5py
import joblib
from pathlib import Path
from deepsim_analyzer.io import save_feature

def calculate_projection(dataset_file, feature_name, overwrite=False, reducer_filename=""):
    with h5py.File(dataset_file, "r") as f:
        keys = list(f.keys())
        image_features = []
        for key in keys:
            feature = f[f"{key}/features/{feature_name}/full"]
            image_features.append(feature)
        image_features = np.stack(image_features, axis=0)

    print(f"calculating umap projection for image feature matrix with shape: {image_features.shape}")
    reducer = UMAP(n_neighbors=6, n_components=2, metric='cosine').fit(image_features)
    transformed_data = reducer.transform(image_features)
    for i, key in enumerate(keys):
        vector_2d = transformed_data[i]
        save_feature(dataset_file, key, vector_2d, feature_name, is_projection=True, overwrite=overwrite)
    
    if reducer_filename != "":
        filename = reducer_filename
    else:
        filename = f"{feature_name}.umap"
    filepath = Path(__file__).parents[1] / "data" / "processed" / "reducers" / filename
    print(f"saving umap reducer model at: {filepath}")
    joblib.dump(reducer, filepath)
    return transformed_data

def project_feature(feature_vector, reducer_path):
    reducer = joblib.load(reducer_path)
    transformed_feature = reducer.transform(feature_vector)
    return transformed_feature


