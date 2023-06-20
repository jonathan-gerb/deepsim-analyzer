# main python file to run UI
import os
import argparse
from pathlib import Path
from deepsim_analyzer import create_dataset, read_dataset_keys, calculate_features
from deepsim_analyzer import calculate_projection, project_feature

# from deepsim_analyzer.deepsim_dashboard.home import start_dashboard
from deepsim_analyzer.ds_dashboard.mainwindow import start_dashboard


def prepare_dataset(args):
    target_features = args.target_features
    print("preparing dataset!")
    if os.path.exists(args.dataset_file):
        print(f"    found dataset file at: {args.dataset_file}")
        if args.refresh:
            print("        Refreshing dataset...")
            print(f"        Image folder filepath: {args.image_folder}")

            dataset_path = Path(args.dataset_file)
            print(
                f"        Keeping old dataset at: {dataset_path.with_name('backup_dataset.h5')} "
            )
            backup_path = dataset_path.rename(
                dataset_path.with_name("backup_dataset.h5")
            )
            create_dataset(args.image_folder, args.dataset_file)
            print(f"        Finished creating new dataset at {dataset_path}! ")
            print(f"        Removing old dataset at: {backup_path}")
            os.remove(backup_path)

            print(f"Calculating features for images")
            # CHANGE THIS HERE TO ADD NEW FEAUTRES TO THE LIST OF FEATURES TO CALCULATE
            calculate_features(
                args.image_folder, args.dataset_file, target_features=["dummy", "texture", "dino"]
            )
    else:
        print(f"    creating new dataset from images in {args.image_folder}")
        create_dataset(args.image_folder, args.dataset_file)

        print(f"Calculating features for images")
        calculate_features(
            args.image_folder, args.dataset_file, target_features=["dummy", "texture", "dino"]
        )

    if args.project:
        print("calculating projection of image features")
        target_features = ["dummy", "dino", "texture"]
        for feature_name in target_features:
            calculate_projection(
                args.dataset_file, feature_name, overwrite=args.refresh
            )

    print(f"dataset prepared at: {args.dataset_file}")


def parse_arguments():
    dataset_default_location = str(
        Path(f"{__file__}").parents[1] / "data" / "processed" / "dataset.h5"
    )
    images_default_location = str(
        Path(f"{__file__}").parents[1] / "data" / "raw_immutable" / "test_images"
    )

    parser = argparse.ArgumentParser(description="Dataset Refresh Script")
    parser.add_argument(
        "-r", "--refresh", action="store_true", help="Refresh the dataset"
    )
    parser.add_argument(
        "-p",
        "--project",
        action="store_true",
        help="Calculates umap for all the features in the dataset",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        help="Filepath to the image folder",
        default=images_default_location,
    )
    parser.add_argument(
        "-d",
        "--dataset_file",
        type=str,
        help="Filepath to the dataset file",
        default=dataset_default_location,
    )
    # please update the default of this argument when more features are implemented
    parser.add_argument(
        "-f",
        "--target_features",
        nargs="+",
        default=["dummy"],
        help="default features to calculate for each image",
    )
    return parser.parse_args()


def start_gui(args):
    print(f"Loading keys from dataset: {args.dataset_file}")
    key_dict = read_dataset_keys(args.dataset_file)
    print(f"Found {len(key_dict)} image keys in dataset")
    print("starting DeepSim dashboard")
    start_dashboard(key_dict, args.dataset_file, args.image_folder)


def main():
    # Example usage:
    args = parse_arguments()
    prepare_dataset(args)
    start_gui(args)


if __name__ == "__main__":
    main()
