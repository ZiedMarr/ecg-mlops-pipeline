# run_preprocessing.py (separate entry point, not called from main.py)
from ecg_bsa.config import get_config
from ecg_bsa.pipeline_preprocessing import preprocess_dataset

def main():
    config = get_config()
    

    #set number of subjects
    num_pat = config["raw_dataset"]["num_subjects"]
    segment_size = config["dataset"]["segment_length"]
    overlap_ratio = config["preprocess"]["overlap_ratio"]
    preprocess_dataset(
        segment_size=config["dataset"]["segment_length"],
        overlap_ratio=config["preprocess"]["overlap_ratio"],
        num_pat=config["raw_dataset"]["num_subjects"],
        sampling_rate=config["dataset"]["sampling_rate"],
        powerline=config["preprocess"]["powerline"],
        lowcut=config["preprocess"]["lowcut"],
        highcut=config["preprocess"]["highcut"],
        downsampled_rate=config["preprocess"]["downsampled_rate"],
        raw_path=config["paths"]["raw_data"],
        save_path=config["paths"]["processed_data"],
        expected_channels=config["dataset"]["input_channels"],
        expected_segment_length=config["dataset"]["segment_length"],
        expected_num_classes=config["dataset"]["num_classes"],
)

if __name__ == "__main__":
    main()