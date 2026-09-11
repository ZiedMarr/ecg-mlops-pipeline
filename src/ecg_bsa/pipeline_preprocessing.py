from prefect import task, flow, unmapped, get_run_logger
from ecg_bsa.preprocessing import form_subject_dict, preprocessing_pipeline, segmentation, save_subject_file,clear_processed_files,validate_processed_files, print_processed_label_statistics
import os
import pandas as pd


@task(retries=2)
def process_subject(subject_number, filter_params, signals_path, reference_df, segment_size, overlap_ratio, expected_num_classes, save_path):
    subject_dict = form_subject_dict(signals_path, reference_df, subject_number, expected_num_classes)
    subject_dict["signals"] = preprocessing_pipeline(subject_dict["signals"], filter_params)
    segmented_list = segmentation(subject_dict=subject_dict, segment_size=segment_size, overlap_ratio=overlap_ratio)
    return save_subject_file(segmented_list, save_path)


@flow
def preprocess_dataset(segment_size, overlap_ratio, num_pat, 
                       sampling_rate, powerline, lowcut, highcut, downsampled_rate, 
                       raw_path="./data/raw/", save_path="./data/processed/", 
                       expected_channels=12, expected_segment_length=1500, expected_num_classes=9):
    #set paths
    logger = get_run_logger()

    filter_params = {
        "sampling_rate": sampling_rate,
        "powerline": powerline,
        "lowcut": lowcut,
        "highcut": highcut,
        "downsampled_rate": downsampled_rate,
    }

    signals_path = os.path.join(raw_path, "Training_WFDB")
    ref_path = os.path.join(raw_path, "REFERENCE.csv")
    reference_df = pd.read_csv(ref_path)
    clear_processed_files(save_path)


    futures = process_subject.map(range(1, num_pat + 1),
        filter_params=unmapped(filter_params),
        signals_path=unmapped(signals_path),
        reference_df=unmapped(reference_df),
        segment_size=unmapped(segment_size),
        overlap_ratio=unmapped(overlap_ratio),
        expected_num_classes=unmapped(expected_num_classes),
        save_path=unmapped(save_path),)

    processed, skipped = 0, 0
    for f in futures:
        try:
            f.result()
            processed += 1
        except Exception:
            skipped += 1

    logger.info(f"Subjects processed: {processed}, skipped: {skipped}")

    validate_processed_files(save_path, expected_channels,
            expected_segment_length,
            expected_num_classes )
    print_processed_label_statistics(save_path, expected_num_classes)

    return {"processed": processed, "skipped": skipped}

