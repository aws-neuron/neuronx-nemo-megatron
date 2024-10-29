import tensorboard.backend.event_processing.event_accumulator as event_acc
import argparse
from collections import defaultdict
import os


# Constants for step thresholds
LATE_TRAINING_STEP_THRESHOLD = 450
EARLY_TRAINING_STEP_MIN = 5


def load_events(event_file):
    accumulator = event_acc.EventAccumulator(event_file)
    accumulator.Reload()
    tags = accumulator.Tags()

    data = {}
    for tag in tags['scalars']:
        data[tag] = accumulator.Scalars(tag)
    return data

def should_compare_step(trn_step, args):
    """Determine if the current step should be compared."""
    if args.compare_first_iters:
        return EARLY_TRAINING_STEP_MIN < trn_step <= args.train_iters
    else:
        return trn_step > LATE_TRAINING_STEP_THRESHOLD

def calculate_percentage_difference(delta, max_val):
    """Calculate the percentage difference, avoiding division by zero."""
    if max_val > 0:
        return (delta / max_val) * 100
    else:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Compare GPU and Trn1 TensorBoard event files.")
    parser.add_argument("gpu_event_file", help="Path to the GPU TensorBoard file.")
    parser.add_argument("trn1_event_file", help="Path to the Trn1 TensorBoard file.")
    parser.add_argument("tags", nargs='+',
                        help="List of tags to compare. if the tag is the form of 'tag1,tag2', then the first one is gpu tag \
                              the second one is the trn1 tag")
    parser.add_argument("--smoothed_weight", type=float, default=0, 
                                                help="Smoothing factor for the values.Value between 0 and 1")
    parser.add_argument("--delta_percentage", type=float, default=1.0,
                        help="The tolerated percentage difference.")
    parser.add_argument("--compare_first_iters", action="store_true",
                                                help="Compare first n iters only")
    parser.add_argument("--train_iters", type=int, default=100, help="Number of iters to compare up to")

    args = parser.parse_args()

    if not os.path.exists(args.gpu_event_file):
        raise FileNotFoundError(f"{args.gpu_event_file} not found")
    if not os.path.exists(args.trn1_event_file):
        raise FileNotFoundError(f"{args.trn1_event_file} not found")

    gpu_events = load_events(args.gpu_event_file)
    trn1_events = load_events(args.trn1_event_file)
    for tag in args.tags:
        tags = tag.split(',')
        if len(tags) == 1:
            gpu_tag = trn_tag = tags[0]
        elif len(tags) == 2:
            gpu_tag, trn_tag = tags
        else:
            raise ValueError(f"The tag field is incorrect {tag}")

        if gpu_tag in gpu_events and trn_tag in trn1_events:
            trn1_last_value = trn1_events[trn_tag][0].value # First value in the trn plot (first timestep)
            gpu_last_value = gpu_events[gpu_tag][0].value # First value in the gpu plot (first timestep)

            # Create a lookup for step to smoothed gpu value for efficient comparisons
            gpu_events_lookup = defaultdict(lambda: None)
            for gpu in gpu_events[gpu_tag]:
                gpu_value = gpu.value
                smoothed_gpu_val = gpu_last_value * args.smoothed_weight + (1 - args.smoothed_weight) * gpu_value
                gpu_events_lookup[gpu.step] = smoothed_gpu_val
                gpu_last_value = smoothed_gpu_val
            missing_steps = 0

            for trn in trn1_events[trn_tag]:
                trn1_value = trn.value
                smoothed_trn_val = trn1_last_value * args.smoothed_weight + (1 - args.smoothed_weight) * trn1_value
                smoothed_gpu_val = gpu_events_lookup[trn.step]

                if(smoothed_gpu_val is not None):
                    smoothed_gpu_val = gpu_events_lookup[trn.step]
                    delta = abs(smoothed_gpu_val - smoothed_trn_val)
                    max_val = max(abs(smoothed_gpu_val), abs(smoothed_trn_val))
                    trn1_last_value = smoothed_trn_val

                    if should_compare_step(trn.step, args):
                        percentage_difference = calculate_percentage_difference(delta, max_val)
                        if percentage_difference is not None and percentage_difference > args.delta_percentage:
                            raise ValueError(
                                f"\n====== Delta percentage exceeds tolerance for tag '{trn_tag}' at step {trn.step}.\n"
                                f"TRN1 value: {smoothed_trn_val}, GPU value: {smoothed_gpu_val}, "
                                f"Difference: {percentage_difference:.2f}% (Tolerance: {args.delta_percentage}%)"
                            )
                else:
                    if args.compare_first_iters:
                        is_within_comparison_range = trn.step <= args.train_iters
                    else:
                        is_within_comparison_range = trn.step > LATE_TRAINING_STEP_THRESHOLD

                    # Increment missing_steps if the step is within the comparison range
                    if is_within_comparison_range:
                        missing_steps += 1
            print(f"Missing steps were {missing_steps}")
        else:
            raise ValueError(f"Tag 'gpu:{gpu_tag}' or 'trn:{trn_tag}' not found in one of the event files")


if __name__ == "__main__":
    main()
