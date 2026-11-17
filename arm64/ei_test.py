import csv
import importlib.util
import sys
import time

runner_path = "/usr/local/lib/python3.11/site-packages/edge_impulse_linux/runner.py"

spec = importlib.util.spec_from_file_location("edge_impulse_runner", runner_path)

runner_module = importlib.util.module_from_spec(spec)
sys.modules["edge_impulse_runner"] = runner_module
spec.loader.exec_module(runner_module)

ImpulseRunner = runner_module.ImpulseRunner

MODEL = "arm64/lab4.1_embeddedai-linux-aarch64-v4-impulse-#1.eim"
CSV_FILE = "data/ecg_validated_balanced_200.csv"

runner = ImpulseRunner(MODEL)

try:
    model_info = runner.init()

    print("Model loaded:")
    print(model_info["project"]["name"])
    print("Labels:", model_info["model_parameters"]["labels"])
    print()

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    features = [float(row[f"x{i}"]) for i in range(360)]
    true_label = int(row["label"])

    start = time.perf_counter()
    result = runner.classify(features)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print("True label:", "arrhythmia" if true_label == 1 else "normal")
    print("Classification:")
    print(result["result"])
    print()

    if "timing" in result:
        print("Edge Impulse timing:")
        print(result["timing"])

    print(f"Total runner time: {elapsed_ms:.3f} ms")

finally:
    runner.stop()
