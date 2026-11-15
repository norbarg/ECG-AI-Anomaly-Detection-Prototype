ECG MODEL TESTER — first working version

Models included:
- models/colab_int8.tflite
- models/jupyter_int8.tflite

Run on Windows/macOS/Linux:
1) Open a terminal in this folder.
2) Run: python -m http.server 8000 --bind 0.0.0.0
3) On the PC open: http://localhost:8000

Do NOT open index.html by double-clicking (file://), because browsers block fetching .tflite/WASM files.
Internet connection is currently required for the LiteRT.js runtime files from CDN.

ECG input format:
- .csv or .txt
- exactly 360 numeric ECG samples is ideal
- if the file contains more values, the first 360 numeric values are used
- separators can be commas, semicolons, spaces or newlines

Workflow:
1) Select Colab or Jupyter model.
2) Click Load model.
3) Choose an ECG CSV/TXT file.
4) Click Run inference.
5) Click Benchmark ×100 for stable latency.

If an INT8 quantization error is shown:
The browser runtime did not expose scale/zero-point in a field this app recognizes.
Open “INT8 quantization override” and enter input/output scale and zero-point from TensorFlow Lite interpreter details.

Later steps:
- add Edge Impulse runtime as third model
- test the same page from iPhone Safari
- add ARM64/Raspberry Pi-like run
