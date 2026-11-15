import { loadLiteRt, loadAndCompile, Tensor } from '@litertjs/core';

const MODELS = {
    colab: {
        name: 'Colab · Weight-only quantization',
        runtime: 'litert',
        url: './models/colab_weight_quant.tflite',
        size: 61276,
    },

    jupyter: {
        name: 'Jupyter · Weight-only quantization',
        runtime: 'litert',
        url: './models/jupyter_weight_quant.tflite',
        size: 61276,
    },

    edge: {
        name: 'Edge Impulse · Quantized INT8',
        runtime: 'edge',
        size: 113.7 * 1024,
    },
};

const $ = (id) => document.getElementById(id);

/* =========================================================
   LOG
========================================================= */

function log(message, object = null) {
    $('log').textContent += `\n${message}${
        object ? `\n${JSON.stringify(object, null, 2)}` : ''
    }`;

    $('log').scrollTop = $('log').scrollHeight;
}

/* =========================================================
   STATE
========================================================= */

let liteRtReady = false;

let compiledModel = null;
let loadedLiteRtModel = null;

let inputDetails = null;
let outputDetails = null;

let dataset = null;

/* Edge Impulse worker */

let edgeWorker = null;

let edgeWorkerReady = false;

let edgeRequestId = 0;

const edgeRequests = new Map();

/* =========================================================
   ENVIRONMENT
========================================================= */

function detectEnvironment() {
    $('platform').textContent = navigator.platform || 'Unknown';

    $('threads').textContent = navigator.hardwareConcurrency || 'Unknown';

    const ua = navigator.userAgent;

    let architecture = 'Unknown / browser-hidden';

    if (/iPhone|iPad|iPod/i.test(ua)) {
        architecture = 'Apple mobile / Safari';
    } else if (/arm64|aarch64/i.test(ua)) {
        architecture = 'ARM64';
    } else if (/arm/i.test(ua)) {
        architecture = 'ARM';
    } else if (/x86_64|Win64|x64|amd64/i.test(ua)) {
        architecture = 'x86-64';
    }

    $('architecture').textContent = architecture;
}

/* =========================================================
   LITERT
========================================================= */

async function initLiteRt() {
    try {
        await loadLiteRt(
            'https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/',
        );

        liteRtReady = true;

        log('LiteRT.js runtime initialized.');

        updateRuntimeBadge();

        updateAnalyzeButton();
    } catch (error) {
        liteRtReady = false;

        log(
            'LiteRT.js initialization failed: ' +
                (error?.message || String(error)),
        );

        updateRuntimeBadge();

        updateAnalyzeButton();
    }
}

/* =========================================================
   EDGE IMPULSE WORKER
========================================================= */

function initEdgeWorker() {
    if (edgeWorker) {
        return;
    }

    edgeWorker = new Worker('./edge-impulse/edge-worker.js');

    edgeWorker.onmessage = (event) => {
        const { id, ok, error, ...payload } = event.data;

        const pending = edgeRequests.get(id);

        if (!pending) {
            return;
        }

        edgeRequests.delete(id);

        if (ok) {
            pending.resolve(payload);
        } else {
            pending.reject(new Error(error || 'Edge Impulse worker error.'));
        }
    };

    edgeWorker.onerror = (error) => {
        log('Edge Impulse Worker error: ' + (error.message || String(error)));
    };
}

function callEdgeWorker(type, payload = {}) {
    initEdgeWorker();

    return new Promise((resolve, reject) => {
        const id = ++edgeRequestId;

        edgeRequests.set(id, {
            resolve,
            reject,
        });

        edgeWorker.postMessage({
            id,
            type,
            ...payload,
        });
    });
}

async function loadEdgeImpulseModel() {
    if (edgeWorkerReady) {
        $('modelSize').textContent = '113.70 KB';

        $('inputInfo').textContent = '360 features · Edge Impulse WASM';

        return;
    }

    log('Loading Edge Impulse · Quantized INT8...');

    const info = await callEdgeWorker('init');

    edgeWorkerReady = true;

    $('modelSize').textContent = '113.70 KB';

    $('inputInfo').textContent = '360 features · Edge Impulse WASM';

    log('Edge Impulse runtime initialized.', info);
}

/* =========================================================
   RUNTIME UI
========================================================= */

function updateRuntimeBadge() {
    const key = $('modelSelect').value;

    if (key === 'edge') {
        $('runtimeBadge').textContent = 'Runtime: Edge Impulse / WASM SIMD';
    } else {
        $('runtimeBadge').textContent = liteRtReady
            ? 'Runtime: LiteRT.js / WASM'
            : 'Runtime: LiteRT.js unavailable';
    }
}

/* =========================================================
   MODEL DETAILS
========================================================= */

function normalizeDetails(details) {
    if (!details) {
        return null;
    }

    return JSON.parse(
        JSON.stringify(details, (key, value) =>
            typeof value === 'bigint' ? String(value) : value,
        ),
    );
}

function inferShape(detail) {
    const shape =
        detail?.shape ??
        detail?.dimensions ??
        detail?.layout?.dimensions ??
        detail?.type?.layout?.dimensions ??
        detail?.tensorType?.layout?.dimensions;

    if (!shape) {
        return [1, 360, 1];
    }

    return Object.values(shape).map(Number);
}

/* =========================================================
   LITERT MODEL
========================================================= */

async function loadLiteRtModel(modelKey) {
    if (!liteRtReady) {
        throw new Error('LiteRT.js runtime is not available.');
    }

    if (compiledModel && loadedLiteRtModel === modelKey) {
        return;
    }

    const info = MODELS[modelKey];

    log(`Loading ${info.name}...`);

    if (compiledModel?.delete) {
        compiledModel.delete();
    }

    compiledModel = await loadAndCompile(info.url, {
        accelerator: 'wasm',
    });

    loadedLiteRtModel = modelKey;

    inputDetails = compiledModel.getInputDetails?.()?.[0] ?? null;

    outputDetails = compiledModel.getOutputDetails?.()?.[0] ?? null;

    const shape = inferShape(inputDetails);

    $('modelSize').textContent = `${(info.size / 1024).toFixed(2)} KB`;

    $('inputInfo').textContent = `${shape.join(' × ')} · float32`;

    log(`Loaded ${info.name}.`, {
        input: normalizeDetails(inputDetails),

        output: normalizeDetails(outputDetails),
    });
}

/* =========================================================
   MODEL ROUTER
========================================================= */

async function loadSelectedModel() {
    const key = $('modelSelect').value;

    if (key === 'edge') {
        await loadEdgeImpulseModel();
    } else {
        await loadLiteRtModel(key);
    }
}

/* =========================================================
   CSV
========================================================= */

function detectDelimiter(line) {
    const commas = (line.match(/,/g) || []).length;

    const semicolons = (line.match(/;/g) || []).length;

    return semicolons > commas ? ';' : ',';
}

function parseLabel(value) {
    const label = String(value).trim().toLowerCase();

    if (label === '0' || label === 'normal' || label === 'n') {
        return 0;
    }

    if (
        label === '1' ||
        label === 'arrhythmia' ||
        label === 'abnormal' ||
        label === 'a'
    ) {
        return 1;
    }

    return null;
}

function parseDatasetCSV(text) {
    const lines = text
        .replace(/\r/g, '')
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);

    if (lines.length < 2) {
        throw new Error('CSV contains no data.');
    }

    const delimiter = detectDelimiter(lines[0]);

    const header = lines[0].split(delimiter).map((item) => item.trim());

    const lower = header.map((item) => item.toLowerCase());

    const labelIndex = lower.indexOf('label');

    if (labelIndex === -1) {
        throw new Error('CSV must contain label column.');
    }

    const idIndex = lower.indexOf('sample_id');

    const featureIndexes = [];

    for (let i = 0; i < header.length; i++) {
        if (/^x\d+$/i.test(header[i])) {
            featureIndexes.push(i);
        }
    }

    featureIndexes.sort(
        (a, b) =>
            Number(header[a].replace(/\D/g, '')) -
            Number(header[b].replace(/\D/g, '')),
    );

    if (featureIndexes.length !== 360) {
        throw new Error(
            `Expected 360 ECG values, found ${featureIndexes.length}.`,
        );
    }

    const samples = [];

    for (let row = 1; row < lines.length; row++) {
        const columns = lines[row]
            .split(delimiter)
            .map((value) => value.trim());

        const label = parseLabel(columns[labelIndex]);

        if (label === null) {
            continue;
        }

        const signal = featureIndexes.map((index) => Number(columns[index]));

        if (signal.some((value) => !Number.isFinite(value))) {
            continue;
        }

        samples.push({
            sampleId: idIndex >= 0 ? columns[idIndex] : String(samples.length),

            label,

            signal,
        });
    }

    if (samples.length === 0) {
        throw new Error('No valid ECG samples.');
    }

    return {
        samples,

        normal: samples.filter((item) => item.label === 0).length,

        arrhythmia: samples.filter((item) => item.label === 1).length,
    };
}

/* =========================================================
   FILE INPUT
========================================================= */

$('fileInput').addEventListener('change', async (event) => {
    try {
        const file = event.target.files?.[0];

        if (!file) {
            return;
        }

        dataset = parseDatasetCSV(await file.text());

        $('fileStatus').textContent =
            `${file.name} · ` +
            `${dataset.samples.length} ECG segments · ` +
            `${dataset.normal} normal / ` +
            `${dataset.arrhythmia} arrhythmia`;

        $('sampleCount').textContent = `${dataset.samples.length} segments`;

        $('totalSegments').textContent = dataset.samples.length;

        drawSignal(flattenSignals(dataset.samples));

        resetResults();

        $('totalSegments').textContent = dataset.samples.length;

        updateAnalyzeButton();

        log(`Loaded ECG dataset ${file.name}.`, {
            samples: dataset.samples.length,

            normal: dataset.normal,

            arrhythmia: dataset.arrhythmia,

            segmentLength: 360,
        });
    } catch (error) {
        dataset = null;

        log('Dataset loading failed: ' + error.message);

        updateAnalyzeButton();
    }
});

/* =========================================================
   GRAPH
========================================================= */

function flattenSignals(samples) {
    const values = [];

    for (const sample of samples) {
        values.push(...sample.signal);
    }

    return values;
}

function downsample(values, maxPoints = 6000) {
    if (values.length <= maxPoints) {
        return values;
    }

    const output = [];

    const step = values.length / maxPoints;

    for (let i = 0; i < maxPoints; i++) {
        output.push(values[Math.floor(i * step)]);
    }

    return output;
}

function drawSignal(values) {
    const canvas = $('ecgCanvas');

    const context = canvas.getContext('2d');

    const width = canvas.width;

    const height = canvas.height;

    context.clearRect(0, 0, width, height);

    context.strokeStyle = '#e8e8e3';

    context.lineWidth = 1;

    for (let i = 1; i < 10; i++) {
        const x = (width * i) / 10;

        context.beginPath();

        context.moveTo(x, 0);

        context.lineTo(x, height);

        context.stroke();
    }

    for (let i = 1; i < 6; i++) {
        const y = (height * i) / 6;

        context.beginPath();

        context.moveTo(0, y);

        context.lineTo(width, y);

        context.stroke();
    }

    if (!values?.length) {
        return;
    }

    const data = downsample(values);

    const min = Math.min(...data);

    const max = Math.max(...data);

    const span = max - min || 1;

    context.strokeStyle = '#111';

    context.lineWidth = 1.5;

    context.beginPath();

    data.forEach((value, index) => {
        const x = (index / (data.length - 1)) * width;

        const y = height - (((value - min) / span) * (height - 32) + 16);

        if (index === 0) {
            context.moveTo(x, y);
        } else {
            context.lineTo(x, y);
        }
    });

    context.stroke();
}

/* =========================================================
   COLAB / JUPYTER INFERENCE
========================================================= */

async function inferLiteRt(signal) {
    const input = new Tensor(new Float32Array(signal), [1, 360, 1]);

    const start = performance.now();

    const outputs = await compiledModel.run(input);

    const elapsed = performance.now() - start;

    const resultTensor = await outputs[0].moveTo('wasm');

    const output = resultTensor.toTypedArray();

    const probability = Number(output[0]);

    input.delete?.();

    outputs.delete?.();

    resultTensor.delete?.();

    if (!Number.isFinite(probability)) {
        throw new Error('Invalid LiteRT prediction.');
    }

    return {
        probability: Math.max(0, Math.min(1, probability)),

        elapsed,
    };
}

/* =========================================================
   EDGE IMPULSE INFERENCE
========================================================= */

async function inferEdgeImpulse(signal) {
    const response = await callEdgeWorker('classify', {
        signal,
    });

    const result = response.result;

    let probability = null;

    const arrhythmia = result.results.find((item) =>
        String(item.label).toLowerCase().includes('arrhythmia'),
    );

    if (arrhythmia) {
        probability = Number(arrhythmia.value);
    }

    if (probability === null) {
        const normal = result.results.find((item) =>
            String(item.label).toLowerCase().includes('normal'),
        );

        if (normal) {
            probability = 1 - Number(normal.value);
        }
    }

    if (!Number.isFinite(probability)) {
        throw new Error(
            'Cannot determine Edge Impulse arrhythmia probability.',
        );
    }

    return {
        probability: Math.max(0, Math.min(1, probability)),

        elapsed: response.inferenceMs,

        rawResult: result,
    };
}

/* =========================================================
   INFERENCE ROUTER
========================================================= */

async function inferSample(signal) {
    const key = $('modelSelect').value;

    return key === 'edge' ? inferEdgeImpulse(signal) : inferLiteRt(signal);
}

/* =========================================================
   METRICS
========================================================= */

function calculateMetrics(results) {
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;

    let normalPredicted = 0;
    let arrhythmiaPredicted = 0;

    let confidenceTotal = 0;

    for (const item of results) {
        const predicted = item.probability >= 0.5 ? 1 : 0;

        const confidence =
            predicted === 1 ? item.probability : 1 - item.probability;

        confidenceTotal += confidence;

        if (predicted === 0) {
            normalPredicted++;
        } else {
            arrhythmiaPredicted++;
        }

        if (item.label === 1 && predicted === 1) {
            tp++;
        } else if (item.label === 0 && predicted === 0) {
            tn++;
        } else if (item.label === 0 && predicted === 1) {
            fp++;
        } else {
            fn++;
        }
    }

    const total = results.length;

    const accuracy = (tp + tn) / total;

    const precision = tp + fp ? tp / (tp + fp) : 0;

    const recall = tp + fn ? tp / (tp + fn) : 0;

    const f1 =
        precision + recall
            ? (2 * precision * recall) / (precision + recall)
            : 0;

    return {
        tp,
        tn,
        fp,
        fn,

        accuracy,
        precision,
        recall,
        f1,

        normalPredicted,
        arrhythmiaPredicted,

        avgConfidence: confidenceTotal / total,
    };
}

/* =========================================================
   UI
========================================================= */

function resetResults() {
    const ids = [
        'normalPredicted',
        'arrhythmiaPredicted',
        'avgConfidence',
        'accuracy',
        'precision',
        'recall',
        'f1',
        'avgInference',
        'minInference',
        'maxInference',
        'totalInference',
        'modelSize',
        'inputInfo',
        'tp',
        'tn',
        'fp',
        'fn',
    ];

    $('finalResult').textContent = '—';

    $('finalDescription').textContent = 'Ready for analysis.';

    for (const id of ids) {
        $(id).textContent = '—';
    }
}

function updateAnalyzeButton() {
    const key = $('modelSelect').value;

    const runtimeAvailable = key === 'edge' ? true : liteRtReady;

    $('analyzeBtn').disabled = !dataset || !runtimeAvailable;
}

function setProgress(current, total) {
    $('progressText').textContent = `${current} / ${total}`;

    $('progressBar').style.width = `${(current / total) * 100}%`;
}

/* =========================================================
   ANALYZE
========================================================= */

async function analyzeDataset() {
    if (!dataset) {
        throw new Error('Load ECG dataset first.');
    }

    const key = $('modelSelect').value;

    const model = MODELS[key];

    await loadSelectedModel();

    $('analyzeBtn').disabled = true;

    $('analyzeBtn').textContent = 'Analyzing…';

    $('progressBlock').classList.remove('hidden');

    setProgress(0, dataset.samples.length);

    /* Warm-up */

    for (let i = 0; i < Math.min(5, dataset.samples.length); i++) {
        await inferSample(dataset.samples[i].signal);
    }

    const results = [];

    const completeStart = performance.now();

    for (let i = 0; i < dataset.samples.length; i++) {
        const sample = dataset.samples[i];

        const inference = await inferSample(sample.signal);

        results.push({
            sampleId: sample.sampleId,

            label: sample.label,

            probability: inference.probability,

            elapsed: inference.elapsed,
        });

        if (i < 5) {
            log(
                `Sample ${i}: ` +
                    `label=${sample.label}, ` +
                    `probability=${inference.probability}`,
            );

            if (key === 'edge') {
                log(`Edge Impulse raw result ${i}:`, inference.rawResult);
            }
        }

        setProgress(i + 1, dataset.samples.length);

        if (i % 10 === 0) {
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
    }

    const completeTime = performance.now() - completeStart;

    const metrics = calculateMetrics(results);

    const times = results.map((result) => result.elapsed);

    const totalInference = times.reduce((a, b) => a + b, 0);

    const averageInference = totalInference / times.length;

    const minimumInference = Math.min(...times);

    const maximumInference = Math.max(...times);

    const arrhythmiaRate = metrics.arrhythmiaPredicted / results.length;

    $('finalResult').textContent = 'Dataset analysis completed';

    $('finalDescription').textContent =
        `${metrics.arrhythmiaPredicted} of ${results.length} ECG segments ` +
        `were classified as arrhythmia ` +
        `(${(arrhythmiaRate * 100).toFixed(1)}%), ` +
        `${metrics.normalPredicted} as normal.`;

    $('totalSegments').textContent = results.length;

    $('normalPredicted').textContent = metrics.normalPredicted;

    $('arrhythmiaPredicted').textContent = metrics.arrhythmiaPredicted;

    $('avgConfidence').textContent = `${(metrics.avgConfidence * 100).toFixed(
        2,
    )}%`;

    $('accuracy').textContent = `${(metrics.accuracy * 100).toFixed(2)}%`;

    $('precision').textContent = `${(metrics.precision * 100).toFixed(2)}%`;

    $('recall').textContent = `${(metrics.recall * 100).toFixed(2)}%`;

    $('f1').textContent = `${(metrics.f1 * 100).toFixed(2)}%`;

    $('avgInference').textContent = `${averageInference.toFixed(3)} ms`;

    $('minInference').textContent = `${minimumInference.toFixed(3)} ms`;

    $('maxInference').textContent = `${maximumInference.toFixed(3)} ms`;

    $('totalInference').textContent = `${totalInference.toFixed(3)} ms`;

    $('tn').textContent = metrics.tn;

    $('fp').textContent = metrics.fp;

    $('fn').textContent = metrics.fn;

    $('tp').textContent = metrics.tp;

    log(`Analysis completed with ${model.name}.`, {
        runtime: model.runtime,

        validation: {
            accuracy: metrics.accuracy,

            precision: metrics.precision,

            recall: metrics.recall,

            f1: metrics.f1,

            averageConfidence: metrics.avgConfidence,

            confusionMatrix: {
                TN: metrics.tn,

                FP: metrics.fp,

                FN: metrics.fn,

                TP: metrics.tp,
            },
        },

        performance: {
            averageInferenceMs: averageInference,

            minInferenceMs: minimumInference,

            maxInferenceMs: maximumInference,

            inferenceTotalMs: totalInference,

            completeAnalysisMs: completeTime,
        },
    });

    $('analyzeBtn').disabled = false;

    $('analyzeBtn').textContent = 'Analyze ECG';
}

/* =========================================================
   EVENTS
========================================================= */

$('analyzeBtn').addEventListener('click', async () => {
    try {
        await analyzeDataset();
    } catch (error) {
        log('Analysis failed: ' + (error?.message || String(error)));

        $('analyzeBtn').textContent = 'Analyze ECG';

        updateAnalyzeButton();
    }
});

$('modelSelect').addEventListener('change', () => {
    resetResults();

    updateRuntimeBadge();

    updateAnalyzeButton();
});

/* =========================================================
   START
========================================================= */

detectEnvironment();

drawSignal([]);

initEdgeWorker();

initLiteRt();

updateRuntimeBadge();
