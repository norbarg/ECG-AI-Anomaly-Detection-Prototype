importScripts('./edge-impulse-standalone.js', './run-impulse.js');

let classifier = null;

async function initClassifier() {
    if (classifier) {
        return;
    }

    classifier = new EdgeImpulseClassifier();

    await classifier.init();
}

self.onmessage = async (event) => {
    const { id, type, signal } = event.data;

    try {
        if (type === 'init') {
            await initClassifier();

            const project = classifier.getProjectInfo();

            const properties = classifier.getProperties();

            self.postMessage({
                id,
                ok: true,
                project,
                properties,
            });

            return;
        }

        if (type === 'classify') {
            await initClassifier();

            const start = performance.now();

            const result = classifier.classify(signal);

            const inferenceMs = performance.now() - start;

            self.postMessage({
                id,
                ok: true,
                result,
                inferenceMs,
            });

            return;
        }

        throw new Error(`Unknown worker action: ${type}`);
    } catch (error) {
        self.postMessage({
            id,
            ok: false,
            error: error?.message || String(error),
        });
    }
};
