// our script loads before core.js creates window.wasmReady,
// so we wait for DOMContentLoaded (which core.js also uses), then await the promise
window.addEventListener('DOMContentLoaded', async () => {
await window.wasmReady;
const { GpuPipeline, CpuPipeline } = wasm_bindgen;
const gpu = new GpuPipeline('gl');
const cpu = new CpuPipeline();
let useGpu = true;
let fileBytes = null;

// init three.js viz
window.initViz();

function showResult(hex, ms) {
    // hex is #rrggbbaa
    const rgb = hex.slice(0, 7);
    document.getElementById('pixel-rgb').style.background = rgb;
    document.getElementById('pixel-rgba').style.background = hex;
    document.getElementById('label-rgb').textContent = rgb;
    document.getElementById('label-rgba').textContent = hex;
    document.getElementById('time').textContent = ms + 'ms';
}
function clearResult() {
    document.getElementById('pixel-rgb').style.background = 'transparent';
    document.getElementById('pixel-rgba').style.background = 'transparent';
    document.getElementById('label-rgb').textContent = '';
    document.getElementById('label-rgba').textContent = '';
    document.getElementById('time').textContent = '';
}
function computeText(text) {
    const t0 = performance.now();
    const hex = useGpu ? gpu.compute(text) : cpu.compute(text);
    showResult(hex, (performance.now() - t0).toFixed(3));
    // viz uses CPU activations (GPU path doesn't expose them)
    window.updateViz(cpu.activations(text));
}
function computeBytes(bytes) {
    const t0 = performance.now();
    const hex = useGpu ? gpu.compute_bytes(bytes) : cpu.compute_bytes(bytes);
    showResult(hex, (performance.now() - t0).toFixed(3));
    window.updateViz(cpu.activations_bytes(bytes));
}

const btn = document.getElementById('toggle');
btn.addEventListener('click', () => {
    useGpu = !useGpu;
    btn.textContent = useGpu ? 'GPU' : 'CPU';
    if (fileBytes) {
        computeBytes(fileBytes);
    } else {
        const text = document.getElementById('input').value;
        if (text) computeText(text);
    }
});

document.getElementById('input').addEventListener('input', e => {
    const text = e.target.value;
    if (!text) { clearResult(); return; }
    fileBytes = null;
    document.getElementById('file').value = '';
    document.getElementById('file-name').textContent = '';
    computeText(text);
});

document.getElementById('file').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) { clearResult(); return; }
    document.getElementById('input').value = '';
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-error').style.display = 'none';
    document.getElementById('spinner').style.display = 'block';
    clearResult();
    try {
        fileBytes = new Uint8Array(await file.arrayBuffer());
        computeBytes(fileBytes);
    } catch (err) {
        fileBytes = null;
        document.getElementById('file-error').textContent = 'file too large to load (' + (file.size / 1e9).toFixed(1) + ' GB)';
        document.getElementById('file-error').style.display = 'block';
    } finally {
        document.getElementById('spinner').style.display = 'none';
    }
});
}); // DOMContentLoaded
