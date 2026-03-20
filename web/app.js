// our script loads before core.js creates window.wasmReady,
// so we wait for DOMContentLoaded (which core.js also uses), then await the promise
window.addEventListener('DOMContentLoaded', async () => {
await window.wasmReady;
const { GpuPipeline, CpuPipeline } = wasm_bindgen;
const gpu = new GpuPipeline('gl');
const cpu = new CpuPipeline();
let useGpu = true;
let fileBytes = null;

function showResult(hex, ms) {
    document.getElementById('pixel').style.background = hex;
    document.getElementById('label').textContent = hex;
    document.getElementById('time').textContent = ms + 'ms';
}
function clearResult() {
    document.getElementById('pixel').style.background = 'transparent';
    document.getElementById('label').textContent = '';
    document.getElementById('time').textContent = '';
}
function computeBytes(bytes) {
    const t0 = performance.now();
    const hex = useGpu ? gpu.compute_bytes(bytes) : cpu.compute_bytes(bytes);
    showResult(hex, (performance.now() - t0).toFixed(3));
}

const btn = document.getElementById('toggle');
btn.addEventListener('click', () => {
    useGpu = !useGpu;
    btn.textContent = useGpu ? 'GPU' : 'CPU';
    if (fileBytes) {
        computeBytes(fileBytes);
    } else {
        document.getElementById('input').dispatchEvent(new Event('input'));
    }
});

document.getElementById('input').addEventListener('input', e => {
    const text = e.target.value;
    if (!text) { clearResult(); return; }
    fileBytes = null;
    document.getElementById('file').value = '';
    document.getElementById('file-name').textContent = '';
    const t0 = performance.now();
    const hex = useGpu ? gpu.compute(text) : cpu.compute(text);
    showResult(hex, (performance.now() - t0).toFixed(3));
});

document.getElementById('file').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) { clearResult(); return; }
    document.getElementById('input').value = '';
    document.getElementById('file-name').textContent = file.name;
    fileBytes = new Uint8Array(await file.arrayBuffer());
    computeBytes(fileBytes);
});
}); // DOMContentLoaded
