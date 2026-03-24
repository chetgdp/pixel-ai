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

const btnCpu = document.getElementById('toggle-cpu');
const btnWebgl2 = document.getElementById('toggle-webgl2');
function setPipeline(gpu) {
    useGpu = gpu;
    btnCpu.classList.toggle('active', !gpu);
    btnWebgl2.classList.toggle('active', gpu);
    if (fileBytes) {
        computeBytes(fileBytes);
    } else {
        const text = document.getElementById('input').value;
        if (text) computeText(text);
    }
}
btnCpu.addEventListener('click', () => setPipeline(false));
btnWebgl2.addEventListener('click', () => setPipeline(true));

// training: like reinforces current output, dislike opens inline color picker
const stepsEl = document.getElementById('train-steps');
const pickerPanel = document.getElementById('picker-panel');
const svCanvas = document.getElementById('picker-sv');
const hueCanvas = document.getElementById('picker-hue');
const pickerPreview = document.getElementById('picker-preview');
const pickerHexEl = document.getElementById('picker-hex');
const svCtx = svCanvas.getContext('2d');
const hueCtx = hueCanvas.getContext('2d');
let pickerHue = 0;
let pickedColor = '#ff0000';

function drawHueBar() {
    const w = hueCanvas.width, h = hueCanvas.height;
    for (let x = 0; x < w; x++) {
        hueCtx.fillStyle = 'hsl(' + (x / w * 360) + ',100%,50%)';
        hueCtx.fillRect(x, 0, 1, h);
    }
}

function drawSV() {
    const w = svCanvas.width, h = svCanvas.height;
    svCtx.fillStyle = 'hsl(' + pickerHue + ',100%,50%)';
    svCtx.fillRect(0, 0, w, h);
    const white = svCtx.createLinearGradient(0, 0, w, 0);
    white.addColorStop(0, 'rgba(255,255,255,1)');
    white.addColorStop(1, 'rgba(255,255,255,0)');
    svCtx.fillStyle = white;
    svCtx.fillRect(0, 0, w, h);
    const black = svCtx.createLinearGradient(0, 0, 0, h);
    black.addColorStop(0, 'rgba(0,0,0,0)');
    black.addColorStop(1, 'rgba(0,0,0,1)');
    svCtx.fillStyle = black;
    svCtx.fillRect(0, 0, w, h);
}

function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(v => Math.round(v).toString(16).padStart(2, '0')).join('');
}

// convert HSV to RGB (h in [0,360], s,v in [0,1])
function hsvToRgb(h, s, v) {
    const c = v * s, x = c * (1 - Math.abs((h / 60) % 2 - 1)), m = v - c;
    let r, g, b;
    if (h < 60)       { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else              { r = c; g = 0; b = x; }
    return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

let pickerS = 1, pickerV = 1;

function updatePickerPreview(hex) {
    pickedColor = hex;
    pickerPreview.style.background = hex;
    pickerHexEl.value = hex;
}

function pickFromSV(e) {
    const rect = svCanvas.getBoundingClientRect();
    pickerS = Math.max(0, Math.min((e.clientX - rect.left) / rect.width, 1));
    pickerV = Math.max(0, Math.min(1 - (e.clientY - rect.top) / rect.height, 1));
    const [r, g, b] = hsvToRgb(pickerHue, pickerS, pickerV);
    updatePickerPreview(rgbToHex(r, g, b));
}

function pickFromHue(e) {
    const rect = hueCanvas.getBoundingClientRect();
    const x = Math.max(0, Math.min((e.clientX - rect.left) / rect.width, 1));
    pickerHue = x * 360;
    drawSV();
    const [r, g, b] = hsvToRgb(pickerHue, pickerS, pickerV);
    updatePickerPreview(rgbToHex(r, g, b));
}

let svDragging = false;
svCanvas.addEventListener('mousedown', e => { svDragging = true; pickFromSV(e); });
svCanvas.addEventListener('mousemove', e => { if (svDragging) pickFromSV(e); });
window.addEventListener('mouseup', () => { svDragging = false; });

let hueDragging = false;
hueCanvas.addEventListener('mousedown', e => { hueDragging = true; pickFromHue(e); });
hueCanvas.addEventListener('mousemove', e => { if (hueDragging) pickFromHue(e); });
window.addEventListener('mouseup', () => { hueDragging = false; });

function getCurrentInput() {
    if (fileBytes) return { type: 'bytes', data: fileBytes };
    const text = document.getElementById('input').value;
    if (text) return { type: 'text', data: text };
    return null;
}

function doTrain(targetHex) {
    const inp = getCurrentInput();
    if (!inp) return;
    const t0 = performance.now();
    const hex = inp.type === 'bytes'
        ? cpu.train_bytes(inp.data, targetHex)
        : cpu.train(inp.data, targetHex);
    const ms = (performance.now() - t0).toFixed(1);
    showResult(hex, ms);
    stepsEl.textContent = 'step ' + cpu.training_steps() + ' (' + ms + 'ms)';
    if (inp.type === 'bytes') {
        window.updateViz(cpu.activations_bytes(inp.data));
    } else {
        window.updateViz(cpu.activations(inp.data));
    }
}

document.getElementById('btn-like').addEventListener('click', () => {
    const hex = document.getElementById('label-rgba').textContent;
    if (hex) doTrain(hex);
});

document.getElementById('btn-dislike').addEventListener('click', () => {
    const visible = pickerPanel.style.display !== 'none';
    pickerPanel.style.display = visible ? 'none' : 'flex';
    if (!visible) {
        pickerHue = 0; pickerS = 1; pickerV = 1;
        drawHueBar(); drawSV();
        updatePickerPreview('#ff0000');
    }
});

pickerHexEl.addEventListener('input', e => {
    const hex = e.target.value;
    if (/^#[0-9a-fA-F]{6}$/.test(hex)) {
        pickedColor = hex;
        pickerPreview.style.background = hex;
    }
});

document.getElementById('picker-train').addEventListener('click', () => {
    doTrain(pickedColor + 'ff');
    pickerPanel.style.display = 'none';
});

const sidebarToggle = document.getElementById('sidebar-toggle');
const ui = document.getElementById('ui');
sidebarToggle.addEventListener('click', () => {
    ui.classList.toggle('collapsed');
    sidebarToggle.innerHTML = ui.classList.contains('collapsed') ? '&rsaquo;' : '&lsaquo;';
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
