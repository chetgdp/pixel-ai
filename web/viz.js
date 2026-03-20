// three.js concentric ring visualization of the forward pass
// each layer is a ring of nodes, edges connect adjacent layers
// the pixel sits at the center

const LAYER_SIZES = [4096, 2048, 1024, 256, 64, 16, 4];
const LAYER_RADII = [18, 15, 12.5, 9.5, 7, 4.5, 1.5];
const LAYER_DEPTHS = [15, 12.5, 10, 7.5, 5, 2.5, 0.5];

// max edges to draw between adjacent layers (sampled randomly if exceeded)
const MAX_EDGES_PER_LAYER = 2000;

let scene, camera, renderer, controls, stats;
let nodeMeshes = [];   // one Points object per layer
let edgeMeshes = [];   // one LineSegments per layer gap
let pixelMesh;         // center pixel quad
let nodeColors = [];   // Float32Array per layer for updating colors
let lastActivations = null;
let raycaster, mouse, tooltip;
const CHANNEL_NAMES = ['R', 'G', 'B', 'A'];

function initViz() {
    const container = document.getElementById('viz');
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 200);
    camera.position.set(0, 0, 42);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x111111);
    container.appendChild(renderer.domElement);

    controls = new THREE.TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 2.0;
    controls.zoomSpeed = 0.8;
    controls.panSpeed = 0.6;
    controls.noPan = false;
    controls.keys = [];
    controls.minDistance = 5;
    controls.maxDistance = 80;
    controls.dynamicDampingFactor = 0.25;

    stats = [new Stats(), new Stats(), new Stats()];
    for (let i = 0; i < 3; i++) {
        stats[i].showPanel(i);
        stats[i].dom.style.left = 'auto';
        stats[i].dom.style.right = (i * 80) + 'px';
        container.appendChild(stats[i].dom);
    }

    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.5;
    mouse = new THREE.Vector2();

    tooltip = document.createElement('div');
    tooltip.id = 'node-tooltip';
    container.appendChild(tooltip);

    renderer.domElement.addEventListener('mousemove', onMouseMove);

    buildNodes();
    buildEdges();
    buildPixel();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    animate();
}

function buildNodes() {
    for (let l = 0; l < LAYER_SIZES.length; l++) {
        const count = LAYER_SIZES[l];
        const radius = LAYER_RADII[l];
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);

        // stagger nodes in z and radius so dense layers spread into a 3D band
        const zSpread = Math.min(count / 500, 1.5);  // larger layers get more z spread
        const rSpread = radius * 0.06;                // slight radial jitter

        for (let i = 0; i < count; i++) {
            const angle = (i / count) * Math.PI * 2;
            const rOff = ((i % 3) - 1) * rSpread;    // cycle -1, 0, +1
            const zOff = Math.sin(angle * 7) * zSpread;
            positions[i * 3]     = Math.cos(angle) * (radius + rOff);
            positions[i * 3 + 1] = Math.sin(angle) * (radius + rOff);
            positions[i * 3 + 2] = LAYER_DEPTHS[l] + zOff;
            // default dark
            colors[i * 3] = 0.12;
            colors[i * 3 + 1] = 0.12;
            colors[i * 3 + 2] = 0.12;
        }

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // with staggering, nodes have more room — use a larger minimum size
        const circumference = 2 * Math.PI * radius;
        const spacing = circumference / count;
        const size = Math.max(Math.min(spacing * 0.7, 0.8), 0.3);
        const mat = new THREE.PointsMaterial({
            size: size,
            sizeAttenuation: true,
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            depthWrite: false,
        });

        const points = new THREE.Points(geo, mat);
        points.renderOrder = 1;
        scene.add(points);
        nodeMeshes.push(points);
        nodeColors.push(colors);
    }
}

function buildEdges() {
    for (let l = 0; l < LAYER_SIZES.length - 1; l++) {
        const srcCount = LAYER_SIZES[l];
        const dstCount = LAYER_SIZES[l + 1];
        const srcRadius = LAYER_RADII[l];
        const dstRadius = LAYER_RADII[l + 1];
        const totalEdges = srcCount * dstCount;

        let edgePairs = [];

        if (totalEdges <= MAX_EDGES_PER_LAYER) {
            for (let s = 0; s < srcCount; s++) {
                for (let d = 0; d < dstCount; d++) {
                    edgePairs.push([s, d]);
                }
            }
        } else {
            // sample: spread connections evenly across destination nodes
            const perDst = Math.max(1, Math.floor(MAX_EDGES_PER_LAYER / dstCount));
            for (let d = 0; d < dstCount; d++) {
                for (let k = 0; k < perDst; k++) {
                    const s = Math.floor(Math.random() * srcCount);
                    edgePairs.push([s, d]);
                }
            }
            edgePairs = edgePairs.slice(0, MAX_EDGES_PER_LAYER);
        }

        const edgeCount = edgePairs.length;
        const positions = new Float32Array(edgeCount * 6);
        const colors = new Float32Array(edgeCount * 6);

        const srcPositions = nodeMeshes[l].geometry.attributes.position.array;
        const dstPositions = nodeMeshes[l + 1].geometry.attributes.position.array;

        for (let e = 0; e < edgeCount; e++) {
            const [s, d] = edgePairs[e];

            positions[e * 6]     = srcPositions[s * 3];
            positions[e * 6 + 1] = srcPositions[s * 3 + 1];
            positions[e * 6 + 2] = srcPositions[s * 3 + 2];
            positions[e * 6 + 3] = dstPositions[d * 3];
            positions[e * 6 + 4] = dstPositions[d * 3 + 1];
            positions[e * 6 + 5] = dstPositions[d * 3 + 2];

            for (let c = 0; c < 6; c++) colors[e * 6 + c] = 0.12;
        }

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const mat = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.4,
            depthWrite: false,
        });

        const lines = new THREE.LineSegments(geo, mat);
        lines.renderOrder = 0;
        scene.add(lines);
        edgeMeshes.push({ mesh: lines, pairs: edgePairs, colors: colors });
    }
}

function buildPixel() {
    const geo = new THREE.PlaneGeometry(2.0, 2.0);
    const mat = new THREE.MeshBasicMaterial({ color: 0x000000, depthTest: false, transparent: true, opacity: 1.0, side: THREE.DoubleSide });
    pixelMesh = new THREE.Mesh(geo, mat);
    pixelMesh.position.z = 0;
    pixelMesh.renderOrder = 999;
    scene.add(pixelMesh);
}

// activation value (0-255 bias-encoded) to color
function valToColor(v, r, g, b) {
    const t = (v - 128) / 127;
    const abs = Math.abs(t);
    // diverging: orange (#C96B30) ← white (#F5EDE5) → blue (#2978B5)
    if (t < 0) {
        r[0] = 0.96 - 0.17 * abs;
        g[0] = 0.93 - 0.51 * abs;
        b[0] = 0.90 - 0.71 * abs;
    } else {
        r[0] = 0.96 - 0.80 * abs;
        g[0] = 0.93 - 0.46 * abs;
        b[0] = 0.90 - 0.19 * abs;
    }
}

function updateViz(activations) {
    if (!activations || activations.length === 0) return;
    lastActivations = activations;

    const r = [0], g = [0], b = [0];
    let offset = 0;

    for (let l = 0; l < LAYER_SIZES.length; l++) {
        const count = LAYER_SIZES[l];
        const colors = nodeColors[l];

        for (let i = 0; i < count; i++) {
            const v = activations[offset + i];
            if (l === LAYER_SIZES.length - 1) {
                // output layer: each node is an RGBA channel
                const t = v / 255;
                colors[i * 3]     = i === 0 ? t : 0; // R
                colors[i * 3 + 1] = i === 1 ? t : 0; // G
                colors[i * 3 + 2] = i === 2 ? t : 0; // B
                if (i === 3) { colors[i * 3] = t; colors[i * 3 + 1] = t; colors[i * 3 + 2] = t; } // A as white
            } else {
                valToColor(v, r, g, b);
                colors[i * 3]     = r[0];
                colors[i * 3 + 1] = g[0];
                colors[i * 3 + 2] = b[0];
            }
        }

        nodeMeshes[l].geometry.attributes.color.needsUpdate = true;
        offset += count;
    }

    for (let l = 0; l < edgeMeshes.length; l++) {
        const { pairs, colors, mesh } = edgeMeshes[l];
        const srcOffset = layerOffset(l);
        const dstOffset = layerOffset(l + 1);

        for (let e = 0; e < pairs.length; e++) {
            const [s, d] = pairs[e];
            const sv = activations[srcOffset + s];
            const dv = activations[dstOffset + d];

            const srcAbs = Math.abs(sv - 128) / 127;
            const dstAbs = Math.abs(dv - 128) / 127;
            const strength = (srcAbs + dstAbs) * 0.5;

            valToColor(sv, r, g, b);
            colors[e * 6]     = r[0] * strength;
            colors[e * 6 + 1] = g[0] * strength;
            colors[e * 6 + 2] = b[0] * strength;

            valToColor(dv, r, g, b);
            colors[e * 6 + 3] = r[0] * strength;
            colors[e * 6 + 4] = g[0] * strength;
            colors[e * 6 + 5] = b[0] * strength;
        }

        mesh.geometry.attributes.color.needsUpdate = true;
    }

    // parse RGB from the hex string in the label, ignoring alpha
    const hex = document.getElementById('label-rgb').textContent;
    if (hex && hex.startsWith('#') && hex.length >= 7) {
        const r = parseInt(hex.slice(1, 3), 16) / 255;
        const g = parseInt(hex.slice(3, 5), 16) / 255;
        const b = parseInt(hex.slice(5, 7), 16) / 255;
        pixelMesh.material.color.setRGB(r, g, b);
    }
}

function layerOffset(l) {
    let off = 0;
    for (let i = 0; i < l; i++) off += LAYER_SIZES[i];
    return off;
}

function animate() {
    requestAnimationFrame(animate);
    for (let i = 0; i < 3; i++) stats[i].begin();
    controls.update();
    renderer.render(scene, camera);
    for (let i = 0; i < 3; i++) stats[i].end();
}

function onMouseMove(e) {
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

    if (!lastActivations) { tooltip.style.display = 'none'; return; }

    raycaster.setFromCamera(mouse, camera);
    const outputLayer = nodeMeshes[LAYER_SIZES.length - 1];
    const hits = raycaster.intersectObject(outputLayer);

    if (hits.length > 0) {
        const idx = hits[0].index;
        const offset = layerOffset(LAYER_SIZES.length - 1);
        const v = lastActivations[offset + idx];
        const name = CHANNEL_NAMES[idx];
        const hex = '0x' + v.toString(16).toUpperCase().padStart(2, '0');
        const bin = '0b' + v.toString(2).padStart(8, '0');
        tooltip.textContent = name + ': ' + v + ' ' + hex + ' ' + bin;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 12) + 'px';
        tooltip.style.top = (e.clientY - 8) + 'px';
    } else {
        tooltip.style.display = 'none';
    }
}

window.initViz = initViz;
window.updateViz = updateViz;
