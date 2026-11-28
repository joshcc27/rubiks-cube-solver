// Rendering and input for the cube page. The cube's state lives in
// cube-model.js; this file draws it and turns user input into moves.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  BASE_MOVES, FACELETS, parseMove, invertMove, solvedState, applyMove, isSolved, generateScramble, relabelByCentres,
} from './cube-model.js';

const COLORS = {
  U: 0xffffff, // white
  D: 0xffff00, // yellow
  F: 0xff4500, // red
  B: 0xff8c00, // orange
  R: 0x0077cc, // blue
  L: 0x00aa44, // green
};
const INSIDE_COLOR = 0x1a1a2e;

const GAP = 0.04;
const SIZE = 1;
const STEP = SIZE + GAP;
const TURN_MS = 180;
const SCRAMBLE_TURN_MS = 80;
const DRAG_THRESHOLD_PX = 10;
const AXES = ['x', 'y', 'z'];
const AXIS_VECTORS = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) };

// Scene
const container = document.getElementById('canvas-container');
const width = () => container.clientWidth;
const height = () => container.clientHeight;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(width(), height());
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, width() / height(), 0.1, 100);
camera.position.set(5, 4, 6);

scene.add(new THREE.AmbientLight(0xffffff, 1.3));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
dirLight.position.set(8, 12, 10);
scene.add(dirLight);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;
orbit.minDistance = 4;
orbit.maxDistance = 18;

window.addEventListener('resize', () => {
  camera.aspect = width() / height();
  camera.updateProjectionMatrix();
  renderer.setSize(width(), height());
});

// Cubies
// 27 meshes that never change position. A turn is animated by rotating a
// layer, then the meshes snap back and are repainted from the model. One
// geometry and seven materials are shared by every mesh, so nothing has to be
// created or disposed after startup.
const geometry = new THREE.BoxGeometry(SIZE, SIZE, SIZE);
const insideMaterial = new THREE.MeshLambertMaterial({ color: INSIDE_COLOR });
const faceMaterials = Object.fromEntries(
  Object.entries(COLORS).map(([face, color]) => [face, new THREE.MeshLambertMaterial({ color })]),
);

const cubeGroup = new THREE.Group();
scene.add(cubeGroup);

const cubies = [];
const cubieAt = new Map();
for (let x = -1; x <= 1; x++) {
  for (let y = -1; y <= 1; y++) {
    for (let z = -1; z <= 1; z++) {
      const mesh = new THREE.Mesh(geometry, Array(6).fill(insideMaterial));
      mesh.position.set(x * STEP, y * STEP, z * STEP);
      mesh.userData.pos = [x, y, z];
      cubeGroup.add(mesh);
      cubies.push(mesh);
      cubieAt.set(`${x},${y},${z}`, mesh);
    }
  }
}

// BoxGeometry material slots: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z.
function slotForNormal([nx, ny, nz]) {
  if (nx) return nx > 0 ? 0 : 1;
  if (ny) return ny > 0 ? 2 : 3;
  return nz > 0 ? 4 : 5;
}
const stickerSlots = FACELETS.map(f => ({ mesh: cubieAt.get(f.pos.join(',')), slot: slotForNormal(f.normal) }));

function paint() {
  stickerSlots.forEach(({ mesh, slot }, i) => { mesh.material[slot] = faceMaterials[state[i]]; });
}

// State
let state = solvedState(); // 54 facelets, see cube-model.js
let scramble = [];         // the scramble applied after the last reset
let history = [];          // moves made since the scramble, used by Undo
let queue = [];            // pending { move, kind }; kind is 'user', 'undo', 'scramble' or 'solve'
let current = null;        // the turn being animated

const isBusy = () => current !== null || queue.length > 0;

function enqueue(move, kind = 'user') {
  if (!parseMove(move)) return;
  forgetSolve();
  queue.push({ move, kind });
  updateUI();
}

function startTurn({ move, kind }) {
  const m = parseMove(move);
  const axisIndex = AXES.indexOf(m.axis);
  const pivot = new THREE.Group();
  cubeGroup.add(pivot);
  const layer = cubies.filter(c => c.userData.pos[axisIndex] === m.layer);
  layer.forEach(c => pivot.add(c));
  current = {
    move, kind, pivot, layer,
    axis: AXIS_VECTORS[m.axis],
    angle: (Math.PI / 2) * m.dir * m.turns,
    duration: kind === 'scramble' ? SCRAMBLE_TURN_MS : TURN_MS * (Math.abs(m.turns) === 2 ? 1.5 : 1),
    elapsed: 0,
  };
}

// Put the animated layer back in place without changing the model.
function releaseLayer() {
  if (!current) return;
  current.layer.forEach(c => cubeGroup.add(c));
  cubeGroup.remove(current.pivot);
  current = null;
}

function finishTurn() {
  const { move, kind } = current;
  releaseLayer();
  state = applyMove(state, move);
  if (kind === 'user' || kind === 'solve') history.push(move);
  else if (kind === 'undo') history.pop();
  paint();
  updateUI();
}

function reset() {
  forgetSolve();
  queue = [];
  releaseLayer();
  state = solvedState();
  scramble = [];
  history = [];
  paint();
  updateUI();
}

function startScramble() {
  if (isBusy()) return;
  reset();
  scramble = generateScramble(20);
  scramble.forEach(mv => queue.push({ move: mv, kind: 'scramble' }));
  updateUI();
}

function undo() {
  if (isBusy() || history.length === 0) return;
  enqueue(invertMove(history[history.length - 1]), 'undo');
}

// Solver
// The network and its search run in a Web Worker (solver-worker.js), created
// on the first Solve. Any change to the cube while it searches cancels the
// request, so a solution always belongs to the cube on screen.
let solver = null;
let solveRequest = null;   // { id, state } while the worker is searching
let lastSolveId = 0;
let solution = null;       // { length, seconds } of the solution being played or just played
let solverMessage = '';    // progress, or why the last request found nothing

function getSolver() {
  if (!solver) {
    solver = new Worker(new URL('./solver-worker.js', import.meta.url), { type: 'module' });
    solver.onmessage = ({ data }) => onSolverMessage(data);
    // Fires if the worker script itself fails to load, for example offline.
    solver.onerror = event => {
      event.preventDefault();
      solver?.terminate();
      solver = null;
      solveRequest = null;
      solverMessage = "The solver couldn't start. It needs network access the first time.";
      updateUI();
    };
  }
  return solver;
}

function solve() {
  if (isBusy() || solveRequest || isSolved(state)) return;
  solveRequest = { id: ++lastSolveId, state: state.join('') };
  solverMessage = 'Starting the solver…';
  // The network only knows the home colour layout, see relabelByCentres.
  getSolver().postMessage({ type: 'solve', id: solveRequest.id, state: relabelByCentres(state).join('') });
  updateUI();
}

// Called whenever the cube changes other than by the solution itself: stops
// any search in progress and clears the solver's status.
function forgetSolve() {
  if (solveRequest) solver?.postMessage({ type: 'cancel' });
  solveRequest = null;
  solution = null;
  solverMessage = '';
}

function onSolverMessage(msg) {
  if (!solveRequest || msg.id !== solveRequest.id) return;
  if (msg.type === 'progress') {
    solverMessage = msg.text;
  } else {
    const request = solveRequest;
    solveRequest = null;
    solverMessage = '';
    if (msg.type === 'error') {
      solverMessage = `The solver failed: ${msg.message}`;
    } else if (!msg.moves) {
      solverMessage = `No solution found after ${msg.seconds.toFixed(1)} s of searching.`;
    } else if (request.state === state.join('') && !isBusy()) {
      solution = { length: msg.moves.length, seconds: msg.seconds };
      msg.moves.forEach(move => queue.push({ move, kind: 'solve' }));
    }
  }
  updateUI();
}

// UI
const statusEl = document.getElementById('status');
const scrambleEl = document.getElementById('scramble-text');
const countEl = document.getElementById('move-count');
const historyEl = document.getElementById('history-text');
const scrambleBtn = document.getElementById('btn-scramble');
const solveBtn = document.getElementById('btn-solve');
const undoBtn = document.getElementById('btn-undo');

const MAX_HISTORY_SHOWN = 40;

function updateUI() {
  const busy = isBusy();
  scrambleBtn.disabled = busy;
  solveBtn.disabled = busy || solveRequest !== null || isSolved(state);
  undoBtn.disabled = busy || history.length === 0;

  scrambleEl.textContent = scramble.length ? scramble.join(' ') : 'none';
  countEl.textContent = history.length;
  const shown = history.slice(-MAX_HISTORY_SHOWN).join(' ');
  historyEl.textContent = history.length === 0 ? 'none'
    : history.length > MAX_HISTORY_SHOWN ? `… ${shown}` : shown;

  const scrambling = current?.kind === 'scramble' || queue.some(q => q.kind === 'scramble');
  const playing = current?.kind === 'solve' || queue.some(q => q.kind === 'solve');
  const solved = !busy && isSolved(state);
  const found = solution && `${solution.length} moves, found in ${solution.seconds.toFixed(1)} s`;
  statusEl.classList.toggle('solved', solved);
  if (scrambling) statusEl.textContent = 'Scrambling…';
  else if (solveRequest) statusEl.textContent = solverMessage;
  else if (playing && solution) statusEl.textContent = `Playing the network's solution: ${found}.`;
  else if (solved && solution) statusEl.textContent = `Solved by the network: ${found}.`;
  else if (solved && scramble.length) statusEl.textContent = `Solved in ${history.length} move${history.length === 1 ? '' : 's'}.`;
  else if (solved) statusEl.textContent = 'Solved.';
  else statusEl.textContent = solverMessage;
}

document.querySelectorAll('.face-btn').forEach(btn => {
  btn.addEventListener('click', () => enqueue(btn.dataset.move));
});
scrambleBtn.addEventListener('click', startScramble);
solveBtn.addEventListener('click', solve);
undoBtn.addEventListener('click', undo);
document.getElementById('btn-reset').addEventListener('click', reset);

// Letters turn clockwise, Shift makes the turn counter-clockwise. The key is
// compared case-insensitively so Caps Lock has no effect.
window.addEventListener('keydown', e => {
  if (e.target.closest?.('input, textarea, select')) return;
  if (e.ctrlKey || e.metaKey) {
    if (e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
    return;
  }
  if (e.altKey || e.repeat || e.key.length !== 1) return;
  const base = e.key.toUpperCase();
  if (BASE_MOVES[base]) enqueue(e.shiftKey ? `${base}'` : base);
});

// Dragging a sticker
// A drag that starts on a sticker turns one layer. The layer is chosen by
// projecting the motion of the touched point under each possible rotation
// into screen space and picking the one that best matches the drag, so it
// works from any camera angle. A drag that starts off the cube orbits.
const raycaster = new THREE.Raycaster();
let drag = null;

function pointerNDC(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  return new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
}

function toScreen(worldPoint) {
  const p = worldPoint.clone().project(camera);
  return new THREE.Vector2((p.x + 1) * width() / 2, (1 - p.y) * height() / 2);
}

function pickSticker(event) {
  raycaster.setFromCamera(pointerNDC(event), camera);
  const hit = raycaster.intersectObjects(cubies, false)[0];
  if (!hit) return null;
  const normal = hit.face.normal.clone().transformDirection(hit.object.matrixWorld);
  const comps = [normal.x, normal.y, normal.z].map(Math.abs);
  const normalAxis = comps.indexOf(Math.max(...comps));
  const normalSign = Math.sign([normal.x, normal.y, normal.z][normalAxis]);
  const pos = hit.object.userData.pos;
  // Only outer faces carry stickers; hits on inner faces seen through the gaps are ignored.
  if (pos[normalAxis] !== normalSign) return null;
  return { point: hit.point.clone(), normalAxis, pos };
}

function moveForDrag(d, dragVec) {
  const dragDir = dragVec.clone().normalize();
  const start = toScreen(d.point);
  let best = null;
  AXES.forEach((axis, axisIndex) => {
    if (axisIndex === d.normalAxis) return;
    // Velocity of the touched point under a positive rotation about this axis.
    const velocity = new THREE.Vector3().crossVectors(AXIS_VECTORS[axis], d.point);
    const screenDir = toScreen(d.point.clone().addScaledVector(velocity, 0.2)).sub(start);
    if (screenDir.lengthSq() < 1e-6) return;
    const score = dragDir.dot(screenDir.normalize());
    if (!best || Math.abs(score) > Math.abs(best.score)) best = { axis, axisIndex, score };
  });
  if (!best) return null;
  const sign = best.score > 0 ? 1 : -1;
  const layer = d.pos[best.axisIndex];
  const base = Object.keys(BASE_MOVES).find(k => BASE_MOVES[k].axis === best.axis && BASE_MOVES[k].layer === layer);
  return BASE_MOVES[base].dir === sign ? base : `${base}'`;
}

// Capture phase, so this runs before OrbitControls sees the event.
renderer.domElement.addEventListener('pointerdown', e => {
  if (e.button !== 0 || drag) return;
  const sticker = pickSticker(e);
  if (!sticker) return;
  orbit.enabled = false;
  drag = { ...sticker, pointerId: e.pointerId, x: e.clientX, y: e.clientY, used: false };
  try { renderer.domElement.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
}, { capture: true });

renderer.domElement.addEventListener('pointermove', e => {
  if (!drag || drag.used || e.pointerId !== drag.pointerId) return;
  const dragVec = new THREE.Vector2(e.clientX - drag.x, e.clientY - drag.y);
  if (dragVec.length() < DRAG_THRESHOLD_PX) return;
  drag.used = true;
  const move = moveForDrag(drag, dragVec);
  if (move) enqueue(move);
});

function endDrag(e) {
  if (!drag || e.pointerId !== drag.pointerId) return;
  drag = null;
  orbit.enabled = true;
}
renderer.domElement.addEventListener('pointerup', endDrag);
renderer.domElement.addEventListener('pointercancel', endDrag);

// Render loop
let lastTime = null;
function animate(time) {
  requestAnimationFrame(animate);
  const dt = lastTime === null ? 0 : time - lastTime;
  lastTime = time;

  if (!current && queue.length) startTurn(queue.shift());
  if (current) {
    current.elapsed += dt;
    const t = Math.min(current.elapsed / current.duration, 1);
    const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    current.pivot.setRotationFromAxisAngle(current.axis, current.angle * eased);
    if (t >= 1) finishTurn();
  }

  orbit.update();
  renderer.render(scene, camera);
}

paint();
updateUI();
requestAnimationFrame(animate);
