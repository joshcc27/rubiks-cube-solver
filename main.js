// Rendering, input and solver controls for the cube page. The cube's state
// and the record of how it got there live in session.js, and the moves
// themselves in cube-model.js. This file draws the cube, turns input into
// moves, plays solutions, and talks to the solver in solver-worker.js.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  BASE_MOVES, FACES, FACELETS, parseMove, isSolved, generateScramble, relabelByCentres, parseSequence,
} from './cube-model.js';
import { CubeSession } from './session.js';

const COLORS = {
  U: 0xffffff, // white
  D: 0xffff00, // yellow
  F: 0xc8102e, // red
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
const SPEEDS = [0.5, 1, 1.5, 2, 3]; // solution playback, as multiples of normal speed
const DEFAULT_SPEED = 1;            // index into SPEEDS
const DRAG_THRESHOLD_PX = 10;
const AXES = ['x', 'y', 'z'];
const AXIS_VECTORS = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) };
const DEFAULT_VIEW = new THREE.Vector3(5, 4, 6).normalize();
const CUBE_RADIUS = Math.sqrt(3) * (STEP + SIZE / 2); // centre to the outermost corner
// The download progress counts models/policy.onnx only. The ONNX Runtime
// engine (about 3 MB more) loads afterwards, during "Starting the network…".
const MODEL_DOWNLOAD = '9 MB';

const $ = id => document.getElementById(id);
const wideQuery = window.matchMedia('(min-width: 900px)'); // matches the layout breakpoint in index.html
const count = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

// Per-viewer preferences (help open, labels, turn buttons, speed). Storage can
// be missing or blocked, so every access is guarded and the defaults work.
function loadPref(key, fallback) {
  try {
    const raw = localStorage.getItem(`cube.${key}`);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}
function savePref(key, value) {
  try { localStorage.setItem(`cube.${key}`, JSON.stringify(value)); } catch { /* not saved */ }
}

// Scene
const stage = $('stage');
const width = () => stage.clientWidth;
const height = () => stage.clientHeight;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(width(), height());
renderer.domElement.setAttribute('role', 'img');
renderer.domElement.setAttribute('aria-label', '3D cube. Turn it with the turn buttons, the keyboard, or by dragging a sticker.');
stage.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, width() / height(), 0.1, 100);

// The camera distance at which the whole cube fits the stage, whichever way
// it is turned, so it never touches the edges of a narrow phone screen. A
// stage with no size yet (a hidden page) counts as square, so the distance
// stays finite until the first real resize.
function fitDistance() {
  const aspect = camera.aspect > 0 && Number.isFinite(camera.aspect) ? camera.aspect : 1;
  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
  return (1.25 * CUBE_RADIUS) / Math.sin(Math.min(vFov, hFov) / 2);
}
camera.position.copy(DEFAULT_VIEW).multiplyScalar(fitDistance());

scene.add(new THREE.AmbientLight(0xffffff, 1.3));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
dirLight.position.set(8, 12, 10);
scene.add(dirLight);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;
orbit.minDistance = CUBE_RADIUS * 1.5;
orbit.maxDistance = fitDistance() * 2.5;

function resetView() {
  // With damping on, OrbitControls keeps the drift from a recent drag and
  // would carry the camera away again. One update without damping uses it up.
  orbit.enableDamping = false;
  orbit.update();
  orbit.target.set(0, 0, 0);
  camera.position.copy(DEFAULT_VIEW).multiplyScalar(fitDistance());
  orbit.update();
  orbit.enableDamping = true;
}

// When the stage changes size, keep the camera's direction and its zoom
// relative to the fitted distance.
let fittedDistance = fitDistance();
new ResizeObserver(() => {
  if (!width() || !height()) return;
  camera.aspect = width() / height();
  camera.updateProjectionMatrix();
  renderer.setSize(width(), height());
  const fit = fitDistance();
  const offset = camera.position.clone().sub(orbit.target).multiplyScalar(fit / fittedDistance);
  camera.position.copy(orbit.target).add(offset);
  orbit.maxDistance = fit * 2.5;
  fittedDistance = fit;
}).observe(stage);

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
  stickerSlots.forEach(({ mesh, slot }, i) => { mesh.material[slot] = faceMaterials[cube.state[i]]; });
}

// Face labels
// A letter on each centre sticker naming its face in the fixed frame the
// moves use. The centre cubies never really move, so a label stays on its
// face even after slice turns carry the centre colours around.
const LABEL_ROTATIONS = {
  U: [-Math.PI / 2, 0, 0],
  D: [Math.PI / 2, 0, 0],
  F: [0, 0, 0],
  B: [0, Math.PI, 0],
  R: [0, Math.PI / 2, 0],
  L: [0, -Math.PI / 2, 0],
};

function labelTexture(letter) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const g = canvas.getContext('2d');
  g.font = '700 86px "Segoe UI", system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.lineJoin = 'round';
  g.lineWidth = 10;
  g.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  g.strokeText(letter, 64, 68);
  g.fillStyle = '#15152a';
  g.fillText(letter, 64, 68);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

const labelGeometry = new THREE.PlaneGeometry(0.6, 0.6);
const labels = FACES.map(face => {
  const { pos, normal } = FACELETS[FACES.indexOf(face) * 9 + 4];
  const material = new THREE.MeshBasicMaterial({ map: labelTexture(face), transparent: true, depthWrite: false });
  const label = new THREE.Mesh(labelGeometry, material);
  label.position.set(...normal.map(n => n * (SIZE / 2 + 0.004)));
  label.rotation.set(...LABEL_ROTATIONS[face]);
  cubieAt.get(pos.join(',')).add(label); // a child, so raycasts on the cubies ignore it
  return label;
});

// Turns
// The cube's state, scramble, history and any solution showing, see session.js.
const cube = new CubeSession();
let queue = [];     // turns waiting to be animated: { move, kind }, kinds as in CubeSession.applyTurn
let current = null; // the turn being animated

const isAnimating = () => current !== null || queue.length > 0;
const isBusy = () => isAnimating() || Boolean(cube.solution?.playing);

// Queue a turn that changes the cube outside the solution player. That puts
// any solution away and stops any search, since neither fits the cube any more.
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
  const double = Math.abs(m.turns) === 2 ? 1.5 : 1;
  let duration = TURN_MS * double;
  if (kind === 'scramble') duration = SCRAMBLE_TURN_MS;
  else if (kind === 'solve' || kind === 'solve-back') duration = (TURN_MS * double) / SPEEDS[speedIndex];
  current = {
    move, kind, pivot, layer, duration,
    axis: AXIS_VECTORS[m.axis],
    angle: (Math.PI / 2) * m.dir * m.turns,
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
  cube.applyTurn(move, kind);
  paint();
  if (queue.length === 0) {
    const solved = isSolved(cube.state);
    if (kind === 'scramble') announce('Scrambled.');
    else if (solved && kind === 'solve' && cube.solution) announce(`Solved by the network in ${count(cube.solution.moves.length, 'move')}.`);
    else if (solved && kind === 'user' && cube.scramble.length) announce(`Solved in ${count(cube.history.length, 'move')}.`);
  }
  updateUI();
}

function reset() {
  forgetSolve();
  queue = [];
  releaseLayer();
  cube.reset();
  paint();
  updateUI();
}

function startScramble(moves = generateScramble(20)) {
  if (isBusy()) return false;
  reset();
  cube.startScramble(moves);
  cube.scramble.forEach(mv => queue.push({ move: mv, kind: 'scramble' }));
  updateUI();
  return true;
}

// With a solution part way played, Undo steps back through it. Otherwise it
// takes back the last move. Like the Undo button, it waits while turns are
// animating or a solution is playing.
function undo() {
  if (isBusy()) return;
  const turn = cube.undoTurn();
  if (!turn) return;
  if (turn.kind === 'solve-back') stepBack();
  else enqueue(turn.move, turn.kind);
}

// Solver
// The network and its search run in a Web Worker (solver-worker.js), created
// on the first Solve. Any change to the cube while it searches cancels the
// request, so a solution always belongs to the cube on screen.
let solver = null;
let solveRequest = null;   // while the worker works: { id, state, phase, loaded, total, depth, wide, searchStarted }
let lastSolveId = 0;
let solverNote = '';       // why the last request found nothing, shown until the cube changes

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
      solverNote = "The solver couldn't start. It needs network access the first time.";
      announce(solverNote);
      updateUI();
    };
  }
  return solver;
}

// The Solve button reads Stop while the worker is searching.
function solveOrStop() {
  if (solveRequest) {
    forgetSolve();
    announce('Search stopped.');
    updateUI();
    return;
  }
  if (isBusy() || isSolved(cube.state)) return;
  cube.dropSolution(); // a paused solution makes way for the new one
  solverNote = '';
  solveRequest = { id: ++lastSolveId, state: cube.state.join(''), phase: 'starting-solver' };
  // The network only knows the home colour layout, see relabelByCentres.
  getSolver().postMessage({ type: 'solve', id: solveRequest.id, state: relabelByCentres(cube.state).join('') });
  announce('Searching for a solution.');
  updateUI();
}

// Stops any search and puts away any solution and the solver's messages.
// Called whenever the cube changes other than through the solution player.
function forgetSolve() {
  if (solveRequest) solver?.postMessage({ type: 'cancel' });
  solveRequest = null;
  solverNote = '';
  cube.dropSolution();
  queue = queue.filter(q => q.kind !== 'solve' && q.kind !== 'solve-back');
}

function onSolverMessage(msg) {
  if (!solveRequest || msg.id !== solveRequest.id) return;
  const request = solveRequest;
  if (msg.type === 'loading') {
    Object.assign(request, { phase: 'loading', loaded: msg.loaded, total: msg.total });
  } else if (msg.type === 'starting') {
    request.phase = 'starting';
  } else if (msg.type === 'searching') {
    if (request.phase !== 'searching' || request.wide !== msg.wide) request.searchStarted = performance.now();
    Object.assign(request, { phase: 'searching', depth: msg.depth, wide: msg.wide });
  } else if (msg.type === 'error') {
    solveRequest = null;
    solverNote = `The solver failed: ${msg.message}`;
    announce(solverNote);
  } else if (msg.type === 'done') {
    solveRequest = null;
    if (!msg.moves) {
      solverNote = `No solution found after ${msg.seconds.toFixed(1)} s of searching.`;
      announce(solverNote);
    } else if (request.state === cube.state.join('') && !isBusy()) {
      showSolution(msg.moves, msg.seconds);
    }
  }
  updateUI();
}

// Solution player
const solutionPanel = $('solution-panel');
const solutionMeta = $('solution-meta');
const solutionList = $('solution-moves');
const startBtn = $('sol-start');
const backBtn = $('sol-back');
const playBtn = $('sol-play');
const fwdBtn = $('sol-fwd');
const endBtn = $('sol-end');
const speedInput = $('sol-speed');
const speedOut = $('sol-speed-out');
let moveButtons = [];

function storedSpeedIndex() {
  const i = Math.round(Number(loadPref('speed', DEFAULT_SPEED)));
  return Number.isFinite(i) ? Math.min(SPEEDS.length - 1, Math.max(0, i)) : DEFAULT_SPEED;
}
let speedIndex = storedSpeedIndex();

function showSolution(moves, seconds) {
  cube.showSolution(moves, seconds);
  solutionMeta.textContent = `${count(moves.length, 'move')} · found in ${seconds.toFixed(1)} s`;
  solutionList.replaceChildren();
  moveButtons = moves.map((move, i) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.textContent = move;
    button.setAttribute('aria-label', `Go to after move ${i + 1}, ${move}`);
    button.addEventListener('click', () => jumpTo(i + 1));
    li.append(button);
    solutionList.append(li);
    return button;
  });
  announce(`Found a ${moves.length}-move solution. Playing it.`);
}

function play() {
  const s = cube.solution;
  if (!s) return;
  if (s.pos === s.moves.length) jumpTo(0);
  s.playing = true;
  updateUI();
}

function pause() {
  if (!cube.solution) return;
  cube.solution.playing = false;
  updateUI();
}

// A step pauses playback. A step asked for while a turn is still animating
// only pauses, so steps never pile up.
function step(turn) {
  if (!cube.solution) return;
  cube.solution.playing = false;
  if (!isAnimating() && turn) queue.push(turn);
  updateUI();
}
const stepForward = () => step(cube.nextSolutionTurn());
const stepBack = () => step(cube.prevSolutionTurn());

// Jump straight to the state after the first k solution moves, without animating.
function jumpTo(k) {
  const s = cube.solution;
  if (!s) return;
  queue = [];
  releaseLayer();
  cube.jumpTo(k);
  paint();
  if (s.pos === s.moves.length && isSolved(cube.state)) announce(`Solved by the network in ${count(s.moves.length, 'move')}.`);
  updateUI();
}

function updateSolutionPanel() {
  const s = cube.solution;
  solutionPanel.hidden = !s;
  if (!s) return;
  const atEnd = s.pos === s.moves.length;
  // Highlight the move being made, or else the last one made.
  const highlight = current?.kind === 'solve' ? s.pos : s.pos - 1;
  moveButtons.forEach((button, i) => {
    button.classList.toggle('done', i < s.pos);
    if (i === highlight) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
  playBtn.classList.toggle('playing', s.playing);
  playBtn.setAttribute('aria-label', s.playing ? 'Pause' : atEnd ? 'Play again from the start' : 'Play');
  startBtn.disabled = s.pos === 0;
  backBtn.disabled = s.pos === 0;
  fwdBtn.disabled = atEnd;
  endBtn.disabled = atEnd;
}

startBtn.addEventListener('click', () => jumpTo(0));
backBtn.addEventListener('click', stepBack);
playBtn.addEventListener('click', () => (cube.solution?.playing ? pause() : play()));
fwdBtn.addEventListener('click', stepForward);
endBtn.addEventListener('click', () => jumpTo(cube.solution ? cube.solution.moves.length : 0));

function showSpeed() {
  speedInput.value = String(speedIndex);
  speedOut.textContent = `${SPEEDS[speedIndex]}×`;
}
speedInput.addEventListener('input', () => {
  speedIndex = Number(speedInput.value);
  savePref('speed', speedIndex);
  showSpeed();
});
showSpeed();

// Status
// The pill over the cube shows what is happening now. Screen readers get
// only the milestones, through the separate announcer, not every update.
const statusEl = $('status');
const announcer = $('announcer');
let shownStatus = '';
let announceTimer = 0;

// A live region only speaks when its text changes, so it is cleared first and
// refilled a moment later. That way a repeat, such as a second "Scrambled.", is read too.
function announce(text) {
  announcer.textContent = '';
  clearTimeout(announceTimer);
  announceTimer = setTimeout(() => { announcer.textContent = text; }, 60);
}

function statusMessage() {
  if (current?.kind === 'scramble' || queue.some(q => q.kind === 'scramble')) return { text: 'Scrambling…' };
  if (solveRequest) {
    const r = solveRequest;
    if (r.phase === 'loading') {
      const percent = r.total ? ` ${Math.min(99, Math.floor((100 * r.loaded) / r.total))}%` : '';
      return { text: `Loading the network (${MODEL_DOWNLOAD}, first time only)…${percent}` };
    }
    if (r.phase === 'starting') return { text: 'Starting the network…' };
    if (r.phase === 'searching') {
      const seconds = ((performance.now() - r.searchStarted) / 1000).toFixed(1);
      return { text: `${r.wide ? 'No luck yet, searching wider' : 'Searching'}… ${count(r.depth, 'move')} deep · ${seconds} s` };
    }
    return { text: 'Starting the solver…' };
  }
  const s = cube.solution;
  if (s) {
    const n = s.moves.length;
    if (s.pos === n && !isAnimating() && isSolved(cube.state)) return { text: `Solved by the network in ${count(n, 'move')}.`, tone: 'solved' };
    if (s.playing || current?.kind === 'solve') return { text: `Playing move ${Math.min(s.pos + 1, n)} of ${n}` };
    if (current?.kind === 'solve-back') {
      return { text: s.pos - 1 === 0 ? 'Stepping back to the start' : `Stepping back to move ${s.pos - 1} of ${n}` };
    }
    return { text: s.pos === 0 ? `Solution ready: ${count(n, 'move')}` : `Paused after move ${s.pos} of ${n}` };
  }
  if (solverNote) return { text: solverNote, tone: 'error' };
  if (!isAnimating() && isSolved(cube.state) && (cube.scramble.length || cube.history.length)) {
    return { text: cube.scramble.length ? `Solved in ${count(cube.history.length, 'move')}.` : 'Solved.', tone: 'solved' };
  }
  return { text: '' };
}

function renderStatus() {
  const { text, tone = '' } = statusMessage();
  const key = `${tone}|${text}`;
  if (key === shownStatus) return;
  shownStatus = key;
  statusEl.textContent = text;
  statusEl.hidden = !text;
  statusEl.classList.toggle('solved', tone === 'solved');
  statusEl.classList.toggle('error', tone === 'error');
}

// UI
const scrambleEl = $('scramble-text');
const countEl = $('move-count');
const historyEl = $('history-text');
const scrambleBtn = $('btn-scramble');
const solveBtn = $('btn-solve');
const undoBtn = $('btn-undo');
const useScrambleBtn = $('btn-use-scramble');

const MAX_HISTORY_SHOWN = 40;

function updateUI() {
  const busy = isBusy();
  scrambleBtn.disabled = busy;
  useScrambleBtn.disabled = busy;
  solveBtn.textContent = solveRequest ? 'Stop' : 'Solve';
  solveBtn.classList.toggle('stop', solveRequest !== null);
  solveBtn.disabled = solveRequest === null && (busy || isSolved(cube.state));
  undoBtn.disabled = busy || cube.undoTurn() === null;

  const { scramble, history } = cube;
  scrambleEl.textContent = scramble.length ? scramble.join(' ') : 'none';
  countEl.textContent = history.length;
  const shown = history.slice(-MAX_HISTORY_SHOWN).join(' ');
  historyEl.textContent = history.length === 0 ? 'none'
    : history.length > MAX_HISTORY_SHOWN ? `… ${shown}` : shown;

  updateSolutionPanel();
  renderStatus();
}

document.querySelectorAll('.face-btn').forEach(btn => {
  btn.addEventListener('click', () => enqueue(btn.dataset.move));
});
scrambleBtn.addEventListener('click', () => startScramble());
solveBtn.addEventListener('click', solveOrStop);
undoBtn.addEventListener('click', undo);
$('btn-reset').addEventListener('click', reset);
$('btn-view').addEventListener('click', resetView);

// Help, labels and turn buttons remember whether they were open or on.
// On wide screens the help sits at the bottom of the side panel and is always
// shown. On narrow ones it is an overlay on the cube, opened with the ? button.
const helpEl = $('help');
const helpBtn = $('btn-help');
function setHelp(open, remember = false) {
  helpEl.hidden = !open;
  helpBtn.setAttribute('aria-expanded', String(open));
  if (remember) savePref('help', open);
}
helpBtn.addEventListener('click', () => setHelp(helpEl.hidden, true));
function placeHelp() {
  if (wideQuery.matches) {
    $('side').append(helpEl);
    helpEl.hidden = false;
  } else {
    stage.append(helpEl);
    setHelp(loadPref('help', false));
  }
}
wideQuery.addEventListener('change', placeHelp);
placeHelp();

const labelsBtn = $('btn-labels');
function setLabels(on) {
  labels.forEach(label => { label.visible = on; });
  labelsBtn.setAttribute('aria-pressed', String(on));
}
labelsBtn.addEventListener('click', () => {
  const on = labelsBtn.getAttribute('aria-pressed') !== 'true';
  setLabels(on);
  savePref('labels', on);
});
setLabels(loadPref('labels', false));

// Opening the turn buttons from code fires a toggle event too. That one is
// skipped, so only the person's own choice is remembered and the default can
// still follow the screen width.
const turnsEl = $('turns');
let skipToggle = loadPref('turns', wideQuery.matches);
turnsEl.open = skipToggle;
turnsEl.addEventListener('toggle', () => {
  if (skipToggle) {
    skipToggle = false;
    return;
  }
  savePref('turns', turnsEl.open);
});

// Typed moves: apply them to the cube as it is, or reset and use them as the scramble.
const movesInput = $('moves-input');
const movesError = $('moves-error');
function typedMoves() {
  try {
    const moves = parseSequence(movesInput.value);
    movesError.textContent = moves.length ? '' : "Type some moves first, for example R U R' U'.";
    return moves.length ? moves : null;
  } catch (err) {
    movesError.textContent = err.message;
    return null;
  }
}
$('moves-form').addEventListener('submit', event => {
  event.preventDefault();
  const moves = typedMoves();
  if (moves) moves.forEach(mv => enqueue(mv));
});
useScrambleBtn.addEventListener('click', () => {
  const moves = typedMoves();
  if (moves && !startScramble(moves)) movesError.textContent = 'Wait until the cube stops turning.';
});

// About. The dialog has no padding of its own, so a click on the dialog
// element itself can only be a click on the backdrop.
const aboutDialog = $('about');
$('btn-about').addEventListener('click', () => aboutDialog.showModal());
aboutDialog.addEventListener('click', event => {
  if (event.target === aboutDialog) aboutDialog.close();
});

// Keyboard
// Letters turn clockwise, Shift makes the turn counter-clockwise. The key is
// compared case-insensitively so Caps Lock has no effect. With a solution
// showing, the arrow keys step through it and Space plays or pauses.
window.addEventListener('keydown', e => {
  if (aboutDialog.open || e.target.closest?.('input, textarea, select')) return;
  if (e.key === 'Escape') {
    if (!wideQuery.matches) setHelp(false, true);
    return;
  }
  if (e.ctrlKey || e.metaKey) {
    if (e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
    return;
  }
  if (e.altKey || e.repeat) return;
  if (cube.solution && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
    e.preventDefault();
    if (e.key === 'ArrowLeft') stepBack(); else stepForward();
    return;
  }
  if (cube.solution && e.key === ' ' && !e.target.closest?.('button, summary, a')) {
    e.preventDefault();
    if (cube.solution.playing) pause(); else play();
    return;
  }
  if (e.key.length !== 1) return;
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

  // Playing the solution feeds it into the queue one move at a time, so a
  // pause takes effect after the current move.
  if (cube.solution?.playing && !isAnimating()) {
    const next = cube.nextSolutionTurn();
    if (next) queue.push(next);
    else {
      cube.solution.playing = false;
      updateUI();
    }
  }
  if (!current && queue.length) {
    startTurn(queue.shift());
    updateUI();
  }
  if (current) {
    current.elapsed += dt;
    const t = Math.min(current.elapsed / current.duration, 1);
    const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    current.pivot.setRotationFromAxisAngle(current.axis, current.angle * eased);
    if (t >= 1) finishTurn();
  }
  if (solveRequest) renderStatus(); // the elapsed time ticks

  orbit.update();
  renderer.render(scene, camera);
}

paint();
updateUI();
requestAnimationFrame(animate);
