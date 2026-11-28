// Logical model of a 3x3 cube. No rendering code lives here, so the same
// module is used by the page, by the tests, and later by a solver.
//
// Coordinates: x points to the R face, y to U, z to F. Every cubie sits at
// integer coordinates in {-1, 0, 1}. Rotations follow the right-hand rule,
// which is also what Three.js uses: a positive quarter turn about an axis is
// counter-clockwise when viewed from the positive end of that axis.
//
// State: an array of 54 face letters, one per sticker, in the order
// U1..U9 R1..R9 F1..F9 D1..D9 L1..L9 B1..B9. Each letter names the face whose
// colour the sticker currently shows. For a solved cube in the home
// orientation, state.join('') gives the usual facelet string
// "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB".
//
// Each face is read row by row as you look straight at it:
//   U: back row first, left to right (B edge at the top)
//   R, F, L, B: top row first, with U at the top
//   D: front row first, left to right (F edge at the top)

export const FACES = ['U', 'R', 'F', 'D', 'L', 'B'];

// The 18 face turns, three per face in FACES order. The solver network's
// outputs are in this order (see training/moves.json).
export const FACE_MOVES = FACES.flatMap(f => [f, `${f}'`, `${f}2`]);

const AXIS_INDEX = { x: 0, y: 1, z: 2 };

// For each face: its outward normal, and how row r and column c (0..2) map
// to a cubie position.
const FACE_LAYOUT = {
  U: { normal: [0, 1, 0],  pos: (r, c) => [c - 1, 1, r - 1] },
  R: { normal: [1, 0, 0],  pos: (r, c) => [1, 1 - r, 1 - c] },
  F: { normal: [0, 0, 1],  pos: (r, c) => [c - 1, 1 - r, 1] },
  D: { normal: [0, -1, 0], pos: (r, c) => [c - 1, -1, 1 - r] },
  L: { normal: [-1, 0, 0], pos: (r, c) => [-1, 1 - r, c - 1] },
  B: { normal: [0, 0, -1], pos: (r, c) => [1 - c, 1 - r, -1] },
};

// FACELETS[i] describes sticker i: which face it starts on, the position of
// the cubie it is stuck to, and the direction it faces.
export const FACELETS = [];
for (const face of FACES) {
  const { normal, pos } = FACE_LAYOUT[face];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      FACELETS.push({ face, pos: pos(r, c), normal: normal.slice() });
    }
  }
}

// Quarter turns. dir is the sign of the right-hand rotation about the axis.
//   U, R, F turn clockwise as seen from outside that face (the positive end
//   of the axis), which is a negative rotation: dir -1.
//   D, L, B turn clockwise as seen from their own side (the negative end of
//   the axis), which is a positive rotation: dir +1.
//   M turns like L, E like D, S like F.
export const BASE_MOVES = {
  U: { axis: 'y', layer: 1,  dir: -1 },
  E: { axis: 'y', layer: 0,  dir: 1 },
  D: { axis: 'y', layer: -1, dir: 1 },
  R: { axis: 'x', layer: 1,  dir: -1 },
  M: { axis: 'x', layer: 0,  dir: 1 },
  L: { axis: 'x', layer: -1, dir: 1 },
  F: { axis: 'z', layer: 1,  dir: -1 },
  S: { axis: 'z', layer: 0,  dir: -1 },
  B: { axis: 'z', layer: -1, dir: 1 },
};

// One positive quarter turn of an integer vector about an axis.
function quarterTurn([x, y, z], axis) {
  if (axis === 'x') return [x, -z, y];
  if (axis === 'y') return [z, y, -x];
  return [-y, x, z];
}

function rotate(v, axis, quarterTurns) {
  let out = v;
  const n = ((quarterTurns % 4) + 4) % 4;
  for (let i = 0; i < n; i++) out = quarterTurn(out, axis);
  return out;
}

const key = (pos, normal) => `${pos.join(',')}|${normal.join(',')}`;
const INDEX_BY_KEY = new Map(FACELETS.map((f, i) => [key(f.pos, f.normal), i]));

// PERMUTATIONS[name][i] is where sticker i ends up after one quarter turn.
const PERMUTATIONS = {};
for (const [name, { axis, layer, dir }] of Object.entries(BASE_MOVES)) {
  const a = AXIS_INDEX[axis];
  PERMUTATIONS[name] = FACELETS.map((f, i) => {
    if (f.pos[a] !== layer) return i;
    const target = INDEX_BY_KEY.get(key(rotate(f.pos, axis, dir), rotate(f.normal, axis, dir)));
    if (target === undefined) throw new Error(`bad permutation for ${name}`);
    return target;
  });
}

// Parse a move like "R", "R'" or "R2". Returns null for anything else.
export function parseMove(name) {
  const m = /^([URFDLBMES])(['2]?)$/.exec(name);
  if (!m) return null;
  const turns = m[2] === "'" ? -1 : m[2] === '2' ? 2 : 1;
  return { base: m[1], turns, ...BASE_MOVES[m[1]] };
}

export function invertMove(name) {
  const m = parseMove(name);
  if (!m) throw new Error(`unknown move: ${name}`);
  if (m.turns === 2) return name;
  return m.turns === 1 ? `${m.base}'` : m.base;
}

// Parse moves typed by a person, such as "R U R' U'", "RUR'U'" or "R2' U’".
// Typographic apostrophes count as primes, and "2'" or "'2" as a double turn.
// Returns move names in the form parseMove accepts. Throws an Error whose
// message can be shown to the person.
export function parseSequence(text, maxMoves = 500) {
  const src = text.replace(/[’‘´`′]/g, "'").trim();
  const token = /\s*([URFDLBMES])(2'|'2|'|2)?/y;
  const moves = [];
  while (token.lastIndex < src.length) {
    const start = token.lastIndex;
    const m = token.exec(src);
    if (!m) {
      const bad = src.slice(start).trim().split(/\s+/)[0];
      throw new Error(`"${bad}" isn't a move. Use the letters U D R L F B M E S, optionally followed by ' or 2.`);
    }
    moves.push(m[1] + (m[2] === "'" ? "'" : m[2] ? '2' : ''));
    if (moves.length > maxMoves) throw new Error(`That's more than ${maxMoves} moves.`);
  }
  return moves;
}

// Accepts "R U R' U'" or ['R', 'U', "R'", "U'"].
export function splitMoves(seq) {
  if (Array.isArray(seq)) return seq;
  return seq.trim().split(/\s+/).filter(Boolean);
}

export function solvedState() {
  return FACELETS.map(f => f.face);
}

export function applyMove(state, name) {
  const m = parseMove(name);
  if (!m) throw new Error(`unknown move: ${name}`);
  const perm = PERMUTATIONS[m.base];
  let cur = state;
  const quarters = ((m.turns % 4) + 4) % 4;
  for (let q = 0; q < quarters; q++) {
    const next = new Array(54);
    for (let i = 0; i < 54; i++) next[perm[i]] = cur[i];
    cur = next;
  }
  return cur === state ? state.slice() : cur;
}

export function applyMoves(state, seq) {
  return splitMoves(seq).reduce(applyMove, state);
}

// Solved means every face shows one colour. Slice moves move the centres, so
// this does not require the cube to be in its home orientation.
export function isSolved(state) {
  for (let f = 0; f < 6; f++) {
    const first = state[f * 9];
    for (let i = 1; i < 9; i++) if (state[f * 9 + i] !== first) return false;
  }
  return true;
}

// Rename every colour after the face whose centre it now shows. Slice moves
// carry the centres around, and the solver only knows the home layout. After
// relabelling, the centres are home and the cube can be solved with face turns
// alone. Face turns never move centres, so they commute with this relabelling,
// and a face-turn solution of the relabelled state also solves the original.
export function relabelByCentres(state) {
  const rename = {};
  FACES.forEach((face, f) => { rename[state[f * 9 + 4]] = face; });
  return state.map(x => rename[x]);
}

// Random-move scramble using face turns only. Two consecutive moves never
// share an axis, so there are no cancellations like "U U'" and no runs like
// "U D U".
export function generateScramble(length = 20, random = Math.random) {
  const faces = ['U', 'D', 'R', 'L', 'F', 'B'];
  const suffixes = ['', "'", '2'];
  const moves = [];
  let lastAxis = null;
  while (moves.length < length) {
    const face = faces[Math.floor(random() * faces.length)];
    const axis = BASE_MOVES[face].axis;
    if (axis === lastAxis) continue;
    lastAxis = axis;
    moves.push(face + suffixes[Math.floor(random() * suffixes.length)]);
  }
  return moves;
}
