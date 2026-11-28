// Beam search guided by the trained network, the same search as
// training/src/search.py. This file has no network runtime in it: the caller
// passes logPolicy, which takes a batch of encoded states and returns
// log-probabilities over FACE_MOVES. The page's Web Worker passes the ONNX
// model; the tests pass a stand-in.

import { FACES, FACE_MOVES, applyMove } from './cube-model.js';

const NUM_MOVES = FACE_MOVES.length;
const CENTRES = [4, 13, 22, 31, 40, 49];
const NON_CENTRES = [...Array(54).keys()].filter(i => !CENTRES.includes(i));
export const INPUT_SIZE = NON_CENTRES.length * 6;

// SOURCE[m][i] is the position whose sticker lands on position i after move m.
const IDENTITY = Array.from({ length: 54 }, (_, i) => i);
const SOURCE = FACE_MOVES.map(mv => Int32Array.from(applyMove(IDENTITY, mv)));

// Two turns of one face in a row are never needed, and turns of opposite faces
// commute, so only one order is kept (U before D, R before L, F before B). In
// FACES order the opposite of face f is f + 3. prev is -1 at the start.
const faceOf = m => Math.floor(m / 3);
function allowedAfter(prev, m) {
  if (prev < 0) return true;
  return faceOf(m) !== faceOf(prev) && faceOf(m) + 3 !== faceOf(prev);
}

function isSolvedAt(states, base) {
  for (let i = 0; i < 54; i++) if (states[base + i] !== Math.floor(i / 9)) return false;
  return true;
}

// One-hot colour of each of the 48 non-centre stickers, matching
// training/src/model.py encode().
export function encode(states, n) {
  const x = new Float32Array(n * INPUT_SIZE);
  for (let s = 0; s < n; s++) {
    NON_CENTRES.forEach((pos, k) => { x[s * INPUT_SIZE + k * 6 + states[s * 54 + pos]] = 1; });
  }
  return x;
}

export class SearchAborted extends Error {
  constructor() { super('search aborted'); this.name = 'SearchAborted'; }
}

// state: 54 face letters with the centres home (pass it through
// relabelByCentres first). Returns the solution as move names, [] if already
// solved, or null if none is found within maxDepth moves.
//
// logPolicy(encoded, n) returns n * 18 log-probabilities and may be async.
// signal.aborted is checked once per move; when set, SearchAborted is thrown.
export async function beamSearch(state, logPolicy, { width = 512, maxDepth = 60, signal, onDepth } = {}) {
  let states = Uint8Array.from(state, c => FACES.indexOf(c));
  if (CENTRES.some((pos, f) => states[pos] !== f)) throw new Error('centres must be home: use relabelByCentres');
  if (isSolvedAt(states, 0)) return [];

  let n = 1;
  let scores = new Float64Array(1);
  let last = new Int8Array([-1]);
  const levels = []; // for each depth: the parent and move of every state kept in the beam

  const pathTo = (parent, move) => {
    const moves = [move];
    for (let d = levels.length - 1; d >= 0; d--) {
      moves.push(levels[d].move[parent]);
      parent = levels[d].parent[parent];
    }
    return moves.reverse().map(m => FACE_MOVES[m]);
  };

  for (let depth = 1; depth <= maxDepth; depth++) {
    const logp = await logPolicy(encode(states, n), n);
    if (signal?.aborted) throw new SearchAborted();

    // Every allowed child of every state in the beam.
    const children = new Uint8Array(n * NUM_MOVES * 54);
    const childScore = new Float64Array(n * NUM_MOVES);
    const childParent = new Int32Array(n * NUM_MOVES);
    const childMove = new Uint8Array(n * NUM_MOVES);
    let count = 0;
    for (let p = 0; p < n; p++) {
      for (let m = 0; m < NUM_MOVES; m++) {
        if (!allowedAfter(last[p], m)) continue;
        const src = SOURCE[m];
        const base = count * 54;
        for (let i = 0; i < 54; i++) children[base + i] = states[p * 54 + src[i]];
        // Check every child for solved, not only the ones the beam keeps.
        if (isSolvedAt(children, base)) return pathTo(p, m);
        childScore[count] = scores[p] + logp[p * NUM_MOVES + m];
        childParent[count] = p;
        childMove[count] = m;
        count++;
      }
    }

    // Keep the best-scoring copy of each distinct state, then the top `width`.
    const order = Array.from({ length: count }, (_, i) => i).sort((a, b) => childScore[b] - childScore[a]);
    const seen = new Set();
    const keep = [];
    for (const c of order) {
      const key = String.fromCharCode.apply(null, children.subarray(c * 54, c * 54 + 54));
      if (seen.has(key)) continue;
      seen.add(key);
      keep.push(c);
      if (keep.length === width) break;
    }

    n = keep.length;
    const next = new Uint8Array(n * 54);
    keep.forEach((c, k) => next.set(children.subarray(c * 54, c * 54 + 54), k * 54));
    states = next;
    scores = Float64Array.from(keep, c => childScore[c]);
    last = Int8Array.from(keep, c => childMove[c]);
    levels.push({ parent: Int32Array.from(keep, c => childParent[c]), move: Uint8Array.from(keep, c => childMove[c]) });
    onDepth?.(depth);
  }
  return null;
}
