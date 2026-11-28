// Writes training/moves.json, which gives the Python training code the same
// moves as cube-model.js. Run it again whenever cube-model.js changes:
//   node scripts/export-moves.mjs      (or: npm run export-moves)
// A test fails if the file is out of date.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { FACES, FACE_MOVES, BASE_MOVES, solvedState, applyMove, applyMoves } from '../cube-model.js';

export const OUTPUT_URL = new URL('../training/moves.json', import.meta.url);

// Every turn the page knows, slice turns included. The network only uses the
// 18 face turns, but src.solve reads scrambles typed with any of these.
export const EVERY_MOVE = Object.keys(BASE_MOVES).flatMap(b => [b, `${b}'`, `${b}2`]);

// Small seeded generator, so the examples are the same on every run.
function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildMoveData() {
  // Applying a move to the sticker indices 0..53 gives, for each position,
  // the index of the sticker that lands there.
  const identity = Array.from({ length: 54 }, (_, i) => i);
  const source = FACE_MOVES.map(mv => applyMove(identity, mv));

  // Random face-turn sequences and the states they reach, for checking the
  // Python implementation against this one.
  const random = mulberry32(2504);
  const randomSequence = (moveSet, length) => Array.from({ length }, () => moveSet[Math.floor(random() * moveSet.length)]);
  const example = moves => ({ moves: moves.join(' '), state: applyMoves(solvedState(), moves).join('') });
  const examples = Array.from({ length: 20 }, () => example(randomSequence(FACE_MOVES, 30)));
  // With slice turns the centres move, so these states are as the page shows them, before relabelling.
  const everyMoveExamples = Array.from({ length: 10 }, () => example(randomSequence(EVERY_MOVE, 20)));

  return {
    faces: FACES.join(''),
    moves: FACE_MOVES.join(' '),
    source,
    examples,
    every_move: EVERY_MOVE.join(' '),
    every_source: EVERY_MOVE.map(mv => applyMove(identity, mv)),
    every_move_examples: everyMoveExamples,
  };
}

// One line per number array, so the file stays readable.
export function formatMoveData(data) {
  return JSON.stringify(data, null, 2).replace(/\[\s+(\d+(?:,\s+\d+)*)\s+\]/g,
    (_, nums) => `[${nums.split(/,\s+/).join(', ')}]`) + '\n';
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  writeFileSync(OUTPUT_URL, formatMoveData(buildMoveData()));
  console.log(`wrote ${fileURLToPath(OUTPUT_URL)}`);
}
