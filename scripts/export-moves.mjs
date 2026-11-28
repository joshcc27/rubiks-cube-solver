// Writes training/moves.json, which gives the Python training code the same
// moves as cube-model.js. Run it again whenever cube-model.js changes:
//   node scripts/export-moves.mjs      (or: npm run export-moves)
// A test fails if the file is out of date.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { FACES, FACE_MOVES, solvedState, applyMove, applyMoves } from '../cube-model.js';

export const OUTPUT_URL = new URL('../training/moves.json', import.meta.url);

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
  const examples = Array.from({ length: 20 }, () => {
    const moves = Array.from({ length: 30 }, () => FACE_MOVES[Math.floor(random() * FACE_MOVES.length)]);
    return { moves: moves.join(' '), state: applyMoves(solvedState(), moves).join('') };
  });

  return { faces: FACES.join(''), moves: FACE_MOVES.join(' '), source, examples };
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
