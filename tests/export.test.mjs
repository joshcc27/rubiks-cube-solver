import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { OUTPUT_URL, buildMoveData, formatMoveData } from '../scripts/export-moves.mjs';

test('training/moves.json matches cube-model.js (rerun scripts/export-moves.mjs if not)', () => {
  assert.equal(readFileSync(OUTPUT_URL, 'utf8').replace(/\r\n/g, '\n'), formatMoveData(buildMoveData()));
});
