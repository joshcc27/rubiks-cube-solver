// Runs the network solver off the main thread, so the page keeps animating
// while it searches. The network (models/policy.onnx) runs with ONNX Runtime
// Web on the CPU through WebAssembly. The search is in solver.js.
//
// Messages in:  { type: 'solve', id, state }  state is 54 letters, centres home
//               { type: 'cancel' }
// Messages out: { id, type: 'progress', text }
//               { id, type: 'done', moves, seconds }  moves is null if none found
//               { id, type: 'error', message }
// A new 'solve' or a 'cancel' stops the search in progress, which then sends nothing.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.min.mjs';
import { beamSearch, INPUT_SIZE, SearchAborted } from './solver.js';

// Wider beams solve more cubes but take longer. The second width is only tried
// if the first finds nothing within MAX_DEPTH moves.
const WIDTHS = [512, 2048];
const MAX_DEPTH = 60;
const MODEL_URL = new URL('./models/policy.onnx', import.meta.url).href;

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
// Several threads need a cross-origin isolated page, which GitHub Pages can't provide.
ort.env.wasm.numThreads = 1;

let session = null;
let current = null; // { id, signal } of the newest request
let chain = Promise.resolve(); // runs searches one at a time

self.onmessage = ({ data }) => {
  if (current) current.signal.aborted = true;
  current = null;
  if (data.type !== 'solve') return;
  const job = { id: data.id, state: data.state, signal: { aborted: false } };
  current = job;
  chain = chain.then(() => run(job));
};

async function run(job) {
  if (job.signal.aborted) return;
  const post = msg => { if (!job.signal.aborted) self.postMessage({ id: job.id, ...msg }); };
  try {
    if (!session) {
      post({ type: 'progress', text: 'Loading the network…' });
      session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    }
    const logPolicy = async (x, n) => {
      const out = await session.run({ onehot: new ort.Tensor('float32', x, [n, INPUT_SIZE]) });
      return out.log_probs.data;
    };
    const start = performance.now();
    for (const [attempt, width] of WIDTHS.entries()) {
      const label = attempt === 0 ? 'Searching' : 'No luck yet, searching wider';
      const moves = await beamSearch(job.state, logPolicy, {
        width,
        maxDepth: MAX_DEPTH,
        signal: job.signal,
        onDepth: depth => post({ type: 'progress', text: `${label}… ${depth} move${depth === 1 ? '' : 's'} deep` }),
      });
      if (moves) {
        post({ type: 'done', moves, seconds: (performance.now() - start) / 1000 });
        return;
      }
    }
    post({ type: 'done', moves: null, seconds: (performance.now() - start) / 1000 });
  } catch (err) {
    if (err instanceof SearchAborted) return;
    post({ type: 'error', message: err?.message ?? String(err) });
  }
}
