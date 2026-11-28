// Runs the network solver off the main thread, so the page keeps animating
// while it searches. The network (models/policy.onnx) runs with ONNX Runtime
// Web on the CPU through WebAssembly. The search is in solver.js.
//
// Messages in:  { type: 'solve', id, state }  state is 54 letters, centres home
//               { type: 'cancel' }
// Messages out: { id, type: 'loading', loaded, total }  downloading the network (total 0 if unknown)
//               { id, type: 'starting' }                 creating the ONNX Runtime session
//               { id, type: 'searching', depth, wide }   wide is true on the second, wider attempt
//               { id, type: 'done', moves, seconds }     moves is null if none was found
//               { id, type: 'error', message }
// A new 'solve' or a 'cancel' stops the search in progress, which then sends nothing more.
// The network is downloaded and started once and shared. Its loading and
// starting messages always go to the newest request, even when a request that
// was stopped began the loading.
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

let sessionPromise = null; // the started network, once a download has begun
let setup = null;          // the latest 'loading' or 'starting' message, until the network is ready
let current = null;        // { id, signal } of the newest request
let chain = Promise.resolve(); // runs searches one at a time

self.onmessage = ({ data }) => {
  if (current) current.signal.aborted = true;
  current = null;
  if (data.type !== 'solve') return;
  const job = { id: data.id, state: data.state, signal: { aborted: false } };
  current = job;
  // Loading already under way reports to this request from now on.
  if (setup) postToCurrent(setup);
  chain = chain.then(() => run(job));
};

function postToCurrent(msg) {
  if (current && !current.signal.aborted) self.postMessage({ id: current.id, ...msg });
}

// Report a loading step to the newest request, and remember it for the next one.
function reportSetup(msg) {
  setup = msg;
  postToCurrent(msg);
}

function getSession() {
  sessionPromise ??= loadSession().catch(err => {
    sessionPromise = null; // the next request tries again
    throw err;
  }).finally(() => { setup = null; });
  return sessionPromise;
}

// Download the network, reporting progress, then start a session on it.
async function loadSession() {
  reportSetup({ type: 'loading', loaded: 0, total: 0 });
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`couldn't download the network (HTTP ${response.status})`);
  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  let lastPercent = -1;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    const percent = total ? Math.floor((100 * loaded) / total) : -1;
    if (percent !== lastPercent) {
      lastPercent = percent;
      reportSetup({ type: 'loading', loaded, total });
    } else {
      setup = { type: 'loading', loaded, total };
    }
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  reportSetup({ type: 'starting' });
  return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
}

async function run(job) {
  if (job.signal.aborted) return;
  const post = msg => { if (!job.signal.aborted) self.postMessage({ id: job.id, ...msg }); };
  try {
    const session = await getSession();
    if (job.signal.aborted) return;
    const logPolicy = async (x, n) => {
      const out = await session.run({ onehot: new ort.Tensor('float32', x, [n, INPUT_SIZE]) });
      return out.log_probs.data;
    };
    const start = performance.now();
    for (const [attempt, width] of WIDTHS.entries()) {
      const wide = attempt > 0;
      post({ type: 'searching', depth: 0, wide });
      const moves = await beamSearch(job.state, logPolicy, {
        width,
        maxDepth: MAX_DEPTH,
        signal: job.signal,
        onDepth: depth => post({ type: 'searching', depth, wide }),
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
