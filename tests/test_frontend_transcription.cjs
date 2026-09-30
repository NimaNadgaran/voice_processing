// Pure rendering regressions, executable with node; no browser package needed.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const context = vm.createContext({ document: { addEventListener() {} }, console });
vm.runInContext(fs.readFileSync('frontend/app.js', 'utf8'), context);
const render = (transcript) => vm.runInContext(
  `transcriptPanel(${JSON.stringify(transcript)}, 'job_test', 'path1')`, context);
const good = render({ status: 'ok', method: 'faster_whisper', language: 'fa',
  text: 'سلام دنیا\n<script>alert(1)</script>', file: 'speaker01.txt',
  metrics: { warnings: ['Review repetition <unsafe>'] } });
assert.match(good, /Download text file/);
assert.match(good, /dir="auto"/);
assert.match(good, /سلام دنیا/);
assert.match(good, /&lt;script&gt;/);
assert.doesNotMatch(good, /<script>/);
assert.match(good, /api\/files\/job_test\/path1\/speaker01.txt/);
assert.match(good, /transcript-warning/);
assert.match(good, /Review repetition &lt;unsafe&gt;/);
const failed = render({ status: 'failed', method: 'vosk', error: 'No model <unsafe>' });
assert.match(failed, /disabled>Download text file/);
assert.match(failed, /&lt;unsafe&gt;/);
assert.doesNotMatch(failed, /href=/);
assert.equal(render({ status: 'disabled' }), '');
console.log('Transcript rendering, Persian direction, safe escaping and download states passed.');
