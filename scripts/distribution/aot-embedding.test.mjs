import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, test } from 'node:test';
import { fileURLToPath } from 'node:url';

const root = mkdtempSync(join(tmpdir(), 'runmat-aot-embedding-test-'));
after(() => rmSync(root, { recursive: true, force: true }));
const executable = process.platform === 'win32' ? '.exe' : '';
const builder = join(root, `build${executable}`);
const manifest = join(root, 'manifest.json');
const payload = join(root, 'payload');
// The build script copies metadata; runtime archive/pack admission checks its contents
// later. This fixture exercises copying/embedding, not archive admission.
const metadata = '{"fixture":"compile-time pack"}\n';
writeFileSync(manifest, metadata);
writeFileSync(payload, Buffer.from([1, 2, 3, 4]));

function run(command, args, env = process.env) {
  const result = spawnSync(command, args, { env, encoding: 'utf8' });
  if (result.error) throw result.error;
  return result;
}

function compile(source, output, env) {
  const result = run(process.env.RUSTC || 'rustc', ['--edition=2021', '-Dwarnings', source, '-o', output], env);
  assert.equal(result.status, 0, result.stderr);
}

compile(fileURLToPath(new URL('../../crates/runmat-aot/build.rs', import.meta.url)), builder, process.env);

function environment(name, includePayload, includeManifest) {
  const out = join(root, name);
  mkdirSync(out);
  const env = { ...process.env, OUT_DIR: out, CARGO_CFG_TARGET_ENV: '' };
  delete env.RUNMAT_AOT_RUNTIME_ARCHIVE;
  delete env.RUNMAT_AOT_RUNTIME_MANIFEST;
  if (includePayload) env.RUNMAT_AOT_RUNTIME_ARCHIVE = payload;
  if (includeManifest) env.RUNMAT_AOT_RUNTIME_MANIFEST = manifest;
  return env;
}

for (const [name, includePayload, includeManifest] of [
  ['no-pack', false, false],
  ['external-pack', false, true],
  ['embedded-pack', true, true],
]) {
  test(name, () => {
    const env = environment(name, includePayload, includeManifest);
    assert.equal(run(builder, [], env).status, 0);
    const probe = join(env.OUT_DIR, 'probe.rs');
    writeFileSync(probe, `
mod generated { include!(concat!(env!("OUT_DIR"), "/embedded.rs")); }
fn main() {
    assert_eq!(generated::PAYLOAD, ${includePayload ? 'Some(&[1u8, 2, 3, 4][..])' : 'None'});
    assert_eq!(generated::MANIFEST, ${includeManifest ? 'Some("{\\\"fixture\\\":\\\"compile-time pack\\\"}\\n")' : 'None'});
}
`);
    const program = join(env.OUT_DIR, `probe${executable}`);
    compile(probe, program, env);
    const result = run(program, [], env);
    assert.equal(result.status, 0, result.stderr);
    if (includeManifest) assert.equal(readFileSync(join(env.OUT_DIR, 'runtime-archive.json'), 'utf8'), metadata);
  });
}

test('a payload without its manifest is refused', () => {
  const result = run(builder, [], environment('invalid-payload-only', true, false));
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /RUNMAT_AOT_RUNTIME_ARCHIVE requires RUNMAT_AOT_RUNTIME_MANIFEST/);
});
