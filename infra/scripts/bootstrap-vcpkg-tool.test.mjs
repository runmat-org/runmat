import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { stagePinnedTool, verifyPayload } from './bootstrap-vcpkg-tool.mjs';

test('payload verification accepts exact bytes and rejects changed bytes', () => {
  const bytes = Buffer.from('tool fixture');
  const hash = createHash('sha256').update(bytes).digest('hex');
  verifyPayload(bytes, hash);
  assert.throws(() => verifyPayload(Buffer.from('changed'), hash), /checksum mismatch/);
});

test('unsupported hosts and changed upstream releases never download', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'runmat-vcpkg-test-'));
  try {
    const download = () => assert.fail('unexpected download');
    assert.throws(() => stagePinnedTool(root, { platform: 'linux', arch: 'arm64', download }), /No pinned fallback/);
    fs.mkdirSync(path.join(root, 'scripts'));
    fs.writeFileSync(path.join(root, 'scripts', 'vcpkg-tool-metadata.txt'), 'VCPKG_TOOL_RELEASE_TAG=other\n');
    assert.throws(() => stagePinnedTool(root, { platform: 'linux', arch: 'x64', download }), /different tool release/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('a corrupt download preserves an existing tool and removes its own scratch', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'runmat-vcpkg-test-'));
  try {
    fs.mkdirSync(path.join(root, 'scripts'));
    fs.writeFileSync(path.join(root, 'scripts', 'vcpkg-tool-metadata.txt'), 'VCPKG_TOOL_RELEASE_TAG=2026-05-27\n');
    fs.writeFileSync(path.join(root, 'vcpkg'), 'existing');
    assert.throws(() => stagePinnedTool(root, {
      platform: 'linux', arch: 'x64',
      download: (_asset, file) => fs.writeFileSync(file, 'corrupt'),
    }), /checksum mismatch/);
    assert.equal(fs.readFileSync(path.join(root, 'vcpkg'), 'utf8'), 'existing');
    assert.deepEqual(fs.readdirSync(root).sort(), ['scripts', 'vcpkg']);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
