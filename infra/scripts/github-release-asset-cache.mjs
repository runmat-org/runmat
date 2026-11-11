// vcpkg's x-script protocol supplies the original URL, SHA512 and destination.
// https://learn.microsoft.com/vcpkg/users/assetcaching
import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export function parseReleaseAsset(value) {
  const url = new URL(value);
  if (url.origin !== 'https://github.com' || url.username || url.password || url.search || url.hash) return null;
  const pieces = url.pathname.split('/');
  if (pieces.length !== 7 || pieces[3] !== 'releases' || pieces[4] !== 'download') return null;
  const [, owner, repository, , , encodedTag, encodedName] = pieces;
  if (![owner, repository].every(part => /^[A-Za-z0-9_.-]+$/.test(part))) return null;
  const tag = decodeURIComponent(encodedTag), name = decodeURIComponent(encodedName);
  if (!tag || !name || name.includes('/') || name.includes('\\') || /[\x00-\x1f]/.test(tag + name)) return null;
  return { owner, repository, tag, name };
}

export function downloadAsset(owner, repository, asset, destination, gh = 'gh') {
  const fd = fs.openSync(destination, 'wx', 0o700);
  try {
    const result = spawnSync(gh, [
      'api', '-H', 'Accept: application/octet-stream',
      `repos/${owner}/${repository}/releases/assets/${asset}`,
    ], { stdio: ['ignore', fd, 'inherit'], timeout: 120_000, windowsHide: true });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`GitHub asset download failed: ${result.status}`);
  } finally {
    fs.closeSync(fd);
  }
}

export function verifyFile(file, algorithm, expected) {
  const digest = createHash(algorithm), fd = fs.openSync(file, 'r'), buffer = Buffer.alloc(1024 * 1024);
  try {
    let count;
    while ((count = fs.readSync(fd, buffer, 0, buffer.length, null))) digest.update(buffer.subarray(0, count));
  } finally {
    fs.closeSync(fd);
  }
  if (digest.digest('hex') !== expected) throw new Error('GitHub asset checksum mismatch');
}

export function fetchReleaseAsset(url, sha512, destination, gh) {
  const selected = parseReleaseAsset(url);
  if (!selected) return false; // Other origins retain vcpkg's normal downloader.
  if (!/^[0-9a-f]{128}$/i.test(sha512)) throw new Error('Expected a complete vcpkg SHA512');
  if (!path.isAbsolute(destination) || !path.basename(destination)) throw new Error('Expected a named absolute asset destination');
  const { owner, repository, tag, name } = selected;
  const release = JSON.parse(execFileSync(gh, [
    'api', `repos/${owner}/${repository}/releases/tags/${encodeURIComponent(tag)}`,
  ], { encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, timeout: 30_000, windowsHide: true }));
  const assets = release.assets.filter(asset => asset.name === name);
  if (assets.length !== 1 || !Number.isSafeInteger(assets[0].id) || assets[0].id <= 0) {
    throw new Error('The pinned URL has no unique GitHub release asset');
  }
  const scratch = `${destination}.runmat-${randomUUID()}`;
  try {
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    downloadAsset(owner, repository, assets[0].id, scratch, gh);
    verifyFile(scratch, 'sha512', sha512.toLowerCase());
    fs.renameSync(scratch, destination);
    return true;
  } finally {
    fs.rmSync(scratch, { force: true });
  }
}

function ghExecutable() {
  const name = process.platform === 'win32' ? 'gh.exe' : 'gh';
  for (const directory of (process.env.PATH ?? '').split(path.delimiter)) {
    const file = path.resolve(directory, name);
    if (fs.existsSync(file) && fs.statSync(file).isFile()) return file;
  }
  throw new Error('GitHub CLI is required for the release-asset source');
}

export function assetSource(node, script, gh, platform = process.platform) {
  const quote = value => {
    if (/[\r\n{}]/.test(value)) throw new Error('Unsupported asset-provider executable path');
    if (platform === 'win32') {
      if (/["%]/.test(value)) throw new Error('Unsupported Windows asset-provider executable path');
      return `"${value}"`;
    }
    return `'${value.replaceAll("'", "'\\''")}'`;
  };
  // vcpkg quotes substituted arguments; only executable paths need quoting here.
  const command = `${quote(node)} ${quote(script)} {url} {sha512} {dst} ${quote(gh)}`;
  return 'x-script,' + command.replaceAll('`', '``').replaceAll(',', '`,').replaceAll(';', '`;');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length === 3 && process.argv[2] === '--config') {
    console.log(assetSource(process.execPath, fileURLToPath(import.meta.url), ghExecutable()));
  } else {
    const [url, sha512, destination, gh] = process.argv.slice(2);
    try {
      if (process.argv.length !== 6 || !fetchReleaseAsset(url, sha512, destination, gh)) process.exitCode = 1;
    } catch (error) {
      console.error(`GitHub release-asset source: ${error.message}`);
      process.exitCode = 1;
    }
  }
}
