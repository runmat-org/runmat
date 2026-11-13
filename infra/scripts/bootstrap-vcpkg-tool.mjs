// Alternate transport for the exact tools selected by the pinned vcpkg trees.
import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { downloadAsset } from './github-release-asset-cache.mjs';

const release = '2026-05-27';
const tools = {
  'linux-x64': {
    asset: 431450369,
    name: 'vcpkg',
    sha256: '459de9d0d4dbdfbec760d87a2d62e48f94c1f60b5473498b4505689df045c35b',
  },
  'win32-x64': {
    asset: 431450492,
    name: 'vcpkg.exe',
    sha256: 'da75e3312ff6881c89f6171363eedb92933b0f79456cd6ee636316edef860ff7',
  },
};

export function verifyPayload(bytes, sha256) {
  if (createHash('sha256').update(bytes).digest('hex') !== sha256) {
    throw new Error('Pinned vcpkg tool checksum mismatch');
  }
}

function downloadWithGh(asset, destination) {
  downloadAsset('microsoft', 'vcpkg-tool', asset, destination);
}

export function stagePinnedTool(root, {
  platform = process.platform, arch = process.arch, download = downloadWithGh,
} = {}) {
  const tool = tools[`${platform}-${arch}`];
  if (!tool) throw new Error(`No pinned fallback vcpkg tool for ${platform}-${arch}`);
  const metadata = fs.readFileSync(path.join(root, 'scripts', 'vcpkg-tool-metadata.txt'), 'utf8');
  const tags = metadata.split(/\r?\n/).filter(line => line.startsWith('VCPKG_TOOL_RELEASE_TAG='));
  if (tags.length !== 1 || tags[0].slice('VCPKG_TOOL_RELEASE_TAG='.length) !== release) {
    throw new Error('vcpkg source selects a different tool release; review the fallback pins');
  }
  const target = path.join(root, tool.name);
  const scratch = path.join(root, `.runmat-vcpkg-${randomUUID()}`);
  try {
    download(tool.asset, scratch);
    verifyPayload(fs.readFileSync(scratch), tool.sha256);
    // Replace only the bootstrap destination, after verifying the complete file.
    fs.renameSync(scratch, target);
    if (platform !== 'win32') fs.chmodSync(target, 0o755);
    const metricsMarker = path.join(root, 'vcpkg.disable-metrics');
    if (!fs.existsSync(metricsMarker)) fs.writeFileSync(metricsMarker, '', { flag: 'wx' });
    return target;
  } finally {
    // This invocation owns this uniquely named download, including failed reads.
    fs.rmSync(scratch, { force: true });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node bootstrap-vcpkg-tool.mjs VCPKG_ROOT');
  const executable = stagePinnedTool(fs.realpathSync(process.argv[2]));
  execFileSync(executable, ['version', '--disable-metrics'], { stdio: 'inherit' });
}
