import fs from "node:fs";

const POSIX_POLICY = Object.freeze({
  kind: "directory-fsync-required",
  guarantee: "file-data-and-parent-directory-metadata",
});
const WINDOWS_POLICY = Object.freeze({
  kind: "directory-fsync-unavailable",
  guarantee: "file-data-and-atomic-filesystem-operation-only",
});

export function directoryDurabilityPolicy(platform = process.platform) {
  return platform === "win32" ? WINDOWS_POLICY : POSIX_POLICY;
}

export function syncDirectory(directory, options = {}) {
  const filesystem = options.filesystem ?? fs;
  const platform = options.platform ?? process.platform;
  const policy = directoryDurabilityPolicy(platform);
  if (policy.kind === "directory-fsync-unavailable") return policy;

  const constants = filesystem.constants ?? fs.constants;
  const descriptor = filesystem.openSync(
    directory,
    constants.O_RDONLY | (constants.O_DIRECTORY ?? 0) | (constants.O_NOFOLLOW ?? 0),
  );
  try {
    filesystem.fsyncSync(descriptor);
  } finally {
    filesystem.closeSync(descriptor);
  }
  return policy;
}
