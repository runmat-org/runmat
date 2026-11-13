import assert from "node:assert/strict";
import test from "node:test";

import {
  directoryDurabilityPolicy, syncDirectory,
} from "../durability.mjs";

test("POSIX directory durability requires a no-follow directory fsync", () => {
  const calls = [];
  const filesystem = {
    constants: { O_RDONLY: 1, O_DIRECTORY: 2, O_NOFOLLOW: 4 },
    openSync(target, flags) { calls.push(["open", target, flags]); return 7; },
    fsyncSync(descriptor) { calls.push(["fsync", descriptor]); },
    closeSync(descriptor) { calls.push(["close", descriptor]); },
  };
  assert.deepEqual(syncDirectory("/repository", { platform: "linux", filesystem }), {
    kind: "directory-fsync-required",
    guarantee: "file-data-and-parent-directory-metadata",
  });
  assert.deepEqual(calls, [
    ["open", "/repository", 7], ["fsync", 7], ["close", 7],
  ]);
});

test("POSIX directory durability never hides an unsupported fsync", () => {
  let closed = false;
  const filesystem = {
    constants: { O_RDONLY: 1, O_DIRECTORY: 2, O_NOFOLLOW: 4 },
    openSync() { return 7; },
    fsyncSync() { const error = new Error("unsupported"); error.code = "EINVAL"; throw error; },
    closeSync() { closed = true; },
  };
  assert.throws(
    () => syncDirectory("/repository", { platform: "darwin", filesystem }),
    /unsupported/,
  );
  assert.equal(closed, true);
});

test("Windows policy explicitly limits durability to file fsync and atomic operations", () => {
  const filesystem = new Proxy({}, {
    get() { throw new Error("Windows directory operations must not be attempted"); },
  });
  assert.deepEqual(syncDirectory("C:\\repository", { platform: "win32", filesystem }), {
    kind: "directory-fsync-unavailable",
    guarantee: "file-data-and-atomic-filesystem-operation-only",
  });
  assert.equal(directoryDurabilityPolicy("win32").kind, "directory-fsync-unavailable");
});
