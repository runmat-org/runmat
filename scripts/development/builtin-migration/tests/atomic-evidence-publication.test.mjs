import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import {
  canonicalEvidencePath, EvidenceTargetExistsError, publishEvidenceBytes,
} from "../atomic-evidence-publication.mjs";
import {
  cleanupTemporaryDirectories, createTemporaryDirectory,
} from "./temporary-directories.mjs";

test.afterEach(cleanupTemporaryDirectories);

test("publication preserves caller-supplied bytes and durably creates one new file", () => {
  const root = createTemporaryDirectory("runmat-evidence-publication-");
  const target = path.join(root, "evidence.json");
  const bytes = Buffer.from("{\n  \"z\": 1, \"a\": 2\n}\n\0", "utf8");
  let syncs = 0;
  const filesystem = interceptedFilesystem({
    fsyncSync(descriptor) { syncs += 1; return fs.fsyncSync(descriptor); },
  });
  const result = publishEvidenceBytes(target, bytes, {
    filesystem, temporaryToken: "exact-bytes",
  });
  assert.deepEqual(fs.readFileSync(target), bytes);
  assert.equal(result.path, target);
  assert.equal(result.byteLength, bytes.length);
  assert.equal(result.durability.kind, "directory-fsync-required");
  assert.equal(syncs, 2, "the staged file and containing directory must both be synced");
  assert.deepEqual(temporaryEntries(root), []);
});

test("publication uses ordinary create permissions under the process umask", () => {
  const root = createTemporaryDirectory("runmat-evidence-publication-");
  const target = path.join(root, "default-mode.json");
  publishEvidenceBytes(target, "{}\n", { temporaryToken: "default-mode" });
  const expected = 0o666 & ~process.umask();
  assert.equal(fs.statSync(target).mode & 0o777, expected);

  const explicit = path.join(root, "explicit-mode.json");
  publishEvidenceBytes(explicit, "{}\n", {
    mode: 0o640, temporaryToken: "explicit-mode",
  });
  assert.equal(fs.statSync(explicit).mode & 0o777, 0o640 & ~process.umask());
  assert.throws(
    () => publishEvidenceBytes(path.join(root, "invalid-mode.json"), "{}\n", { mode: 0o1000 }),
    /mode must contain only Unix permission bits/,
  );
});

test("parent creation is opt-in and creates only real directory components", () => {
  const root = createTemporaryDirectory("runmat-evidence-publication-");
  const target = path.join(root, "new", "nested", "evidence.json");
  assert.throws(
    () => publishEvidenceBytes(target, "{}\n", { temporaryToken: "no-parent" }),
    /parent directory does not exist/,
  );
  publishEvidenceBytes(target, "{}\n", {
    createParentDirectories: true, temporaryToken: "with-parent",
  });
  assert.equal(fs.readFileSync(target, "utf8"), "{}\n");
});

test("existing regular, directory, and symbolic-link targets are never replaced", () => {
  const root = createTemporaryDirectory("runmat-evidence-publication-");
  const existing = path.join(root, "existing.json");
  fs.writeFileSync(existing, "original\n");
  const directory = path.join(root, "directory.json");
  fs.mkdirSync(directory);
  const symlink = path.join(root, "symlink.json");
  fs.symlinkSync(existing, symlink);
  for (const target of [existing, directory, symlink]) {
    assert.throws(
      () => publishEvidenceBytes(target, "replacement\n", { temporaryToken: "exists" }),
      (error) => error instanceof EvidenceTargetExistsError
        && error.code === "RUNMAT_EVIDENCE_TARGET_EXISTS"
        && error.target === target,
    );
  }
  assert.equal(
    new Error(`evidence target already exists: ${existing}`) instanceof EvidenceTargetExistsError,
    false,
    "matching message text must not acquire the typed idempotency discriminator",
  );
  assert.equal(fs.readFileSync(existing, "utf8"), "original\n");
  assert.deepEqual(temporaryEntries(root), []);
});

test("symbolic-link and non-directory parent components are rejected", () => {
  const root = createTemporaryDirectory("runmat-evidence-publication-");
  const real = path.join(root, "real");
  fs.mkdirSync(real);
  const alias = path.join(root, "alias");
  fs.symlinkSync(real, alias);
  assert.throws(
    () => publishEvidenceBytes(path.join(alias, "evidence.json"), "{}\n"),
    /parent component is not a real directory/,
  );
  const normalized = canonicalEvidencePath(path.join(alias, "normalized.json"));
  assert.equal(normalized, path.join(fs.realpathSync(real), "normalized.json"));
  publishEvidenceBytes(normalized, "normalized\n");
  assert.equal(fs.readFileSync(path.join(real, "normalized.json"), "utf8"), "normalized\n");
  const regular = path.join(root, "regular");
  fs.writeFileSync(regular, "not a directory");
  assert.throws(
    () => publishEvidenceBytes(path.join(regular, "evidence.json"), "{}\n"),
    /parent component is not a real directory/,
  );
});

test("a target racing publication wins without being overwritten", () => {
  const root = createTemporaryDirectory("runmat-evidence-publication-");
  const target = path.join(root, "evidence.json");
  assert.throws(
    () => publishEvidenceBytes(target, "ours\n", {
      temporaryToken: "race",
      hooks: { beforePublish() { fs.writeFileSync(target, "racer\n"); } },
    }),
    (error) => error instanceof EvidenceTargetExistsError && error.target === target,
  );
  assert.equal(fs.readFileSync(target, "utf8"), "racer\n");
  assert.deepEqual(temporaryEntries(root), []);
});

test("write, file-sync, and atomic-link failures clean the owned temporary", async (context) => {
  const cases = [
    ["write", { writeFileSync() { throw new Error("injected write failure"); } }],
    ["file sync", { fsyncSync() { throw new Error("injected sync failure"); } }],
    ["atomic link", { linkSync() { throw new Error("injected link failure"); } }],
  ];
  for (const [name, overrides] of cases) {
    await context.test(name, () => {
      const root = createTemporaryDirectory("runmat-evidence-publication-");
      const target = path.join(root, "evidence.json");
      assert.throws(() => publishEvidenceBytes(target, "{}\n", {
        filesystem: interceptedFilesystem(overrides), temporaryToken: name.replace(" ", "-"),
      }), /injected/);
      assert.equal(fs.existsSync(target), false);
      assert.deepEqual(temporaryEntries(root), []);
    });
  }
});

test("a target created after staging is detected before the atomic link", () => {
  const root = createTemporaryDirectory("runmat-evidence-publication-");
  const target = path.join(root, "evidence.json");
  assert.throws(() => publishEvidenceBytes(target, "ours\n", {
    temporaryToken: "post-stage",
    hooks: { afterStage() { fs.writeFileSync(target, "racer\n"); } },
  }), /target already exists/);
  assert.equal(fs.readFileSync(target, "utf8"), "racer\n");
  assert.deepEqual(temporaryEntries(root), []);
});

test("a staged-file symlink substitution is rejected without deleting the substitute", () => {
  const root = createTemporaryDirectory("runmat-evidence-publication-");
  const target = path.join(root, "evidence.json");
  const external = path.join(root, "external.json");
  fs.writeFileSync(external, "external\n");
  assert.throws(
    () => publishEvidenceBytes(target, "ours\n", {
      temporaryToken: "substitution",
      hooks: {
        afterStage({ temporary }) {
          fs.unlinkSync(temporary);
          fs.symlinkSync(external, temporary);
        },
      },
    }),
    (error) => error instanceof AggregateError
      && error.errors.some((entry) => /temporary changed before publication/.test(entry.message))
      && error.errors.some((entry) => /temporary changed before cleanup/.test(entry.message)),
  );
  assert.equal(fs.existsSync(target), false);
  assert.equal(fs.readFileSync(external, "utf8"), "external\n");
  const substitute = path.join(root, ".evidence.json.runmat-new-substitution");
  assert.equal(fs.lstatSync(substitute).isSymbolicLink(), true);
});

test("directory-sync failure reports a published target and leaves no owned temporary", () => {
  const root = createTemporaryDirectory("runmat-evidence-publication-");
  const target = path.join(root, "evidence.json");
  let syncs = 0;
  const filesystem = interceptedFilesystem({
    fsyncSync(descriptor) {
      syncs += 1;
      if (syncs === 2) throw new Error("injected directory sync failure");
      return fs.fsyncSync(descriptor);
    },
  });
  assert.throws(() => publishEvidenceBytes(target, "published\n", {
    filesystem, temporaryToken: "directory-sync",
  }), /target was published.*directory sync failure/);
  assert.equal(fs.readFileSync(target, "utf8"), "published\n");
  assert.deepEqual(temporaryEntries(root), []);
});

function interceptedFilesystem(overrides) {
  return new Proxy(fs, {
    get(target, property) {
      if (Object.hasOwn(overrides, property)) return overrides[property];
      const value = Reflect.get(target, property);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

function temporaryEntries(directory) {
  return fs.readdirSync(directory).filter((entry) => entry.includes(".runmat-new-"));
}
