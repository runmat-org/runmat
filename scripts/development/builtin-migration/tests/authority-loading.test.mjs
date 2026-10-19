import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import { contentDigest } from "../evidence.mjs";
import {
  assertAuthorityLoadSession, assertAuthorityRoot, assertLoadedJsonArtifact,
  assertSessionArtifact, loadJsonArtifact, loadedJsonValue, openAuthorityLoadSession,
  openAuthorityRoot, revalidateObservedArtifacts, withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import {
  cleanupTemporaryDirectories, createTemporaryDirectory,
} from "./temporary-directories.mjs";

const DIGEST_A = `sha256:${"a".repeat(64)}`;

test.afterEach(cleanupTemporaryDirectories);

test("root canonicalizes once and loaded JSON is byte-bound, detached, and session-pinned", () => {
  const directory = createTemporaryDirectory("runmat-authority-loading-");
  const real = path.join(directory, "real");
  fs.mkdirSync(real);
  const alias = path.join(directory, "alias");
  fs.symlinkSync(real, alias, process.platform === "win32" ? "junction" : "dir");
  const bytes = Buffer.from("{\n  \"nested\": { \"value\": 1 }\n}\n");
  fs.writeFileSync(path.join(real, "authority.json"), bytes);

  const root = openAuthorityRoot(alias);
  assert.equal(root.path, fs.realpathSync(real));
  const session = openAuthorityLoadSession(root);
  const artifact = loadJsonArtifact(session, "authority.json");
  assert.equal(artifact.contentDigest, contentDigest(bytes));
  assert.equal(artifact.byteLength, bytes.length);
  assert.equal(loadJsonArtifact(session, "authority.json"), artifact);
  const first = loadedJsonValue(artifact, root);
  first.nested.value = 9;
  assert.deepEqual(loadedJsonValue(artifact, root), { nested: { value: 1 } });
  assert.equal(assertSessionArtifact(session, artifact, "authority.json"), artifact);
  assert.deepEqual(revalidateObservedArtifacts(session), { artifacts: 1 });
});

test("relative paths reject absolute, traversal, alternate separators, and aliases below the root", () => {
  const rootPath = createTemporaryDirectory("runmat-authority-loading-");
  const real = path.join(rootPath, "real");
  fs.mkdirSync(real);
  fs.writeFileSync(path.join(real, "value.json"), "{}\n");
  fs.symlinkSync(
    real, path.join(rootPath, "alias"), process.platform === "win32" ? "junction" : "dir",
  );
  const session = openAuthorityLoadSession(openAuthorityRoot(rootPath));
  for (const candidate of [
    path.join(rootPath, "real/value.json"), "../value.json", "real/../value.json",
    "./real/value.json", "real\\value.json", "real//value.json", "C:/value.json",
    "c:value.json", "//server/share/value.json", "//?/C:/value.json", "bad\0value.json",
    "NUL/value.json", "area/COM1.json", "value.json:stream", "area./value.json",
  ]) {
    assert.throws(() => loadJsonArtifact(session, candidate), /canonical relative POSIX path/);
  }
  assert.throws(
    () => loadJsonArtifact(session, "alias/value.json"),
    /symbolic-link or non-directory ancestor/,
  );
  fs.writeFileSync(path.join(rootPath, "file-parent"), "not a directory\n");
  assert.throws(
    () => loadJsonArtifact(session, "file-parent/value.json"),
    /symbolic-link or non-directory ancestor/,
  );
});

test("symbolic-link and non-regular final artifacts are rejected", (context) => {
  const rootPath = createTemporaryDirectory("runmat-authority-loading-");
  const regular = path.join(rootPath, "regular.json");
  fs.writeFileSync(regular, "{}\n");
  const session = openAuthorityLoadSession(openAuthorityRoot(rootPath));
  fs.mkdirSync(path.join(rootPath, "directory.json"));
  assert.throws(() => loadJsonArtifact(session, "directory.json"), /real regular file/);
  if (process.platform !== "win32") {
    fs.symlinkSync(regular, path.join(rootPath, "alias.json"));
    assert.throws(() => loadJsonArtifact(session, "alias.json"), /real regular file/);
    const fifo = path.join(rootPath, "fifo.json");
    execFileSync("mkfifo", [fifo]);
    assert.throws(() => loadJsonArtifact(session, "fifo.json"), /real regular file/);
  } else {
    context.diagnostic("symbolic-link and FIFO creation are not portable for an unprivileged Windows test");
  }
});

test("JSON observation rejects malformed UTF-8 and malformed JSON", () => {
  const rootPath = createTemporaryDirectory("runmat-authority-loading-");
  fs.writeFileSync(path.join(rootPath, "utf8.json"), Buffer.from([0xc3, 0x28]));
  fs.writeFileSync(path.join(rootPath, "json.json"), "{\"missing\":}\n");
  const session = openAuthorityLoadSession(openAuthorityRoot(rootPath));
  assert.throws(() => loadJsonArtifact(session, "utf8.json"), /not valid UTF-8 JSON/);
  assert.throws(() => loadJsonArtifact(session, "json.json"), /not valid UTF-8 JSON/);
});

test("memoized observations stay pinned and final revalidation detects mutation", () => {
  const rootPath = createTemporaryDirectory("runmat-authority-loading-");
  const target = path.join(rootPath, "value.json");
  fs.writeFileSync(target, "{\"value\":1}\n");
  const root = openAuthorityRoot(rootPath);
  const session = openAuthorityLoadSession(root);
  const artifact = loadJsonArtifact(session, "value.json");
  fs.writeFileSync(target, "{\"value\":2}\n");
  assert.equal(loadJsonArtifact(session, "value.json"), artifact);
  assert.deepEqual(loadedJsonValue(artifact, root), { value: 1 });
  assert.throws(() => revalidateObservedArtifacts(session), /changed after observation/);
});

test("final revalidation detects replacement and disappearance", async (context) => {
  for (const operation of ["replace", "remove"]) {
    await context.test(operation, () => {
      const rootPath = createTemporaryDirectory("runmat-authority-loading-");
      const target = path.join(rootPath, "value.json");
      fs.writeFileSync(target, "{}\n");
      const session = openAuthorityLoadSession(openAuthorityRoot(rootPath));
      loadJsonArtifact(session, "value.json");
      if (operation === "replace") {
        fs.renameSync(target, path.join(rootPath, "prior.json"));
        fs.writeFileSync(target, "{}\n");
      } else fs.unlinkSync(target);
      const expected = operation === "replace" ? /changed after observation/ : /cannot revalidate/;
      assert.throws(() => revalidateObservedArtifacts(session), expected);
    });
  }
});

test("final revalidation detects replacement of the canonical root", () => {
  const parent = createTemporaryDirectory("runmat-authority-loading-");
  const rootPath = path.join(parent, "root");
  fs.mkdirSync(rootPath);
  const session = openAuthorityLoadSession(openAuthorityRoot(rootPath));
  fs.renameSync(rootPath, path.join(parent, "prior-root"));
  fs.mkdirSync(rootPath);
  assert.throws(() => revalidateObservedArtifacts(session), /root changed after observation/);
});

test("traversal detects active path and domain-semantic cycles and always releases scope", () => {
  const rootPath = createTemporaryDirectory("runmat-authority-loading-");
  const session = openAuthorityLoadSession(openAuthorityRoot(rootPath));
  withAuthorityTraversal(session, {
    domain: "queue-state", path: "states/current.json", semanticDigest: DIGEST_A,
  }, () => {
    assert.throws(() => withAuthorityTraversal(session, {
      domain: "queue-checkpoint", path: "states/current.json", semanticDigest: `sha256:${"b".repeat(64)}`,
    }, () => {}), /path cycle/);
    assert.throws(() => withAuthorityTraversal(session, {
      domain: "queue-state", path: "states/copy.json", semanticDigest: DIGEST_A,
    }, () => {}), /semantic authority cycle/);
  });
  assert.doesNotThrow(() => withAuthorityTraversal(session, {
    domain: "queue-state", path: "states/current.json", semanticDigest: DIGEST_A,
  }, () => {}));
});

test("root, session, and artifact capabilities cannot be forged or cloned", () => {
  const rootPath = createTemporaryDirectory("runmat-authority-loading-");
  fs.writeFileSync(path.join(rootPath, "value.json"), "{}\n");
  const root = openAuthorityRoot(rootPath);
  const session = openAuthorityLoadSession(root);
  const artifact = loadJsonArtifact(session, "value.json");
  assert.throws(() => assertAuthorityRoot({ path: root.path }), /exact authority root/);
  assert.throws(() => assertAuthorityLoadSession({ root }), /exact authority load session/);
  assert.throws(
    () => assertLoadedJsonArtifact({ ...artifact }, root),
    /exact loaded JSON authority artifact/,
  );
  const second = openAuthorityLoadSession(root);
  assert.throws(
    () => assertSessionArtifact(second, artifact),
    /does not belong to this session/,
  );
});
