import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { evidenceDigest } from "../../evidence.mjs";
import { deriveModuleCompositionBaselineCandidate } from "../baseline-candidate.mjs";
import { freezeReviewedModuleCompositionBaseline } from "../baseline-authority.mjs";
import { buildModuleCompositionBaselineReviewTemplate, sealModuleCompositionBaselineReview } from "../baseline-review.mjs";
import { bootstrapModuleComposition } from "../bootstrap.mjs";
import { renderModuleCompositionProduct } from "../generate.mjs";
import { materializeEffectiveModuleComposition } from "../materialize.mjs";
import { moduleCompositionProductRegistry } from "../registry.mjs";
import {
  inspectCompositionRepository,
  inspectMaterializationTargets,
} from "../repository-state.mjs";
import { installCompositionSet, renderCompositionSetTwice, withCompositionTransaction } from "../transaction.mjs";
import { markTransactionCommitted, writeTransactionJournal } from "../transaction-journal.mjs";
import { cleanupRepositoryFixtures, controlledFixture } from "../../tests/helpers.mjs";

test.afterEach(() => cleanupRepositoryFixtures());

test("bootstrap audits and atomically materializes the complete present product set", () => withRepository(({ root, projection, baseline }) => {
  const result = bootstrap(root, baseline);
  assert.deepEqual(result.installed.map((entry) => entry.product_id), ["runtime-math", "runtime-root"]);
  for (const id of ["runtime-math", "runtime-root"]) {
    const product = projection.products.find((entry) => entry.product_id === id);
    assert.equal(fs.readFileSync(path.join(root, product.path), "utf8"), renderModuleCompositionProduct(product));
  }
  assert.equal(fs.existsSync(path.join(root, "crates/runmat-builtins/src/catalog/entries/argument_validation/mod.rs")), false);
}));

test("bootstrap rejects unsupported handwritten behavior", () => withRepository(({ root, projection, baseline }) => {
  write(root, product(projection, "runtime-math").path, "mod alpha;\nfn hidden_policy() {}\n");
  assert.throws(() => bootstrap(root, baseline), /unsupported handwritten Rust syntax/);
}));

test("bootstrap rejects missing and unexpected direct children", () => withRepository(({ root, projection, baseline }) => {
  write(root, product(projection, "runtime-math").path, "mod beta;\n");
  assert.throws(() => bootstrap(root, baseline), /deterministic source observation|resolve exactly once/);
}));

test("repository inspection keeps product state orthogonal to reviewed children and rejects unreviewed sources", () => {
  withRepository(({ root, projection }) => {
    fs.unlinkSync(path.join(root, product(projection, "runtime-math").path));
    const runtimeMath = product(projection, "runtime-math");
    const inventory = inspectCompositionRepository(root, [runtimeMath]);
    assert.equal(
      inventory.products.find((entry) => entry.product_id === "runtime-math").state,
      "absent",
    );
  });
  withRepository(({ root, projection, baseline }) => {
    const absent = product(projection, "catalog-argument-validation");
    write(root, path.posix.join(path.posix.dirname(absent.path), "orphan.rs"), "// orphan\n");
    assert.throws(() => bootstrap(root, baseline), /unreviewed direct module source/);
  });
});

test("repository inspection binds declarations to reviewed children by module identity", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-declaration-order-"));
  try {
    const definition = moduleCompositionProductRegistry()
      .find((entry) => entry.product_id === "runtime-math");
    const alpha = {
      ...child("alpha", "file", "crates/runmat-runtime/src/builtins/math/ops/alpha.rs"),
      declaration_order: 1,
    };
    const beta = {
      ...child("beta", "file", "crates/runmat-runtime/src/builtins/math/ops/beta.rs"),
      declaration_order: 0,
    };
    const product = {
      ...structuredClone(definition), state: "present", children: [alpha, beta],
    };
    write(root, product.path, [
      "#[path = \"ops/beta.rs\"]", "pub mod beta;",
      "#[path = \"ops/alpha.rs\"]", "pub mod alpha;", "",
    ].join("\n"));
    write(root, alpha.source_path, "pub fn alpha() {}\n");
    write(root, beta.source_path, "pub fn beta() {}\n");
    const inventory = inspectCompositionRepository(root, [product]);
    assert.deepEqual(
      inventory.products[0].children.map((entry) => entry.module),
      ["beta", "alpha"],
    );
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("bootstrap rejects source-kind mismatches and file-directory collisions", () => {
  withRepository(({ root, projection, baseline }) => {
    const child = product(projection, "runtime-math").children[0];
    fs.rmSync(path.join(root, path.dirname(child.source_path)), { recursive: true });
    write(root, path.posix.join(path.posix.dirname(product(projection, "runtime-math").path), "alpha"), "not a directory\n");
    assert.throws(() => bootstrap(root, baseline), /deterministic source observation|resolve exactly once/);
  });
  withRepository(({ root, projection, baseline }) => {
    const parent = path.posix.dirname(product(projection, "runtime-math").path);
    write(root, `${parent}/alpha.rs`, "// collision\n");
    assert.throws(() => bootstrap(root, baseline), /resolve exactly once/);
  });
});

test("bootstrap rejects nondeterministic output and staged-byte tampering", () => {
  withRepository(({ root, projection, baseline }) => {
    let calls = 0;
    assert.throws(() => bootstrap(root, baseline, {
      render(value) { calls += 1; return `${renderModuleCompositionProduct(value)}${calls % 2 ? "" : "\n"}`; },
    }), /nondeterministic/);
  });
  withRepository(({ root, projection, baseline }) => {
    assert.throws(() => bootstrap(root, baseline, {
      installHooks: { afterStage({ temporary }) { fs.appendFileSync(temporary, "// tampered\n"); } },
    }), /staged composition bytes changed/);
    const directory = path.dirname(path.join(root, product(projection, "runtime-math").path));
    assert.equal(fs.readdirSync(directory).some((entry) => entry.includes(".runmat-stage-")), false);
  });
});

test("bootstrap repeats signed source authority after every staged product is verified", () => withRepository(({ root, projection, baseline }) => {
  const childPath = product(projection, "runtime-math").children[0].source_path;
  let changed = false;
  assert.throws(() => bootstrap(root, baseline, {
    installHooks: {
      afterStage() {
        if (changed) return;
        changed = true;
        fs.appendFileSync(path.join(root, childPath), "// changed after staging\n");
      },
    },
  }), /differs from deterministic reconstruction|clean source HEAD/);
  for (const definition of [product(projection, "runtime-math"), product(projection, "runtime-root")]) {
    const directory = path.dirname(path.join(root, definition.path));
    assert.equal(fs.readdirSync(directory).some((entry) => entry.includes(".runmat-stage-")), false);
  }
}));

test("bootstrap restores every earlier product after a mid-install failure", () => withRepository(({ root, projection, baseline }) => {
  const originals = new Map(["runtime-math", "runtime-root"].map((id) => {
    const sourcePath = product(projection, id).path;
    return [sourcePath, fs.readFileSync(path.join(root, sourcePath), "utf8")];
  }));
  assert.throws(() => bootstrap(root, baseline, {
    installHooks: { beforeInstall({ index }) { if (index === 1) throw new Error("injected install failure"); } },
  }), /injected install failure/);
  for (const [sourcePath, contents] of originals) assert.equal(fs.readFileSync(path.join(root, sourcePath), "utf8"), contents);
}));

test("bootstrap rejects a product ancestor that escapes through a symlink", (context) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-root-"));
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-outside-"));
  context.after(() => { fs.rmSync(root, { recursive: true, force: true }); fs.rmSync(outside, { recursive: true, force: true }); });
  fs.mkdirSync(path.join(root, "crates/runmat-runtime/src"), { recursive: true });
  fs.symlinkSync(outside, path.join(root, "crates/runmat-runtime/src/builtins"));
  assert.throws(() => deriveModuleCompositionBaselineCandidate(root, SIGNER, sourceOptions()), /canonical Git worktree root|realpath|ENOENT|symbolic link/);
});

test("materialization rejects staged symlinks and concurrent lock acquisition", () => {
  withRepository(({ root, projection, baseline }) => {
    assert.throws(() => bootstrap(root, baseline, {
      installHooks: { afterStage({ temporary, target }) { fs.unlinkSync(temporary); fs.symlinkSync(target, temporary); } },
    }), /ELOOP|symbolic link|staged file/);
  });
  withRepository(({ root, baseline }) => {
    assert.throws(() => bootstrap(root, baseline, {
      afterAudit() { bootstrap(root, baseline); },
    }), /retains the repository process fence/);
  });
});

test("materialization rejects target and canonical ancestor drift after audit", () => {
  withRepository(({ root, projection, baseline }) => {
    assert.throws(() => bootstrap(root, baseline, {
      afterAudit() {
        const target = path.join(root, product(projection, "runtime-math").path);
        const bytes = fs.readFileSync(target);
        fs.renameSync(target, `${target}.prior`);
        fs.writeFileSync(target, bytes);
      },
    }), /before-state changed/);
  });
  withRepository(({ root, projection, baseline }) => {
    const directory = path.join(root, "crates/runmat-runtime/src/builtins/math");
    assert.throws(() => bootstrap(root, baseline, {
      afterAudit() { fs.renameSync(directory, `${directory}-prior`); fs.mkdirSync(directory); },
    }), /before-state changed|canonical regular file|canonical child source/);
  });
});

test("expected-absent installation never overwrites a racing target", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-absent-"));
  try {
    const definition = moduleCompositionProductRegistry().find((entry) => entry.product_id === "runtime-math");
    const value = { ...structuredClone(definition), state: "present", children: [child("alpha", "directory", "crates/runmat-runtime/src/builtins/math/alpha/mod.rs")] };
    write(root, value.children[0].source_path, "pub fn value() {}\n");
    withCompositionTransaction(root, (lock) => {
      const before = inspectMaterializationTargets(root, [value]);
      const rendered = renderCompositionSetTwice([value]);
      write(root, value.path, "// racing writer\n");
      assert.throws(() => installCompositionSet(root, rendered, before, lock), /before-state changed/);
      assert.equal(fs.readFileSync(path.join(root, value.path), "utf8"), "// racing writer\n");
    });
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("installation requires exact before-state coverage and cleans staging after authority failure", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-authority-"));
  try {
    const definition = moduleCompositionProductRegistry().find((entry) => entry.product_id === "runtime-math");
    const value = { ...structuredClone(definition), state: "present", children: [child("alpha", "directory", "crates/runmat-runtime/src/builtins/math/alpha/mod.rs")] };
    write(root, value.children[0].source_path, "pub fn value() {}\n");
    withCompositionTransaction(root, (lock) => {
      const before = inspectMaterializationTargets(root, [value]);
      const rendered = renderCompositionSetTwice([value]);
      const surplus = { ...before[0], product_id: "runtime-root", path: "crates/runmat-runtime/src/builtins/mod.rs" };
      assert.throws(
        () => installCompositionSet(root, rendered, [...before, surplus], lock),
        /before-state does not exactly cover/,
      );
      assert.throws(
        () => installCompositionSet(root, rendered, before, lock, {}, () => {
          throw new Error("authority changed");
        }),
        /authority changed/,
      );
      const directory = path.dirname(path.join(root, value.path));
      assert.equal(fs.readdirSync(directory).some((entry) => entry.includes(".runmat-stage-")), false);
      assert.equal(fs.existsSync(path.join(root, value.path)), false);
    });
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("committed cleanup failure is reported and recovered before the next transaction", () => withRepository(({ root, baseline }) => {
  const first = bootstrap(root, baseline, {
    installHooks: { beforeCleanup({ index }) { if (index === 0) throw new Error("injected cleanup failure"); } },
  });
  assert.equal(first.transaction.cleanup, "recovery-required");
  assert.deepEqual(first.transaction.cleanup_errors, ["injected cleanup failure"]);
  const second = bootstrap(root, baselineFor(root));
  assert.deepEqual(second.transaction.recovery, { recovered: true, phase: "committed" });
  assert.equal(second.transaction.cleanup, "complete");
}));

test("an interrupted installing journal rolls the complete set back before new work", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-recovery-"));
  try {
    const relative = "crates/runmat-runtime/src/builtins/math/mod.rs";
    const target = path.join(root, relative);
    const original = "// original\n";
    const replacement = "// replacement\n";
    write(root, relative, original);
    const stat = fs.statSync(target);
    const token = `999-${"a".repeat(24)}`;
    const backup = `${target}.runmat-backup-${token}`;
    fs.renameSync(target, backup);
    fs.writeFileSync(target, replacement);
    writeTransactionJournal(root, { schema_version: 2, kind: "runmat-module-composition-transaction", phase: "installing", registry_digest: registryDigest(), token, entries: [{
      product_id: "runtime-math", path: relative, existed: true,
      before_digest: sha256(original), before_file_identity: { device: String(stat.dev), inode: String(stat.ino) },
      desired_state: "present", new_digest: sha256(replacement),
    }] });
    withCompositionTransaction(root, (_lock, recovery) => {
      assert.deepEqual(recovery, { recovered: true, phase: "installing" });
      assert.equal(fs.readFileSync(target, "utf8"), original);
    });
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("recovery rejects symlink substitution for backups and committed targets", () => {
  const makeJournal = (root, relative, token, existed, before, replacement) => {
    const target = path.join(root, relative);
    const stat = fs.statSync(target);
    writeTransactionJournal(root, { schema_version: 2, kind: "runmat-module-composition-transaction", phase: "installing", registry_digest: registryDigest(), token, entries: [{
      product_id: "runtime-math", path: relative, existed,
      before_digest: existed ? sha256(before) : null,
      before_file_identity: existed ? { device: String(stat.dev), inode: String(stat.ino) } : null,
      desired_state: "present", new_digest: sha256(replacement),
    }] });
    return { target, stat };
  };
  const first = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-recovery-link-"));
  try {
    const relative = "crates/runmat-runtime/src/builtins/math/mod.rs";
    const original = "// original\n";
    const replacement = "// replacement\n";
    write(first, relative, original);
    const token = `999-${"b".repeat(24)}`;
    const { target } = makeJournal(first, relative, token, true, original, replacement);
    const external = path.join(first, "external-backup.rs");
    fs.writeFileSync(external, original);
    fs.unlinkSync(target);
    fs.writeFileSync(target, replacement);
    fs.symlinkSync(external, `${target}.runmat-backup-${token}`);
    assert.throws(() => withCompositionTransaction(first, () => {}), /not a regular recovery file/);
    assert.equal(fs.readFileSync(external, "utf8"), original);
  } finally { fs.rmSync(first, { recursive: true, force: true }); }

  const second = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-committed-link-"));
  try {
    const relative = "crates/runmat-runtime/src/builtins/math/mod.rs";
    const original = "// original\n";
    const replacement = "// replacement\n";
    write(second, relative, original);
    const token = `999-${"c".repeat(24)}`;
    const { target } = makeJournal(second, relative, token, true, original, replacement);
    const external = path.join(second, "external-target.rs");
    fs.writeFileSync(external, replacement);
    fs.unlinkSync(target);
    fs.symlinkSync(external, target);
    markTransactionCommitted(second, token);
    assert.throws(() => withCompositionTransaction(second, () => {}), /noncanonical repository ancestor|escapes the repository root/);
    assert.equal(fs.readFileSync(external, "utf8"), replacement);
  } finally { fs.rmSync(second, { recursive: true, force: true }); }
});

test("recovery rejects registry drift and duplicate products before mutation", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-recovery-registry-"));
  try {
    const relative = "crates/runmat-runtime/src/builtins/math/mod.rs";
    const target = path.join(root, relative);
    const original = "// original\n";
    const replacement = "// replacement\n";
    write(root, relative, replacement);
    const token = `999-${"d".repeat(24)}`;
    const entry = {
      product_id: "runtime-math", path: relative, existed: false,
      before_digest: null, before_file_identity: null,
      desired_state: "present", new_digest: sha256(replacement),
    };
    writeTransactionJournal(root, {
      schema_version: 2, kind: "runmat-module-composition-transaction", phase: "installing",
      registry_digest: `sha256:${"0".repeat(64)}`, token, entries: [entry],
    });
    assert.throws(() => withCompositionTransaction(root, () => {}), /recovery journal is invalid/);
    assert.equal(fs.readFileSync(target, "utf8"), replacement);
    fs.unlinkSync(path.join(root, ".runmat-module-composition.transaction.json"));
    writeTransactionJournal(root, {
      schema_version: 2, kind: "runmat-module-composition-transaction", phase: "installing",
      registry_digest: registryDigest(), token, entries: [entry, entry],
    });
    assert.throws(() => withCompositionTransaction(root, () => {}), /duplicate products/);
    assert.equal(fs.readFileSync(target, "utf8"), replacement);
    fs.unlinkSync(path.join(root, ".runmat-module-composition.transaction.json"));
    writeTransactionJournal(root, {
      schema_version: 2, kind: "runmat-module-composition-transaction", phase: "installing",
      registry_digest: registryDigest(), token,
      entries: [{ ...entry, path: "crates/runmat-runtime/src/builtins/mod.rs" }],
    });
    assert.throws(() => withCompositionTransaction(root, () => {}), /differs from the fixed registry/);
    assert.equal(fs.readFileSync(target, "utf8"), replacement);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("factory materialization requires the lease-base inventory authority", () => {
  const factory = path.resolve("scripts/development/builtin-migration-factory.mjs");
  const common = ["--control", "x", "--baseline-inventory", "x", "--lease", "x", "--state", "x", "--queue-checkpoint", "x", "--trusted-queue-checkpoint-digest", `sha256:${"1".repeat(64)}`,
    "--component-graph", "x", "--draft", "x", "--c01-c03-review", "x", "--c04-c05-review", "x", "--c06-c07-review", "x", "--reconciliation", "x", "--stability-corrections", "x", "--candidate", "x", "--attestation", "x", "--topology", "x", "--control-scaffold", "x", "--control-review-set", "x", "--control-candidate", "x", "--control-attestation", "x"];
  const result = spawnSync(process.execPath, [factory, "materialize-composition", ...common], { encoding: "utf8" });
  assert.equal(result.status, 2);
  assert.match(result.stderr, /requires --control, --lease-base-inventory/);
});

test("operational materialization derives authority and writes active plus baseline-only products as one set", () => {
  const fixture = controlledFixture({ composition: true });
  const result = materializeEffectiveModuleComposition({
    repository: fixture.repository, control: fixture.control,
    queueState: fixture.queueState, queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  });
  assert.deepEqual(result.product_ids, moduleCompositionProductRegistry().map((entry) => entry.product_id));
  assert.deepEqual(result.installed.map((entry) => entry.product_id), ["runtime-math"]);
  const effective = fixture.control.moduleComposition.transitions.get(fixture.bundleId).changes[0].after;
  assert.match(fs.readFileSync(path.join(fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs"), "utf8"), new RegExp(`mod ${effective.module};`));
});

test("operational materialization activates an absent reviewed parent", () => {
  const fixture = controlledFixture({
    composition: true, compositionTransition: "activate",
  });
  const target = path.join(
    fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs",
  );
  assert.equal(fs.existsSync(target), false);
  const result = materializeEffectiveModuleComposition({
    repository: fixture.repository, control: fixture.control,
    queueState: fixture.queueState, queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  });
  assert.equal(fs.existsSync(target), true);
  assert.deepEqual(
    result.installed.find((entry) => entry.product_id === "runtime-math"),
    {
      product_id: "runtime-math",
      path: "crates/runmat-runtime/src/builtins/math/mod.rs",
      state: "present",
    },
  );
});

test("operational materialization deactivates a present parent while retaining reviewed children", () => {
  const fixture = controlledFixture({
    composition: true, compositionTransition: "deactivate",
  });
  const target = path.join(
    fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs",
  );
  assert.equal(fs.existsSync(target), true);
  const result = materializeEffectiveModuleComposition({
    repository: fixture.repository, control: fixture.control,
    queueState: fixture.queueState, queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  });
  assert.equal(fs.existsSync(target), false);
  assert.deepEqual(
    result.installed.find((entry) => entry.product_id === "runtime-math"),
    {
      product_id: "runtime-math",
      path: "crates/runmat-runtime/src/builtins/math/mod.rs",
      state: "absent",
    },
  );
});

test("operational child composition while absent does not create its parent", () => {
  const fixture = controlledFixture({
    composition: true, compositionTransition: "stage-while-absent",
  });
  const target = path.join(
    fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs",
  );
  assert.equal(fs.existsSync(target), false);
  const result = materializeEffectiveModuleComposition({
    repository: fixture.repository, control: fixture.control,
    queueState: fixture.queueState, queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  });
  assert.equal(fs.existsSync(target), false);
  assert.equal(
    result.installed.some((entry) => entry.product_id === "runtime-math"),
    false,
  );
});

test("operational materialization rejects stale or unvalidated queue authority before writing", () => {
  const fixture = controlledFixture({ composition: true });
  const original = fs.readFileSync(path.join(fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs"), "utf8");
  assert.throws(() => materializeEffectiveModuleComposition({
    repository: fixture.repository, control: fixture.control,
    queueState: fixture.queueState.value, queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  }), /validated queue state/);
  assert.equal(fs.readFileSync(path.join(fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs"), "utf8"), original);
});

test("operational materialization rejects parent bytes outside the authorized prior projection", () => {
  const fixture = controlledFixture({ composition: true });
  const target = path.join(
    fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs",
  );
  fs.writeFileSync(target, "// unreviewed parent edit\n");
  assert.throws(() => materializeEffectiveModuleComposition({
    repository: fixture.repository, control: fixture.control,
    queueState: fixture.queueState, queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  }), /parent bytes differ from the canonical prior projection/);
  assert.equal(fs.readFileSync(target, "utf8"), "// unreviewed parent edit\n");
});

test("operational materialization rejects an unauthorized present parent", () => {
  const fixture = controlledFixture({ composition: true });
  const definition = moduleCompositionProductRegistry()
    .find((entry) => entry.product_id === "catalog-argument-validation");
  write(
    fixture.repository,
    definition.path,
    renderModuleCompositionProduct({
      ...structuredClone(definition), state: "present", children: [],
    }),
  );
  assert.throws(() => materializeEffectiveModuleComposition({
    repository: fixture.repository, control: fixture.control,
    queueState: fixture.queueState, queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  }), /parent presence differs from the authorized prior projection/);
});

test("operational authority rejects child storage changed after its initial audit", () => {
  const fixture = controlledFixture({ composition: true });
  const parent = path.join(
    fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs",
  );
  const original = fs.readFileSync(parent, "utf8");
  assert.throws(() => materializeEffectiveModuleComposition({
    repository: fixture.repository, control: fixture.control,
    queueState: fixture.queueState, queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  }, {
    afterAudit() {
      fs.appendFileSync(path.join(
        fixture.repository,
        "crates/runmat-runtime/src/builtins/math/basic/mod.rs",
      ), "// changed after audit\n");
    },
  }), /effective child storage changed during materialization/);
  assert.equal(fs.readFileSync(parent, "utf8"), original);
});

test("operational authority rejects child disappearance after staging", () => {
  const fixture = controlledFixture({ composition: true });
  const parent = path.join(
    fixture.repository, "crates/runmat-runtime/src/builtins/math/mod.rs",
  );
  const original = fs.readFileSync(parent, "utf8");
  let removed = false;
  assert.throws(() => materializeEffectiveModuleComposition({
    repository: fixture.repository, control: fixture.control,
    queueState: fixture.queueState, queueCheckpoint: fixture.queueCheckpoint,
    lease: fixture.lease,
  }, {
    installHooks: {
      afterStage() {
        if (removed) return;
        removed = true;
        fs.unlinkSync(path.join(
          fixture.repository,
          "crates/runmat-runtime/src/builtins/math/basic/mod.rs",
        ));
      },
    },
  }), /reviewed directory source is not a canonical regular file/);
  assert.equal(fs.readFileSync(parent, "utf8"), original);
  assert.equal(
    fs.existsSync(path.join(
      fixture.repository, ".runmat-module-composition.transaction.json",
    )),
    false,
  );
});

function withRepository(run) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-composition-bootstrap-"));
  const projection = fixedProjection();
  try {
    write(root, product(projection, "runtime-root").path, "pub mod math;\n");
    write(root, product(projection, "runtime-math").path, "pub mod alpha;\n");
    write(root, product(projection, "runtime-math").children[0].source_path, "pub fn value() -> i32 { 1 }\n");
    return run({ root, projection, baseline: baselineFor(root) });
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
}

const SIGNER = "0123456789ABCDEF0123456789ABCDEF01234567";

function baselineFor(root) {
  const candidate = deriveModuleCompositionBaselineCandidate(root, SIGNER, sourceOptions());
  const review = buildModuleCompositionBaselineReviewTemplate(candidate);
  for (const role of review.roles) role.role = "group";
  review.review = { status: "reviewed", evidence: ["fixture review"] };
  return freezeReviewedModuleCompositionBaseline(candidate, sealModuleCompositionBaselineReview(review, candidate));
}

function bootstrap(root, baseline, options = {}) {
  return bootstrapModuleComposition({
    repository: root, reviewedBaseline: baseline,
    trustedReviewedBaselineDigest: baseline.digest,
    trustedSignerFingerprint: SIGNER,
  }, { ...options, sourceObservation: sourceOptions(true) });
}

function sourceOptions(allowCompositionLock = false) {
  return { allowCompositionLock, git(root, arguments_) {
    const command = arguments_.join(" ");
    if (command === "rev-parse --show-toplevel") return root;
    if (command === "rev-parse HEAD") return `${"a".repeat(40)}\n`;
    if (command === "rev-parse HEAD^{tree}") return `${"b".repeat(40)}\n`;
    if (command.startsWith("status ") || command === "verify-commit HEAD") return "";
    if (command.startsWith(`ls-tree -z ${"a".repeat(40)} -- `)) {
      const sourcePath = command.slice(`ls-tree -z ${"a".repeat(40)} -- `.length);
      const mode = fs.statSync(path.join(root, sourcePath)).mode & 0o111 ? "100755" : "100644";
      return `${mode} blob ${"c".repeat(40)}\t${sourcePath}\0`;
    }
    if (command.startsWith(`ls-tree -z ${"a".repeat(40)} -- `)) {
      const sourcePath = command.slice(`ls-tree -z ${"a".repeat(40)} -- `.length);
      return `100644 blob ${"c".repeat(40)}\t${sourcePath}\0`;
    }
    if (command.startsWith(`show ${"a".repeat(40)}:`)) {
      return fs.readFileSync(path.join(root, command.slice(`show ${"a".repeat(40)}:`.length)), "utf8");
    }
    if (command.startsWith("log -1 ")) return `G\0${SIGNER}\0${SIGNER}\n`;
    throw new Error(`unexpected git command ${command}`);
  } };
}

function fixedProjection() {
  const children = new Map([
    ["runtime-root", [child("math", "directory", "crates/runmat-runtime/src/builtins/math/mod.rs")]],
    ["runtime-math", [child("alpha", "directory", "crates/runmat-runtime/src/builtins/math/alpha/mod.rs")]],
  ]);
  return {
    schema_version: 5, kind: "runmat-builtin-module-composition-projection",
    products: moduleCompositionProductRegistry().map((entry) => {
      const productChildren = children.get(entry.product_id) ?? [];
      return {
        ...structuredClone(entry),
        state: productChildren.length ? "present" : "absent",
        children: productChildren,
      };
    }),
  };
}

function child(module, sourceKind, sourcePath) {
  return { module, source_kind: sourceKind, source_path: sourcePath, role: "group", visibility: "public", declaration_condition: { kind: "always" }, declaration_order: 0, macro_use: false, reexports: [], aggregation_sources: [] };
}
function product(projection, id) { return projection.products.find((entry) => entry.product_id === id); }
function sha256(value) { return `sha256:${crypto.createHash("sha256").update(value).digest("hex")}`; }
function registryDigest() { return evidenceDigest(moduleCompositionProductRegistry()); }
function write(root, relative, contents) { const target = path.join(root, relative); fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, contents); }
