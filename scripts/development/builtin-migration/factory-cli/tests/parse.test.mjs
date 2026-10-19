import assert from "node:assert/strict";
import test from "node:test";

import { factoryCliHelp } from "../help.mjs";
import { parseFactoryCliArguments } from "../parse.mjs";

test("factory help exposes representative and review-validation command contracts", () => {
  const help = factoryCliHelp();
  assert.match(help, /^Usage:\n/);
  assert.match(help, /builtin-migration-factory\.mjs inventory\|seed-dispositions/);
  assert.match(help, /builtin-migration-factory\.mjs prepare NAME/);
  assert.match(help, /validate-global-control-review .*--global-review PATH/);
  assert.match(
    help,
    /validate-bundle-control-review .*--bundle-review PATH --bundle ID/,
  );
  assert.match(help, /pilot-session-start .*--authority-root PATH/);
  assert.match(help, /initialize-queue .*--initial-queue-review PATH/);
  assert.match(help, /pilot-transition .*--evaluation PATH --evaluation-digest SHA256/);
  assert.match(help, /never production authority/);
  assert.deepEqual(parseFactoryCliArguments(["--help"]), { help: true });
  assert.deepEqual(parseFactoryCliArguments(["invalid", "-h"]), { help: true });
});

test("factory parser rejects invalid commands, missing values, and unknown options", () => {
  assert.throws(
    () => parseFactoryCliArguments([]),
    /expected inventory, queue, seed-dispositions.*use --help/,
  );
  assert.throws(
    () => parseFactoryCliArguments(["not-a-command"]),
    /expected inventory, queue, seed-dispositions.*use --help/,
  );
  assert.throws(
    () => parseFactoryCliArguments(["inventory"]),
    /inventory requires --compiled-inventory/,
  );
  assert.throws(
    () => parseFactoryCliArguments(["inventory", "--compiled-inventory"]),
    /--compiled-inventory requires a value/,
  );
  assert.throws(
    () => parseFactoryCliArguments([
      "inventory", "--compiled-inventory", "inventory.json", "--invented", "value",
    ]),
    /unknown option --invented/,
  );
});

test("factory parser preserves representative command and positional contracts", () => {
  const inventory = parseFactoryCliArguments([
    "inventory", "--compiled-inventory", "compiled.json",
    "--dispositions", "dispositions.json", "--output", "inventory.json",
  ]);
  assert.equal(inventory.command, "inventory");
  assert.equal(inventory.compiledInventory, "compiled.json");
  assert.equal(inventory.dispositions, "dispositions.json");
  assert.equal(inventory.output, "inventory.json");

  const verify = parseFactoryCliArguments([
    "verify", "--manifest", "verification.json", ...controlAuthorityArguments(),
    "--output", "result.json",
  ]);
  assert.equal(verify.command, "verify");
  assert.equal(verify.manifest, "verification.json");
  assert.equal(verify.output, "result.json");
});

function controlAuthorityArguments() {
  return [
    "--control", "control.json", "--baseline-inventory", "inventory.json",
    "--topology", "topology.json", "--candidate", "topology-candidate.json",
    "--attestation", "topology-attestation.json", "--component-graph", "graph.json",
    "--draft", "draft.json", "--c01-c03-review", "c01-c03.json",
    "--c04-c05-review", "c04-c05.json", "--c06-c07-review", "c06-c07.json",
    "--reconciliation", "reconciliation.json",
    "--stability-corrections", "stability.json",
    "--control-scaffold", "scaffold.json", "--control-review-set", "reviews.json",
    "--control-candidate", "control-candidate.json",
    "--control-attestation", "control-attestation.json",
  ];
}

test("prepare requires its positional identity before option validation", () => {
  assert.throws(
    () => parseFactoryCliArguments(["prepare"]),
    /prepare identity requires a value/,
  );
  assert.throws(
    () => parseFactoryCliArguments(["prepare", "--compiled-inventory", "compiled.json"]),
    /prepare identity requires a value/,
  );
});

test("review validators reject repeated and unrelated recognized options", () => {
  assert.throws(
    () => parseFactoryCliArguments([
      "validate-global-control-review",
      "--baseline-inventory", "one.json",
      "--baseline-inventory", "two.json",
    ]),
    /does not accept repeated options/,
  );
  assert.throws(
    () => parseFactoryCliArguments([
      "validate-bundle-control-review", "--request", "request.json",
    ]),
    /does not accept --request/,
  );
});

test("pilot lifecycle commands require exact authority references", () => {
  assert.throws(
    () => parseFactoryCliArguments([
      "pilot-evaluate", ...controlAuthorityArguments(),
    ]),
    /requires --authority-root, --measurement, --measurement-digest/,
  );
  assert.throws(
    () => parseFactoryCliArguments([
      "pilot-evaluate", ...controlAuthorityArguments(),
      "--authority-root", "authority", "--measurement", "measurement.json",
      "--measurement-digest", "sha256:abc", "--limiter", "limiter.json",
    ]),
    /requires --limiter and --limiter-digest together/,
  );
  const transition = parseFactoryCliArguments([
    "pilot-transition", ...controlAuthorityArguments(),
    "--authority-root", "authority", "--evaluation", "evaluation.json",
    "--evaluation-digest", "sha256:abc",
  ]);
  assert.equal(transition.command, "pilot-transition");
  assert.equal(transition.authorityRoot, "authority");
});

test("initial queue publication requires its exact reviewed authority", () => {
  assert.throws(
    () => parseFactoryCliArguments(["initialize-queue", ...controlAuthorityArguments()]),
    /requires --authority-root, --initial-queue-review, and --initial-queue-review-digest/,
  );
  const parsed = parseFactoryCliArguments([
    "initialize-queue", ...controlAuthorityArguments(),
    "--authority-root", "authority", "--initial-queue-review", "review.json",
    "--initial-queue-review-digest", "sha256:abc",
  ]);
  assert.equal(parsed.command, "initialize-queue");
  assert.equal(parsed.initialQueueReview, "review.json");
  assert.throws(() => parseFactoryCliArguments([
    "initialize-queue", ...controlAuthorityArguments(),
    "--authority-root", "authority", "--initial-queue-review", "one.json",
    "--initial-queue-review", "two.json",
    "--initial-queue-review-digest", "sha256:abc",
  ]), /does not accept repeated options/);
  assert.throws(() => parseFactoryCliArguments([
    "initialize-queue", ...controlAuthorityArguments(),
    "--authority-root", "authority", "--initial-queue-review", "review.json",
    "--initial-queue-review-digest", "sha256:abc", "--request", "ignored.json",
  ]), /initialize-queue does not accept --request/);
});

test("all factory commands reject duplicate option spellings", () => {
  assert.throws(() => parseFactoryCliArguments([
    "inventory", "--compiled-inventory", "one.json",
    "--compiled-inventory", "two.json",
  ]), /inventory does not accept repeated options/);
});

test("materialization accepts only an explicit, typed product selection", () => {
  const common = [
    ...controlAuthorityArguments(), "--lease-base-inventory", "lease-base.json",
    "--lease", "lease.json", "--state", "state.json",
    "--queue-checkpoint", "checkpoint.json",
    "--trusted-queue-checkpoint-digest", "sha256:abc",
  ];
  const selected = parseFactoryCliArguments([
    "materialize-composition", "--product", "catalog-array-creation",
    "--product", "runtime-array-creation", ...common,
  ]);
  assert.deepEqual(selected.productIds, [
    "catalog-array-creation", "runtime-array-creation",
  ]);
  assert.equal(selected.noProducts, false);

  const empty = parseFactoryCliArguments([
    "materialize-composition", "--no-products", ...common,
  ]);
  assert.deepEqual(empty.productIds, []);
  assert.equal(empty.noProducts, true);

  const authority = parseFactoryCliArguments([
    "materialize-composition", "--authority-products", ...common,
  ]);
  assert.deepEqual(authority.productIds, []);
  assert.equal(authority.authorityProducts, true);

  assert.throws(() => parseFactoryCliArguments([
    "materialize-composition", ...common,
  ]), /requires exactly one of --product, --no-products, or --authority-products/);
  assert.throws(() => parseFactoryCliArguments([
    "materialize-composition", "--no-products", "--product", "catalog-array-creation",
    ...common,
  ]), /requires exactly one of --product, --no-products, or --authority-products/);
  assert.throws(() => parseFactoryCliArguments([
    "materialize-composition", "--authority-products", "--product", "catalog-array-creation",
    ...common,
  ]), /requires exactly one of --product, --no-products, or --authority-products/);
  assert.throws(() => parseFactoryCliArguments([
    "materialize-composition", "--product", "catalog-array-creation",
    "--product", "catalog-array-creation", ...common,
  ]), /does not accept duplicate --product values/);
});

test("initial queue review options are rejected outside their owning command", () => {
  for (const option of ["--initial-queue-review", "--initial-queue-review-digest"]) {
    assert.throws(() => parseFactoryCliArguments([
      "inventory", "--compiled-inventory", "compiled.json", option, "ignored.json",
    ]), new RegExp(`${option} is not accepted by inventory`));
  }
});
