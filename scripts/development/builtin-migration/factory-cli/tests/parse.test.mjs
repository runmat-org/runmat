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
    "verify", "--manifest", "verification.json", "--output", "result.json",
  ]);
  assert.equal(verify.command, "verify");
  assert.equal(verify.manifest, "verification.json");
  assert.equal(verify.output, "result.json");
});

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
