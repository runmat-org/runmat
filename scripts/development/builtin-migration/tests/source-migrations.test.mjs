import assert from "node:assert/strict";
import test from "node:test";

import { subjectAuthorityPathFailures } from "../authority-paths.mjs";
import { validateMigrationPhasePaths } from "../integration-phases.mjs";
import { renderModuleCompositionProduct } from "../module-composition/generate.mjs";
import { moduleCompositionProductRegistry } from "../module-composition/registry.mjs";
import {
  parseSourceMigrationEvidence, parseSourceMigrations, validateSourceMigrationControl,
} from "../source-migrations.mjs";

const DIGEST = `sha256:${"a".repeat(64)}`;
const ROOT = "crates/runmat-runtime/src/builtins/datetime/calendar_duration";
const SOURCE = `${ROOT}.rs`;
const PROMOTED = `${ROOT}/mod.rs`;
const SUPPORT = `${ROOT}/support.rs`;
const IDENTITY = `${ROOT}/calendar_duration.rs`;
const PRODUCT_ID = "runtime-datetime-calendar-duration";
const PUBLIC_GLOB = Object.freeze({
  kind: "glob", visibility: "public", condition: { kind: "always" }, doc_hidden: false,
});

test("all reviewed deep promoted targets have paired composition products", () => {
  const ids = new Set(moduleCompositionProductRegistry().map(({ product_id: id }) => id));
  for (const id of [
    "runtime-datetime-business-calendar",
    "runtime-datetime-calendar-duration",
    "runtime-datetime-conversion",
    "runtime-deep-learning-autodiff",
    "runtime-geometry-triangulation",
    "runtime-timing-timer",
  ]) assert.ok(ids.has(id), `missing ${id}`);
});

test("authored support reparent binds an explicit support subset without identity authority", () => {
  const migration = supportReparent({ registered: false });
  assert.deepEqual(parseSourceMigrations(
    [migration], "bundle-example", [], inventory(), [],
  ), [migration]);
  assert.doesNotThrow(() => validateSourceMigrationControl(
    bundles(migration), new Map(), new Map(), moduleComposition(),
  ));
});

test("registered calendar-duration reparent preserves its exact reviewed glob export", () => {
  const migration = supportReparent();
  const product = compositionProduct();
  assert.doesNotThrow(() => validateSourceMigrationControl(
    bundles(migration), new Map(), integrationProducts(), moduleComposition([product]),
  ));

  const definition = moduleCompositionProductRegistry()
    .find((entry) => entry.product_id === PRODUCT_ID);
  const rendered = renderModuleCompositionProduct({
    ...structuredClone(definition), state: "present", children: product.children,
  });
  assert.match(rendered, /mod support;\n\npub use support::\*;/);
});

test("decomposition can retain typed support and exact identity destinations", () => {
  const migration = decomposition();
  const parsed = parseSourceMigrations(
    [migration], "bundle-example", ["calendar_duration"], inventory(), baselineEvidence(),
  );
  assert.deepEqual(parsed, [migration]);
  assert.doesNotThrow(() => validateSourceMigrationControl(
    bundles(migration), identityControls(), integrationProducts(),
    moduleComposition([compositionProduct()]),
  ));
});

test("rejects malformed, stale, inferred, and ambiguous migration authority", () => {
  const cases = [
    ["unknown strategy", (migration) => { migration.strategy = "copy"; }, /must be one of/],
    ["stale source digest", (migration) => { migration.source_baseline_digest = `sha256:${"b".repeat(64)}`; }, /differs from the exact baseline source/],
    ["existing destination", (migration, input) => { input.source.files.push({ path: IDENTITY, content_digest: DIGEST }); }, /destination exists at the baseline/],
    ["missing decomposition ownership", (migration) => {
      migration.destination_paths = [SUPPORT];
      migration.identity_destinations = [];
    }, /requires identity destinations/],
    ["identity outside bundle", (migration) => { migration.identity_destinations[0].identity = "other"; }, /must belong to the bundle/],
    ["noncanonical destinations", (migration) => { migration.destination_paths.reverse(); }, /canonically ordered/],
    ["duplicate destinations", (migration) => { migration.destination_paths = [IDENTITY, IDENTITY]; }, /unique and canonically ordered/],
    ["case-colliding endpoint", (migration) => { migration.destination_paths = [SOURCE.toUpperCase()]; }, /case-insensitively/],
    ["unreviewed decision", (migration) => { migration.review.status = "unreviewed"; }, /status must be reviewed/],
  ];
  for (const [name, mutate, expected] of cases) {
    const migration = decomposition();
    const input = inventory();
    mutate(migration, input);
    assert.throws(
      () => parseSourceMigrations(
        [migration], "bundle-example", ["calendar_duration"], input, baselineEvidence(),
      ), expected, name,
    );
  }
});

test("support and identity ownership are typed against exact migration destinations", () => {
  const missingSupport = decomposition();
  missingSupport.support_destinations = [];
  assert.throws(
    () => parseSourceMigrationEvidence([missingSupport]),
    /requires a typed support destination/,
  );

  const untyped = decomposition();
  untyped.destination_paths.push(`${ROOT}/untyped.rs`);
  untyped.destination_paths.sort();
  assert.throws(
    () => parseSourceMigrationEvidence([untyped]),
    /not exactly partitioned/,
  );

  const absent = decomposition();
  absent.support_destinations[0].destination_path = `${ROOT}/absent.rs`;
  assert.throws(() => parseSourceMigrationEvidence([absent]), /absent from destination paths/);

  const metadataOnMod = supportReparent();
  metadataOnMod.destination_paths = [PROMOTED, SUPPORT].sort();
  metadataOnMod.support_destinations[0].destination_path = PROMOTED;
  assert.throws(
    () => validateSourceMigrationControl(
      bundles(metadataOnMod), new Map(), integrationProducts(),
      moduleComposition([compositionProduct()]),
    ),
    /effective composition product|integration-owned|not an exact support child/,
  );
});

test("reviewed internal identities may remain owned by the shared support child", () => {
  const migration = decomposition();
  migration.identity_destinations[0].destination_paths = [SUPPORT];
  const identities = identityControls();
  identities.get("calendar_duration").implementation.callable.owner_path = SUPPORT;
  identities.get("calendar_duration").public_identity = { kind: "internal" };
  assert.doesNotThrow(() => validateSourceMigrationControl(
    bundles(migration), identities, integrationProducts(),
    moduleComposition([compositionProduct()]),
  ));
});

test("public identities cannot remain hidden in a migrated support child", () => {
  const migration = decomposition();
  migration.identity_destinations[0].destination_paths = [SUPPORT];
  const identities = identityControls();
  identities.get("calendar_duration").implementation.callable.owner_path = SUPPORT;
  assert.throws(
    () => validateSourceMigrationControl(
      bundles(migration), identities, integrationProducts(),
      moduleComposition([compositionProduct()]),
    ),
    /only an internal identity may remain owned by migrated support/,
  );
});

test("a support reparent cannot absorb baseline identity authority", () => {
  const migration = supportReparent();
  migration.destination_paths = [IDENTITY, SUPPORT];
  migration.identity_destinations = [{
    identity: "calendar_duration", destination_paths: [IDENTITY],
  }];
  assert.throws(
    () => parseSourceMigrations(
      [migration], "bundle-example", ["calendar_duration"], inventory(), baselineEvidence(),
    ),
    /support reparent cannot claim identity destinations/,
  );
});

test("registered and authored promoted targets fail closed against composition authority", () => {
  const registered = supportReparent();
  assert.throws(
    () => validateSourceMigrationControl(
      bundles(registered), new Map(), new Map(), moduleComposition(),
    ), /absent from exact integration-product authority/,
  );
  assert.throws(
    () => validateSourceMigrationControl(
      bundles(registered), new Map(), integrationProducts(), moduleComposition(),
    ), /absent from effective composition/,
  );

  const authored = supportReparent({ registered: false });
  assert.throws(
    () => validateSourceMigrationControl(
      bundles(authored), new Map(), integrationProducts(),
      moduleComposition([compositionProduct()]),
    ), /effective composition product|authored promoted target is registered/,
  );
});

test("identity ownership maps exactly to final callable and constant owner paths", () => {
  const migration = decomposition();
  const wrong = identityControls();
  wrong.get("calendar_duration").implementation.callable.owner_path = `${ROOT}/wrong.rs`;
  assert.throws(
    () => validateSourceMigrationControl(
      bundles(migration), wrong, integrationProducts(), moduleComposition([compositionProduct()]),
    ), /differ from final owned callable\/constant paths/,
  );
  const missing = new Map();
  assert.throws(
    () => validateSourceMigrationControl(
      bundles(migration), missing, integrationProducts(), moduleComposition([compositionProduct()]),
    ), /lacks exact bundle identity authority/,
  );
});

test("registered support metadata must match exact effective support children", () => {
  const migration = supportReparent();
  const product = compositionProduct();
  product.children.find((child) => child.role === "support").reexports = [];
  assert.throws(
    () => validateSourceMigrationControl(
      bundles(migration), new Map(), integrationProducts(), moduleComposition([product]),
    ), /reexports differ/,
  );
  const wrongRole = compositionProduct();
  wrongRole.children.find((child) => child.role === "support").role = "identity";
  assert.throws(
    () => validateSourceMigrationControl(
      bundles(migration), new Map(), integrationProducts(), moduleComposition([wrongRole]),
    ), /do not exactly cover migrated support children/,
  );
});

test("the migration-owning bundle must introduce every registered target child", () => {
  const migration = decomposition();
  const controlled = bundles(migration);
  controlled.get("bundle-example").integration_product_refs = [];
  assert.throws(
    () => validateSourceMigrationControl(
      controlled, identityControls(), integrationProducts(),
      moduleComposition([compositionProduct()]),
    ),
    /not referenced by its source-migration bundle/,
  );

  const missingChild = bundles(migration);
  missingChild.get("bundle-example").module_composition_transition.changes =
    missingChild.get("bundle-example").module_composition_transition.changes
      .filter((change) => change.after?.source_path !== IDENTITY);
  assert.throws(
    () => validateSourceMigrationControl(
      missingChild, identityControls(), integrationProducts(),
      moduleComposition([compositionProduct()]),
    ),
    /migrated identity is not introduced by its exact bundle transition/,
  );
});

test("globally rejects reused endpoints, including migration chains and cycles", () => {
  const first = bundle(supportReparent({ registered: false }), "bundle-first");
  const secondMigration = supportReparent({ registered: false });
  secondMigration.source_path = SUPPORT;
  secondMigration.source_baseline_digest = DIGEST;
  const second = bundle(secondMigration, "bundle-second");
  second.module_composition_transition.changes[0].before.source_path = SUPPORT;
  const controlled = new Map([[first.id, first], [second.id, second]]);
  assert.throws(
    () => validateSourceMigrationControl(controlled, new Map(), new Map(), moduleComposition()),
    /already owned/,
  );
});

test("phase and final-state evidence require every reviewed migration endpoint", () => {
  const controlled = bundle(supportReparent({ registered: false }), "bundle-example");
  controlled.integration_outputs = [];
  assert.doesNotThrow(() => validateMigrationPhasePaths(
    controlled, [SOURCE, PROMOTED, SUPPORT], [],
  ));
  assert.throws(
    () => validateMigrationPhasePaths(controlled, [SUPPORT], []),
    /did not change reviewed source migration endpoints/,
  );

  const control = { bundles: new Map([[controlled.id, controlled]]), identities: new Map() };
  assert.deepEqual(subjectAuthorityPathFailures(control, {
    source: { files: [{ path: PROMOTED }, { path: SUPPORT }] }, identities: [],
  }, [controlled.id]), []);
});

function decomposition() {
  return {
    ...baseMigration(),
    strategy: "module-decomposition",
    destination_paths: [IDENTITY, SUPPORT],
    support_destinations: [{ destination_path: SUPPORT, reexports: [PUBLIC_GLOB] }],
    identity_destinations: [{
      identity: "calendar_duration", destination_paths: [IDENTITY],
    }],
    reason: "Split identity authority and retained support beneath the registered module",
  };
}

function supportReparent({ registered = true } = {}) {
  return {
    ...baseMigration(),
    strategy: "module-support-reparent",
    promoted_target: registered
      ? { kind: "registered", path: PROMOTED, product_id: PRODUCT_ID }
      : { kind: "authored", path: PROMOTED },
    destination_paths: registered ? [SUPPORT] : [PROMOTED, SUPPORT],
    support_destinations: [{ destination_path: SUPPORT, reexports: [PUBLIC_GLOB] }],
    identity_destinations: [],
    reason: "Reparent reviewed shared support beneath its promoted module",
  };
}

function baseMigration() {
  return {
    strategy: "module-decomposition",
    source_path: SOURCE,
    source_baseline_digest: DIGEST,
    promoted_target: { kind: "registered", path: PROMOTED, product_id: PRODUCT_ID },
    destination_paths: [IDENTITY],
    support_destinations: [],
    identity_destinations: [],
    reason: "Reviewed source migration",
    review: { status: "reviewed", evidence: ["review:fixture source migration"] },
  };
}

function inventory() {
  return { source: { files: [{ path: SOURCE, content_digest: DIGEST }] } };
}

function baselineEvidence() {
  return [{
    path: SOURCE,
    source_snapshot: "present",
    content_digest: DIGEST,
    affected_identities: ["calendar_duration"],
  }];
}

function identityControls() {
  return new Map([["calendar_duration", {
    bundle_id: "bundle-example",
    public_identity: { kind: "primary" },
    implementation: {
      callable: { kind: "owned", owner_path: IDENTITY, bindings: [] },
      constant: { kind: "none", reason: "no-constant-form" },
    },
  }]]);
}

function bundles(migration) {
  const value = bundle(migration, "bundle-example");
  return new Map([[value.id, value]]);
}

function bundle(migration, id) {
  const targetChildren = [
    {
      product_id: PRODUCT_ID,
      operation: "add",
      before: null,
      after: compositionProduct().children.find((child) => child.source_path === SUPPORT),
    },
    ...migration.identity_destinations.flatMap(({ destination_paths: paths }) => paths
      .filter((destination) => destination !== SUPPORT)
      .map((destination) => ({
        product_id: PRODUCT_ID,
        operation: "add",
        before: null,
        after: compositionProduct().children.find((child) => child.source_path === destination)
          ?? { source_path: destination, role: "identity" },
      }))),
  ];
  return {
    id,
    integration_product_refs: migration.promoted_target.kind === "registered" ? [PRODUCT_ID] : [],
    authored_write_set: [
      { kind: "tree", path: ROOT },
      { kind: "file", path: migration.source_path },
    ],
    expected_removals: [],
    source_migrations: [structuredClone(migration)],
    module_composition_transition: {
      product_states: migration.promoted_target.kind === "registered"
        ? [{ product_id: PRODUCT_ID, before_state: "absent", after_state: "present" }]
        : [],
      changes: [{
        operation: "replace",
        before: { source_kind: "file", source_path: migration.source_path },
        after: { source_kind: "directory", source_path: PROMOTED },
      }, ...(migration.promoted_target.kind === "registered" ? targetChildren : [])],
    },
  };
}

function integrationProducts() {
  return new Map([[PRODUCT_ID, {
    product_id: PRODUCT_ID,
    path: PROMOTED,
    verification: { kind: "rust_module_composition" },
  }]]);
}

function compositionProduct() {
  return {
    product_id: PRODUCT_ID,
    path: PROMOTED,
    children: [
      {
        module: "calendar_duration",
        source_kind: "file",
        source_path: IDENTITY,
        role: "identity",
        visibility: "private",
        declaration_condition: { kind: "always" },
        declaration_order: 0,
        macro_use: false,
        reexports: [],
        aggregation_sources: [],
      },
      {
        module: "support",
        source_kind: "file",
        source_path: SUPPORT,
        role: "support",
        visibility: "private",
        declaration_condition: { kind: "always" },
        declaration_order: 1,
        macro_use: false,
        reexports: [PUBLIC_GLOB],
        aggregation_sources: [],
      },
    ],
  };
}

function moduleComposition(products = []) {
  return { effective: { products } };
}
