import assert from "node:assert/strict";
import test from "node:test";

import {
  observedIdentityForms, parseIdentityForms, parseImplementationAuthority, parsePublicIdentity,
  validateIdentityAuthorityGraph,
} from "../identity-authority.mjs";

test("one callable owner preserves multiple exact binding variants", () => {
  const owner = "crates/runmat-runtime/src/builtins/math/foo.rs";
  const callable = {
    kind: "owned",
    owner_path: owner,
    bindings: [binding("default"), binding("two-output")],
  };
  const implementation = implementationByForm(callable, absent("constant"));
  assert.equal(parseImplementationAuthority(implementation, "foo"), implementation);
  const repeatedOwner = structuredClone(implementation);
  repeatedOwner.callable.bindings[1].path = "crates/runmat-runtime/src/builtins/math/other.rs";
  assert.throws(() => parseImplementationAuthority(repeatedOwner, "foo"), /fields must be exactly/);
  const wrongSymbol = structuredClone(implementation);
  wrongSymbol.callable.bindings[0].native_symbol = "guessed";
  assert.throws(() => parseImplementationAuthority(wrongSymbol, "foo"), /exact binding identity/);
  const malformedFunction = structuredClone(implementation);
  malformedFunction.callable.bindings[0].function = "module::foo-builtin";
  assert.throws(() => parseImplementationAuthority(malformedFunction, "foo"), /Rust identifier/);
  const repeatedVariant = structuredClone(implementation);
  repeatedVariant.callable.bindings[1] = {
    ...repeatedVariant.callable.bindings[0],
    function: "other_builtin",
    builtin_path: "builtins::math::other",
  };
  repeatedVariant.callable.bindings.sort((left, right) => `${left.function}\0${left.variant}`.localeCompare(`${right.function}\0${right.variant}`));
  assert.throws(() => parseImplementationAuthority(repeatedVariant, "foo"), /semantic registration identity/);
});

test("public identity and form spelling records are closed and case-folded", () => {
  assert.doesNotThrow(() => parsePublicIdentity({
    kind: "primary",
    primary_spelling: { identity: "foo", spelling: "Foo" },
  }, "foo"));
  assert.throws(() => parseIdentityForms({
    kind: "callable", callable_spellings: ["other"], constant_spellings: [],
  }, "foo"), /owns the form/);
  assert.throws(() => parseIdentityForms({
    kind: "callable", callable_spellings: ["foo", "foo"], constant_spellings: [],
  }, "foo"), /must be unique/);
  assert.doesNotThrow(() => parseIdentityForms({
    kind: "callable_and_constant", callable_spellings: ["inf"],
    constant_spellings: ["Inf", "inf"],
  }, "inf"));
  assert.doesNotThrow(() => parseIdentityForms({
    kind: "callable_and_constant", callable_spellings: ["nan"],
    constant_spellings: ["NaN", "nan"],
  }, "nan"));
});

test("identity authority graph derives aliases from their sole edge authority", () => {
  const primary = row("foo", {
    kind: "primary",
    primary_spelling: { identity: "foo", spelling: "foo" },
  }, implementationByForm(owned("foo"), absent("constant")));
  const alias = row("foalias", {
    kind: "alias",
    alias_spelling: { identity: "foalias", spelling: "foalias" },
    canonical_identity: "foo",
  }, aliasImplementation());
  assert.doesNotThrow(() => validateIdentityAuthorityGraph(new Map([
    ["foo", primary], ["foalias", alias],
  ])));

  const aliasOwner = structuredClone(alias);
  aliasOwner.implementation = implementationByForm(owned("foalias"), absent("constant"));
  assert.throws(() => validateIdentityAuthorityGraph(new Map([
    ["foo", primary], ["foalias", aliasOwner],
  ])), /cannot own an independent implementation/);
});

test("compiled alias edges remain callable independently of migration history", () => {
  assert.deepEqual(observedIdentityForms({ semantic_authority: {
    catalog_aliases: [{ alias: { name: "foalias" }, canonical: { name: "foo" } }],
  } }), {
    kind: "callable",
    callable_spellings: ["foalias"],
    constant_spellings: [],
  });
});

test("callable and constant forms own independent typed implementations", () => {
  const constant = row("eps", { kind: "primary", primary_spelling: { identity: "eps", spelling: "eps" } }, implementationByForm(
    absent("callable"), ownedConstant("eps"),
  ), {
    kind: "constant", callable_spellings: [], constant_spellings: ["eps"],
  });
  assert.doesNotThrow(() => validateIdentityAuthorityGraph(new Map([["eps", constant]])));
  const dual = row("inf", {
    kind: "primary", primary_spelling: { identity: "inf", spelling: "inf" },
  }, implementationByForm(owned("inf"), ownedConstant("inf")), {
    kind: "callable_and_constant", callable_spellings: ["inf"],
    constant_spellings: ["Inf", "inf"],
  });
  assert.notEqual(dual.implementation.callable.owner_path, dual.implementation.constant.owner_path);
  assert.doesNotThrow(() => validateIdentityAuthorityGraph(new Map([["inf", dual]])));
  const crossedForms = structuredClone(dual.implementation);
  crossedForms.callable.bindings = crossedForms.constant.bindings;
  assert.throws(() => parseImplementationAuthority(crossedForms, "inf"), /unsupported kind/);
  const internal = row("__helper", {
    kind: "internal", reason: "helper", evidence: ["review"],
  }, implementationByForm(absent("callable"), absent("constant")));
  assert.throws(() => validateIdentityAuthorityGraph(new Map([["__helper", internal]])), /requires one implementation owner/);
  const repeatedConstant = ownedConstant("inf");
  repeatedConstant.bindings.push({ kind: "constant_registration", constant: "inf", builtin_path: "builtins::other" });
  repeatedConstant.bindings.sort((left, right) => left.builtin_path.localeCompare(right.builtin_path));
  assert.throws(
    () => parseImplementationAuthority(implementationByForm(absent("callable"), repeatedConstant), "inf"),
    /semantic registration identity/,
  );
});

function row(id, publicIdentity, implementation, forms = {
  kind: "callable", callable_spellings: [id], constant_spellings: [],
}) {
  return { identity: id, public_identity: publicIdentity, forms, implementation };
}

function owned(id) {
  const owner = `crates/runmat-runtime/src/builtins/math/${id}.rs`;
  return { kind: "owned", owner_path: owner, bindings: [binding("default", id)] };
}

function ownedConstant(id) {
  return {
    kind: "owned",
    owner_path: "crates/runmat-runtime/src/builtins/constants/mod.rs",
    bindings: [{ kind: "constant_registration", constant: id, builtin_path: "builtins::constants" }],
  };
}

function absent(form) { return { kind: "none", reason: `no-${form}-form` }; }

function aliasImplementation() {
  return {
    callable: { kind: "none", reason: "alias-resolution" },
    constant: { kind: "none", reason: "alias-resolution" },
  };
}

function implementationByForm(callable, constant) { return { callable, constant }; }

function binding(variant, id = "foo") {
  return {
    kind: "canonical_binding", function: `${id}_builtin`, variant,
    builtin_path: `builtins::math::${id}`,
    native_symbol: `runmat_builtin_binding_v1_${Buffer.from(id).toString("hex")}_${Buffer.from(variant).toString("hex")}`,
  };
}
