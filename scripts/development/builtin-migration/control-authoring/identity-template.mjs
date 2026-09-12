import { compareCodePoint } from "../constants.mjs";
import { implementationBindingKey } from "../identity-authority.mjs";

export function identityControlAuthorityTemplate(scaffold, identity) {
  const proposal = scaffold.authority_proposals.identity_rows
    .find((row) => row.identity === identity)?.proposal;
  if (!proposal) throw new Error(`${identity}: scaffold has no authority proposal`);
  return {
    public_identity: publicIdentityTemplate(proposal.public_identity),
    forms: structuredClone(proposal.forms),
    implementation: implementationTemplate(proposal, identity),
  };
}

function publicIdentityTemplate(value) {
  if (value.kind === "internal") return structuredClone(value);
  const spelling = (entry) => ({ identity: entry.identity, spelling: entry.spelling });
  if (value.kind === "alias") {
    return {
      kind: "alias",
      alias_spelling: spelling(value.alias_spelling),
      canonical_identity: value.canonical_identity,
    };
  }
  return {
    kind: "primary",
    primary_spelling: spelling(value.primary_spelling),
  };
}

function implementationTemplate(proposal, identity) {
  if (proposal.public_identity.kind === "alias") {
    return {
      callable: { kind: "none", reason: "alias-resolution" },
      constant: { kind: "none", reason: "alias-resolution" },
    };
  }
  return {
    callable: callableImplementationTemplate(proposal, identity),
    constant: constantImplementationTemplate(proposal),
  };
}

function callableImplementationTemplate(proposal, identity) {
  if (proposal.forms.callable_spellings.length === 0) {
    return { kind: "none", reason: "no-callable-form" };
  }
  const observed = proposal.implementation.callable;
  if (observed.proposed_owner_path === null) return null;
  const bindings = observed.observed_bindings.map((entry) => {
    if (entry.authority === "canonical_binding") {
      return {
        kind: "canonical_binding",
        function: entry.function,
        variant: entry.binding_variant,
        builtin_path: entry.builtin_path,
        native_symbol: nativeSymbol(identity, entry.binding_variant),
      };
    }
    return {
      kind: "legacy_function",
      function: entry.function,
      builtin_path: entry.builtin_path,
    };
  }).sort((left, right) => compareCodePoint(
    implementationBindingKey(left), implementationBindingKey(right),
  ));
  return { kind: "owned", owner_path: observed.proposed_owner_path, bindings };
}

function constantImplementationTemplate(proposal) {
  if (proposal.forms.constant_spellings.length === 0) {
    return { kind: "none", reason: "no-constant-form" };
  }
  const observed = proposal.implementation.constant;
  if (observed.proposed_owner_path === null) return null;
  const bindings = observed.observed_bindings.map((entry) => ({
      kind: "constant_registration",
      constant: entry.name,
      builtin_path: entry.builtin_path,
    })).sort((left, right) => compareCodePoint(
    implementationBindingKey(left), implementationBindingKey(right),
  ));
  return { kind: "owned", owner_path: observed.proposed_owner_path, bindings };
}

function nativeSymbol(name, variant) {
  return `runmat_builtin_binding_v1_${Buffer.from(name).toString("hex")}_${Buffer.from(variant).toString("hex")}`;
}
