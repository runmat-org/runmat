//! Constraint constructors. The generated testing parent owns composition.

use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

const EXPECTED_INPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "expected",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Expected host value, retained without numeric conversion.",
}];
const NO_INPUTS: &[BuiltinParamDescriptor] = &[];
const OBJECT_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "constraint",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A scalar constraint object with its concrete runtime class.",
}];
const EQUAL_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "constraint = matlab.unittest.constraints.IsEqualTo(expected)",
    inputs: EXPECTED_INPUT,
    outputs: OBJECT_OUTPUT,
}];
const TRUE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "constraint = matlab.unittest.constraints.IsTrue()",
    inputs: NO_INPUTS,
    outputs: OBJECT_OUTPUT,
}];
const FALSE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "constraint = matlab.unittest.constraints.IsFalse()",
    inputs: NO_INPUTS,
    outputs: OBJECT_OUTPUT,
}];
const EQUAL_ERRORS: &[BuiltinErrorDescriptor] = &[BuiltinErrorDescriptor {
    code: "RM.TESTING.IS_EQUAL_TO_INPUT",
    identifier: Some("RunMat:Testing:InvalidConstraint"),
    when: "The constructor receives a missing expected value or unsupported additional arguments.",
    message: "IsEqualTo requires exactly one expected value",
}];
const BOOLEAN_ERRORS: &[BuiltinErrorDescriptor] = &[BuiltinErrorDescriptor {
    code: "RM.TESTING.BOOLEAN_CONSTRAINT_INPUT",
    identifier: Some("RunMat:Testing:InvalidConstraint"),
    when: "The constructor receives an argument.",
    message: "IsTrue and IsFalse accept no inputs",
}];

pub const IS_EQUAL_TO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: EQUAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: EQUAL_ERRORS,
};
pub const IS_TRUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: TRUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: BOOLEAN_ERRORS,
};
pub const IS_FALSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: FALSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: BOOLEAN_ERRORS,
};

const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

const fn entry(
    name: &'static str,
    docs: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "testing/constraints",
        documentation: docs,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[EffectKind::MayThrow],
            capabilities: &[],
        },
        placement: PLACEMENT,
        link: LINK,
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const IS_EQUAL_TO_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.constraints.IsEqualTo",
    documentation::IS_EQUAL_TO,
    &IS_EQUAL_TO_DESCRIPTOR,
    inference::is_equal_to,
);
pub const IS_TRUE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.constraints.IsTrue",
    documentation::IS_TRUE,
    &IS_TRUE_DESCRIPTOR,
    inference::is_true,
);
pub const IS_FALSE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.constraints.IsFalse",
    documentation::IS_FALSE,
    &IS_FALSE_DESCRIPTOR,
    inference::is_false,
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &IS_EQUAL_TO_CATALOG_ENTRY,
    &IS_TRUE_CATALOG_ENTRY,
    &IS_FALSE_CATALOG_ENTRY,
];
