mod documentation;

use super::super::support::{provider_unary_numeric_catalog_entry, UnaryNumericCatalogSpec};
use crate::*;

use documentation::DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real inverse complementary error-function values with the input class and shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense real single or double input.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = erfcinv(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ERFCINV_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERFCINV.INVALID_ARGUMENT",
    identifier: Some("RunMat:erfcinv:InvalidArgument"),
    when: "The invocation does not have exactly one input.",
    message: "erfcinv: invalid argument",
};
pub const ERFCINV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERFCINV.INVALID_INPUT",
    identifier: Some("RunMat:erfcinv:InvalidInput"),
    when: "Input is not dense real single or double numeric data.",
    message: "erfcinv: invalid input",
};
pub const ERFCINV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERFCINV.INTERNAL",
    identifier: Some("RunMat:erfcinv:Internal"),
    when: "Internal allocation, provider execution, gather, or residency restoration fails.",
    message: "erfcinv: internal error",
};
pub const ERFCINV_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERFCINV.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:erfcinv:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "erfcinv: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERFCINV_ERROR_INVALID_ARGUMENT,
    ERFCINV_ERROR_INVALID_INPUT,
    ERFCINV_ERROR_INTERNAL,
    ERFCINV_ERROR_TOO_MANY_OUTPUTS,
];
pub const ERFCINV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Fixed-width integer input is rejected before host evaluation or provider dispatch.",
}];
pub const ERFCINV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = erfcinv(X) with fixed-width integer X",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "All eight fixed-width integer classes are unsupported in both compatibility modes; ordinary numeric literals remain double and are supported.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
pub const ERFCINV_CATALOG_ENTRY: BuiltinCatalogEntry =
    provider_unary_numeric_catalog_entry(UnaryNumericCatalogSpec {
        identity: BuiltinCatalogIdentity { name: "erfcinv" },
        documentation: DOCUMENTATION,
        descriptor: &ERFCINV_DESCRIPTOR,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::ErrorFunction(
            ErrorFunctionInferenceRule::InverseComplementary,
        )),
        bindings: &BINDINGS,
        extensions: &[],
        integer_capabilities: &ERFCINV_INTEGER_CAPABILITIES,
        integer_audit: None,
        fusion: BuiltinFusionPolicy::Never,
    });
