mod documentation;

use self::documentation::REALSQRT_DOCUMENTATION;
use super::super::support::{unary_numeric_catalog_entry, UnaryNumericCatalogSpec};
use crate::{
    BuiltinBindingDeclaration, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinFusionPolicy,
    BuiltinInferenceRule, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    MathInferenceRule, RootKind,
};

const REALSQRT_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real elementwise square-root result.",
}];
const REALSQRT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real single or double numeric input.",
}];
const REALSQRT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = realsqrt(X)",
    inputs: &REALSQRT_INPUTS,
    outputs: &REALSQRT_OUTPUTS,
}];
pub const REALSQRT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REALSQRT.INVALID_INPUT",
    identifier: Some("RunMat:realsqrt:InvalidInput"),
    when: "Input is not real single or double numeric data.",
    message: "realsqrt: invalid input",
};
pub const REALSQRT_ERROR_DOMAIN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REALSQRT.DOMAIN",
    identifier: Some("RunMat:realsqrt:ComplexResult"),
    when: "At least one real input value is negative and would require a complex result.",
    message: "realsqrt: input must be nonnegative",
};
pub const REALSQRT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REALSQRT.INTERNAL",
    identifier: Some("RunMat:realsqrt:Internal"),
    when: "Internal tensor construction or provider interaction fails.",
    message: "realsqrt: internal error",
};
const REALSQRT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    REALSQRT_ERROR_INVALID_INPUT,
    REALSQRT_ERROR_DOMAIN,
    REALSQRT_ERROR_INTERNAL,
];
pub const REALSQRT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REALSQRT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REALSQRT_ERRORS,
};
const REALSQRT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Only real single and double inputs are accepted; integer values reject by class before domain evaluation or provider dispatch.",
    }];
pub const REALSQRT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = realsqrt(X)",
        inputs: &REALSQRT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "realsqrt has no integer overload; the empty accepted-class set intentionally prevents generic numeric coercion from admitting integers.",
    }];
const REALSQRT_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
pub const REALSQRT_CATALOG_ENTRY: BuiltinCatalogEntry =
    unary_numeric_catalog_entry(UnaryNumericCatalogSpec {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name: "realsqrt" },
        documentation: REALSQRT_DOCUMENTATION,
        descriptor: &REALSQRT_DESCRIPTOR,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Root(RootKind::RealOnly)),
        bindings: &REALSQRT_BINDINGS,
        extensions: &[],
        integer_capabilities: &REALSQRT_INTEGER_CAPABILITIES,
        integer_audit: None,
        fusion: BuiltinFusionPolicy::Never,
    });
