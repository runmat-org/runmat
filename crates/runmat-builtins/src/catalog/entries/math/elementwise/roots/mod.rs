mod documentation;

use self::documentation::{REALSQRT_DOCUMENTATION, SQRT_DOCUMENTATION};
use super::support::{unary_numeric_catalog_entry, UnaryNumericCatalogSpec};
use crate::{
    BuiltinBindingDeclaration, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, MathInferenceRule, RootKind, ALL_INTEGER_CLASSES,
};

const SQRT_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise principal square root, promoted to complex for negative real input.",
}];
const SQRT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SQRT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = sqrt(X)",
    inputs: &SQRT_INPUTS,
    outputs: &SQRT_OUTPUTS,
}];
pub const SQRT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQRT.INVALID_INPUT",
    identifier: Some("RunMat:sqrt:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or complex data.",
    message: "sqrt: invalid input",
};
pub const SQRT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQRT.INTERNAL",
    identifier: Some("RunMat:sqrt:Internal"),
    when: "Internal tensor construction or provider interaction fails.",
    message: "sqrt: internal error",
};
const SQRT_ERRORS: [BuiltinErrorDescriptor; 2] = [SQRT_ERROR_INVALID_INPUT, SQRT_ERROR_INTERNAL];
pub const SQRT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SQRT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SQRT_ERRORS,
};
pub const SQRT_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sqrt-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sqrt with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SqrtIntegerInputExtension"),
};
pub const SQRT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SQRT_INTEGER_INPUT_EXTENSION];
const SQRT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode accepts real typed integers whose authoritative values are exactly representable at the binary64 square-root boundary; typed complex integers remain unsupported.",
    }];
pub const SQRT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = sqrt(integer_X)",
        inputs: &SQRT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The result is real double for nonnegative input and complex double when any value is negative. Exactness validation precedes provider access or gather.",
    }];
const SQRT_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
pub const SQRT_CATALOG_ENTRY: BuiltinCatalogEntry =
    unary_numeric_catalog_entry(UnaryNumericCatalogSpec {
        identity: BuiltinCatalogIdentity { name: "sqrt" },
        documentation: SQRT_DOCUMENTATION,
        descriptor: &SQRT_DESCRIPTOR,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Root(RootKind::Principal)),
        bindings: &SQRT_BINDINGS,
        extensions: &SQRT_EXTENSIONS,
        integer_capabilities: &SQRT_INTEGER_CAPABILITIES,
        integer_audit: None,
        fusion: BuiltinFusionPolicy::Never,
    });

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

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&SQRT_CATALOG_ENTRY, &REALSQRT_CATALOG_ENTRY];
