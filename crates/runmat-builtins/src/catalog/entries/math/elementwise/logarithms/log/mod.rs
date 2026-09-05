use super::super::support::{unary_numeric_catalog_entry, UnaryNumericCatalogSpec};
use crate::{
    BuiltinBindingDeclaration, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, LogarithmKind, MathInferenceRule, ALL_INTEGER_CLASSES,
};

mod documentation;

use documentation::LOG_DOCUMENTATION;

const LOG_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise natural-log result, promoted to complex for negative real input.",
}];
const LOG_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input or a supported table; integer, logical, and character forms are RunMat-only extensions.",
}];
const LOG_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = log(X)",
    inputs: &LOG_INPUTS,
    outputs: &LOG_OUTPUTS,
}];

pub const LOG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG.INVALID_INPUT",
    identifier: Some("RunMat:log:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or table data.",
    message: "log: invalid input",
};
pub const LOG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG.INTERNAL",
    identifier: Some("RunMat:log:Internal"),
    when: "Internal tensor construction, table mapping, or provider interaction fails.",
    message: "log: internal error",
};
const LOG_ERRORS: [BuiltinErrorDescriptor; 2] = [LOG_ERROR_INVALID_INPUT, LOG_ERROR_INTERNAL];
pub const LOG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOG_ERRORS,
};

pub const LOG_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LogIntegerInputExtension"),
};
pub const LOG_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LogLogicalInputExtension"),
};
pub const LOG_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LogCharacterInputExtension"),
};
pub const LOG_EXPLICIT_GPU_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "log-explicit-real-gpu-complex-promotion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "complex promotion from an explicit real gpuArray is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LogExplicitGpuComplexExtension"),
    };
pub const LOG_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    LOG_INTEGER_INPUT_EXTENSION,
    LOG_LOGICAL_INPUT_EXTENSION,
    LOG_CHARACTER_INPUT_EXTENSION,
    LOG_EXPLICIT_GPU_COMPLEX_EXTENSION,
];

const LOG_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are accepted only in RunMat mode and only when every value lies in the inclusive exact binary64 interval [-2^53, 2^53].",
    }];
pub const LOG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = log(integer_X)",
        inputs: &LOG_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only overload validates exact binary64 conversion before computation. Negative real values produce complex double output; resident values gather through their exact owner.",
    }];

const LOG_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
pub const LOG_CATALOG_ENTRY: BuiltinCatalogEntry =
    unary_numeric_catalog_entry(UnaryNumericCatalogSpec {
        identity: BuiltinCatalogIdentity { name: "log" },
        documentation: LOG_DOCUMENTATION,
        descriptor: &LOG_DESCRIPTOR,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Logarithm(
            LogarithmKind::Natural,
        )),
        bindings: &LOG_BINDINGS,
        extensions: &LOG_EXTENSIONS,
        integer_capabilities: &LOG_INTEGER_CAPABILITIES,
        integer_audit: None,
        fusion: BuiltinFusionPolicy::Never,
    });
