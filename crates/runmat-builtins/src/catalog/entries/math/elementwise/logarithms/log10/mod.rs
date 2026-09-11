mod documentation;

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

use documentation::LOG10_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise base-10 logarithm, promoted to complex for negative real input.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input or a supported table; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = log10(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const LOG10_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG10.INVALID_INPUT",
    identifier: Some("RunMat:log10:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or table data.",
    message: "log10: invalid input",
};
pub const LOG10_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG10.INTERNAL",
    identifier: Some("RunMat:log10:Internal"),
    when: "Internal tensor construction, table mapping, or provider interaction fails.",
    message: "log10: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [LOG10_ERROR_INVALID_INPUT, LOG10_ERROR_INTERNAL];
pub const LOG10_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const LOG10_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log10-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log10 with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log10IntegerInputExtension"),
};
pub const LOG10_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log10-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log10 with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log10LogicalInputExtension"),
};
pub const LOG10_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "log10-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "log10 with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Log10CharacterInputExtension"),
    };
pub const LOG10_EXPLICIT_GPU_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "log10-explicit-real-gpu-complex-promotion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "complex promotion from an explicit real gpuArray is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Log10ExplicitGpuComplexExtension"),
    };
pub const LOG10_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    LOG10_INTEGER_INPUT_EXTENSION,
    LOG10_LOGICAL_INPUT_EXTENSION,
    LOG10_CHARACTER_INPUT_EXTENSION,
    LOG10_EXPLICIT_GPU_COMPLEX_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are accepted only in RunMat mode and only when every value lies in the inclusive exact binary64 interval [-2^53, 2^53].",
}];
pub const LOG10_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = log10(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only overload validates exact binary64 conversion before computation. Negative real values produce complex double output; resident values gather through their exact owner.",
    }];
const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
pub const LOG10_CATALOG_ENTRY: BuiltinCatalogEntry =
    unary_numeric_catalog_entry(UnaryNumericCatalogSpec {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name: "log10" },
        documentation: LOG10_DOCUMENTATION,
        descriptor: &LOG10_DESCRIPTOR,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Logarithm(
            LogarithmKind::Common,
        )),
        bindings: &BINDINGS,
        extensions: &LOG10_EXTENSIONS,
        integer_capabilities: &LOG10_INTEGER_CAPABILITIES,
        integer_audit: None,
        fusion: BuiltinFusionPolicy::Never,
    });
