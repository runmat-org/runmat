use crate::*;

const Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite nonempty real predictions in a numeric array, a host dlarray, or a plain resident gpuArray.",
};
const TARGETS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "targets",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real target probabilities with the same element count as Y; the resident provider form requires equal shape.",
};
const WEIGHTS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "weights",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional scalar, full-sized, channel, or observation weights.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Name, Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "ClassificationMode, TargetCategories, Reduction, NormalizationFactor, DataFormat, WeightsFormat, Weights, or Mask pairs.",
};
const PLAIN_INPUTS: [BuiltinParamDescriptor; 3] = [Y, TARGETS, OPTIONS];
const WEIGHTED_INPUTS: [BuiltinParamDescriptor; 4] = [Y, TARGETS, WEIGHTS, OPTIONS];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "loss",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar reduced loss, or prediction-shaped loss for Reduction='none'; a host dlarray input returns an unformatted dlarray.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "loss = crossentropy(Y, targets, Name, Value, ...)",
        inputs: &PLAIN_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "loss = crossentropy(Y, targets, weights, Name, Value, ...)",
        inputs: &WEIGHTED_INPUTS,
        outputs: &OUTPUT,
    },
];

pub const CROSSENTROPY_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "Input class, shape, finite values, target range, weights, mask, option list, or requested output count is invalid.",
    message: "crossentropy: invalid prediction, target, weighting, or option input",
};
pub const CROSSENTROPY_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "An explicit gpuArray input has no supported resident provider form; automatic residency alone may use the admitted host fallback.",
    message: "crossentropy: explicit gpuArray requires a supported resident provider path",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [CROSSENTROPY_INVALID_INPUT, CROSSENTROPY_UNSUPPORTED];
pub const CROSSENTROPY_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "crossentropy-integer-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Native integer prediction, target, or weight arrays in floating-point cross-entropy are a RunMat extension.",
    error_identifier: Some("RunMat:compatibility:CrossentropyIntegerInputExtension"),
};
pub const CROSSENTROPY_PLAIN_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "crossentropy-plain-numeric-pair",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "A plain numeric prediction and target pair without a dlarray operand is a RunMat extension.",
    error_identifier: Some("RunMat:compatibility:CrossentropyPlainNumericExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CROSSENTROPY_INTEGER_EXTENSION,
    CROSSENTROPY_PLAIN_NUMERIC_EXTENSION,
];
pub const CROSSENTROPY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Y, targets, weights",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Host typed integers enter floating-point cross-entropy arithmetic and may round, including wide values; dlarray construction itself rejects integer storage.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "crossentropy(integer_Y, integer_targets, ...)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Current host numeric-array extension converts integer data at the floating loss boundary. It is not an exact-integer computation.",
    }];
