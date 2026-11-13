use crate::*;

pub const RANDOM_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "random-integer-parameters",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "random with typed-integer distribution parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandomIntegerParametersExtension"),
    };
pub const RANDOM_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "random-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "random with typed-integer size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandomIntegerSizeExtension"),
};
pub const RANDOM_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    RANDOM_INTEGER_PARAMETER_EXTENSION,
    RANDOM_INTEGER_SIZE_EXTENSION,
];
const RANDOM_INTEGER_PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A...D",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single/double named-distribution parameters; typed integers are gated before conversion and must be exactly representable as binary64.",
    }];
const RANDOM_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sz1...szN or sz",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents single/double size controls; RunMat typed sizes are exact structural values and do not cross the distribution computation boundary.",
    }];
pub const RANDOM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "R = random(name,integer_A...D,...)",
        inputs: &RANDOM_INTEGER_PARAMETER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each typed parameter is admitted independently and converted once after exactness validation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = random(pd,integer_sz) or random(name,A...D,integer_sz)",
        inputs: &RANDOM_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed sizes gate before provider access and are decoded from authoritative integer storage without floating conversion.",
    },
];
