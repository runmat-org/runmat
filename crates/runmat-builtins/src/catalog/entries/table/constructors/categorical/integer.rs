use crate::*;

pub const CATEGORICAL_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "categorical-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "categorical with an interactive resident GPU argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CategoricalGpuInputExtension"),
    };

pub const CATEGORICAL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [CATEGORICAL_GPU_INPUT_EXTENSION];

const CATEGORICAL_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric input accepts every real integer class and maps exact source values to categorical codes without a binary64 intermediary.",
    }];

const CATEGORICAL_INTEGER_VALUESET_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric input accepts every real integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "valueset",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A documented numeric valueset must use the same integer class as A, contain unique values, and is matched through exact native storage.",
    },
];

const CATEGORICAL_INTEGER_FLAG_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Ordinal or Protected",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented flags accept a scalar numeric zero or one; other integer values reject rather than being coerced by generic truthiness.",
    }];

const CATEGORICAL_RESIDENT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident argument",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Interactive resident numeric input is not a documented gpuArray form; RunMat mode gates it before exact gather into the host categorical constructor.",
    }];

pub const CATEGORICAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = categorical(integer_A, ...)",
        inputs: &CATEGORICAL_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact integer values determine sorted categories and one-based categorical codes; the result is a host categorical metadata object rather than an integer array.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = categorical(integer_A, integer_valueset, ...)",
        inputs: &CATEGORICAL_INTEGER_VALUESET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A and valueset retain their exact common integer class for uniqueness, ordering, and membership; category names are object metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = categorical(A, ..., \"Ordinal\" or \"Protected\", integer_flag)",
        inputs: &CATEGORICAL_INTEGER_FLAG_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Every integer class is accepted only at exact scalar values zero and one, matching the documented numeric flag domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = categorical(gpuArray(integer_argument), ...)",
        inputs: &CATEGORICAL_RESIDENT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat mode gathers resident integer arguments exactly after the compatibility gate and returns a host categorical metadata object.",
    },
];
