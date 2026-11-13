//! datasample declarations; shared sampling parameters remain family-owned.
use super::super::contract::{
    ERROR_INTERNAL, ERROR_INVALID_ARGUMENT, INPUTS_DATA_K, INPUTS_DATA_K_OPTIONS, OUTPUT_Y,
    OUTPUT_Y_IDX,
};
use crate::*;

pub const DATASAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "y = datasample(data, k)",
        inputs: &INPUTS_DATA_K,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = datasample(data, k, dim)",
        inputs: &INPUTS_DATA_K_OPTIONS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = datasample(___, Name, Value)",
        inputs: &INPUTS_DATA_K_OPTIONS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[y, idx] = datasample(___)",
        inputs: &INPUTS_DATA_K_OPTIONS,
        outputs: &OUTPUT_Y_IDX,
    },
];

pub const DATASAMPLE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with typed-integer population data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleIntegerDataExtension"),
    };

pub const DATASAMPLE_INTEGER_K_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datasample-integer-k",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "datasample with a typed-integer sample count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DatasampleIntegerKExtension"),
};

pub const DATASAMPLE_INTEGER_DIM_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-integer-dim",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with a typed-integer dimension is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleIntegerDimExtension"),
    };

pub const DATASAMPLE_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-integer-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with typed-integer weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleIntegerWeightsExtension"),
    };

pub const DATASAMPLE_LOGICAL_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-logical-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with logical weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleLogicalWeightsExtension"),
    };

pub const DATASAMPLE_NUMERIC_REPLACE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-numeric-replace",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with a numeric Replace value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleNumericReplaceExtension"),
    };

pub const DATASAMPLE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with resident accelerator input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleResidentInputExtension"),
    };

pub const DATASAMPLE_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    DATASAMPLE_INTEGER_DATA_EXTENSION,
    DATASAMPLE_INTEGER_K_EXTENSION,
    DATASAMPLE_INTEGER_DIM_EXTENSION,
    DATASAMPLE_INTEGER_WEIGHTS_EXTENSION,
    DATASAMPLE_LOGICAL_WEIGHTS_EXTENSION,
    DATASAMPLE_NUMERIC_REPLACE_EXTENSION,
    DATASAMPLE_RESIDENT_INPUT_EXTENSION,
];

pub const DATASAMPLE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "data",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are sampled by exact storage indexing; the documented ordinary numeric population classes are single and double.",
    }];

pub const DATASAMPLE_INTEGER_K_INPUTS: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "Typed k is independently gated and read exactly as a positive structural sample count.",
    },
];

pub const DATASAMPLE_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed dim is independently gated and read exactly as a positive one-based structural dimension.",
    }];

pub const DATASAMPLE_INTEGER_WEIGHTS_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Weights",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed nonnegative weights are independently gated and converted only at the floating probability boundary.",
    }];

pub const DATASAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = datasample(integer_data,k,___)",
        inputs: &DATASAMPLE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Sampling preserves the exact selected integer values and native integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = datasample(data,integer_k,___)",
        inputs: &DATASAMPLE_INTEGER_K_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed k controls output extent and never determines the sampled value class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = datasample(data,k,integer_dim,___)",
        inputs: &DATASAMPLE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed dim selects an axis and never determines the sampled value class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = datasample(data,k,Weights=integer_weights,___)",
        inputs: &DATASAMPLE_INTEGER_WEIGHTS_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "Weights affect selection probabilities only; sampled data retains its ordinary class.",
    },
];

pub const DATASAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATASAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.datasample.INVALID_ARGUMENT",
            identifier: Some("RunMat:datasample:InvalidArgument"),
            when: ERROR_INVALID_ARGUMENT.when,
            message: ERROR_INVALID_ARGUMENT.message,
        },
        BuiltinErrorDescriptor {
            code: "RM.datasample.INTERNAL",
            identifier: Some("RunMat:datasample:Internal"),
            when: ERROR_INTERNAL.when,
            message: ERROR_INTERNAL.message,
        },
    ],
};
