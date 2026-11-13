use crate::*;

const ISMEMBERTOL_OUTPUT_MASK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "LIA",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical mask over A.",
}];

const ISMEMBERTOL_OUTPUT_MASK_LOC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "LIA",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical mask over A.",
    },
    BuiltinParamDescriptor {
        name: "LocB",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First or all matching indices into B.",
    },
];

const ISMEMBERTOL_INPUTS_AB: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query values or rows.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference values or rows.",
    },
];

const ISMEMBERTOL_INPUTS_AB_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query values or rows.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference values or rows.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1e-12 for double, 1e-6 for single"),
        description: "Relative tolerance.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: ByRows, DataScale, OutputAllIndices.",
    },
];

const ISMEMBERTOL_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "LIA = ismembertol(A, B)",
        inputs: &ISMEMBERTOL_INPUTS_AB,
        outputs: &ISMEMBERTOL_OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "LIA = ismembertol(A, B, tol, Name, Value)",
        inputs: &ISMEMBERTOL_INPUTS_AB_OPTIONS,
        outputs: &ISMEMBERTOL_OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "[LIA, LocB] = ismembertol(A, B)",
        inputs: &ISMEMBERTOL_INPUTS_AB,
        outputs: &ISMEMBERTOL_OUTPUT_MASK_LOC,
    },
    BuiltinSignatureDescriptor {
        label: "[LIA, LocB] = ismembertol(A, B, tol, Name, Value)",
        inputs: &ISMEMBERTOL_INPUTS_AB_OPTIONS,
        outputs: &ISMEMBERTOL_OUTPUT_MASK_LOC,
    },
];

pub const ISMEMBERTOL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBERTOL.INVALID_INPUT",
    identifier: Some("RunMat:ismembertol:InvalidInput"),
    when: "A or B is not a supported real full numeric input.",
    message: "ismembertol: inputs must be real full numeric arrays",
};

pub const ISMEMBERTOL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBERTOL.INVALID_ARGUMENT",
    identifier: Some("RunMat:ismembertol:InvalidArgument"),
    when: "Tolerance or name-value arguments are malformed.",
    message: "ismembertol: invalid argument",
};

pub const ISMEMBERTOL_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBERTOL.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:ismembertol:RowsColumnMismatch"),
    when: "ByRows is true and A/B column counts differ.",
    message: "ismembertol: inputs must have the same number of columns when ByRows is true",
};

pub const ISMEMBERTOL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBERTOL.INTERNAL",
    identifier: Some("RunMat:ismembertol:Internal"),
    when: "Internal conversion or allocation fails.",
    message: "ismembertol: internal error",
};

const ISMEMBERTOL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ISMEMBERTOL_ERROR_INVALID_INPUT,
    ISMEMBERTOL_ERROR_INVALID_ARGUMENT,
    ISMEMBERTOL_ERROR_ROWS_COLUMN_MISMATCH,
    ISMEMBERTOL_ERROR_INTERNAL,
];

pub const ISMEMBERTOL_HOST_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ismembertol-host-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ismembertol with host typed-integer A or B data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsmembertolHostIntegerDataExtension"),
    };

pub const ISMEMBERTOL_HOST_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ismembertol-host-logical-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ismembertol with host logical A or B data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsmembertolHostLogicalDataExtension"),
    };

pub const ISMEMBERTOL_TYPED_TOLERANCE_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ismembertol-typed-tolerance-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "ismembertol with a typed-integer tolerance or DataScale is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsmembertolTypedToleranceControlExtension"),
    };

pub const ISMEMBERTOL_LOGICAL_TOLERANCE_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ismembertol-logical-tolerance-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ismembertol with a logical tolerance or DataScale is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsmembertolLogicalToleranceControlExtension"),
    };

pub const ISMEMBERTOL_GPU_WIDE_INTEGER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ismembertol-gpu-wide-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ismembertol with a resident 64-bit integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsmembertolGpuWideIntegerInputExtension"),
    };

pub const ISMEMBERTOL_GPU_OPTIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ismembertol-gpu-options",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "ismembertol ByRows or OutputAllIndices with resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsmembertolGpuOptionsExtension"),
    };

pub const ISMEMBERTOL_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    ISMEMBERTOL_GPU_OPTIONS_EXTENSION,
    ISMEMBERTOL_GPU_WIDE_INTEGER_EXTENSION,
    ISMEMBERTOL_HOST_INTEGER_DATA_EXTENSION,
    ISMEMBERTOL_HOST_LOGICAL_DATA_EXTENSION,
    ISMEMBERTOL_LOGICAL_TOLERANCE_CONTROL_EXTENSION,
    ISMEMBERTOL_TYPED_TOLERANCE_CONTROL_EXTENSION,
];

const ISMEMBERTOL_WIDE_INTEGER_CLASSES: [BuiltinIntegerClass; 2] =
    [BuiltinIntegerClass::Int64, BuiltinIntegerClass::Uint64];

const ISMEMBERTOL_HOST_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented host data domain is single or double; RunMat mode additionally accepts every real integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented host data domain is single or double; RunMat mode additionally accepts every real integer class.",
    },
];

const ISMEMBERTOL_GPU_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "resident_A",
        classes: &crate::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Interactive GPU arrays document signed and unsigned integer data through 32 bits.",
    },
    BuiltinIntegerInputCapability {
        name: "resident_B",
        classes: &crate::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Interactive GPU arrays document signed and unsigned integer data through 32 bits.",
    },
];

const ISMEMBERTOL_GPU_WIDE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident_A_or_B",
        classes: &ISMEMBERTOL_WIDE_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public interactive GPU support excludes 64-bit integers; RunMat mode preserves them through typed gather before the tolerance boundary.",
    }];

const ISMEMBERTOL_TYPED_TOLERANCE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "tol_or_DataScale",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented tolerance and DataScale classes are single and double; RunMat mode additionally accepts typed integers and converts them once into the floating tolerance domain.",
    }];

const ISMEMBERTOL_INTEGER_BOOLEAN_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ByRows_or_OutputAllIndices",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented numeric boolean controls accept only zero or one; resident ByRows and OutputAllIndices forms remain independently restricted.",
    }];

pub const ISMEMBERTOL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[LIA, LocB] = ismembertol(host_integer_A, host_integer_B, options)",
        inputs: &ISMEMBERTOL_HOST_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only host integer observations retain authoritative storage until one explicit f64 tolerance/DataScale boundary; LIA is logical and first locations are double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[LIA, LocB] = ismembertol(resident_integer_A, resident_integer_B, options)",
        inputs: &ISMEMBERTOL_GPU_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented resident integer inputs through 32 bits use typed gather fallback and restore logical/double outputs to the input owner; ByRows and OutputAllIndices remain mode-gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[LIA, LocB] = ismembertol(resident_int64_or_uint64_A_or_B, options)",
        inputs: &ISMEMBERTOL_GPU_WIDE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Resident 64-bit integer data is a RunMat-only extension and enters the same explicit floating tolerance domain after authoritative typed gather.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "LIA = ismembertol(A, B, integer_tol_or_DataScale, options)",
        inputs: &ISMEMBERTOL_TYPED_TOLERANCE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed integer tolerance controls are RunMat-only and convert once to f64 before threshold arithmetic.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "LIA = ismembertol(A, B, ByRows_or_OutputAllIndices=integer_zero_or_one)",
        inputs: &ISMEMBERTOL_INTEGER_BOOLEAN_OPTION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed integer zero/one boolean controls are documented numeric forms; interactive GPU execution excludes enabled ByRows and OutputAllIndices.",
    },
];

pub const ISMEMBERTOL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISMEMBERTOL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISMEMBERTOL_ERRORS,
};
