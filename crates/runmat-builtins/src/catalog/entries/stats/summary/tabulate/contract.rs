use crate::*;

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, or text observations.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tbl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Values, counts, and percentages in three columns.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tbl = tabulate(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.tabulate.INVALID_ARGUMENT",
        identifier: Some("RunMat:tabulate:InvalidArgument"),
        when: "Inputs, flags, dimensions, or options are malformed.",
        message: "descriptive statistics: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.tabulate.INTERNAL",
        identifier: Some("RunMat:tabulate:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "descriptive statistics: internal error",
    },
];
pub const TABULATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TABULATE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "tabulate-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "tabulate with typed-integer input data and exact cell output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TabulateIntegerDataExtension"),
    };
pub const TABULATE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tabulate-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tabulate with a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TabulateGpuInputExtension"),
};
pub const TABULATE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TABULATE_INTEGER_DATA_EXTENSION,
    TABULATE_GPU_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes group by exact value; the first cell column retains exact wide labels.",
}];
pub const TABULATE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tbl = tabulate(integer_X)", inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer labels remain exact in a cell table; counts and percentages are double. Bounded positive-level expansion preserves small gaps; larger domains group observed labels.",
    },
];
