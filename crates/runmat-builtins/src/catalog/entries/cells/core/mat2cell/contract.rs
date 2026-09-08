use crate::*;

pub const MAT2CELL_INTEGER_PARTITIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mat2cell-integer-partitions",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "typed integer partition vectors are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Mat2cellIntegerPartitionsExtension"),
    };
pub const MAT2CELL_EXTENSIONS: &[BuiltinExtensionDescriptor] =
    &[MAT2CELL_INTEGER_PARTITIONS_EXTENSION];

const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array containing contiguous blocks of A.",
}];
const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array to partition.",
};
const INPUT_DIM_REQUIRED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dimdist",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A non-negative partition-size vector for the first dimension.",
};
const INPUT_DIM_VARIADIC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dimdist",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One non-negative partition-size vector per selected dimension.",
};
const SINGLE_DIM_INPUTS: &[BuiltinParamDescriptor] = &[INPUT_A, INPUT_DIM_REQUIRED];
const MULTI_DIM_INPUTS: &[BuiltinParamDescriptor] = &[INPUT_A, INPUT_DIM_VARIADIC];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "C = mat2cell(A, dim1dist)",
        inputs: SINGLE_DIM_INPUTS,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = mat2cell(A, dim1dist, dim2dist, ...)",
        inputs: MULTI_DIM_INPUTS,
        outputs: OUTPUTS,
    },
];

pub const MAT2CELL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAT2CELL.INVALID_INPUT",
    identifier: Some("RunMat:mat2cell:InvalidInput"),
    when: "The input array type or argument count is invalid.",
    message: "mat2cell: invalid input arguments",
};
pub const MAT2CELL_ERROR_INVALID_PARTITION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAT2CELL.INVALID_PARTITION",
    identifier: Some("RunMat:mat2cell:InvalidPartition"),
    when: "A partition vector is malformed or inconsistent with its input dimension.",
    message: "mat2cell: invalid partition sizes",
};
pub const MAT2CELL_ERROR_SIZE_EXCEEDED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAT2CELL.SIZE_EXCEEDED",
    identifier: Some("RunMat:mat2cell:SizeExceeded"),
    when: "A partition or output shape exceeds platform limits.",
    message: "mat2cell: size exceeds platform limits",
};
pub const MAT2CELL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAT2CELL.INTERNAL",
    identifier: None,
    when: "An internal storage or indexing invariant fails.",
    message: "mat2cell: internal error",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    MAT2CELL_ERROR_INVALID_INPUT,
    MAT2CELL_ERROR_INVALID_PARTITION,
    MAT2CELL_ERROR_SIZE_EXCEEDED,
    MAT2CELL_ERROR_INTERNAL,
];
pub const MAT2CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

const DATA_INPUT: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Each block preserves A's exact native integer class and values.",
}];
const PARTITION_INPUT: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "dimdist",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed partition values are decoded exactly and range checked.",
}];
pub const MAT2CELL_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor { form: "C = mat2cell(integer_A, dim1dist, ...)", inputs: DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Automatic residency gathers before host cell construction; explicit resident input is unsupported." },
    BuiltinIntegerCapabilityDescriptor { form: "C = mat2cell(A, integer_dim1dist, ...)", inputs: PARTITION_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The RunMat-only typed form validates exact partition values before block construction." },
];
