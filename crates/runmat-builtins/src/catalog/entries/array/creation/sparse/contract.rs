use crate::*;

const SPARSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse matrix.",
}];

pub const SPARSE_INPUT_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Full or sparse matrix to convert.",
}];

pub const SPARSE_INPUT_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of rows.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of columns.",
    },
];

const SPARSE_INPUT_DIMS_TYPENAME: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of rows.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of columns.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sparse storage type: double, single, or logical.",
    },
];

const SPARSE_INPUT_TRIPLETS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "i",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based row subscripts.",
    },
    BuiltinParamDescriptor {
        name: "j",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based column subscripts.",
    },
    BuiltinParamDescriptor {
        name: "v",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values for each row/column pair.",
    },
];

const SPARSE_INPUT_TRIPLETS_DIMS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "i",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based row subscripts.",
    },
    BuiltinParamDescriptor {
        name: "j",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based column subscripts.",
    },
    BuiltinParamDescriptor {
        name: "v",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values for each row/column pair.",
    },
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of rows.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of columns.",
    },
];

const SPARSE_INPUT_TRIPLETS_DIMS_NZMAX: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "i",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based row subscripts.",
    },
    BuiltinParamDescriptor {
        name: "j",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based column subscripts.",
    },
    BuiltinParamDescriptor {
        name: "v",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values for each row/column pair.",
    },
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of rows.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of columns.",
    },
    BuiltinParamDescriptor {
        name: "nzmax",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Allocation hint accepted for MATLAB compatibility.",
    },
];

const SPARSE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "S = sparse(A)",
        inputs: &SPARSE_INPUT_A,
        outputs: &SPARSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sparse(m, n)",
        inputs: &SPARSE_INPUT_DIMS,
        outputs: &SPARSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sparse(m, n, typename)",
        inputs: &SPARSE_INPUT_DIMS_TYPENAME,
        outputs: &SPARSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sparse(i, j, v)",
        inputs: &SPARSE_INPUT_TRIPLETS,
        outputs: &SPARSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sparse(i, j, v, m, n)",
        inputs: &SPARSE_INPUT_TRIPLETS_DIMS,
        outputs: &SPARSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sparse(i, j, v, m, n, nzmax)",
        inputs: &SPARSE_INPUT_TRIPLETS_DIMS_NZMAX,
        outputs: &SPARSE_OUTPUT,
    },
];

pub const SPARSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPARSE.INVALID_INPUT",
    identifier: Some("RunMat:sparse:InvalidInput"),
    when: "Inputs are not a supported sparse construction form.",
    message: "sparse: invalid input",
};

pub const SPARSE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPARSE.INVALID_INDEX",
    identifier: Some("RunMat:sparse:InvalidIndex"),
    when: "Row or column subscripts are nonpositive, noninteger, or outside explicit dimensions.",
    message: "sparse: invalid index",
};

pub const SPARSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPARSE.INTERNAL",
    identifier: Some("RunMat:sparse:Internal"),
    when: "Sparse matrix materialisation fails internally.",
    message: "sparse: internal error",
};

pub const SPARSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SPARSE_ERROR_INVALID_INPUT,
    SPARSE_ERROR_INVALID_INDEX,
    SPARSE_ERROR_INTERNAL,
];

pub const SPARSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPARSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPARSE_ERRORS,
};

const SPARSE_INTEGER_A_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer sparse value storage is a RunMat extension; the public sparse value domain is double, single, or logical.",
    }];
const SPARSE_INTEGER_DIMS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "m, n, or nzmax",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Structural size and allocation controls are decoded exactly and range checked before conversion to platform dimensions.",
    }];
const SPARSE_INTEGER_SUBSCRIPT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "i and j",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Integer row and column subscripts are decoded without binary64 conversion and must use one shared integer datatype.",
    }];
const SPARSE_INTEGER_TRIPLET_VALUE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "i and j",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Integer row and column subscripts use one shared datatype and remain exact.",
    },
    BuiltinIntegerInputCapability {
        name: "v",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer stored values are independently compatibility-gated and retain their native class in RunMat mode.",
    },
];
pub const SPARSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "S = sparse(integer_A)", inputs: &SPARSE_INTEGER_A_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "RunMat mode constructs authoritative typed CSC value storage; MATLAB-compatible modes reject before provider access." },
    BuiltinIntegerCapabilityDescriptor { form: "S = sparse(integer_m, integer_n, typename?)", inputs: &SPARSE_INTEGER_DIMS_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::OptionDependent, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Dimension controls select host sparse shape while typename selects documented double, single, or logical storage." },
    BuiltinIntegerCapabilityDescriptor { form: "S = sparse(integer_i, integer_j, v, m?, n?, nzmax?)", inputs: &SPARSE_INTEGER_SUBSCRIPT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Documented integer subscripts remain exact through duplicate accumulation and CSC ordering; the stored value class follows v." },
    BuiltinIntegerCapabilityDescriptor { form: "S = sparse(integer_i, integer_j, integer_v, m?, n?, nzmax?)", inputs: &SPARSE_INTEGER_TRIPLET_VALUE_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The typed-value form is RunMat-only, sums duplicate entries with class-specific saturation, and stores exact native integer CSC values." },
];
