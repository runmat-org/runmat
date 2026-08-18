//! MATLAB-compatible `sparse` construction for real double matrices.

use std::collections::{BTreeMap, BTreeSet};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType,
    ResolveContext, SparseTensor, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod descriptors;
pub use descriptors::{
    NONZEROS_DESCRIPTOR, SPDIAGS_DESCRIPTOR, SPEYE_DESCRIPTOR, SPONES_DESCRIPTOR, SPRAND_DESCRIPTOR,
};

const NAME: &str = "sparse";
const SPARSE_DENSE_INPUT_VECTOR_LIMIT: usize = 10_000_000;
const SPARSE_HELPER_DENSE_INPUT_LIMIT: usize = 10_000_000;
const SPRAND_CONDITION_DENSE_INPUT_LIMIT: usize = 1_000_000;
const SPRAND_CONDITION_ROTATION_WORK_LIMIT: usize = 50_000_000;
const SPRAND_CONDITION_MAX_ROTATION_ATTEMPTS: usize = 10_000;

const NONZEROS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for full and sparse inputs; authoritative elements are filtered without numeric conversion.",
    }];
pub const NONZEROS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "v = nonzeros(integer_A)", inputs: &NONZEROS_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Stored nonzero values are copied in column order into a full same-class column. Host, sparse, scalar, and supported real gpuArray inputs preserve exact integer class; automatic residency may gather transparently and explicit residency is restored through the exact owner." }];

const SPARSE_INTEGER_A_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer sparse value storage is a RunMat extension; the public sparse value domain is double, single, or logical.",
    }];
const SPARSE_INTEGER_DIMS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "m, n, or nzmax",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Structural size and allocation controls are decoded exactly and range checked before conversion to platform dimensions.",
    }];
const SPARSE_INTEGER_SUBSCRIPT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "i and j",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Integer row and column subscripts are decoded without binary64 conversion and must use one shared integer datatype.",
    }];
const SPARSE_INTEGER_TRIPLET_VALUE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "i and j",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Integer row and column subscripts use one shared datatype and remain exact.",
    },
    BuiltinIntegerInputCapability {
        name: "v",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
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

const SPEYE_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n, [m n], or m and n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every built-in integer class is documented for sparse identity dimensions and is decoded directly into platform extents.",
    }];
pub const SPEYE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "S = speye(integer_dimensions, typename?)",
        inputs: &SPEYE_INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Dimensions affect shape only; typename selects documented double, single, or logical sparse storage.",
    }];

const SPONES_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spones-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spones with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SponesIntegerInputExtension"),
};
pub const SPONES_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SPONES_INTEGER_INPUT_EXTENSION];
const SPONES_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "S",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB sparse storage does not establish typed-integer classes; RunMat admits them only to inspect the exact nonzero pattern.",
    }];
pub const SPONES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "S = spones(integer_A)",
        inputs: &SPONES_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The nonzero pattern is read exactly and emitted as a host sparse double matrix of ones; documented GPU input may use the gather fallback.",
    }];

const SPRAND_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sprand-integer-numeric-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "sprand with typed-integer dimensions, density, or condition profile is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SprandIntegerNumericControlExtension"),
};
pub const SPRAND_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SPRAND_INTEGER_CONTROL_EXTENSION];
const SPRAND_INTEGER_PATTERN_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "S",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All built-in integer classes are documented for the input whose nonzero pattern is copied.",
    }];
const SPRAND_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "m and n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public documentation requires nonnegative integer values but does not establish typed storage; RunMat gates typed dimensions and decodes them exactly.",
    },
    BuiltinIntegerInputCapability {
        name: "density or rc",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public documentation does not establish typed storage for density or condition values; RunMat gates them and requires exact binary64 admission.",
    },
];
pub const SPRAND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "R = sprand(integer_S, typename?)", inputs: &SPRAND_INTEGER_PATTERN_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::OptionDependent, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Only S's exact nonzero pattern is consumed; random values use double unless typename selects single." },
    BuiltinIntegerCapabilityDescriptor { form: "R = sprand(integer_m, integer_n, integer_density/rc, typename?)", inputs: &SPRAND_INTEGER_CONTROL_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::OptionDependent, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Typed controls are gated RunMat extensions: dimensions remain exact structural values, while density and condition values require exact binary64 admission." },
];

const SPDIAGS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spdiags-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spdiags with typed-integer matrix data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SpdiagsIntegerDataExtension"),
};
const SPDIAGS_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spdiags-integer-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spdiags with typed-integer diagonal or dimension controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SpdiagsIntegerControlExtension"),
};
pub const SPDIAGS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    SPDIAGS_INTEGER_DATA_EXTENSION,
    SPDIAGS_INTEGER_CONTROL_EXTENSION,
];
const SPDIAGS_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A, B, or Bin",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented matrix storage is single, double, or logical; RunMat gates typed matrix values and processes them directly in their integer class.",
    }];
const SPDIAGS_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "d", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Diagonal offsets are documented as integer-valued but typed storage is not established; RunMat gates typed offsets and decodes them exactly." },
    BuiltinIntegerInputCapability { name: "m and n", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Output dimensions are documented as nonnegative integers but typed storage is not established; RunMat gates typed dimensions and decodes them exactly." },
];
pub const SPDIAGS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "B/S = spdiags(integer_A/B/Bin, ...)", inputs: &SPDIAGS_INTEGER_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Typed matrix data is a gated extension. Extraction and construction preserve the integer class exactly; replacement follows the target class, and duplicate same-class integer diagonals add with saturation." },
    BuiltinIntegerCapabilityDescriptor { form: "B/S = spdiags(A/B, integer_d, integer_m?, integer_n?)", inputs: &SPDIAGS_INTEGER_CONTROL_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Typed offsets and dimensions are gated RunMat extensions decoded exactly before diagonal selection or allocation." },
];

const SPARSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse matrix.",
}];

const SPARSE_INPUT_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Full or sparse matrix to convert.",
}];

const SPARSE_INPUT_DIMS: [BuiltinParamDescriptor; 2] = [
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
        description: "Sparse storage type: double or single.",
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

const SPARSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPARSE.INVALID_INPUT",
    identifier: Some("RunMat:sparse:InvalidInput"),
    when: "Inputs are not a supported sparse construction form.",
    message: "sparse: invalid input",
};

const SPARSE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPARSE.INVALID_INDEX",
    identifier: Some("RunMat:sparse:InvalidIndex"),
    when: "Row or column subscripts are nonpositive, noninteger, or outside explicit dimensions.",
    message: "sparse: invalid index",
};

const SPARSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPARSE.INTERNAL",
    identifier: Some("RunMat:sparse:Internal"),
    when: "Sparse matrix materialisation fails internally.",
    message: "sparse: internal error",
};

const SPARSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::sparse")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("sparse"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Sparse matrices are host-resident CSC values. GPU inputs are gathered before sparse construction because RunMat's acceleration API currently exposes dense tensor handles only.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::sparse")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sparse construction is a representation-changing operation and is not fused.",
};

fn sparse_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args {
        [Type::Tensor { shape: Some(shape) }] => Type::Tensor {
            shape: Some(shape.clone()),
        },
        [Type::Logical { shape: Some(shape) }] => Type::Tensor {
            shape: Some(shape.clone()),
        },
        [Type::Num | Type::Int, Type::Num | Type::Int] => Type::Tensor {
            shape: Some(vec![None, None]),
        },
        _ => Type::tensor(),
    }
}

fn sparse_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "sparse",
    category = "array/creation",
    summary = "Create sparse matrices from full arrays or row/column/value triplets.",
    keywords = "sparse,csc,matrix,nonzero,gpu",
    accel = "custom",
    type_resolver(sparse_type),
    descriptor(crate::builtins::array::creation::sparse::SPARSE_DESCRIPTOR),
    extensions(crate::compatibility::SPARSE_EXTENSIONS),
    integer_capabilities(crate::builtins::array::creation::sparse::SPARSE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn sparse_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    reject_disabled_integer_sparse_constructor(&args)?;
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gpu_helpers::gather_value_async(&arg).await?);
    }
    construct_sparse(gathered).map(Value::SparseTensor)
}

#[runtime_builtin(
    name = "speye",
    category = "array/creation",
    summary = "Create a sparse identity matrix.",
    keywords = "speye,sparse,identity,matrix",
    accel = "custom",
    descriptor(crate::builtins::array::creation::sparse::SPEYE_DESCRIPTOR),
    integer_capabilities(crate::builtins::array::creation::sparse::SPEYE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::sparse"
)]
fn speye_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (rows, cols, dtype) = parse_speye_shape(&args)?;
    Ok(Value::SparseTensor(sparse_identity(rows, cols, dtype)?))
}

#[runtime_builtin(
    name = "nonzeros",
    category = "array/creation",
    summary = "Return nonzero matrix elements in column order.",
    keywords = "nonzeros,sparse,nonzero,column vector",
    accel = "custom",
    descriptor(crate::builtins::array::creation::sparse::NONZEROS_DESCRIPTOR),
    integer_capabilities(crate::builtins::array::creation::sparse::NONZEROS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn nonzeros_builtin(value: Value) -> BuiltinResult<Value> {
    let Value::GpuTensor(source) = value else {
        return nonzeros_value(&value);
    };
    let owner = gpu_helpers::exact_provider_for_handle(&source).ok_or_else(|| {
        sparse_error(
            &SPARSE_ERROR_INTERNAL,
            "nonzeros: no acceleration provider owns the input handle",
        )
    })?;
    let host = gpu_helpers::download_value_preserving_residency_async(owner, &source).await?;
    let output = nonzeros_value(&host)?;
    let output = gpu_helpers::restore_class_preserving_value(&source, output, "nonzeros")?;
    if runmat_accelerate_api::handle_is_explicit(&source) && !matches!(output, Value::GpuTensor(_))
    {
        return Err(sparse_error(
            &SPARSE_ERROR_INTERNAL,
            "nonzeros: provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(output)
}

#[runtime_builtin(
    name = "spones",
    category = "array/creation",
    summary = "Replace nonzero sparse matrix elements with ones.",
    keywords = "spones,sparse,pattern,nonzero",
    accel = "custom",
    descriptor(crate::builtins::array::creation::sparse::SPONES_DESCRIPTOR),
    extensions(crate::builtins::array::creation::sparse::SPONES_EXTENSIONS),
    integer_capabilities(crate::builtins::array::creation::sparse::SPONES_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn spones_builtin(value: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SPONES_INTEGER_INPUT_EXTENSION,
            "spones",
        )?;
    }
    let sparse = sparse_pattern_from_value(gpu_helpers::gather_value_async(&value).await?)?;
    let nnz = sparse.nnz();
    Ok(Value::SparseTensor(
        SparseTensor::new(
            sparse.rows,
            sparse.cols,
            sparse.col_ptrs,
            sparse.row_indices,
            vec![1.0; nnz],
        )
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("spones: {err}")))?,
    ))
}

#[runtime_builtin(
    name = "sprand",
    category = "array/creation",
    summary = "Create a sparse uniformly distributed random matrix.",
    keywords = "sprand,sparse,random,density",
    accel = "custom",
    descriptor(crate::builtins::array::creation::sparse::SPRAND_DESCRIPTOR),
    extensions(crate::builtins::array::creation::sparse::SPRAND_EXTENSIONS),
    integer_capabilities(crate::builtins::array::creation::sparse::SPRAND_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn sprand_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(args.len(), 3 | 4) {
        for value in &args[..2] {
            if crate::builtins::common::validation::value_has_native_integer_class(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &SPRAND_INTEGER_CONTROL_EXTENSION,
                    "sprand",
                )?;
            }
        }
        for value in &args[2..] {
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                value,
                &SPRAND_INTEGER_CONTROL_EXTENSION,
                "sprand",
                "density-or-condition",
            )
            .await?;
        }
    }
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gpu_helpers::gather_value_async(&arg).await?);
    }
    Ok(Value::SparseTensor(sprand_sparse(gathered)?))
}

#[runtime_builtin(
    name = "spdiags",
    category = "array/creation",
    summary = "Extract sparse diagonals or create sparse diagonal matrices.",
    keywords = "spdiags,sparse,diagonal,banded",
    accel = "custom",
    descriptor(crate::builtins::array::creation::sparse::SPDIAGS_DESCRIPTOR),
    extensions(crate::builtins::array::creation::sparse::SPDIAGS_EXTENSIONS),
    integer_capabilities(crate::builtins::array::creation::sparse::SPDIAGS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn spdiags_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if let Some(data) = args.first() {
        if crate::builtins::common::validation::value_has_native_integer_class(data) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SPDIAGS_INTEGER_DATA_EXTENSION,
                "spdiags",
            )?;
        }
    }
    if let Some(offsets) = args.get(1) {
        if crate::builtins::common::validation::value_has_native_integer_class(offsets) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SPDIAGS_INTEGER_CONTROL_EXTENSION,
                "spdiags",
            )?;
        }
    }
    if args.len() == 3 {
        if crate::builtins::common::validation::value_has_native_integer_class(&args[2]) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SPDIAGS_INTEGER_DATA_EXTENSION,
                "spdiags",
            )?;
        }
    } else if args.len() == 4 {
        for dimension in &args[2..] {
            if crate::builtins::common::validation::value_has_native_integer_class(dimension) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &SPDIAGS_INTEGER_CONTROL_EXTENSION,
                    "spdiags",
                )?;
            }
        }
    }
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gpu_helpers::gather_value_async(&arg).await?);
    }
    spdiags_value(gathered)
}

fn construct_sparse(args: Vec<Value>) -> BuiltinResult<SparseTensor> {
    reject_disabled_integer_sparse_constructor(&args)?;
    match args.len() {
        1 => sparse_from_value(args.into_iter().next().expect("one argument")),
        2 => {
            let mut iter = args.into_iter();
            let rows = parse_size_arg(iter.next().as_ref().expect("rows"), "m")?;
            let cols = parse_size_arg(iter.next().as_ref().expect("cols"), "n")?;
            Ok(SparseTensor::zeros(rows, cols))
        }
        3 if keyword_of(&args[2]).is_some() => {
            let rows = parse_size_arg(&args[0], "m")?;
            let cols = parse_size_arg(&args[1], "n")?;
            match sparse_storage_class(&args[2], "sparse")? {
                MatrixClass::Floating(NumericDType::F64) => Ok(SparseTensor::zeros(rows, cols)),
                MatrixClass::Floating(NumericDType::F32) => Ok(SparseTensor::zeros_f32(rows, cols)),
                MatrixClass::Logical => Ok(SparseTensor::zeros_logical(rows, cols)),
                MatrixClass::Floating(_) => {
                    unreachable!("sparse typename parser returned integer dtype")
                }
            }
        }
        3 | 5 | 6 => sparse_from_triplet_form(args),
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sparse: expected sparse(A), sparse(m,n), or sparse(i,j,v[,m,n[,nzmax]])",
        )),
    }
}

fn reject_disabled_integer_sparse_constructor(args: &[Value]) -> BuiltinResult<()> {
    let integer_payload = match args {
        [value] => value_is_real_integer(value),
        [_, _, value] if keyword_of(value).is_none() => value_is_real_integer(value),
        [_, _, value, _, _] | [_, _, value, _, _, _] => value_is_real_integer(value),
        _ => false,
    };
    if integer_payload {
        crate::compatibility::ensure_sparse_integer_extension_enabled("sparse")?;
    }
    Ok(())
}

fn value_is_real_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::SparseTensor(sparse) => sparse.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn parse_speye_shape(args: &[Value]) -> BuiltinResult<(usize, usize, MatrixClass)> {
    let mut args = args;
    let mut class = MatrixClass::Floating(NumericDType::F64);
    if let Some((last, rest)) = args.split_last() {
        if keyword_of(last).is_some() {
            class = sparse_storage_class(last, "speye")?;
            args = rest;
        }
    }

    let shape = match args {
        [] => Ok((1, 1)),
        [n] => match n {
            Value::Tensor(tensor)
                if is_vector_shape(&tensor.shape)
                    && tensor_utils::tensor_element_len(tensor) == 2 =>
            {
                let dims = parse_speye_shape_tensor(tensor)?;
                let rows = dims[0];
                let cols = dims[1];
                Ok((rows, cols))
            }
            Value::SparseTensor(sparse) if is_vector_shape(&[sparse.rows, sparse.cols]) => {
                let dense = if sparse.is_logical() {
                    let logical = sparse.to_dense_logical().map_err(|err| {
                        sparse_error(&SPARSE_ERROR_INTERNAL, format!("speye: {err}"))
                    })?;
                    Tensor::new(
                        logical.data.into_iter().map(f64::from).collect(),
                        logical.shape,
                    )
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("speye: {err}")))?
                } else {
                    sparse.to_dense().map_err(|err| {
                        sparse_error(&SPARSE_ERROR_INTERNAL, format!("speye: {err}"))
                    })?
                };
                if dense.len() != 2 {
                    return Err(sparse_error(
                        &SPARSE_ERROR_INVALID_INPUT,
                        "speye: size vector must have two elements",
                    ));
                }
                let dims = parse_speye_shape_tensor(&dense)?;
                let rows = dims[0];
                let cols = dims[1];
                Ok((rows, cols))
            }
            _ => {
                let size = parse_speye_size_value(n, "n")?;
                Ok((size, size))
            }
        },
        [m, n] => Ok((
            parse_speye_size_value(m, "m")?,
            parse_speye_size_value(n, "n")?,
        )),
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "speye: expected speye(), speye(n), speye([m n]), or speye(m,n)",
        )),
    }?;
    Ok((shape.0, shape.1, class))
}

fn parse_speye_shape_tensor(tensor: &Tensor) -> BuiltinResult<Vec<usize>> {
    if let Some(values) = tensor_integer_values(tensor) {
        return values
            .iter()
            .zip(["m", "n"])
            .map(|(value, name)| parse_speye_integer_size(value, name))
            .collect();
    }
    tensor_utils::tensor_values_f64_cow(tensor)
        .iter()
        .copied()
        .zip(["m", "n"])
        .map(|(value, name)| parse_speye_size_raw(value, name))
        .collect()
}

fn sparse_float_dtype(value: &Value, label: &str) -> BuiltinResult<NumericDType> {
    let type_name = keyword_of(value).ok_or_else(|| {
        sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("{label}: typename must be \"double\" or \"single\""),
        )
    })?;
    match type_name.as_str() {
        "double" => Ok(NumericDType::F64),
        "single" => Ok(NumericDType::F32),
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("{label}: typename must be \"double\" or \"single\", got {type_name}"),
        )),
    }
}

fn sparse_storage_class(value: &Value, label: &str) -> BuiltinResult<MatrixClass> {
    let type_name = keyword_of(value).ok_or_else(|| {
        sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("{label}: typename must be \"double\", \"single\", or \"logical\""),
        )
    })?;
    match type_name.as_str() {
        "double" => Ok(MatrixClass::Floating(NumericDType::F64)),
        "single" => Ok(MatrixClass::Floating(NumericDType::F32)),
        "logical" => Ok(MatrixClass::Logical),
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "{label}: typename must be \"double\", \"single\", or \"logical\", got {type_name}"
            ),
        )),
    }
}

fn parse_speye_size_value(value: &Value, name: &str) -> BuiltinResult<usize> {
    if let Some(integer) = scalar_integer_value(value) {
        return parse_speye_integer_size(&integer, name);
    }
    let raw = match value {
        Value::Num(n) => *n,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => {
            return Err(sparse_error(
                &SPARSE_ERROR_INVALID_INPUT,
                format!("speye: {name} must be a scalar size"),
            ))
        }
    };
    parse_speye_size_raw(raw, name)
}

fn parse_speye_integer_size(value: &IntValue, name: &str) -> BuiltinResult<usize> {
    if value.try_to_i64().is_some_and(|raw| raw <= 0) {
        return Ok(0);
    }
    value.try_to_usize().ok_or_else(|| {
        sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("speye: {name} exceeds the maximum supported size"),
        )
    })
}

fn parse_speye_size_raw(raw: f64, name: &str) -> BuiltinResult<usize> {
    if !raw.is_finite() || raw.fract() != 0.0 {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("speye: {name} must be an integer size"),
        ));
    }
    if raw <= 0.0 {
        return Ok(0);
    }
    if raw > max_usize_cast_value() {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("speye: {name} exceeds the maximum supported size"),
        ));
    }
    Ok(raw as usize)
}

fn sparse_identity(rows: usize, cols: usize, class: MatrixClass) -> BuiltinResult<SparseTensor> {
    let diagonal = rows.min(cols);
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::with_capacity(diagonal);
    col_ptrs.push(0);
    for col in 0..cols {
        if col < diagonal {
            row_indices.push(col);
        }
        col_ptrs.push(row_indices.len());
    }
    match class {
        MatrixClass::Floating(NumericDType::F64) => {
            SparseTensor::new(rows, cols, col_ptrs, row_indices, vec![1.0; diagonal])
        }
        MatrixClass::Floating(NumericDType::F32) => {
            SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, vec![1.0; diagonal])
        }
        MatrixClass::Logical => SparseTensor::new_logical(rows, cols, col_ptrs, row_indices),
        MatrixClass::Floating(_) => unreachable!("sparse identity requested with integer dtype"),
    }
    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("speye: {err}")))
}

fn nonzeros_value(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::SparseTensor(sparse) => {
            if sparse.is_logical() {
                return LogicalArray::new(vec![1; sparse.nnz()], vec![sparse.nnz(), 1])
                    .map(Value::LogicalArray)
                    .map_err(|err| {
                        sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}"))
                    });
            }
            if let Some(storage) = sparse.integer_storage() {
                return integer_tensor_column(storage.clone(), "nonzeros");
            }
            if let Some(values) = sparse.as_f32_slice() {
                let values = values.to_vec();
                let len = values.len();
                return Tensor::from_f32(values, vec![len, 1])
                    .map(Value::Tensor)
                    .map_err(|err| {
                        sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}"))
                    });
            }
            tensor_column(
                sparse
                    .as_f64_slice()
                    .expect("double sparse storage")
                    .to_vec(),
                "nonzeros",
            )
        }
        Value::Tensor(tensor) => nonzeros_dense_tensor(tensor),
        Value::ComplexTensor(tensor) => {
            let indices = (0..tensor.len())
                .filter(|&index| match tensor.complex_storage() {
                    ComplexStorage::F64(values) => {
                        let (real, imag) = values[index];
                        is_stored_value(real) || is_stored_value(imag)
                    }
                    ComplexStorage::F32(values) => {
                        let (real, imag) = values[index];
                        real.is_nan() || imag.is_nan() || real != 0.0 || imag != 0.0
                    }
                    ComplexStorage::Integer(storage) => storage
                        .is_nonzero_at(index)
                        .expect("complex integer storage length matches tensor shape"),
                })
                .collect::<Vec<_>>();
            let storage = tensor
                .complex_storage()
                .gather(&indices)
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}")))?;
            ComplexTensor::from_complex_storage(storage, vec![indices.len(), 1])
                .map(Value::ComplexTensor)
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}")))
        }
        Value::LogicalArray(logical) => {
            let data = logical
                .data
                .iter()
                .filter_map(|&bit| if bit != 0 { Some(1) } else { None })
                .collect::<Vec<_>>();
            let len = data.len();
            LogicalArray::new(data, vec![len, 1])
                .map(Value::LogicalArray)
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}")))
        }
        Value::CharArray(array) => {
            let data = array
                .data
                .iter()
                .copied()
                .filter(|value| *value != '\0')
                .collect::<Vec<_>>();
            let len = data.len();
            CharArray::new(data, len, 1)
                .map(Value::CharArray)
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}")))
        }
        Value::Num(n) => tensor_column(
            if is_stored_value(*n) {
                vec![*n]
            } else {
                Vec::new()
            },
            "nonzeros",
        ),
        Value::Int(i) => integer_tensor_column(integer_storage_from_scalar(i), "nonzeros"),
        Value::Bool(b) => LogicalArray::new(
            if *b { vec![1] } else { Vec::new() },
            vec![usize::from(*b), 1],
        )
        .map(Value::LogicalArray)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}"))),
        other => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("nonzeros: unsupported input {other:?}"),
        )),
    }
}

fn integer_storage_from_scalar(value: &IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => {
            IntegerStorage::I8(((*value != 0).then_some(*value)).into_iter().collect())
        }
        IntValue::I16(value) => {
            IntegerStorage::I16(((*value != 0).then_some(*value)).into_iter().collect())
        }
        IntValue::I32(value) => {
            IntegerStorage::I32(((*value != 0).then_some(*value)).into_iter().collect())
        }
        IntValue::I64(value) => {
            IntegerStorage::I64(((*value != 0).then_some(*value)).into_iter().collect())
        }
        IntValue::U8(value) => {
            IntegerStorage::U8(((*value != 0).then_some(*value)).into_iter().collect())
        }
        IntValue::U16(value) => {
            IntegerStorage::U16(((*value != 0).then_some(*value)).into_iter().collect())
        }
        IntValue::U32(value) => {
            IntegerStorage::U32(((*value != 0).then_some(*value)).into_iter().collect())
        }
        IntValue::U64(value) => {
            IntegerStorage::U64(((*value != 0).then_some(*value)).into_iter().collect())
        }
    }
}

fn tensor_column(data: Vec<f64>, label: &str) -> BuiltinResult<Value> {
    Tensor::new(data.clone(), vec![data.len(), 1])
        .map(Value::Tensor)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
}

fn integer_tensor_column(storage: IntegerStorage, label: &str) -> BuiltinResult<Value> {
    let len = storage.len();
    Tensor::new_integer(storage, vec![len, 1])
        .map(Value::Tensor)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
}

fn nonzeros_dense_tensor(tensor: &Tensor) -> BuiltinResult<Value> {
    if let Some(storage) = tensor.integer_storage() {
        let values = storage
            .exact_values()
            .into_iter()
            .filter(|value| !value.is_zero())
            .collect();
        let storage = storage
            .from_same_class_values(values)
            .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}")))?;
        return integer_tensor_column(storage, "nonzeros");
    }
    let values = tensor_utils::tensor_values_f64_cow(tensor)
        .iter()
        .copied()
        .filter(|value| is_stored_value(*value))
        .collect::<Vec<_>>();
    if tensor.numeric_dtype() == NumericDType::F32 {
        let values = values
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let len = values.len();
        return Tensor::from_f32(values, vec![len, 1])
            .map(Value::Tensor)
            .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}")));
    }
    tensor_column(values, "nonzeros")
}

fn sparse_pattern_from_value(value: Value) -> BuiltinResult<SparseTensor> {
    sparse_from_value(value)
}

fn sprand_sparse(mut args: Vec<Value>) -> BuiltinResult<SparseTensor> {
    let dtype = take_optional_sparse_float_typename(&mut args, "sprand")?;
    let sparse = match args.len() {
        1 => {
            let pattern = sparse_pattern_from_value(args.into_iter().next().expect("pattern"))?;
            let values = random::generate_uniform(pattern.nnz(), "sprand")?;
            SparseTensor::new(
                pattern.rows,
                pattern.cols,
                pattern.col_ptrs,
                pattern.row_indices,
                values,
            )
            .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sprand: {err}")))?
        }
        3 | 4 => {
            let rows = parse_size_arg(&args[0], "m")?;
            let cols = parse_size_arg(&args[1], "n")?;
            let density = parse_density_arg(&args[2])?;
            if args.len() == 4 {
                let profile = parse_sprand_condition_profile(&args[3], rows, cols)?;
                sprand_from_condition_profile(rows, cols, density, &profile)?
            } else {
                sprand_from_density(rows, cols, density)?
            }
        }
        _ => {
            return Err(sparse_error(
                &SPARSE_ERROR_INVALID_INPUT,
                "sprand: expected sprand(S), sprand(m,n,density), or sprand(m,n,density,rc)",
            ))
        }
    };
    sparse_with_float_dtype(sparse, dtype, "sprand")
}

fn take_optional_sparse_float_typename(
    args: &mut Vec<Value>,
    label: &str,
) -> BuiltinResult<NumericDType> {
    if args.last().and_then(keyword_of).is_none() {
        return Ok(NumericDType::F64);
    }
    let value = args.pop().expect("typename argument exists");
    sparse_float_dtype(&value, label)
}

fn sparse_with_float_dtype(
    sparse: SparseTensor,
    dtype: NumericDType,
    label: &str,
) -> BuiltinResult<SparseTensor> {
    match dtype {
        NumericDType::F64 => Ok(sparse),
        NumericDType::F32 => {
            let values = sparse
                .materialize_f64()
                .into_iter()
                .map(|value| value as f32)
                .collect();
            SparseTensor::new_f32(
                sparse.rows,
                sparse.cols,
                sparse.col_ptrs,
                sparse.row_indices,
                values,
            )
            .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
        }
        _ => unreachable!("sparse floating typename parser returned integer dtype"),
    }
}

fn parse_density_arg(value: &Value) -> BuiltinResult<f64> {
    let density = scalar_f64(value, "density", "sprand")?;
    if !density.is_finite() || !(0.0..=1.0).contains(&density) {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sprand: density must be finite and between 0 and 1",
        ));
    }
    Ok(density)
}

fn sprand_from_density(rows: usize, cols: usize, density: f64) -> BuiltinResult<SparseTensor> {
    let total = rows.checked_mul(cols).ok_or_else(|| {
        sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sprand: matrix dimensions overflow",
        )
    })?;
    if total == 0 || density == 0.0 {
        return Ok(SparseTensor::zeros(rows, cols));
    }
    let target = ((total as f64) * density).round().clamp(0.0, total as f64) as usize;
    if target == 0 {
        return Ok(SparseTensor::zeros(rows, cols));
    }
    if target > SPARSE_HELPER_DENSE_INPUT_LIMIT {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "sprand: requested sparse pattern has {target} stored entries, exceeding safe threshold"
            ),
        ));
    }

    let positions = sample_unique_positions(total, target)?;
    let values = random::generate_uniform(target, "sprand")?;
    let mut entries = BTreeMap::new();
    for (linear, value) in positions.into_iter().zip(values) {
        let row = linear % rows;
        let col = linear / rows;
        entries.insert((col, row), value);
    }
    sparse_from_entries(rows, cols, entries, "sprand")
}

fn parse_sprand_condition_profile(
    value: &Value,
    rows: usize,
    cols: usize,
) -> BuiltinResult<Vec<f64>> {
    let rank_limit = rows.min(cols);
    if rank_limit == 0 {
        let values = numeric_vector_for_label(value, "rc", "sprand")?;
        if values
            .iter()
            .any(|&value| !valid_sprand_condition_value(value))
        {
            return Err(invalid_sprand_condition_error());
        }
        return Ok(Vec::new());
    }

    let values = numeric_vector_for_label(value, "rc", "sprand")?;
    if values.is_empty() || values.len() > rank_limit {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("sprand: rc must have between 1 and {rank_limit} elements"),
        ));
    }
    if values
        .iter()
        .any(|&value| !valid_sprand_condition_value(value))
    {
        return Err(invalid_sprand_condition_error());
    }

    if values.len() == 1 {
        Ok(scalar_rcond_singular_profile(rank_limit, values[0]))
    } else {
        Ok(values)
    }
}

fn valid_sprand_condition_value(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn invalid_sprand_condition_error() -> RuntimeError {
    sparse_error(
        &SPARSE_ERROR_INVALID_INPUT,
        "sprand: rc values must be finite and between 0 and 1",
    )
}

fn scalar_rcond_singular_profile(rank_limit: usize, rcond: f64) -> Vec<f64> {
    match rank_limit {
        0 => Vec::new(),
        1 => vec![1.0],
        _ if rcond == 0.0 => (0..rank_limit)
            .map(|idx| 1.0 - (idx as f64 / (rank_limit - 1) as f64))
            .collect(),
        _ => (0..rank_limit)
            .map(|idx| rcond.powf(idx as f64 / (rank_limit - 1) as f64))
            .collect(),
    }
}

fn sprand_from_condition_profile(
    rows: usize,
    cols: usize,
    density: f64,
    singular_values: &[f64],
) -> BuiltinResult<SparseTensor> {
    let total = rows.checked_mul(cols).ok_or_else(|| {
        sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sprand: matrix dimensions overflow",
        )
    })?;
    if total == 0 || density == 0.0 || singular_values.is_empty() {
        return Ok(SparseTensor::zeros(rows, cols));
    }
    if rows == 1 || cols == 1 {
        return sprand_condition_vector(rows, cols, density, singular_values);
    }
    if total > SPRAND_CONDITION_DENSE_INPUT_LIMIT {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "sprand: condition-number form requires bounded dense working storage and refuses {total} elements"
            ),
        ));
    }

    let target = ((total as f64) * density).round().clamp(0.0, total as f64) as usize;
    if target == 0 {
        return Ok(SparseTensor::zeros(rows, cols));
    }
    if target > SPARSE_HELPER_DENSE_INPUT_LIMIT {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "sprand: requested sparse pattern has {target} stored entries, exceeding safe threshold"
            ),
        ));
    }

    let dense = condition_profile_sparse_rotation_matrix(rows, cols, singular_values, target)?;
    let mut entries = BTreeMap::new();
    for (linear, value) in dense.into_iter().enumerate() {
        if is_stored_value(value) {
            let row = linear % rows;
            let col = linear / rows;
            entries.insert((col, row), value);
        }
    }
    sparse_from_entries(rows, cols, entries, "sprand")
}

fn sprand_condition_vector(
    rows: usize,
    cols: usize,
    density: f64,
    singular_values: &[f64],
) -> BuiltinResult<SparseTensor> {
    let total = rows.checked_mul(cols).ok_or_else(|| {
        sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sprand: matrix dimensions overflow",
        )
    })?;
    if total == 0 || density == 0.0 {
        return Ok(SparseTensor::zeros(rows, cols));
    }
    let sigma = singular_values.first().copied().unwrap_or(0.0);
    if sigma == 0.0 {
        return Ok(SparseTensor::zeros(rows, cols));
    }
    let target = ((total as f64) * density).round().clamp(1.0, total as f64) as usize;
    if target > SPARSE_HELPER_DENSE_INPUT_LIMIT {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "sprand: requested sparse pattern has {target} stored entries, exceeding safe threshold"
            ),
        ));
    }

    let positions = sample_unique_positions(total, target)?;
    let draws = random::generate_uniform(target, "sprand")?;
    let norm = draws.iter().map(|value| value * value).sum::<f64>().sqrt();
    let scale = if norm > 0.0 { sigma / norm } else { sigma };
    let mut entries = BTreeMap::new();
    for (linear, draw) in positions.into_iter().zip(draws) {
        let row = linear % rows;
        let col = linear / rows;
        entries.insert((col, row), draw * scale);
    }
    sparse_from_entries(rows, cols, entries, "sprand")
}

fn condition_profile_sparse_rotation_matrix(
    rows: usize,
    cols: usize,
    singular_values: &[f64],
    target_nnz: usize,
) -> BuiltinResult<Vec<f64>> {
    let rank = singular_values.len().min(rows.min(cols));
    let mut dense = vec![0.0; rows * cols];
    for component in 0..rank {
        dense[component + component * rows] = singular_values[component];
    }

    let mut current_nnz = dense
        .iter()
        .filter(|&&value| is_stored_value(value))
        .count();
    if current_nnz >= target_nnz {
        return Ok(dense);
    }

    let mut work = 0usize;
    let mut attempts = 0usize;
    let mut stale_attempts = 0usize;
    while current_nnz < target_nnz && attempts < SPRAND_CONDITION_MAX_ROTATION_ATTEMPTS {
        let prefer_rows = rows > 1 && (attempts.is_multiple_of(2) || cols <= 1);
        let rotation_cost = if prefer_rows { cols } else { rows };
        work = work.checked_add(rotation_cost).ok_or_else(|| {
            sparse_error(
                &SPARSE_ERROR_INVALID_INPUT,
                "sprand: condition-number rotation work overflow",
            )
        })?;
        if work > SPRAND_CONDITION_ROTATION_WORK_LIMIT {
            return Err(sparse_error(
                &SPARSE_ERROR_INVALID_INPUT,
                "sprand: condition-number form exceeded bounded plane-rotation work before reaching requested density",
            ));
        }

        let (c, s) = random_plane_rotation("sprand")?;
        if prefer_rows {
            let first = attempts % rows;
            let second = (first + 1 + stale_attempts.min(rows.saturating_sub(2))) % rows;
            apply_row_rotation(&mut dense, rows, cols, first, second, c, s);
        } else {
            let first = attempts % cols;
            let second = (first + 1 + stale_attempts.min(cols.saturating_sub(2))) % cols;
            apply_col_rotation(&mut dense, rows, cols, first, second, c, s);
        }

        attempts += 1;
        let next_nnz = dense
            .iter()
            .filter(|&&value| is_stored_value(value))
            .count();
        if next_nnz > current_nnz {
            current_nnz = next_nnz;
            stale_attempts = 0;
        } else {
            stale_attempts += 1;
        }
    }

    if current_nnz < target_nnz {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sprand: condition-number form exceeded bounded plane-rotation attempts before reaching requested density",
        ));
    }

    Ok(dense)
}

fn random_plane_rotation(label: &str) -> BuiltinResult<(f64, f64)> {
    let draw = random::generate_uniform(1, label)?[0];
    let theta = (0.1 + 0.8 * draw) * std::f64::consts::FRAC_PI_2;
    Ok((theta.cos(), theta.sin()))
}

fn apply_row_rotation(
    dense: &mut [f64],
    rows: usize,
    cols: usize,
    first: usize,
    second: usize,
    c: f64,
    s: f64,
) {
    if first == second {
        return;
    }
    for col in 0..cols {
        let first_idx = first + col * rows;
        let second_idx = second + col * rows;
        let a = dense[first_idx];
        let b = dense[second_idx];
        dense[first_idx] = c * a + s * b;
        dense[second_idx] = -s * a + c * b;
    }
}

fn apply_col_rotation(
    dense: &mut [f64],
    rows: usize,
    _cols: usize,
    first: usize,
    second: usize,
    c: f64,
    s: f64,
) {
    if first == second {
        return;
    }
    let first_base = first * rows;
    let second_base = second * rows;
    for row in 0..rows {
        let first_idx = first_base + row;
        let second_idx = second_base + row;
        let a = dense[first_idx];
        let b = dense[second_idx];
        dense[first_idx] = c * a + s * b;
        dense[second_idx] = -s * a + c * b;
    }
}

fn sample_unique_positions(total: usize, target: usize) -> BuiltinResult<Vec<usize>> {
    if target == total {
        return Ok((0..total).collect());
    }

    if target > total / 2 {
        let omitted = sample_unique_positions_floyd(total, total - target)?;
        let mut positions = Vec::with_capacity(target);
        for position in 0..total {
            if !omitted.contains(&position) {
                positions.push(position);
            }
        }
        return Ok(positions);
    }

    Ok(sample_unique_positions_floyd(total, target)?
        .into_iter()
        .collect())
}

fn sample_unique_positions_floyd(total: usize, target: usize) -> BuiltinResult<BTreeSet<usize>> {
    let mut positions = BTreeSet::new();
    if target == 0 {
        return Ok(positions);
    }

    let draws = random::generate_uniform(target, "sprand")?;
    for (offset, draw) in draws.into_iter().enumerate() {
        let upper = total - target + offset;
        let candidate = ((draw * (upper + 1) as f64).floor() as usize).min(upper);
        if positions.contains(&candidate) {
            positions.insert(upper);
        } else {
            positions.insert(candidate);
        }
    }

    debug_assert_eq!(positions.len(), target);
    Ok(positions)
}

enum ExtractionMatrix {
    Dense(DenseMatrix),
    Sparse(SparseTensor),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixClass {
    Floating(NumericDType),
    Logical,
}

impl ExtractionMatrix {
    fn rows(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.rows,
            Self::Sparse(sparse) => sparse.rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.cols,
            Self::Sparse(sparse) => sparse.cols,
        }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        match self {
            Self::Dense(matrix) => matrix.get(row, col),
            Self::Sparse(sparse) => sparse.get(row, col).unwrap_or(0.0),
        }
    }

    fn class(&self) -> MatrixClass {
        match self {
            Self::Dense(matrix) => matrix.class,
            Self::Sparse(sparse) if sparse.is_logical() => MatrixClass::Logical,
            Self::Sparse(sparse) if sparse.numeric_dtype() == Some(NumericDType::F32) => {
                MatrixClass::Floating(NumericDType::F32)
            }
            Self::Sparse(_) => MatrixClass::Floating(NumericDType::F64),
        }
    }
}

fn extraction_matrix_from_value(value: &Value, label: &str) -> BuiltinResult<ExtractionMatrix> {
    match value {
        Value::SparseTensor(sparse) => Ok(ExtractionMatrix::Sparse(sparse.clone())),
        _ => dense_matrix_from_value(value, label).map(ExtractionMatrix::Dense),
    }
}

fn add_entry(entries: &mut BTreeMap<(usize, usize), f64>, col: usize, row: usize, value: f64) {
    let key = (col, row);
    let entry = entries.entry(key).or_insert(0.0);
    *entry += value;
    if !is_stored_value(*entry) {
        entries.remove(&key);
    }
}

fn add_integer_entry(
    entries: &mut BTreeMap<(usize, usize), IntValue>,
    col: usize,
    row: usize,
    value: IntValue,
) -> BuiltinResult<()> {
    let key = (col, row);
    if let Some(current) = entries.get(&key) {
        let sum = current.saturating_add(&value).map_err(|err| {
            sparse_error(
                &SPARSE_ERROR_INTERNAL,
                format!("spdiags: cannot combine duplicate integer diagonals: {err}"),
            )
        })?;
        if sum.is_zero() {
            entries.remove(&key);
        } else {
            entries.insert(key, sum);
        }
    } else if !value.is_zero() {
        entries.insert(key, value);
    }
    Ok(())
}

fn spdiags_value(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.len() {
        1 => {
            if let Some(matrix) = integer_extraction_matrix_from_value(&args[0], "spdiags") {
                let matrix = matrix?;
                let offsets = nonzero_integer_diagonal_offsets(&matrix);
                let bout = extract_integer_diagonals(&matrix, &offsets)?;
                let id = tensor_column(offsets.iter().map(|&d| d as f64).collect(), "spdiags")?;
                return match crate::output_count::current_output_count() {
                    Some(0) => Ok(Value::OutputList(Vec::new())),
                    Some(1) => Ok(Value::OutputList(vec![bout])),
                    Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                        out_count,
                        vec![bout, id],
                    )),
                    None => Ok(bout),
                };
            }
            let matrix = extraction_matrix_from_value(&args[0], "spdiags")?;
            let offsets = nonzero_diagonal_offsets(&matrix);
            let bout = extract_diagonals(&matrix, &offsets)?;
            let id = tensor_column(offsets.iter().map(|&d| d as f64).collect(), "spdiags")?;
            match crate::output_count::current_output_count() {
                Some(0) => Ok(Value::OutputList(Vec::new())),
                Some(1) => Ok(Value::OutputList(vec![bout])),
                Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                    out_count,
                    vec![bout, id],
                )),
                None => Ok(bout),
            }
        }
        2 => {
            if let Some(matrix) = integer_extraction_matrix_from_value(&args[0], "spdiags") {
                let matrix = matrix?;
                let offsets = parse_diag_offsets(&args[1], "spdiags")?;
                return extract_integer_diagonals(&matrix, &offsets);
            }
            let matrix = extraction_matrix_from_value(&args[0], "spdiags")?;
            let offsets = parse_diag_offsets(&args[1], "spdiags")?;
            extract_diagonals(&matrix, &offsets)
        }
        3 => {
            let target = sparse_from_value(args[2].clone())?;
            if target.integer_storage().is_some() {
                replace_sparse_integer_diagonals(&args[0], &args[1], target)
                    .map(Value::SparseTensor)
            } else {
                replace_sparse_diagonals(&args[0], &args[1], &args[2]).map(Value::SparseTensor)
            }
        }
        4 => {
            let offsets = parse_diag_offsets(&args[1], "spdiags")?;
            let rows = parse_size_arg(&args[2], "m")?;
            let cols = parse_size_arg(&args[3], "n")?;
            if let Some(bin) = integer_dense_matrix_from_value(&args[0], "spdiags") {
                construct_sparse_integer_diagonals(&bin?, &offsets, rows, cols)
                    .map(Value::SparseTensor)
            } else {
                construct_sparse_diagonals(&args[0], &offsets, rows, cols).map(Value::SparseTensor)
            }
        }
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "spdiags: expected spdiags(A), spdiags(A,d), spdiags(B,d,m,n), or spdiags(B,d,A)",
        )),
    }
}

#[derive(Clone)]
struct IntegerMatrix {
    rows: usize,
    cols: usize,
    storage: IntegerStorage,
}

impl IntegerMatrix {
    fn get(&self, row: usize, col: usize) -> IntValue {
        self.storage
            .value_at(row + col * self.rows)
            .expect("validated integer matrix shape matches storage")
    }
}

enum IntegerExtractionMatrix {
    Dense(IntegerMatrix),
    Sparse(SparseTensor),
}

impl IntegerExtractionMatrix {
    fn rows(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.rows,
            Self::Sparse(matrix) => matrix.rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.cols,
            Self::Sparse(matrix) => matrix.cols,
        }
    }

    fn prototype(&self) -> &IntegerStorage {
        match self {
            Self::Dense(matrix) => &matrix.storage,
            Self::Sparse(matrix) => matrix
                .integer_storage()
                .expect("integer sparse extraction matrix retains integer storage"),
        }
    }

    fn get(&self, row: usize, col: usize) -> IntValue {
        match self {
            Self::Dense(matrix) => matrix.get(row, col),
            Self::Sparse(matrix) => matrix.integer_at(row, col).unwrap_or_else(|| {
                matrix
                    .integer_storage()
                    .expect("integer sparse extraction matrix retains integer storage")
                    .zeros_like(1)
                    .value_at(0)
                    .expect("one-element integer zero storage")
            }),
        }
    }
}

fn integer_extraction_matrix_from_value(
    value: &Value,
    label: &str,
) -> Option<BuiltinResult<IntegerExtractionMatrix>> {
    match value {
        Value::SparseTensor(sparse) if sparse.integer_storage().is_some() => {
            Some(Ok(IntegerExtractionMatrix::Sparse(sparse.clone())))
        }
        _ => integer_dense_matrix_from_value(value, label)
            .map(|result| result.map(IntegerExtractionMatrix::Dense)),
    }
}

fn integer_dense_matrix_from_value(
    value: &Value,
    label: &str,
) -> Option<BuiltinResult<IntegerMatrix>> {
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if tensor.shape.len() > 2 {
                return Some(Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("{label}: input must be a 2-D matrix"),
                )));
            }
            Some(Ok(IntegerMatrix {
                rows: tensor.rows(),
                cols: tensor.cols(),
                storage: tensor
                    .integer_storage()
                    .expect("checked integer tensor")
                    .clone(),
            }))
        }
        Value::SparseTensor(sparse) if sparse.integer_storage().is_some() => {
            let total = match sparse.rows.checked_mul(sparse.cols) {
                Some(total) => total,
                None => {
                    return Some(Err(sparse_error(
                        &SPARSE_ERROR_INVALID_INPUT,
                        format!("{label}: sparse dimensions overflow"),
                    )))
                }
            };
            if total > SPARSE_HELPER_DENSE_INPUT_LIMIT {
                return Some(Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!(
                        "{label}: cannot densify sparse matrix with {total} elements for diagonal processing"
                    ),
                )));
            }
            Some(
                sparse
                    .to_dense()
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
                    .map(|tensor| IntegerMatrix {
                        rows: sparse.rows,
                        cols: sparse.cols,
                        storage: tensor
                            .integer_storage()
                            .expect("integer sparse tensor densifies to integer storage")
                            .clone(),
                    }),
            )
        }
        Value::Int(value) => Some(Ok(IntegerMatrix {
            rows: 1,
            cols: 1,
            storage: IntegerStorage::from_scalar(value.clone()),
        })),
        _ => None,
    }
}

#[derive(Clone)]
struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
    class: MatrixClass,
}

impl DenseMatrix {
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }
}

fn dense_matrix_from_value(value: &Value, label: &str) -> BuiltinResult<DenseMatrix> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.shape.len() > 2 {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("{label}: input must be a 2-D matrix"),
                ));
            }
            Ok(DenseMatrix {
                rows: tensor.rows(),
                cols: tensor.cols(),
                data: tensor_utils::tensor_values_f64(tensor),
                class: if tensor.numeric_dtype() == NumericDType::F32 {
                    MatrixClass::Floating(NumericDType::F32)
                } else {
                    MatrixClass::Floating(NumericDType::F64)
                },
            })
        }
        Value::SparseTensor(sparse) => {
            let total = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
                sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("{label}: sparse dimensions overflow"),
                )
            })?;
            if total > SPARSE_HELPER_DENSE_INPUT_LIMIT {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!(
                        "{label}: cannot densify sparse matrix with {total} elements for diagonal processing"
                    ),
                ));
            }
            let data = if sparse.is_logical() {
                sparse
                    .to_dense_logical()
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))?
                    .data
                    .into_iter()
                    .map(f64::from)
                    .collect()
            } else {
                tensor_utils::tensor_into_values_f64(sparse.to_dense().map_err(|err| {
                    sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}"))
                })?)
            };
            Ok(DenseMatrix {
                rows: sparse.rows,
                cols: sparse.cols,
                data,
                class: if sparse.is_logical() {
                    MatrixClass::Logical
                } else if sparse.numeric_dtype() == Some(NumericDType::F32) {
                    MatrixClass::Floating(NumericDType::F32)
                } else {
                    MatrixClass::Floating(NumericDType::F64)
                },
            })
        }
        Value::LogicalArray(logical) => {
            if logical.shape.len() > 2 {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("{label}: logical input must be a 2-D matrix"),
                ));
            }
            let rows = logical.shape.first().copied().unwrap_or(1);
            let cols = logical.shape.get(1).copied().unwrap_or(1);
            Ok(DenseMatrix {
                rows,
                cols,
                data: logical
                    .data
                    .iter()
                    .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                    .collect(),
                class: MatrixClass::Logical,
            })
        }
        Value::Num(n) => Ok(DenseMatrix {
            rows: 1,
            cols: 1,
            data: vec![*n],
            class: MatrixClass::Floating(NumericDType::F64),
        }),
        Value::Int(i) => Ok(DenseMatrix {
            rows: 1,
            cols: 1,
            data: vec![i.to_f64()],
            class: MatrixClass::Floating(NumericDType::F64),
        }),
        Value::Bool(b) => Ok(DenseMatrix {
            rows: 1,
            cols: 1,
            data: vec![if *b { 1.0 } else { 0.0 }],
            class: MatrixClass::Logical,
        }),
        other => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("{label}: unsupported matrix input {other:?}"),
        )),
    }
}

fn nonzero_diagonal_offsets(matrix: &ExtractionMatrix) -> Vec<isize> {
    let mut offsets = BTreeSet::new();
    match matrix {
        ExtractionMatrix::Sparse(sparse) => {
            let values = sparse.materialize_f64();
            for col in 0..sparse.cols {
                for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
                    if is_stored_value(values[idx]) {
                        offsets.insert(col as isize - sparse.row_indices[idx] as isize);
                    }
                }
            }
        }
        ExtractionMatrix::Dense(_) => {
            for col in 0..matrix.cols() {
                for row in 0..matrix.rows() {
                    if is_stored_value(matrix.get(row, col)) {
                        offsets.insert(col as isize - row as isize);
                    }
                }
            }
        }
    }
    offsets.into_iter().collect()
}

fn parse_diag_offsets(value: &Value, label: &str) -> BuiltinResult<Vec<isize>> {
    if let Some(offsets) = parse_integer_diag_offsets(value, label) {
        return offsets;
    }
    numeric_vector(value, "d")?
        .into_iter()
        .map(|raw| {
            if !raw.is_finite() || raw.fract() != 0.0 {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("{label}: diagonal numbers must be finite integers"),
                ));
            }
            if raw < isize::MIN as f64 || raw > isize::MAX as f64 {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("{label}: diagonal number exceeds supported range"),
                ));
            }
            Ok(raw as isize)
        })
        .collect()
}

/// Parse typed diagonal offsets directly from integer storage. Offsets are
/// selectors, so wide values must not pass through floating-point conversion.
fn parse_integer_diag_offsets(value: &Value, label: &str) -> Option<BuiltinResult<Vec<isize>>> {
    let values = match value {
        Value::Int(value) => vec![value.clone()],
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) {
                return None;
            }
            tensor_integer_values(tensor)?
        }
        Value::SparseTensor(sparse) => {
            if !is_vector_shape(&[sparse.rows, sparse.cols]) || sparse.integer_storage().is_none() {
                return None;
            }
            let dense = match dense_typed_triplet_sparse(sparse, "d") {
                Ok(dense) => dense,
                Err(err) => return Some(Err(err)),
            };
            tensor_integer_values(&dense).expect("typed sparse tensor remains typed when densified")
        }
        _ => return None,
    };

    Some(
        values
            .iter()
            .map(|value| {
                value.try_to_isize().ok_or_else(|| {
                    sparse_error(
                        &SPARSE_ERROR_INVALID_INPUT,
                        format!("{label}: diagonal number exceeds supported range"),
                    )
                })
            })
            .collect(),
    )
}

fn nonzero_integer_diagonal_offsets(matrix: &IntegerExtractionMatrix) -> Vec<isize> {
    let mut offsets = BTreeSet::new();
    match matrix {
        IntegerExtractionMatrix::Sparse(sparse) => {
            let storage = sparse
                .integer_storage()
                .expect("integer sparse extraction matrix retains integer storage");
            for col in 0..sparse.cols {
                for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
                    if !storage
                        .value_at(idx)
                        .expect("validated sparse integer storage")
                        .is_zero()
                    {
                        offsets.insert(col as isize - sparse.row_indices[idx] as isize);
                    }
                }
            }
        }
        IntegerExtractionMatrix::Dense(_) => {
            for col in 0..matrix.cols() {
                for row in 0..matrix.rows() {
                    if !matrix.get(row, col).is_zero() {
                        offsets.insert(col as isize - row as isize);
                    }
                }
            }
        }
    }
    offsets.into_iter().collect()
}

fn extract_integer_diagonals(
    matrix: &IntegerExtractionMatrix,
    offsets: &[isize],
) -> BuiltinResult<Value> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let out_rows = rows.min(cols);
    let mut storage = matrix
        .prototype()
        .zeros_like(out_rows.saturating_mul(offsets.len()));
    for (out_col, &offset) in offsets.iter().enumerate() {
        for t in 0..diag_len(rows, cols, offset) {
            let Some((row, col)) = diag_coord(offset, t) else {
                continue;
            };
            if row >= rows || col >= cols {
                continue;
            }
            let out_row = diag_bout_row(offset, t);
            if out_row < out_rows {
                storage
                    .set_value(out_row + out_col * out_rows, matrix.get(row, col))
                    .map_err(|err| {
                        sparse_error(&SPARSE_ERROR_INTERNAL, format!("spdiags: {err}"))
                    })?;
            }
        }
    }
    Tensor::new_integer(storage, vec![out_rows, offsets.len()])
        .map(Value::Tensor)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("spdiags: {err}")))
}

fn construct_sparse_integer_diagonals(
    bin: &IntegerMatrix,
    offsets: &[isize],
    rows: usize,
    cols: usize,
) -> BuiltinResult<SparseTensor> {
    if offsets.is_empty() {
        return Ok(SparseTensor::zeros_with_integer_storage(
            rows,
            cols,
            &bin.storage,
        ));
    }
    if bin.cols != offsets.len() {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "spdiags: Bin must have one column per diagonal, got {} columns for {} diagonals",
                bin.cols,
                offsets.len()
            ),
        ));
    }

    let mut entries = BTreeMap::new();
    for (diag_index, &offset) in offsets.iter().enumerate() {
        for t in 0..diag_len(rows, cols, offset) {
            let Some((row, col)) = diag_coord(offset, t) else {
                continue;
            };
            let source_row = diag_bout_row(offset, t);
            if source_row < bin.rows {
                add_integer_entry(&mut entries, col, row, bin.get(source_row, diag_index))?;
            }
        }
    }
    sparse_integer_from_entries(rows, cols, entries, &bin.storage, "spdiags")
}

fn replace_sparse_integer_diagonals(
    bin: &Value,
    offsets: &Value,
    target: SparseTensor,
) -> BuiltinResult<SparseTensor> {
    let offsets = parse_diag_offsets(offsets, "spdiags")?;
    let prototype = target
        .integer_storage()
        .expect("integer replacement target retains integer storage")
        .clone();
    let selected: BTreeSet<isize> = offsets.iter().copied().collect();
    let mut entries = BTreeMap::new();
    for col in 0..target.cols {
        for idx in target.col_ptrs[col]..target.col_ptrs[col + 1] {
            let row = target.row_indices[idx];
            if !selected.contains(&(col as isize - row as isize)) {
                entries.insert(
                    (col, row),
                    prototype
                        .value_at(idx)
                        .expect("validated sparse integer storage"),
                );
            }
        }
    }

    let mut replacements = BTreeMap::new();
    if let Some(integer_bin) = integer_dense_matrix_from_value(bin, "spdiags") {
        let integer_bin = integer_bin?;
        validate_spdiags_bin_columns(integer_bin.cols, offsets.len())?;
        for (diag_index, &offset) in offsets.iter().enumerate() {
            for t in 0..diag_len(target.rows, target.cols, offset) {
                let Some((row, col)) = diag_coord(offset, t) else {
                    continue;
                };
                let source_row = diag_bout_row(offset, t);
                if source_row < integer_bin.rows {
                    let value =
                        prototype.cast_exact_assignment(&integer_bin.get(source_row, diag_index));
                    add_integer_entry(&mut replacements, col, row, value)?;
                }
            }
        }
    } else {
        let floating_bin = dense_matrix_from_value(bin, "spdiags")?;
        validate_spdiags_bin_columns(floating_bin.cols, offsets.len())?;
        for (diag_index, &offset) in offsets.iter().enumerate() {
            for t in 0..diag_len(target.rows, target.cols, offset) {
                let Some((row, col)) = diag_coord(offset, t) else {
                    continue;
                };
                let source_row = diag_bout_row(offset, t);
                if source_row < floating_bin.rows {
                    let value =
                        prototype.cast_f64_assignment(floating_bin.get(source_row, diag_index));
                    add_integer_entry(&mut replacements, col, row, value)?;
                }
            }
        }
    }
    entries.extend(replacements);
    sparse_integer_from_entries(target.rows, target.cols, entries, &prototype, "spdiags")
}

fn validate_spdiags_bin_columns(actual: usize, expected: usize) -> BuiltinResult<()> {
    if expected != 0 && actual != expected {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "spdiags: Bin must have one column per diagonal, got {actual} columns for {expected} diagonals"
            ),
        ));
    }
    Ok(())
}

fn sparse_integer_from_entries(
    rows: usize,
    cols: usize,
    entries: BTreeMap<(usize, usize), IntValue>,
    prototype: &IntegerStorage,
    label: &str,
) -> BuiltinResult<SparseTensor> {
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    col_ptrs.push(0);
    for col in 0..cols {
        for (&(entry_col, row), value) in entries.range((col, 0)..=(col, usize::MAX)) {
            debug_assert_eq!(entry_col, col);
            if !value.is_zero() {
                row_indices.push(row);
                values.push(value.clone());
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new_integer_like(rows, cols, col_ptrs, row_indices, values, prototype)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
}

fn extract_diagonals(matrix: &ExtractionMatrix, offsets: &[isize]) -> BuiltinResult<Value> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let out_rows = rows.min(cols);
    let mut data = vec![0.0; out_rows.saturating_mul(offsets.len())];
    for (out_col, &offset) in offsets.iter().enumerate() {
        for t in 0..diag_len(rows, cols, offset) {
            let Some((row, col)) = diag_coord(offset, t) else {
                continue;
            };
            if row >= rows || col >= cols {
                continue;
            }
            let out_row = diag_bout_row(offset, t);
            if out_row < out_rows {
                data[out_row + out_col * out_rows] = matrix.get(row, col);
            }
        }
    }
    match matrix.class() {
        MatrixClass::Floating(NumericDType::F32) => Tensor::from_f32(
            data.into_iter().map(|value| value as f32).collect(),
            vec![out_rows, offsets.len()],
        )
        .map(Value::Tensor),
        MatrixClass::Floating(NumericDType::F64) => {
            Tensor::new(data, vec![out_rows, offsets.len()]).map(Value::Tensor)
        }
        MatrixClass::Logical => LogicalArray::new(
            data.into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect(),
            vec![out_rows, offsets.len()],
        )
        .map(Value::LogicalArray),
        MatrixClass::Floating(_) => {
            unreachable!("diagonal extraction normalizes integer inputs to double")
        }
    }
    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("spdiags: {err}")))
}

fn construct_sparse_diagonals(
    bin: &Value,
    offsets: &[isize],
    rows: usize,
    cols: usize,
) -> BuiltinResult<SparseTensor> {
    let bin = dense_matrix_from_value(bin, "spdiags")?;
    if offsets.is_empty() {
        return Ok(match bin.class {
            MatrixClass::Floating(NumericDType::F32) => SparseTensor::zeros_f32(rows, cols),
            MatrixClass::Floating(NumericDType::F64) => SparseTensor::zeros(rows, cols),
            MatrixClass::Logical => SparseTensor::zeros_logical(rows, cols),
            MatrixClass::Floating(_) => {
                unreachable!("diagonal construction normalizes integer inputs to double")
            }
        });
    }
    if bin.cols != offsets.len() {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "spdiags: Bin must have one column per diagonal, got {} columns for {} diagonals",
                bin.cols,
                offsets.len()
            ),
        ));
    }

    let mut entries = BTreeMap::new();
    for (diag_index, &offset) in offsets.iter().enumerate() {
        let source_col = diag_index;
        for t in 0..diag_len(rows, cols, offset) {
            let Some((row, col)) = diag_coord(offset, t) else {
                continue;
            };
            let source_row = diag_bout_row(offset, t);
            if source_row >= bin.rows {
                continue;
            }
            let value = bin.get(source_row, source_col);
            if is_stored_value(value) {
                add_entry(&mut entries, col, row, value);
            }
        }
    }
    sparse_from_entries_with_class(rows, cols, entries, bin.class, "spdiags")
}

fn replace_sparse_diagonals(
    bin: &Value,
    offsets: &Value,
    target: &Value,
) -> BuiltinResult<SparseTensor> {
    let offsets = parse_diag_offsets(offsets, "spdiags")?;
    let target_sparse = sparse_from_value(target.clone())?;
    let target_values = target_sparse.materialize_f64();
    let selected: BTreeSet<isize> = offsets.iter().copied().collect();
    let mut entries = BTreeMap::new();
    for col in 0..target_sparse.cols {
        for idx in target_sparse.col_ptrs[col]..target_sparse.col_ptrs[col + 1] {
            let row = target_sparse.row_indices[idx];
            let offset = col as isize - row as isize;
            if !selected.contains(&offset) {
                entries.insert((col, row), target_values[idx]);
            }
        }
    }
    let replacement =
        construct_sparse_diagonals(bin, &offsets, target_sparse.rows, target_sparse.cols)?;
    let replacement_values = replacement.materialize_f64();
    for col in 0..replacement.cols {
        for idx in replacement.col_ptrs[col]..replacement.col_ptrs[col + 1] {
            let row = replacement.row_indices[idx];
            let value = replacement_values[idx];
            if is_stored_value(value) {
                entries.insert((col, row), value);
            }
        }
    }
    let class = if target_sparse.is_logical() {
        MatrixClass::Logical
    } else if target_sparse.numeric_dtype() == Some(NumericDType::F32) {
        MatrixClass::Floating(NumericDType::F32)
    } else {
        MatrixClass::Floating(NumericDType::F64)
    };
    sparse_from_entries_with_class(
        target_sparse.rows,
        target_sparse.cols,
        entries,
        class,
        "spdiags",
    )
}

fn diag_len(rows: usize, cols: usize, offset: isize) -> usize {
    if offset >= 0 {
        let col_start = offset as usize;
        if col_start >= cols {
            return 0;
        }
        rows.min(cols - col_start)
    } else {
        let row_start = offset.unsigned_abs();
        if row_start >= rows {
            return 0;
        }
        (rows - row_start).min(cols)
    }
}

fn diag_coord(offset: isize, t: usize) -> Option<(usize, usize)> {
    if offset >= 0 {
        Some((t, offset as usize + t))
    } else {
        Some((offset.unsigned_abs() + t, t))
    }
}

fn diag_bout_row(offset: isize, t: usize) -> usize {
    if offset >= 0 {
        offset as usize + t
    } else {
        t
    }
}

fn sparse_from_entries(
    rows: usize,
    cols: usize,
    entries: BTreeMap<(usize, usize), f64>,
    label: &str,
) -> BuiltinResult<SparseTensor> {
    sparse_from_entries_with_dtype(rows, cols, entries, NumericDType::F64, label)
}

fn sparse_from_entries_with_dtype(
    rows: usize,
    cols: usize,
    entries: BTreeMap<(usize, usize), f64>,
    dtype: NumericDType,
    label: &str,
) -> BuiltinResult<SparseTensor> {
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..cols {
        for (&(entry_col, row), &value) in entries.range((col, 0)..=(col, usize::MAX)) {
            if entry_col != col {
                break;
            }
            if row < rows && is_stored_value(value) {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    let sparse = match dtype {
        NumericDType::F32 => SparseTensor::new_f32(
            rows,
            cols,
            col_ptrs,
            row_indices,
            values.into_iter().map(|value| value as f32).collect(),
        ),
        NumericDType::F64 => SparseTensor::new(rows, cols, col_ptrs, row_indices, values),
        _ => unreachable!("sparse entry reconstruction requires a floating dtype"),
    };
    sparse.map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
}

fn sparse_from_entries_with_class(
    rows: usize,
    cols: usize,
    entries: BTreeMap<(usize, usize), f64>,
    class: MatrixClass,
    label: &str,
) -> BuiltinResult<SparseTensor> {
    if let MatrixClass::Floating(dtype) = class {
        return sparse_from_entries_with_dtype(rows, cols, entries, dtype, label);
    }

    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for col in 0..cols {
        for (&(entry_col, row), &value) in entries.range((col, 0)..=(col, usize::MAX)) {
            if entry_col != col {
                break;
            }
            if row < rows && value != 0.0 {
                row_indices.push(row);
            }
        }
        col_ptrs.push(row_indices.len());
    }
    SparseTensor::new_logical(rows, cols, col_ptrs, row_indices)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
}

fn scalar_f64(value: &Value, name: &str, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("{label}: {name} must be a scalar, got {other:?}"),
        )),
    }
}

fn sparse_from_value(value: Value) -> BuiltinResult<SparseTensor> {
    match value {
        Value::SparseTensor(sparse) => Ok(sparse),
        Value::Tensor(tensor) => {
            if tensor.shape.len() != 2 {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!(
                        "sparse: input must be a 2-D matrix, got {}-D tensor",
                        tensor.shape.len()
                    ),
                ));
            }
            sparse_from_dense_tensor(&tensor)
        }
        Value::LogicalArray(logical) => {
            if logical.shape.len() != 2 {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!(
                        "sparse: input must be a 2-D matrix, got {}-D logical array",
                        logical.shape.len()
                    ),
                ));
            }
            sparse_from_logical_array(&logical)
        }
        Value::Num(n) => sparse_from_dense_tensor(
            &Tensor::new(vec![n], vec![1, 1])
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))?,
        ),
        Value::Int(i) => sparse_from_integer_scalar(i),
        Value::Bool(b) => {
            if b {
                SparseTensor::new_logical(1, 1, vec![0, 1], vec![0])
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
            } else {
                Ok(SparseTensor::zeros_logical(1, 1))
            }
        }
        other => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("sparse: unsupported conversion input {other:?}"),
        )),
    }
}

fn sparse_from_dense_tensor(tensor: &Tensor) -> BuiltinResult<SparseTensor> {
    if let Some(storage) = tensor.integer_storage() {
        return sparse_from_integer_dense_tensor(tensor.rows(), tensor.cols(), storage);
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    if let Some(dense_values) = tensor.as_f32_slice() {
        let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        col_ptrs.push(0);
        for col in 0..cols {
            for row in 0..rows {
                let value = dense_values[row + col * rows];
                if value != 0.0 {
                    row_indices.push(row);
                    values.push(value);
                }
            }
            col_ptrs.push(values.len());
        }
        return SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, values)
            .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")));
    }
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    let dense_values = tensor_utils::tensor_values_f64_cow(tensor);
    col_ptrs.push(0);
    for col in 0..cols {
        for row in 0..rows {
            let value = dense_values[row + col * rows];
            if is_stored_value(value) {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
}

fn sparse_from_integer_scalar(value: runmat_builtins::IntValue) -> BuiltinResult<SparseTensor> {
    let storage = match value {
        runmat_builtins::IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        runmat_builtins::IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        runmat_builtins::IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        runmat_builtins::IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        runmat_builtins::IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        runmat_builtins::IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        runmat_builtins::IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        runmat_builtins::IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    };
    sparse_from_integer_dense_tensor(1, 1, &storage)
}

fn sparse_from_integer_dense_tensor(
    rows: usize,
    cols: usize,
    storage: &IntegerStorage,
) -> BuiltinResult<SparseTensor> {
    macro_rules! construct_integer_sparse {
        ($source:expr, $variant:ident) => {{
            let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
            let mut row_indices = Vec::new();
            let mut values = Vec::new();
            col_ptrs.push(0);
            for col in 0..cols {
                for row in 0..rows {
                    let value = $source[row + col * rows];
                    if value != 0 {
                        row_indices.push(row);
                        values.push(value);
                    }
                }
                col_ptrs.push(values.len());
            }
            SparseTensor::new_integer(
                rows,
                cols,
                col_ptrs,
                row_indices,
                IntegerStorage::$variant(values),
            )
        }};
    }

    let sparse = match storage {
        IntegerStorage::I8(values) => construct_integer_sparse!(values, I8),
        IntegerStorage::I16(values) => construct_integer_sparse!(values, I16),
        IntegerStorage::I32(values) => construct_integer_sparse!(values, I32),
        IntegerStorage::I64(values) => construct_integer_sparse!(values, I64),
        IntegerStorage::U8(values) => construct_integer_sparse!(values, U8),
        IntegerStorage::U16(values) => construct_integer_sparse!(values, U16),
        IntegerStorage::U32(values) => construct_integer_sparse!(values, U32),
        IntegerStorage::U64(values) => construct_integer_sparse!(values, U64),
    };
    sparse.map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
}

fn sparse_from_logical_array(logical: &LogicalArray) -> BuiltinResult<SparseTensor> {
    if logical.shape.len() != 2 {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "sparse: input must be a 2-D matrix, got {}-D logical array",
                logical.shape.len()
            ),
        ));
    }
    let shape = match logical.shape.as_slice() {
        [] => vec![1, 1],
        [n] => vec![1, *n],
        [rows, cols, ..] => vec![*rows, *cols],
    };
    let rows = shape[0];
    let cols = shape[1];
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for col in 0..cols {
        for row in 0..rows {
            if logical.data[row + col * rows] != 0 {
                row_indices.push(row);
            }
        }
        col_ptrs.push(row_indices.len());
    }
    SparseTensor::new_logical(rows, cols, col_ptrs, row_indices)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
}

fn sparse_from_triplet_form(args: Vec<Value>) -> BuiltinResult<SparseTensor> {
    validate_triplet_subscript_classes(&args[0], &args[1])?;
    let rows_vec = triplet_subscripts(&args[0], "row")?;
    let cols_vec = triplet_subscripts(&args[1], "column")?;
    let values = triplet_values(&args[2])?;

    let target_length = rows_vec.len().max(cols_vec.len()).max(values.len());

    let rows_vec = if rows_vec.len() == 1 && target_length > 1 {
        vec![rows_vec[0]; target_length]
    } else {
        rows_vec
    };
    let cols_vec = if cols_vec.len() == 1 && target_length > 1 {
        vec![cols_vec[0]; target_length]
    } else {
        cols_vec
    };
    let values = values.expand_to(target_length)?;

    if rows_vec.len() != cols_vec.len() || rows_vec.len() != values.len() {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sparse: i, j, and v must have the same number of elements",
        ));
    }

    let explicit_dims = args.len() >= 5;
    let mut rows = if explicit_dims {
        parse_size_arg(&args[3], "m")?
    } else {
        0
    };
    let mut cols = if explicit_dims {
        parse_size_arg(&args[4], "n")?
    } else {
        0
    };
    if args.len() == 6 {
        let _ = parse_size_arg(&args[5], "nzmax")?;
    }

    let mut coordinates = Vec::with_capacity(target_length);
    for (&row, &col) in rows_vec.iter().zip(cols_vec.iter()) {
        if explicit_dims {
            if row > rows || col > cols {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INDEX,
                    "sparse: subscript exceeds matrix dimensions",
                ));
            }
        } else {
            rows = rows.max(row);
            cols = cols.max(col);
        }
        coordinates.push((col - 1, row - 1));
    }

    match values {
        TripletValues::F64(values) => sparse_from_float_triplets(rows, cols, coordinates, values),
        TripletValues::F32(values) => sparse_from_f32_triplets(rows, cols, coordinates, values),
        TripletValues::Logical(values) => {
            sparse_from_logical_triplets(rows, cols, coordinates, values)
        }
        TripletValues::Integer(values) => {
            sparse_from_integer_triplets(rows, cols, coordinates, values)
        }
    }
}

fn validate_triplet_subscript_classes(rows: &Value, cols: &Value) -> BuiltinResult<()> {
    let row_class = integer_subscript_class(rows);
    let col_class = integer_subscript_class(cols);
    if row_class.is_none() && col_class.is_none() {
        return Ok(());
    }
    if row_class == col_class {
        return Ok(());
    }
    Err(sparse_error(
        &SPARSE_ERROR_INVALID_INPUT,
        match (row_class, col_class) {
            (Some(row), Some(col)) => format!(
                "sparse: integer row and column subscripts must have the same datatype, got {row} and {col}"
            ),
            _ => "sparse: when either row or column subscripts use an integer datatype, both must use the same integer datatype".to_string(),
        },
    ))
}

fn integer_subscript_class(value: &Value) -> Option<&'static str> {
    match value {
        Value::Int(value) => Some(value.class_name()),
        Value::Tensor(tensor) => tensor.integer_storage().map(IntegerStorage::class_name),
        Value::SparseTensor(sparse) => sparse.integer_storage().map(IntegerStorage::class_name),
        _ => None,
    }
}

enum TripletValues {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Logical(Vec<u8>),
    Integer(IntegerStorage),
}

impl TripletValues {
    fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Logical(values) => values.len(),
            Self::Integer(values) => values.len(),
        }
    }

    fn expand_to(self, len: usize) -> BuiltinResult<Self> {
        if self.len() != 1 || len <= 1 {
            return Ok(self);
        }
        match self {
            Self::F64(values) => Ok(Self::F64(vec![values[0]; len])),
            Self::F32(values) => Ok(Self::F32(vec![values[0]; len])),
            Self::Logical(values) => Ok(Self::Logical(vec![values[0]; len])),
            Self::Integer(storage) => {
                let value = storage.value_at(0).expect("single integer storage value");
                let values = vec![value; len];
                storage
                    .from_same_class_values(values)
                    .map(Self::Integer)
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
            }
        }
    }
}

fn triplet_values(value: &Value) -> BuiltinResult<TripletValues> {
    match value {
        Value::Bool(value) => Ok(TripletValues::Logical(vec![u8::from(*value)])),
        Value::LogicalArray(values) => Ok(TripletValues::Logical(values.data.clone())),
        Value::SparseTensor(sparse) if sparse.is_logical() => sparse
            .to_dense_logical()
            .map(|dense| TripletValues::Logical(dense.data))
            .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}"))),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => Ok(TripletValues::Integer(
            tensor
                .integer_storage()
                .expect("checked integer storage")
                .clone(),
        )),
        Value::Int(value) => {
            let storage = match value {
                runmat_builtins::IntValue::I8(value) => IntegerStorage::I8(vec![*value]),
                runmat_builtins::IntValue::I16(value) => IntegerStorage::I16(vec![*value]),
                runmat_builtins::IntValue::I32(value) => IntegerStorage::I32(vec![*value]),
                runmat_builtins::IntValue::I64(value) => IntegerStorage::I64(vec![*value]),
                runmat_builtins::IntValue::U8(value) => IntegerStorage::U8(vec![*value]),
                runmat_builtins::IntValue::U16(value) => IntegerStorage::U16(vec![*value]),
                runmat_builtins::IntValue::U32(value) => IntegerStorage::U32(vec![*value]),
                runmat_builtins::IntValue::U64(value) => IntegerStorage::U64(vec![*value]),
            };
            Ok(TripletValues::Integer(storage))
        }
        Value::SparseTensor(sparse) if sparse.integer_storage().is_some() => {
            dense_typed_triplet_sparse(sparse, "v")
                .and_then(|dense| triplet_values(&Value::Tensor(dense)))
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            Ok(TripletValues::F32(
                tensor
                    .as_f32_slice()
                    .expect("single tensor storage")
                    .to_vec(),
            ))
        }
        Value::SparseTensor(sparse) if sparse.numeric_dtype() == Some(NumericDType::F32) => {
            dense_typed_triplet_sparse(sparse, "v")
                .and_then(|dense| triplet_values(&Value::Tensor(dense)))
        }
        _ => numeric_triplet_array(value, "v").map(TripletValues::F64),
    }
}

fn sparse_from_logical_triplets(
    rows: usize,
    cols: usize,
    coordinates: Vec<(usize, usize)>,
    values: Vec<u8>,
) -> BuiltinResult<SparseTensor> {
    let mut entries = BTreeMap::new();
    for (coordinate, value) in coordinates.into_iter().zip(values) {
        if value != 0 {
            entries.insert(coordinate, ());
        }
    }

    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for col in 0..cols {
        for (&(entry_col, row), ()) in entries.range((col, 0)..=(col, usize::MAX)) {
            if entry_col != col {
                break;
            }
            if row >= rows {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INDEX,
                    "sparse: row index exceeds matrix dimensions",
                ));
            }
            row_indices.push(row);
        }
        col_ptrs.push(row_indices.len());
    }
    SparseTensor::new_logical(rows, cols, col_ptrs, row_indices)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
}

fn sparse_from_float_triplets(
    rows: usize,
    cols: usize,
    coordinates: Vec<(usize, usize)>,
    values: Vec<f64>,
) -> BuiltinResult<SparseTensor> {
    let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for (coordinate, value) in coordinates.into_iter().zip(values) {
        if is_stored_value(value) {
            let entry = entries.entry(coordinate).or_insert(0.0);
            *entry += value;
        }
    }

    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..cols {
        for (&(entry_col, row), value) in entries.range((col, 0)..=(col, usize::MAX)) {
            if entry_col != col {
                break;
            }
            if row >= rows {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INDEX,
                    "sparse: row index exceeds matrix dimensions",
                ));
            }
            if is_stored_value(*value) {
                row_indices.push(row);
                values.push(*value);
            }
        }
        col_ptrs.push(values.len());
    }

    SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
}

fn sparse_from_f32_triplets(
    rows: usize,
    cols: usize,
    coordinates: Vec<(usize, usize)>,
    values: Vec<f32>,
) -> BuiltinResult<SparseTensor> {
    let mut entries: BTreeMap<(usize, usize), f32> = BTreeMap::new();
    for (coordinate, value) in coordinates.into_iter().zip(values) {
        if value != 0.0 {
            *entries.entry(coordinate).or_insert(0.0) += value;
        }
    }

    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..cols {
        for (&(entry_col, row), value) in entries.range((col, 0)..=(col, usize::MAX)) {
            if entry_col != col {
                break;
            }
            if row >= rows {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INDEX,
                    "sparse: row index exceeds matrix dimensions",
                ));
            }
            if *value != 0.0 {
                row_indices.push(row);
                values.push(*value);
            }
        }
        col_ptrs.push(values.len());
    }

    SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, values)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
}

fn sparse_from_integer_triplets(
    rows: usize,
    cols: usize,
    coordinates: Vec<(usize, usize)>,
    storage: IntegerStorage,
) -> BuiltinResult<SparseTensor> {
    let mut entries: BTreeMap<(usize, usize), runmat_builtins::IntValue> = BTreeMap::new();
    for (index, coordinate) in coordinates.into_iter().enumerate() {
        let value = storage.value_at(index).ok_or_else(|| {
            sparse_error(&SPARSE_ERROR_INTERNAL, "sparse: integer value is missing")
        })?;
        if value.is_zero() {
            continue;
        }
        match entries.get_mut(&coordinate) {
            Some(existing) => {
                *existing = existing.saturating_add(&value).map_err(|err| {
                    sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}"))
                })?;
            }
            None => {
                entries.insert(coordinate, value);
            }
        }
    }

    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..cols {
        for (&(entry_col, row), value) in entries.range((col, 0)..=(col, usize::MAX)) {
            if entry_col != col {
                break;
            }
            if row >= rows {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INDEX,
                    "sparse: row index exceeds matrix dimensions",
                ));
            }
            if !value.is_zero() {
                row_indices.push(row);
                values.push(value.clone());
            }
        }
        col_ptrs.push(values.len());
    }
    let values = storage
        .from_same_class_values(values)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))?;
    SparseTensor::new_integer(rows, cols, col_ptrs, row_indices, values)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
}

fn numeric_triplet_array(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor_utils::tensor_values_f64(tensor)),
        Value::SparseTensor(sparse) => {
            let total_elements = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
                sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("sparse: {name} sparse dimensions overflow"),
                )
            })?;
            if total_elements > SPARSE_DENSE_INPUT_VECTOR_LIMIT {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!(
                        "sparse: cannot densify sparse {name} input {}x{} with {} stored entries ({} elements exceeds safe threshold)",
                        sparse.rows,
                        sparse.cols,
                        sparse.nnz(),
                        total_elements
                    ),
                ));
            }
            if sparse.is_logical() {
                sparse
                    .to_dense_logical()
                    .map(|logical| logical.data.into_iter().map(f64::from).collect())
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
            } else {
                sparse
                    .to_dense()
                    .map(tensor_utils::tensor_into_values_f64)
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
            }
        }
        Value::LogicalArray(logical) => Ok(logical
            .data
            .iter()
            .map(|&bit| if bit == 0 { 0.0 } else { 1.0 })
            .collect()),
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        Value::Bool(value) => Ok(vec![if *value { 1.0 } else { 0.0 }]),
        other => Err(numeric_vector_error(other, name)),
    }
}

fn triplet_subscripts(value: &Value, name: &str) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let storage = tensor.integer_storage().expect("checked integer storage");
            (0..storage.len())
                .map(|index| {
                    let integer = storage
                        .value_at(index)
                        .expect("integer storage length matches tensor data");
                    parse_integer_subscript(&integer, name)
                })
                .collect()
        }
        Value::Int(integer) => Ok(vec![parse_integer_subscript(integer, name)?]),
        Value::SparseTensor(sparse) if sparse.integer_storage().is_some() => {
            dense_typed_triplet_sparse(sparse, name)
                .and_then(|dense| triplet_subscripts(&Value::Tensor(dense), name))
        }
        _ => numeric_triplet_array(value, name).and_then(|values| {
            values
                .into_iter()
                .map(|value| parse_subscript(value, name))
                .collect()
        }),
    }
}

fn dense_typed_triplet_sparse(sparse: &SparseTensor, name: &str) -> BuiltinResult<Tensor> {
    let total_elements = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
        sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("sparse: {name} sparse dimensions overflow"),
        )
    })?;
    if total_elements > SPARSE_DENSE_INPUT_VECTOR_LIMIT {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "sparse: cannot densify sparse {name} input {}x{} with {} stored entries ({} elements exceeds safe threshold)",
                sparse.rows,
                sparse.cols,
                sparse.nnz(),
                total_elements
            ),
        ));
    }
    sparse
        .to_dense()
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
}

fn parse_integer_subscript(value: &runmat_builtins::IntValue, name: &str) -> BuiltinResult<usize> {
    match integer_to_usize(value) {
        Some(index) if index > 0 => Ok(index),
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INDEX,
            format!("sparse: {name} indices must be positive integers"),
        )),
    }
}

fn integer_to_usize(value: &runmat_builtins::IntValue) -> Option<usize> {
    match value {
        runmat_builtins::IntValue::I8(value) => usize::try_from(*value).ok(),
        runmat_builtins::IntValue::I16(value) => usize::try_from(*value).ok(),
        runmat_builtins::IntValue::I32(value) => usize::try_from(*value).ok(),
        runmat_builtins::IntValue::I64(value) => usize::try_from(*value).ok(),
        runmat_builtins::IntValue::U8(value) => Some(usize::from(*value)),
        runmat_builtins::IntValue::U16(value) => Some(usize::from(*value)),
        runmat_builtins::IntValue::U32(value) => usize::try_from(*value).ok(),
        runmat_builtins::IntValue::U64(value) => usize::try_from(*value).ok(),
    }
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) {
                return Err(numeric_vector_error(value, name));
            }
            Ok(tensor_utils::tensor_values_f64(tensor))
        }
        Value::SparseTensor(sparse) => {
            let shape = [sparse.rows, sparse.cols];
            if !is_vector_shape(&shape) {
                return Err(numeric_vector_error(value, name));
            }
            let total_elements = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
                sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("sparse: {name} sparse vector dimensions overflow"),
                )
            })?;
            if total_elements > SPARSE_DENSE_INPUT_VECTOR_LIMIT {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!(
                        "sparse: cannot densify sparse {name} vector {}x{} with {} stored entries ({} elements exceeds safe threshold)",
                        sparse.rows,
                        sparse.cols,
                        sparse.nnz(),
                        total_elements
                    ),
                ));
            }
            if sparse.is_logical() {
                sparse
                    .to_dense_logical()
                    .map(|logical| logical.data.into_iter().map(f64::from).collect())
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
            } else {
                sparse
                    .to_dense()
                    .map(tensor_utils::tensor_into_values_f64)
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
            }
        }
        Value::LogicalArray(logical) => {
            if !is_vector_shape(&logical.shape) {
                return Err(numeric_vector_error(value, name));
            }
            Ok(logical
                .data
                .iter()
                .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                .collect())
        }
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        other => Err(numeric_vector_error(other, name)),
    }
}

fn numeric_vector_for_label(value: &Value, name: &str, label: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) {
                return Err(numeric_vector_label_error(value, name, label));
            }
            Ok(tensor_utils::tensor_values_f64(tensor))
        }
        Value::SparseTensor(sparse) => {
            let shape = [sparse.rows, sparse.cols];
            if !is_vector_shape(&shape) {
                return Err(numeric_vector_label_error(value, name, label));
            }
            let total_elements = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
                sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("{label}: {name} sparse vector dimensions overflow"),
                )
            })?;
            if total_elements > SPARSE_DENSE_INPUT_VECTOR_LIMIT {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!(
                        "{label}: cannot densify sparse {name} vector {}x{} with {} stored entries ({} elements exceeds safe threshold)",
                        sparse.rows,
                        sparse.cols,
                        sparse.nnz(),
                        total_elements
                    ),
                ));
            }
            if sparse.is_logical() {
                sparse
                    .to_dense_logical()
                    .map(|logical| logical.data.into_iter().map(f64::from).collect())
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
            } else {
                sparse
                    .to_dense()
                    .map(tensor_utils::tensor_into_values_f64)
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
            }
        }
        Value::LogicalArray(logical) => {
            if !is_vector_shape(&logical.shape) {
                return Err(numeric_vector_label_error(value, name, label));
            }
            Ok(logical
                .data
                .iter()
                .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                .collect())
        }
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        other => Err(numeric_vector_label_error(other, name, label)),
    }
}

fn numeric_vector_error(value: &Value, name: &str) -> RuntimeError {
    sparse_error(
        &SPARSE_ERROR_INVALID_INPUT,
        format!("sparse: {name} must be a real numeric vector, got {value:?}"),
    )
}

fn numeric_vector_label_error(value: &Value, name: &str, label: &str) -> RuntimeError {
    sparse_error(
        &SPARSE_ERROR_INVALID_INPUT,
        format!("{label}: {name} must be a real numeric vector, got {value:?}"),
    )
}

fn is_vector_shape(shape: &[usize]) -> bool {
    if shape.len() > 2 {
        return false;
    }
    shape.iter().filter(|&&dim| dim != 1).count() <= 1
}

fn parse_size_arg(value: &Value, name: &str) -> BuiltinResult<usize> {
    if let Some(value) = scalar_integer_value(value) {
        return value.try_to_usize().ok_or_else(|| {
            sparse_error(
                &SPARSE_ERROR_INVALID_INPUT,
                format!("sparse: {name} exceeds the maximum supported size"),
            )
        });
    }
    let raw = match value {
        Value::Num(n) => *n,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => {
            return Err(sparse_error(
                &SPARSE_ERROR_INVALID_INPUT,
                format!("sparse: {name} must be a scalar size"),
            ))
        }
    };
    if !raw.is_finite() || raw < 0.0 || raw.fract() != 0.0 {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("sparse: {name} must be a nonnegative integer"),
        ));
    }
    if raw > max_usize_cast_value() {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("sparse: {name} exceeds the maximum supported size"),
        ));
    }
    Ok(raw as usize)
}

fn scalar_integer_value(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn tensor_integer_values(tensor: &Tensor) -> Option<Vec<IntValue>> {
    tensor.integer_storage().map(|storage| {
        (0..storage.len())
            .map(|index| {
                storage
                    .value_at(index)
                    .expect("integer tensor storage length matches tensor data")
            })
            .collect()
    })
}

fn parse_subscript(raw: f64, name: &str) -> BuiltinResult<usize> {
    if !raw.is_finite() || raw < 1.0 || raw.fract() != 0.0 {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INDEX,
            format!("sparse: {name} indices must be positive integers"),
        ));
    }
    if raw > max_usize_cast_value() {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("sparse: {name} index exceeds the maximum supported size"),
        ));
    }
    Ok(raw as usize)
}

fn max_usize_cast_value() -> f64 {
    if usize::BITS <= f64::MANTISSA_DIGITS {
        usize::MAX as f64
    } else {
        f64::from_bits((usize::MAX as f64).to_bits() - 1)
    }
}

fn is_stored_value(value: f64) -> bool {
    value.is_nan() || value != 0.0
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use nalgebra::DMatrix;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView, HostTensorView};
    use runmat_builtins::IntValue;

    fn sparse_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::sparse_builtin(args))
    }

    fn nonzeros_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::nonzeros_builtin(value))
    }

    fn spones_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::spones_builtin(value))
    }

    fn sprand_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::sprand_builtin(args))
    }

    fn spdiags_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::spdiags_builtin(args))
    }

    fn expect_sparse(value: Value) -> SparseTensor {
        match value {
            Value::SparseTensor(sparse) => sparse,
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    fn exact_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("integer tensor")
    }

    fn integer_storage_in_same_class(storage: &IntegerStorage, value: i64) -> IntegerStorage {
        match storage {
            IntegerStorage::I8(_) => IntegerStorage::I8(vec![value as i8]),
            IntegerStorage::I16(_) => IntegerStorage::I16(vec![value as i16]),
            IntegerStorage::I32(_) => IntegerStorage::I32(vec![value as i32]),
            IntegerStorage::I64(_) => IntegerStorage::I64(vec![value]),
            IntegerStorage::U8(_) => IntegerStorage::U8(vec![value as u8]),
            IntegerStorage::U16(_) => IntegerStorage::U16(vec![value as u16]),
            IntegerStorage::U32(_) => IntegerStorage::U32(vec![value as u32]),
            IntegerStorage::U64(_) => IntegerStorage::U64(vec![value as u64]),
        }
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn dense_matrix_for_svd(sparse: &SparseTensor) -> DMatrix<f64> {
        let dense = sparse.to_dense().expect("dense sparse test matrix");
        let values = tensor_utils::tensor_values_f64_cow(&dense);
        DMatrix::from_column_slice(sparse.rows, sparse.cols, &values)
    }

    #[test]
    fn sparse_dims_constructs_empty_matrix() {
        let sparse = expect_sparse(
            sparse_builtin(vec![
                Value::Int(IntValue::I32(3)),
                Value::Int(IntValue::I32(4)),
            ])
            .expect("sparse"),
        );
        assert_eq!(sparse.shape(), vec![3, 4]);
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.col_ptrs, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn sparse_size_args_accept_typed_integer_scalar_tensors_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let rows = exact_integer_tensor(IntegerStorage::U64(vec![large]), vec![1, 1]);
        let cols = exact_integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]);

        let sparse = expect_sparse(
            sparse_builtin(vec![Value::Tensor(rows), Value::Tensor(cols)]).expect("sparse"),
        );
        assert_eq!(sparse.shape(), vec![large as usize, 1]);
        assert_eq!(sparse.nnz(), 0);
    }

    #[test]
    fn sparse_triplet_structural_args_read_all_integer_storage_classes() {
        let classes = [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];

        for index in classes {
            let dimension = integer_storage_in_same_class(&index, 2);
            let count = integer_storage_in_same_class(&index, 0);
            let sparse = expect_sparse(
                sparse_builtin(vec![
                    Value::Tensor(exact_integer_tensor(index.clone(), vec![1, 1])),
                    Value::Tensor(exact_integer_tensor(index, vec![1, 1])),
                    Value::Num(0.0),
                    Value::Tensor(exact_integer_tensor(dimension.clone(), vec![1, 1])),
                    Value::Tensor(exact_integer_tensor(dimension, vec![1, 1])),
                    Value::Tensor(exact_integer_tensor(count, vec![1, 1])),
                ])
                .expect("sparse typed structural arguments"),
            );
            assert_eq!(sparse.shape(), vec![2, 2]);
            assert_eq!(sparse.nnz(), 0);
        }
    }

    #[test]
    fn sparse_triplet_integer_subscripts_do_not_round_through_double() {
        let large = 9_007_199_254_740_993_u64;
        let row =
            Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).expect("row index");
        let col = Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).expect("col index");
        let value = Tensor::new(vec![0.0], vec![1, 1]).expect("zero value");
        let rows = Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).expect("rows");
        let cols = Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).expect("cols");

        let sparse = expect_sparse(
            sparse_builtin(vec![
                Value::Tensor(row.clone()),
                Value::Tensor(col.clone()),
                Value::Tensor(value.clone()),
                Value::Tensor(rows),
                Value::Tensor(cols),
            ])
            .expect("sparse exact row"),
        );
        assert_eq!(sparse.shape(), vec![large as usize, 1]);
        assert_eq!(sparse.nnz(), 0);

        let smaller_rows =
            Tensor::new_integer(IntegerStorage::U64(vec![large - 1]), vec![1, 1]).expect("rows");
        let err = sparse_builtin(vec![
            Value::Tensor(row),
            Value::Tensor(col),
            Value::Tensor(value),
            Value::Tensor(smaller_rows),
            Value::Int(IntValue::U64(1)),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidIndex"));
        assert!(err
            .message()
            .contains("subscript exceeds matrix dimensions"));
    }

    #[test]
    fn speye_shape_parsing_accepts_typed_integer_tensors_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let shape =
            Tensor::new_integer(IntegerStorage::U64(vec![large, 1]), vec![1, 2]).expect("shape");
        let sparse =
            expect_sparse(super::speye_builtin(vec![Value::Tensor(shape)]).expect("speye"));
        assert_eq!(sparse.shape(), vec![large as usize, 1]);
        assert_eq!(sparse.nnz(), 1);
        assert_eq!(sparse.get(0, 0), Some(1.0));

        let negative =
            Tensor::new_integer(IntegerStorage::I64(vec![-3]), vec![1, 1]).expect("negative");
        let empty =
            expect_sparse(super::speye_builtin(vec![Value::Tensor(negative)]).expect("speye"));
        assert_eq!(empty.shape(), vec![0, 0]);
        assert_eq!(empty.nnz(), 0);
    }

    #[test]
    fn sparse_from_integer_tensor_preserves_every_integer_class_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cases = vec![
            IntegerStorage::I8(vec![0, i8::MIN, i8::MAX, 0]),
            IntegerStorage::I16(vec![0, i16::MIN, i16::MAX, 0]),
            IntegerStorage::I32(vec![0, i32::MIN, i32::MAX, 0]),
            IntegerStorage::I64(vec![0, i64::MIN, i64::MAX, 0]),
            IntegerStorage::U8(vec![0, 1, u8::MAX, 0]),
            IntegerStorage::U16(vec![0, 1, u16::MAX, 0]),
            IntegerStorage::U32(vec![0, 1, u32::MAX, 0]),
            IntegerStorage::U64(vec![0, 1, u64::MAX, 0]),
        ];

        for storage in cases {
            let expected_class = storage.class_name();
            let tensor = Tensor::new_integer(storage.clone(), vec![2, 2]).expect("integer tensor");
            let sparse = expect_sparse(
                sparse_builtin(vec![Value::Tensor(tensor)]).expect("sparse integer tensor"),
            );

            assert_eq!(sparse.class_name(), expected_class);
            assert_eq!(
                sparse.integer_storage().map(IntegerStorage::class_name),
                Some(expected_class)
            );
            assert_eq!(sparse.integer_at(1, 0), storage.value_at(1));
            assert_eq!(sparse.integer_at(0, 1), storage.value_at(2));
            assert_eq!(sparse.integer_at(0, 0), None);
            assert_eq!(sparse.integer_at(1, 1), None);
            assert_eq!(
                sparse
                    .to_dense()
                    .expect("dense integer sparse")
                    .integer_storage(),
                Some(&storage)
            );
        }
    }

    #[test]
    fn nonzeros_of_uint64_sparse_preserves_exact_values() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let source = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX, 7, 0]), vec![2, 2])
            .expect("uint64 tensor");
        let sparse = expect_sparse(
            sparse_builtin(vec![Value::Tensor(source)]).expect("sparse uint64 tensor"),
        );

        let values = expect_tensor(
            nonzeros_builtin(Value::SparseTensor(sparse)).expect("nonzeros uint64 sparse"),
        );
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 7]))
        );
    }

    #[test]
    fn nonzeros_of_dense_tensors_preserves_native_numeric_class() {
        let integer = Tensor::new_integer(
            IntegerStorage::U64(vec![0, (1_u64 << 53) + 1, u64::MAX]),
            vec![1, 3],
        )
        .expect("uint64 tensor");
        let integer =
            expect_tensor(nonzeros_builtin(Value::Tensor(integer)).expect("dense uint64 nonzeros"));
        assert_eq!(
            integer.integer_storage(),
            Some(&IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]))
        );

        let single =
            Tensor::from_f32(vec![0.0, -2.5, f32::NAN], vec![1, 3]).expect("single tensor");
        let single =
            expect_tensor(nonzeros_builtin(Value::Tensor(single)).expect("dense single nonzeros"));
        assert_eq!(single.numeric_dtype(), NumericDType::F32);
        assert_eq!(single.len(), 2);
        assert_eq!(
            single.numeric_value_at(0),
            Some(runmat_builtins::NumericScalar::F32(-2.5))
        );
        assert!(matches!(
            single.numeric_value_at(1),
            Some(runmat_builtins::NumericScalar::F32(value)) if value.is_nan()
        ));
    }

    #[test]
    fn nonzeros_preserves_every_integer_class_and_exact_scalar_values() {
        let cases = [
            IntegerStorage::I8(vec![0, i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![0, i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![0, i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![0, i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, 1, u8::MAX]),
            IntegerStorage::U16(vec![0, 1, u16::MAX]),
            IntegerStorage::U32(vec![0, 1, u32::MAX]),
            IntegerStorage::U64(vec![0, 1, u64::MAX]),
        ];
        for storage in cases {
            let expected = storage
                .from_same_class_values(
                    storage
                        .exact_values()
                        .into_iter()
                        .filter(|value| !value.is_zero())
                        .collect(),
                )
                .expect("same-class expected storage");
            let tensor = Tensor::new_integer(storage, vec![3, 1]).expect("integer tensor");
            let output =
                expect_tensor(nonzeros_builtin(Value::Tensor(tensor)).expect("integer nonzeros"));
            assert_eq!(output.shape, vec![2, 1]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }

        let scalar = expect_tensor(
            nonzeros_builtin(Value::Int(IntValue::U64(u64::MAX)))
                .expect("exact integer scalar nonzeros"),
        );
        assert_eq!(
            scalar.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX]))
        );
    }

    #[test]
    fn nonzeros_preserves_character_logical_and_complex_component_classes() {
        let characters = CharArray::new(vec!['a', '\0', 'z'], 1, 3).expect("character array");
        let output = nonzeros_builtin(Value::CharArray(characters)).expect("character nonzeros");
        assert!(
            matches!(output, Value::CharArray(array) if array.shape == vec![2, 1] && array.data == vec!['a', 'z'])
        );

        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).expect("logical array");
        let output = nonzeros_builtin(Value::LogicalArray(logical)).expect("logical nonzeros");
        assert!(
            matches!(output, Value::LogicalArray(array) if array.shape == vec![2, 1] && array.data == vec![1, 1])
        );

        let single = ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(0.0, 0.0), (1.5, 0.0), (0.0, -2.5)]),
            vec![3, 1],
        )
        .expect("single complex tensor");
        let output =
            nonzeros_builtin(Value::ComplexTensor(single)).expect("single complex nonzeros");
        assert!(
            matches!(output, Value::ComplexTensor(tensor) if tensor.complex_storage() == &ComplexStorage::F32(vec![(1.5, 0.0), (0.0, -2.5)]))
        );

        let integer = runmat_builtins::IntegerComplexStorage::new(
            IntegerStorage::U64(vec![0, u64::MAX, 0]),
            IntegerStorage::U64(vec![0, 0, u64::MAX]),
        )
        .expect("paired integer storage");
        let integer = ComplexTensor::new_integer(integer, vec![3, 1]).expect("complex integer");
        let output =
            nonzeros_builtin(Value::ComplexTensor(integer)).expect("integer complex nonzeros");
        assert!(
            matches!(output, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some_and(|storage| storage.real == IntegerStorage::U64(vec![u64::MAX, 0]) && storage.imag == IntegerStorage::U64(vec![0, u64::MAX])))
        );
    }

    #[test]
    fn sparse_triplets_sum_duplicates_and_drop_zeros() {
        let i = Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![4, 1]).unwrap();
        let j = Tensor::new(vec![1.0, 1.0, 1.0, 3.0], vec![4, 1]).unwrap();
        let v = Tensor::new(vec![2.0, 0.0, 5.0, 9.0], vec![4, 1]).unwrap();
        let sparse = expect_sparse(
            sparse_builtin(vec![
                Value::Tensor(i),
                Value::Tensor(j),
                Value::Tensor(v),
                Value::Num(3.0),
                Value::Num(4.0),
            ])
            .expect("sparse"),
        );
        assert_eq!(sparse.shape(), vec![3, 4]);
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get(0, 0), Some(7.0));
        assert_eq!(sparse.get(1, 2), Some(9.0));
    }

    #[test]
    fn sparse_triplet_integer_subscripts_require_one_shared_datatype() {
        let rows = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap();
        let cols = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap();
        let values = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let sparse = expect_sparse(
            sparse_builtin(vec![
                Value::Tensor(rows),
                Value::Tensor(cols),
                Value::Tensor(values.clone()),
            ])
            .expect("matching integer subscript classes"),
        );
        assert_eq!(sparse.shape(), vec![2, 2]);

        let rows = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap();
        let cols = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).unwrap();
        let err = sparse_builtin(vec![
            Value::Tensor(rows),
            Value::Tensor(cols),
            Value::Tensor(values.clone()),
        ])
        .expect_err("mixed integer subscript classes reject");
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(err.message().contains("uint16 and int16"));

        let rows = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap();
        let cols = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = sparse_builtin(vec![
            Value::Tensor(rows),
            Value::Tensor(cols),
            Value::Tensor(values),
        ])
        .expect_err("integer and floating subscript classes reject");
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(err
            .message()
            .contains("both must use the same integer datatype"));
    }

    #[test]
    fn registered_sparse_integer_results_require_runmat_mode() {
        let integer = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("integer tensor");
        let triplet_rows =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("row index");
        let triplet_cols =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("column index");
        let triplet_values =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("values");
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = crate::dispatcher::call_builtin("sparse", &[Value::Tensor(integer.clone())])
                .expect_err("MATLAB mode must reject sparse integer output");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:SparseIntegerExtension")
            );
            let err = crate::dispatcher::call_builtin(
                "sparse",
                &[
                    Value::Tensor(triplet_rows.clone()),
                    Value::Tensor(triplet_cols.clone()),
                    Value::Tensor(triplet_values.clone()),
                ],
            )
            .expect_err("MATLAB mode must reject sparse integer triplet values");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:SparseIntegerExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let result =
                crate::dispatcher::call_builtin("sparse", &[Value::Tensor(integer)]).unwrap();
            let sparse = expect_sparse(result);
            assert_eq!(
                sparse.integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX]))
            );
            let result = crate::dispatcher::call_builtin(
                "sparse",
                &[
                    Value::Tensor(triplet_rows),
                    Value::Tensor(triplet_cols),
                    Value::Tensor(triplet_values),
                ],
            )
            .expect("RunMat mode accepts sparse integer triplet values");
            let sparse = expect_sparse(result);
            assert_eq!(
                sparse.integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX]))
            );
        }
    }

    #[test]
    fn registered_cast_and_arithmetic_cannot_expose_sparse_integer_in_matlab_mode() {
        let floating =
            SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![7.0]).expect("floating sparse");
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err =
                crate::dispatcher::call_builtin("uint16", &[Value::SparseTensor(floating.clone())])
                    .expect_err("MATLAB mode must reject sparse integer cast output");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:SparseIntegerExtension")
            );
        }

        let integer =
            SparseTensor::new_integer(1, 1, vec![0, 1], vec![0], IntegerStorage::U16(vec![7]))
                .expect("integer sparse");
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = crate::dispatcher::call_builtin(
                "plus",
                &[
                    Value::SparseTensor(integer.clone()),
                    Value::Int(IntValue::U16(0)),
                ],
            )
            .expect_err("MATLAB mode must reject sparse integer arithmetic output");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:SparseIntegerExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let cast = crate::dispatcher::call_builtin("uint16", &[Value::SparseTensor(floating)])
                .unwrap();
            assert!(matches!(
                cast,
                Value::SparseTensor(ref sparse)
                    if sparse.integer_storage() == Some(&IntegerStorage::U16(vec![7]))
            ));
            let sum = crate::dispatcher::call_builtin(
                "plus",
                &[Value::SparseTensor(integer), Value::Int(IntValue::U16(0))],
            )
            .unwrap();
            assert!(matches!(
                sum,
                Value::SparseTensor(ref sparse)
                    if sparse.integer_storage() == Some(&IntegerStorage::U16(vec![7]))
            ));
        }
    }

    #[test]
    fn sparse_numeric_helpers_read_typed_integer_storage_before_float_boundary() {
        let vector = Value::Tensor(exact_integer_tensor(
            IntegerStorage::I16(vec![1, -2, 3]),
            vec![3, 1],
        ));
        assert_eq!(
            numeric_vector(&vector, "i").expect("numeric vector"),
            vec![1.0, -2.0, 3.0]
        );
        assert_eq!(
            numeric_triplet_array(&vector, "v").expect("numeric triplet array"),
            vec![1.0, -2.0, 3.0]
        );

        let matrix = Value::Tensor(exact_integer_tensor(
            IntegerStorage::I16(vec![10, 20, 0, 1, 2, 0]),
            vec![3, 2],
        ));
        let dense = dense_matrix_from_value(&matrix, "spdiags").expect("dense matrix");
        assert_eq!(dense.data, vec![10.0, 20.0, 0.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn sparse_triplets_accept_dense_matrices_but_reject_sparse_and_logical_matrices() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let vector = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let sparse = expect_sparse(
            sparse_builtin(vec![
                Value::Tensor(matrix),
                Value::Tensor(vector.clone()),
                Value::Tensor(vector.clone()),
            ])
            .expect("dense matrix triplets"),
        );
        assert_eq!(sparse.shape(), vec![4, 4]);
        assert_eq!(sparse.nnz(), 4);

        let sparse_matrix =
            SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let err = sparse_builtin(vec![
            Value::SparseTensor(sparse_matrix),
            Value::Tensor(vector.clone()),
            Value::Tensor(vector.clone()),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidIndex"));
        assert!(err.message().contains("positive integers"));

        let logical_matrix = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let err = sparse_builtin(vec![
            Value::LogicalArray(logical_matrix),
            Value::Tensor(vector.clone()),
            Value::Tensor(vector),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidIndex"));
        assert!(err.message().contains("positive integers"));
    }

    #[test]
    fn sparse_from_dense_preserves_column_major_entries() {
        let dense = Tensor::new(vec![0.0, 4.0, 5.0, 0.0], vec![2, 2]).unwrap();
        let sparse = expect_sparse(sparse_builtin(vec![Value::Tensor(dense)]).expect("sparse"));
        assert_eq!(sparse.col_ptrs, vec![0, 1, 2]);
        assert_eq!(sparse.row_indices, vec![1, 0]);
        assert_eq!(sparse.materialize_f64(), vec![4.0, 5.0]);
    }

    #[test]
    fn sparse_from_logical_uses_pattern_as_authoritative_storage() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
        let sparse =
            expect_sparse(sparse_builtin(vec![Value::LogicalArray(logical)]).expect("sparse"));
        assert!(sparse.is_logical());
        assert_eq!(sparse.numeric_dtype(), None);
        assert_eq!(sparse.col_ptrs, vec![0, 1, 2]);
        assert_eq!(sparse.row_indices, vec![1, 0]);
        assert_eq!(
            sparse.to_dense_logical().expect("dense logical").data,
            vec![0, 1, 1, 0]
        );
    }

    #[test]
    fn sparse_logical_triplets_merge_with_or_and_discard_false_values() {
        let i = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).unwrap();
        let j = Tensor::new(vec![1.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let values = LogicalArray::new(vec![0, 1, 1, 0], vec![4, 1]).unwrap();
        let sparse = expect_sparse(
            sparse_builtin(vec![
                Value::Tensor(i),
                Value::Tensor(j),
                Value::LogicalArray(values),
                Value::Num(2.0),
                Value::Num(3.0),
            ])
            .expect("logical triplet sparse"),
        );
        assert!(sparse.is_logical());
        assert_eq!(sparse.col_ptrs, vec![0, 1, 2, 2]);
        assert_eq!(sparse.row_indices, vec![0, 1]);
    }

    #[test]
    fn native_single_sparse_constructors_and_nonzeros_preserve_class() {
        let dense = Tensor::from_f32(vec![0.0, 4.25, 5.5, 0.0], vec![2, 2]).unwrap();
        let from_dense =
            expect_sparse(sparse_builtin(vec![Value::Tensor(dense)]).expect("single sparse"));
        assert_eq!(from_dense.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(from_dense.as_f32_slice(), Some(&[4.25, 5.5][..]));

        let rows = Tensor::new(vec![1.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let cols = Tensor::new(vec![1.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let values = Tensor::from_f32(vec![0.25, 0.5, 3.0], vec![3, 1]).unwrap();
        let triplets = expect_sparse(
            sparse_builtin(vec![
                Value::Tensor(rows),
                Value::Tensor(cols),
                Value::Tensor(values),
            ])
            .expect("single triplets"),
        );
        assert_eq!(triplets.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(triplets.as_f32_slice(), Some(&[0.75, 3.0][..]));

        let empty = expect_sparse(
            sparse_builtin(vec![
                Value::Num(3.0),
                Value::Num(4.0),
                Value::String("single".to_string()),
            ])
            .expect("typed empty sparse"),
        );
        assert_eq!(empty.shape(), vec![3, 4]);
        assert_eq!(empty.numeric_dtype(), Some(NumericDType::F32));

        let eye = expect_sparse(
            super::speye_builtin(vec![Value::Num(3.0), Value::String("single".to_string())])
                .expect("single speye"),
        );
        assert_eq!(eye.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(eye.as_f32_slice(), Some(&[1.0, 1.0, 1.0][..]));

        let random_empty = expect_sparse(
            sprand_builtin(vec![
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(0.0),
                Value::String("single".to_string()),
            ])
            .expect("single sprand"),
        );
        assert_eq!(random_empty.shape(), vec![2, 3]);
        assert_eq!(random_empty.numeric_dtype(), Some(NumericDType::F32));

        let nonzeros =
            expect_tensor(nonzeros_builtin(Value::SparseTensor(triplets)).expect("nonzeros"));
        assert_eq!(nonzeros.numeric_dtype(), NumericDType::F32);
        assert_eq!(nonzeros.as_f32_slice(), Some(&[0.75, 3.0][..]));
    }

    #[test]
    fn logical_typename_constructs_empty_sparse_and_identity_without_value_payload() {
        let empty = expect_sparse(
            sparse_builtin(vec![
                Value::Num(3.0),
                Value::Num(4.0),
                Value::String("logical".to_string()),
            ])
            .expect("logical empty sparse"),
        );
        assert!(empty.is_logical());
        assert_eq!(empty.shape(), vec![3, 4]);
        assert_eq!(empty.nnz(), 0);

        let eye = expect_sparse(
            super::speye_builtin(vec![Value::Num(3.0), Value::String("logical".to_string())])
                .expect("logical speye"),
        );
        assert!(eye.is_logical());
        assert_eq!(eye.col_ptrs, vec![0, 1, 2, 3]);
        assert_eq!(eye.row_indices, vec![0, 1, 2]);

        let err = sprand_builtin(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(0.5),
            Value::String("logical".to_string()),
        ])
        .expect_err("sprand logical typename is not documented");
        assert!(err.message().contains("double"));
        assert!(err.message().contains("single"));
    }

    #[test]
    fn sparse_gathers_gpu_input() {
        test_support::with_test_provider(|provider| {
            let dense = Tensor::new(vec![0.0, 8.0, 0.0, 3.0], vec![2, 2]).unwrap();
            let dense_values = dense.as_f64_slice().expect("double test tensor");
            let handle = provider
                .upload(&HostTensorView {
                    data: dense_values,
                    shape: &dense.shape,
                })
                .expect("upload");
            let sparse = expect_sparse(sparse_builtin(vec![Value::GpuTensor(handle)]).unwrap());
            assert_eq!(sparse.nnz(), 2);
            assert_eq!(sparse.get(1, 0), Some(8.0));
            assert_eq!(sparse.get(1, 1), Some(3.0));
        });
    }

    #[test]
    fn sparse_provider_contract_is_explicitly_host_resident() {
        assert!(GPU_SPEC.provider_hooks.is_empty());
        assert!(matches!(
            GPU_SPEC.residency,
            ResidencyPolicy::GatherImmediately
        ));
        assert_eq!(
            GPU_SPEC.supported_precisions,
            &[ScalarType::F32, ScalarType::F64]
        );
        assert!(GPU_SPEC.notes.contains("host-resident CSC"));
        assert!(GPU_SPEC.notes.contains("dense tensor handles only"));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn sparse_wgpu_input_gathers_into_host_csc_storage() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::AccelProvider;

        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let values = [0.0, 8.0, 0.0, 3.0];
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &[2, 2],
            })
            .expect("upload WGPU input");
        let sparse = expect_sparse(
            crate::dispatcher::call_builtin("sparse", &[Value::GpuTensor(handle)])
                .expect("gather WGPU sparse input"),
        );
        assert_eq!(sparse.shape(), vec![2, 2]);
        assert_eq!(sparse.col_ptrs, vec![0, 1, 2]);
        assert_eq!(sparse.row_indices, vec![1, 1]);
        assert_eq!(sparse.materialize_f64(), vec![8.0, 3.0]);
    }

    #[test]
    fn sparse_integer_gpu_input_gathers_to_host_and_remains_runmat_mode_only() {
        test_support::with_test_provider(|provider| {
            let values = [0_u64, u64::MAX];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &[1, 2],
                })
                .expect("upload integer input");

            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error =
                    crate::dispatcher::call_builtin("sparse", &[Value::GpuTensor(handle.clone())])
                        .expect_err("MATLAB mode must reject gathered sparse integer output");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:SparseIntegerExtension")
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                let sparse = expect_sparse(
                    crate::dispatcher::call_builtin("sparse", &[Value::GpuTensor(handle)])
                        .expect("RunMat mode sparse integer gather"),
                );
                assert_eq!(
                    sparse.integer_storage(),
                    Some(&IntegerStorage::U64(vec![u64::MAX]))
                );
                assert_eq!(sparse.row_indices, vec![0]);
                assert_eq!(sparse.col_ptrs, vec![0, 0, 1]);
            }
        });
    }

    #[test]
    fn sparse_rejects_size_and_subscript_values_too_large_for_usize() {
        let too_large = max_usize_cast_value() * 2.0;

        let size_err = parse_size_arg(&Value::Num(too_large), "m").unwrap_err();
        assert_eq!(size_err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(size_err.message().contains("maximum supported size"));

        let index_err = parse_subscript(too_large, "row").unwrap_err();
        assert_eq!(index_err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(index_err.message().contains("maximum supported size"));
    }

    #[test]
    fn sparse_triplets_reject_oversized_sparse_vector_before_densifying() {
        let i = SparseTensor::zeros(SPARSE_DENSE_INPUT_VECTOR_LIMIT + 1, 1);
        let j = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let v = Tensor::new(vec![1.0], vec![1, 1]).unwrap();

        let err = sparse_builtin(vec![
            Value::SparseTensor(i),
            Value::Tensor(j),
            Value::Tensor(v),
        ])
        .unwrap_err();

        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(err.message().contains("exceeds safe threshold"));
    }

    #[test]
    fn speye_supports_square_rectangular_vector_and_negative_sizes() {
        let square = expect_sparse(super::speye_builtin(vec![Value::Num(3.0)]).expect("speye"));
        assert_eq!(square.shape(), vec![3, 3]);
        assert_eq!(square.col_ptrs, vec![0, 1, 2, 3]);
        assert_eq!(square.row_indices, vec![0, 1, 2]);
        assert_eq!(square.materialize_f64(), vec![1.0, 1.0, 1.0]);

        let rect = expect_sparse(
            super::speye_builtin(vec![Value::Num(2.0), Value::Num(4.0)]).expect("speye"),
        );
        assert_eq!(rect.shape(), vec![2, 4]);
        assert_eq!(rect.get(0, 0), Some(1.0));
        assert_eq!(rect.get(1, 1), Some(1.0));
        assert_eq!(rect.nnz(), 2);

        let shape = Tensor::new(vec![4.0, 2.0], vec![1, 2]).unwrap();
        let from_shape =
            expect_sparse(super::speye_builtin(vec![Value::Tensor(shape)]).expect("speye"));
        assert_eq!(from_shape.shape(), vec![4, 2]);
        assert_eq!(from_shape.nnz(), 2);

        let empty =
            expect_sparse(super::speye_builtin(vec![Value::Num(-5.0)]).expect("speye negative"));
        assert_eq!(empty.shape(), vec![0, 0]);
        assert_eq!(empty.nnz(), 0);
    }

    #[test]
    fn nonzeros_returns_column_order_for_dense_sparse_logical_and_complex() {
        let dense = Tensor::new(vec![0.0, 4.0, 5.0, 0.0], vec![2, 2]).unwrap();
        let out = expect_tensor(nonzeros_builtin(Value::Tensor(dense)).expect("nonzeros"));
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.as_f64_slice().expect("double nonzeros"), &[4.0, 5.0]);

        let sparse =
            SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![10.0, 30.0, 20.0]).unwrap();
        let out = expect_tensor(nonzeros_builtin(Value::SparseTensor(sparse)).expect("nonzeros"));
        assert_eq!(out.shape, vec![3, 1]);
        assert_eq!(
            out.as_f64_slice().expect("double nonzeros"),
            &[10.0, 30.0, 20.0]
        );

        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let out = nonzeros_builtin(Value::LogicalArray(logical)).expect("nonzeros");
        assert!(
            matches!(out, Value::LogicalArray(array) if array.shape == vec![2, 1] && array.data == vec![1, 1])
        );

        let complex =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 2.0), (0.0, 3.0)], vec![3, 1]).unwrap();
        let out = nonzeros_builtin(Value::ComplexTensor(complex)).expect("nonzeros");
        match out {
            Value::ComplexTensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.materialize_f64(), vec![(1.0, 2.0), (0.0, 3.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn nonzeros_of_logical_sparse_preserves_logical_class() {
        let sparse =
            SparseTensor::new_logical(3, 2, vec![0, 2, 3], vec![0, 2, 1]).expect("logical sparse");
        let out = nonzeros_builtin(Value::SparseTensor(sparse)).expect("nonzeros");
        match out {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![3, 1]);
                assert_eq!(logical.data, vec![1, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn nonzeros_wgpu_integer_fallback_preserves_exact_explicit_residency_and_source() {
        let _guard = test_support::accel_test_lock();
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("register WGPU provider");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
            vec![3, 1],
        )
        .expect("integer source");
        let handle = gpu_helpers::upload_tensor(provider, &source).expect("upload integer source");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let output =
            nonzeros_builtin(Value::GpuTensor(handle.clone())).expect("resident integer nonzeros");
        let Value::GpuTensor(output) = output else {
            panic!("explicit gpuArray result must remain resident");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        let output = block_on(gpu_helpers::download_value_preserving_residency_async(
            provider, &output,
        ))
        .expect("download output");
        assert!(
            matches!(output, Value::Tensor(tensor) if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX])))
        );
        let source = block_on(gpu_helpers::download_value_preserving_residency_async(
            provider, &handle,
        ))
        .expect("source remains live");
        assert!(
            matches!(source, Value::Tensor(tensor) if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX])))
        );
    }

    #[test]
    fn spones_preserves_sparse_pattern_and_accepts_dense_gpu_input() {
        let sparse = SparseTensor::new(
            3,
            3,
            vec![0, 1, 1, 3],
            vec![1, 0, 2],
            vec![4.0, 5.0, f64::NAN],
        )
        .unwrap();
        let ones = expect_sparse(spones_builtin(Value::SparseTensor(sparse)).expect("spones"));
        assert_eq!(ones.col_ptrs, vec![0, 1, 1, 3]);
        assert_eq!(ones.row_indices, vec![1, 0, 2]);
        assert_eq!(ones.materialize_f64(), vec![1.0, 1.0, 1.0]);

        test_support::with_test_provider(|provider| {
            let dense = Tensor::new(vec![0.0, 2.0, 3.0, 0.0], vec![2, 2]).unwrap();
            let dense_values = dense.as_f64_slice().expect("double test tensor");
            let handle = provider
                .upload(&HostTensorView {
                    data: dense_values,
                    shape: &dense.shape,
                })
                .expect("upload");
            let sparse = expect_sparse(spones_builtin(Value::GpuTensor(handle)).expect("spones"));
            assert_eq!(sparse.shape(), vec![2, 2]);
            assert_eq!(sparse.materialize_f64(), vec![1.0, 1.0]);
            assert_eq!(sparse.get(1, 0), Some(1.0));
            assert_eq!(sparse.get(0, 1), Some(1.0));
        });
    }

    #[test]
    fn sprand_pattern_form_preserves_shape_and_pattern() {
        let pattern =
            SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![10.0, 30.0, 20.0]).unwrap();
        let random = expect_sparse(sprand_builtin(vec![Value::SparseTensor(pattern)]).unwrap());
        assert_eq!(random.shape(), vec![3, 2]);
        assert_eq!(random.col_ptrs, vec![0, 2, 3]);
        assert_eq!(random.row_indices, vec![0, 2, 1]);
        assert_eq!(random.materialize_f64().len(), 3);
        assert!(random
            .materialize_f64()
            .iter()
            .all(|value| *value >= 0.0 && *value < 1.0));
    }

    #[test]
    fn sprand_density_form_constructs_requested_sparse_shape() {
        let random = expect_sparse(
            sprand_builtin(vec![Value::Num(4.0), Value::Num(5.0), Value::Num(0.25)]).unwrap(),
        );
        assert_eq!(random.shape(), vec![4, 5]);
        assert_eq!(random.nnz(), 5);
        assert!(random
            .materialize_f64()
            .iter()
            .all(|value| *value >= 0.0 && *value < 1.0));
    }

    #[test]
    fn sprand_density_sampling_is_bounded_near_full_density() {
        let random = expect_sparse(
            sprand_builtin(vec![Value::Num(5.0), Value::Num(5.0), Value::Num(0.92)]).unwrap(),
        );
        assert_eq!(random.shape(), vec![5, 5]);
        assert_eq!(random.nnz(), 23);
    }

    #[test]
    fn sprand_scalar_rc_form_matches_requested_condition_at_full_density() {
        random::set_seed(11).expect("seed");
        let random = expect_sparse(
            sprand_builtin(vec![
                Value::Num(4.0),
                Value::Num(4.0),
                Value::Num(1.0),
                Value::Num(0.25),
            ])
            .expect("sprand rc"),
        );
        assert_eq!(random.shape(), vec![4, 4]);
        assert_eq!(random.nnz(), 16);

        let matrix = dense_matrix_for_svd(&random);
        let singular = matrix.svd(false, false).singular_values;
        let reciprocal_condition = singular[singular.len() - 1] / singular[0];
        assert!((reciprocal_condition - 0.25).abs() < 1.0e-10);
    }

    #[test]
    fn sprand_vector_rc_form_uses_requested_singular_values() {
        random::set_seed(13).expect("seed");
        let profile = Tensor::new(vec![1.0, 0.5], vec![2, 1]).unwrap();
        let random = expect_sparse(
            sprand_builtin(vec![
                Value::Num(4.0),
                Value::Num(3.0),
                Value::Num(1.0),
                Value::Tensor(profile),
            ])
            .expect("sprand rc vector"),
        );
        assert_eq!(random.shape(), vec![4, 3]);
        assert_eq!(random.nnz(), 12);

        let matrix = dense_matrix_for_svd(&random);
        let singular = matrix.svd(false, false).singular_values;
        assert!((singular[0] - 1.0).abs() < 1.0e-10);
        assert!((singular[1] - 0.5).abs() < 1.0e-10);
        assert!(singular[2].abs() < 1.0e-10);
    }

    #[test]
    fn sprand_rc_form_respects_density_and_typename_validation() {
        random::set_seed(17).expect("seed");
        let random = expect_sparse(
            sprand_builtin(vec![
                Value::Num(4.0),
                Value::Num(4.0),
                Value::Num(0.5),
                Value::Num(0.25),
                Value::String("double".into()),
            ])
            .expect("sprand rc density"),
        );
        assert_eq!(random.shape(), vec![4, 4]);
        assert!((8..=10).contains(&random.nnz()));
        let matrix = dense_matrix_for_svd(&random);
        let singular = matrix.svd(false, false).singular_values;
        let reciprocal_condition = singular[singular.len() - 1] / singular[0];
        assert!((reciprocal_condition - 0.25).abs() < 1.0e-10);

        let err = sprand_builtin(vec![
            Value::Num(4.0),
            Value::Num(4.0),
            Value::Num(0.25),
            Value::Num(1.2),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(err.message().contains("between 0 and 1"));

        let single = expect_sparse(
            sprand_builtin(vec![
                Value::Num(4.0),
                Value::Num(4.0),
                Value::Num(0.25),
                Value::String("single".into()),
            ])
            .expect("single sprand"),
        );
        assert_eq!(single.numeric_dtype(), Some(NumericDType::F32));
    }

    #[test]
    fn sprand_rc_vector_shape_preserves_singular_value_without_rotation_cap() {
        random::set_seed(19).expect("seed");
        let random = expect_sparse(
            sprand_builtin(vec![
                Value::Num(1.0),
                Value::Num(20.0),
                Value::Num(0.5),
                Value::Num(0.75),
            ])
            .expect("sprand row vector rc"),
        );
        assert_eq!(random.shape(), vec![1, 20]);
        assert_eq!(random.nnz(), 10);

        let matrix = dense_matrix_for_svd(&random);
        let singular = matrix.svd(false, false).singular_values;
        assert!((singular[0] - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn spdiags_extracts_diagonals_and_ids_with_padding() {
        let matrix = Tensor::new(
            vec![1.0, 2.0, 0.0, 4.0, 0.0, 6.0, 7.0, 0.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let offsets = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let bout = expect_tensor(
            spdiags_builtin(vec![Value::Tensor(matrix.clone()), Value::Tensor(offsets)])
                .expect("spdiags"),
        );
        assert_eq!(bout.shape, vec![3, 3]);
        assert_eq!(
            bout.as_f64_slice().expect("double spdiags output"),
            &[
                2.0, 6.0, 0.0, // -1 diagonal
                1.0, 0.0, 9.0, // 0 diagonal
                0.0, 4.0, 0.0, // +1 diagonal, padded at top
            ]
        );

        let _guard = crate::output_count::push_output_count(Some(2));
        let outputs = spdiags_builtin(vec![Value::Tensor(matrix)]).expect("spdiags");
        match outputs {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                let ids = expect_tensor(values[1].clone());
                assert_eq!(
                    ids.as_f64_slice().expect("double diagonal ids"),
                    &[-1.0, 0.0, 1.0, 2.0]
                );
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn spdiags_extracts_from_large_sparse_without_densifying() {
        let sparse = SparseTensor::new(
            SPARSE_HELPER_DENSE_INPUT_LIMIT + 1,
            1,
            vec![0, 1],
            vec![0],
            vec![42.0],
        )
        .unwrap();
        let offsets = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let bout = expect_tensor(
            spdiags_builtin(vec![Value::SparseTensor(sparse), Value::Tensor(offsets)])
                .expect("spdiags"),
        );
        assert_eq!(bout.shape, vec![1, 1]);
        assert_eq!(bout.as_f64_slice().expect("double spdiags output"), &[42.0]);
    }

    #[test]
    fn spdiags_constructs_and_replaces_sparse_diagonals() {
        let bin = Tensor::new(
            vec![
                10.0, 20.0, 0.0, // -1 source column
                0.0, 1.0, 2.0, // +1 source column
            ],
            vec![3, 2],
        )
        .unwrap();
        let offsets = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let sparse = expect_sparse(
            spdiags_builtin(vec![
                Value::Tensor(bin.clone()),
                Value::Tensor(offsets.clone()),
                Value::Num(3.0),
                Value::Num(3.0),
            ])
            .expect("spdiags"),
        );
        assert_eq!(sparse.get(1, 0), Some(10.0));
        assert_eq!(sparse.get(2, 1), Some(20.0));
        assert_eq!(sparse.get(0, 1), Some(1.0));
        assert_eq!(sparse.get(1, 2), Some(2.0));

        let target = SparseTensor::new(
            3,
            3,
            vec![0, 2, 3, 4],
            vec![0, 1, 1, 2],
            vec![9.0, 8.0, 7.0, 6.0],
        )
        .unwrap();
        let replaced = expect_sparse(
            spdiags_builtin(vec![
                Value::Tensor(bin),
                Value::Tensor(offsets),
                Value::SparseTensor(target),
            ])
            .expect("spdiags"),
        );
        assert_eq!(replaced.get(0, 0), Some(9.0));
        assert_eq!(replaced.get(1, 0), Some(10.0));
        assert_eq!(replaced.get(1, 1), Some(7.0));
        assert_eq!(replaced.get(2, 2), Some(6.0));
        assert_eq!(replaced.get(1, 2), Some(2.0));
    }

    #[test]
    fn spdiags_preserves_native_single_extraction_construction_and_replacement() {
        let bin = Tensor::from_f32(
            vec![
                10.0, 20.0, 0.0, // -1 source column
                0.0, 1.0, 2.0, // +1 source column
            ],
            vec![3, 2],
        )
        .unwrap();
        let offsets = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let constructed = expect_sparse(
            spdiags_builtin(vec![
                Value::Tensor(bin.clone()),
                Value::Tensor(offsets.clone()),
                Value::Num(3.0),
                Value::Num(3.0),
            ])
            .expect("construct single spdiags"),
        );
        assert_eq!(constructed.numeric_dtype(), Some(NumericDType::F32));

        let extracted = expect_tensor(
            spdiags_builtin(vec![
                Value::SparseTensor(constructed.clone()),
                Value::Tensor(offsets.clone()),
            ])
            .expect("extract single spdiags"),
        );
        assert_eq!(extracted.numeric_dtype(), NumericDType::F32);

        let target =
            SparseTensor::new_f32(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![9.0, 7.0, 6.0])
                .unwrap();
        let replaced = expect_sparse(
            spdiags_builtin(vec![
                Value::Tensor(bin),
                Value::Tensor(offsets),
                Value::SparseTensor(target),
            ])
            .expect("replace single spdiags"),
        );
        assert_eq!(replaced.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(replaced.get(0, 0), Some(9.0));
        assert_eq!(replaced.get(1, 0), Some(10.0));
        assert_eq!(replaced.get(1, 1), Some(7.0));
        assert_eq!(replaced.get(2, 2), Some(6.0));
    }

    #[test]
    fn spdiags_preserves_logical_extraction_construction_and_replacement() {
        let bin = LogicalArray::new(
            vec![
                1, 0, 0, // -1 source column
                0, 1, 1, // +1 source column
            ],
            vec![3, 2],
        )
        .unwrap();
        let offsets = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let constructed = expect_sparse(
            spdiags_builtin(vec![
                Value::LogicalArray(bin.clone()),
                Value::Tensor(offsets.clone()),
                Value::Num(3.0),
                Value::Num(3.0),
            ])
            .expect("construct logical spdiags"),
        );
        assert!(constructed.is_logical());
        assert_eq!(constructed.col_ptrs, vec![0, 1, 2, 3]);
        assert_eq!(constructed.row_indices, vec![1, 0, 1]);

        let extracted = spdiags_builtin(vec![
            Value::SparseTensor(constructed.clone()),
            Value::Tensor(offsets.clone()),
        ])
        .expect("extract logical spdiags");
        match extracted {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![3, 2]);
                assert_eq!(logical.data, bin.data);
            }
            other => panic!("expected logical extraction, got {other:?}"),
        }

        let target = SparseTensor::new_logical(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2]).unwrap();
        let replaced = expect_sparse(
            spdiags_builtin(vec![
                Value::LogicalArray(bin),
                Value::Tensor(offsets),
                Value::SparseTensor(target),
            ])
            .expect("replace logical spdiags"),
        );
        assert!(replaced.is_logical());
        assert_eq!(
            replaced.to_dense_logical().expect("dense logical").data,
            vec![
                1, 1, 0, // column 1
                1, 1, 0, // column 2
                0, 1, 1, // column 3
            ]
        );
    }

    #[test]
    fn spdiags_preserves_integer_construction_and_exact_offsets() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let bin = exact_integer_tensor(
            IntegerStorage::I16(vec![
                10, 20, 0, // -1 source column
                0, 1, 2, // +1 source column
            ]),
            vec![3, 2],
        );
        let offsets = exact_integer_tensor(IntegerStorage::I16(vec![-1, 1]), vec![1, 2]);
        let sparse = expect_sparse(
            spdiags_builtin(vec![
                Value::Tensor(bin),
                Value::Tensor(offsets),
                Value::Num(3.0),
                Value::Num(3.0),
            ])
            .expect("spdiags"),
        );
        assert_eq!(
            sparse.integer_storage(),
            Some(&IntegerStorage::I16(vec![10, 1, 20, 2]))
        );
        assert_eq!(sparse.integer_at(1, 0), Some(IntValue::I16(10)));
        assert_eq!(sparse.integer_at(2, 1), Some(IntValue::I16(20)));
        assert_eq!(sparse.integer_at(0, 1), Some(IntValue::I16(1)));
        assert_eq!(sparse.integer_at(1, 2), Some(IntValue::I16(2)));
    }

    #[test]
    fn spdiags_offset_parser_decodes_all_integer_classes_exactly() {
        let cases = [
            (IntegerStorage::I8(vec![-1, 1]), vec![-1, 1]),
            (IntegerStorage::I16(vec![-2, 2]), vec![-2, 2]),
            (IntegerStorage::I32(vec![-3, 3]), vec![-3, 3]),
            (IntegerStorage::I64(vec![-4, 4]), vec![-4, 4]),
            (IntegerStorage::U8(vec![0, 1]), vec![0, 1]),
            (IntegerStorage::U16(vec![0, 2]), vec![0, 2]),
            (IntegerStorage::U32(vec![0, 3]), vec![0, 3]),
            (IntegerStorage::U64(vec![0, 4]), vec![0, 4]),
        ];

        for (storage, expected) in cases {
            let tensor = exact_integer_tensor(storage, vec![1, 2]);
            assert_eq!(
                parse_diag_offsets(&Value::Tensor(tensor), "spdiags").expect("typed offsets"),
                expected
            );
        }

        let wide = 9_007_199_254_740_993_u64;
        let wide_offsets = parse_diag_offsets(&Value::Int(IntValue::U64(wide)), "spdiags");
        if let Ok(wide) = isize::try_from(wide) {
            assert_eq!(wide_offsets.expect("wide host offset"), vec![wide]);
        } else {
            assert!(
                wide_offsets.is_err(),
                "unrepresentable wide offset must fail"
            );
        }

        let err = parse_diag_offsets(&Value::Int(IntValue::U64(u64::MAX)), "spdiags")
            .expect_err("wide uint64 offset must be rejected instead of rounded");
        assert!(err.to_string().contains("exceeds supported range"));
    }

    #[test]
    fn spdiags_preserves_wide_integer_extraction_and_sparse_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = 9_007_199_254_740_993_u64;
        let dense = exact_integer_tensor(
            IntegerStorage::U64(vec![wide, 0, 0, 0, wide + 2, 0, 0, 0, u64::MAX]),
            vec![3, 3],
        );
        let offsets = exact_integer_tensor(IntegerStorage::I8(vec![0]), vec![1, 1]);
        let extracted = expect_tensor(
            spdiags_builtin(vec![Value::Tensor(dense), Value::Tensor(offsets.clone())])
                .expect("extract dense integer diagonal"),
        );
        assert_eq!(
            extracted.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide, wide + 2, u64::MAX]))
        );

        let sparse = SparseTensor::new_integer(
            SPARSE_HELPER_DENSE_INPUT_LIMIT + 1,
            1,
            vec![0, 1],
            vec![0],
            IntegerStorage::U64(vec![wide]),
        )
        .unwrap();
        let extracted = expect_tensor(
            spdiags_builtin(vec![Value::SparseTensor(sparse), Value::Tensor(offsets)])
                .expect("extract sparse integer diagonal"),
        );
        assert_eq!(
            extracted.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide]))
        );
    }

    #[test]
    fn spdiags_integer_replacement_preserves_target_class_and_saturates_duplicates() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let target = SparseTensor::new_integer(
            3,
            3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            IntegerStorage::U64(vec![u64::MAX, 7, 6]),
        )
        .unwrap();
        let bin = exact_integer_tensor(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]);
        let offsets = exact_integer_tensor(IntegerStorage::I8(vec![1]), vec![1, 1]);
        let replaced = expect_sparse(
            spdiags_builtin(vec![
                Value::Tensor(bin),
                Value::Tensor(offsets),
                Value::SparseTensor(target),
            ])
            .expect("replace integer diagonal"),
        );
        assert_eq!(replaced.integer_at(0, 0), Some(IntValue::U64(u64::MAX)));
        assert_eq!(replaced.integer_at(1, 1), Some(IntValue::U64(7)));
        assert_eq!(replaced.integer_at(2, 2), Some(IntValue::U64(6)));
        assert_eq!(replaced.integer_at(0, 1), Some(IntValue::U64(2)));
        assert_eq!(replaced.integer_at(1, 2), Some(IntValue::U64(3)));

        let bin = exact_integer_tensor(IntegerStorage::U8(vec![250, 10]), vec![1, 2]);
        let offsets = exact_integer_tensor(IntegerStorage::I8(vec![0, 0]), vec![1, 2]);
        let saturated = expect_sparse(
            spdiags_builtin(vec![
                Value::Tensor(bin),
                Value::Tensor(offsets),
                Value::Num(1.0),
                Value::Num(1.0),
            ])
            .expect("construct duplicate integer diagonals"),
        );
        assert_eq!(saturated.integer_at(0, 0), Some(IntValue::U8(u8::MAX)));
    }

    #[test]
    fn sparse_random_and_diagonal_integer_extensions_gate_independently() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_data = exact_integer_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]);
        let err = spdiags_builtin(vec![Value::Tensor(integer_data)])
            .expect_err("typed spdiags data must be gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:SpdiagsIntegerDataExtension")
        );

        let floating = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let offsets = exact_integer_tensor(IntegerStorage::I8(vec![0]), vec![1, 1]);
        let err = spdiags_builtin(vec![Value::Tensor(floating), Value::Tensor(offsets)])
            .expect_err("typed spdiags offsets must be gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:SpdiagsIntegerControlExtension")
        );

        let err = sprand_builtin(vec![
            Value::Int(IntValue::U8(2)),
            Value::Num(2.0),
            Value::Num(0.5),
        ])
        .expect_err("typed sprand dimensions must be gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:SprandIntegerNumericControlExtension")
        );
    }

    #[test]
    fn spdiags_duplicate_offsets_sum_and_bin_columns_must_match_offsets() {
        let duplicate_bin = Tensor::new(
            vec![
                1.0, 2.0, 3.0, // first main diagonal contribution
                10.0, 20.0, 30.0, // second main diagonal contribution
            ],
            vec![3, 2],
        )
        .unwrap();
        let duplicate_offsets = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let sparse = expect_sparse(
            spdiags_builtin(vec![
                Value::Tensor(duplicate_bin),
                Value::Tensor(duplicate_offsets),
                Value::Num(3.0),
                Value::Num(3.0),
            ])
            .expect("spdiags"),
        );
        assert_eq!(sparse.get(0, 0), Some(11.0));
        assert_eq!(sparse.get(1, 1), Some(22.0));
        assert_eq!(sparse.get(2, 2), Some(33.0));

        let one_column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let two_offsets = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let err = spdiags_builtin(vec![
            Value::Tensor(one_column),
            Value::Tensor(two_offsets),
            Value::Num(3.0),
            Value::Num(3.0),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(err.message().contains("one column per diagonal"));
    }
}
