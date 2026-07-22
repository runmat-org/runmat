//! MATLAB-compatible `sparse` construction for real double matrices.

use std::collections::{BTreeMap, BTreeSet};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntegerStorage, LogicalArray, ResolveContext, SparseTensor, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
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

const SPARSE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
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
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn sparse_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::array::creation::sparse"
)]
fn speye_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (rows, cols) = parse_speye_shape(&args)?;
    Ok(Value::SparseTensor(sparse_identity(rows, cols)?))
}

#[runtime_builtin(
    name = "nonzeros",
    category = "array/creation",
    summary = "Return nonzero matrix elements in column order.",
    keywords = "nonzeros,sparse,nonzero,column vector",
    accel = "custom",
    descriptor(crate::builtins::array::creation::sparse::NONZEROS_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn nonzeros_builtin(value: Value) -> BuiltinResult<Value> {
    let value = gpu_helpers::gather_value_async(&value).await?;
    nonzeros_value(&value)
}

#[runtime_builtin(
    name = "spones",
    category = "array/creation",
    summary = "Replace nonzero sparse matrix elements with ones.",
    keywords = "spones,sparse,pattern,nonzero",
    accel = "custom",
    descriptor(crate::builtins::array::creation::sparse::SPONES_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn spones_builtin(value: Value) -> BuiltinResult<Value> {
    let sparse = sparse_pattern_from_value(gpu_helpers::gather_value_async(&value).await?)?;
    Ok(Value::SparseTensor(
        SparseTensor::new(
            sparse.rows,
            sparse.cols,
            sparse.col_ptrs,
            sparse.row_indices,
            vec![1.0; sparse.values.len()],
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
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn sprand_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::array::creation::sparse"
)]
async fn spdiags_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gpu_helpers::gather_value_async(&arg).await?);
    }
    spdiags_value(gathered)
}

fn construct_sparse(args: Vec<Value>) -> BuiltinResult<SparseTensor> {
    match args.len() {
        1 => sparse_from_value(args.into_iter().next().expect("one argument")),
        2 => {
            let mut iter = args.into_iter();
            let rows = parse_size_arg(iter.next().as_ref().expect("rows"), "m")?;
            let cols = parse_size_arg(iter.next().as_ref().expect("cols"), "n")?;
            Ok(SparseTensor::zeros(rows, cols))
        }
        3 | 5 | 6 => sparse_from_triplet_form(args),
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sparse: expected sparse(A), sparse(m,n), or sparse(i,j,v[,m,n[,nzmax]])",
        )),
    }
}

fn parse_speye_shape(args: &[Value]) -> BuiltinResult<(usize, usize)> {
    let mut args = args;
    if let Some((last, rest)) = args.split_last() {
        if let Some(type_name) = optional_sparse_double_typename(last) {
            if !type_name.eq_ignore_ascii_case("double") {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!(
                        "speye: only double sparse storage is currently supported, got {type_name}"
                    ),
                ));
            }
            args = rest;
        }
    }

    match args {
        [] => Ok((1, 1)),
        [n] => match n {
            Value::Tensor(tensor) if is_vector_shape(&tensor.shape) && tensor.data.len() == 2 => {
                let rows = parse_speye_size_raw(tensor.data[0], "m")?;
                let cols = parse_speye_size_raw(tensor.data[1], "n")?;
                Ok((rows, cols))
            }
            Value::SparseTensor(sparse) if is_vector_shape(&[sparse.rows, sparse.cols]) => {
                let dense = sparse
                    .to_dense()
                    .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("speye: {err}")))?;
                if dense.data.len() != 2 {
                    return Err(sparse_error(
                        &SPARSE_ERROR_INVALID_INPUT,
                        "speye: size vector must have two elements",
                    ));
                }
                let rows = parse_speye_size_raw(dense.data[0], "m")?;
                let cols = parse_speye_size_raw(dense.data[1], "n")?;
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
    }
}

fn optional_sparse_double_typename(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn parse_speye_size_value(value: &Value, name: &str) -> BuiltinResult<usize> {
    let raw = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
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

fn sparse_identity(rows: usize, cols: usize) -> BuiltinResult<SparseTensor> {
    let diagonal = rows.min(cols);
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::with_capacity(diagonal);
    let mut values = Vec::with_capacity(diagonal);
    col_ptrs.push(0);
    for col in 0..cols {
        if col < diagonal {
            row_indices.push(col);
            values.push(1.0);
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("speye: {err}")))
}

fn nonzeros_value(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::SparseTensor(sparse) => match sparse.integer_storage() {
            Some(storage) => integer_tensor_column(storage.clone(), "nonzeros"),
            None => tensor_column(sparse.values.clone(), "nonzeros"),
        },
        Value::Tensor(tensor) => {
            let data = tensor
                .data
                .iter()
                .copied()
                .filter(|value| is_stored_value(*value))
                .collect();
            tensor_column(data, "nonzeros")
        }
        Value::ComplexTensor(tensor) => {
            let data: Vec<(f64, f64)> = tensor
                .data
                .iter()
                .copied()
                .filter(|(re, im)| is_stored_value(*re) || is_stored_value(*im))
                .collect();
            ComplexTensor::new(data.clone(), vec![data.len(), 1])
                .map(Value::ComplexTensor)
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("nonzeros: {err}")))
        }
        Value::LogicalArray(logical) => {
            let data = logical
                .data
                .iter()
                .filter_map(|&bit| if bit != 0 { Some(1.0) } else { None })
                .collect();
            tensor_column(data, "nonzeros")
        }
        Value::Num(n) => tensor_column(
            if is_stored_value(*n) {
                vec![*n]
            } else {
                Vec::new()
            },
            "nonzeros",
        ),
        Value::Int(i) => tensor_column(
            if !i.is_zero() {
                vec![i.to_f64()]
            } else {
                Vec::new()
            },
            "nonzeros",
        ),
        Value::Bool(b) => tensor_column(if *b { vec![1.0] } else { Vec::new() }, "nonzeros"),
        other => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("nonzeros: unsupported input {other:?}"),
        )),
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

fn sparse_pattern_from_value(value: Value) -> BuiltinResult<SparseTensor> {
    sparse_from_value(value)
}

fn sprand_sparse(mut args: Vec<Value>) -> BuiltinResult<SparseTensor> {
    take_optional_sparse_double_typename(&mut args, "sprand")?;
    match args.len() {
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
            .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sprand: {err}")))
        }
        3 | 4 => {
            let rows = parse_size_arg(&args[0], "m")?;
            let cols = parse_size_arg(&args[1], "n")?;
            let density = parse_density_arg(&args[2])?;
            if args.len() == 4 {
                let profile = parse_sprand_condition_profile(&args[3], rows, cols)?;
                return sprand_from_condition_profile(rows, cols, density, &profile);
            }
            sprand_from_density(rows, cols, density)
        }
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sprand: expected sprand(S), sprand(m,n,density), or sprand(m,n,density,rc)",
        )),
    }
}

fn take_optional_sparse_double_typename(args: &mut Vec<Value>, label: &str) -> BuiltinResult<()> {
    let Some(type_name) = args.last().and_then(keyword_of) else {
        return Ok(());
    };
    match type_name.as_str() {
        "double" => {
            args.pop();
            Ok(())
        }
        "single" => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("{label}: single sparse storage is not supported yet"),
        )),
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!("{label}: typename must be \"double\" or \"single\", got {type_name}"),
        )),
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

fn spdiags_value(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.len() {
        1 => {
            let matrix = extraction_matrix_from_value(&args[0], "spdiags")?;
            let offsets = nonzero_diagonal_offsets(&matrix);
            let bout = extract_diagonals(&matrix, &offsets)?;
            let id = tensor_column(offsets.iter().map(|&d| d as f64).collect(), "spdiags")?;
            match crate::output_count::current_output_count() {
                Some(0) => Ok(Value::OutputList(Vec::new())),
                Some(1) => Ok(Value::OutputList(vec![Value::Tensor(bout)])),
                Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                    out_count,
                    vec![Value::Tensor(bout), id],
                )),
                None => Ok(Value::Tensor(bout)),
            }
        }
        2 => {
            let matrix = extraction_matrix_from_value(&args[0], "spdiags")?;
            let offsets = parse_diag_offsets(&args[1], "spdiags")?;
            extract_diagonals(&matrix, &offsets).map(Value::Tensor)
        }
        3 => replace_sparse_diagonals(&args[0], &args[1], &args[2]).map(Value::SparseTensor),
        4 => {
            let offsets = parse_diag_offsets(&args[1], "spdiags")?;
            let rows = parse_size_arg(&args[2], "m")?;
            let cols = parse_size_arg(&args[3], "n")?;
            construct_sparse_diagonals(&args[0], &offsets, rows, cols).map(Value::SparseTensor)
        }
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "spdiags: expected spdiags(A), spdiags(A,d), spdiags(B,d,m,n), or spdiags(B,d,A)",
        )),
    }
}

#[derive(Clone)]
struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
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
                data: tensor.data.clone(),
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
            let dense = sparse
                .to_dense()
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))?;
            Ok(DenseMatrix {
                rows: dense.rows(),
                cols: dense.cols(),
                data: dense.data,
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
            })
        }
        Value::Num(n) => Ok(DenseMatrix {
            rows: 1,
            cols: 1,
            data: vec![*n],
        }),
        Value::Int(i) => Ok(DenseMatrix {
            rows: 1,
            cols: 1,
            data: vec![i.to_f64()],
        }),
        Value::Bool(b) => Ok(DenseMatrix {
            rows: 1,
            cols: 1,
            data: vec![if *b { 1.0 } else { 0.0 }],
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
            for col in 0..sparse.cols {
                for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
                    if is_stored_value(sparse.values[idx]) {
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

fn extract_diagonals(matrix: &ExtractionMatrix, offsets: &[isize]) -> BuiltinResult<Tensor> {
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
    Tensor::new(data, vec![out_rows, offsets.len()])
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
        return Ok(SparseTensor::zeros(rows, cols));
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
    sparse_from_entries(rows, cols, entries, "spdiags")
}

fn replace_sparse_diagonals(
    bin: &Value,
    offsets: &Value,
    target: &Value,
) -> BuiltinResult<SparseTensor> {
    let offsets = parse_diag_offsets(offsets, "spdiags")?;
    let target_sparse = sparse_from_value(target.clone())?;
    let selected: BTreeSet<isize> = offsets.iter().copied().collect();
    let mut entries = BTreeMap::new();
    for col in 0..target_sparse.cols {
        for idx in target_sparse.col_ptrs[col]..target_sparse.col_ptrs[col + 1] {
            let row = target_sparse.row_indices[idx];
            let offset = col as isize - row as isize;
            if !selected.contains(&offset) {
                entries.insert((col, row), target_sparse.values[idx]);
            }
        }
    }
    let replacement =
        construct_sparse_diagonals(bin, &offsets, target_sparse.rows, target_sparse.cols)?;
    for col in 0..replacement.cols {
        for idx in replacement.col_ptrs[col]..replacement.col_ptrs[col + 1] {
            let row = replacement.row_indices[idx];
            let value = replacement.values[idx];
            if is_stored_value(value) {
                entries.insert((col, row), value);
            }
        }
    }
    sparse_from_entries(target_sparse.rows, target_sparse.cols, entries, "spdiags")
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
    SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
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
        Value::Bool(b) => sparse_from_dense_tensor(
            &Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))?,
        ),
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
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for col in 0..cols {
        for row in 0..rows {
            let value = tensor.data[row + col * rows];
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
    let data = logical
        .data
        .iter()
        .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
        .collect();
    let tensor = Tensor::new(data, shape)
        .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))?;
    sparse_from_dense_tensor(&tensor)
}

fn sparse_from_triplet_form(args: Vec<Value>) -> BuiltinResult<SparseTensor> {
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
        TripletValues::Float(values) => sparse_from_float_triplets(rows, cols, coordinates, values),
        TripletValues::Integer(values) => {
            sparse_from_integer_triplets(rows, cols, coordinates, values)
        }
    }
}

enum TripletValues {
    Float(Vec<f64>),
    Integer(IntegerStorage),
}

impl TripletValues {
    fn len(&self) -> usize {
        match self {
            Self::Float(values) => values.len(),
            Self::Integer(values) => values.len(),
        }
    }

    fn expand_to(self, len: usize) -> BuiltinResult<Self> {
        if self.len() != 1 || len <= 1 {
            return Ok(self);
        }
        match self {
            Self::Float(values) => Ok(Self::Float(vec![values[0]; len])),
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
        _ => numeric_triplet_array(value, "v").map(TripletValues::Float),
    }
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
        Value::Tensor(tensor) => Ok(tensor.data.clone()),
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
            sparse
                .to_dense()
                .map(|dense| dense.data)
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
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
            Ok(tensor.data.clone())
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
            sparse
                .to_dense()
                .map(|dense| dense.data)
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
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
            Ok(tensor.data.clone())
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
            sparse
                .to_dense()
                .map(|dense| dense.data)
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("{label}: {err}")))
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
    if let Value::Int(value) = value {
        return integer_to_usize(value).ok_or_else(|| {
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
    use runmat_accelerate_api::HostTensorView;
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

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn dense_matrix_for_svd(sparse: &SparseTensor) -> DMatrix<f64> {
        let dense = sparse.to_dense().expect("dense sparse test matrix");
        DMatrix::from_column_slice(sparse.rows, sparse.cols, &dense.data)
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
    fn sparse_from_integer_tensor_preserves_every_integer_class_exactly() {
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
    fn sparse_triplets_reject_matrix_inputs() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let vector = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let err = sparse_builtin(vec![
            Value::Tensor(matrix),
            Value::Tensor(vector.clone()),
            Value::Tensor(vector.clone()),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(err.message().contains("i must be a real numeric vector"));

        let sparse_matrix =
            SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let err = sparse_builtin(vec![
            Value::SparseTensor(sparse_matrix),
            Value::Tensor(vector.clone()),
            Value::Tensor(vector.clone()),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(err.message().contains("i must be a real numeric vector"));

        let logical_matrix = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let err = sparse_builtin(vec![
            Value::LogicalArray(logical_matrix),
            Value::Tensor(vector.clone()),
            Value::Tensor(vector),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(err.message().contains("i must be a real numeric vector"));
    }

    #[test]
    fn sparse_from_dense_preserves_column_major_entries() {
        let dense = Tensor::new(vec![0.0, 4.0, 5.0, 0.0], vec![2, 2]).unwrap();
        let sparse = expect_sparse(sparse_builtin(vec![Value::Tensor(dense)]).expect("sparse"));
        assert_eq!(sparse.col_ptrs, vec![0, 1, 2]);
        assert_eq!(sparse.row_indices, vec![1, 0]);
        assert_eq!(sparse.values, vec![4.0, 5.0]);
    }

    #[test]
    fn sparse_gathers_gpu_input() {
        test_support::with_test_provider(|provider| {
            let dense = Tensor::new(vec![0.0, 8.0, 0.0, 3.0], vec![2, 2]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &dense.data,
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
        assert_eq!(square.values, vec![1.0, 1.0, 1.0]);

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
        assert_eq!(out.data, vec![4.0, 5.0]);

        let sparse =
            SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![10.0, 30.0, 20.0]).unwrap();
        let out = expect_tensor(nonzeros_builtin(Value::SparseTensor(sparse)).expect("nonzeros"));
        assert_eq!(out.shape, vec![3, 1]);
        assert_eq!(out.data, vec![10.0, 30.0, 20.0]);

        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let out = expect_tensor(nonzeros_builtin(Value::LogicalArray(logical)).expect("nonzeros"));
        assert_eq!(out.data, vec![1.0, 1.0]);

        let complex =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 2.0), (0.0, 3.0)], vec![3, 1]).unwrap();
        let out = nonzeros_builtin(Value::ComplexTensor(complex)).expect("nonzeros");
        match out {
            Value::ComplexTensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![(1.0, 2.0), (0.0, 3.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
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
        assert_eq!(ones.values, vec![1.0, 1.0, 1.0]);

        test_support::with_test_provider(|provider| {
            let dense = Tensor::new(vec![0.0, 2.0, 3.0, 0.0], vec![2, 2]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &dense.data,
                    shape: &dense.shape,
                })
                .expect("upload");
            let sparse = expect_sparse(spones_builtin(Value::GpuTensor(handle)).expect("spones"));
            assert_eq!(sparse.shape(), vec![2, 2]);
            assert_eq!(sparse.values, vec![1.0, 1.0]);
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
        assert_eq!(random.values.len(), 3);
        assert!(random
            .values
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
            .values
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

        let err = sprand_builtin(vec![
            Value::Num(4.0),
            Value::Num(4.0),
            Value::Num(0.25),
            Value::String("single".into()),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:sparse:InvalidInput"));
        assert!(err.message().contains("single sparse storage"));
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
            bout.data,
            vec![
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
                assert_eq!(ids.data, vec![-1.0, 0.0, 1.0, 2.0]);
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
        assert_eq!(bout.data, vec![42.0]);
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
