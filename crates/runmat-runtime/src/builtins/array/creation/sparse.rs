//! MATLAB-compatible `sparse` construction for real double matrices.

use std::collections::BTreeMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LogicalArray, ResolveContext, SparseTensor, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "sparse";

const SPARSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse double matrix.",
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
    summary = "Create sparse double matrices from full arrays or row/column/value triplets.",
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

fn sparse_from_value(value: Value) -> BuiltinResult<SparseTensor> {
    match value {
        Value::SparseTensor(sparse) => Ok(sparse),
        Value::Tensor(tensor) => {
            if tensor.shape.len() != 2 {
                return Err(sparse_error(
                    &SPARSE_ERROR_INVALID_INPUT,
                    format!("sparse: input must be a 2-D matrix, got {}-D tensor", tensor.shape.len()),
                ));
            }
            sparse_from_dense_tensor(&tensor)
        }
        Value::LogicalArray(logical) => sparse_from_logical_array(&logical),
        Value::Num(n) => sparse_from_dense_tensor(
            &Tensor::new(vec![n], vec![1, 1])
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))?,
        ),
        Value::Int(i) => sparse_from_dense_tensor(
            &Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))?,
        ),
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

fn sparse_from_logical_array(logical: &LogicalArray) -> BuiltinResult<SparseTensor> {
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
    let rows_vec = numeric_vector(&args[0], "i")?;
    let cols_vec = numeric_vector(&args[1], "j")?;
    let values_vec = numeric_vector(&args[2], "v")?;

    let target_length = rows_vec.len().max(cols_vec.len()).max(values_vec.len());

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
    let values_vec = if values_vec.len() == 1 && target_length > 1 {
        vec![values_vec[0]; target_length]
    } else {
        values_vec
    };

    if rows_vec.len() != cols_vec.len() || rows_vec.len() != values_vec.len() {
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

    let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for ((&row_raw, &col_raw), &value) in
        rows_vec.iter().zip(cols_vec.iter()).zip(values_vec.iter())
    {
        let row = parse_subscript(row_raw, "row")?;
        let col = parse_subscript(col_raw, "column")?;
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
        if is_stored_value(value) {
            let key = (col - 1, row - 1);
            let entry = entries.entry(key).or_insert(0.0);
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

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.shape.len() > 2 {
                return Err(numeric_vector_error(value, name));
            }
            Ok(tensor.data.clone())
        }
        Value::SparseTensor(sparse) => {
            let shape = vec![sparse.rows, sparse.cols];
            if shape.len() > 2 {
                return Err(numeric_vector_error(value, name));
            }
            sparse
                .to_dense()
                .map(|dense| dense.data)
                .map_err(|err| sparse_error(&SPARSE_ERROR_INTERNAL, format!("sparse: {err}")))
        }
        Value::LogicalArray(logical) => {
            if logical.shape.len() > 2 {
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

fn numeric_vector_error(value: &Value, name: &str) -> RuntimeError {
    sparse_error(
        &SPARSE_ERROR_INVALID_INPUT,
        format!("sparse: {name} must be a real numeric vector, got {value:?}"),
    )
}

fn is_vector_shape(shape: &[usize]) -> bool {
    if shape.len() > 2 {
        return false;
    }
    shape.iter().filter(|&&dim| dim != 1).count() <= 1
}

fn parse_size_arg(value: &Value, name: &str) -> BuiltinResult<usize> {
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
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntValue;

    fn sparse_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::sparse_builtin(args))
    }

    fn expect_sparse(value: Value) -> SparseTensor {
        match value {
            Value::SparseTensor(sparse) => sparse,
            other => panic!("expected sparse tensor, got {other:?}"),
        }
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
}
