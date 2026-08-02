//! MATLAB-compatible `isdiag`, `istril`, and `istriu` matrix-structure predicates.

use log::debug;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderBandwidth};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntValue, LogicalArray, NumericStorage, SparseTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::math::linalg::type_resolvers::logical_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the input matrix satisfies the requested structure predicate.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input numeric, logical, complex, sparse, or gpuArray matrix.",
}];

const ISDIAG_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isdiag(A)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ISTRIL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = istril(A)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ISTRIU_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = istriu(A)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ISDIAG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISDIAG.INVALID_INPUT",
    identifier: Some("RunMat:isdiag:InvalidInput"),
    when: "Input cannot be interpreted as a supported numeric, logical, sparse, complex, or gpuArray matrix.",
    message: "isdiag: invalid input",
};

const ISDIAG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISDIAG.INTERNAL",
    identifier: Some("RunMat:isdiag:Internal"),
    when: "Runtime fails while gathering GPU input or constructing an internal matrix view.",
    message: "isdiag: internal runtime failure",
};

const ISTRIL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISTRIL.INVALID_INPUT",
    identifier: Some("RunMat:istril:InvalidInput"),
    when: "Input cannot be interpreted as a supported numeric, logical, sparse, complex, or gpuArray matrix.",
    message: "istril: invalid input",
};

const ISTRIL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISTRIL.INTERNAL",
    identifier: Some("RunMat:istril:Internal"),
    when: "Runtime fails while gathering GPU input or constructing an internal matrix view.",
    message: "istril: internal runtime failure",
};

const ISTRIU_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISTRIU.INVALID_INPUT",
    identifier: Some("RunMat:istriu:InvalidInput"),
    when: "Input cannot be interpreted as a supported numeric, logical, sparse, complex, or gpuArray matrix.",
    message: "istriu: invalid input",
};

const ISTRIU_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISTRIU.INTERNAL",
    identifier: Some("RunMat:istriu:Internal"),
    when: "Runtime fails while gathering GPU input or constructing an internal matrix view.",
    message: "istriu: internal runtime failure",
};

const ISDIAG_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ISDIAG_ERROR_INVALID_INPUT, ISDIAG_ERROR_INTERNAL];
const ISTRIL_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ISTRIL_ERROR_INVALID_INPUT, ISTRIL_ERROR_INTERNAL];
const ISTRIU_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ISTRIU_ERROR_INVALID_INPUT, ISTRIU_ERROR_INTERNAL];

pub const ISDIAG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISDIAG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISDIAG_ERRORS,
};

pub const ISTRIL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISTRIL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISTRIL_ERRORS,
};

pub const ISTRIU_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISTRIU_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISTRIU_ERRORS,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::structure::diagonal_triangular"
)]
pub const ISDIAG_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isdiag",
    op_kind: GpuOpKind::Custom("structure_analysis"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("bandwidth")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Real and logical gpuArray inputs use the provider bandwidth hook and read back only the small [lower, upper] result. Complex-interleaved inputs and providers without bandwidth support fall back to the host path.",
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::structure::diagonal_triangular"
)]
pub const ISTRIL_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "istril",
    op_kind: GpuOpKind::Custom("structure_analysis"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("bandwidth")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Real and logical gpuArray inputs use the provider bandwidth hook and read back only the small [lower, upper] result. Complex-interleaved inputs and providers without bandwidth support fall back to the host path.",
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::structure::diagonal_triangular"
)]
pub const ISTRIU_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "istriu",
    op_kind: GpuOpKind::Custom("structure_analysis"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("bandwidth")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Real and logical gpuArray inputs use the provider bandwidth hook and read back only the small [lower, upper] result. Complex-interleaved inputs and providers without bandwidth support fall back to the host path.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::structure::diagonal_triangular"
)]
pub const ISDIAG_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isdiag",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Returns a host logical scalar and acts as a fusion sink.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::structure::diagonal_triangular"
)]
pub const ISTRIL_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "istril",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Returns a host logical scalar and acts as a fusion sink.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::structure::diagonal_triangular"
)]
pub const ISTRIU_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "istriu",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Returns a host logical scalar and acts as a fusion sink.",
};

#[derive(Clone, Copy)]
enum StructurePredicate {
    Diagonal,
    LowerTriangular,
    UpperTriangular,
}

#[derive(Clone, Copy)]
struct BuiltinContext {
    name: &'static str,
    invalid_input: &'static BuiltinErrorDescriptor,
    internal: &'static BuiltinErrorDescriptor,
}

#[runtime_builtin(
    name = "isdiag",
    category = "math/linalg/structure",
    summary = "Determine whether a matrix is diagonal.",
    keywords = "isdiag,diagonal,matrix structure,sparse,gpu",
    accel = "structure",
    type_resolver(logical_scalar_type),
    descriptor(crate::builtins::math::linalg::structure::diagonal_triangular::ISDIAG_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::structure::diagonal_triangular"
)]
async fn isdiag_builtin(value: Value) -> BuiltinResult<Value> {
    structure_predicate_builtin(
        value,
        StructurePredicate::Diagonal,
        BuiltinContext {
            name: "isdiag",
            invalid_input: &ISDIAG_ERROR_INVALID_INPUT,
            internal: &ISDIAG_ERROR_INTERNAL,
        },
    )
    .await
}

#[runtime_builtin(
    name = "istril",
    category = "math/linalg/structure",
    summary = "Determine whether a matrix is lower triangular.",
    keywords = "istril,lower triangular,matrix structure,sparse,gpu",
    accel = "structure",
    type_resolver(logical_scalar_type),
    descriptor(crate::builtins::math::linalg::structure::diagonal_triangular::ISTRIL_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::structure::diagonal_triangular"
)]
async fn istril_builtin(value: Value) -> BuiltinResult<Value> {
    structure_predicate_builtin(
        value,
        StructurePredicate::LowerTriangular,
        BuiltinContext {
            name: "istril",
            invalid_input: &ISTRIL_ERROR_INVALID_INPUT,
            internal: &ISTRIL_ERROR_INTERNAL,
        },
    )
    .await
}

#[runtime_builtin(
    name = "istriu",
    category = "math/linalg/structure",
    summary = "Determine whether a matrix is upper triangular.",
    keywords = "istriu,upper triangular,matrix structure,sparse,gpu",
    accel = "structure",
    type_resolver(logical_scalar_type),
    descriptor(crate::builtins::math::linalg::structure::diagonal_triangular::ISTRIU_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::structure::diagonal_triangular"
)]
async fn istriu_builtin(value: Value) -> BuiltinResult<Value> {
    structure_predicate_builtin(
        value,
        StructurePredicate::UpperTriangular,
        BuiltinContext {
            name: "istriu",
            invalid_input: &ISTRIU_ERROR_INVALID_INPUT,
            internal: &ISTRIU_ERROR_INTERNAL,
        },
    )
    .await
}

async fn structure_predicate_builtin(
    value: Value,
    predicate: StructurePredicate,
    ctx: BuiltinContext,
) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = value {
        return Ok(Value::Bool(
            gpu_structure_predicate(handle, predicate, ctx).await?,
        ));
    }
    let matrix = MatrixInput::from_value(value, ctx).await?;
    Ok(Value::Bool(matrix.satisfies(predicate)))
}

async fn gpu_structure_predicate(
    handle: GpuTensorHandle,
    predicate: StructurePredicate,
    ctx: BuiltinContext,
) -> BuiltinResult<bool> {
    if runmat_accelerate_api::handle_storage(&handle) != GpuTensorStorage::ComplexInterleaved {
        let Some((_rows, _cols)) = matrix_dims_or_false(&handle.shape) else {
            return Ok(false);
        };
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle)
            .or_else(runmat_accelerate_api::provider)
        {
            match provider.bandwidth(&handle) {
                Ok(bandwidth) => return Ok(bandwidth_satisfies(bandwidth, predicate)),
                Err(err) => debug!("{}: provider bandwidth fallback: {err}", ctx.name),
            }
        }
    }

    let matrix = matrix_from_gpu(handle, ctx).await?;
    Ok(matrix.satisfies(predicate))
}

fn bandwidth_satisfies(bandwidth: ProviderBandwidth, predicate: StructurePredicate) -> bool {
    match predicate {
        StructurePredicate::Diagonal => bandwidth.lower == 0 && bandwidth.upper == 0,
        StructurePredicate::LowerTriangular => bandwidth.upper == 0,
        StructurePredicate::UpperTriangular => bandwidth.lower == 0,
    }
}

enum MatrixInput {
    DenseReal {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
    DenseInteger {
        rows: usize,
        cols: usize,
        data: Vec<IntValue>,
    },
    DenseComplex {
        rows: usize,
        cols: usize,
        data: Vec<(f64, f64)>,
    },
    Sparse(SparseTensor),
    TooManyDimensions,
}

impl MatrixInput {
    async fn from_value(value: Value, ctx: BuiltinContext) -> BuiltinResult<Self> {
        if let Value::GpuTensor(handle) = value {
            return matrix_from_gpu(handle, ctx).await;
        }
        matrix_from_host_value(value, ctx)
    }

    fn satisfies(&self, predicate: StructurePredicate) -> bool {
        match self {
            Self::DenseReal { rows, cols, data } => {
                dense_real_satisfies(*rows, *cols, data, predicate)
            }
            Self::DenseInteger { rows, cols, data } => {
                dense_integer_satisfies(*rows, *cols, data, predicate)
            }
            Self::DenseComplex { rows, cols, data } => {
                dense_complex_satisfies(*rows, *cols, data, predicate)
            }
            Self::Sparse(sparse) => sparse_satisfies(sparse, predicate),
            Self::TooManyDimensions => false,
        }
    }
}

fn matrix_from_host_value(value: Value, ctx: BuiltinContext) -> BuiltinResult<MatrixInput> {
    match value {
        Value::Tensor(tensor) => matrix_from_real_tensor(tensor, ctx),
        Value::LogicalArray(logical) => matrix_from_logical(logical, ctx),
        Value::SparseTensor(sparse) => Ok(MatrixInput::Sparse(sparse)),
        Value::ComplexTensor(tensor) => matrix_from_complex_tensor(tensor, ctx),
        Value::Num(n) => Ok(MatrixInput::DenseReal {
            rows: 1,
            cols: 1,
            data: vec![n],
        }),
        Value::Int(i) => Ok(MatrixInput::DenseReal {
            rows: 1,
            cols: 1,
            data: vec![i.to_f64()],
        }),
        Value::Bool(b) => Ok(MatrixInput::DenseReal {
            rows: 1,
            cols: 1,
            data: vec![if b { 1.0 } else { 0.0 }],
        }),
        Value::Complex(re, im) => Ok(MatrixInput::DenseComplex {
            rows: 1,
            cols: 1,
            data: vec![(re, im)],
        }),
        other => Err(runtime_error_with_detail(
            ctx,
            ctx.invalid_input,
            format!(
                "unsupported input type {:?}; expected numeric, logical, sparse, complex, or gpuArray matrix",
                other
            ),
        )),
    }
}

async fn matrix_from_gpu(
    handle: GpuTensorHandle,
    ctx: BuiltinContext,
) -> BuiltinResult<MatrixInput> {
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|err| runtime_error_with_detail(ctx, ctx.internal, err))?;
    matrix_from_host_value(gathered, ctx)
}

fn matrix_from_real_tensor(tensor: Tensor, ctx: BuiltinContext) -> BuiltinResult<MatrixInput> {
    let Some((rows, cols)) = matrix_dims_or_false(&tensor.shape) else {
        return Ok(MatrixInput::TooManyDimensions);
    };
    if rows
        .checked_mul(cols)
        .is_none_or(|len| len > tensor_utils::tensor_element_len(&tensor))
    {
        return Err(runtime_error_with_detail(
            ctx,
            ctx.internal,
            "tensor shape exceeds backing data length",
        ));
    }
    match tensor
        .into_numeric_storage()
        .map_err(|error| runtime_error_with_detail(ctx, ctx.internal, error))?
    {
        NumericStorage::F64(data) => Ok(MatrixInput::DenseReal { rows, cols, data }),
        NumericStorage::F32(data) => Ok(MatrixInput::DenseReal {
            rows,
            cols,
            data: data.into_iter().map(f64::from).collect(),
        }),
        storage => Ok(MatrixInput::DenseInteger {
            rows,
            cols,
            data: storage
                .into_integer_storage()
                .expect("non-floating numeric storage is integer")
                .exact_values(),
        }),
    }
}

fn matrix_from_complex_tensor(
    tensor: ComplexTensor,
    ctx: BuiltinContext,
) -> BuiltinResult<MatrixInput> {
    let Some((rows, cols)) = matrix_dims_or_false(&tensor.shape) else {
        return Ok(MatrixInput::TooManyDimensions);
    };
    if rows
        .checked_mul(cols)
        .is_none_or(|len| len > tensor_utils::complex_tensor_element_len(&tensor))
    {
        return Err(runtime_error_with_detail(
            ctx,
            ctx.internal,
            "complex tensor shape exceeds backing data length",
        ));
    }
    Ok(MatrixInput::DenseComplex {
        rows,
        cols,
        data: tensor_utils::complex_tensor_into_values_complex64(tensor)
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect(),
    })
}

fn matrix_from_logical(logical: LogicalArray, ctx: BuiltinContext) -> BuiltinResult<MatrixInput> {
    let Some((rows, cols)) = matrix_dims_or_false(&logical.shape) else {
        return Ok(MatrixInput::TooManyDimensions);
    };
    if rows
        .checked_mul(cols)
        .is_none_or(|len| len > logical.data.len())
    {
        return Err(runtime_error_with_detail(
            ctx,
            ctx.internal,
            "logical array shape exceeds backing data length",
        ));
    }
    Ok(MatrixInput::DenseReal {
        rows,
        cols,
        data: logical
            .data
            .into_iter()
            .map(|value| if value != 0 { 1.0 } else { 0.0 })
            .collect(),
    })
}

fn matrix_dims_or_false(shape: &[usize]) -> Option<(usize, usize)> {
    match shape.len() {
        0 => Some((1, 1)),
        1 => Some((1, shape[0])),
        _ if shape[2..].iter().any(|&dim| dim != 1) => None,
        _ => Some((shape[0], shape[1])),
    }
}

fn dense_real_satisfies(
    rows: usize,
    cols: usize,
    data: &[f64],
    predicate: StructurePredicate,
) -> bool {
    scan_dense(
        rows,
        cols,
        |row, col| {
            let value = data[row + col * rows];
            value != 0.0 || value.is_nan()
        },
        predicate,
    )
}

fn dense_integer_satisfies(
    rows: usize,
    cols: usize,
    data: &[IntValue],
    predicate: StructurePredicate,
) -> bool {
    scan_dense(
        rows,
        cols,
        |row, col| !data[row + col * rows].is_zero(),
        predicate,
    )
}

fn dense_complex_satisfies(
    rows: usize,
    cols: usize,
    data: &[(f64, f64)],
    predicate: StructurePredicate,
) -> bool {
    scan_dense(
        rows,
        cols,
        |row, col| {
            let (re, im) = data[row + col * rows];
            !(re == 0.0 && im == 0.0)
        },
        predicate,
    )
}

fn scan_dense(
    rows: usize,
    cols: usize,
    mut is_nonzero: impl FnMut(usize, usize) -> bool,
    predicate: StructurePredicate,
) -> bool {
    for col in 0..cols {
        for row in 0..rows {
            if is_forbidden_nonzero(row, col, predicate) && is_nonzero(row, col) {
                return false;
            }
        }
    }
    true
}

fn sparse_satisfies(sparse: &SparseTensor, predicate: StructurePredicate) -> bool {
    for col in 0..sparse.cols {
        for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let row = sparse.row_indices[idx];
            let value = sparse.values[idx];
            if is_forbidden_nonzero(row, col, predicate) && (value != 0.0 || value.is_nan()) {
                return false;
            }
        }
    }
    true
}

fn is_forbidden_nonzero(row: usize, col: usize, predicate: StructurePredicate) -> bool {
    match predicate {
        StructurePredicate::Diagonal => row != col,
        StructurePredicate::LowerTriangular => row < col,
        StructurePredicate::UpperTriangular => row > col,
    }
}

fn runtime_error_with_detail(
    ctx: BuiltinContext,
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(ctx.name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, HostTensorOwned, HostTensorView,
    };
    use runmat_builtins::{
        IntValue, IntegerComplexStorage, IntegerStorage, NumericStorage, ResolveContext, Type,
    };

    use crate::builtins::common::test_support;

    fn call_isdiag(value: Value) -> BuiltinResult<Value> {
        block_on(isdiag_builtin(value))
    }

    fn call_istril(value: Value) -> BuiltinResult<Value> {
        block_on(istril_builtin(value))
    }

    fn call_istriu(value: Value) -> BuiltinResult<Value> {
        block_on(istriu_builtin(value))
    }

    fn expect_bool(value: Value) -> bool {
        match value {
            Value::Bool(value) => value,
            other => panic!("expected bool, got {other:?}"),
        }
    }

    #[test]
    fn descriptors_cover_core_forms() {
        assert_eq!(ISDIAG_DESCRIPTOR.signatures[0].label, "tf = isdiag(A)");
        assert_eq!(ISTRIL_DESCRIPTOR.signatures[0].label, "tf = istril(A)");
        assert_eq!(ISTRIU_DESCRIPTOR.signatures[0].label, "tf = istriu(A)");
    }

    #[test]
    fn type_resolver_returns_logical_scalar() {
        let out = logical_scalar_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Bool);
    }

    #[test]
    fn scalars_are_diagonal_and_triangular() {
        assert!(expect_bool(
            call_isdiag(Value::Int(IntValue::I32(7))).unwrap()
        ));
        assert!(expect_bool(call_istril(Value::Num(7.0)).unwrap()));
        assert!(expect_bool(call_istriu(Value::Bool(true)).unwrap()));
        assert!(expect_bool(call_isdiag(Value::Complex(0.0, 2.0)).unwrap()));
    }

    #[test]
    fn dense_diagonal_matrix_satisfies_all_three_predicates() {
        let tensor = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
            vec![3, 3],
        )
        .unwrap();
        assert!(expect_bool(
            call_isdiag(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(expect_bool(call_istriu(Value::Tensor(tensor)).unwrap()));
    }

    #[test]
    fn lower_and_upper_triangular_are_distinguished() {
        let lower = Tensor::new(
            vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        assert!(!expect_bool(
            call_isdiag(Value::Tensor(lower.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::Tensor(lower.clone())).unwrap()
        ));
        assert!(!expect_bool(call_istriu(Value::Tensor(lower)).unwrap()));

        let upper = Tensor::new(
            vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        assert!(!expect_bool(
            call_isdiag(Value::Tensor(upper.clone())).unwrap()
        ));
        assert!(!expect_bool(
            call_istril(Value::Tensor(upper.clone())).unwrap()
        ));
        assert!(expect_bool(call_istriu(Value::Tensor(upper)).unwrap()));
    }

    #[test]
    fn dense_structure_predicates_read_typed_integer_storage_exactly() {
        let lower = Tensor::new_integer(
            IntegerStorage::I16(vec![1, 2, 3, 0, 4, 5, 0, 0, 6]),
            vec![3, 3],
        )
        .unwrap();

        assert!(!expect_bool(
            call_isdiag(Value::Tensor(lower.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::Tensor(lower.clone())).unwrap()
        ));
        assert!(!expect_bool(call_istriu(Value::Tensor(lower)).unwrap()));
    }

    #[test]
    fn dense_structure_predicates_read_native_single_storage() {
        let lower = Tensor::from_numeric_storage(
            NumericStorage::F32(vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]),
            vec![3, 3],
        )
        .unwrap();
        assert!(!expect_bool(
            call_isdiag(Value::Tensor(lower.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::Tensor(lower.clone())).unwrap()
        ));
        assert!(!expect_bool(call_istriu(Value::Tensor(lower)).unwrap()));
    }

    #[test]
    fn dense_structure_predicates_read_wide_integer_storage_without_float_mirror() {
        let upper = Tensor::new_integer(
            IntegerStorage::U64(vec![1, 0, 0, u64::MAX, 1_u64 << 63, 0, 0, 1, 1]),
            vec![3, 3],
        )
        .expect("upper triangular integer matrix");

        assert!(!expect_bool(
            call_isdiag(Value::Tensor(upper.clone())).unwrap()
        ));
        assert!(!expect_bool(
            call_istril(Value::Tensor(upper.clone())).unwrap()
        ));
        assert!(expect_bool(call_istriu(Value::Tensor(upper)).unwrap()));
    }

    #[test]
    fn rectangular_diagonal_matrix_can_be_true() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0], vec![3, 2]).unwrap();
        assert!(expect_bool(
            call_isdiag(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(expect_bool(call_istriu(Value::Tensor(tensor)).unwrap()));
    }

    #[test]
    fn zero_and_empty_matrices_are_vacuously_true() {
        let zeros = Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap();
        assert!(expect_bool(
            call_isdiag(Value::Tensor(zeros.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::Tensor(zeros.clone())).unwrap()
        ));
        assert!(expect_bool(call_istriu(Value::Tensor(zeros)).unwrap()));

        let empty = Tensor::new(Vec::new(), vec![0, 4]).unwrap();
        assert!(expect_bool(
            call_isdiag(Value::Tensor(empty.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::Tensor(empty.clone())).unwrap()
        ));
        assert!(expect_bool(call_istriu(Value::Tensor(empty)).unwrap()));
    }

    #[test]
    fn one_dimensional_tensor_shape_uses_row_vector_semantics() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(!expect_bool(
            call_isdiag(Value::Tensor(row.clone())).unwrap()
        ));
        assert!(!expect_bool(
            call_istril(Value::Tensor(row.clone())).unwrap()
        ));
        assert!(expect_bool(call_istriu(Value::Tensor(row)).unwrap()));
    }

    #[test]
    fn arrays_with_nonsingleton_trailing_dimensions_return_false() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        assert!(!expect_bool(
            call_isdiag(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(!expect_bool(
            call_istril(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(!expect_bool(call_istriu(Value::Tensor(tensor)).unwrap()));
    }

    #[test]
    fn arrays_with_trailing_zero_dimensions_return_false() {
        let tensor = Tensor::new(Vec::new(), vec![2, 2, 0]).unwrap();
        assert!(!expect_bool(
            call_isdiag(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(!expect_bool(
            call_istril(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(!expect_bool(call_istriu(Value::Tensor(tensor)).unwrap()));
    }

    #[test]
    fn logical_arrays_are_supported() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        assert!(expect_bool(
            call_isdiag(Value::LogicalArray(logical)).unwrap()
        ));
    }

    #[test]
    fn complex_off_structure_value_is_nonzero() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 0.0), (0.0, 2.0), (0.0, 0.0), (1.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        assert!(!expect_bool(
            call_isdiag(Value::ComplexTensor(tensor.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::ComplexTensor(tensor.clone())).unwrap()
        ));
        assert!(!expect_bool(
            call_istriu(Value::ComplexTensor(tensor)).unwrap()
        ));
    }

    #[test]
    fn dense_structure_predicates_read_typed_complex_integer_storage_exactly() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 0, 0, 0]),
            IntegerStorage::I16(vec![0, 2, 0, 0]),
        )
        .unwrap();
        let mut tensor = ComplexTensor::new_integer(storage, vec![2, 2]).unwrap();
        tensor.data.clear();

        assert!(!expect_bool(
            call_isdiag(Value::ComplexTensor(tensor.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::ComplexTensor(tensor.clone())).unwrap()
        ));
        assert!(!expect_bool(
            call_istriu(Value::ComplexTensor(tensor)).unwrap()
        ));
    }

    #[test]
    fn sparse_inputs_are_scanned_without_densifying() {
        let sparse =
            SparseTensor::new(3, 3, vec![0, 2, 3, 3], vec![0, 2, 1], vec![1.0, 4.0, 2.0]).unwrap();
        assert!(!expect_bool(
            call_isdiag(Value::SparseTensor(sparse.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::SparseTensor(sparse.clone())).unwrap()
        ));
        assert!(!expect_bool(
            call_istriu(Value::SparseTensor(sparse)).unwrap()
        ));
    }

    #[test]
    fn explicit_nan_off_structure_counts_as_nonzero() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 0.0, 1.0], vec![2, 2]).unwrap();
        assert!(!expect_bool(
            call_isdiag(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::Tensor(tensor.clone())).unwrap()
        ));
        assert!(!expect_bool(call_istriu(Value::Tensor(tensor)).unwrap()));
    }

    #[derive(Clone, Copy)]
    struct FixedBandwidthProvider {
        lower: u32,
        upper: u32,
        device_id: u32,
    }

    impl AccelProvider for FixedBandwidthProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: self.device_id,
                buffer_id: 1,
            })
        }

        fn download<'a>(&'a self, _h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move { Err(anyhow::anyhow!("download should not be called")) })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "fixed-bandwidth-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            self.device_id
        }

        fn bandwidth(&self, _matrix: &GpuTensorHandle) -> anyhow::Result<ProviderBandwidth> {
            Ok(ProviderBandwidth {
                lower: self.lower,
                upper: self.upper,
            })
        }
    }

    static DIAGONAL_PROVIDER: FixedBandwidthProvider = FixedBandwidthProvider {
        lower: 0,
        upper: 0,
        device_id: 92_001,
    };
    static LOWER_PROVIDER: FixedBandwidthProvider = FixedBandwidthProvider {
        lower: 2,
        upper: 0,
        device_id: 92_002,
    };
    static UPPER_PROVIDER: FixedBandwidthProvider = FixedBandwidthProvider {
        lower: 0,
        upper: 2,
        device_id: 92_003,
    };

    fn fixed_gpu_handle(provider: &FixedBandwidthProvider) -> Value {
        Value::GpuTensor(GpuTensorHandle {
            shape: vec![3, 3],
            device_id: provider.device_id(),
            buffer_id: 1,
        })
    }

    #[derive(Clone, Copy)]
    struct FallbackDownloadProvider;

    impl AccelProvider for FallbackDownloadProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: self.device_id(),
                buffer_id: 1,
            })
        }

        fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![1.0, 3.0, 0.0, 2.0],
                    shape: h.shape.clone(),
                    storage: GpuTensorStorage::Real,
                })
            })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "fallback-download-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            92_004
        }

        fn bandwidth(&self, _matrix: &GpuTensorHandle) -> anyhow::Result<ProviderBandwidth> {
            Err(anyhow::anyhow!("bandwidth intentionally unavailable"))
        }
    }

    static FALLBACK_PROVIDER: FallbackDownloadProvider = FallbackDownloadProvider;

    #[test]
    fn gpu_inputs_use_provider_bandwidth_without_gathering() {
        let _guard = test_support::accel_test_lock();
        let _thread_provider =
            runmat_accelerate_api::ThreadProviderGuard::set(Some(&DIAGONAL_PROVIDER));
        assert!(expect_bool(
            call_isdiag(fixed_gpu_handle(&DIAGONAL_PROVIDER)).unwrap()
        ));
        assert!(expect_bool(
            call_istril(fixed_gpu_handle(&DIAGONAL_PROVIDER)).unwrap()
        ));
        assert!(expect_bool(
            call_istriu(fixed_gpu_handle(&DIAGONAL_PROVIDER)).unwrap()
        ));

        let _thread_provider =
            runmat_accelerate_api::ThreadProviderGuard::set(Some(&LOWER_PROVIDER));
        assert!(!expect_bool(
            call_isdiag(fixed_gpu_handle(&LOWER_PROVIDER)).unwrap()
        ));
        assert!(expect_bool(
            call_istril(fixed_gpu_handle(&LOWER_PROVIDER)).unwrap()
        ));
        assert!(!expect_bool(
            call_istriu(fixed_gpu_handle(&LOWER_PROVIDER)).unwrap()
        ));

        let _thread_provider =
            runmat_accelerate_api::ThreadProviderGuard::set(Some(&UPPER_PROVIDER));
        assert!(!expect_bool(
            call_isdiag(fixed_gpu_handle(&UPPER_PROVIDER)).unwrap()
        ));
        assert!(!expect_bool(
            call_istril(fixed_gpu_handle(&UPPER_PROVIDER)).unwrap()
        ));
        assert!(expect_bool(
            call_istriu(fixed_gpu_handle(&UPPER_PROVIDER)).unwrap()
        ));
    }

    #[test]
    fn gpu_inputs_fall_back_to_download_when_bandwidth_is_unavailable() {
        let _guard = test_support::accel_test_lock();
        let _thread_provider =
            runmat_accelerate_api::ThreadProviderGuard::set(Some(&FALLBACK_PROVIDER));
        let handle = Value::GpuTensor(GpuTensorHandle {
            shape: vec![2, 2],
            device_id: FALLBACK_PROVIDER.device_id(),
            buffer_id: 1,
        });
        assert!(!expect_bool(call_isdiag(handle.clone()).unwrap()));
        assert!(expect_bool(call_istril(handle.clone()).unwrap()));
        assert!(!expect_bool(call_istriu(handle).unwrap()));
    }

    #[test]
    fn gpu_input_with_inprocess_provider_works() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .unwrap();
            assert!(expect_bool(call_isdiag(Value::GpuTensor(handle)).unwrap()));
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_inputs_use_bandwidth_predicates() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let lower = Tensor::new(
            vec![1.0, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 3.0],
            vec![3, 3],
        )
        .unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &lower.materialize_f64(),
                shape: &lower.shape,
            })
            .expect("upload lower");
        assert!(!expect_bool(
            call_isdiag(Value::GpuTensor(handle.clone())).unwrap()
        ));
        assert!(expect_bool(
            call_istril(Value::GpuTensor(handle.clone())).unwrap()
        ));
        assert!(!expect_bool(call_istriu(Value::GpuTensor(handle)).unwrap()));
    }

    #[test]
    fn unsupported_inputs_error() {
        let err = call_isdiag(Value::String("x".to_string())).unwrap_err();
        assert!(err.message.contains("isdiag: invalid input"));
    }
}
