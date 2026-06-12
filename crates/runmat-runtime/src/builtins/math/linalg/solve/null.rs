//! MATLAB-compatible `null` builtin for matrix null-space bases.

use std::convert::TryFrom;

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::linalg::{
    matrix_dimensions_for, parse_tolerance_arg, svd_default_tolerance,
};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::solve::rref::{
    default_complex_tolerance, default_real_tolerance, rref_complex_impl, rref_real_impl,
};
use crate::builtins::math::linalg::type_resolvers::null_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "null";

const NULL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Basis for the null space of A.",
}];

const NULL_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

const NULL_INPUTS_A_TOL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Rank tolerance.",
    },
];

const NULL_INPUTS_A_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "`\"r\"` requests a row-reduction basis.",
    },
];

const NULL_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Z = null(A)",
        inputs: &NULL_INPUTS_A,
        outputs: &NULL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Z = null(A, tol)",
        inputs: &NULL_INPUTS_A_TOL,
        outputs: &NULL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Z = null(A, \"r\")",
        inputs: &NULL_INPUTS_A_OPTION,
        outputs: &NULL_OUTPUT,
    },
];

const NULL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NULL.INVALID_ARGUMENT",
    identifier: Some("RunMat:null:InvalidArgument"),
    when: "Optional tolerance or basis option is malformed or outside accepted bounds.",
    message: "null: invalid argument",
};

const NULL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NULL.INVALID_INPUT",
    identifier: Some("RunMat:null:InvalidInput"),
    when: "Input shape/type cannot be processed for null-space evaluation.",
    message: "null: invalid input",
};

const NULL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NULL.INTERNAL",
    identifier: Some("RunMat:null:Internal"),
    when: "Runtime fails while computing the null space or executing fallback/upload paths.",
    message: "null: internal runtime failure",
};

const NULL_ERRORS: [BuiltinErrorDescriptor; 3] = [
    NULL_ERROR_INVALID_ARGUMENT,
    NULL_ERROR_INVALID_INPUT,
    NULL_ERROR_INTERNAL,
];

pub const NULL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NULL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NULL_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::null")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("null-space"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "`null` gathers GPU inputs to the host null-space implementation and re-uploads real bases when a provider is available.",
};

fn null_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn input_error(message: impl Into<String>) -> RuntimeError {
    null_error_with_message(message, &NULL_ERROR_INVALID_INPUT)
}

fn argument_error(message: impl Into<String>) -> RuntimeError {
    null_error_with_message(message, &NULL_ERROR_INVALID_ARGUMENT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    null_error_with_message(message, &NULL_ERROR_INTERNAL)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::null")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`null` is an eager null-space solve and terminates fusion plans.",
};

#[derive(Clone, Copy, Debug)]
enum NullMode {
    Orthonormal { tolerance: Option<f64> },
    RowReduction,
}

#[runtime_builtin(
    name = "null",
    category = "math/linalg/solve",
    summary = "Compute a basis for a matrix null space.",
    keywords = "null,null space,kernel,svd,rref,rational,gpu",
    accel = "sink",
    type_resolver(null_type),
    descriptor(crate::builtins::math::linalg::solve::null::NULL_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::solve::null"
)]
async fn null_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let mode = parse_null_mode(&rest).map_err(argument_error)?;
    match value {
        Value::GpuTensor(handle) => null_gpu(handle, mode).await,
        Value::ComplexTensor(tensor) => null_complex_value(tensor, mode),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(input_error)?;
            null_complex_value(tensor, mode)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(input_error)?;
            null_real_value(tensor, mode)
        }
    }
}

fn parse_null_mode(args: &[Value]) -> Result<NullMode, String> {
    match args.len() {
        0 => Ok(NullMode::Orthonormal { tolerance: None }),
        1 => {
            if let Some(option) = string_option_from_value(&args[0]) {
                let option = option?;
                let normalized = option.trim().to_ascii_lowercase();
                return match normalized.as_str() {
                    "r" | "rational" => Ok(NullMode::RowReduction),
                    _ => Err(format!("{NAME}: unsupported basis option '{option}'")),
                };
            }
            let tolerance = parse_tolerance_arg(NAME, args)?;
            Ok(NullMode::Orthonormal { tolerance })
        }
        _ => Err(format!("{NAME}: too many input arguments")),
    }
}

fn string_option_from_value(value: &Value) -> Option<Result<String, String>> {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Some(String::try_from(value))
        }
        _ => None,
    }
}

async fn null_gpu(handle: GpuTensorHandle, mode: NullMode) -> BuiltinResult<Value> {
    let gathered = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(map_control_flow)?;
    let basis = null_real_tensor(&gathered, mode)?;

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(uploaded) = provider.upload(&HostTensorView {
            data: &basis.data,
            shape: &basis.shape,
        }) {
            return Ok(Value::GpuTensor(uploaded));
        }
    }

    Ok(tensor::tensor_into_value(basis))
}

fn null_real_value(matrix: Tensor, mode: NullMode) -> BuiltinResult<Value> {
    let basis = null_real_tensor(&matrix, mode)?;
    Ok(tensor::tensor_into_value(basis))
}

fn null_complex_value(matrix: ComplexTensor, mode: NullMode) -> BuiltinResult<Value> {
    let basis = null_complex_tensor(&matrix, mode)?;
    Ok(complex_tensor_into_value(basis))
}

fn null_real_tensor(matrix: &Tensor, mode: NullMode) -> BuiltinResult<Tensor> {
    match mode {
        NullMode::Orthonormal { tolerance } => null_real_orthonormal_tensor(matrix, tolerance),
        NullMode::RowReduction => null_real_row_reduction_tensor(matrix),
    }
}

fn null_complex_tensor(matrix: &ComplexTensor, mode: NullMode) -> BuiltinResult<ComplexTensor> {
    match mode {
        NullMode::Orthonormal { tolerance } => null_complex_orthonormal_tensor(matrix, tolerance),
        NullMode::RowReduction => null_complex_row_reduction_tensor(matrix),
    }
}

fn null_real_orthonormal_tensor(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<Tensor> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(input_error)?;
    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|e| internal_error(format!("{NAME}: {e}")));
    }
    if rows == 0 {
        return Tensor::new(identity_basis_real(cols), vec![cols, cols])
            .map_err(|e| internal_error(format!("{NAME}: {e}")));
    }

    let (basis, basis_cols) = real_orthonormal_basis_from_svd(&matrix.data, rows, cols, tol)?;
    Tensor::new(basis, vec![cols, basis_cols]).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn null_complex_orthonormal_tensor(
    matrix: &ComplexTensor,
    tol: Option<f64>,
) -> BuiltinResult<ComplexTensor> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(input_error)?;
    if cols == 0 {
        return ComplexTensor::new(Vec::new(), vec![0, 0])
            .map_err(|e| internal_error(format!("{NAME}: {e}")));
    }
    if rows == 0 {
        let data = identity_basis_complex(cols)
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect();
        return ComplexTensor::new(data, vec![cols, cols])
            .map_err(|e| internal_error(format!("{NAME}: {e}")));
    }

    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let (basis, basis_cols) = complex_orthonormal_basis_from_svd(&data, rows, cols, tol)?;
    let data: Vec<(f64, f64)> = basis
        .into_iter()
        .map(|value| (value.re, value.im))
        .collect();
    ComplexTensor::new(data, vec![cols, basis_cols])
        .map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn null_real_row_reduction_tensor(matrix: &Tensor) -> BuiltinResult<Tensor> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(input_error)?;
    let tolerance = default_real_tolerance(&matrix.data, rows, cols);
    let (basis, basis_cols) = real_row_reduction_basis(matrix.data.clone(), rows, cols, tolerance)?;
    Tensor::new(basis, vec![cols, basis_cols]).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn null_complex_row_reduction_tensor(matrix: &ComplexTensor) -> BuiltinResult<ComplexTensor> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(input_error)?;
    let tolerance = default_complex_tolerance(&matrix.data, rows, cols);
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let (basis, basis_cols) = complex_row_reduction_basis(data, rows, cols, tolerance)?;
    let data: Vec<(f64, f64)> = basis
        .into_iter()
        .map(|value| (value.re, value.im))
        .collect();
    ComplexTensor::new(data, vec![cols, basis_cols])
        .map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn real_orthonormal_basis_from_svd(
    data: &[f64],
    rows: usize,
    cols: usize,
    tol: Option<f64>,
) -> BuiltinResult<(Vec<f64>, usize)> {
    let matrix = DMatrix::from_column_slice(rows, cols, data);
    let svd = SVD::new(matrix, false, true);
    let tolerance =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    let v_t = svd.v_t.ok_or_else(|| {
        internal_error(format!("{NAME}: failed to compute right singular vectors"))
    })?;
    let v = v_t.adjoint();
    let mut row_space = Vec::new();
    let mut basis = Vec::new();
    for idx in 0..svd.singular_values.len() {
        let vector: Vec<f64> = (0..cols).map(|row| v[(row, idx)]).collect();
        if is_rank_singular_value(svd.singular_values[idx], tolerance) {
            row_space.push(vector);
        } else {
            push_orthonormal_real(&mut basis, &row_space, vector, cols);
        }
    }
    let target_nullity = cols.saturating_sub(row_space.len());
    complete_real_null_basis(&mut basis, &row_space, cols, target_nullity);
    let basis_cols = basis.len();
    Ok((flatten_real_columns(basis, cols), basis_cols))
}

fn complex_orthonormal_basis_from_svd(
    data: &[Complex64],
    rows: usize,
    cols: usize,
    tol: Option<f64>,
) -> BuiltinResult<(Vec<Complex64>, usize)> {
    let matrix = DMatrix::from_column_slice(rows, cols, data);
    let svd = SVD::new(matrix, false, true);
    let tolerance =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    let v_t = svd.v_t.ok_or_else(|| {
        internal_error(format!("{NAME}: failed to compute right singular vectors"))
    })?;
    let v = v_t.adjoint();
    let mut row_space = Vec::new();
    let mut basis = Vec::new();
    for idx in 0..svd.singular_values.len() {
        let vector: Vec<Complex64> = (0..cols).map(|row| v[(row, idx)]).collect();
        if is_rank_singular_value(svd.singular_values[idx], tolerance) {
            row_space.push(vector);
        } else {
            push_orthonormal_complex(&mut basis, &row_space, vector, cols);
        }
    }
    let target_nullity = cols.saturating_sub(row_space.len());
    complete_complex_null_basis(&mut basis, &row_space, cols, target_nullity);
    let basis_cols = basis.len();
    Ok((flatten_complex_columns(basis, cols), basis_cols))
}

fn complete_real_null_basis(
    basis: &mut Vec<Vec<f64>>,
    row_space: &[Vec<f64>],
    dim: usize,
    target_cols: usize,
) {
    for idx in 0..dim {
        if basis.len() >= target_cols {
            break;
        }
        let mut vector = vec![0.0; dim];
        vector[idx] = 1.0;
        push_orthonormal_real(basis, row_space, vector, dim);
    }
}

fn complete_complex_null_basis(
    basis: &mut Vec<Vec<Complex64>>,
    row_space: &[Vec<Complex64>],
    dim: usize,
    target_cols: usize,
) {
    for idx in 0..dim {
        if basis.len() >= target_cols {
            break;
        }
        let mut vector = vec![Complex64::new(0.0, 0.0); dim];
        vector[idx] = Complex64::new(1.0, 0.0);
        push_orthonormal_complex(basis, row_space, vector, dim);
    }
}

fn push_orthonormal_real(
    basis: &mut Vec<Vec<f64>>,
    row_space: &[Vec<f64>],
    vector: Vec<f64>,
    dim: usize,
) {
    let mut vector = vector;
    orthogonalize_real(&mut vector, row_space);
    orthogonalize_real(&mut vector, basis);
    orthogonalize_real(&mut vector, row_space);
    orthogonalize_real(&mut vector, basis);
    let norm = real_vector_norm(&vector);
    if norm > completion_threshold(dim) {
        let inv_norm = 1.0 / norm;
        for value in &mut vector {
            *value *= inv_norm;
        }
        basis.push(vector);
    }
}

fn push_orthonormal_complex(
    basis: &mut Vec<Vec<Complex64>>,
    row_space: &[Vec<Complex64>],
    vector: Vec<Complex64>,
    dim: usize,
) {
    let mut vector = vector;
    orthogonalize_complex(&mut vector, row_space);
    orthogonalize_complex(&mut vector, basis);
    orthogonalize_complex(&mut vector, row_space);
    orthogonalize_complex(&mut vector, basis);
    let norm = complex_vector_norm(&vector);
    if norm > completion_threshold(dim) {
        let inv_norm = 1.0 / norm;
        for value in &mut vector {
            *value *= inv_norm;
        }
        basis.push(vector);
    }
}

fn orthogonalize_real(vector: &mut [f64], columns: &[Vec<f64>]) {
    for column in columns {
        let dot = vector
            .iter()
            .zip(column.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>();
        for (value, basis_value) in vector.iter_mut().zip(column.iter()) {
            *value -= dot * basis_value;
        }
    }
}

fn orthogonalize_complex(vector: &mut [Complex64], columns: &[Vec<Complex64>]) {
    for column in columns {
        let dot = column
            .iter()
            .zip(vector.iter())
            .map(|(basis_value, value)| basis_value.conj() * value)
            .sum::<Complex64>();
        for (value, basis_value) in vector.iter_mut().zip(column.iter()) {
            *value -= basis_value * dot;
        }
    }
}

fn real_vector_norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn complex_vector_norm(vector: &[Complex64]) -> f64 {
    vector
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn completion_threshold(dim: usize) -> f64 {
    (dim.max(1) as f64) * f64::EPSILON.sqrt()
}

fn is_rank_singular_value(singular_value: f64, tolerance: f64) -> bool {
    singular_value.is_infinite() || singular_value > tolerance
}

fn flatten_real_columns(columns: Vec<Vec<f64>>, rows: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(rows * columns.len());
    for column in columns {
        data.extend(column);
    }
    data
}

fn flatten_complex_columns(columns: Vec<Vec<Complex64>>, rows: usize) -> Vec<Complex64> {
    let mut data = Vec::with_capacity(rows * columns.len());
    for column in columns {
        data.extend(column);
    }
    data
}

fn identity_basis_real(cols: usize) -> Vec<f64> {
    let mut data = vec![0.0; cols * cols];
    for idx in 0..cols {
        data[idx + idx * cols] = 1.0;
    }
    data
}

fn identity_basis_complex(cols: usize) -> Vec<Complex64> {
    let mut data = vec![Complex64::new(0.0, 0.0); cols * cols];
    for idx in 0..cols {
        data[idx + idx * cols] = Complex64::new(1.0, 0.0);
    }
    data
}

fn real_row_reduction_basis(
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    tolerance: f64,
) -> BuiltinResult<(Vec<f64>, usize)> {
    let (reduced, pivots) = rref_real_impl(data, rows, cols, tolerance)
        .map_err(|err| internal_error(format!("{NAME}: row reduction failed ({err})")))?;
    Ok(basis_from_real_rref(&reduced, rows, cols, &pivots))
}

fn complex_row_reduction_basis(
    data: Vec<Complex64>,
    rows: usize,
    cols: usize,
    tolerance: f64,
) -> BuiltinResult<(Vec<Complex64>, usize)> {
    let (reduced, pivots) = rref_complex_impl(data, rows, cols, tolerance)
        .map_err(|err| internal_error(format!("{NAME}: row reduction failed ({err})")))?;
    Ok(basis_from_complex_rref(&reduced, rows, cols, &pivots))
}

fn basis_from_real_rref(
    reduced: &[f64],
    rows: usize,
    cols: usize,
    pivots: &[usize],
) -> (Vec<f64>, usize) {
    let free_cols = free_columns(cols, pivots);
    let mut basis = vec![0.0; cols * free_cols.len()];
    for (basis_col, &free_col) in free_cols.iter().enumerate() {
        basis[free_col + basis_col * cols] = 1.0;
        for (pivot_row, &pivot_one_based) in pivots.iter().enumerate() {
            let pivot_col = pivot_one_based - 1;
            basis[pivot_col + basis_col * cols] = -reduced[pivot_row + free_col * rows];
        }
    }
    (basis, free_cols.len())
}

fn basis_from_complex_rref(
    reduced: &[Complex64],
    rows: usize,
    cols: usize,
    pivots: &[usize],
) -> (Vec<Complex64>, usize) {
    let free_cols = free_columns(cols, pivots);
    let mut basis = vec![Complex64::new(0.0, 0.0); cols * free_cols.len()];
    for (basis_col, &free_col) in free_cols.iter().enumerate() {
        basis[free_col + basis_col * cols] = Complex64::new(1.0, 0.0);
        for (pivot_row, &pivot_one_based) in pivots.iter().enumerate() {
            let pivot_col = pivot_one_based - 1;
            basis[pivot_col + basis_col * cols] = -reduced[pivot_row + free_col * rows];
        }
    }
    (basis, free_cols.len())
}

fn free_columns(cols: usize, pivots: &[usize]) -> Vec<usize> {
    let mut pivot_marks = vec![false; cols];
    for &pivot in pivots {
        if (1..=cols).contains(&pivot) {
            pivot_marks[pivot - 1] = true;
        }
    }
    pivot_marks
        .into_iter()
        .enumerate()
        .filter_map(|(col, is_pivot)| (!is_pivot).then_some(col))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntValue, ResolveContext, StringArray, Type};

    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!(
                (lhs - rhs).abs() <= tol,
                "expected {lhs} ~= {rhs} within {tol}"
            );
        }
    }

    fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for ((ar, ai), (er, ei)) in actual.iter().zip(expected.iter()) {
            assert!(
                (ar - er).abs() <= tol && (ai - ei).abs() <= tol,
                "expected ({ar}, {ai}) ~= ({er}, {ei}) within {tol}"
            );
        }
    }

    fn assert_az_zero(a: &Tensor, z: &Tensor, tol: f64) {
        assert_eq!(a.shape.len(), 2);
        assert_eq!(z.shape.len(), 2);
        assert_eq!(a.shape[1], z.shape[0], "inner dimension mismatch");
        let rows = a.shape[0];
        let inner = a.shape[1];
        let cols = z.shape[1];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0;
                for k in 0..inner {
                    sum += a.data[row + k * rows] * z.data[k + col * z.shape[0]];
                }
                assert!(
                    sum.abs() <= tol,
                    "expected A*Z ~= 0 at ({row},{col}), got {sum}"
                );
            }
        }
    }

    fn assert_complex_az_zero(a: &ComplexTensor, z: &ComplexTensor, tol: f64) {
        assert_eq!(a.shape.len(), 2);
        assert_eq!(z.shape.len(), 2);
        assert_eq!(a.shape[1], z.shape[0], "inner dimension mismatch");
        let rows = a.shape[0];
        let inner = a.shape[1];
        let cols = z.shape[1];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..inner {
                    let lhs = a.data[row + k * rows];
                    let rhs = z.data[k + col * z.shape[0]];
                    sum += Complex64::new(lhs.0, lhs.1) * Complex64::new(rhs.0, rhs.1);
                }
                assert!(
                    sum.norm() <= tol,
                    "expected A*Z ~= 0 at ({row},{col}), got {sum}"
                );
            }
        }
    }

    fn assert_columns_orthonormal(z: &Tensor, tol: f64) {
        let rows = z.shape[0];
        let cols = z.shape[1];
        for col in 0..cols {
            let norm = (0..rows)
                .map(|row| z.data[row + col * rows] * z.data[row + col * rows])
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() <= tol, "column norm {norm}");
            for other in 0..col {
                let dot = (0..rows)
                    .map(|row| z.data[row + col * rows] * z.data[row + other * rows])
                    .sum::<f64>();
                assert!(dot.abs() <= tol, "column dot {dot}");
            }
        }
    }

    fn assert_complex_columns_orthonormal(z: &ComplexTensor, tol: f64) {
        let rows = z.shape[0];
        let cols = z.shape[1];
        for col in 0..cols {
            let norm = (0..rows)
                .map(|row| {
                    let value = z.data[row + col * rows];
                    value.0 * value.0 + value.1 * value.1
                })
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() <= tol, "column norm {norm}");
            for other in 0..col {
                let dot = (0..rows)
                    .map(|row| {
                        let lhs = z.data[row + other * rows];
                        let rhs = z.data[row + col * rows];
                        Complex64::new(lhs.0, -lhs.1) * Complex64::new(rhs.0, rhs.1)
                    })
                    .sum::<Complex64>();
                assert!(dot.norm() <= tol, "column dot {dot}");
            }
        }
    }

    #[test]
    fn null_type_uses_input_column_count_and_unknown_nullity() {
        let out = null_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), None])
            }
        );
    }

    #[test]
    fn null_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = NULL_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"Z = null(A)"));
        assert!(labels.contains(&"Z = null(A, tol)"));
        assert!(labels.contains(&"Z = null(A, \"r\")"));
    }

    #[test]
    fn null_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = NULL_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.NULL.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.NULL.INVALID_INPUT"));
        assert!(codes.contains(&"RM.NULL.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_full_rank_square_returns_empty_basis() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let result = null_builtin(Value::Tensor(tensor), Vec::new()).expect("null");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 0]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected empty tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_rank_deficient_square_returns_orthonormal_basis() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = null_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("null");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_az_zero(&tensor, &out, 1e-12);
                assert_columns_orthonormal(&out, 1e-12);
            }
            other => panic!("expected tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_rectangular_wide_matrix_returns_full_nullity() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let result = null_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("null");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_az_zero(&tensor, &out, 1e-12);
                assert_columns_orthonormal(&out, 1e-12);
            }
            other => panic!("expected tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_large_scale_default_tolerance_keeps_exact_null_column() {
        let tensor = Tensor::new(vec![1e20, 0.0], vec![1, 2]).unwrap();
        let result = null_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("null");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_az_zero(&tensor, &out, 1e-6);
                assert_columns_orthonormal(&out, 1e-12);
                assert!(out.data[0].abs() <= 1e-12);
                assert!((out.data[1].abs() - 1.0).abs() <= 1e-12);
            }
            other => panic!("expected tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_row_reduction_option_returns_pivot_basis() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let chars = CharArray::new("r".chars().collect(), 1, 1).unwrap();
        let result = null_builtin(Value::Tensor(tensor), vec![Value::CharArray(chars)])
            .expect("null rref basis");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_close(&out.data, &[-2.0, 1.0, 0.0, -3.0, 0.0, 1.0], 1e-12);
            }
            other => panic!("expected tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_row_reduction_accepts_string_array_option() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let option = StringArray::new(vec!["r".to_string()], vec![1, 1]).unwrap();
        let result = null_builtin(Value::Tensor(tensor), vec![Value::StringArray(option)])
            .expect("null rref basis");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_close(&out.data, &[-2.0, 1.0], 1e-12);
            }
            other => panic!("expected tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_custom_tolerance_expands_null_space() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1e-8], vec![2, 2]).unwrap();
        let result = null_builtin(Value::Tensor(tensor), vec![Value::Num(1e-6)]).expect("null");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_close(&out.data, &[0.0, 1.0], 1e-12);
            }
            other => panic!("expected tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_custom_large_tolerance_keeps_full_zero_matrix_basis() {
        let tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let result = null_builtin(Value::Tensor(tensor.clone()), vec![Value::Num(2.0)])
            .expect("null with large tolerance");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_az_zero(&tensor, &out, 1e-12);
                assert_columns_orthonormal(&out, 1e-12);
            }
            other => panic!("expected tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_complex_rank_deficient_matrix() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (2.0, 2.0), (2.0, 0.0), (4.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = null_builtin(Value::ComplexTensor(tensor.clone()), Vec::new()).expect("null");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_complex_az_zero(&tensor, &out, 1e-12);
                assert_complex_columns_orthonormal(&out, 1e-12);
            }
            other => panic!("expected complex tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_complex_row_reduction_option_returns_complex_basis() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (0.0, 2.0)],
            vec![2, 2],
        )
        .unwrap();
        let chars = CharArray::new("r".chars().collect(), 1, 1).unwrap();
        let result = null_builtin(Value::ComplexTensor(tensor), vec![Value::CharArray(chars)])
            .expect("null rref basis");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_complex_close(&out.data, &[(-0.0, -1.0), (1.0, 0.0)], 1e-12);
            }
            other => panic!("expected complex tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_empty_matrices_have_expected_basis_shapes() {
        let zero_by_three = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = null_builtin(Value::Tensor(zero_by_three), Vec::new()).expect("null");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 3]);
                assert_close(
                    &out.data,
                    &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    1e-12,
                );
            }
            other => panic!("expected identity basis, got {other:?}"),
        }

        let three_by_zero = Tensor::new(Vec::<f64>::new(), vec![3, 0]).unwrap();
        let result = null_builtin(Value::Tensor(three_by_zero), Vec::new()).expect("null");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![0, 0]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected 0x0 tensor basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_scalars_follow_matrix_null_space_rules() {
        let zero = null_builtin(Value::Bool(false), Vec::new()).expect("null zero");
        let nonzero = null_builtin(Value::Int(IntValue::I32(5)), Vec::new()).expect("null int");
        match zero {
            Value::Num(value) => assert_eq!(value, 1.0),
            other => panic!("expected scalar one basis, got {other:?}"),
        }
        match nonzero {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 0]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected empty scalar null basis, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_rejects_invalid_option() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let chars = CharArray::new("q".chars().collect(), 1, 1).unwrap();
        let err = unwrap_error(
            null_builtin(Value::Tensor(tensor), vec![Value::CharArray(chars)]).unwrap_err(),
        );
        assert!(err.message().contains("unsupported basis option"));
        assert_eq!(err.identifier(), NULL_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_rejects_negative_tolerance() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = unwrap_error(
            null_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(-1))]).unwrap_err(),
        );
        assert!(err.message().contains("tolerance"));
        assert_eq!(err.identifier(), NULL_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn null_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = null_builtin(Value::GpuTensor(handle), Vec::new()).expect("gpu null");
            let gathered = test_support::gather(result).expect("gather");
            let cpu = null_real_tensor(&tensor, NullMode::Orthonormal { tolerance: None })
                .expect("cpu null");
            assert_eq!(gathered.shape, cpu.shape);
            assert_close(&gathered.data, &cpu.data, 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn null_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let cpu =
            null_real_tensor(&tensor, NullMode::Orthonormal { tolerance: None }).expect("cpu null");

        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = null_builtin(Value::GpuTensor(handle), Vec::new()).expect("gpu null");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_close(&gathered.data, &cpu.data, 5e-5);
    }

    fn null_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::null_builtin(value, rest))
    }
}
