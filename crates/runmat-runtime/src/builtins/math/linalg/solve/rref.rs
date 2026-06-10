//! MATLAB-compatible `rref` builtin for reduced row echelon form.

use num_complex::Complex64;
use runmat_accelerate_api::{
    GpuTensorHandle, HostTensorView, ProviderRrefOptions, ProviderRrefResult,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::linalg::{eps_like, matrix_dimensions_for, parse_tolerance_arg};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::rref_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rref";

const RREF_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Reduced row echelon form of A.",
}];

const RREF_OUTPUT_R_PIVOT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reduced row echelon form of A.",
    },
    BuiltinParamDescriptor {
        name: "pivcol",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based pivot column indices.",
    },
];

const RREF_INPUT_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

const RREF_INPUT_A_TOL: [BuiltinParamDescriptor; 2] = [
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
        description: "Pivot tolerance.",
    },
];

const RREF_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "R = rref(A)",
        inputs: &RREF_INPUT_A,
        outputs: &RREF_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = rref(A, tol)",
        inputs: &RREF_INPUT_A_TOL,
        outputs: &RREF_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, pivcol] = rref(A)",
        inputs: &RREF_INPUT_A,
        outputs: &RREF_OUTPUT_R_PIVOT,
    },
    BuiltinSignatureDescriptor {
        label: "[R, pivcol] = rref(A, tol)",
        inputs: &RREF_INPUT_A_TOL,
        outputs: &RREF_OUTPUT_R_PIVOT,
    },
];

const RREF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.INVALID_ARGUMENT",
    identifier: Some("RunMat:rref:InvalidArgument"),
    when: "Tolerance argument, argument count, or requested output count is invalid.",
    message: "rref: invalid argument",
};

const RREF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.INVALID_INPUT",
    identifier: Some("RunMat:rref:InvalidInput"),
    when: "Input value cannot be converted to a supported numeric matrix.",
    message: "rref: invalid input",
};

const RREF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.INTERNAL",
    identifier: Some("RunMat:rref:Internal"),
    when: "Runtime cannot compute or materialize rref outputs.",
    message: "rref: internal runtime failure",
};

const RREF_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RREF_ERROR_INVALID_ARGUMENT,
    RREF_ERROR_INVALID_INPUT,
    RREF_ERROR_INTERNAL,
];

pub const RREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RREF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RREF_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::rref")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rref"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("rref")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement native RREF; the reference WGPU path gathers to host and re-uploads R and pivot columns.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::rref")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`rref` is an eager matrix reduction/factorization and terminates fusion plans.",
};

fn rref_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    rref_error_with_message(message, &RREF_ERROR_INVALID_ARGUMENT)
}

fn invalid_input(message: impl Into<String>) -> RuntimeError {
    rref_error_with_message(message, &RREF_ERROR_INVALID_INPUT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    rref_error_with_message(message, &RREF_ERROR_INTERNAL)
}

fn with_rref_context(mut error: RuntimeError) -> RuntimeError {
    if error.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(NAME);
    }
    error
}

#[runtime_builtin(
    name = "rref",
    category = "math/linalg/solve",
    summary = "Compute reduced row echelon form and pivot columns.",
    keywords = "rref,reduced row echelon form,pivot,tolerance,matrix,gpu",
    accel = "rref",
    type_resolver(rref_type),
    descriptor(crate::builtins::math::linalg::solve::rref::RREF_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::solve::rref"
)]
async fn rref_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let tol = parse_tolerance_arg(NAME, &rest).map_err(invalid_argument)?;
    let eval = evaluate(value, tol).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        return match out_count {
            0 => Ok(Value::OutputList(Vec::new())),
            1 => Ok(Value::OutputList(vec![eval.reduced()])),
            2 => Ok(Value::OutputList(vec![eval.reduced(), eval.pivots()])),
            _ => Err(invalid_argument(
                "rref currently supports at most two outputs",
            )),
        };
    }
    Ok(eval.reduced())
}

#[derive(Clone)]
pub struct RrefEval {
    reduced: Value,
    pivots: Value,
}

impl RrefEval {
    pub fn reduced(&self) -> Value {
        self.reduced.clone()
    }

    pub fn pivots(&self) -> Value {
        self.pivots.clone()
    }

    fn from_real(result: RrefRealResult) -> Self {
        Self {
            reduced: tensor::tensor_into_value(result.reduced),
            pivots: Value::Tensor(result.pivots),
        }
    }

    fn from_complex(result: RrefComplexResult) -> Self {
        Self {
            reduced: complex_tensor_into_value(result.reduced),
            pivots: Value::Tensor(result.pivots),
        }
    }

    fn from_provider(result: ProviderRrefResult) -> Self {
        Self {
            reduced: Value::GpuTensor(result.reduced),
            pivots: Value::GpuTensor(result.pivots),
        }
    }
}

pub async fn evaluate(value: Value, tol: Option<f64>) -> BuiltinResult<RrefEval> {
    match value {
        Value::GpuTensor(handle) => evaluate_gpu(handle, tol).await,
        other => evaluate_host_value(other, tol),
    }
}

async fn evaluate_gpu(handle: GpuTensorHandle, tol: Option<f64>) -> BuiltinResult<RrefEval> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let options = ProviderRrefOptions { tolerance: tol };
        if let Ok(result) = provider.rref(&handle, options).await {
            return Ok(RrefEval::from_provider(result));
        }
    }

    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(with_rref_context)?;
    let eval = evaluate_host_value(gathered, tol)?;

    if let (Value::Tensor(reduced), Value::Tensor(pivots)) = (eval.reduced(), eval.pivots()) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let reduced_handle = provider
                .upload(&HostTensorView {
                    data: &reduced.data,
                    shape: &reduced.shape,
                })
                .map_err(|err| internal_error(format!("{NAME}: {err}")))?;
            let pivots_handle = provider
                .upload(&HostTensorView {
                    data: &pivots.data,
                    shape: &pivots.shape,
                })
                .map_err(|err| internal_error(format!("{NAME}: {err}")))?;
            return Ok(RrefEval::from_provider(ProviderRrefResult {
                reduced: reduced_handle,
                pivots: pivots_handle,
            }));
        }
    }

    Ok(eval)
}

fn evaluate_host_value(value: Value, tol: Option<f64>) -> BuiltinResult<RrefEval> {
    match value {
        Value::ComplexTensor(tensor) => {
            rref_complex_tensor(&tensor, tol).map(RrefEval::from_complex)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(invalid_input)?;
            rref_complex_tensor(&tensor, tol).map(RrefEval::from_complex)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(invalid_input)?;
            rref_real_tensor(&tensor, tol).map(RrefEval::from_real)
        }
    }
}

#[derive(Clone, Debug)]
pub struct RrefRealResult {
    pub reduced: Tensor,
    pub pivots: Tensor,
}

#[derive(Clone, Debug)]
struct RrefComplexResult {
    reduced: ComplexTensor,
    pivots: Tensor,
}

pub fn rref_host_real_for_provider(
    matrix: &Tensor,
    tol: Option<f64>,
) -> BuiltinResult<RrefRealResult> {
    rref_real_tensor(matrix, tol)
}

fn rref_real_tensor(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<RrefRealResult> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape).map_err(invalid_input)?;
    let tol = tol.unwrap_or_else(|| default_tolerance_real(matrix, rows, cols));
    let mut data = matrix.data.clone();
    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0usize;

    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }
        let mut best_row = pivot_row;
        let mut best_abs = 0.0;
        for row in pivot_row..rows {
            let abs = data[idx(row, col, rows)].abs();
            if abs > best_abs {
                best_abs = abs;
                best_row = row;
            }
        }
        if best_abs <= tol {
            continue;
        }
        if best_row != pivot_row {
            swap_rows_real(&mut data, rows, cols, best_row, pivot_row);
        }

        let pivot = data[idx(pivot_row, col, rows)];
        for c in col..cols {
            data[idx(pivot_row, c, rows)] /= pivot;
        }
        data[idx(pivot_row, col, rows)] = 1.0;

        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = data[idx(row, col, rows)];
            if factor.abs() <= tol {
                data[idx(row, col, rows)] = 0.0;
                continue;
            }
            for c in col..cols {
                let value = data[idx(row, c, rows)] - factor * data[idx(pivot_row, c, rows)];
                data[idx(row, c, rows)] = value;
            }
            data[idx(row, col, rows)] = 0.0;
        }

        pivot_cols.push((col + 1) as f64);
        pivot_row += 1;
    }

    zero_small_real(&mut data, tol);
    let reduced = Tensor::new(data, matrix.shape.clone()).map_err(internal_error)?;
    let pivots = pivot_tensor(pivot_cols)?;
    Ok(RrefRealResult { reduced, pivots })
}

fn rref_complex_tensor(
    matrix: &ComplexTensor,
    tol: Option<f64>,
) -> BuiltinResult<RrefComplexResult> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape).map_err(invalid_input)?;
    let tol = tol.unwrap_or_else(|| default_tolerance_complex(matrix, rows, cols));
    let mut data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0usize;

    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }
        let mut best_row = pivot_row;
        let mut best_abs = 0.0;
        for row in pivot_row..rows {
            let abs = data[idx(row, col, rows)].norm();
            if abs > best_abs {
                best_abs = abs;
                best_row = row;
            }
        }
        if best_abs <= tol {
            continue;
        }
        if best_row != pivot_row {
            swap_rows_complex(&mut data, rows, cols, best_row, pivot_row);
        }

        let pivot = data[idx(pivot_row, col, rows)];
        for c in col..cols {
            data[idx(pivot_row, c, rows)] /= pivot;
        }
        data[idx(pivot_row, col, rows)] = Complex64::new(1.0, 0.0);

        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = data[idx(row, col, rows)];
            if factor.norm() <= tol {
                data[idx(row, col, rows)] = Complex64::new(0.0, 0.0);
                continue;
            }
            for c in col..cols {
                let value = data[idx(row, c, rows)] - factor * data[idx(pivot_row, c, rows)];
                data[idx(row, c, rows)] = value;
            }
            data[idx(row, col, rows)] = Complex64::new(0.0, 0.0);
        }

        pivot_cols.push((col + 1) as f64);
        pivot_row += 1;
    }

    zero_small_complex(&mut data, tol);
    let reduced_data = data.into_iter().map(|value| (value.re, value.im)).collect();
    let reduced = ComplexTensor::new(reduced_data, matrix.shape.clone()).map_err(internal_error)?;
    let pivots = pivot_tensor(pivot_cols)?;
    Ok(RrefComplexResult { reduced, pivots })
}

fn default_tolerance_real(matrix: &Tensor, rows: usize, cols: usize) -> f64 {
    let norm_inf = real_inf_norm(&matrix.data, rows, cols);
    rows.max(cols) as f64 * eps_like(norm_inf)
}

fn default_tolerance_complex(matrix: &ComplexTensor, rows: usize, cols: usize) -> f64 {
    let norm_inf = complex_inf_norm(&matrix.data, rows, cols);
    rows.max(cols) as f64 * eps_like(norm_inf)
}

fn real_inf_norm(data: &[f64], rows: usize, cols: usize) -> f64 {
    let mut best = 0.0;
    for row in 0..rows {
        let mut sum = 0.0;
        for col in 0..cols {
            sum += data[idx(row, col, rows)].abs();
        }
        if sum > best {
            best = sum;
        }
    }
    best
}

fn complex_inf_norm(data: &[(f64, f64)], rows: usize, cols: usize) -> f64 {
    let mut best = 0.0;
    for row in 0..rows {
        let mut sum = 0.0;
        for col in 0..cols {
            let (re, im) = data[idx(row, col, rows)];
            sum += Complex64::new(re, im).norm();
        }
        if sum > best {
            best = sum;
        }
    }
    best
}

#[inline]
fn idx(row: usize, col: usize, rows: usize) -> usize {
    row + col * rows
}

fn swap_rows_real(data: &mut [f64], rows: usize, cols: usize, a: usize, b: usize) {
    for col in 0..cols {
        data.swap(idx(a, col, rows), idx(b, col, rows));
    }
}

fn swap_rows_complex(data: &mut [Complex64], rows: usize, cols: usize, a: usize, b: usize) {
    for col in 0..cols {
        data.swap(idx(a, col, rows), idx(b, col, rows));
    }
}

fn zero_small_real(data: &mut [f64], tol: f64) {
    for value in data {
        if value.abs() <= tol {
            *value = 0.0;
        }
    }
}

fn zero_small_complex(data: &mut [Complex64], tol: f64) {
    for value in data {
        if value.norm() <= tol {
            *value = Complex64::new(0.0, 0.0);
        }
    }
}

fn pivot_tensor(pivots: Vec<f64>) -> BuiltinResult<Tensor> {
    let cols = pivots.len();
    Tensor::new(pivots, vec![1, cols]).map_err(internal_error)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn tensor_from_value(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            Value::GpuTensor(handle) => test_support::gather(Value::GpuTensor(handle)).unwrap(),
            other => panic!("expected tensor-compatible value, got {other:?}"),
        }
    }

    fn complex_tensor_from_value(value: Value) -> ComplexTensor {
        match value {
            Value::ComplexTensor(tensor) => tensor,
            Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1]).unwrap(),
            other => panic!("expected complex tensor-compatible value, got {other:?}"),
        }
    }

    fn assert_tensor_close(actual: &Tensor, expected: &[f64], tol: f64) {
        assert_eq!(actual.data.len(), expected.len(), "length mismatch");
        for (lhs, rhs) in actual.data.iter().zip(expected.iter()) {
            assert!(
                (lhs - rhs).abs() <= tol,
                "expected {lhs} to be within {tol} of {rhs}; actual={:?}",
                actual.data
            );
        }
    }

    #[test]
    fn rref_full_rank_matrix() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = rref_real_tensor(&tensor, None).expect("rref");
        assert_tensor_close(&result.reduced, &[1.0, 0.0, 0.0, 1.0], 1e-12);
        assert_eq!(result.pivots.shape, vec![1, 2]);
        assert_eq!(result.pivots.data, vec![1.0, 2.0]);
    }

    #[test]
    fn rref_rank_deficient_matrix() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = rref_real_tensor(&tensor, None).expect("rref");
        assert_tensor_close(&result.reduced, &[1.0, 0.0, 2.0, 0.0], 1e-12);
        assert_eq!(result.pivots.data, vec![1.0]);
    }

    #[test]
    fn rref_all_zero_has_no_pivots() {
        let tensor = Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap();
        let result = rref_real_tensor(&tensor, None).expect("rref");
        assert_tensor_close(&result.reduced, &[0.0; 6], 0.0);
        assert_eq!(result.pivots.shape, vec![1, 0]);
        assert!(result.pivots.data.is_empty());
    }

    #[test]
    fn rref_scalar_values() {
        let nonzero = rref_real_tensor(&Tensor::new(vec![5.0], vec![1, 1]).unwrap(), None).unwrap();
        assert_eq!(nonzero.reduced.data, vec![1.0]);
        assert_eq!(nonzero.pivots.data, vec![1.0]);

        let zero = rref_real_tensor(&Tensor::new(vec![0.0], vec![1, 1]).unwrap(), None).unwrap();
        assert_eq!(zero.reduced.data, vec![0.0]);
        assert!(zero.pivots.data.is_empty());
    }

    #[test]
    fn rref_row_and_column_vectors() {
        let row = Tensor::new(vec![2.0, 4.0, 6.0], vec![1, 3]).unwrap();
        let row_result = rref_real_tensor(&row, None).unwrap();
        assert_tensor_close(&row_result.reduced, &[1.0, 2.0, 3.0], 1e-12);
        assert_eq!(row_result.pivots.data, vec![1.0]);

        let col = Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap();
        let col_result = rref_real_tensor(&col, None).unwrap();
        assert_tensor_close(&col_result.reduced, &[1.0, 0.0, 0.0], 1e-12);
        assert_eq!(col_result.pivots.data, vec![1.0]);
    }

    #[test]
    fn rref_empty_matrix() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = rref_real_tensor(&tensor, None).expect("rref");
        assert_eq!(result.reduced.shape, vec![0, 3]);
        assert!(result.reduced.data.is_empty());
        assert_eq!(result.pivots.shape, vec![1, 0]);
    }

    #[test]
    fn rref_explicit_tolerance_suppresses_small_pivot() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1e-10], vec![2, 2]).unwrap();
        let result = rref_real_tensor(&tensor, Some(1e-9)).expect("rref");
        assert_tensor_close(&result.reduced, &[1.0, 0.0, 0.0, 0.0], 1e-12);
        assert_eq!(result.pivots.data, vec![1.0]);
    }

    #[test]
    fn rref_default_tolerance_uses_inf_norm() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, f64::EPSILON], vec![2, 2]).unwrap();
        let result = rref_real_tensor(&tensor, None).expect("rref");
        assert_tensor_close(&result.reduced, &[1.0, 0.0, 0.0, 0.0], 1e-12);
        assert_eq!(result.pivots.data, vec![1.0]);
    }

    #[test]
    fn rref_complex_matrix() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (2.0, -1.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = rref_complex_tensor(&tensor, None).expect("rref");
        let reduced = complex_tensor_from_value(complex_tensor_into_value(result.reduced));
        assert_eq!(reduced.shape, vec![2, 2]);
        for (actual, expected) in
            reduced
                .data
                .iter()
                .zip([(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)])
        {
            assert!((actual.0 - expected.0).abs() <= 1e-12);
            assert!((actual.1 - expected.1).abs() <= 1e-12);
        }
        assert_eq!(result.pivots.data, vec![1.0, 2.0]);
    }

    #[test]
    fn rref_logical_and_integer_inputs_promote() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let eval = evaluate_host_value(Value::LogicalArray(logical), None).expect("rref");
        assert_tensor_close(
            &tensor_from_value(eval.reduced()),
            &[1.0, 0.0, 0.0, 1.0],
            1e-12,
        );

        let eval = evaluate_host_value(Value::Int(IntValue::I32(7)), None).expect("rref");
        assert_eq!(tensor_from_value(eval.reduced()).data, vec![1.0]);
    }

    #[test]
    fn rref_rejects_invalid_tolerance() {
        let err = block_on(rref_builtin(
            Value::Num(1.0),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
            )],
        ))
        .expect_err("expected error");
        assert_eq!(err.identifier(), Some("RunMat:rref:InvalidArgument"));

        let err = block_on(rref_builtin(Value::Num(1.0), vec![Value::Num(-1.0)]))
            .expect_err("expected error");
        assert_eq!(err.identifier(), Some("RunMat:rref:InvalidArgument"));
    }

    #[test]
    fn rref_output_count_forms() {
        let tensor = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap());

        let _guard = crate::output_count::push_output_count(Some(0));
        let out = block_on(rref_builtin(tensor.clone(), Vec::new())).expect("rref");
        assert!(matches!(out, Value::OutputList(values) if values.is_empty()));
        drop(_guard);

        let _guard = crate::output_count::push_output_count(Some(1));
        let out = block_on(rref_builtin(tensor.clone(), Vec::new())).expect("rref");
        assert!(matches!(out, Value::OutputList(values) if values.len() == 1));
        drop(_guard);

        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(rref_builtin(tensor.clone(), Vec::new())).expect("rref");
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(tensor_from_value(values[1].clone()).data, vec![1.0, 2.0]);
            }
            other => panic!("expected output list, got {other:?}"),
        }
        drop(_guard);

        let _guard = crate::output_count::push_output_count(Some(3));
        let err = block_on(rref_builtin(tensor, Vec::new())).expect_err("expected error");
        assert_eq!(err.identifier(), Some("RunMat:rref:InvalidArgument"));
    }

    #[test]
    fn rref_descriptor_and_type_resolver() {
        let labels: Vec<&str> = RREF_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"R = rref(A)"));
        assert!(labels.contains(&"R = rref(A, tol)"));
        assert!(labels.contains(&"[R, pivcol] = rref(A)"));
        assert!(labels.contains(&"[R, pivcol] = rref(A, tol)"));
        assert_eq!(
            RREF_DESCRIPTOR.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );
        assert_eq!(
            RREF_DESCRIPTOR.completion_policy,
            BuiltinCompletionPolicy::Public
        );
        let codes: Vec<&str> = RREF_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.RREF.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.RREF.INVALID_INPUT"));
        assert!(codes.contains(&"RM.RREF.INTERNAL"));

        let out = rref_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn rref_gpu_provider_round_trip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                })
                .unwrap();
            let _output_guard = crate::output_count::push_output_count(Some(2));
            let out = block_on(rref_builtin(Value::GpuTensor(handle), Vec::new())).expect("rref");
            match out {
                Value::OutputList(values) => {
                    assert_eq!(values.len(), 2);
                    assert!(matches!(values[0], Value::GpuTensor(_)));
                    assert!(matches!(values[1], Value::GpuTensor(_)));
                    assert_tensor_close(
                        &tensor_from_value(values[0].clone()),
                        &[1.0, 0.0, 0.0, 1.0],
                        1e-12,
                    );
                    assert_eq!(tensor_from_value(values[1].clone()).data, vec![1.0, 2.0]);
                }
                other => panic!("expected output list, got {other:?}"),
            }
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn rref_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("provider");
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            })
            .unwrap();
        let _output_guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(rref_builtin(Value::GpuTensor(handle), Vec::new())).expect("rref");
        match out {
            Value::OutputList(values) => {
                assert!(matches!(values[0], Value::GpuTensor(_)));
                assert!(matches!(values[1], Value::GpuTensor(_)));
                assert_tensor_close(
                    &tensor_from_value(values[0].clone()),
                    &[1.0, 0.0, 0.0, 1.0],
                    1e-12,
                );
                assert_eq!(tensor_from_value(values[1].clone()).data, vec![1.0, 2.0]);
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }
}
