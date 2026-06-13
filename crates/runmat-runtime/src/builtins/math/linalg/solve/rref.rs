//! MATLAB-compatible `rref` builtin using Gauss-Jordan row reduction.

use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
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
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
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

const RREF_OUTPUT_R_PIVOTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reduced row echelon form of A.",
    },
    BuiltinParamDescriptor {
        name: "pivots",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based pivot column indices.",
    },
];

const RREF_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

const RREF_INPUTS_A_TOL: [BuiltinParamDescriptor; 2] = [
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
        inputs: &RREF_INPUTS_A,
        outputs: &RREF_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = rref(A, tol)",
        inputs: &RREF_INPUTS_A_TOL,
        outputs: &RREF_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, pivots] = rref(A)",
        inputs: &RREF_INPUTS_A,
        outputs: &RREF_OUTPUT_R_PIVOTS,
    },
    BuiltinSignatureDescriptor {
        label: "[R, pivots] = rref(A, tol)",
        inputs: &RREF_INPUTS_A_TOL,
        outputs: &RREF_OUTPUT_R_PIVOTS,
    },
];

const RREF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.INVALID_ARGUMENT",
    identifier: Some("RunMat:rref:InvalidArgument"),
    when: "Optional tolerance argument is malformed or outside accepted bounds.",
    message: "rref: invalid argument",
};

const RREF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.INVALID_INPUT",
    identifier: Some("RunMat:rref:InvalidInput"),
    when: "Input shape/type cannot be processed for row reduction.",
    message: "rref: invalid input",
};

const RREF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rref:TooManyOutputs"),
    when: "More than two output arguments are requested.",
    message: "rref: too many output arguments",
};

const RREF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.INTERNAL",
    identifier: Some("RunMat:rref:Internal"),
    when: "Runtime fails while computing row reduction or executing gather/upload paths.",
    message: "rref: internal runtime failure",
};

const RREF_ERRORS: [BuiltinErrorDescriptor; 4] = [
    RREF_ERROR_INVALID_ARGUMENT,
    RREF_ERROR_INVALID_INPUT,
    RREF_ERROR_TOO_MANY_OUTPUTS,
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
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "`rref` gathers GPU inputs to the host row-reduction implementation and re-uploads real reduced matrices when a provider is available.",
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

fn input_error(message: impl Into<String>) -> RuntimeError {
    rref_error_with_message(message, &RREF_ERROR_INVALID_INPUT)
}

fn argument_error(message: impl Into<String>) -> RuntimeError {
    rref_error_with_message(message, &RREF_ERROR_INVALID_ARGUMENT)
}

fn too_many_outputs_error() -> RuntimeError {
    rref_error_with_message(
        RREF_ERROR_TOO_MANY_OUTPUTS.message,
        &RREF_ERROR_TOO_MANY_OUTPUTS,
    )
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    rref_error_with_message(message, &RREF_ERROR_INTERNAL)
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::rref")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`rref` is an eager row-reduction solve and terminates fusion plans.",
};

#[runtime_builtin(
    name = "rref",
    category = "math/linalg/solve",
    summary = "Compute reduced row echelon form and pivot columns.",
    keywords = "rref,reduced row echelon form,row reduction,pivots,rank",
    accel = "sink",
    type_resolver(rref_type),
    descriptor(crate::builtins::math::linalg::solve::rref::RREF_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::solve::rref"
)]
async fn rref_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let tol = parse_tolerance_arg(NAME, &rest).map_err(argument_error)?;
    let eval = match value {
        Value::GpuTensor(handle) => rref_gpu(handle, tol).await?,
        other => rref_eval_from_value(other, tol)?,
    };
    eval.into_requested_outputs()
}

async fn rref_gpu(handle: GpuTensorHandle, tol: Option<f64>) -> BuiltinResult<RrefEval> {
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(map_control_flow)?;
    let mut eval = rref_eval_from_value(gathered, tol)?;

    if let Value::Tensor(matrix) = &eval.reduced {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &matrix.data,
                shape: &matrix.shape,
            };
            let uploaded = provider.upload(&view).map_err(|err| {
                internal_error(format!("{NAME}: failed to upload reduced matrix ({err})"))
            })?;
            eval.reduced = Value::GpuTensor(uploaded);
        }
    }

    Ok(eval)
}

fn rref_eval_from_value(value: Value, tol: Option<f64>) -> BuiltinResult<RrefEval> {
    match value {
        Value::ComplexTensor(tensor) => rref_complex_eval(tensor, tol),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(input_error)?;
            rref_complex_eval(tensor, tol)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(input_error)?;
            rref_real_eval(tensor, tol)
        }
    }
}

#[derive(Debug, Clone)]
struct RrefEval {
    reduced: Value,
    pivots: Tensor,
}

impl RrefEval {
    fn into_requested_outputs(self) -> BuiltinResult<Value> {
        if let Some(out_count) = crate::output_count::current_output_count() {
            return match out_count {
                0 => Ok(Value::OutputList(Vec::new())),
                1 => Ok(Value::OutputList(vec![self.reduced])),
                2 => Ok(Value::OutputList(vec![
                    self.reduced,
                    tensor::tensor_into_value(self.pivots),
                ])),
                _ => Err(too_many_outputs_error()),
            };
        }
        Ok(self.reduced)
    }
}

fn rref_real_eval(matrix: Tensor, tol: Option<f64>) -> BuiltinResult<RrefEval> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(input_error)?;
    let tolerance = tol.unwrap_or_else(|| default_real_tolerance(&matrix.data, rows, cols));
    let (reduced, pivots) = rref_real_impl(matrix.data, rows, cols, tolerance)?;
    let reduced = Tensor::new(reduced, vec![rows, cols])
        .map_err(|e| internal_error(format!("{NAME}: {e}")))?;
    Ok(RrefEval {
        reduced: tensor::tensor_into_value(reduced),
        pivots: pivot_tensor(pivots)?,
    })
}

fn rref_complex_eval(matrix: ComplexTensor, tol: Option<f64>) -> BuiltinResult<RrefEval> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(input_error)?;
    let tolerance = tol.unwrap_or_else(|| default_complex_tolerance(&matrix.data, rows, cols));
    let data: Vec<Complex64> = matrix
        .data
        .into_iter()
        .map(|(re, im)| Complex64::new(re, im))
        .collect();
    let (reduced, pivots) = rref_complex_impl(data, rows, cols, tolerance)?;
    let reduced_pairs: Vec<(f64, f64)> = reduced
        .into_iter()
        .map(|value| (value.re, value.im))
        .collect();
    let reduced = ComplexTensor::new(reduced_pairs, vec![rows, cols])
        .map_err(|e| internal_error(format!("{NAME}: {e}")))?;
    Ok(RrefEval {
        reduced: complex_tensor_into_value(reduced),
        pivots: pivot_tensor(pivots)?,
    })
}

fn rref_real_impl(
    mut data: Vec<f64>,
    rows: usize,
    cols: usize,
    tolerance: f64,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    if rows == 0 || cols == 0 {
        return Ok((data, Vec::new()));
    }
    let mut pivot_row = 0;
    let mut pivots = Vec::new();

    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }

        let Some((max_row, max_abs)) = max_real_pivot(&data, rows, col, pivot_row) else {
            continue;
        };
        if max_abs <= tolerance {
            zero_real_column_from(&mut data, rows, col, pivot_row);
            continue;
        }

        if max_row != pivot_row {
            swap_rows_real(&mut data, rows, cols, max_row, pivot_row);
        }

        let pivot = data[index(pivot_row, col, rows)];
        for c in 0..cols {
            data[index(pivot_row, c, rows)] /= pivot;
        }

        for r in 0..rows {
            if r == pivot_row {
                continue;
            }
            let factor = data[index(r, col, rows)];
            if factor == 0.0 {
                continue;
            }
            for c in 0..cols {
                let idx = index(r, c, rows);
                let pivot_value = data[index(pivot_row, c, rows)];
                data[idx] -= factor * pivot_value;
            }
        }

        for r in 0..rows {
            data[index(r, col, rows)] = if r == pivot_row { 1.0 } else { 0.0 };
        }
        pivots.push(col + 1);
        pivot_row += 1;
    }

    zero_small_real(&mut data, tolerance);
    Ok((data, pivots))
}

fn rref_complex_impl(
    mut data: Vec<Complex64>,
    rows: usize,
    cols: usize,
    tolerance: f64,
) -> BuiltinResult<(Vec<Complex64>, Vec<usize>)> {
    if rows == 0 || cols == 0 {
        return Ok((data, Vec::new()));
    }
    let mut pivot_row = 0;
    let mut pivots = Vec::new();

    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }

        let Some((max_row, max_abs)) = max_complex_pivot(&data, rows, col, pivot_row) else {
            continue;
        };
        if max_abs <= tolerance {
            zero_complex_column_from(&mut data, rows, col, pivot_row);
            continue;
        }

        if max_row != pivot_row {
            swap_rows_complex(&mut data, rows, cols, max_row, pivot_row);
        }

        let pivot = data[index(pivot_row, col, rows)];
        for c in 0..cols {
            data[index(pivot_row, c, rows)] /= pivot;
        }

        for r in 0..rows {
            if r == pivot_row {
                continue;
            }
            let factor = data[index(r, col, rows)];
            if factor == Complex64::new(0.0, 0.0) {
                continue;
            }
            for c in 0..cols {
                let idx = index(r, c, rows);
                let pivot_value = data[index(pivot_row, c, rows)];
                data[idx] -= factor * pivot_value;
            }
        }

        for r in 0..rows {
            data[index(r, col, rows)] = if r == pivot_row {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
        pivots.push(col + 1);
        pivot_row += 1;
    }

    zero_small_complex(&mut data, tolerance);
    Ok((data, pivots))
}

fn default_real_tolerance(data: &[f64], rows: usize, cols: usize) -> f64 {
    default_tolerance(real_inf_norm(data, rows, cols), rows, cols)
}

fn default_complex_tolerance(data: &[(f64, f64)], rows: usize, cols: usize) -> f64 {
    default_tolerance(complex_inf_norm(data, rows, cols), rows, cols)
}

fn default_tolerance(norm_inf: f64, rows: usize, cols: usize) -> f64 {
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    rows.max(cols) as f64 * eps_like(norm_inf)
}

fn real_inf_norm(data: &[f64], rows: usize, cols: usize) -> f64 {
    let mut norm = 0.0_f64;
    for r in 0..rows {
        let mut row_sum = 0.0_f64;
        for c in 0..cols {
            let abs = data[index(r, c, rows)].abs();
            if abs.is_nan() {
                return f64::NAN;
            }
            row_sum += abs;
        }
        norm = norm.max(row_sum);
    }
    norm
}

fn complex_inf_norm(data: &[(f64, f64)], rows: usize, cols: usize) -> f64 {
    let mut norm = 0.0_f64;
    for r in 0..rows {
        let mut row_sum = 0.0_f64;
        for c in 0..cols {
            let (re, im) = data[index(r, c, rows)];
            let abs = re.hypot(im);
            if abs.is_nan() {
                return f64::NAN;
            }
            row_sum += abs;
        }
        norm = norm.max(row_sum);
    }
    norm
}

fn max_real_pivot(data: &[f64], rows: usize, col: usize, start_row: usize) -> Option<(usize, f64)> {
    if start_row >= rows {
        return None;
    }
    let mut best_row = start_row;
    let mut best_abs = data[index(start_row, col, rows)].abs();
    if best_abs.is_nan() {
        return Some((best_row, best_abs));
    }
    for r in start_row..rows {
        let abs = data[index(r, col, rows)].abs();
        if abs.is_nan() {
            return Some((r, abs));
        }
        if abs > best_abs {
            best_abs = abs;
            best_row = r;
        }
    }
    Some((best_row, best_abs))
}

fn max_complex_pivot(
    data: &[Complex64],
    rows: usize,
    col: usize,
    start_row: usize,
) -> Option<(usize, f64)> {
    if start_row >= rows {
        return None;
    }
    let mut best_row = start_row;
    let mut best_abs = data[index(start_row, col, rows)].norm();
    if best_abs.is_nan() {
        return Some((best_row, best_abs));
    }
    for r in start_row..rows {
        let abs = data[index(r, col, rows)].norm();
        if abs.is_nan() {
            return Some((r, abs));
        }
        if abs > best_abs {
            best_abs = abs;
            best_row = r;
        }
    }
    Some((best_row, best_abs))
}

fn zero_real_column_from(data: &mut [f64], rows: usize, col: usize, start_row: usize) {
    for r in start_row..rows {
        data[index(r, col, rows)] = 0.0;
    }
}

fn zero_complex_column_from(data: &mut [Complex64], rows: usize, col: usize, start_row: usize) {
    for r in start_row..rows {
        data[index(r, col, rows)] = Complex64::new(0.0, 0.0);
    }
}

fn swap_rows_real(data: &mut [f64], rows: usize, cols: usize, a: usize, b: usize) {
    for c in 0..cols {
        data.swap(index(a, c, rows), index(b, c, rows));
    }
}

fn swap_rows_complex(data: &mut [Complex64], rows: usize, cols: usize, a: usize, b: usize) {
    for c in 0..cols {
        data.swap(index(a, c, rows), index(b, c, rows));
    }
}

fn zero_small_real(data: &mut [f64], tolerance: f64) {
    if !tolerance.is_finite() {
        return;
    }
    for value in data {
        if value.abs() <= tolerance {
            *value = 0.0;
        }
    }
}

fn zero_small_complex(data: &mut [Complex64], tolerance: f64) {
    if !tolerance.is_finite() {
        return;
    }
    for value in data {
        if value.norm() <= tolerance {
            *value = Complex64::new(0.0, 0.0);
        }
    }
}

fn pivot_tensor(pivots: Vec<usize>) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = pivots.into_iter().map(|idx| idx as f64).collect();
    let len = data.len();
    Tensor::new(data, vec![1, len]).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

#[inline]
fn index(row: usize, col: usize, rows: usize) -> usize {
    row + col * rows
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, ResolveContext, Type};

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

    fn output_list(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn rref_type_preserves_matrix_shape() {
        let out = rref_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(4)])
            }
        );
    }

    #[test]
    fn rref_type_normalizes_vector_and_trailing_singleton_shapes() {
        let vector = rref_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            vector,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );

        let trailing_singleton = rref_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            trailing_singleton,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn rref_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = RREF_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"R = rref(A)"));
        assert!(labels.contains(&"R = rref(A, tol)"));
        assert!(labels.contains(&"[R, pivots] = rref(A)"));
        assert!(labels.contains(&"[R, pivots] = rref(A, tol)"));
    }

    #[test]
    fn rref_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = RREF_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.RREF.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.RREF.INVALID_INPUT"));
        assert!(codes.contains(&"RM.RREF.TOO_MANY_OUTPUTS"));
        assert!(codes.contains(&"RM.RREF.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_rank_deficient_matrix_returns_pivots() {
        let tensor = Tensor::new(
            vec![1.0, 2.0, 1.0, 2.0, 4.0, 1.0, 3.0, 6.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let values = output_list(rref_builtin(Value::Tensor(tensor), Vec::new()).expect("rref"));
        match &values[0] {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 3]);
                assert_close(
                    &out.data,
                    &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 2.0, 0.0],
                    1e-12,
                );
            }
            other => panic!("expected tensor R, got {other:?}"),
        }
        match &values[1] {
            Value::Tensor(pivots) => {
                assert_eq!(pivots.shape, vec![1, 2]);
                assert_eq!(pivots.data, vec![1.0, 2.0]);
            }
            other => panic!("expected pivot tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_rectangular_wide_matrix() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = rref_builtin(Value::Tensor(tensor), Vec::new()).expect("rref");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_close(&out.data, &[1.0, 0.0, 0.0, 1.0, -1.0, 2.0], 1e-12);
            }
            other => panic!("expected tensor R, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_row_vector_pivots_after_leading_zero() {
        let tensor = Tensor::new(vec![0.0, 2.0, 4.0], vec![1, 3]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let values = output_list(rref_builtin(Value::Tensor(tensor), Vec::new()).expect("rref"));
        match &values[0] {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_close(&out.data, &[0.0, 1.0, 2.0], 1e-12);
            }
            other => panic!("expected tensor R, got {other:?}"),
        }
        match &values[1] {
            Value::Num(pivot) => assert_eq!(*pivot, 2.0),
            other => panic!("expected scalar pivot, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_tall_vector_normalizes_shape() {
        let tensor = Tensor::new(vec![0.0, 2.0, 4.0], vec![3, 1]).unwrap();
        let result = rref_builtin(Value::Tensor(tensor), Vec::new()).expect("rref");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_close(&out.data, &[1.0, 0.0, 0.0], 1e-12);
            }
            other => panic!("expected tensor R, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_empty_matrices_preserve_normalized_shapes() {
        let zero_by_three = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = rref_builtin(Value::Tensor(zero_by_three), Vec::new()).expect("rref");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![0, 3]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected 0x3 tensor R, got {other:?}"),
        }

        let three_by_zero = Tensor::new(Vec::<f64>::new(), vec![3, 0]).unwrap();
        let result = rref_builtin(Value::Tensor(three_by_zero), Vec::new()).expect("rref");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 0]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected 3x0 tensor R, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_custom_tolerance_suppresses_small_pivot() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1e-12], vec![2, 2]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let values =
            output_list(rref_builtin(Value::Tensor(tensor), vec![Value::Num(1e-6)]).expect("rref"));
        match &values[0] {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_close(&out.data, &[1.0, 0.0, 0.0, 0.0], 1e-12);
            }
            other => panic!("expected tensor R, got {other:?}"),
        }
        match &values[1] {
            Value::Num(pivot) => assert_eq!(*pivot, 1.0),
            other => panic!("expected scalar pivot, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_zero_matrix_has_no_pivots() {
        let tensor = Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let values = output_list(rref_builtin(Value::Tensor(tensor), Vec::new()).expect("rref"));
        match &values[0] {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![0.0; 6]);
            }
            other => panic!("expected tensor R, got {other:?}"),
        }
        match &values[1] {
            Value::Tensor(pivots) => {
                assert_eq!(pivots.shape, vec![1, 0]);
                assert!(pivots.data.is_empty());
            }
            other => panic!("expected empty pivot tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_complex_diagonal() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (2.0, -3.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = rref_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("rref");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_complex_close(
                    &out.data,
                    &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
                    1e-12,
                );
            }
            other => panic!("expected complex tensor R, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_complex_row_swap_and_elimination() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (3.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = rref_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("rref");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_complex_close(
                    &out.data,
                    &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
                    1e-12,
                );
            }
            other => panic!("expected complex tensor R, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_output_count_zero_and_one() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let zero_output_guard = crate::output_count::push_output_count(Some(0));
        let values =
            output_list(rref_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("rref"));
        assert!(values.is_empty());
        drop(zero_output_guard);

        let _one_output_guard = crate::output_count::push_output_count(Some(1));
        let values = output_list(rref_builtin(Value::Tensor(tensor), Vec::new()).expect("rref"));
        assert_eq!(values.len(), 1);
        match &values[0] {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_close(&out.data, &[1.0, 0.0, 0.0, 1.0], 1e-12);
            }
            other => panic!("expected tensor R, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_infinite_default_tolerance_zeros_columns() {
        let tensor = Tensor::new(vec![f64::INFINITY, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let values = output_list(rref_builtin(Value::Tensor(tensor), Vec::new()).expect("rref"));
        match &values[0] {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0.0, 0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor R, got {other:?}"),
        }
        match &values[1] {
            Value::Tensor(pivots) => assert!(pivots.data.is_empty()),
            other => panic!("expected empty pivot tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_nan_default_tolerance_propagates_nan_arithmetic() {
        let tensor = Tensor::new(vec![f64::NAN, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let values = output_list(rref_builtin(Value::Tensor(tensor), Vec::new()).expect("rref"));
        match &values[0] {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert!(out.data[0].is_nan());
                assert!(out.data[1].is_nan());
                assert_eq!(out.data[2], 0.0);
                assert_eq!(out.data[3], 1.0);
            }
            other => panic!("expected tensor R, got {other:?}"),
        }
        match &values[1] {
            Value::Tensor(pivots) => assert_eq!(pivots.data, vec![1.0, 2.0]),
            other => panic!("expected pivot tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_scalars_and_logicals() {
        let zero = rref_builtin(Value::Bool(false), Vec::new()).expect("rref zero");
        let nonzero = rref_builtin(Value::Int(IntValue::I32(5)), Vec::new()).expect("rref int");
        match zero {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar zero, got {other:?}"),
        }
        match nonzero {
            Value::Num(value) => assert_eq!(value, 1.0),
            other => panic!("expected scalar one, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_invalid_shape_errors() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let err = unwrap_error(rref_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err());
        assert!(
            err.message().contains("2-D matrices or vectors"),
            "unexpected error message: {err}"
        );
        assert_eq!(err.identifier(), RREF_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_invalid_tolerance_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err =
            unwrap_error(rref_builtin(Value::Tensor(tensor), vec![Value::Num(-1.0)]).unwrap_err());
        assert!(
            err.message().contains("tolerance must be >= 0"),
            "unexpected error message: {err}"
        );
        assert_eq!(err.identifier(), RREF_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_too_many_outputs_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = unwrap_error(rref_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err());
        assert_eq!(err.identifier(), RREF_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_gpu_input_reduces_matrix() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = rref_builtin(Value::GpuTensor(handle), Vec::new()).expect("rref");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_close(&gathered.data, &[1.0, 0.0, 2.0, 0.0], 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rref_gpu_two_outputs_returns_gpu_matrix_and_host_pivots() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let _guard = crate::output_count::push_output_count(Some(2));
            let values =
                output_list(rref_builtin(Value::GpuTensor(handle), Vec::new()).expect("rref"));
            match &values[0] {
                Value::GpuTensor(_) => {
                    let gathered = test_support::gather(values[0].clone()).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    assert_close(&gathered.data, &[1.0, 0.0, 2.0, 0.0], 1e-12);
                }
                other => panic!("expected gpu tensor R, got {other:?}"),
            }
            match &values[1] {
                Value::Num(pivot) => assert_eq!(*pivot, 1.0),
                other => panic!("expected scalar pivot, got {other:?}"),
            }
        });
    }

    fn rref_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::rref_builtin(value, rest))
    }
}
