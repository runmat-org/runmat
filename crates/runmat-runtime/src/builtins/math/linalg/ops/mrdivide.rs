//! MATLAB-compatible `mrdivide` builtin (`/`) for solving right-sided linear systems.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ComplexTensor, IntegerStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{try_integer_binary, IntegerBinaryOp};
use crate::builtins::math::linalg::type_resolvers::right_divide_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "mrdivide";

const MRDIVIDE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Solution to X * B = A.",
}];

const MRDIVIDE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left-hand side matrix or scalar.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right coefficient matrix or scalar.",
    },
];

const MRDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = mrdivide(A, B)",
    inputs: &MRDIVIDE_INPUTS,
    outputs: &MRDIVIDE_OUTPUT,
}];

const MRDIVIDE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MRDIVIDE.INVALID_INPUT",
    identifier: Some("RunMat:mrdivide:InvalidInput"),
    when: "Inputs are unsupported or incompatible for right division.",
    message: "mrdivide: invalid input",
};

const MRDIVIDE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MRDIVIDE.INTERNAL",
    identifier: Some("RunMat:mrdivide:Internal"),
    when: "Runtime cannot materialize right-division outputs.",
    message: "mrdivide: internal runtime failure",
};

const MRDIVIDE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [MRDIVIDE_ERROR_INVALID_INPUT, MRDIVIDE_ERROR_INTERNAL];

pub const MRDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MRDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MRDIVIDE_ERRORS,
};

const MRDIVIDE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The integer numerator supplies the preserved output class and shape.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "When an integer participates, B must be scalar; same-class integer and scalar-double forms follow scalar element-wise division.",
    },
];

pub const MRDIVIDE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "X = integer_A / scalar_B",
        inputs: &MRDIVIDE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Integer linear-system solving is unsupported; the documented scalar-right form is exact class-preserving element-wise division, and resident results return to the first input owner.",
    }];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::mrdivide")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mrdivide",
    op_kind: GpuOpKind::Custom("solve"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("mrdivide")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prefers the provider `mrdivide` hook for supported floating solves; scalar-right integer division gathers through exact typed storage, applies class-preserving division, and restores GPU residency for integer results when an input was resident.",
};

fn mrdivide_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn mrdivide_invalid_input(message: impl Into<String>) -> RuntimeError {
    mrdivide_error_with_message(message, &MRDIVIDE_ERROR_INVALID_INPUT)
}

fn mrdivide_internal_error(message: impl Into<String>) -> RuntimeError {
    mrdivide_error_with_message(message, &MRDIVIDE_ERROR_INTERNAL)
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

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::ops::mrdivide"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mrdivide",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Right-division is a terminal operation and does not fuse with surrounding kernels.",
};

#[runtime_builtin(
    name = "mrdivide",
    category = "math/linalg/ops",
    summary = "Solve linear systems using right division.",
    keywords = "mrdivide,matrix division,linear algebra,least squares,gpu",
    accel = "mrdivide",
    type_resolver(right_divide_type),
    descriptor(crate::builtins::math::linalg::ops::mrdivide::MRDIVIDE_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::math::linalg::ops::mrdivide::MRDIVIDE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::linalg::ops::mrdivide"
)]
async fn mrdivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(&rhs)
    {
        return Err(mrdivide_invalid_input(
            "complex integer arithmetic is not supported",
        ));
    }
    mrdivide_eval(&lhs, &rhs).await
}

pub(crate) async fn mrdivide_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    if contains_integer(lhs) || contains_integer(rhs) {
        return mrdivide_integer_eval(lhs, rhs).await;
    }
    if let Some(result) = try_gpu_mrdivide(lhs, rhs).await? {
        return Ok(result);
    }

    let lhs_host = crate::dispatcher::gather_if_needed_async(lhs)
        .await
        .map_err(map_control_flow)?;
    let rhs_host = crate::dispatcher::gather_if_needed_async(rhs)
        .await
        .map_err(map_control_flow)?;
    mrdivide_cpu(lhs_host, rhs_host)
}

async fn mrdivide_integer_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    let restore_provider = [lhs, rhs].into_iter().find_map(|value| match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::provider_for_handle(handle),
        _ => None,
    });
    let lhs = crate::dispatcher::gather_if_needed_async(lhs)
        .await
        .map_err(map_control_flow)?;
    let rhs = crate::dispatcher::gather_if_needed_async(rhs)
        .await
        .map_err(map_control_flow)?;
    let result = mrdivide_cpu(lhs, rhs)?;
    let Some(provider) = restore_provider else {
        return Ok(result);
    };
    let tensor = match result {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => tensor,
        Value::Int(value) => Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
            .map_err(mrdivide_internal_error)?,
        other => return Ok(other),
    };
    let handle = gpu_helpers::upload_tensor(provider, &tensor)
        .map_err(|error| mrdivide_internal_error(format!("{NAME}: {error}")))?;
    Ok(gpu_helpers::resident_gpu_value(handle))
}

async fn try_gpu_mrdivide(lhs: &Value, rhs: &Value) -> BuiltinResult<Option<Value>> {
    let provider = match solve_provider(lhs, rhs) {
        Some(p) => p,
        None => return Ok(None),
    };

    if contains_complex(lhs) || contains_complex(rhs) {
        return Ok(None);
    }
    if selected_floating_output_precision(lhs, rhs) != Some(provider.precision()) {
        return Ok(None);
    }

    let mut lhs_operand = match prepare_gpu_operand(lhs, provider)? {
        Some(op) => op,
        None => return Ok(None),
    };
    let mut rhs_operand = match prepare_gpu_operand(rhs, provider) {
        Err(error) => {
            release_operand(provider, &mut lhs_operand);
            return Err(error);
        }
        Ok(value) => match value {
            Some(op) => op,
            None => {
                release_operand(provider, &mut lhs_operand);
                return Ok(None);
            }
        },
    };

    if is_scalar_handle(rhs_operand.handle()) {
        release_operand(provider, &mut lhs_operand);
        release_operand(provider, &mut rhs_operand);
        return Ok(None);
    }

    let expected_shape = vec![
        matrix_rows(&lhs_operand.handle().shape),
        matrix_rows(&rhs_operand.handle().shape),
    ];
    let result = match provider
        .mrdivide(lhs_operand.handle(), rhs_operand.handle())
        .await
    {
        Ok(output)
            if native_solve_output_matches(&output, provider, &expected_shape)
                && !same_owned_buffer(&output, provider, lhs_operand.handle(), provider)
                && !same_owned_buffer(&output, provider, rhs_operand.handle(), provider) =>
        {
            Some(output)
        }
        Ok(output) => {
            free_rejected_native_output(
                &output,
                provider,
                &[lhs_operand.handle(), rhs_operand.handle()],
            );
            None
        }
        Err(_) => None,
    };
    release_operand(provider, &mut lhs_operand);
    release_operand(provider, &mut rhs_operand);
    Ok(result.map(Value::GpuTensor))
}

fn mrdivide_cpu(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if (contains_integer(&lhs) || contains_integer(&rhs))
        && !contains_complex(&lhs)
        && !contains_complex(&rhs)
    {
        if !scalar_divide_input(&rhs) {
            return Err(mrdivide_invalid_input(
                "mrdivide: integer inputs are only supported for scalar right division",
            ));
        }
        if let Some(result) = try_integer_binary(&lhs, &rhs, IntegerBinaryOp::Divide, NAME)
            .map_err(mrdivide_invalid_input)?
        {
            return Ok(result);
        }
    }
    let lhs_numeric = classify_numeric(lhs)?;
    let rhs_numeric = classify_numeric(rhs)?;

    match (lhs_numeric, rhs_numeric) {
        (NumericInput::Real(lhs_r), NumericInput::Real(rhs_r)) => {
            let result = mrdivide_real(&lhs_r, &rhs_r)?;
            Ok(tensor::tensor_into_value(result))
        }
        (NumericInput::Complex(lhs_c), NumericInput::Complex(rhs_c)) => {
            let result = mrdivide_complex(&lhs_c, &rhs_c)?;
            Ok(complex_tensor_into_value(result))
        }
        (NumericInput::Complex(lhs_c), NumericInput::Real(rhs_r)) => {
            let rhs_c = promote_real_tensor(&rhs_r)?;
            let result = mrdivide_complex(&lhs_c, &rhs_c)?;
            Ok(complex_tensor_into_value(result))
        }
        (NumericInput::Real(lhs_r), NumericInput::Complex(rhs_c)) => {
            let lhs_c = promote_real_tensor(&lhs_r)?;
            let result = mrdivide_complex(&lhs_c, &rhs_c)?;
            Ok(complex_tensor_into_value(result))
        }
    }
}

fn contains_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn scalar_divide_input(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::LogicalArray(logical) => logical.data.len() == 1,
        _ => false,
    }
}

/// Host implementation shared with acceleration providers that keep data on the CPU.
pub fn mrdivide_host_real_for_provider(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    mrdivide_real(lhs, rhs)
}

enum NumericInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_numeric(value: Value) -> BuiltinResult<NumericInput> {
    match value {
        Value::ComplexTensor(tensor) => {
            ensure_matrix_shape("mrdivide", &tensor.shape)?;
            Ok(NumericInput::Complex(tensor))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| mrdivide_invalid_input(format!("{NAME}: {e}")))?;
            Ok(NumericInput::Complex(tensor))
        }
        other => {
            let tensor =
                tensor::value_into_tensor_for(NAME, other).map_err(mrdivide_invalid_input)?;
            ensure_matrix_shape(NAME, &tensor.shape)?;
            Ok(NumericInput::Real(tensor))
        }
    }
}

fn mrdivide_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    ensure_matrix_shape(NAME, &lhs.shape)?;
    ensure_matrix_shape(NAME, &rhs.shape)?;

    if tensor::is_scalar_tensor(rhs) {
        let divisor = tensor::tensor_value_f64(rhs, 0);
        let scaled = scale_real_preserving_float_class(lhs, divisor.recip())?;
        return Ok(scaled);
    }

    ensure_column_match(lhs.cols(), rhs.cols())?;

    if rhs.cols() == 0 {
        let rows = lhs.rows();
        let cols = rhs.rows();
        let result = Tensor::new(vec![0.0; rows * cols], vec![rows, cols])
            .map_err(|e| mrdivide_internal_error(format!("{NAME}: {e}")))?;
        return Ok(result);
    }

    let lhs_values = tensor::tensor_values_f64_cow(lhs);
    let rhs_values = tensor::tensor_values_f64_cow(rhs);
    let lhs_matrix = DMatrix::from_column_slice(lhs.rows(), lhs.cols(), lhs_values.as_ref());
    let rhs_matrix = DMatrix::from_column_slice(rhs.rows(), rhs.cols(), rhs_values.as_ref());
    let solution = solve_real_matrix(&lhs_matrix, &rhs_matrix)?;
    matrix_real_to_tensor(solution, floating_result_dtype(lhs, rhs))
}

fn mrdivide_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> BuiltinResult<ComplexTensor> {
    ensure_matrix_shape(NAME, &lhs.shape)?;
    ensure_matrix_shape(NAME, &rhs.shape)?;

    if complex_tensor_is_scalar(rhs) {
        let divisor = Complex64::new(rhs.materialize_f64()[0].0, rhs.materialize_f64()[0].1);
        let inv = Complex64::new(1.0, 0.0) / divisor;
        let scaled = scale_complex_preserving_float_class(lhs, inv)?;
        return Ok(scaled);
    }

    ensure_column_match(lhs.cols, rhs.cols)?;

    if rhs.cols == 0 {
        let rows = lhs.rows;
        let cols = rhs.rows;
        let result = ComplexTensor::new(vec![(0.0, 0.0); rows * cols], vec![rows, cols])
            .map_err(|e| mrdivide_internal_error(format!("{NAME}: {e}")))?;
        return Ok(result);
    }

    let lhs_data: Vec<Complex64> = lhs
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let rhs_data: Vec<Complex64> = rhs
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let lhs_matrix = DMatrix::from_column_slice(lhs.rows, lhs.cols, &lhs_data);
    let rhs_matrix = DMatrix::from_column_slice(rhs.rows, rhs.cols, &rhs_data);
    let solution = solve_complex_matrix(&lhs_matrix, &rhs_matrix)?;
    matrix_complex_to_tensor(
        solution,
        if lhs.numeric_dtype() == runmat_builtins::NumericDType::F32
            || rhs.numeric_dtype() == runmat_builtins::NumericDType::F32
        {
            runmat_builtins::NumericDType::F32
        } else {
            runmat_builtins::NumericDType::F64
        },
    )
}

fn solve_real_matrix(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> BuiltinResult<DMatrix<f64>> {
    let rhs_t = rhs.transpose();
    let lhs_t = lhs.transpose();
    let svd = SVD::new(rhs_t.clone(), true, true);
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), rhs_t.nrows(), rhs_t.ncols());
    let solved = svd
        .solve(&lhs_t, tol)
        .map_err(|e| mrdivide_invalid_input(format!("{NAME}: {e}")))?;
    Ok(solved.transpose())
}

fn solve_complex_matrix(
    lhs: &DMatrix<Complex64>,
    rhs: &DMatrix<Complex64>,
) -> BuiltinResult<DMatrix<Complex64>> {
    let rhs_t = rhs.transpose();
    let lhs_t = lhs.transpose();
    let svd = SVD::new(rhs_t.clone(), true, true);
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), rhs_t.nrows(), rhs_t.ncols());
    let solved = svd
        .solve(&lhs_t, tol)
        .map_err(|e| mrdivide_invalid_input(format!("{NAME}: {e}")))?;
    Ok(solved.transpose())
}

fn compute_svd_tolerance(singular_values: &[f64], rows: usize, cols: usize) -> f64 {
    let max_sv = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_dim = rows.max(cols) as f64;
    f64::EPSILON * max_dim * max_sv.max(1.0)
}

fn matrix_real_to_tensor(
    matrix: DMatrix<f64>,
    dtype: runmat_builtins::NumericDType,
) -> BuiltinResult<Tensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    Tensor::new_with_dtype(matrix.as_slice().to_vec(), vec![rows, cols], dtype)
        .map_err(|e| mrdivide_internal_error(format!("{NAME}: {e}")))
}

fn matrix_complex_to_tensor(
    matrix: DMatrix<Complex64>,
    dtype: runmat_builtins::NumericDType,
) -> BuiltinResult<ComplexTensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<(f64, f64)> = matrix.as_slice().iter().map(|c| (c.re, c.im)).collect();
    ComplexTensor::from_f64_values_with_dtype(data, vec![rows, cols], dtype)
        .map_err(|e| mrdivide_internal_error(format!("{NAME}: {e}")))
}

fn promote_real_tensor(tensor: &Tensor) -> BuiltinResult<ComplexTensor> {
    let values = tensor::tensor_values_f64_cow(tensor);
    let data: Vec<(f64, f64)> = values.iter().map(|&re| (re, 0.0)).collect();
    let dtype = if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32 {
        runmat_builtins::NumericDType::F32
    } else {
        runmat_builtins::NumericDType::F64
    };
    ComplexTensor::from_f64_values_with_dtype(data, tensor.shape.clone(), dtype)
        .map_err(|e| mrdivide_internal_error(format!("{NAME}: {e}")))
}

fn floating_result_dtype(lhs: &Tensor, rhs: &Tensor) -> runmat_builtins::NumericDType {
    if lhs.numeric_dtype() == runmat_builtins::NumericDType::F32
        || rhs.numeric_dtype() == runmat_builtins::NumericDType::F32
    {
        runmat_builtins::NumericDType::F32
    } else {
        runmat_builtins::NumericDType::F64
    }
}

fn scale_real_preserving_float_class(tensor: &Tensor, scalar: f64) -> BuiltinResult<Tensor> {
    let dtype = if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32 {
        runmat_builtins::NumericDType::F32
    } else {
        runmat_builtins::NumericDType::F64
    };
    Tensor::new_with_dtype(
        tensor::tensor_values_f64_cow(tensor)
            .iter()
            .map(|value| value * scalar)
            .collect(),
        tensor.shape.clone(),
        dtype,
    )
    .map_err(mrdivide_internal_error)
}

fn scale_complex_preserving_float_class(
    tensor: &ComplexTensor,
    scalar: Complex64,
) -> BuiltinResult<ComplexTensor> {
    let dtype = if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32 {
        runmat_builtins::NumericDType::F32
    } else {
        runmat_builtins::NumericDType::F64
    };
    let data = tensor
        .materialize_f64()
        .iter()
        .map(|&(real, imag)| {
            let value = Complex64::new(real, imag) * scalar;
            (value.re, value.im)
        })
        .collect();
    ComplexTensor::from_f64_values_with_dtype(data, tensor.shape.clone(), dtype)
        .map_err(|error| mrdivide_internal_error(format!("{NAME}: {error}")))
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> BuiltinResult<()> {
    if is_effectively_matrix(shape) {
        Ok(())
    } else {
        Err(mrdivide_invalid_input(format!(
            "{name}: inputs must be 2-D matrices or vectors"
        )))
    }
}

fn ensure_column_match(lhs_cols: usize, rhs_cols: usize) -> BuiltinResult<()> {
    if lhs_cols == rhs_cols {
        Ok(())
    } else {
        Err(mrdivide_invalid_input("Matrix dimensions must agree."))
    }
}

fn is_effectively_matrix(shape: &[usize]) -> bool {
    match shape.len() {
        0..=2 => true,
        _ => shape.iter().skip(2).all(|&dim| dim == 1),
    }
}

fn contains_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn complex_tensor_is_scalar(tensor: &ComplexTensor) -> bool {
    tensor.materialize_f64().len() == 1
}

fn is_scalar_handle(handle: &GpuTensorHandle) -> bool {
    handle.shape.iter().copied().product::<usize>() == 1
}

struct PreparedOperand {
    handle: GpuTensorHandle,
    owned: bool,
}

impl PreparedOperand {
    fn borrowed(handle: &GpuTensorHandle) -> Self {
        Self {
            handle: handle.clone(),
            owned: false,
        }
    }

    fn owned(handle: GpuTensorHandle) -> Self {
        Self {
            handle,
            owned: true,
        }
    }

    fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
}

fn prepare_gpu_operand(
    value: &Value,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Option<PreparedOperand>> {
    match value {
        Value::GpuTensor(handle) => {
            let owner = runmat_accelerate_api::provider_for_handle(handle);
            if is_scalar_handle(handle)
                || owner.is_none_or(|owner| !std::ptr::eq(owner, provider))
                || runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
                || runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_precision(handle)
                    .is_some_and(|precision| precision != provider.precision())
            {
                Ok(None)
            } else {
                Ok(Some(PreparedOperand::borrowed(handle)))
            }
        }
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                Ok(None)
            } else {
                let uploaded = upload_tensor(provider, tensor)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(None)
            } else {
                let tensor = tensor::logical_to_tensor(logical).map_err(mrdivide_invalid_input)?;
                let uploaded = upload_tensor(provider, &tensor)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        _ => Ok(None),
    }
}

fn upload_tensor(
    provider: &'static dyn AccelProvider,
    tensor: &Tensor,
) -> BuiltinResult<GpuTensorHandle> {
    gpu_helpers::upload_tensor(provider, tensor)
        .map_err(|e| mrdivide_internal_error(format!("{NAME}: {e}")))
}

fn release_operand(provider: &'static dyn AccelProvider, operand: &mut PreparedOperand) {
    if operand.owned {
        let _ = provider.free(&operand.handle);
        operand.owned = false;
    }
}

fn solve_provider(lhs: &Value, rhs: &Value) -> Option<&'static dyn AccelProvider> {
    let handles: Vec<&GpuTensorHandle> = [lhs, rhs]
        .into_iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect();
    let Some(first) = handles.first() else {
        return runmat_accelerate_api::provider();
    };
    let provider = runmat_accelerate_api::provider_for_handle(first)?;
    if handles.iter().skip(1).all(|handle| {
        runmat_accelerate_api::provider_for_handle(handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
    }) {
        Some(provider)
    } else {
        None
    }
}

fn selected_floating_output_precision(lhs: &Value, rhs: &Value) -> Option<ProviderPrecision> {
    let lhs = value_floating_precision(lhs)?;
    let rhs = value_floating_precision(rhs)?;
    if lhs == ProviderPrecision::F32 || rhs == ProviderPrecision::F32 {
        Some(ProviderPrecision::F32)
    } else {
        Some(ProviderPrecision::F64)
    }
}

fn value_floating_precision(value: &Value) -> Option<ProviderPrecision> {
    match value {
        Value::Num(_) | Value::Bool(_) | Value::LogicalArray(_) => Some(ProviderPrecision::F64),
        Value::Tensor(tensor) => Some(
            if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32 {
                ProviderPrecision::F32
            } else {
                ProviderPrecision::F64
            },
        ),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_precision(handle),
        _ => None,
    }
}

fn native_solve_output_matches(
    output: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    expected_shape: &[usize],
) -> bool {
    output.shape == expected_shape
        && output.device_id == provider.device_id()
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::handle_precision(output)
            == Some(match provider.precision() {
                ProviderPrecision::F32 => ProviderPrecision::F32,
                ProviderPrecision::F64 => ProviderPrecision::F64,
            })
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn free_rejected_native_output(
    output: &GpuTensorHandle,
    invoked_provider: &dyn AccelProvider,
    inputs: &[&GpuTensorHandle],
) {
    let output_owner =
        runmat_accelerate_api::provider_for_handle(output).unwrap_or(invoked_provider);
    if inputs.iter().any(|input| {
        let input_owner =
            runmat_accelerate_api::provider_for_handle(input).unwrap_or(invoked_provider);
        same_owned_buffer(input, input_owner, output, output_owner)
    }) {
        return;
    }
    let _ = output_owner.free(output);
}

fn same_owned_buffer(
    lhs: &GpuTensorHandle,
    lhs_owner: &dyn AccelProvider,
    rhs: &GpuTensorHandle,
    rhs_owner: &dyn AccelProvider,
) -> bool {
    lhs.buffer_id == rhs.buffer_id && std::ptr::eq(lhs_owner, rhs_owner)
}

fn matrix_rows(shape: &[usize]) -> usize {
    shape.first().copied().unwrap_or(1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostTensorView, IntegerElementType, ProviderTelemetry};
    use runmat_builtins::{IntValue, IntegerStorage, ResolveContext, Type};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    fn fallback_count(telemetry: &ProviderTelemetry, reason: &str) -> u64 {
        telemetry
            .solve_fallbacks
            .iter()
            .find(|entry| entry.reason == reason)
            .map(|entry| entry.count)
            .unwrap_or(0)
    }

    fn host_mrdivide_real(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        super::mrdivide_host_real_for_provider(lhs, rhs).expect("host mrdivide")
    }

    fn clear_accel_provider_state() {
        runmat_accelerate_api::set_thread_provider(None);
        runmat_accelerate_api::clear_provider();
    }

    fn integer_scalar_mrdivide_cases() -> Vec<(IntegerStorage, IntValue, IntegerStorage)> {
        vec![
            (
                IntegerStorage::I8(vec![i8::MIN, 6, 4]),
                IntValue::I8(2),
                IntegerStorage::I8(vec![i8::MIN / 2, 3, 2]),
            ),
            (
                IntegerStorage::I16(vec![i16::MIN, 6, 4]),
                IntValue::I16(2),
                IntegerStorage::I16(vec![i16::MIN / 2, 3, 2]),
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, 6, 4]),
                IntValue::I32(2),
                IntegerStorage::I32(vec![i32::MIN / 2, 3, 2]),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN, 6, 4]),
                IntValue::I64(2),
                IntegerStorage::I64(vec![i64::MIN / 2, 3, 2]),
            ),
            (
                IntegerStorage::U8(vec![u8::MAX - 1, 6, 4]),
                IntValue::U8(2),
                IntegerStorage::U8(vec![(u8::MAX - 1) / 2, 3, 2]),
            ),
            (
                IntegerStorage::U16(vec![u16::MAX - 1, 6, 4]),
                IntValue::U16(2),
                IntegerStorage::U16(vec![(u16::MAX - 1) / 2, 3, 2]),
            ),
            (
                IntegerStorage::U32(vec![u32::MAX - 1, 6, 4]),
                IntValue::U32(2),
                IntegerStorage::U32(vec![(u32::MAX - 1) / 2, 3, 2]),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX - 1, (1_u64 << 53) + 2, 4]),
                IntValue::U64(2),
                IntegerStorage::U64(vec![(u64::MAX - 1) / 2, (1_u64 << 52) + 1, 2]),
            ),
        ]
    }

    fn integer_element_type(storage: &IntegerStorage) -> IntegerElementType {
        match storage {
            IntegerStorage::I8(_) => IntegerElementType::I8,
            IntegerStorage::I16(_) => IntegerElementType::I16,
            IntegerStorage::I32(_) => IntegerElementType::I32,
            IntegerStorage::I64(_) => IntegerElementType::I64,
            IntegerStorage::U8(_) => IntegerElementType::U8,
            IntegerStorage::U16(_) => IntegerElementType::U16,
            IntegerStorage::U32(_) => IntegerElementType::U32,
            IntegerStorage::U64(_) => IntegerElementType::U64,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn divides_scalar_by_scalar() {
        let result = mrdivide_builtin(Value::Num(6.0), Value::Num(2.0)).expect("mrdivide");
        match result {
            Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn mrdivide_type_uses_rhs_rows() {
        let out = right_divide_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(4), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4)])
            }
        );
    }

    #[test]
    fn mrdivide_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = MRDIVIDE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"X = mrdivide(A, B)"));
    }

    #[test]
    fn mrdivide_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = MRDIVIDE_DESCRIPTOR
            .errors
            .iter()
            .map(|err| err.code)
            .collect();
        assert!(codes.contains(&"RM.MRDIVIDE.INVALID_INPUT"));
        assert!(codes.contains(&"RM.MRDIVIDE.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn divides_matrix_by_scalar() {
        let tensor = Tensor::new(vec![2.0, 4.0, 6.0], vec![1, 3]).expect("tensor");
        let result = mrdivide_builtin(Value::Tensor(tensor), Value::Num(2.0)).expect("mrdivide");
        match result {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 3.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn host_single_solve_preserves_single_storage() {
        let lhs = Tensor::from_f32(vec![8.0, 12.0], vec![1, 2]).expect("single A");
        let rhs = Tensor::from_f32(vec![2.0, 0.0, 0.0, 4.0], vec![2, 2]).expect("single B");
        let Value::Tensor(result) =
            mrdivide_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("single solve")
        else {
            panic!("expected tensor")
        };
        assert_eq!(result.numeric_dtype(), runmat_builtins::NumericDType::F32);
        assert_eq!(result.materialize_f64(), vec![4.0, 3.0]);
    }

    #[test]
    fn ambient_f64_provider_does_not_widen_host_single_solve() {
        test_support::with_test_provider(|_| {
            let lhs = Tensor::from_f32(vec![8.0, 12.0], vec![1, 2]).expect("A");
            let rhs = Tensor::from_f32(vec![2.0, 0.0, 0.0, 4.0], vec![2, 2]).expect("B");
            let Value::Tensor(result) =
                mrdivide_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("solve")
            else {
                panic!("precision mismatch must use host fallback")
            };
            assert_eq!(result.numeric_dtype(), runmat_builtins::NumericDType::F32);
        });
    }

    #[test]
    fn mrdivide_declares_documented_integer_scalar_capability() {
        let builtin = runmat_builtins::builtin_function_by_name(NAME).expect("mrdivide");
        assert_eq!(builtin.integer_capabilities.len(), 1);
        let capability = &builtin.integer_capabilities[0];
        assert_eq!(capability.inputs.len(), 2);
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(capability.inputs[1].classes.len(), 8);
        assert_eq!(
            capability.output_class,
            BuiltinIntegerOutputClassRule::PreserveInput
        );
        assert_eq!(
            capability.backend,
            BuiltinIntegerBackendRule::GatherFallback
        );
    }

    #[test]
    fn complex_single_scalar_right_division_preserves_single_storage() {
        let lhs = ComplexTensor::from_f32(vec![(8.0, 4.0), (12.0, 6.0)], vec![1, 2]).expect("lhs");
        let divisor = ComplexTensor::from_f32(vec![(2.0, 1.0)], vec![1, 1]).expect("divisor");
        let Value::ComplexTensor(result) =
            mrdivide_builtin(Value::ComplexTensor(lhs), Value::ComplexTensor(divisor))
                .expect("complex single scalar solve")
        else {
            panic!("expected complex tensor")
        };
        assert_eq!(result.numeric_dtype(), runmat_builtins::NumericDType::F32);
        assert_eq!(result.shape, vec![1, 2]);
    }

    #[test]
    fn rejected_native_output_with_foreign_id_collision_is_freed_by_its_owner() {
        let _guard = test_support::accel_test_lock();
        let invoked = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let owner = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        unsafe {
            runmat_accelerate_api::register_provider(invoked);
            runmat_accelerate_api::register_provider(owner);
        }
        let rejected = owner
            .upload(&HostTensorView {
                data: &[99.0; 4],
                shape: &[2, 2],
            })
            .expect("rejected upload");
        let foreign_collision = GpuTensorHandle {
            shape: rejected.shape.clone(),
            device_id: invoked.device_id(),
            buffer_id: rejected.buffer_id,
            descriptor: Default::default(),
        };

        assert!(!same_owned_buffer(
            &rejected,
            owner,
            &foreign_collision,
            invoked
        ));
        free_rejected_native_output(&rejected, invoked, &[&foreign_collision]);
        assert!(block_on(owner.download(&rejected)).is_err());
    }

    #[test]
    fn resident_integer_result_restores_to_the_input_owner_not_the_ambient_provider() {
        let _guard = test_support::accel_test_lock();
        let owner = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let ambient = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        unsafe {
            runmat_accelerate_api::register_provider(owner);
            runmat_accelerate_api::register_provider(ambient);
        }
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![8, 12]), vec![1, 2]).expect("integer row");
        let input_handle = gpu_helpers::upload_tensor(owner, &input).expect("owner upload");

        let result = mrdivide_builtin(
            Value::GpuTensor(input_handle.clone()),
            Value::Int(IntValue::U64(2)),
        )
        .expect("integer divide");
        let Value::GpuTensor(result_handle) = result else {
            panic!("expected resident result")
        };
        let result_owner =
            runmat_accelerate_api::provider_for_handle(&result_handle).expect("restored owner");
        assert!(std::ptr::eq(result_owner, owner));
        assert!(!std::ptr::eq(result_owner, ambient));
        let gathered = block_on(owner.download_integer(&result_handle)).expect("owner download");
        assert_eq!(
            gathered.data,
            runmat_accelerate_api::HostIntegerDataOwned::U64(vec![4, 6])
        );

        owner.free(&input_handle).expect("free input");
        owner.free(&result_handle).expect("free result");
    }

    #[test]
    fn integer_matrix_right_division_by_scalar_preserves_uint64_storage() {
        let values =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
                .expect("integer values");
        let result = mrdivide_builtin(Value::Tensor(values), Value::Num(1.0)).expect("mrdivide");
        assert_eq!(
            result,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
                    .expect("integer result")
            )
        );
    }

    #[test]
    fn integer_array_scalar_right_division_is_exact_for_all_classes() {
        for (array, scalar, expected) in integer_scalar_mrdivide_cases() {
            let array =
                Value::Tensor(Tensor::new_integer(array, vec![1, 3]).expect("integer array"));
            for divisor in [
                Value::Int(scalar.clone()),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::from_scalar(scalar), vec![1, 1])
                        .expect("integer scalar tensor"),
                ),
                Value::Num(2.0),
            ] {
                let result =
                    mrdivide_builtin(array.clone(), divisor).expect("integer scalar mrdivide");
                assert_eq!(
                    result,
                    Value::Tensor(
                        Tensor::new_integer(expected.clone(), vec![1, 3]).expect("integer result")
                    )
                );
            }
        }
    }

    #[test]
    fn integer_scalar_divisor_rejects_nonscalar_double_numerator() {
        let numerator = Tensor::new(vec![2.0, 4.0], vec![1, 2]).expect("numerator");
        let err = mrdivide_builtin(Value::Tensor(numerator), Value::Int(IntValue::I32(2)))
            .expect_err("integer scalar divisor needs scalar numerator");
        assert_eq!(err.identifier(), MRDIVIDE_ERROR_INVALID_INPUT.identifier);
        assert!(err
            .message()
            .contains("integer arrays can only be combined with scalar double values"));
    }

    #[test]
    fn integer_mrdivide_rejects_nonscalar_divisors_and_mixed_classes() {
        let lhs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![6, 4]), vec![1, 2]).expect("lhs"),
        );
        let rhs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![2, 2]), vec![1, 2]).expect("rhs"),
        );
        let error = mrdivide_builtin(lhs, rhs).expect_err("nonscalar integer divisor must reject");
        assert_eq!(error.identifier(), MRDIVIDE_ERROR_INVALID_INPUT.identifier);
        assert!(error.message().contains("scalar right division"));

        let lhs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![6, 4]), vec![1, 2]).expect("lhs"),
        );
        let error = mrdivide_builtin(lhs, Value::Int(IntValue::U16(2)))
            .expect_err("mixed integer classes must reject");
        assert_eq!(error.identifier(), MRDIVIDE_ERROR_INVALID_INPUT.identifier);
        assert!(error.message().contains("same integer class"));
    }

    #[test]
    fn mrdivide_complex_scalar_promotion_reads_typed_integer_storage_exactly() {
        let lhs = ComplexTensor::new(vec![(6.0, 4.0), (2.0, -8.0)], vec![1, 2]).unwrap();
        let divisor =
            Tensor::new_integer(IntegerStorage::I64(vec![2]), vec![1, 1]).expect("integer scalar");

        let result =
            mrdivide_builtin(Value::ComplexTensor(lhs), Value::Tensor(divisor)).expect("mrdivide");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };
        assert_eq!(out.shape, vec![1, 2]);
        assert!((out.materialize_f64()[0].0 - 3.0).abs() < 1e-12);
        assert!((out.materialize_f64()[0].1 - 2.0).abs() < 1e-12);
        assert!((out.materialize_f64()[1].0 - 1.0).abs() < 1e-12);
        assert!((out.materialize_f64()[1].1 + 4.0).abs() < 1e-12);
    }

    #[test]
    fn mrdivide_host_real_reads_typed_integer_storage_exactly() {
        let lhs =
            Tensor::new_integer(IntegerStorage::I16(vec![6, 10]), vec![1, 2]).expect("typed lhs");
        let rhs =
            Tensor::new_integer(IntegerStorage::I16(vec![2]), vec![1, 1]).expect("typed divisor");

        let out = mrdivide_host_real_for_provider(&lhs, &rhs).expect("host mrdivide");

        assert_eq!(out.materialize_f64(), vec![3.0, 5.0]);
        assert!(out.integer_storage().is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn solves_square_system() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result = mrdivide_builtin(Value::Tensor(a), Value::Tensor(b)).expect("mrdivide");
        let gathered = test_support::gather(result).expect("gather");
        let expected = vec![3.0, 2.0, -2.0, -1.0];
        assert_eq!(gathered.shape, vec![2, 2]);
        for (val, exp) in gathered.materialize_f64().iter().zip(expected.into_iter()) {
            assert!((val - exp).abs() < 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn solves_least_squares() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let result =
            mrdivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("mrdivide");
        let gathered = test_support::gather(result).expect("gather");
        let expected = host_mrdivide_real(&a, &b);
        assert_eq!(gathered.shape, expected.shape);
        for (actual, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64().iter())
        {
            assert!(
                (actual - expected).abs() < 1e-10,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn supports_complex_inputs() {
        let a = ComplexTensor::new(
            vec![(1.0, 2.0), (5.0, 6.0), (3.0, -4.0), (7.0, -2.0)],
            vec![2, 2],
        )
        .unwrap();
        let b = ComplexTensor::new(
            vec![(2.0, -1.0), (1.0, 0.5), (0.5, 1.0), (3.0, 2.0)],
            vec![2, 2],
        )
        .unwrap();
        let result =
            mrdivide_builtin(Value::ComplexTensor(a), Value::ComplexTensor(b)).expect("mrdivide");
        match result {
            Value::ComplexTensor(out) => {
                let expected = [
                    (-0.7902439, 1.28780488),
                    (-0.72780488, 3.2897561),
                    (0.48780488, -1.6097561),
                    (2.0097561, -2.31219512),
                ];
                for (value, (er, ei)) in out.materialize_f64().iter().zip(expected.into_iter()) {
                    let (vr, vi) = *value;
                    assert!((vr - er).abs() < 1e-6);
                    assert!((vi - ei).abs() < 1e-6);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reports_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = unwrap_error(mrdivide_builtin(Value::Tensor(a), Value::Tensor(b)).unwrap_err());
        assert_eq!(err.identifier(), MRDIVIDE_ERROR_INVALID_INPUT.identifier);
        assert!(
            err.message().contains("Matrix dimensions must agree"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

            let cpu = mrdivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone()))
                .expect("cpu mrdivide");
            let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

            let view_a = HostTensorView {
                data: &a.materialize_f64(),
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.materialize_f64(),
                shape: &b.shape,
            };
            let ha = provider.upload(&view_a).expect("upload A");
            let hb = provider.upload(&view_b).expect("upload B");
            let result =
                mrdivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
                    .expect("gpu mrdivide");
            let gathered = test_support::gather(result).expect("gather");
            let _ = provider.free(&ha);
            let _ = provider.free(&hb);

            assert_eq!(gathered.shape, cpu_tensor.shape);
            for (gpu, cpu) in gathered
                .materialize_f64()
                .iter()
                .zip(cpu_tensor.materialize_f64().iter())
            {
                assert!((gpu - cpu).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn provider_telemetry_records_gpu_host_reupload_path() {
        test_support::with_test_provider(|provider| {
            provider.reset_telemetry();
            let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
            let ha = provider
                .upload(&HostTensorView {
                    data: &a.materialize_f64(),
                    shape: &a.shape,
                })
                .expect("upload A");
            let hb = provider
                .upload(&HostTensorView {
                    data: &b.materialize_f64(),
                    shape: &b.shape,
                })
                .expect("upload B");

            let _ = mrdivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
                .expect("gpu mrdivide");

            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.mrdivide.count, 1);
            assert!(telemetry.upload_bytes > 0);
            assert!(telemetry.download_bytes > 0);
            assert_eq!(fallback_count(&telemetry, "mrdivide:host_reupload"), 1);

            let _ = provider.free(&ha);
            let _ = provider.free(&hb);
        });
    }

    #[test]
    fn scalar_gpu_rhs_falls_back_without_provider_solve_dispatch() {
        test_support::with_test_provider(|provider| {
            provider.reset_telemetry();
            let matrix = Tensor::new(vec![2.0, 4.0, 6.0], vec![1, 3]).unwrap();
            let scalar = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
            let hm = provider
                .upload(&HostTensorView {
                    data: &matrix.materialize_f64(),
                    shape: &matrix.shape,
                })
                .expect("upload matrix");
            let hs = provider
                .upload(&HostTensorView {
                    data: &scalar.materialize_f64(),
                    shape: &scalar.shape,
                })
                .expect("upload scalar");

            let result =
                mrdivide_eval(&Value::GpuTensor(hm.clone()), &Value::GpuTensor(hs.clone()))
                    .expect("fallback mrdivide");
            let gathered = test_support::gather(result).expect("gather fallback");
            assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 3.0]);

            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.mrdivide.count, 0);
            assert_eq!(fallback_count(&telemetry, "mrdivide:host_reupload"), 0);
            assert!(telemetry.download_bytes > 0);

            let _ = provider.free(&hm);
            let _ = provider.free(&hs);
        });
    }

    #[test]
    fn resident_integer_scalar_mrdivide_preserves_all_classes_and_residency() {
        test_support::with_test_provider(|provider| {
            for (array, scalar, expected) in integer_scalar_mrdivide_cases() {
                let array = Tensor::new_integer(array, vec![1, 3]).expect("resident integer array");
                let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
                let result =
                    mrdivide_builtin(Value::GpuTensor(array_handle), Value::Int(scalar.clone()))
                        .expect("resident integer array divided by host scalar");
                let Value::GpuTensor(result_handle) = &result else {
                    panic!("expected resident integer result, got {result:?}");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(result_handle),
                    Some(integer_element_type(&expected))
                );
                let gathered = test_support::gather(result).expect("gather integer result");
                assert_eq!(gathered.integer_storage(), Some(&expected));

                let scalar = Tensor::new_integer(IntegerStorage::from_scalar(scalar), vec![1, 1])
                    .expect("resident scalar");
                let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
                let scalar_handle = gpu_helpers::upload_tensor(provider, &scalar).expect("upload");
                let result = mrdivide_builtin(
                    Value::GpuTensor(array_handle),
                    Value::GpuTensor(scalar_handle),
                )
                .expect("resident integer array divided by resident scalar");
                let Value::GpuTensor(result_handle) = &result else {
                    panic!("expected resident integer result, got {result:?}");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(result_handle),
                    Some(integer_element_type(&expected))
                );
                let gathered = test_support::gather(result).expect("gather integer result");
                assert_eq!(gathered.integer_storage(), Some(&expected));
            }

            let array =
                Tensor::new_integer(IntegerStorage::U16(vec![6, 4]), vec![1, 2]).expect("array");
            let scalar =
                Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1, 1]).expect("scalar");
            let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
            let scalar_handle = gpu_helpers::upload_tensor(provider, &scalar).expect("upload");
            let result = mrdivide_builtin(
                Value::GpuTensor(array_handle),
                Value::GpuTensor(scalar_handle),
            )
            .expect("singleton-N-D resident scalar mrdivide");
            let gathered = test_support::gather(result).expect("gather integer result");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U16(vec![3, 2]))
            );
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_integer_scalar_mrdivide_preserves_all_classes_and_residency() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        for (array, scalar, expected) in integer_scalar_mrdivide_cases() {
            let array = Tensor::new_integer(array, vec![1, 3]).expect("wgpu integer array");
            let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
            let result =
                mrdivide_builtin(Value::GpuTensor(array_handle), Value::Int(scalar.clone()))
                    .expect("wgpu integer scalar mrdivide");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident integer result, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(result_handle),
                Some(integer_element_type(&expected))
            );
            let gathered = test_support::gather(result).expect("gather wgpu integer result");
            assert_eq!(gathered.integer_storage(), Some(&expected));

            let scalar = Tensor::new_integer(IntegerStorage::from_scalar(scalar), vec![1, 1])
                .expect("scalar");
            let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
            let scalar_handle = gpu_helpers::upload_tensor(provider, &scalar).expect("upload");
            let result = mrdivide_builtin(
                Value::GpuTensor(array_handle),
                Value::GpuTensor(scalar_handle),
            )
            .expect("wgpu integer array divided by resident scalar");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident integer result, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(result_handle),
                Some(integer_element_type(&expected))
            );
            let gathered = test_support::gather(result).expect("gather wgpu integer result");
            assert_eq!(gathered.integer_storage(), Some(&expected));
        }
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_wide_path_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = match runmat_accelerate_api::provider() {
            Some(p) => p,
            None => panic!("wgpu provider not available"),
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        provider.reset_telemetry();

        let a = Tensor::new(vec![1.0, 2.0, 2.0], vec![1, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let cpu_tensor = host_mrdivide_real(&a, &b);
        provider.reset_telemetry();

        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload A");
        let hb = provider.upload(&view_b).expect("upload B");
        let gpu_value = mrdivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
            .expect("gpu mrdivide");
        let gathered = test_support::gather(gpu_value).expect("gather");
        let _ = provider.free(&ha);
        let _ = provider.free(&hb);

        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert!(gathered
            .materialize_f64()
            .iter()
            .all(|value| value.is_finite()));

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.mrdivide.count, 1);
        assert_eq!(fallback_count(&telemetry, "mrdivide:host_reupload"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_square_path_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = match runmat_accelerate_api::provider() {
            Some(p) => p,
            None => panic!("wgpu provider not available"),
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        provider.reset_telemetry();

        let a = Tensor::new(vec![7.0, 8.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 2.0, 1.0, 4.0], vec![2, 2]).unwrap();
        let cpu = mrdivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone()))
            .expect("cpu mrdivide");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload A");
        let hb = provider.upload(&view_b).expect("upload B");
        let gpu_value = mrdivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
            .expect("gpu mrdivide");
        let gathered = test_support::gather(gpu_value).expect("gather");
        let _ = provider.free(&ha);
        let _ = provider.free(&hb);

        assert_eq!(gathered.shape, cpu_tensor.shape);
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.mrdivide.count, 1);
        assert_eq!(fallback_count(&telemetry, "mrdivide:host_reupload"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_round_trip_matches_cpu() {
        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = match runmat_accelerate_api::provider() {
            Some(p) => p,
            None => panic!("wgpu provider not available"),
        };

        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let cpu = mrdivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone()))
            .expect("cpu mrdivide");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload A");
        let hb = provider.upload(&view_b).expect("upload B");
        let gpu_value = mrdivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
            .expect("gpu mrdivide");
        let gathered = test_support::gather(gpu_value).expect("gather");
        let _ = provider.free(&ha);
        let _ = provider.free(&hb);

        assert_eq!(gathered.shape, cpu_tensor.shape);
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-10);
        }
    }

    fn mrdivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(super::mrdivide_builtin(lhs, rhs))
    }

    fn mrdivide_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
        block_on(super::mrdivide_eval(lhs, rhs))
    }
}
