//! MATLAB-compatible `det` builtin with GPU-aware semantics for RunMat.

use std::collections::HashSet;

use nalgebra::{DMatrix, LU};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, NumericScalar, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::numeric_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "det";

const DET_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "det-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "det with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DetIntegerInputExtension"),
};
const DET_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "det-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "det with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DetLogicalInputExtension"),
};
const DET_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [DET_INTEGER_INPUT_EXTENSION, DET_LOGICAL_INPUT_EXTENSION];
const DET_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents square single/double matrices; RunMat mode additionally admits all eight real integer classes.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "d = det(integer_A)",
        inputs: &DET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Authoritative integer values enter one explicit binary64 LU boundary. Resident integer input gathers exactly before LU and restores the double scalar to the owning provider.",
    }];

const DET_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Determinant of A.",
}];

const DET_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

const DET_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "d = det(A)",
    inputs: &DET_INPUTS,
    outputs: &DET_OUTPUT,
}];

const DET_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DET.INVALID_INPUT",
    identifier: Some("RunMat:det:InvalidInput"),
    when: "Input shape/type or numeric domain is unsupported for determinant evaluation.",
    message: "det: invalid input",
};

const DET_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DET.INTERNAL",
    identifier: Some("RunMat:det:Internal"),
    when: "Runtime fails while evaluating determinant or device fallback paths.",
    message: "det: internal runtime failure",
};
const DET_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DET.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:det:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "det: too many output arguments",
};

const DET_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DET_ERROR_INVALID_INPUT,
    DET_ERROR_INTERNAL,
    DET_ERROR_TOO_MANY_OUTPUTS,
];

pub const DET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DET_ERRORS,
};

#[derive(Debug, Clone, Copy)]
enum Determinant {
    Real(f64),
    Complex(f64, f64),
}

impl Determinant {
    fn apply_sign(self, sign: f64) -> Self {
        match self {
            Self::Real(value) => Self::Real(value * sign),
            Self::Complex(re, im) => Self::Complex(re * sign, im * sign),
        }
    }

    fn into_value(self) -> Value {
        match self {
            Self::Real(value) => Value::Num(value),
            Self::Complex(re, im) => Value::Complex(re, im),
        }
    }

    fn into_value_for_prototype(self, prototype: &GpuTensorHandle) -> BuiltinResult<Value> {
        let single = runmat_accelerate_api::handle_integer_type(prototype).is_none()
            && !runmat_accelerate_api::handle_is_logical(prototype)
            && runmat_accelerate_api::handle_precision(prototype)
                == Some(runmat_accelerate_api::ProviderPrecision::F32);
        if !single {
            return Ok(self.into_value());
        }
        match self {
            Self::Real(value) => Tensor::from_f32(vec![value as f32], vec![1, 1])
                .map(Value::Tensor)
                .map_err(builtin_error),
            Self::Complex(re, im) => {
                ComplexTensor::from_f32(vec![(re as f32, im as f32)], vec![1, 1])
                    .map(Value::ComplexTensor)
                    .map_err(builtin_error)
            }
        }
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::det")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("det"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("lu")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Floating inputs use owner-resolved LU when available; typed integer, logical, and complex storage uses an explicit gathered fallback. Restored results preserve input single/double precision and owner/device residency.",
};

fn det_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    det_error_with_message(message, &DET_ERROR_INVALID_INPUT)
}

fn interaction_pending_error() -> RuntimeError {
    build_runtime_error("interaction pending...")
        .with_builtin(NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::det")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Determinant evaluation is a terminal scalar operation and does not participate in fusion plans.",
};

#[runtime_builtin(
    name = "det",
    category = "math/linalg/solve",
    summary = "Compute determinants of square matrices.",
    keywords = "det,determinant,linear algebra,matrix,gpu",
    accel = "det",
    type_resolver(numeric_scalar_type),
    extensions(DET_EXTENSIONS),
    integer_capabilities(INTEGER_CAPABILITIES),
    descriptor(crate::builtins::math::linalg::solve::det::DET_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::solve::det"
)]
async fn det_builtin(value: Value) -> BuiltinResult<Value> {
    crate::builtins::math::trigonometry::inverse_helpers::reject_excess_outputs(NAME)?;
    ensure_extensions(&value)?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(&value, NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, NAME)?;
    match value {
        Value::GpuTensor(handle) => det_gpu(handle).await,
        Value::ComplexTensor(tensor) => det_complex_value(tensor),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            det_real_value(tensor)
        }
    }
}

async fn det_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle)
        .ok_or_else(|| builtin_error(format!("{NAME}: GPU input has no owning provider")))?;
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
        || runmat_accelerate_api::handle_storage(&handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
        let result = determinant_from_value(gathered)?.into_value_for_prototype(&handle)?;
        return crate::builtins::math::trigonometry::inverse_helpers::upload_value_like(
            provider, result, NAME, &handle,
        );
    }
    {
        match det_gpu_via_provider(provider, &handle).await {
            Ok(Some(value)) => return Ok(value),
            Ok(None) => {}
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(interaction_pending_error());
                }
                return Err(err);
            }
        }
    }

    let gathered_value = {
        let proxy = Value::GpuTensor(handle.clone());
        gpu_helpers::gather_value_async(&proxy).await?
    };

    let det_result = determinant_from_value(gathered_value)?;
    match det_result {
        Determinant::Real(det) => {
            {
                match upload_scalar(provider, det, &handle) {
                    Ok(uploaded) => return Ok(Value::GpuTensor(uploaded)),
                    Err(err) => {
                        if err.message() == "interaction pending..." {
                            return Err(interaction_pending_error());
                        }
                    }
                }
            }
            Ok(Value::Num(det))
        }
        Determinant::Complex(re, im) => Ok(Value::Complex(re, im)),
    }
}

fn ensure_extensions(value: &Value) -> BuiltinResult<()> {
    let is_integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if is_integer {
        crate::compatibility::ensure_builtin_extension_enabled(&DET_INTEGER_INPUT_EXTENSION, NAME)?;
    }
    let is_logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if is_logical {
        crate::compatibility::ensure_builtin_extension_enabled(&DET_LOGICAL_INPUT_EXTENSION, NAME)?;
    }
    Ok(())
}

fn det_real_value(tensor: Tensor) -> BuiltinResult<Value> {
    if tensor.numeric_dtype() == NumericDType::F32 {
        let (rows, cols) = matrix_dimensions(tensor.shape.as_slice())?;
        if rows != cols {
            return Err(builtin_error(format!(
                "{NAME}: input must be a square matrix."
            )));
        }
        let values = match tensor.into_numeric_storage().map_err(builtin_error)? {
            NumericStorage::F32(values) => values,
            _ => unreachable!("F32 dtype must have F32 storage"),
        };
        let value = if rows == 0 {
            1.0f32
        } else if values.len() == 1 {
            values[0]
        } else {
            LU::new(DMatrix::from_column_slice(rows, cols, &values)).determinant()
        };
        let output = Tensor::from_f32(vec![value], vec![1, 1]).map_err(builtin_error)?;
        return Ok(Value::Tensor(output));
    }
    Ok(Determinant::Real(det_real_tensor(&tensor)?).into_value())
}

fn det_complex_value(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let (re, im) = det_complex_tensor(&tensor)?;
    if tensor.numeric_dtype() == NumericDType::F32 {
        let output = ComplexTensor::from_f64_values_with_dtype(
            vec![(re, im)],
            vec![1, 1],
            NumericDType::F32,
        )
        .map_err(builtin_error)?;
        return Ok(Value::ComplexTensor(output));
    }
    Ok(Determinant::Complex(re, im).into_value())
}

fn determinant_from_value(value: Value) -> BuiltinResult<Determinant> {
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(&value, NAME)?;
    match value {
        Value::Num(n) => Ok(Determinant::Real(n)),
        Value::Tensor(tensor) => det_real_tensor(&tensor).map(Determinant::Real),
        Value::ComplexTensor(tensor) => {
            det_complex_tensor(&tensor).map(|(re, im)| Determinant::Complex(re, im))
        }
        Value::Complex(re, im) => Ok(Determinant::Complex(re, im)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(builtin_error)?;
            det_real_tensor(&tensor).map(Determinant::Real)
        }
        Value::Int(int_value) => Ok(Determinant::Real(int_value.to_f64())),
        Value::Bool(flag) => Ok(Determinant::Real(if flag { 1.0 } else { 0.0 })),
        other => Err(builtin_error(format!(
            "{NAME}: unsupported input type {:?}; expected numeric or logical values",
            other
        ))),
    }
}

fn det_real_tensor(matrix: &Tensor) -> BuiltinResult<f64> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 && cols == 0 {
        return Ok(1.0);
    }
    let values = tensor::tensor_values_f64_cow(matrix);
    if values.len() == 1 {
        return Ok(values[0]);
    }
    let lu = LU::new(DMatrix::from_column_slice(rows, cols, &values));
    Ok(lu.determinant())
}

fn det_complex_tensor(matrix: &ComplexTensor) -> BuiltinResult<(f64, f64)> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 && cols == 0 {
        return Ok((1.0, 0.0));
    }
    if matrix.materialize_f64().len() == 1 {
        return Ok(matrix.materialize_f64()[0]);
    }
    let data: Vec<Complex64> = matrix
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let lu = LU::new(DMatrix::from_column_slice(rows, cols, &data));
    let det = lu.determinant();
    Ok((det.re, det.im))
}

fn matrix_dimensions(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    match shape {
        [] => Ok((1, 1)),
        [rows] => {
            if *rows == 1 {
                Ok((1, 1))
            } else {
                Err(builtin_error(format!(
                    "{NAME}: input must be a square matrix."
                )))
            }
        }
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        ))),
    }
}

fn upload_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: f64,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<GpuTensorHandle> {
    let data = [value];
    let shape = [1usize, 1usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    let expected_precision = if runmat_accelerate_api::handle_integer_type(prototype).is_some()
        || runmat_accelerate_api::handle_is_logical(prototype)
    {
        runmat_accelerate_api::ProviderPrecision::F64
    } else {
        runmat_accelerate_api::handle_precision(prototype).unwrap_or_else(|| provider.precision())
    };
    if provider.precision() != expected_precision {
        return Err(builtin_error(format!(
            "{NAME}: input provider cannot restore the requested result precision"
        )));
    }
    let handle = provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let valid = handle.shape == shape
        && handle.device_id == prototype.device_id
        && runmat_accelerate_api::handle_storage(&handle)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle)
        && runmat_accelerate_api::handle_precision(&handle) == Some(expected_precision)
        && runmat_accelerate_api::provider_for_handle(&handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider));
    if !valid {
        let owner = runmat_accelerate_api::provider_for_handle(&handle).unwrap_or(provider);
        let _ = owner.free(&handle);
        return Err(builtin_error(format!(
            "{NAME}: provider returned an incompatible scalar result"
        )));
    }
    Ok(handle)
}

async fn det_gpu_via_provider(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    let (rows, cols) = matrix_dimensions(handle.shape.as_slice())?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 {
        let uploaded = upload_scalar(provider, 1.0, handle)?;
        return Ok(Some(Value::GpuTensor(uploaded)));
    }

    let lu_result = match provider.lu(handle).await {
        Ok(result) => result,
        Err(_) => return Ok(None),
    };
    if !validate_lu_result(&lu_result, provider, handle, rows) {
        free_lu_result_unique(&lu_result, provider, handle);
        return Ok(None);
    }

    let outcome = {
        async {
            enum UpperFactor {
                Real(Tensor),
                Complex(ComplexTensor),
            }

            let upper_factor = match gpu_helpers::gather_tensor_async(&lu_result.upper).await {
                Ok(tensor) => UpperFactor::Real(tensor),
                Err(err) => {
                    if err.message() == "interaction pending..." {
                        return Err(interaction_pending_error());
                    }
                    let value = Value::GpuTensor(lu_result.upper.clone());
                    match gpu_helpers::gather_value_async(&value).await {
                        Ok(Value::Tensor(tensor)) => UpperFactor::Real(tensor),
                        Ok(Value::ComplexTensor(tensor)) => UpperFactor::Complex(tensor),
                        Ok(Value::Num(n)) => {
                            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(builtin_error)?;
                            UpperFactor::Real(tensor)
                        }
                        Ok(_) => return Ok(None),
                        Err(err) => {
                            if err.message() == "interaction pending..." {
                                return Err(interaction_pending_error());
                            }
                            return Ok(None);
                        }
                    }
                }
            };

            let pivot_tensor = match gpu_helpers::gather_tensor_async(&lu_result.perm_vector).await
            {
                Ok(tensor) => tensor,
                Err(err) => {
                    if err.message() == "interaction pending..." {
                        return Err(interaction_pending_error());
                    }
                    return Ok(None);
                }
            };

            let determinant = match upper_factor {
                UpperFactor::Real(tensor) => match diagonal_product_real(&tensor, rows) {
                    Ok(value) => Determinant::Real(value),
                    Err(err) => {
                        if err.message() == "interaction pending..." {
                            return Err(interaction_pending_error());
                        }
                        return Ok(None);
                    }
                },
                UpperFactor::Complex(tensor) => match diagonal_product_complex(&tensor, rows) {
                    Ok((re, im)) => Determinant::Complex(re, im),
                    Err(err) => {
                        if err.message() == "interaction pending..." {
                            return Err(interaction_pending_error());
                        }
                        return Ok(None);
                    }
                },
            };

            let permutation_sign = match permutation_sign_from_tensor(&pivot_tensor, rows) {
                Ok(value) => value,
                Err(err) => {
                    if err.message() == "interaction pending..." {
                        return Err(interaction_pending_error());
                    }
                    return Ok(None);
                }
            };

            let determinant = determinant.apply_sign(permutation_sign);

            match determinant {
                Determinant::Real(value) => match upload_scalar(provider, value, handle) {
                    Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
                    Err(err) => {
                        if err.message() == "interaction pending..." {
                            Err(interaction_pending_error())
                        } else {
                            Ok(None)
                        }
                    }
                },
                Determinant::Complex(re, im) => Ok(Some(Value::Complex(re, im))),
            }
        }
        .await
    };

    free_lu_result_unique(&lu_result, provider, handle);

    outcome
}

fn lu_handles(result: &runmat_accelerate_api::ProviderLuResult) -> [&GpuTensorHandle; 5] {
    [
        &result.combined,
        &result.lower,
        &result.upper,
        &result.perm_matrix,
        &result.perm_vector,
    ]
}

fn validate_lu_result(
    result: &runmat_accelerate_api::ProviderLuResult,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    dimension: usize,
) -> bool {
    let expected_precision =
        runmat_accelerate_api::handle_precision(input).unwrap_or_else(|| provider.precision());
    let expected_shapes = [
        vec![dimension, dimension],
        vec![dimension, dimension],
        vec![dimension, dimension],
        vec![dimension, dimension],
        vec![dimension, 1],
    ];
    let mut identities = HashSet::new();
    for (handle, expected_shape) in lu_handles(result).into_iter().zip(expected_shapes.iter()) {
        if handle.buffer_id == input.buffer_id && handle.device_id == input.device_id {
            return false;
        }
        if !identities.insert((handle.device_id, handle.buffer_id)) {
            return false;
        }
        if handle.device_id != input.device_id
            || handle.shape != *expected_shape
            || runmat_accelerate_api::provider_for_handle(handle)
                .is_none_or(|owner| !std::ptr::eq(owner, provider))
            || runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            || runmat_accelerate_api::handle_precision(handle) != Some(expected_precision)
            || runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return false;
        }
    }
    true
}

fn free_lu_result_unique(
    result: &runmat_accelerate_api::ProviderLuResult,
    invoked_provider: &dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
) {
    let mut freed = HashSet::new();
    for handle in lu_handles(result) {
        let identity = (handle.device_id, handle.buffer_id);
        if identity == (input.device_id, input.buffer_id) || !freed.insert(identity) {
            continue;
        }
        let owner = runmat_accelerate_api::provider_for_handle(handle).unwrap_or(invoked_provider);
        let _ = owner.free(handle);
    }
}

fn diagonal_product_real(upper: &Tensor, dimension: usize) -> BuiltinResult<f64> {
    if dimension == 0 {
        return Ok(1.0);
    }
    let rows = upper.rows();
    let cols = upper.cols();
    if rows < dimension || cols < dimension {
        return Err(builtin_error(format!(
            "{NAME}: upper factor shape mismatch"
        )));
    }
    let mut product = 1.0f64;
    for i in 0..dimension {
        let idx = i + i * rows;
        let value = upper
            .numeric_value_at(idx)
            .ok_or_else(|| builtin_error(format!("{NAME}: upper factor diagonal out of range")))?;
        product *= floating_scalar_to_f64(value)?;
    }
    Ok(product)
}

fn diagonal_product_complex(upper: &ComplexTensor, dimension: usize) -> BuiltinResult<(f64, f64)> {
    if dimension == 0 {
        return Ok((1.0, 0.0));
    }
    let rows = upper.rows;
    let cols = upper.cols;
    if rows < dimension || cols < dimension {
        return Err(builtin_error(format!(
            "{NAME}: upper factor shape mismatch"
        )));
    }
    let mut product = Complex64::new(1.0, 0.0);
    for i in 0..dimension {
        let idx = i + i * rows;
        let (re, im) = *upper
            .materialize_f64()
            .get(idx)
            .ok_or_else(|| builtin_error(format!("{NAME}: upper factor diagonal out of range")))?;
        product *= Complex64::new(re, im);
    }
    Ok((product.re, product.im))
}

fn permutation_sign_from_tensor(pivots: &Tensor, expected_len: usize) -> BuiltinResult<f64> {
    if expected_len == 0 {
        return Ok(1.0);
    }
    if pivots.len() != expected_len {
        return Err(builtin_error(format!(
            "{NAME}: pivot vector length mismatch"
        )));
    }
    let len = pivots.len();
    let mut permutation = Vec::with_capacity(len);
    let mut seen = vec![false; len];
    for position in 0..len {
        let raw = pivots
            .numeric_value_at(position)
            .ok_or_else(|| builtin_error(format!("{NAME}: pivot vector length mismatch")))?;
        let one_based = pivot_index(raw, len)?;
        let idx = one_based - 1;
        if seen[idx] {
            return Err(builtin_error(format!(
                "{NAME}: pivot vector must describe a permutation"
            )));
        }
        seen[idx] = true;
        permutation.push(idx);
    }
    Ok(permutation_sign(&permutation))
}

fn floating_scalar_to_f64(value: NumericScalar) -> BuiltinResult<f64> {
    match value {
        NumericScalar::F64(value) => Ok(value),
        NumericScalar::F32(value) => Ok(f64::from(value)),
        _ => Err(builtin_error(format!(
            "{NAME}: provider upper factor must use floating storage"
        ))),
    }
}

fn pivot_index(value: NumericScalar, len: usize) -> BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => floating_pivot_index(value, len),
        NumericScalar::F32(value) => floating_pivot_index(f64::from(value), len),
        integer => {
            let one_based = integer
                .into_int_value()
                .expect("non-floating numeric scalar is integer")
                .try_to_usize()
                .ok_or_else(|| builtin_error(format!("{NAME}: pivot vector index out of range")))?;
            if one_based == 0 || one_based > len {
                return Err(builtin_error(format!(
                    "{NAME}: pivot vector index out of range"
                )));
            }
            Ok(one_based)
        }
    }
}

fn floating_pivot_index(value: f64, len: usize) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(builtin_error(format!(
            "{NAME}: pivot vector contains non-finite entries"
        )));
    }
    let rounded = value.round();
    if (rounded - value).abs() > 1.0e-6 {
        return Err(builtin_error(format!(
            "{NAME}: pivot vector must contain integer values"
        )));
    }
    if rounded < 1.0 || rounded > len as f64 {
        return Err(builtin_error(format!(
            "{NAME}: pivot vector index out of range"
        )));
    }
    Ok(rounded as usize)
}

fn permutation_sign(permutation: &[usize]) -> f64 {
    let mut visited = vec![false; permutation.len()];
    let mut sign = 1.0f64;
    for start in 0..permutation.len() {
        if visited[start] {
            continue;
        }
        let mut length = 0usize;
        let mut current = start;
        while current < permutation.len() && !visited[current] {
            visited[current] = true;
            current = permutation[current];
            length += 1;
        }
        if length > 0 && length.is_multiple_of(2) {
            sign = -sign;
        }
    }
    sign
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntValue, IntegerStorage};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg(feature = "wgpu")]
    fn det_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::det_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_basic_2x2() {
        let tensor = Tensor::new(vec![4.0, 1.0, -2.0, 3.0], vec![2, 2]).unwrap();
        let result = det_real_value(tensor).expect("det");
        match result {
            Value::Num(v) => assert!((v - 14.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn det_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![4, 1, 2, 3]), vec![2, 2])
            .expect("integer");
        let result = det_real_value(tensor).expect("det");
        match result {
            Value::Num(v) => assert!((v - 10.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn det_integer_extension_is_gated_and_wide_values_reject() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::det_builtin(Value::Int(IntValue::I32(1))))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            error.identifier(),
            DET_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(strict);

        let compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = block_on(super::det_builtin(Value::Int(IntValue::U64(
            (1u64 << 53) + 1,
        ))))
        .expect_err("inexact binary64 boundary rejects");
        assert!(error.message().contains("exactly representable as double"));
        drop(compat);
    }

    #[test]
    fn det_preserves_single_result_class() {
        let tensor = Tensor::from_f32(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).expect("single");
        match block_on(super::det_builtin(Value::Tensor(tensor))).expect("det") {
            Value::Tensor(output) => assert_eq!(output.numeric_dtype(), NumericDType::F32),
            other => panic!("expected single tensor result, got {other:?}"),
        }
    }

    #[test]
    fn det_provider_helpers_read_authoritative_storage() {
        let upper =
            Tensor::from_f32(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).expect("single upper factor");
        assert!((diagonal_product_real(&upper, 2).unwrap() - 6.0).abs() < 1.0e-12);

        let pivots =
            Tensor::new_integer(IntegerStorage::U64(vec![2, 1]), vec![2, 1]).expect("pivots");
        assert_eq!(permutation_sign_from_tensor(&pivots, 2).unwrap(), -1.0);
    }

    #[test]
    fn det_type_returns_scalar() {
        let out = numeric_scalar_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn det_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = DET_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"d = det(A)"));
    }

    #[test]
    fn det_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = DET_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.DET.INVALID_INPUT"));
        assert!(codes.contains(&"RM.DET.INTERNAL"));
        assert!(codes.contains(&"RM.DET.TOO_MANY_OUTPUTS"));
    }

    #[test]
    fn det_rejects_aliased_or_mistyped_provider_lu_results_and_frees_safely() {
        use runmat_accelerate_api::ProviderLuResult;

        test_support::with_test_provider(|provider| {
            let upload = |data: &[f64], shape: &[usize]| {
                provider
                    .upload(&HostTensorView { data, shape })
                    .expect("upload")
            };
            let input = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let combined = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let lower = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let upper = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let perm_matrix = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let perm_vector = upload(&[1.0, 2.0], &[2, 1]);
            let valid = ProviderLuResult {
                combined: combined.clone(),
                lower: lower.clone(),
                upper: upper.clone(),
                perm_matrix: perm_matrix.clone(),
                perm_vector: perm_vector.clone(),
            };
            assert!(validate_lu_result(&valid, provider, &input, 2));

            let mut duplicate = valid.clone();
            duplicate.lower = duplicate.combined.clone();
            assert!(!validate_lu_result(&duplicate, provider, &input, 2));

            let mistyped = valid.clone();
            runmat_accelerate_api::set_handle_logical(&mistyped.upper, true);
            assert!(!validate_lu_result(&mistyped, provider, &input, 2));
            runmat_accelerate_api::set_handle_logical(&mistyped.upper, false);

            let aliasing = ProviderLuResult {
                combined: input.clone(),
                lower: combined.clone(),
                upper: combined.clone(),
                perm_matrix: perm_matrix.clone(),
                perm_vector: perm_vector.clone(),
            };
            free_lu_result_unique(&aliasing, provider, &input);
            assert!(block_on(provider.download(&input)).is_ok());
            assert!(block_on(provider.download(&combined)).is_err());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_non_square_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err = unwrap_error(det_real_value(tensor).unwrap_err());
        assert!(err
            .message()
            .contains("det: input must be a square matrix."));
        assert_eq!(err.identifier(), DET_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn det_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(
            vec![3.0, 1.0, 0.0, 0.0, 5.0, 2.0, 4.0, 0.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        let cpu_det = det_real_tensor(&tensor).expect("cpu det");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data = tensor.as_f64_slice().expect("double tensor");
        let view = HostTensorView {
            data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = det_builtin(Value::GpuTensor(handle)).expect("det");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 1]);
        let det_gpu = gathered.numeric_value_at(0).unwrap();
        let det_gpu = floating_scalar_to_f64(det_gpu).unwrap();
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
        };
        assert!(
            (det_gpu - cpu_det).abs() < tol,
            "gpu det {det_gpu} differs from cpu det {cpu_det}"
        );
    }
}
