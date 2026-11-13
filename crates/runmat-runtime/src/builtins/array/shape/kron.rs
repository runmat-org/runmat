//! MATLAB-compatible `kron` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, NumericDType, NumericStorage, SparseTensor,
    Tensor, Value,
};

use crate::builtins::math::elementwise::integer_arithmetic::{
    integer_binary_scalar, IntegerBinaryOp,
};

type AlignedShapes = (Vec<usize>, Vec<usize>, Vec<usize>);
const BUILTIN_NAME: &str = "kron";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::kron")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "kron",
    op_kind: GpuOpKind::Custom("kronecker"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("kron")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes on-device when the provider implements `kron`; a certified pre-effect unsupported result may gather and restore, while operational provider failures are terminal.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::kron")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "kron",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Kronecker products allocate a fresh dense or CSC sparse result and terminate fusion graphs.",
};

fn kron_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    kron_error_with_message(error.message, error)
}

fn kron_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone)]
enum KronNumericResult {
    Real(Tensor),
    Complex(ComplexTensor),
}

#[derive(Clone)]
enum KronInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

#[runtime_builtin(
    name = "kron",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::kron"
)]
async fn kron_builtin(
    a: Value,
    b: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &KRON_DESCRIPTOR,
        &KRON_ERROR_UNSUPPORTED_INPUT,
        BUILTIN_NAME,
    )?;
    if !rest.is_empty() {
        return Err(kron_error(&KRON_ERROR_TOO_MANY_INPUTS));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&a, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, BUILTIN_NAME)?;
    ensure_kron_shape_policy(&a)?;
    ensure_kron_shape_policy(&b)?;
    if (is_integer_value(&a) && contains_complex(&b))
        || (contains_complex(&a) && is_integer_value(&b))
    {
        return Err(kron_error_with_message(
            "kron: complex integer arithmetic is not supported",
            &KRON_ERROR_UNSUPPORTED_INPUT,
        ));
    }
    validate_kron_integer_admission(&a, &b)?;
    if (matches!(a, Value::SparseTensor(_)) || matches!(b, Value::SparseTensor(_)))
        && [&a, &b].into_iter().any(|value| {
            matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
        })
    {
        return Err(kron_error_with_message(
            "kron: sparse output cannot preserve explicit gpuArray residency",
            &KRON_ERROR_UNSUPPORTED_INPUT,
        ));
    }
    let source = preferred_kron_source(&a, &b);
    if let (Value::GpuTensor(left), Value::GpuTensor(right)) = (&a, &b) {
        if let Some(mut result) = try_kron_gpu_pair(left, right).await? {
            if let Err(error) = set_kron_output_provenance(&mut result, left, Some(right)) {
                return Err(gpu_helpers::reject_gpu_output(result, error));
            }
            return gpu_helpers::resident_gpu_value(result);
        }
    }
    match (&a, &b) {
        (Value::GpuTensor(source), host) => {
            if let Some(mut result) = try_kron_gpu_host(source, host, true).await? {
                if let Err(error) = set_kron_output_provenance(&mut result, source, None) {
                    return Err(gpu_helpers::reject_gpu_output(result, error));
                }
                return gpu_helpers::resident_gpu_value(result);
            }
        }
        (host, Value::GpuTensor(source)) => {
            if let Some(mut result) = try_kron_gpu_host(source, host, false).await? {
                if let Err(error) = set_kron_output_provenance(&mut result, source, None) {
                    return Err(gpu_helpers::reject_gpu_output(result, error));
                }
                return gpu_helpers::resident_gpu_value(result);
            }
        }
        _ => {}
    }
    let a = gather_kron_operand(a).await?;
    let b = gather_kron_operand(b).await?;
    let host = kron_host(a, b)?;
    match source {
        Some(source) => restore_kron_residency(&source, host),
        None => Ok(host),
    }
}

fn preferred_kron_source(left: &Value, right: &Value) -> Option<GpuTensorHandle> {
    let handles = [left, right].map(|value| match value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    });
    handles
        .iter()
        .flatten()
        .find(|handle| runmat_accelerate_api::handle_is_explicit(handle))
        .or_else(|| handles.iter().flatten().next())
        .map(|handle| (*handle).clone())
}

fn set_kron_output_provenance(
    output: &mut GpuTensorHandle,
    source: &GpuTensorHandle,
    other: Option<&GpuTensorHandle>,
) -> crate::BuiltinResult<()> {
    gpu_helpers::propagate_output_provenance(output, std::iter::once(source).chain(other))
}

async fn try_kron_gpu_pair(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> crate::BuiltinResult<Option<GpuTensorHandle>> {
    if runmat_accelerate_api::handle_integer_type(left).is_some()
        || runmat_accelerate_api::handle_integer_type(right).is_some()
        || runmat_accelerate_api::handle_is_logical(left)
        || runmat_accelerate_api::handle_is_logical(right)
        || runmat_accelerate_api::handle_storage(left)
            != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_storage(right)
            != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Ok(None);
    }
    gpu_helpers::require_resident_compute_owners(&[left, right], false, || {
        kron_error_with_message(
            "kron: no exact provider owns the resident operands",
            &KRON_ERROR_INTERNAL,
        )
    })?;
    let Some(owner) = gpu_helpers::exact_provider_for_handle(left) else {
        return Ok(None);
    };
    let Some(right_owner) = gpu_helpers::exact_provider_for_handle(right) else {
        return Ok(None);
    };
    if !runmat_accelerate_api::same_provider_instance(owner, right_owner)
        || left.device_id() != right.device_id()
    {
        return Ok(None);
    }
    let expected_shape = aligned_shapes(left.shape(), right.shape())?.2;
    let Some(expected_precision) = runmat_accelerate_api::handle_precision(left) else {
        return Ok(None);
    };
    if Some(expected_precision) != runmat_accelerate_api::handle_precision(right) {
        return Ok(None);
    }
    let output = match owner.kron(left, right).await {
        Ok(output) => output,
        Err(error) if error.is_unsupported() => return Ok(None),
        Err(error) => {
            return Err(crate::runtime_error::with_provider_execution_source(
                kron_error_with_message("kron: provider operation failed", &KRON_ERROR_INTERNAL),
                error,
            ));
        }
    };
    let valid = output.allocation_lease().validate_live().is_ok()
        && output.shape() == expected_shape
        && output.device_id() == left.device_id()
        && gpu_helpers::exact_provider_for_handle(&output).is_some_and(|output_owner| {
            runmat_accelerate_api::same_provider_instance(output_owner, owner)
        })
        && !output.shares_allocation_with(left)
        && !output.shares_allocation_with(right)
        && runmat_accelerate_api::handle_storage(&output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&output) == Some(expected_precision)
        && runmat_accelerate_api::handle_integer_type(&output).is_none()
        && !runmat_accelerate_api::handle_is_logical(&output);
    if !valid {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            kron_error_with_message(
                "kron: provider returned malformed or aliased output",
                &KRON_ERROR_INTERNAL,
            ),
        ));
    }
    Ok(Some(output))
}

async fn try_kron_gpu_host(
    source: &GpuTensorHandle,
    host: &Value,
    source_is_left: bool,
) -> crate::BuiltinResult<Option<GpuTensorHandle>> {
    if runmat_accelerate_api::handle_integer_type(source).is_some()
        || runmat_accelerate_api::handle_is_logical(source)
        || runmat_accelerate_api::handle_storage(source)
            != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Ok(None);
    }
    let expected_precision = runmat_accelerate_api::handle_precision(source);
    let tensor = match host {
        Value::Num(value)
            if expected_precision == Some(runmat_accelerate_api::ProviderPrecision::F64) =>
        {
            Tensor::new(vec![*value], vec![1, 1])
                .map_err(|error| kron_error_with_message(error, &KRON_ERROR_INTERNAL))?
        }
        Value::Tensor(tensor)
            if tensor.integer_storage().is_none()
                && matches!(
                    (tensor.numeric_dtype(), expected_precision),
                    (
                        NumericDType::F32,
                        Some(runmat_accelerate_api::ProviderPrecision::F32)
                    ) | (
                        NumericDType::F64,
                        Some(runmat_accelerate_api::ProviderPrecision::F64)
                    )
                ) =>
        {
            tensor.clone()
        }
        _ => return Ok(None),
    };
    gpu_helpers::require_resident_compute_owner(source, || {
        kron_error_with_message(
            "kron: no exact provider owns the resident operand",
            &KRON_ERROR_INTERNAL,
        )
    })?;
    let Some(owner) = gpu_helpers::exact_provider_for_handle(source) else {
        return Ok(None);
    };
    let expected_shape = if source_is_left {
        aligned_shapes(source.shape(), tensor.shape())?.2
    } else {
        aligned_shapes(tensor.shape(), source.shape())?.2
    };
    let uploaded =
        gpu_helpers::upload_tensor_protecting(owner, &tensor, &[source]).map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?;
    let uploaded_valid = uploaded.allocation_lease().validate_live().is_ok()
        && uploaded.device_id() == source.device_id()
        && gpu_helpers::exact_provider_for_handle(&uploaded).is_some_and(|uploaded_owner| {
            runmat_accelerate_api::same_provider_instance(uploaded_owner, owner)
        })
        && runmat_accelerate_api::handle_storage(&uploaded)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&uploaded) == expected_precision
        && runmat_accelerate_api::handle_integer_type(&uploaded).is_none()
        && !runmat_accelerate_api::handle_is_logical(&uploaded)
        && !uploaded.shares_allocation_with(source);
    if !uploaded_valid {
        return Err(gpu_helpers::reject_gpu_output(
            uploaded,
            kron_error_with_message(
                "kron: provider returned malformed staged input",
                &KRON_ERROR_INTERNAL,
            ),
        ));
    }
    let output = if source_is_left {
        owner.kron(source, &uploaded).await
    } else {
        owner.kron(&uploaded, source).await
    };
    let result = match output {
        Ok(output) => Ok(Some(output)),
        Err(error) if error.is_unsupported() => Ok(None),
        Err(error) => Err(crate::runtime_error::with_provider_execution_source(
            kron_error_with_message("kron: provider operation failed", &KRON_ERROR_INTERNAL),
            error,
        )),
    };
    let result = result.and_then(|candidate| {
        let Some(output) = candidate else {
            return Ok(None);
        };
        let valid = output.allocation_lease().validate_live().is_ok()
            && output.shape() == expected_shape
            && output.device_id() == source.device_id()
            && gpu_helpers::exact_provider_for_handle(&output).is_some_and(|output_owner| {
                runmat_accelerate_api::same_provider_instance(output_owner, owner)
            })
            && !output.shares_allocation_with(source)
            && !output.shares_allocation_with(&uploaded)
            && runmat_accelerate_api::handle_storage(&output)
                == runmat_accelerate_api::GpuTensorStorage::Real
            && runmat_accelerate_api::handle_precision(&output) == expected_precision
            && runmat_accelerate_api::handle_integer_type(&output).is_none()
            && !runmat_accelerate_api::handle_is_logical(&output);
        if !valid {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                kron_error_with_message(
                    "kron: provider returned malformed or aliased output",
                    &KRON_ERROR_INTERNAL,
                ),
            ));
        }
        Ok(Some(output))
    });
    crate::runtime_error::complete_with_cleanup(result, release_kron_staged(uploaded))
}

fn release_kron_staged(uploaded: GpuTensorHandle) -> crate::BuiltinResult<()> {
    uploaded.release().map(|_| ()).map_err(|error| {
        let mut builder =
            build_runtime_error(format!("kron: staged operand retirement failed: {error}"))
                .with_builtin(BUILTIN_NAME)
                .with_source(error);
        if let Some(identifier) = KRON_ERROR_INTERNAL.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    })
}

fn value_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::ComplexTensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(&array.shape),
        Value::GpuTensor(handle) => Some(handle.shape()),
        _ => None,
    }
}

fn ensure_kron_shape_policy(value: &Value) -> crate::BuiltinResult<()> {
    if value_shape(value).is_some_and(|shape| shape.len() > 2) {
        crate::compatibility::ensure_builtin_extension_enabled(&KRON_ND_EXTENSION, BUILTIN_NAME)?;
    }
    Ok(())
}

fn is_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn integer_value_class(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Int(value) => Some(value.numeric_dtype()),
        Value::Tensor(tensor) => tensor.integer_storage().map(IntegerStorage::numeric_dtype),
        Value::SparseTensor(tensor) => tensor.integer_storage().map(IntegerStorage::numeric_dtype),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).map(|ty| match ty {
                runmat_accelerate_api::IntegerElementType::I8 => NumericDType::I8,
                runmat_accelerate_api::IntegerElementType::I16 => NumericDType::I16,
                runmat_accelerate_api::IntegerElementType::I32 => NumericDType::I32,
                runmat_accelerate_api::IntegerElementType::I64 => NumericDType::I64,
                runmat_accelerate_api::IntegerElementType::U8 => NumericDType::U8,
                runmat_accelerate_api::IntegerElementType::U16 => NumericDType::U16,
                runmat_accelerate_api::IntegerElementType::U32 => NumericDType::U32,
                runmat_accelerate_api::IntegerElementType::U64 => NumericDType::U64,
            })
        }
        _ => None,
    }
}

fn is_kron_scalar_double(value: &Value) -> bool {
    match value {
        Value::Num(_) => true,
        Value::Tensor(tensor) => {
            tensor::is_scalar_tensor(tensor) && tensor.numeric_dtype() == NumericDType::F64
        }
        Value::SparseTensor(tensor) => {
            tensor.rows == 1
                && tensor.cols == 1
                && tensor.numeric_dtype() == Some(NumericDType::F64)
        }
        Value::GpuTensor(handle) => {
            handle.shape().iter().product::<usize>() == 1
                && runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F64)
        }
        _ => false,
    }
}

fn validate_kron_integer_admission(left: &Value, right: &Value) -> crate::BuiltinResult<()> {
    match (integer_value_class(left), integer_value_class(right)) {
        (None, None) => Ok(()),
        (Some(left), Some(right)) if left == right => Ok(()),
        (Some(_), Some(_)) => Err(kron_error_with_message(
            "kron: integer operands must have the same integer class",
            &KRON_ERROR_UNSUPPORTED_INPUT,
        )),
        (Some(_), None) if is_kron_scalar_double(right) => Ok(()),
        (None, Some(_)) if is_kron_scalar_double(left) => Ok(()),
        _ => Err(kron_error_with_message(
            "kron: integer arrays can only be combined with scalar double or logical values",
            &KRON_ERROR_UNSUPPORTED_INPUT,
        )),
    }
}

async fn gather_kron_operand(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(&value).await,
        _ => Ok(value),
    }
}

fn restore_kron_residency(
    source: &GpuTensorHandle,
    mut host: Value,
) -> crate::BuiltinResult<Value> {
    if matches!(host, Value::Num(_) | Value::Int(_) | Value::Bool(_)) {
        host = Value::Tensor(
            tensor::value_into_tensor_for(BUILTIN_NAME, host)
                .map_err(|error| kron_error_with_message(error, &KRON_ERROR_INTERNAL))?,
        );
    }
    let restored = gpu_helpers::restore_class_preserving_value(source, host, BUILTIN_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(source) && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(kron_error_with_message(
            "kron: provider cannot preserve explicit gpuArray output",
            &KRON_ERROR_INTERNAL,
        ));
    }
    Ok(restored)
}

fn kron_host(left: Value, right: Value) -> crate::BuiltinResult<Value> {
    if matches!(left, Value::SparseTensor(_)) || matches!(right, Value::SparseTensor(_)) {
        return sparse::kron(&left, &right);
    }
    if !contains_complex(&left) && !contains_complex(&right) {
        if let Some(result) = try_integer_kron(&left, &right)? {
            return Ok(result);
        }
    }
    let numeric = compute_numeric(left, right)?;
    finalize_numeric(numeric, None)
}

fn contains_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(value, Value::SparseTensor(tensor) if tensor.is_complex())
}

mod sparse;

enum KronIntegerOperand<'a> {
    Scalar(&'a IntValue),
    Array(&'a IntegerStorage, &'a [usize]),
}

impl KronIntegerOperand<'_> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Scalar(_) => &[1, 1],
            Self::Array(_, shape) => shape,
        }
    }

    fn numeric_dtype(&self) -> NumericDType {
        match self {
            Self::Scalar(value) => value.numeric_dtype(),
            Self::Array(storage, _) => storage.numeric_dtype(),
        }
    }

    fn zeros_like(&self, len: usize) -> IntegerStorage {
        match self {
            Self::Scalar(value) => IntegerStorage::from_scalar((*value).clone()).zeros_like(len),
            Self::Array(storage, _) => storage.zeros_like(len),
        }
    }

    fn value_at(&self, index: usize) -> IntValue {
        match self {
            Self::Scalar(value) => (*value).clone(),
            Self::Array(storage, _) => storage
                .value_at(index)
                .expect("kron integer storage index must be valid"),
        }
    }
}

fn integer_kron_operand(value: &Value) -> Option<KronIntegerOperand<'_>> {
    match value {
        Value::Int(value) => Some(KronIntegerOperand::Scalar(value)),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| KronIntegerOperand::Array(storage, tensor.shape())),
        _ => None,
    }
}

fn integer_kron_scalar(value: &Value) -> Option<Value> {
    match value {
        Value::Num(_) | Value::Bool(_) => Some(value.clone()),
        Value::Tensor(tensor)
            if tensor.len() == 1
                && matches!(
                    tensor.numeric_dtype(),
                    NumericDType::F64 | NumericDType::F32
                ) =>
        {
            Some(value.clone())
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Some(value.clone()),
        _ => None,
    }
}

fn try_integer_kron(left: &Value, right: &Value) -> crate::BuiltinResult<Option<Value>> {
    let left_integer = integer_kron_operand(left);
    let right_integer = integer_kron_operand(right);
    if left_integer.is_none() && right_integer.is_none() {
        return Ok(None);
    }

    let left_scalar = left_integer
        .is_none()
        .then(|| integer_kron_scalar(left))
        .flatten();
    let right_scalar = right_integer
        .is_none()
        .then(|| integer_kron_scalar(right))
        .flatten();
    if left_integer.is_none() && left_scalar.is_none()
        || right_integer.is_none() && right_scalar.is_none()
    {
        return Err(kron_error_with_message(
            "kron: integer arrays can only be combined with scalar double or logical values",
            &KRON_ERROR_UNSUPPORTED_INPUT,
        ));
    }

    let left_shape = left_integer
        .as_ref()
        .map(KronIntegerOperand::shape)
        .unwrap_or(&[1, 1]);
    let right_shape = right_integer
        .as_ref()
        .map(KronIntegerOperand::shape)
        .unwrap_or(&[1, 1]);
    let (_, shape_b, shape_out) = aligned_shapes(left_shape, right_shape)?;
    let total_out = checked_total(&shape_out, "kron")?;
    let prototype = left_integer
        .as_ref()
        .or(right_integer.as_ref())
        .expect("integer presence was checked");

    if let (Some(left_integer), Some(right_integer)) = (&left_integer, &right_integer) {
        if left_integer.numeric_dtype() != right_integer.numeric_dtype() {
            return Err(kron_error_with_message(
                "kron: integer operands must have the same integer class",
                &KRON_ERROR_UNSUPPORTED_INPUT,
            ));
        }
    }
    if total_out == 0 {
        let tensor = Tensor::new_integer(prototype.zeros_like(0), shape_out)
            .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))?;
        return Ok(Some(tensor::tensor_into_value(tensor)));
    }

    let strides_out = column_major_strides(&shape_out);
    let mut coords_a = vec![0usize; shape_out.len()];
    let mut coords_b = vec![0usize; shape_out.len()];
    let left_len = left_integer.as_ref().map_or(1, |operand| match operand {
        KronIntegerOperand::Scalar(_) => 1,
        KronIntegerOperand::Array(storage, _) => storage.len(),
    });
    let right_len = right_integer.as_ref().map_or(1, |operand| match operand {
        KronIntegerOperand::Scalar(_) => 1,
        KronIntegerOperand::Array(storage, _) => storage.len(),
    });
    let mut output = prototype.zeros_like(total_out);

    for left_index in 0..left_len {
        unravel_index(left_index, left_shape, &mut coords_a);
        let left_value = left_integer
            .as_ref()
            .map(|operand| Value::Int(operand.value_at(left_index)))
            .or_else(|| left_scalar.clone())
            .expect("integer or scalar left operand was validated");
        for right_index in 0..right_len {
            unravel_index(right_index, right_shape, &mut coords_b);
            let right_value = right_integer
                .as_ref()
                .map(|operand| Value::Int(operand.value_at(right_index)))
                .or_else(|| right_scalar.clone())
                .expect("integer or scalar right operand was validated");
            let value = integer_binary_scalar(
                &left_value,
                &right_value,
                IntegerBinaryOp::Multiply,
                BUILTIN_NAME,
            )
            .map_err(|e| kron_error_with_message(e, &KRON_ERROR_UNSUPPORTED_INPUT))?;
            let out_index = combine_indices(&coords_a, &coords_b, &shape_b, &strides_out)?;
            output
                .set_value(out_index, value)
                .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))?;
        }
    }

    let tensor = Tensor::new_integer(output, shape_out)
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))?;
    Ok(Some(tensor::tensor_into_value(tensor)))
}

fn compute_numeric(left: Value, right: Value) -> crate::BuiltinResult<KronNumericResult> {
    let left_input = value_into_kron_input(left)?;
    let right_input = value_into_kron_input(right)?;
    compute_numeric_inputs(left_input, right_input)
}

fn compute_numeric_inputs(
    left: KronInput,
    right: KronInput,
) -> crate::BuiltinResult<KronNumericResult> {
    match (left, right) {
        (KronInput::Real(a), KronInput::Real(b)) => {
            let tensor = kron_tensor(&a, &b)?;
            Ok(KronNumericResult::Real(tensor))
        }
        (KronInput::Complex(a), KronInput::Complex(b)) => {
            let tensor = kron_complex_tensor(&a, &b)?;
            Ok(KronNumericResult::Complex(tensor))
        }
        (KronInput::Real(a), KronInput::Complex(b)) => {
            let complex_a = tensor_to_complex(&a)?;
            let tensor = kron_complex_tensor(&complex_a, &b)?;
            Ok(KronNumericResult::Complex(tensor))
        }
        (KronInput::Complex(a), KronInput::Real(b)) => {
            let complex_b = tensor_to_complex(&b)?;
            let tensor = kron_complex_tensor(&a, &complex_b)?;
            Ok(KronNumericResult::Complex(tensor))
        }
    }
}

fn finalize_numeric(
    numeric: KronNumericResult,
    provider: Option<&dyn runmat_accelerate_api::AccelProvider>,
) -> crate::BuiltinResult<Value> {
    match numeric {
        KronNumericResult::Real(tensor) => {
            if let Some(provider) = provider {
                if let Ok(handle) = gpu_helpers::upload_tensor(provider, &tensor) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
            Ok(tensor::tensor_into_value(tensor))
        }
        KronNumericResult::Complex(tensor) => Ok(complex_tensor_into_value(tensor)),
    }
}

fn value_into_kron_input(value: Value) -> crate::BuiltinResult<KronInput> {
    match value {
        Value::Tensor(tensor) => Ok(KronInput::Real(tensor)),
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map(KronInput::Real)
            .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL)),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            tensor::value_into_tensor_for("kron", value)
                .map(KronInput::Real)
                .map_err(|e| kron_error_with_message(e, &KRON_ERROR_INTERNAL))
        }
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map(KronInput::Complex)
            .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL)),
        Value::ComplexTensor(tensor) => Ok(KronInput::Complex(tensor)),
        Value::CharArray(chars) => char_array_to_tensor(&chars).map(KronInput::Real),
        other => Err(kron_error_with_message(
            format!(
                "kron: unsupported input type {:?}; expected numeric, logical, or complex values",
                other
            ),
            &KRON_ERROR_UNSUPPORTED_INPUT,
        )),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> crate::BuiltinResult<Tensor> {
    let data: Vec<f64> = chars
        .iter_column_major()
        .map(|ch| f64::from(ch.raw()))
        .collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))
}

fn tensor_to_complex(tensor: &Tensor) -> crate::BuiltinResult<ComplexTensor> {
    if let Some(values) = tensor.as_f32_slice() {
        return ComplexTensor::from_f32(
            values.iter().map(|&re| (re, 0.0)).collect(),
            tensor.shape().to_vec(),
        )
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL));
    }
    let values = tensor::tensor_values_f64_cow(tensor);
    let data: Vec<(f64, f64)> = values.iter().map(|&re| (re, 0.0)).collect();
    ComplexTensor::new(data, tensor.shape().to_vec())
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))
}

fn kron_uses_single(left: &Value, right: &Value) -> bool {
    [left, right].into_iter().any(|value| match value {
        Value::SparseTensor(tensor) => tensor.numeric_dtype() == Some(NumericDType::F32),
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        Value::ComplexTensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        _ => false,
    })
}

fn kron_complex_uses_single(left: &ComplexTensor, right: &ComplexTensor) -> bool {
    left.numeric_dtype() == NumericDType::F32 || right.numeric_dtype() == NumericDType::F32
}

fn kron_complex_product(a: (f64, f64), b: (f64, f64), single: bool) -> (f64, f64) {
    if single {
        let (ar, ai, br, bi) = (a.0 as f32, a.1 as f32, b.0 as f32, b.1 as f32);
        ((ar * br - ai * bi) as f64, (ar * bi + ai * br) as f64)
    } else {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }
}

fn kron_tensor(a: &Tensor, b: &Tensor) -> crate::BuiltinResult<Tensor> {
    let (shape_a, shape_b, shape_out) = aligned_shapes(a.shape(), b.shape())?;
    let storage_a = a
        .clone()
        .into_numeric_storage()
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))?;
    let storage_b = b
        .clone()
        .into_numeric_storage()
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))?;
    let storage = match (storage_a, storage_b) {
        (NumericStorage::F64(a), NumericStorage::F64(b)) => {
            NumericStorage::F64(kron_real_values(&a, &shape_a, &b, &shape_b, &shape_out)?)
        }
        (NumericStorage::F32(a), NumericStorage::F32(b)) => {
            NumericStorage::F32(kron_real_values(&a, &shape_a, &b, &shape_b, &shape_out)?)
        }
        (NumericStorage::F32(a), NumericStorage::F64(b)) => {
            let b = b.into_iter().map(|value| value as f32).collect::<Vec<_>>();
            NumericStorage::F32(kron_real_values(&a, &shape_a, &b, &shape_b, &shape_out)?)
        }
        (NumericStorage::F64(a), NumericStorage::F32(b)) => {
            let a = a.into_iter().map(|value| value as f32).collect::<Vec<_>>();
            NumericStorage::F32(kron_real_values(&a, &shape_a, &b, &shape_b, &shape_out)?)
        }
        _ => {
            return Err(kron_error_with_message(
                "kron: integer storage reached the floating evaluation path",
                &KRON_ERROR_INTERNAL,
            ))
        }
    };
    Tensor::from_numeric_storage(storage, shape_out)
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))
}

fn kron_real_values<T>(
    a: &[T],
    shape_a: &[usize],
    b: &[T],
    shape_b: &[usize],
    shape_out: &[usize],
) -> crate::BuiltinResult<Vec<T>>
where
    T: Copy + Default + std::ops::Mul<Output = T>,
{
    let total_out = checked_total(shape_out, "kron")?;
    let strides_out = column_major_strides(shape_out);
    let mut coords_a = vec![0usize; shape_out.len()];
    let mut coords_b = vec![0usize; shape_out.len()];
    let mut data = vec![T::default(); total_out];

    for (idx_a, &value_a) in a.iter().enumerate() {
        unravel_index(idx_a, shape_a, &mut coords_a);
        for (idx_b, &value_b) in b.iter().enumerate() {
            unravel_index(idx_b, shape_b, &mut coords_b);
            let out_index = combine_indices(&coords_a, &coords_b, shape_b, &strides_out)?;
            data[out_index] = value_a * value_b;
        }
    }
    Ok(data)
}

fn kron_complex_tensor(
    a: &ComplexTensor,
    b: &ComplexTensor,
) -> crate::BuiltinResult<ComplexTensor> {
    let (shape_a, shape_b, shape_out) = aligned_shapes(a.shape(), b.shape())?;
    let total_out = checked_total(&shape_out, "kron")?;
    let single = kron_complex_uses_single(a, b);
    if total_out == 0 {
        return ComplexTensor::from_f64_values_with_dtype(
            Vec::new(),
            shape_out,
            if single {
                NumericDType::F32
            } else {
                NumericDType::F64
            },
        )
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL));
    }

    let strides_out = column_major_strides(&shape_out);
    let mut coords_a = vec![0usize; shape_out.len()];
    let mut coords_b = vec![0usize; shape_out.len()];
    let mut data = vec![(0.0f64, 0.0f64); total_out];

    let left_values = a.materialize_f64();
    let right_values = b.materialize_f64();
    for (idx_a, &(ar, ai)) in left_values.iter().enumerate() {
        unravel_index(idx_a, &shape_a, &mut coords_a);
        for (idx_b, &(br, bi)) in right_values.iter().enumerate() {
            unravel_index(idx_b, &shape_b, &mut coords_b);
            let out_index = combine_indices(&coords_a, &coords_b, &shape_b, &strides_out)?;
            data[out_index] = kron_complex_product((ar, ai), (br, bi), single);
        }
    }

    ComplexTensor::from_f64_values_with_dtype(
        data,
        shape_out,
        if single {
            NumericDType::F32
        } else {
            NumericDType::F64
        },
    )
    .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))
}

fn aligned_shapes(shape_a: &[usize], shape_b: &[usize]) -> crate::BuiltinResult<AlignedShapes> {
    let rank = shape_a.len().max(shape_b.len()).max(1);
    let mut padded_a = vec![1usize; rank];
    let mut padded_b = vec![1usize; rank];

    for (idx, &dim) in shape_a.iter().enumerate() {
        padded_a[idx] = dim;
    }
    for (idx, &dim) in shape_b.iter().enumerate() {
        padded_b[idx] = dim;
    }

    let mut output = Vec::with_capacity(rank);
    for i in 0..rank {
        output.push(padded_a[i].checked_mul(padded_b[i]).ok_or_else(|| {
            kron_error_with_message(
                "kron: requested output exceeds maximum size",
                &KRON_ERROR_INTERNAL,
            )
        })?);
    }

    Ok((padded_a, padded_b, output))
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut current = 1usize;
    for &dim in shape {
        strides.push(current);
        current = current.saturating_mul(dim.max(1));
    }
    strides
}

fn unravel_index(mut index: usize, shape: &[usize], coords: &mut [usize]) {
    for (dim_idx, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            coords[dim_idx] = 0;
        } else {
            coords[dim_idx] = index % dim;
            index /= dim;
        }
    }
}

fn combine_indices(
    coords_a: &[usize],
    coords_b: &[usize],
    shape_b: &[usize],
    strides_out: &[usize],
) -> crate::BuiltinResult<usize> {
    let mut index = 0usize;
    for (dim, stride) in strides_out.iter().enumerate() {
        let scaled = coords_a
            .get(dim)
            .copied()
            .unwrap_or(0)
            .checked_mul(shape_b.get(dim).copied().unwrap_or(1))
            .ok_or_else(|| kron_error_with_message("kron: index overflow", &KRON_ERROR_INTERNAL))?;
        let coord = scaled
            .checked_add(coords_b.get(dim).copied().unwrap_or(0))
            .ok_or_else(|| kron_error_with_message("kron: index overflow", &KRON_ERROR_INTERNAL))?;
        index = index
            .checked_add(coord.checked_mul(*stride).ok_or_else(|| {
                kron_error_with_message("kron: index overflow", &KRON_ERROR_INTERNAL)
            })?)
            .ok_or_else(|| kron_error_with_message("kron: index overflow", &KRON_ERROR_INTERNAL))?;
    }
    Ok(index)
}

fn checked_total(shape: &[usize], context: &str) -> crate::BuiltinResult<usize> {
    let mut total = 1usize;
    for &dim in shape {
        if dim == 0 {
            return Ok(0);
        }
        total = total.checked_mul(dim).ok_or_else(|| {
            kron_error_with_message(
                format!("{context}: requested output exceeds maximum size"),
                &KRON_ERROR_INTERNAL,
            )
        })?;
    }
    Ok(total)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn kron_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::kron_builtin(
            a,
            b,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntegerStorage, LogicalArray, Tensor};

    #[test]
    fn kron_sparse_csc_products_preserve_structure_and_numeric_class() {
        let left = SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]).unwrap();
        let right = Tensor::new(vec![0.0, 5.0], vec![2, 1]).unwrap();
        let Value::SparseTensor(output) =
            kron_builtin(Value::SparseTensor(left), Value::Tensor(right), Vec::new())
                .expect("sparse kron")
        else {
            panic!("expected CSC result")
        };
        assert_eq!((output.rows, output.cols), (4, 2));
        assert_eq!(&*output.col_ptrs, &[0, 1, 2]);
        assert_eq!(&*output.row_indices, &[1, 3]);
        assert_eq!(output.as_f32_slice(), Some(&[10.0, 15.0][..]));
    }

    #[test]
    fn kron_sparse_integer_complex_and_logical_use_existing_classes() {
        let integer = SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![0],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .unwrap();
        let Value::SparseTensor(scaled) =
            kron_builtin(Value::SparseTensor(integer), Value::Num(2.0), Vec::new())
                .expect("integer sparse kron")
        else {
            panic!("expected sparse integer")
        };
        assert_eq!(
            scaled.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX]))
        );

        let complex =
            SparseTensor::new_complex_f32(1, 2, vec![0, 1, 1], vec![0], vec![(1.0, 2.0)]).unwrap();
        let Value::SparseTensor(product) =
            kron_builtin(Value::SparseTensor(complex), Value::Num(2.0), Vec::new())
                .expect("complex sparse kron")
        else {
            panic!("expected sparse complex")
        };
        assert_eq!(
            product.as_complex_f32_slice().unwrap()[0],
            runmat_value::ComplexElement(2.0, 4.0)
        );

        let logical = SparseTensor::new_logical(1, 2, vec![0, 1, 1], vec![0]).unwrap();
        let Value::SparseTensor(product) =
            kron_builtin(Value::SparseTensor(logical), Value::Num(3.0), Vec::new())
                .expect("logical sparse kron")
        else {
            panic!("expected sparse double")
        };
        assert_eq!(product.as_f64_slice(), Some(&[3.0][..]));
    }

    #[test]
    fn kron_integer_scalar_and_sparse_double_scalar_preserve_exact_class_both_orders() {
        let sparse = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![2.0]).unwrap();
        for (left, right) in [
            (
                Value::Int(IntValue::U64(u64::MAX)),
                Value::SparseTensor(sparse.clone()),
            ),
            (
                Value::SparseTensor(sparse.clone()),
                Value::Int(IntValue::U64(u64::MAX)),
            ),
        ] {
            let Value::SparseTensor(output) =
                kron_builtin(left, right, Vec::new()).expect("integer scalar sparse kron")
            else {
                panic!("expected sparse integer")
            };
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX]))
            );
        }
    }

    #[test]
    fn kron_elides_underflowed_products_from_csc_nnz() {
        let sparse =
            SparseTensor::new_f32(2, 1, vec![0, 1], vec![0], vec![f32::MIN_POSITIVE]).unwrap();
        let Value::SparseTensor(output) = kron_builtin(
            Value::SparseTensor(sparse),
            Value::Tensor(Tensor::from_f32(vec![f32::MIN_POSITIVE], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .expect("sparse underflow") else {
            panic!("expected CSC")
        };
        assert_eq!(output.nnz(), 0);
        assert_eq!(&*output.col_ptrs, &[0, 0]);
    }

    #[test]
    fn kron_logical_pairs_agree_on_double_class_dense_and_sparse() {
        let dense = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        let Value::Tensor(full) = kron_builtin(
            Value::LogicalArray(dense.clone()),
            Value::LogicalArray(dense.clone()),
            Vec::new(),
        )
        .expect("dense logical product") else {
            panic!("expected numeric tensor")
        };
        assert_eq!(full.numeric_dtype(), NumericDType::F64);
        let sparse = SparseTensor::new_logical(2, 1, vec![0, 1], vec![0]).unwrap();
        let Value::SparseTensor(csc) = kron_builtin(
            Value::SparseTensor(sparse),
            Value::LogicalArray(dense),
            Vec::new(),
        )
        .expect("sparse logical product") else {
            panic!("expected CSC")
        };
        assert_eq!(csc.numeric_dtype(), Some(NumericDType::F64));
        assert_eq!(csc.nnz(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_matrix_product() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![0.0, 6.0, 5.0, 7.0], vec![2, 2]).unwrap();
        let result = kron_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[4, 4]);
                assert_eq!(
                    t.materialize_f64(),
                    vec![
                        0.0, 6.0, 0.0, 18.0, 5.0, 7.0, 15.0, 21.0, 0.0, 12.0, 0.0, 24.0, 10.0,
                        14.0, 20.0, 28.0
                    ]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_scalar_scaling() {
        let a = Value::Num(3.0);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = kron_builtin(a, Value::Tensor(b), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert_eq!(t.materialize_f64(), vec![3.0, 6.0, 9.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn kron_preserves_native_single_for_single_and_mixed_floating_inputs() {
        let single = Tensor::from_f32(vec![1.5, -2.0], vec![2, 1]).unwrap();
        let single_scale = Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap();
        let result = kron_builtin(
            Value::Tensor(single.clone()),
            Value::Tensor(single_scale),
            Vec::new(),
        )
        .expect("single kron");
        let Value::Tensor(result) = result else {
            panic!("single kron must return a tensor");
        };
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![3.0, -4.0])
        );

        let double_scale = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
        let result = kron_builtin(
            Value::Tensor(single),
            Value::Tensor(double_scale),
            Vec::new(),
        )
        .expect("mixed floating kron");
        let Value::Tensor(result) = result else {
            panic!("mixed floating kron must return a tensor");
        };
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.75, -1.0])
        );
    }

    #[test]
    fn kron_preserves_every_exact_integer_class_and_saturates() {
        let cases = vec![
            (
                IntegerStorage::I8(vec![i8::MAX, -2]),
                IntegerStorage::I8(vec![2, 3]),
                IntegerStorage::I8(vec![i8::MAX, -4, i8::MAX, -6]),
            ),
            (
                IntegerStorage::I16(vec![i16::MAX, -2]),
                IntegerStorage::I16(vec![2, 3]),
                IntegerStorage::I16(vec![i16::MAX, -4, i16::MAX, -6]),
            ),
            (
                IntegerStorage::I32(vec![i32::MAX, -2]),
                IntegerStorage::I32(vec![2, 3]),
                IntegerStorage::I32(vec![i32::MAX, -4, i32::MAX, -6]),
            ),
            (
                IntegerStorage::I64(vec![i64::MAX, -2]),
                IntegerStorage::I64(vec![2, 3]),
                IntegerStorage::I64(vec![i64::MAX, -4, i64::MAX, -6]),
            ),
            (
                IntegerStorage::U8(vec![u8::MAX, 2]),
                IntegerStorage::U8(vec![2, 3]),
                IntegerStorage::U8(vec![u8::MAX, 4, u8::MAX, 6]),
            ),
            (
                IntegerStorage::U16(vec![u16::MAX, 2]),
                IntegerStorage::U16(vec![2, 3]),
                IntegerStorage::U16(vec![u16::MAX, 4, u16::MAX, 6]),
            ),
            (
                IntegerStorage::U32(vec![u32::MAX, 2]),
                IntegerStorage::U32(vec![2, 3]),
                IntegerStorage::U32(vec![u32::MAX, 4, u32::MAX, 6]),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
                IntegerStorage::U64(vec![2, 3]),
                IntegerStorage::U64(vec![u64::MAX, u64::MAX, u64::MAX, u64::MAX]),
            ),
        ];

        for (left, right, expected) in cases {
            let left = Tensor::new_integer(left, vec![2, 1]).expect("left integer tensor");
            let right = Tensor::new_integer(right, vec![1, 2]).expect("right integer tensor");
            let result = kron_builtin(Value::Tensor(left), Value::Tensor(right), Vec::new())
                .expect("integer kron");
            let Value::Tensor(result) = result else {
                panic!("integer matrix kron must return a tensor");
            };
            assert_eq!(result.shape(), &[2, 2]);
            assert_eq!(result.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn kron_integer_scalar_double_and_empty_paths_preserve_exact_class() {
        let values =
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
                .expect("uint64 tensor");
        let result = kron_builtin(Value::Tensor(values), Value::Num(2.0), Vec::new())
            .expect("integer scalar-double kron");
        let Value::Tensor(result) = result else {
            panic!("integer kron must return a tensor");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX]))
        );

        let empty = Tensor::new_integer(IntegerStorage::I64(Vec::new()), vec![0, 2])
            .expect("empty integer tensor");
        let result = kron_builtin(
            Value::Tensor(empty),
            Value::Int(IntValue::I64(3)),
            Vec::new(),
        )
        .expect("empty integer kron");
        let Value::Tensor(result) = result else {
            panic!("empty integer kron must return a tensor");
        };
        assert_eq!(result.shape(), &[0, 2]);
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I64(Vec::new()))
        );
    }

    #[test]
    fn kron_rejects_mixed_class_and_non_scalar_double_integer_inputs() {
        let left = Tensor::new_integer(IntegerStorage::I8(vec![1]), vec![1, 1]).unwrap();
        let right = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
        let error = kron_builtin(Value::Tensor(left), Value::Tensor(right), Vec::new())
            .expect_err("mixed integer classes must fail");
        assert!(error.to_string().contains("same integer class"));

        let left = Tensor::new_integer(IntegerStorage::I8(vec![1, 2]), vec![1, 2]).unwrap();
        let right = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let error = kron_builtin(Value::Tensor(left), Value::Tensor(right), Vec::new())
            .expect_err("non-scalar double array must fail with integer input");
        assert!(error.to_string().contains("scalar double or logical"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_complex_inputs() {
        let a = Value::Complex(1.0, 2.0);
        let b = Value::ComplexTensor(
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 0.0)], vec![2, 1]).unwrap(),
        );
        let result = kron_builtin(a, b, Vec::new()).expect("kron");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape(), &[2, 1]);
                assert_eq!(
                    ct.materialize_f64(),
                    vec![(0.0, 0.0), (1.0, 2.0 * 1.0 + 1.0 * 0.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn kron_complex_single_class_is_shared_by_dense_and_sparse_paths() {
        let single = ComplexTensor::from_f32(vec![(1.5, -2.0)], vec![1, 1]).unwrap();
        let double = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).unwrap();
        let dense = kron_builtin(
            Value::ComplexTensor(single),
            Value::ComplexTensor(double.clone()),
            Vec::new(),
        )
        .expect("mixed complex dense kron");
        let Value::ComplexTensor(dense) = dense else {
            panic!("expected complex dense tensor")
        };
        assert_eq!(dense.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            dense.as_f32_slice(),
            Some(&[runmat_value::ComplexElement(5.0, -2.5)][..])
        );

        let sparse =
            SparseTensor::new_complex_f32(1, 1, vec![0, 1], vec![0], vec![(1.5, -2.0)]).unwrap();
        let Value::SparseTensor(sparse) = kron_builtin(
            Value::SparseTensor(sparse),
            Value::ComplexTensor(double),
            Vec::new(),
        )
        .expect("mixed complex sparse kron") else {
            panic!("expected complex sparse tensor")
        };
        assert_eq!(sparse.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(sparse.as_complex_f32_slice(), dense.as_f32_slice());
    }

    #[test]
    fn kron_rejects_integer_and_complex_mixed_arithmetic() {
        let left = Tensor::new_integer(IntegerStorage::I64(vec![-3, 5]), vec![1, 2])
            .expect("integer tensor");
        let right = ComplexTensor::new(vec![(2.0, -1.0)], vec![1, 1]).unwrap();

        let error = kron_builtin(Value::Tensor(left), Value::ComplexTensor(right), Vec::new())
            .expect_err("complex integer arithmetic must reject");
        assert!(error.message().contains("complex integer arithmetic"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_logical_promotes_to_double() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let result = kron_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 4]);
                assert_eq!(
                    t.materialize_f64(),
                    vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_char_arrays_convert_to_double() {
        let chars = CharArray::from_code_units(
            [65, 0xD800, 66, 0xDC00, 67, 0xFFFF]
                .map(runmat_value::Utf16CodeUnit::from_raw)
                .to_vec(),
            3,
            2,
        )
        .unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result =
            kron_builtin(Value::CharArray(chars), Value::Tensor(tensor), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[6, 2]);
                assert_eq!(
                    t.as_f64_slice(),
                    Some(
                        &[
                            65.0, 130.0, 66.0, 132.0, 67.0, 134.0, 55296.0, 110592.0, 56320.0,
                            112640.0, 65535.0, 131070.0
                        ][..]
                    )
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_rejects_extra_arguments() {
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = kron_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("extra")],
        )
        .unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("too many"),
            "unexpected error: {err}"
        );
        assert_eq!(err.identifier(), KRON_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_rejects_unsupported_input_with_stable_identifier() {
        let a = Value::String("bad".to_string().into());
        let b = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
        let err = kron_builtin(a, b, Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), KRON_ERROR_UNSUPPORTED_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![0.0, 6.0, 5.0, 7.0], vec![2, 2]).unwrap();
            let view_a = HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            };
            let view_b = HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            };
            let handle_a = provider.upload(&view_a).expect("upload a");
            let handle_b = provider.upload(&view_b).expect("upload b");
            let result = kron_builtin(
                Value::GpuTensor(handle_a),
                Value::GpuTensor(handle_b),
                Vec::new(),
            )
            .expect("kron");
            match &result {
                Value::GpuTensor(_) => {}
                other => panic!("expected GPU tensor, got {other:?}"),
            }
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), &[4, 4]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![
                    0.0, 6.0, 0.0, 18.0, 5.0, 7.0, 15.0, 21.0, 0.0, 12.0, 0.0, 24.0, 10.0, 14.0,
                    20.0, 28.0
                ]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_mixed_gpu_host_reuploads() {
        test_support::with_test_provider(|provider| {
            let gpu_tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &gpu_tensor.materialize_f64(),
                shape: gpu_tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let host = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
            let result = kron_builtin(Value::GpuTensor(handle), Value::Tensor(host), Vec::new())
                .expect("kron");
            match result {
                Value::GpuTensor(_) => {}
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn kron_explicit_gpu_and_sparse_reject_before_gather_and_keep_source_alive() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
            let source = gpu_helpers::upload_tensor(provider, &tensor)
                .unwrap()
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
            let sparse = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![3.0]).unwrap();
            let error = kron_builtin(
                Value::GpuTensor(source.clone()),
                Value::SparseTensor(sparse),
                Vec::new(),
            )
            .expect_err("explicit sparse output has no resident representation");
            assert_eq!(error.identifier(), KRON_ERROR_UNSUPPORTED_INPUT.identifier);
            let original = test_support::gather(Value::GpuTensor(source)).unwrap();
            assert_eq!(original.materialize_f64(), vec![2.0]);
        });
    }

    #[test]
    fn kron_resident_fallback_preserves_source_handle_and_explicit_output() {
        test_support::with_test_provider(|provider| {
            let source_tensor = Tensor::new_integer(IntegerStorage::I32(vec![2, 3]), vec![2, 1])
                .expect("integer source");
            let source = gpu_helpers::upload_tensor(provider, &source_tensor).expect("upload");
            let source = source
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("validated fixture provenance");
            let result = kron_builtin(
                Value::GpuTensor(source.clone()),
                Value::Num(2.0),
                Vec::new(),
            )
            .expect("resident kron");
            let Value::GpuTensor(output) = result else {
                panic!("explicit gpuArray kron must remain resident");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            assert!(!source.shares_allocation_with(&output));
            let original = test_support::gather(Value::GpuTensor(source)).expect("source survives");
            assert_eq!(
                original.integer_storage(),
                Some(&IntegerStorage::I32(vec![2, 3]))
            );
        });
    }

    #[test]
    fn kron_fallback_prefers_explicit_second_resident_operand() {
        test_support::with_test_provider(|provider| {
            let scalar = Tensor::new(vec![2.0], vec![1, 1]).expect("scalar");
            let automatic = gpu_helpers::upload_tensor(provider, &scalar).expect("automatic");
            let automatic = automatic
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("validated fixture provenance");
            let integer =
                Tensor::new_integer(IntegerStorage::I32(vec![3, 4]), vec![2, 1]).expect("integer");
            let explicit = gpu_helpers::upload_tensor(provider, &integer).expect("explicit");
            let explicit = explicit
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("validated fixture provenance");

            let result = kron_builtin(
                Value::GpuTensor(automatic),
                Value::GpuTensor(explicit.clone()),
                Vec::new(),
            )
            .expect("fallback kron");
            let Value::GpuTensor(output) = result else {
                panic!("explicit second operand must preserve resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::I32)
            );
            let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::I32(vec![6, 8]))
            );
            let original = test_support::gather(Value::GpuTensor(explicit)).expect("source");
            assert_eq!(
                original.integer_storage(),
                Some(&IntegerStorage::I32(vec![3, 4]))
            );
        });
    }

    #[test]
    fn kron_nd_input_is_gated_in_matlab_mode() {
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 1, 2]).unwrap();
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = kron_builtin(Value::Tensor(input), Value::Num(2.0), Vec::new())
            .expect_err("N-D kron must be gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:KronNdInputExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_empty_inputs() {
        let a = Tensor::new(Vec::new(), vec![0, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = kron_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0, 4]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn kron_wgpu_matches_cpu() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![0.0, 6.0, 5.0, 7.0], vec![2, 2]).unwrap();

        let cpu_value = kron_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            Vec::new(),
        )
        .expect("cpu");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: a.shape(),
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: b.shape(),
        };
        let handle_a = provider.upload(&view_a).expect("upload a");
        let handle_b = provider.upload(&view_b).expect("upload b");

        let gpu_value = kron_builtin(
            Value::GpuTensor(handle_a),
            Value::GpuTensor(handle_b),
            Vec::new(),
        )
        .expect("gpu");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gpu_tensor.shape(), cpu_tensor.shape());
        for (idx, (g, c)) in gpu_tensor
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
            .enumerate()
        {
            assert!(
                (*g - *c).abs() < 1e-9,
                "mismatch at index {idx}: {g} vs {c}"
            );
        }
    }
}
