//! MATLAB-compatible `kron` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, IntValue, IntegerStorage, NumericDType, NumericStorage,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::math::elementwise::integer_arithmetic::{
    integer_binary_scalar, IntegerBinaryOp,
};

type AlignedShapes = (Vec<usize>, Vec<usize>, Vec<usize>);
const BUILTIN_NAME: &str = "kron";

fn kron_type(args: &[Type], _context: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { .. } => Type::tensor(),
        Type::Logical { .. } => Type::logical(),
        Type::Num | Type::Int | Type::Bool => Type::tensor(),
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

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
    notes: "Executes entirely on-device when the provider implements `kron`; otherwise the runtime gathers inputs, computes on the host, and re-uploads the result when possible.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::kron")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "kron",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Kronecker products allocate a fresh tensor and terminate fusion graphs.",
};

const KRON_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Kronecker product of inputs A and B.",
}];

const KRON_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left numeric/logical/complex input array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right numeric/logical/complex input array.",
    },
];

const KRON_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = kron(A, B)",
    inputs: &KRON_INPUTS_A_B,
    outputs: &KRON_OUTPUT,
}];

const KRON_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KRON.TOO_MANY_INPUTS",
    identifier: Some("RunMat:kron:TooManyInputs"),
    when: "Extra arguments were supplied after A and B.",
    message: "kron: too many input arguments",
};

const KRON_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KRON.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:kron:UnsupportedInput"),
    when: "Input values are not numeric/logical/complex arrays.",
    message: "kron: unsupported input type",
};

const KRON_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KRON.INTERNAL",
    identifier: Some("RunMat:kron:Internal"),
    when: "Internal conversion/allocation/provider path fails.",
    message: "kron: internal operation failed",
};

const KRON_ERRORS: [BuiltinErrorDescriptor; 3] = [
    KRON_ERROR_TOO_MANY_INPUTS,
    KRON_ERROR_UNSUPPORTED_INPUT,
    KRON_ERROR_INTERNAL,
];

pub const KRON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &KRON_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &KRON_ERRORS,
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
    category = "array/shape",
    summary = "Compute the Kronecker (tensor) product of two arrays.",
    keywords = "kron,kronecker product,tensor product,block matrix,gpu",
    accel = "custom",
    type_resolver(kron_type),
    descriptor(crate::builtins::array::shape::kron::KRON_DESCRIPTOR),
    builtin_path = "crate::builtins::array::shape::kron"
)]
async fn kron_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(kron_error(&KRON_ERROR_TOO_MANY_INPUTS));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&a, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, BUILTIN_NAME)?;

    match (a, b) {
        (Value::GpuTensor(left), Value::GpuTensor(right)) => Ok(kron_gpu_gpu(left, right).await?),
        (Value::GpuTensor(left), right) => Ok(kron_gpu_mixed_left(left, right).await?),
        (left, Value::GpuTensor(right)) => Ok(kron_gpu_mixed_right(left, right).await?),
        (left, right) => Ok(kron_host(left, right)?),
    }
}

async fn kron_gpu_gpu(
    left: GpuTensorHandle,
    right: GpuTensorHandle,
) -> crate::BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&left);
    if let Some(provider) = provider {
        if runmat_accelerate_api::handle_integer_type(&left).is_none()
            && runmat_accelerate_api::handle_integer_type(&right).is_none()
        {
            if let Ok(handle) = provider.kron(&left, &right) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    let left_tensor = gpu_helpers::gather_tensor_async(&left).await?;
    let right_tensor = gpu_helpers::gather_tensor_async(&right).await?;
    if let Some(provider) = provider {
        let _ = provider.free(&left);
        let _ = provider.free(&right);
    }
    let left_value = tensor::tensor_into_value(left_tensor);
    let right_value = tensor::tensor_into_value(right_tensor);
    let numeric = compute_numeric(left_value, right_value)?;
    finalize_numeric(numeric, provider)
}

async fn kron_gpu_mixed_left(left: GpuTensorHandle, right: Value) -> crate::BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&left);
    if let Some(provider) = provider {
        if let Ok(tensor_right) = tensor::value_into_tensor_for("kron", right.clone()) {
            if runmat_accelerate_api::handle_integer_type(&left).is_none()
                && matches!(
                    tensor_right.numeric_dtype(),
                    NumericDType::F64 | NumericDType::F32
                )
            {
                if let Ok(uploaded) = gpu_helpers::upload_tensor(provider, &tensor_right) {
                    match provider.kron(&left, &uploaded) {
                        Ok(handle) => {
                            let _ = provider.free(&uploaded);
                            return Ok(Value::GpuTensor(handle));
                        }
                        Err(_) => {
                            let _ = provider.free(&uploaded);
                        }
                    }
                }
            }
        }
    }

    let left_tensor = gpu_helpers::gather_tensor_async(&left).await?;
    if let Some(provider) = provider {
        let _ = provider.free(&left);
    }
    let left_value = tensor::tensor_into_value(left_tensor);
    let numeric = compute_numeric(left_value, right)?;
    finalize_numeric(numeric, provider)
}

async fn kron_gpu_mixed_right(left: Value, right: GpuTensorHandle) -> crate::BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&right);
    if let Some(provider) = provider {
        if let Ok(tensor_left) = tensor::value_into_tensor_for("kron", left.clone()) {
            if runmat_accelerate_api::handle_integer_type(&right).is_none()
                && matches!(
                    tensor_left.numeric_dtype(),
                    NumericDType::F64 | NumericDType::F32
                )
            {
                if let Ok(uploaded) = gpu_helpers::upload_tensor(provider, &tensor_left) {
                    match provider.kron(&uploaded, &right) {
                        Ok(handle) => {
                            let _ = provider.free(&uploaded);
                            return Ok(Value::GpuTensor(handle));
                        }
                        Err(_) => {
                            let _ = provider.free(&uploaded);
                        }
                    }
                }
            }
        }
    }

    let right_tensor = gpu_helpers::gather_tensor_async(&right).await?;
    if let Some(provider) = provider {
        let _ = provider.free(&right);
    }
    let right_value = tensor::tensor_into_value(right_tensor);
    let numeric = compute_numeric(left, right_value)?;
    finalize_numeric(numeric, provider)
}

fn kron_host(left: Value, right: Value) -> crate::BuiltinResult<Value> {
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
}

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

    fn class_name(&self) -> &'static str {
        match self {
            Self::Scalar(value) => value.class_name(),
            Self::Array(storage, _) => storage.class_name(),
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
            .map(|storage| KronIntegerOperand::Array(storage, &tensor.shape)),
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
        if left_integer.class_name() != right_integer.class_name() {
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
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))
}

fn tensor_to_complex(tensor: &Tensor) -> crate::BuiltinResult<ComplexTensor> {
    let values = tensor::tensor_values_f64_cow(tensor);
    let data: Vec<(f64, f64)> = values.iter().map(|&re| (re, 0.0)).collect();
    ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL))
}

fn kron_tensor(a: &Tensor, b: &Tensor) -> crate::BuiltinResult<Tensor> {
    let (shape_a, shape_b, shape_out) = aligned_shapes(&a.shape, &b.shape)?;
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
    let (shape_a, shape_b, shape_out) = aligned_shapes(&a.shape, &b.shape)?;
    let total_out = checked_total(&shape_out, "kron")?;
    if total_out == 0 {
        return ComplexTensor::new(Vec::new(), shape_out)
            .map_err(|e| kron_error_with_message(format!("kron: {e}"), &KRON_ERROR_INTERNAL));
    }

    let strides_out = column_major_strides(&shape_out);
    let mut coords_a = vec![0usize; shape_out.len()];
    let mut coords_b = vec![0usize; shape_out.len()];
    let mut data = vec![(0.0f64, 0.0f64); total_out];

    for (idx_a, &(ar, ai)) in a.materialize_f64().iter().enumerate() {
        unravel_index(idx_a, &shape_a, &mut coords_a);
        for (idx_b, &(br, bi)) in b.materialize_f64().iter().enumerate() {
            unravel_index(idx_b, &shape_b, &mut coords_b);
            let out_index = combine_indices(&coords_a, &coords_b, &shape_b, &strides_out)?;
            let real = ar * br - ai * bi;
            let imag = ar * bi + ai * br;
            data[out_index] = (real, imag);
        }
    }

    ComplexTensor::new(data, shape_out)
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
        block_on(super::kron_builtin(a, b, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntegerStorage, LogicalArray, Tensor, Type};

    #[test]
    fn kron_type_logical_returns_logical() {
        let out = kron_type(
            &[Type::Logical { shape: None }, Type::Logical { shape: None }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::logical());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_matrix_product() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![0.0, 6.0, 5.0, 7.0], vec![2, 2]).unwrap();
        let result = kron_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 4]);
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
                assert_eq!(t.shape, vec![2, 2]);
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
            assert_eq!(result.shape, vec![2, 2]);
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
        assert_eq!(result.shape, vec![0, 2]);
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
                assert_eq!(ct.shape, vec![2, 1]);
                assert_eq!(
                    ct.materialize_f64(),
                    vec![(0.0, 0.0), (1.0, 2.0 * 1.0 + 1.0 * 0.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn kron_complex_path_reads_typed_integer_storage_exactly() {
        let left = Tensor::new_integer(IntegerStorage::I64(vec![-3, 5]), vec![1, 2])
            .expect("integer tensor");
        let right = ComplexTensor::new(vec![(2.0, -1.0)], vec![1, 1]).unwrap();

        let result = kron_builtin(Value::Tensor(left), Value::ComplexTensor(right), Vec::new())
            .expect("kron");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(out.materialize_f64(), vec![(-6.0, 3.0), (10.0, -5.0)]);
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
                assert_eq!(t.shape, vec![2, 4]);
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
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result =
            kron_builtin(Value::CharArray(chars), Value::Tensor(tensor), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected: Vec<f64> = "AB"
                    .chars()
                    .flat_map(|ch| [ch as u32 as f64, 2.0 * ch as u32 as f64])
                    .collect();
                assert_eq!(t.materialize_f64(), expected);
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
        let a = Value::String("bad".to_string());
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
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.materialize_f64(),
                shape: &b.shape,
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
            assert_eq!(gathered.shape, vec![4, 4]);
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
                shape: &gpu_tensor.shape,
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_empty_inputs() {
        let a = Tensor::new(Vec::new(), vec![0, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = kron_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 4]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn kron_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

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

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: &b.shape,
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

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
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
