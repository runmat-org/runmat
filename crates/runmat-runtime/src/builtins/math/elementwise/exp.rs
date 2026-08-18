//! MATLAB-compatible `exp` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise exponential for documented floating inputs plus independently gated
//! RunMat integer, logical, and character extensions. GPU fallbacks preserve the source owner.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, IntegerStorage, NumericStorage, ObjectInstance,
    SparseTensor, StructValue, SymbolicFunction, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::symbolic::symbolic_function;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::exp")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "exp",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_exp" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may evaluate real floating exp directly; complex values and unsupported hooks use an owner-preserving host fallback.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::exp")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "exp",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("exp({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `exp` calls; providers can override with fused elementwise kernels.",
};

const BUILTIN_NAME: &str = "exp";

const EXP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise exponential result.",
}];
const EXP_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const EXP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = exp(X)",
    inputs: &EXP_INPUTS,
    outputs: &EXP_OUTPUT,
}];
const EXP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXP.INVALID_INPUT",
    identifier: Some("RunMat:exp:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, char, or complex data.",
    message: "exp: invalid input",
};
const EXP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXP.INTERNAL",
    identifier: Some("RunMat:exp:Internal"),
    when: "Internal tensor construction or provider interaction failed.",
    message: "exp: internal error",
};
const EXP_ERRORS: [BuiltinErrorDescriptor; 2] = [EXP_ERROR_INVALID_INPUT, EXP_ERROR_INTERNAL];

const EXP_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exp-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exp with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExpIntegerInputExtension"),
};
const EXP_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exp-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exp with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExpLogicalInputExtension"),
};
const EXP_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exp-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exp with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExpCharacterInputExtension"),
};
const EXP_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    EXP_INTEGER_INPUT_EXTENSION,
    EXP_LOGICAL_INPUT_EXTENSION,
    EXP_CHARACTER_INPUT_EXTENSION,
];

const EXP_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are accepted only in RunMat extension mode and only when every value lies in the inclusive exact binary64 interval [-2^53, 2^53].",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = exp(integer_X)",
        inputs: &EXP_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only overload validates exact binary64 conversion before exponentiation. Resident integer input gathers exactly through its owning provider; the double result is restored only when that provider physically supports binary64, otherwise it remains a host double.",
    }];
pub const EXP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EXP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXP_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn exp_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "exp",
    category = "math/elementwise",
    summary = "Compute element-wise exponential values.",
    keywords = "exp,exponential,elementwise,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::exp::EXP_DESCRIPTOR),
    extensions(EXP_EXTENSIONS),
    integer_capabilities(crate::builtins::math::elementwise::exp::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::exp"
)]
async fn exp_builtin(value: Value) -> BuiltinResult<Value> {
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Exp) {
        return Ok(symbolic);
    }
    ensure_exp_extensions(&value)?;
    exp_value(value).await
}

async fn exp_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            exp_table(object).await
        }
        other => exp_non_table_value(other).await,
    }
}

async fn exp_non_table_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => exp_gpu(handle).await,
        Value::Complex(re, im) => {
            let (re, im) = exp_complex_parts(re, im);
            Ok(Value::Complex(re, im))
        }
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "exp")?;
            exp_complex_tensor(ct)
        }
        Value::SparseTensor(sparse) => exp_sparse_tensor(sparse),
        Value::CharArray(ca) => exp_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(exp_error_with_detail(
            &EXP_ERROR_INVALID_INPUT,
            "expected numeric input, got string",
        )),
        other => exp_real(other),
    }
}

fn ensure_exp_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXP_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::SparseTensor(sparse) if sparse.is_logical())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXP_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXP_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn exp_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        exp_error_with_detail(&EXP_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let tensor = gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let output = exp_tensor(tensor)?;
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
            return Ok(tensor::tensor_into_value(output));
        }
        return restore_real_gpu_output(provider, &handle, &output);
    }
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let result = exp_host_value(gathered)?;
        return restore_gpu_value(provider, &handle, result);
    }
    match provider.unary_exp(&handle).await {
        Ok(out) if valid_real_gpu_output(&out, &handle, provider) => {
            return Ok(gpu_helpers::resident_gpu_value(out));
        }
        Ok(out) => {
            free_rejected_gpu_output(&out, &handle);
            return Err(exp_error_with_detail(
                &EXP_ERROR_INTERNAL,
                "provider unary_exp returned malformed output",
            ));
        }
        Err(err) if is_unsupported_provider_hook(&err) => {}
        Err(err) => {
            return Err(exp_error_with_detail(
                &EXP_ERROR_INTERNAL,
                format!("provider unary_exp failed: {err}"),
            ));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let output = exp_tensor(tensor)?;
    restore_real_gpu_output(provider, &handle, &output)
}

fn exp_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("exp", value)
        .map_err(|e| builtin_error(format!("exp: {e}")))?;
    Ok(tensor::tensor_into_value(exp_tensor(tensor)?))
}

fn exp_host_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(re, im) => {
            let (re, im) = exp_complex_parts(re, im);
            Ok(Value::Complex(re, im))
        }
        Value::ComplexTensor(tensor) => exp_complex_tensor(tensor),
        other => exp_real(other),
    }
}

fn exp_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("exp: {e}")))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(f64::exp).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(f32::exp).collect())
        }
        storage => NumericStorage::F64(
            promote_integer_storage_to_exp_domain(storage)?
                .into_iter()
                .map(f64::exp)
                .collect(),
        ),
    };
    Tensor::from_numeric_storage(output, shape).map_err(|e| builtin_error(format!("exp: {e}")))
}

fn exp_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| (exp_complex_re(real, imag), exp_complex_im(real, imag)))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| exp_complex_parts_f32(real, imag))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(exp_error_with_detail(
                &EXP_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error(format!("exp: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn promote_integer_storage_to_exp_domain(storage: NumericStorage) -> BuiltinResult<Vec<f64>> {
    let storage = storage
        .into_integer_storage()
        .expect("exp integer-promotion boundary received floating storage");
    ensure_integer_storage_exact_binary64(&storage)?;
    Ok(storage.to_f64_vec())
}

fn exp_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| (ch as u32 as f64).exp()).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("exp: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn exp_sparse_tensor(sparse: SparseTensor) -> BuiltinResult<Value> {
    let rows = sparse.rows;
    let cols = sparse.cols;
    let len = rows.checked_mul(cols).ok_or_else(|| {
        exp_error_with_detail(&EXP_ERROR_INTERNAL, "sparse output element count overflow")
    })?;
    if let Some(values) = sparse.as_f64_slice() {
        let dense = sparse_to_dense(rows, cols, &sparse.col_ptrs, &sparse.row_indices, values)?;
        return Tensor::new(dense.into_iter().map(f64::exp).collect(), vec![rows, cols])
            .map(Value::Tensor)
            .map_err(|err| exp_error_with_detail(&EXP_ERROR_INTERNAL, err));
    }
    if let Some(values) = sparse.as_f32_slice() {
        let dense = sparse_to_dense(rows, cols, &sparse.col_ptrs, &sparse.row_indices, values)?;
        return Tensor::from_f32(dense.into_iter().map(f32::exp).collect(), vec![rows, cols])
            .map(Value::Tensor)
            .map_err(|err| exp_error_with_detail(&EXP_ERROR_INTERNAL, err));
    }
    let mut dense = vec![0.0; len];
    if sparse.is_logical() {
        fill_sparse_dense(
            rows,
            cols,
            &sparse.col_ptrs,
            &sparse.row_indices,
            &vec![1.0; sparse.nnz()],
            &mut dense,
        )?;
    } else if let Some(storage) = sparse.integer_storage() {
        ensure_integer_storage_exact_binary64(storage)?;
        let values = storage.to_f64_vec();
        fill_sparse_dense(
            rows,
            cols,
            &sparse.col_ptrs,
            &sparse.row_indices,
            &values,
            &mut dense,
        )?;
    } else {
        return Err(exp_error_with_detail(
            &EXP_ERROR_INVALID_INPUT,
            "unsupported sparse storage",
        ));
    }
    Tensor::new(dense.into_iter().map(f64::exp).collect(), vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| exp_error_with_detail(&EXP_ERROR_INTERNAL, err))
}

fn sparse_to_dense<T: Copy + Default>(
    rows: usize,
    cols: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[T],
) -> BuiltinResult<Vec<T>> {
    let len = rows.checked_mul(cols).ok_or_else(|| {
        exp_error_with_detail(&EXP_ERROR_INTERNAL, "sparse output element count overflow")
    })?;
    let mut dense = vec![T::default(); len];
    fill_sparse_dense(rows, cols, col_ptrs, row_indices, values, &mut dense)?;
    Ok(dense)
}

fn fill_sparse_dense<T: Copy>(
    rows: usize,
    cols: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[T],
    dense: &mut [T],
) -> BuiltinResult<()> {
    if col_ptrs.len() != cols + 1 || row_indices.len() != values.len() {
        return Err(exp_error_with_detail(
            &EXP_ERROR_INTERNAL,
            "malformed sparse CSC storage",
        ));
    }
    for col in 0..cols {
        for index in col_ptrs[col]..col_ptrs[col + 1] {
            let row = *row_indices.get(index).ok_or_else(|| {
                exp_error_with_detail(&EXP_ERROR_INTERNAL, "malformed sparse row index")
            })?;
            let value = *values.get(index).ok_or_else(|| {
                exp_error_with_detail(&EXP_ERROR_INTERNAL, "malformed sparse value index")
            })?;
            let offset = row
                .checked_add(col.saturating_mul(rows))
                .filter(|offset| row < rows && *offset < dense.len())
                .ok_or_else(|| {
                    exp_error_with_detail(&EXP_ERROR_INTERNAL, "sparse row index out of bounds")
                })?;
            dense[offset] = value;
        }
    }
    Ok(())
}

async fn exp_table(object: ObjectInstance) -> BuiltinResult<Value> {
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|err| exp_error_with_detail(&EXP_ERROR_INVALID_INPUT, err.message))?;
    let mut output = StructValue::new();
    for (name, value) in variables.fields {
        ensure_exp_extensions(&value)?;
        if matches!(value, Value::Object(_)) {
            return Err(exp_error_with_detail(
                &EXP_ERROR_INVALID_INPUT,
                format!("table variable {name} does not support exp"),
            ));
        }
        output.insert(name, exp_non_table_value(value).await?);
    }
    crate::builtins::table::table_replace_variables_like(&object, output)
        .map_err(|err| exp_error_with_detail(&EXP_ERROR_INTERNAL, err.message))
}

const MAX_EXACT_BINARY64_INTEGER: i128 = 9_007_199_254_740_992;

fn ensure_integer_storage_exact_binary64(storage: &IntegerStorage) -> BuiltinResult<()> {
    let valid = match storage {
        IntegerStorage::I8(_) | IntegerStorage::I16(_) | IntegerStorage::I32(_) => true,
        IntegerStorage::I64(values) => values.iter().all(|&value| {
            let value = i128::from(value);
            (-MAX_EXACT_BINARY64_INTEGER..=MAX_EXACT_BINARY64_INTEGER).contains(&value)
        }),
        IntegerStorage::U8(_) | IntegerStorage::U16(_) | IntegerStorage::U32(_) => true,
        IntegerStorage::U64(values) => values
            .iter()
            .all(|&value| u128::from(value) <= MAX_EXACT_BINARY64_INTEGER as u128),
    };
    if valid {
        Ok(())
    } else {
        Err(exp_error_with_detail(
            &EXP_ERROR_INVALID_INPUT,
            "integer input lies outside the inclusive exact binary64 interval [-2^53, 2^53]",
        ))
    }
}

fn restore_real_gpu_output(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    tensor: &Tensor,
) -> BuiltinResult<Value> {
    let output = gpu_helpers::upload_tensor(provider, tensor).map_err(|err| {
        exp_error_with_detail(
            &EXP_ERROR_INTERNAL,
            format!("failed to restore fallback result to input provider: {err}"),
        )
    })?;
    if !valid_real_gpu_output(&output, input, provider) {
        free_rejected_gpu_output(&output, input);
        return Err(exp_error_with_detail(
            &EXP_ERROR_INTERNAL,
            "provider upload returned malformed fallback output",
        ));
    }
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn restore_gpu_value(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    value: Value,
) -> BuiltinResult<Value> {
    match value {
        Value::ComplexTensor(tensor) => {
            let output = gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|err| {
                exp_error_with_detail(
                    &EXP_ERROR_INTERNAL,
                    format!("failed to restore complex result to input provider: {err}"),
                )
            })?;
            if !valid_complex_gpu_output(&output, input, provider) {
                free_rejected_gpu_output(&output, input);
                return Err(exp_error_with_detail(
                    &EXP_ERROR_INTERNAL,
                    "provider upload returned malformed complex fallback output",
                ));
            }
            Ok(gpu_helpers::complex_gpu_value(output))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], input.shape.clone())
                .map_err(|err| exp_error_with_detail(&EXP_ERROR_INTERNAL, err))?;
            restore_gpu_value(provider, input, Value::ComplexTensor(tensor))
        }
        Value::Tensor(tensor) => restore_real_gpu_output(provider, input, &tensor),
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], input.shape.clone())
                .map_err(|err| exp_error_with_detail(&EXP_ERROR_INTERNAL, err))?;
            restore_real_gpu_output(provider, input, &tensor)
        }
        other => Err(exp_error_with_detail(
            &EXP_ERROR_INTERNAL,
            format!("unexpected host fallback result {other:?}"),
        )),
    }
}

fn valid_real_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    valid_gpu_output(output, input, provider, GpuTensorStorage::Real)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
}

fn valid_complex_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    valid_gpu_output(
        output,
        input,
        provider,
        GpuTensorStorage::ComplexInterleaved,
    ) && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
}

fn valid_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    storage: GpuTensorStorage,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_storage(output) == storage
        && runmat_accelerate_api::provider_for_handle(output)
            .filter(|owner| owner.device_id() == output.device_id)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn free_rejected_gpu_output(output: &GpuTensorHandle, input: &GpuTensorHandle) {
    if gpu_handles_alias(output, input) {
        return;
    }
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(output)
        .filter(|owner| owner.device_id() == output.device_id)
    {
        let _ = owner.free(output);
    }
}

fn is_unsupported_provider_hook(err: &anyhow::Error) -> bool {
    err.to_string().contains("unary_exp not supported")
}

fn exp_complex_parts(re: f64, im: f64) -> (f64, f64) {
    if im == 0.0 {
        return (re.exp(), im);
    }
    let exp_re = re.exp();
    (exp_re * im.cos(), exp_re * im.sin())
}

#[inline]
fn exp_complex_re(re: f64, im: f64) -> f64 {
    exp_complex_parts(re, im).0
}

#[inline]
fn exp_complex_im(re: f64, im: f64) -> f64 {
    exp_complex_parts(re, im).1
}

#[inline]
fn exp_complex_parts_f32(re: f32, im: f32) -> (f32, f32) {
    if im == 0.0 {
        return (re.exp(), im);
    }
    let exp_re = re.exp();
    (exp_re * im.cos(), exp_re * im.sin())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, LogicalArray, ResolveContext, Tensor, Type};

    fn exp_builtin(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::exp_builtin(value))
    }

    fn exp_builtin_matlab(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        block_on(super::exp_builtin(value))
    }

    #[test]
    fn exp_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = EXP_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = exp(X)"));
    }

    #[test]
    fn exp_string_rejected_with_stable_identifier() {
        let err = exp_builtin(Value::from("bad")).expect_err("expected input error");
        assert_eq!(err.identifier(), EXP_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn exp_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
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
    fn exp_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_scalar() {
        let result = exp_builtin(Value::Num(1.0)).expect("exp");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::E).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let result = exp_builtin(Value::Tensor(tensor)).expect("exp");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected: Vec<f64> = vec![0.0_f64, 1.0_f64, 2.0_f64]
                    .into_iter()
                    .map(|v| v.exp())
                    .collect();
                for (a, b) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            Value::Num(_) => panic!("expected tensor result"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn exp_preserves_native_single_real_complex_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let Value::Tensor(output) = exp_builtin(Value::Tensor(tensor)).expect("exp") else {
            panic!("expected single tensor");
        };
        assert_eq!(output.shape, vec![2, 1]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 1.0_f32.exp()])
        );

        let complex = ComplexTensor::from_f32(vec![(0.0, 0.0), (1.0, 0.5)], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) = exp_builtin(Value::ComplexTensor(complex)).expect("exp")
        else {
            panic!("expected complex single tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.as_f32_slice(),
            Some(
                &[
                    exp_complex_parts_f32(0.0, 0.0),
                    exp_complex_parts_f32(1.0, 0.5),
                ][..]
            )
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::ComplexTensor(output) = exp_builtin(Value::ComplexTensor(empty)).expect("exp")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn exp_integer_gpu_rejects_inexact_binary64_value_after_exact_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![0, 9_007_199_254_740_993]),
                vec![1, 2],
            )
            .unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let err = exp_builtin(Value::GpuTensor(handle)).expect_err("inexact integer");
            assert_eq!(err.identifier(), EXP_ERROR_INVALID_INPUT.identifier);
        });
    }

    #[test]
    fn exp_resident_integer_stays_host_double_on_f32_owner() {
        test_support::with_f32_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::I16(vec![0, 1]), vec![1, 2])
                .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let Value::Tensor(output) = exp_builtin(Value::GpuTensor(handle)).expect("exp") else {
                panic!("expected host double output");
            };
            assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F64);
            assert_eq!(output.materialize_f64(), &[1.0, 1.0_f64.exp()]);
        });
    }

    #[test]
    fn exp_independent_extensions_are_gated_with_stable_identifiers() {
        let err = exp_builtin_matlab(Value::Int(IntValue::I8(1))).expect_err("integer gate");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:ExpIntegerInputExtension")
        );
        let err = exp_builtin_matlab(Value::Bool(true)).expect_err("logical gate");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:ExpLogicalInputExtension")
        );
        let chars = CharArray::new(vec!['A'], 1, 1).unwrap();
        let err = exp_builtin_matlab(Value::CharArray(chars)).expect_err("character gate");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:ExpCharacterInputExtension")
        );
    }

    #[test]
    fn exp_accepts_all_integer_classes_and_exact_endpoints() {
        let storages = vec![
            IntegerStorage::I8(vec![-1, 1]),
            IntegerStorage::I16(vec![-1, 1]),
            IntegerStorage::I32(vec![-1, 1]),
            IntegerStorage::I64(vec![-9_007_199_254_740_992, 9_007_199_254_740_992]),
            IntegerStorage::U8(vec![0, 1]),
            IntegerStorage::U16(vec![0, 1]),
            IntegerStorage::U32(vec![0, 1]),
            IntegerStorage::U64(vec![0, 9_007_199_254_740_992]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![1, 2]).unwrap();
            let Value::Tensor(output) = exp_builtin(Value::Tensor(tensor)).expect("integer exp")
            else {
                panic!("expected double tensor");
            };
            assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F64);
        }
        for storage in [
            IntegerStorage::I64(vec![-9_007_199_254_740_993]),
            IntegerStorage::I64(vec![9_007_199_254_740_993]),
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
        ] {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let err = exp_builtin(Value::Tensor(tensor)).expect_err("outside exact interval");
            assert_eq!(err.identifier(), EXP_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1]).unwrap();

        let result = exp_builtin(Value::Tensor(tensor)).expect("exp");

        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert!(t.integer_storage().is_none());
                let expected = [1.0, std::f64::consts::E, 2.0_f64.exp()];
                for (actual, expected) in t.materialize_f64().iter().zip(expected) {
                    assert!((*actual - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_int_value_promotes() {
        let value = Value::Int(IntValue::I32(2));
        let result = exp_builtin(value).expect("exp");
        match result {
            Value::Num(v) => assert!((v - 2.0_f64.exp()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_bool_scalar() {
        let result = exp_builtin(Value::Bool(true)).expect("exp");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::E).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_complex_scalar() {
        let result = exp_builtin(Value::Complex(1.0, 2.0)).expect("exp");
        match result {
            Value::Complex(re, im) => {
                let expected = (1.0f64.exp() * 2.0f64.cos(), 1.0f64.exp() * 2.0f64.sin());
                assert!((re - expected.0).abs() < 1e-12);
                assert!((im - expected.1).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_complex_tensor_elements() {
        let tensor = ComplexTensor::new(vec![(0.0, 0.0), (1.0, 1.0)], vec![2, 1]).unwrap();
        let result = exp_builtin(Value::ComplexTensor(tensor)).expect("exp");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let expected: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 1.0)]
                    .into_iter()
                    .map(|(re, im)| (exp_complex_re(re, im), exp_complex_im(re, im)))
                    .collect();
                for (idx, (re, im)) in t.materialize_f64().iter().enumerate() {
                    assert!((re - expected[idx].0).abs() < 1e-12);
                    assert!((im - expected[idx].1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_char_array_roundtrip() {
        let chars = CharArray::new("Hi".chars().collect(), 1, 2).unwrap();
        let result = exp_builtin(Value::CharArray(chars)).expect("exp");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = "Hi".chars().map(|c| (c as u32 as f64).exp()).collect();
                for (a, b) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_logical_array_promotes_to_double() {
        let logical =
            LogicalArray::new(vec![1u8, 0u8, 1u8, 0u8], vec![2, 2]).expect("logical array");
        let result = exp_builtin(Value::LogicalArray(logical)).expect("exp");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [std::f64::consts::E, 1.0, std::f64::consts::E, 1.0];
                for (a, b) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_string_rejected() {
        let err = exp_builtin(Value::from("runmat")).unwrap_err();
        assert!(
            err.message().contains("expected numeric input"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exp_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = exp_builtin(Value::GpuTensor(handle)).expect("exp");
            assert!(matches!(&result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.exp()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (a, b) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn exp_gpu_output_contract_rejects_alias_shape_and_storage_metadata() {
        test_support::with_test_provider(|provider| {
            let input = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
            )
            .unwrap();
            let mut output = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
            )
            .unwrap();
            assert!(valid_real_gpu_output(&output, &input, provider));
            assert!(!valid_real_gpu_output(&input, &input, provider));
            let mut wrong_shape = output.clone();
            wrong_shape.shape = vec![1, 2];
            assert!(!valid_real_gpu_output(&wrong_shape, &input, provider));
            output.descriptor.storage =
                Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
            assert!(!valid_real_gpu_output(&output, &input, provider));
        });
    }

    #[test]
    fn exp_sparse_densifies_implicit_zeros_and_preserves_single_class() {
        let sparse =
            SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let Value::Tensor(output) = exp_builtin(Value::SparseTensor(sparse)).expect("sparse exp")
        else {
            panic!("exp of sparse input must be dense");
        };
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0_f32.exp(), 1.0, 1.0, 2.0_f32.exp()])
        );
    }

    #[test]
    fn exp_table_maps_supported_variables_and_preserves_container() {
        let input = crate::builtins::table::table_from_columns(
            vec!["Double".into(), "Single".into()],
            vec![
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            ],
        )
        .unwrap();
        let Value::Object(output) = exp_builtin(input).expect("table exp") else {
            panic!("expected table");
        };
        let variables = crate::builtins::table::table_variables(&output).unwrap();
        assert_eq!(variables.fields.len(), 2);
        assert!(matches!(
            variables.fields.get("Single"),
            Some(Value::Tensor(tensor)) if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32
        ));
    }

    #[test]
    fn exp_complex_real_axis_preserves_signed_zero_at_infinite_endpoint() {
        let Value::Complex(re, im) = exp_builtin(Value::Complex(f64::INFINITY, -0.0)).unwrap()
        else {
            panic!("expected complex scalar");
        };
        assert_eq!(re, f64::INFINITY);
        assert_eq!(im, 0.0);
        assert!(im.is_sign_negative());
        let Value::Complex(re, im) = exp_builtin(Value::Complex(f64::NAN, 0.0)).unwrap() else {
            panic!("expected complex scalar");
        };
        assert!(re.is_nan());
        assert_eq!(im, 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn exp_wgpu_matches_cpu_elementwise() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let cpu = exp_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let handle = provider.upload(&view).unwrap();
        let gpu = block_on(exp_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu value {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
        }
    }
}
