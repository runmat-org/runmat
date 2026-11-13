//! MATLAB-compatible `ismember` builtin with GPU-aware semantics for RunMat.

use super::set_policy::{ElementKey, Operand};
#[cfg(test)]
use super::type_resolvers::logical_output_type;
use crate::call::{BuiltinArguments, FunctionArgumentView};
pub use runmat_builtins::{
    ISMEMBER_DESCRIPTOR, ISMEMBER_ERROR_INTERNAL, ISMEMBER_ERROR_INVALID_ARGUMENT,
    ISMEMBER_ERROR_LEGACY_OPTION_UNSUPPORTED, ISMEMBER_ERROR_NUMERIC_CLASS_MISMATCH,
    ISMEMBER_ERROR_ROWS_COLUMN_MISMATCH, ISMEMBER_ERROR_UNKNOWN_OPTION,
    ISMEMBER_ERROR_UNSUPPORTED_INPUT_TYPE, ISMEMBER_INTEGER_CAPABILITIES,
};
use std::collections::HashMap;

use runmat_accelerate_api::{
    GpuTensorHandle, HostLogicalOwned, IsMemberOptions as ProviderIsMemberOptions, IsMemberResult,
};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, LogicalArray, NumericDType, NumericStorage,
    StringArray, StringElement, Tensor, Utf16CodeUnit, Value,
};

use super::float_order::{ComplexSequence, SetFloat};
use crate::build_runtime_error;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::integer_conversion::{IntegerClass, IntegerClassExt};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::ismember")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ismember",
    op_kind: GpuOpKind::Custom("ismember"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("ismember")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may supply dedicated membership kernels; exact typed fallback gathers when needed and restores logical membership plus double locations to the input owner.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::ismember"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ismember",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`ismember` materialises logical outputs and terminates fusion chains; upstream tensors are gathered when necessary.",
};

const BUILTIN_NAME: &str = "ismember";

fn ismember_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ismember_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    ismember_error_with(error, error.message)
}

fn ismember_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    ismember_error_with(&ISMEMBER_ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "ismember",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::sorting_sets::ismember"
)]
async fn ismember_builtin(
    a: Value,
    b: Value,
    rest: BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Err(ismember_error_with(
            &ISMEMBER_ERROR_INVALID_ARGUMENT,
            "ismember: too many output arguments; maximum is 2",
        ));
    }
    let provider = super::set_output_provider(&a, &b);
    let eval = evaluate_view(a, b, rest.view()).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            let outputs = super::restore_set_outputs(
                provider,
                BUILTIN_NAME,
                vec![eval.into_mask_value()],
                ismember_internal_error,
            )?;
            return Ok(Value::output_list(outputs));
        }
        let (mask, loc) = eval.into_pair();
        let outputs = super::restore_set_outputs(
            provider,
            BUILTIN_NAME,
            vec![mask, loc],
            ismember_internal_error,
        )?;
        return Ok(Value::output_list(outputs));
    }
    let mut outputs = super::restore_set_outputs(
        provider,
        BUILTIN_NAME,
        vec![eval.into_mask_value()],
        ismember_internal_error,
    )?;
    Ok(outputs.pop().expect("ismember output"))
}

/// Evaluate the `ismember` builtin once and expose all outputs.
pub async fn evaluate(
    a: Value,
    b: Value,
    rest: &[Value],
) -> crate::BuiltinResult<IsMemberEvaluation> {
    evaluate_view(a, b, FunctionArgumentView::from_slice(rest)).await
}

pub(super) async fn evaluate_view(
    a: Value,
    b: Value,
    rest: FunctionArgumentView<'_>,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    crate::builtins::common::validation::reject_typed_complex_integer(&a, "ismember")?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, "ismember")?;
    let opts = parse_options_view(rest)?;
    for value in [&a, &b] {
        if let Value::GpuTensor(handle) = value {
            if super::is_unsupported_set_gpu_integer(handle) {
                return Err(ismember_error_with(
                    &ISMEMBER_ERROR_UNSUPPORTED_INPUT_TYPE,
                    "ismember: resident 64-bit integer inputs are not supported",
                ));
            }
        }
    }
    match (a, b) {
        (Value::GpuTensor(handle_a), Value::GpuTensor(handle_b)) => {
            ismember_gpu_pair(handle_a, handle_b, &opts).await
        }
        (Value::GpuTensor(handle_a), other) => {
            ismember_gpu_mixed(handle_a, other, &opts, true).await
        }
        (other, Value::GpuTensor(handle_b)) => {
            ismember_gpu_mixed(handle_b, other, &opts, false).await
        }
        (left, right) => ismember_host(left, right, &opts),
    }
}

#[derive(Debug, Clone, Copy)]
struct IsMemberOptions {
    rows: bool,
}

impl IsMemberOptions {
    fn into_provider_options(self) -> ProviderIsMemberOptions {
        ProviderIsMemberOptions { rows: self.rows }
    }
}

#[cfg(test)]
fn parse_options(rest: &[Value]) -> crate::BuiltinResult<IsMemberOptions> {
    parse_options_view(FunctionArgumentView::from_slice(rest))
}

fn parse_options_view(rest: FunctionArgumentView<'_>) -> crate::BuiltinResult<IsMemberOptions> {
    let mut opts = IsMemberOptions { rows: false };
    for arg in rest {
        let text = tensor::value_to_string(arg)
            .ok_or_else(|| ismember_error(&ISMEMBER_ERROR_INVALID_ARGUMENT))?;
        let lowered = text.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "rows" => opts.rows = true,
            "legacy" | "r2012a" => {
                return Err(ismember_error(&ISMEMBER_ERROR_LEGACY_OPTION_UNSUPPORTED))
            }
            other => {
                return Err(ismember_error_with(
                    &ISMEMBER_ERROR_UNKNOWN_OPTION,
                    format!("ismember: unrecognised option '{other}'"),
                ))
            }
        }
    }
    Ok(opts)
}

async fn ismember_gpu_pair(
    handle_a: GpuTensorHandle,
    handle_b: GpuTensorHandle,
    opts: &IsMemberOptions,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if super::provider_result::real_pair(&handle_a, &handle_b) {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle_a) {
            let provider_opts = opts.into_provider_options();
            if let Some(result) = super::provider_result::attempt(
                provider.ismember(&handle_a, &handle_b, &provider_opts),
                ismember_internal_error,
            )
            .await?
            {
                let (mask, loc) = super::provider_result::membership(
                    result,
                    &handle_a,
                    &handle_b,
                    opts.rows,
                    ismember_internal_error,
                )?;
                return Ok(IsMemberEvaluation::new(mask, loc));
            }
        }
    }
    let a = gpu_helpers::gather_value_async(&Value::GpuTensor(handle_a)).await?;
    let b = gpu_helpers::gather_value_async(&Value::GpuTensor(handle_b)).await?;
    ismember_host(a, b, opts)
}

async fn ismember_gpu_mixed(
    handle_gpu: GpuTensorHandle,
    other: Value,
    opts: &IsMemberOptions,
    gpu_is_a: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let gpu = gpu_helpers::gather_value_async(&Value::GpuTensor(handle_gpu)).await?;
    if gpu_is_a {
        ismember_host(gpu, other, opts)
    } else {
        ismember_host(other, gpu, opts)
    }
}

fn ismember_host(
    a: Value,
    b: Value,
    opts: &IsMemberOptions,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    match (a, b) {
        (Value::ComplexTensor(at), Value::ComplexTensor(bt)) => ismember_complex(at, bt, opts.rows),
        (Value::ComplexTensor(at), Value::Complex(re, im)) => {
            let bt = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
            ismember_complex(at, bt, opts.rows)
        }
        (Value::Complex(a_re, a_im), Value::ComplexTensor(bt)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
            ismember_complex(at, bt, opts.rows)
        }
        (Value::Complex(a_re, a_im), Value::Complex(b_re, b_im)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
            let bt = ComplexTensor::new(vec![(b_re, b_im)], vec![1, 1])
                .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
            ismember_complex(at, bt, opts.rows)
        }

        (Value::CharArray(ac), Value::CharArray(bc)) => ismember_char(ac, bc, opts.rows),

        (Value::StringArray(astring), Value::StringArray(bstring)) => {
            ismember_string(astring, bstring, opts.rows)
        }
        (Value::StringArray(astring), Value::String(b)) => {
            let bstring = StringArray::from_elements_unbounded(vec![b], vec![1, 1])
                .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
            ismember_string(astring, bstring, opts.rows)
        }
        (Value::String(a), Value::StringArray(bstring)) => {
            let astring = StringArray::from_elements_unbounded(vec![a], vec![1, 1])
                .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
            ismember_string(astring, bstring, opts.rows)
        }
        (Value::String(a), Value::String(b)) => {
            let astring = StringArray::from_elements_unbounded(vec![a], vec![1, 1])
                .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
            let bstring = StringArray::from_elements_unbounded(vec![b], vec![1, 1])
                .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
            ismember_string(astring, bstring, opts.rows)
        }

        (left, right) => {
            let tensor_a = tensor::value_into_tensor_for("ismember", left)
                .map_err(|e| ismember_internal_error(e))?;
            let tensor_b = tensor::value_into_tensor_for("ismember", right)
                .map_err(|e| ismember_internal_error(e))?;
            ismember_numeric_tensors(tensor_a, tensor_b, opts)
        }
    }
}

fn ismember_numeric_tensors(
    a: Tensor,
    b: Tensor,
    opts: &IsMemberOptions,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let a_dtype = a.numeric_dtype();
    let b_dtype = b.numeric_dtype();
    if let (Some(a_storage), Some(b_storage)) = (a.integer_storage(), b.integer_storage()) {
        if a_storage.numeric_dtype() == b_storage.numeric_dtype() {
            return if opts.rows {
                ismember_integer_rows(&a, &b)
            } else {
                ismember_integer_elements(&a, &b)
            };
        }
        return Err(ismember_error(&ISMEMBER_ERROR_NUMERIC_CLASS_MISMATCH));
    }
    match (a.integer_storage(), b.integer_storage()) {
        (Some(storage), None) if b_dtype == NumericDType::F64 => {
            let target = IntegerClass::from_storage(storage);
            let b = target.cast_tensor(b).map_err(ismember_internal_error)?;
            return ismember_numeric_tensors(a, b, opts);
        }
        (None, Some(storage)) if a_dtype == NumericDType::F64 => {
            let target = IntegerClass::from_storage(storage);
            let a = target.cast_tensor(a).map_err(ismember_internal_error)?;
            return ismember_numeric_tensors(a, b, opts);
        }
        _ => {}
    }
    if a_dtype != b_dtype && a_dtype != NumericDType::F64 && b_dtype != NumericDType::F64 {
        return Err(ismember_error(&ISMEMBER_ERROR_NUMERIC_CLASS_MISMATCH));
    }
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    let a_storage = a.into_numeric_storage().map_err(ismember_internal_error)?;
    let b_storage = b.into_numeric_storage().map_err(ismember_internal_error)?;
    match (a_storage, b_storage) {
        (NumericStorage::F64(a), NumericStorage::F64(b)) => {
            ismember_floating(a, a_shape, b, b_shape, opts.rows)
        }
        (NumericStorage::F32(a), NumericStorage::F32(b)) => {
            ismember_floating(a, a_shape, b, b_shape, opts.rows)
        }
        (a, b) => ismember_promoted_f64(a, a_shape, b, b_shape, opts.rows),
    }
}

fn ismember_promoted_f64(
    a: NumericStorage,
    a_shape: Vec<usize>,
    b: NumericStorage,
    b_shape: Vec<usize>,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    ismember_floating(
        a.materialize_f64(),
        a_shape,
        b.materialize_f64(),
        b_shape,
        rows,
    )
}

fn ismember_floating<T: SetFloat>(
    a: Vec<T>,
    a_shape: Vec<usize>,
    b: Vec<T>,
    b_shape: Vec<usize>,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if rows {
        ismember_floating_rows(a, a_shape, b, b_shape)
    } else {
        ismember_floating_elements(a, a_shape, b)
    }
}

fn ismember_integer_elements(a: &Tensor, b: &Tensor) -> crate::BuiltinResult<IsMemberEvaluation> {
    let a_values = a.integer_storage().expect("integer path").exact_values();
    let b_values = b.integer_storage().expect("integer path").exact_values();
    let mut map = HashMap::<IntValue, usize>::new();
    for (index, value) in b_values.into_iter().enumerate() {
        map.entry(value).or_insert(index + 1);
    }
    let mut mask = Vec::with_capacity(a_values.len());
    let mut locations = Vec::with_capacity(a_values.len());
    for value in a_values {
        if let Some(&index) = map.get(&value) {
            mask.push(1);
            locations.push(index as f64);
        } else {
            mask.push(0);
            locations.push(0.0);
        }
    }
    let logical = LogicalArray::new(mask, a.shape().to_vec())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let locations = Tensor::new(locations, a.shape().to_vec())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, locations))
}

fn ismember_integer_rows(a: &Tensor, b: &Tensor) -> crate::BuiltinResult<IsMemberEvaluation> {
    let (rows_a, cols_a) = tensor_rows_cols(a, "ismember")?;
    let (rows_b, cols_b) = tensor_rows_cols(b, "ismember")?;
    if cols_a != cols_b {
        return Err(ismember_error(&ISMEMBER_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let a_values = a.integer_storage().expect("integer path").exact_values();
    let b_values = b.integer_storage().expect("integer path").exact_values();
    let mut map = HashMap::<Vec<IntValue>, usize>::new();
    for row in 0..rows_b {
        let key: Vec<_> = (0..cols_b)
            .map(|col| b_values[row + col * rows_b].clone())
            .collect();
        map.entry(key).or_insert(row + 1);
    }
    let mut mask = vec![0; rows_a];
    let mut locations = vec![0.0; rows_a];
    for row in 0..rows_a {
        let key: Vec<_> = (0..cols_a)
            .map(|col| a_values[row + col * rows_a].clone())
            .collect();
        if let Some(&index) = map.get(&key) {
            mask[row] = 1;
            locations[row] = index as f64;
        }
    }
    let shape = vec![rows_a, 1];
    let logical = LogicalArray::new(mask, shape.clone())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let locations = Tensor::new(locations, shape)
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, locations))
}

/// Helper exposed for acceleration providers handling numeric tensors on the host.
pub fn ismember_numeric_from_tensors(
    a: Tensor,
    b: Tensor,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let opts = IsMemberOptions { rows };
    ismember_numeric_tensors(a, b, &opts)
}

#[cfg(test)]
fn ismember_numeric_elements(a: Tensor, b: Tensor) -> crate::BuiltinResult<IsMemberEvaluation> {
    ismember_numeric_tensors(a, b, &IsMemberOptions { rows: false })
}

fn ismember_floating_elements<T: SetFloat>(
    a_values: Vec<T>,
    a_shape: Vec<usize>,
    b_values: Vec<T>,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let mut map: HashMap<ElementKey, usize> = HashMap::new();
    for (idx, &value) in b_values.iter().enumerate() {
        map.entry(ElementKey::real(value, Operand::B, idx))
            .or_insert(idx + 1);
    }

    let mut mask_data = Vec::<u8>::with_capacity(a_values.len());
    let mut loc_data = Vec::<f64>::with_capacity(a_values.len());

    for (index, &value) in a_values.iter().enumerate() {
        let key = ElementKey::real(value, Operand::A, index);
        if let Some(&pos) = map.get(&key) {
            mask_data.push(1);
            loc_data.push(pos as f64);
        } else {
            mask_data.push(0);
            loc_data.push(0.0);
        }
    }

    let logical = LogicalArray::new(mask_data, a_shape.clone())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, a_shape)
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

#[cfg(test)]
fn ismember_numeric_rows(a: Tensor, b: Tensor) -> crate::BuiltinResult<IsMemberEvaluation> {
    ismember_numeric_tensors(a, b, &IsMemberOptions { rows: true })
}

fn ismember_floating_rows<T: SetFloat>(
    a_values: Vec<T>,
    a_shape: Vec<usize>,
    b_values: Vec<T>,
    b_shape: Vec<usize>,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let (rows_a, cols_a) = shape_rows_cols(&a_shape, "ismember")?;
    let (rows_b, cols_b) = shape_rows_cols(&b_shape, "ismember")?;
    if cols_a != cols_b {
        return Err(ismember_error(&ISMEMBER_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let mut map: HashMap<FloatingRowKey, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols_b);
        for c in 0..cols_b {
            let idx = r + c * rows_b;
            row_values.push(b_values[idx]);
        }
        let key = FloatingRowKey::from_slice(&row_values, Operand::B, r);
        map.entry(key).or_insert(r + 1);
    }

    let mut mask_data = vec![0u8; rows_a];
    let mut loc_data = vec![0.0f64; rows_a];

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols_a);
        for c in 0..cols_a {
            let idx = r + c * rows_a;
            row_values.push(a_values[idx]);
        }
        let key = FloatingRowKey::from_slice(&row_values, Operand::A, r);
        if let Some(&pos) = map.get(&key) {
            mask_data[r] = 1;
            loc_data[r] = pos as f64;
        }
    }

    let shape = vec![rows_a, 1];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, shape)
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_complex(
    a: ComplexTensor,
    b: ComplexTensor,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let (a_storage, a_shape) = a.into_complex_storage_and_shape();
    let (b_storage, b_shape) = b.into_complex_storage_and_shape();
    match (a_storage, b_storage) {
        (ComplexStorage::F64(a), ComplexStorage::F64(b)) => {
            ismember_floating_complex(a, a_shape, b, b_shape, rows)
        }
        (ComplexStorage::F32(a), ComplexStorage::F32(b)) => {
            ismember_floating_complex(a, a_shape, b, b_shape, rows)
        }
        (a, b) => ismember_promoted_complex_f64(a, a_shape, b, b_shape, rows),
    }
}

fn ismember_promoted_complex_f64(
    a: ComplexStorage,
    a_shape: runmat_value::SharedShape,
    b: ComplexStorage,
    b_shape: runmat_value::SharedShape,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    ismember_floating_complex(
        a.materialize_f64(),
        a_shape,
        b.materialize_f64(),
        b_shape,
        rows,
    )
}

fn ismember_floating_complex<T: SetFloat, A: ComplexSequence<T>, B: ComplexSequence<T>>(
    a: A,
    a_shape: runmat_value::SharedShape,
    b: B,
    b_shape: runmat_value::SharedShape,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if rows {
        ismember_floating_complex_rows(a, a_shape, b, b_shape)
    } else {
        ismember_floating_complex_elements(a, a_shape, b)
    }
}

#[cfg(test)]
fn ismember_complex_elements(
    a: ComplexTensor,
    b: ComplexTensor,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    ismember_complex(a, b, false)
}

fn ismember_floating_complex_elements<T: SetFloat, A: ComplexSequence<T>, B: ComplexSequence<T>>(
    a: A,
    a_shape: runmat_value::SharedShape,
    b: B,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let mut map: HashMap<ComplexKey, usize> = HashMap::new();
    for idx in 0..b.len() {
        let value = b.pair_at(idx);
        map.entry(ElementKey::complex(value, Operand::B, idx))
            .or_insert(idx + 1);
    }

    let mut mask_data = Vec::<u8>::with_capacity(a.len());
    let mut loc_data = Vec::<f64>::with_capacity(a.len());

    for index in 0..a.len() {
        let value = a.pair_at(index);
        let key = ElementKey::complex(value, Operand::A, index);
        if let Some(&pos) = map.get(&key) {
            mask_data.push(1);
            loc_data.push(pos as f64);
        } else {
            mask_data.push(0);
            loc_data.push(0.0);
        }
    }

    let logical = LogicalArray::new(mask_data, a_shape.clone())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, a_shape)
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

#[cfg(test)]
fn ismember_complex_rows(
    a: ComplexTensor,
    b: ComplexTensor,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    ismember_complex(a, b, true)
}

fn ismember_floating_complex_rows<T: SetFloat, A: ComplexSequence<T>, B: ComplexSequence<T>>(
    a: A,
    a_shape: runmat_value::SharedShape,
    b: B,
    b_shape: runmat_value::SharedShape,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let (rows_a, cols_a) = shape_rows_cols(&a_shape, "ismember")?;
    let (rows_b, cols_b) = shape_rows_cols(&b_shape, "ismember")?;
    if cols_a != cols_b {
        return Err(ismember_error(&ISMEMBER_ERROR_ROWS_COLUMN_MISMATCH).into());
    }

    let mut map: HashMap<Vec<ComplexKey>, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_keys = Vec::with_capacity(cols_b);
        for c in 0..cols_b {
            let idx = r + c * rows_b;
            row_keys.push(ElementKey::complex(b.pair_at(idx), Operand::B, r));
        }
        map.entry(row_keys).or_insert(r + 1);
    }

    let mut mask_data = vec![0u8; rows_a];
    let mut loc_data = vec![0.0f64; rows_a];

    for r in 0..rows_a {
        let mut row_keys = Vec::with_capacity(cols_a);
        for c in 0..cols_a {
            let idx = r + c * rows_a;
            row_keys.push(ElementKey::complex(a.pair_at(idx), Operand::A, r));
        }
        if let Some(&pos) = map.get(&row_keys) {
            mask_data[r] = 1;
            loc_data[r] = pos as f64;
        }
    }

    let shape = vec![rows_a, 1];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, shape)
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_char(
    a: CharArray,
    b: CharArray,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if rows {
        ismember_char_rows(a, b)
    } else {
        ismember_char_elements(a, b)
    }
}

fn ismember_char_elements(a: CharArray, b: CharArray) -> crate::BuiltinResult<IsMemberEvaluation> {
    let rows_b = b.rows;
    let cols_b = b.cols;
    let mut map: HashMap<Utf16CodeUnit, usize> = HashMap::new();

    for col in 0..cols_b {
        for row in 0..rows_b {
            let data_idx = row * cols_b + col;
            let ch = b.data()[data_idx];
            let linear_idx = row + col * rows_b;
            map.entry(ch).or_insert(linear_idx + 1);
        }
    }

    let rows_a = a.rows;
    let cols_a = a.cols;
    let mut mask_data = vec![0u8; rows_a * cols_a];
    let mut loc_data = vec![0.0f64; rows_a * cols_a];

    for col in 0..cols_a {
        for row in 0..rows_a {
            let data_idx = row * cols_a + col;
            let ch = a.data()[data_idx];
            let linear_idx = row + col * rows_a;
            if let Some(&pos) = map.get(&ch) {
                mask_data[linear_idx] = 1;
                loc_data[linear_idx] = pos as f64;
            }
        }
    }

    let shape = vec![rows_a, cols_a];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, shape)
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_char_rows(a: CharArray, b: CharArray) -> crate::BuiltinResult<IsMemberEvaluation> {
    if a.cols != b.cols {
        return Err(ismember_error(&ISMEMBER_ERROR_ROWS_COLUMN_MISMATCH).into());
    }

    let rows_b = b.rows;
    let cols = b.cols;
    let mut map: HashMap<RowCharKey, usize> = HashMap::new();

    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(b.data()[idx]);
        }
        let key = RowCharKey::from_slice(&row_values);
        map.entry(key).or_insert(r + 1);
    }

    let rows_a = a.rows;
    let mut mask_data = vec![0u8; rows_a];
    let mut loc_data = vec![0.0f64; rows_a];

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(a.data()[idx]);
        }
        let key = RowCharKey::from_slice(&row_values);
        if let Some(&pos) = map.get(&key) {
            mask_data[r] = 1;
            loc_data[r] = pos as f64;
        }
    }

    let shape = vec![rows_a, 1];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, shape)
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_string(
    a: StringArray,
    b: StringArray,
    rows: bool,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if rows {
        ismember_string_rows(a, b)
    } else {
        ismember_string_elements(a, b)
    }
}

fn ismember_string_elements(
    a: StringArray,
    b: StringArray,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    let mut map: HashMap<&StringElement, usize> = HashMap::new();
    for (idx, value) in b.data().iter().enumerate() {
        if !value.is_missing() {
            map.entry(value).or_insert(idx + 1);
        }
    }

    let mut mask_data = Vec::<u8>::with_capacity(a.data().len());
    let mut loc_data = Vec::<f64>::with_capacity(a.data().len());

    for value in a.data() {
        if let Some(&pos) = map.get(value) {
            mask_data.push(1);
            loc_data.push(pos as f64);
        } else {
            mask_data.push(0);
            loc_data.push(0.0);
        }
    }

    let logical = LogicalArray::new(mask_data, a.shape.to_vec())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, a.shape.to_vec())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn ismember_string_rows(
    a: StringArray,
    b: StringArray,
) -> crate::BuiltinResult<IsMemberEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(ismember_internal_error(
            "ismember: 'rows' option requires 2-D string arrays",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(ismember_error(&ISMEMBER_ERROR_ROWS_COLUMN_MISMATCH).into());
    }

    let rows_a = a.shape[0];
    let cols = a.shape[1];
    let rows_b = b.shape[0];

    let mut map: HashMap<RowStringKey, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(&b.data()[idx]);
        }
        if row_values.iter().any(|text| text.is_missing()) {
            continue;
        }
        let key = RowStringKey(row_values);
        map.entry(key).or_insert(r + 1);
    }

    let mut mask_data = vec![0u8; rows_a];
    let mut loc_data = vec![0.0f64; rows_a];

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(&a.data()[idx]);
        }
        let key = RowStringKey(row_values);
        if let Some(&pos) = map.get(&key) {
            mask_data[r] = 1;
            loc_data[r] = pos as f64;
        }
    }

    let shape = vec![rows_a, 1];
    let logical = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    let loc_tensor = Tensor::new(loc_data, shape)
        .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
    Ok(IsMemberEvaluation::new(logical, loc_tensor))
}

fn tensor_rows_cols(t: &Tensor, name: &str) -> crate::BuiltinResult<(usize, usize)> {
    shape_rows_cols(t.shape(), name)
}

fn shape_rows_cols(shape: &[usize], name: &str) -> crate::BuiltinResult<(usize, usize)> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((shape[0], 1)),
        2 => Ok((shape[0], shape[1])),
        _ => Err(ismember_internal_error(format!(
            "{name}: 'rows' option requires 2-D numeric matrices"
        ))
        .into()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FloatingRowKey(Vec<ElementKey>);
impl FloatingRowKey {
    fn from_slice<T: SetFloat>(values: &[T], operand: Operand, record: usize) -> Self {
        Self(
            values
                .iter()
                .map(|&value| ElementKey::real(value, operand, record))
                .collect(),
        )
    }
}
type ComplexKey = ElementKey;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowCharKey(Vec<Utf16CodeUnit>);

impl RowCharKey {
    fn from_slice(values: &[Utf16CodeUnit]) -> Self {
        RowCharKey(values.to_vec())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowStringKey<'a>(Vec<&'a StringElement>);

#[derive(Debug, Clone)]
pub struct IsMemberEvaluation {
    mask: LogicalArray,
    loc: Tensor,
}

impl IsMemberEvaluation {
    fn new(mask: LogicalArray, loc: Tensor) -> Self {
        Self { mask, loc }
    }

    pub fn from_provider_result(result: IsMemberResult) -> crate::BuiltinResult<Self> {
        let mask = LogicalArray::new(result.mask.data, result.mask.shape)
            .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
        let loc = Tensor::new(result.loc.data, result.loc.shape)
            .map_err(|e| ismember_internal_error(format!("ismember: {e}")))?;
        Ok(IsMemberEvaluation::new(mask, loc))
    }

    pub fn into_numeric_ismember_result(self) -> crate::BuiltinResult<IsMemberResult> {
        let IsMemberEvaluation { mask, loc } = self;
        Ok(IsMemberResult {
            mask: HostLogicalOwned {
                data: mask.data.into_vec(),
                shape: mask.shape.to_vec(),
            },
            loc: tensor::tensor_into_host_f64_owned(loc),
        })
    }

    pub fn into_mask_value(self) -> Value {
        logical_array_into_value(self.mask)
    }

    pub fn mask_value(&self) -> Value {
        logical_array_into_value(self.mask.clone())
    }

    pub fn into_pair(self) -> (Value, Value) {
        let mask = logical_array_into_value(self.mask);
        let loc = tensor::tensor_into_value(self.loc);
        (mask, loc)
    }

    pub fn loc_value(&self) -> Value {
        tensor::tensor_into_value(self.loc.clone())
    }
}

fn logical_array_into_value(logical: LogicalArray) -> Value {
    if logical.data.len() == 1 {
        Value::Bool(logical.data[0] != 0)
    } else {
        Value::LogicalArray(logical)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntegerStorage, Tensor};

    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    fn evaluate_sync(
        a: Value,
        b: Value,
        rest: &[Value],
    ) -> crate::BuiltinResult<IsMemberEvaluation> {
        futures::executor::block_on(evaluate(a, b, rest))
    }

    fn builtin_sync(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(ismember_builtin(
            a,
            b,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn public_invocation_preserves_rank_zero_rows_and_rank_one_element_geometry() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let rank_zero = Tensor::new(vec![7.0], vec![]).expect("rank-zero input");
        let Value::OutputList(outputs) = builtin_sync(
            Value::Tensor(rank_zero.clone()),
            Value::Tensor(rank_zero),
            vec![Value::from("rows")],
        )
        .expect("rank-zero rows membership") else {
            panic!("expected two outputs");
        };
        assert_eq!(outputs.as_slice(), &[Value::Bool(true), Value::Num(1.0)]);

        let rank_one = Tensor::new(vec![1.0, 2.0], vec![2]).expect("rank-one input");
        let set = Tensor::new(vec![2.0], vec![1]).expect("rank-one set");
        let Value::OutputList(outputs) =
            builtin_sync(Value::Tensor(rank_one), Value::Tensor(set), Vec::new())
                .expect("rank-one element membership")
        else {
            panic!("expected two outputs");
        };
        let Value::LogicalArray(mask) = &outputs[0] else {
            panic!("expected logical rank-one output");
        };
        assert_eq!(mask.shape, vec![2]);
        assert_eq!(mask.data.as_slice(), &[0, 1]);
        let Value::Tensor(locations) = &outputs[1] else {
            panic!("expected rank-one location tensor");
        };
        assert_eq!(locations.shape(), &[2]);
        assert_eq!(locations.materialize_f64(), vec![0.0, 1.0]);
    }

    #[test]
    fn registered_builtin_restores_resident_outputs_and_rejects_excess_arity() {
        test_support::with_test_provider(|provider| {
            let left = Tensor::new_integer(IntegerStorage::I32(vec![7, 2, 9]), vec![3, 1]).unwrap();
            let right = Tensor::new_integer(IntegerStorage::I32(vec![2, 7]), vec![2, 1]).unwrap();
            let left =
                Value::GpuTensor(gpu_helpers::upload_tensor(provider, &left).expect("upload left"));
            let right = Value::GpuTensor(
                gpu_helpers::upload_tensor(provider, &right).expect("upload right"),
            );

            {
                let _guard =
                    crate::output_count::push_output_count(Some(2)).expect("output-count scope");
                let Value::OutputList(outputs) =
                    builtin_sync(left, right, Vec::new()).expect("resident ismember")
                else {
                    panic!("expected output list");
                };
                assert_eq!(outputs.len(), 2);
                let Value::GpuTensor(mask) = &outputs[0] else {
                    panic!("expected resident membership mask");
                };
                assert!(runmat_accelerate_api::handle_is_logical(mask));
                assert!(matches!(outputs[1], Value::GpuTensor(_)));
                assert_eq!(
                    test_support::gather(outputs[0].clone())
                        .expect("gather mask")
                        .materialize_f64(),
                    vec![1.0, 1.0, 0.0]
                );
                assert_eq!(
                    test_support::gather(outputs[1].clone())
                        .expect("gather locations")
                        .materialize_f64(),
                    vec![2.0, 1.0, 0.0]
                );
            }

            let _guard =
                crate::output_count::push_output_count(Some(3)).expect("output-count scope");
            let err = builtin_sync(Value::Num(1.0), Value::Num(1.0), Vec::new())
                .expect_err("excess outputs must fail");
            assert_eq!(err.identifier(), ISMEMBER_ERROR_INVALID_ARGUMENT.identifier);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_membership_basic() {
        let a = Tensor::new(vec![5.0, 7.0, 2.0, 7.0], vec![1, 4]).unwrap();
        let b = Tensor::new(vec![7.0, 9.0, 5.0], vec![1, 3]).unwrap();
        let eval = ismember_numeric_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1, 0, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![3.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn numeric_membership_uses_native_single_elements_and_rows() {
        let a = Tensor::from_f32(vec![1.0, 2.0, f32::NAN], vec![3, 1]).unwrap();
        let b = Tensor::from_f32(vec![2.0, f32::NAN], vec![2, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("single ismember");
        assert_eq!(eval.mask.data, vec![0, 1, 0]);
        assert_eq!(eval.loc.materialize_f64(), vec![0.0, 1.0, 0.0]);

        let a = Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_f32(vec![3.0, 5.0, 4.0, 6.0], vec![2, 2]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
            .expect("single row ismember");
        assert_eq!(eval.mask.data, vec![0, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![0.0, 1.0]);
    }

    #[test]
    fn integer_membership_uses_exact_values_for_elements_and_rows() {
        let a = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993]),
            vec![3, 1],
        )
        .expect("input");
        let b = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![0, u64::MAX]),
            vec![2, 1],
        )
        .expect("input");
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1, 0]);
        assert_eq!(eval.loc.materialize_f64(), vec![2.0, 1.0, 0.0]);

        let a = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![i64::MAX, i64::MIN, 1, 2]),
            vec![2, 2],
        )
        .expect("input");
        let b = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![i64::MIN, 7, 2, 8]),
            vec![2, 2],
        )
        .expect("input");
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
            .expect("ismember rows");
        assert_eq!(eval.mask.data, vec![0, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![0.0, 1.0]);
    }

    #[test]
    fn mixed_integer_membership_rejects_nondouble_class_mismatch() {
        let a = Tensor::new_integer(
            runmat_value::IntegerStorage::U16(vec![7, 2, 9, 7]),
            vec![4, 1],
        )
        .expect("input");
        let b = Tensor::new_integer(runmat_value::IntegerStorage::I32(vec![2, 7]), vec![2, 1])
            .expect("input");

        let error = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect_err("mixed integer classes must reject");
        assert_eq!(
            error.identifier(),
            ISMEMBER_ERROR_NUMERIC_CLASS_MISMATCH.identifier
        );
    }

    #[test]
    fn resident_integer_set_functions_use_exact_runtime_fallback_and_class_rules() {
        test_support::with_test_provider(|provider| {
            let left = Tensor::new_integer(
                runmat_value::IntegerStorage::I32(vec![7, 2, 9, 7]),
                vec![4, 1],
            )
            .unwrap();
            let right =
                Tensor::new_integer(runmat_value::IntegerStorage::I32(vec![2, 7]), vec![2, 1])
                    .unwrap();
            let left =
                Value::GpuTensor(gpu_helpers::upload_tensor(provider, &left).expect("upload left"));
            let right = Value::GpuTensor(
                gpu_helpers::upload_tensor(provider, &right).expect("upload right"),
            );

            let member = futures::executor::block_on(evaluate(left.clone(), right.clone(), &[]))
                .expect("resident integer ismember");
            assert_eq!(member.mask.data, vec![1, 1, 0, 1]);

            for (builtin, result) in [
                (
                    "intersect",
                    futures::executor::block_on(super::super::intersect::evaluate(
                        left.clone(),
                        right.clone(),
                        &[],
                    ))
                    .map(|eval| eval.values_value()),
                ),
                (
                    "union",
                    futures::executor::block_on(super::super::union::evaluate(
                        left.clone(),
                        right.clone(),
                        &[],
                    ))
                    .map(|eval| eval.values_value()),
                ),
                (
                    "setdiff",
                    futures::executor::block_on(super::super::setdiff::evaluate(
                        left.clone(),
                        right.clone(),
                        &[],
                    ))
                    .map(|eval| eval.values_value()),
                ),
                (
                    "setxor",
                    futures::executor::block_on(super::super::setxor::evaluate(
                        left.clone(),
                        right.clone(),
                        &[],
                    ))
                    .map(|eval| eval.values_value()),
                ),
            ] {
                let value = result.unwrap_or_else(|error| panic!("{builtin}: {error}"));
                let Value::Tensor(tensor) = value else {
                    panic!("{builtin}: expected integer tensor")
                };
                assert_eq!(
                    tensor.integer_storage().map(|storage| storage.class_name()),
                    Some("int32"),
                    "{builtin}"
                );
            }

            let mismatched =
                Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![2, 7]), vec![2, 1])
                    .unwrap();
            let mismatched =
                Value::GpuTensor(gpu_helpers::upload_tensor(provider, &mismatched).unwrap());
            for (builtin, error) in [
                (
                    "ismember",
                    futures::executor::block_on(evaluate(left.clone(), mismatched.clone(), &[]))
                        .expect_err("mismatch"),
                ),
                (
                    "intersect",
                    futures::executor::block_on(super::super::intersect::evaluate(
                        left.clone(),
                        mismatched.clone(),
                        &[],
                    ))
                    .expect_err("mismatch"),
                ),
                (
                    "union",
                    futures::executor::block_on(super::super::union::evaluate(
                        left.clone(),
                        mismatched.clone(),
                        &[],
                    ))
                    .expect_err("mismatch"),
                ),
                (
                    "setdiff",
                    futures::executor::block_on(super::super::setdiff::evaluate(
                        left.clone(),
                        mismatched.clone(),
                        &[],
                    ))
                    .expect_err("mismatch"),
                ),
                (
                    "setxor",
                    futures::executor::block_on(super::super::setxor::evaluate(
                        left.clone(),
                        mismatched,
                        &[],
                    ))
                    .expect_err("mismatch"),
                ),
            ] {
                assert!(
                    error
                        .identifier()
                        .is_some_and(|identifier| identifier.ends_with(":NumericClassMismatch")),
                    "{builtin}: {error}"
                );
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn resident_i32_set_functions_use_exact_wgpu_runtime_fallback() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let left = Tensor::new_integer(
            runmat_value::IntegerStorage::I32(vec![7, 2, 9, 7]),
            vec![4, 1],
        )
        .unwrap();
        let right =
            Tensor::new_integer(runmat_value::IntegerStorage::I32(vec![2, 7]), vec![2, 1]).unwrap();
        let left =
            Value::GpuTensor(gpu_helpers::upload_tensor(provider, &left).expect("upload left"));
        let right =
            Value::GpuTensor(gpu_helpers::upload_tensor(provider, &right).expect("upload right"));

        let member = futures::executor::block_on(evaluate(left.clone(), right.clone(), &[]))
            .expect("wgpu integer ismember");
        assert_eq!(member.mask.data, vec![1, 1, 0, 1]);

        for (builtin, result) in [
            (
                "intersect",
                futures::executor::block_on(super::super::intersect::evaluate(
                    left.clone(),
                    right.clone(),
                    &[],
                ))
                .map(|eval| eval.values_value()),
            ),
            (
                "union",
                futures::executor::block_on(super::super::union::evaluate(
                    left.clone(),
                    right.clone(),
                    &[],
                ))
                .map(|eval| eval.values_value()),
            ),
            (
                "setdiff",
                futures::executor::block_on(super::super::setdiff::evaluate(
                    left.clone(),
                    right.clone(),
                    &[],
                ))
                .map(|eval| eval.values_value()),
            ),
            (
                "setxor",
                futures::executor::block_on(super::super::setxor::evaluate(left, right, &[]))
                    .map(|eval| eval.values_value()),
            ),
        ] {
            let value = result.unwrap_or_else(|error| panic!("{builtin}: {error}"));
            let Value::Tensor(tensor) = value else {
                panic!("{builtin}: expected integer tensor")
            };
            assert_eq!(
                tensor.integer_storage().map(|storage| storage.class_name()),
                Some("int32"),
                "{builtin}"
            );
        }
    }

    #[test]
    fn ismember_type_resolver_logical() {
        assert_eq!(
            logical_output_type(
                &[Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new()),
            ),
            Type::logical()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_nan_membership() {
        let a = Tensor::new(vec![f64::NAN, 1.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![f64::NAN, 2.0], vec![1, 2]).unwrap();
        let eval = ismember_numeric_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![0, 0]);
        assert_eq!(eval.loc.materialize_f64(), vec![0.0, 0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_rows_membership() {
        let a = Tensor::new(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 5.0, 1.0, 4.0, 6.0, 2.0], vec![3, 2]).unwrap();
        let eval = ismember_numeric_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![3.0, 1.0, 3.0]);
        assert_eq!(eval.loc.shape(), &[3, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_membership() {
        let a = ComplexTensor::new(vec![(1.0, 2.0), (0.0, 0.0)], vec![1, 2]).unwrap();
        let b = ComplexTensor::new(vec![(0.0, 0.0), (1.0, 2.0)], vec![1, 2]).unwrap();
        let eval = ismember_complex_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![2.0, 1.0]);
    }

    #[test]
    fn complex_membership_uses_native_single_elements_and_rows() {
        let a = ComplexTensor::from_f32(vec![(1.0, 1.0), (2.0, 0.0)], vec![2, 1]).unwrap();
        let b = ComplexTensor::from_f32(vec![(2.0, 0.0), (1.0, 1.0)], vec![2, 1]).unwrap();
        let eval = evaluate_sync(Value::ComplexTensor(a), Value::ComplexTensor(b), &[])
            .expect("complex single ismember");
        assert_eq!(eval.mask.data, vec![1, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![2.0, 1.0]);

        let a = ComplexTensor::from_f32(
            vec![(1.0, 0.0), (3.0, 0.0), (2.0, 1.0), (4.0, 1.0)],
            vec![2, 2],
        )
        .unwrap();
        let b = ComplexTensor::from_f32(
            vec![(3.0, 0.0), (5.0, 0.0), (4.0, 1.0), (6.0, 1.0)],
            vec![2, 2],
        )
        .unwrap();
        let eval = evaluate_sync(
            Value::ComplexTensor(a),
            Value::ComplexTensor(b),
            &[Value::from("rows")],
        )
        .expect("complex single row ismember");
        assert_eq!(eval.mask.data, vec![0, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![0.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_rows_membership() {
        let a = ComplexTensor::new(
            vec![(1.0, 1.0), (3.0, 0.0), (2.0, 0.0), (4.0, 4.0)],
            vec![2, 2],
        )
        .unwrap();
        let b = ComplexTensor::new(
            vec![
                (1.0, 1.0),
                (5.0, 0.0),
                (3.0, 0.0),
                (2.0, 0.0),
                (6.0, 0.0),
                (4.0, 4.0),
            ],
            vec![3, 2],
        )
        .unwrap();
        let eval = ismember_complex_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![1.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_membership() {
        let a = CharArray::from_unicode_scalars(vec!['r', 'u', 'n', 'm'], 2, 2).unwrap();
        let b = CharArray::from_unicode_scalars(vec!['m', 'a', 'r', 'u'], 2, 2).unwrap();
        let eval = ismember_char_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 0, 1, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![2.0, 0.0, 4.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rows_membership() {
        let a = CharArray::from_unicode_scalars(vec!['m', 'a', 't', 'l'], 2, 2).unwrap();
        let b = CharArray::from_unicode_scalars(vec!['m', 'a', 'g', 'e', 't', 'l'], 3, 2).unwrap();
        let eval = ismember_char_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![1.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_membership() {
        let a = StringArray::new(
            vec![
                "apple".to_string(),
                "pear".to_string(),
                "banana".to_string(),
            ],
            vec![1, 3],
        )
        .unwrap();
        let b = StringArray::new(
            vec![
                "pear".to_string(),
                "orange".to_string(),
                "apple".to_string(),
            ],
            vec![1, 3],
        )
        .unwrap();
        let eval = ismember_string_elements(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1, 0]);
        assert_eq!(eval.loc.materialize_f64(), vec![3.0, 1.0, 0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_rows_membership() {
        let a = StringArray::new(
            vec![
                "alpha".to_string(),
                "gamma".to_string(),
                "beta".to_string(),
                "delta".to_string(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let b = StringArray::new(
            vec![
                "alpha".to_string(),
                "theta".to_string(),
                "gamma".to_string(),
                "beta".to_string(),
                "eta".to_string(),
                "delta".to_string(),
            ],
            vec![3, 2],
        )
        .unwrap();
        let eval = ismember_string_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![1, 1]);
        assert_eq!(eval.loc.materialize_f64(), vec![1.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn options_reject_legacy() {
        let err = parse_options(&[Value::from("legacy")]).unwrap_err();
        assert_eq!(
            err.identifier(),
            ISMEMBER_ERROR_LEGACY_OPTION_UNSUPPORTED.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unknown_option() {
        let err =
            evaluate_sync(Value::Num(1.0), Value::Num(1.0), &[Value::from("stable")]).unwrap_err();
        assert_eq!(err.identifier(), ISMEMBER_ERROR_UNKNOWN_OPTION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismember_runtime_numeric() {
        let a = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap());
        let (mask, loc) = evaluate_sync(a, b, &[]).unwrap().into_pair();
        match mask {
            Value::LogicalArray(arr) => assert_eq!(arr.data, vec![1, 0, 1]),
            other => panic!("expected logical array, got {other:?}"),
        }
        match loc {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![2.0, 0.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_inputs_promoted() {
        let a = Value::Bool(true);
        let logical_b =
            LogicalArray::new(vec![1, 0], vec![2, 1]).expect("logical array construction");
        let eval = evaluate_sync(a, Value::LogicalArray(logical_b), &[]).expect("ismember");
        assert_eq!(eval.mask_value(), Value::Bool(true));
        assert_eq!(eval.loc_value(), Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismember_rows_shape_checks() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        assert!(ismember_numeric_rows(a.clone(), b.clone()).is_ok());
        let bad = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = ismember_numeric_rows(a, bad).unwrap_err();
        assert_eq!(
            err.identifier(),
            ISMEMBER_ERROR_ROWS_COLUMN_MISMATCH.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismember_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 4.0], vec![4, 1]).unwrap();
            let set = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
            let view_a = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let view_b = runmat_accelerate_api::HostTensorView {
                data: &set.materialize_f64(),
                shape: set.shape(),
            };
            let handle_a = provider.upload(&view_a).expect("upload a");
            let handle_b = provider.upload(&view_b).expect("upload b");
            let eval = evaluate_sync(Value::GpuTensor(handle_a), Value::GpuTensor(handle_b), &[])
                .expect("ismember");
            assert_eq!(eval.mask.data, vec![0, 1, 0, 1]);
            assert_eq!(eval.loc.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismember_gpu_rows_roundtrip() {
        test_support::with_test_provider(|provider| {
            let rows = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let bank = Tensor::new(vec![1.0, 5.0, 3.0, 2.0, 6.0, 4.0], vec![3, 2]).unwrap();
            let view_a = runmat_accelerate_api::HostTensorView {
                data: &rows.materialize_f64(),
                shape: rows.shape(),
            };
            let view_b = runmat_accelerate_api::HostTensorView {
                data: &bank.materialize_f64(),
                shape: bank.shape(),
            };
            let handle_a = provider.upload(&view_a).expect("upload a");
            let handle_b = provider.upload(&view_b).expect("upload b");
            let eval = evaluate_sync(
                Value::GpuTensor(handle_a.clone()),
                Value::GpuTensor(handle_b.clone()),
                &[Value::from("rows")],
            )
            .expect("ismember");
            assert_eq!(eval.mask.data, vec![1, 1]);
            assert_eq!(eval.loc.materialize_f64(), vec![1.0, 3.0]);
            drop(handle_a);
            drop(handle_b);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ismember_wgpu_numeric_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 4.0], vec![4, 1]).unwrap();
        let set = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let cpu_eval =
            ismember_numeric_from_tensors(tensor.clone(), set.clone(), false).expect("cpu");

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view_a = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let view_b = HostTensorView {
            data: &set.materialize_f64(),
            shape: set.shape(),
        };
        let handle_a = provider.upload(&view_a).expect("upload a");
        let handle_b = provider.upload(&view_b).expect("upload b");

        let eval = evaluate_sync(
            Value::GpuTensor(handle_a.clone()),
            Value::GpuTensor(handle_b.clone()),
            &[],
        )
        .expect("gpu evaluate");
        assert_eq!(eval.mask.data, cpu_eval.mask.data);
        assert_eq!(eval.loc.materialize_f64(), cpu_eval.loc.materialize_f64());

        drop(handle_a);
        drop(handle_b);

        let matrix = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let bank = Tensor::new(vec![1.0, 7.0, 3.0, 2.0, 9.0, 4.0], vec![3, 2]).unwrap();
        let cpu_rows =
            ismember_numeric_from_tensors(matrix.clone(), bank.clone(), true).expect("cpu rows");
        let view_matrix = HostTensorView {
            data: &matrix.materialize_f64(),
            shape: matrix.shape(),
        };
        let view_bank = HostTensorView {
            data: &bank.materialize_f64(),
            shape: bank.shape(),
        };
        let handle_matrix = provider.upload(&view_matrix).expect("upload matrix");
        let handle_bank = provider.upload(&view_bank).expect("upload bank");
        let eval_rows = evaluate_sync(
            Value::GpuTensor(handle_matrix.clone()),
            Value::GpuTensor(handle_bank.clone()),
            &[Value::from("rows")],
        )
        .expect("gpu rows evaluate");
        assert_eq!(eval_rows.mask.data, cpu_rows.mask.data);
        assert_eq!(
            eval_rows.loc.materialize_f64(),
            cpu_rows.loc.materialize_f64()
        );
        drop(handle_matrix);
        drop(handle_bank);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_return_is_bool() {
        let a = Value::Tensor(Tensor::new(vec![7.0], vec![1, 1]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![7.0], vec![1, 1]).unwrap());
        let mask = evaluate_sync(a, b, &[]).unwrap().into_mask_value();
        assert_eq!(mask, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_rows_option() {
        let opts = parse_options(&[Value::from("rows")]).unwrap();
        assert!(opts.rows);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_rows_with_nan() {
        let a = Tensor::new(vec![f64::NAN, 1.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![f64::NAN, 2.0], vec![2, 1]).unwrap();
        let eval = ismember_numeric_rows(a, b).expect("ismember");
        assert_eq!(eval.mask.data, vec![0, 0]);
        assert_eq!(eval.loc.materialize_f64(), vec![0.0, 0.0]);
    }
}
