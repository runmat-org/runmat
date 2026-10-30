//! MATLAB-compatible `cat` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::concatenation::{
    char_array_from_f64_with_prefix, char_from_f64_code_unit,
};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::integer_conversion::{integer_values, IntegerClass, IntegerClassExt};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use crate::call::{BuiltinArguments, FunctionArgumentView};
use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::AccelProvider;
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntegerComplexStorage, LogicalArray, NumericDType,
    NumericStorage, StringArray, Tensor, Value,
};

fn cat_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("cat").build()
}

const MAX_CAT_DIMENSION: usize = 1024;
const BUILTIN_NAME: &str = "cat";

mod managed;
pub(crate) mod symbolic;

fn is_like_name(value: &Value) -> bool {
    match value {
        Value::String(text) => text
            .as_str()
            .is_some_and(|text| text.eq_ignore_ascii_case("like")),
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .as_str()
            .is_some_and(|text| text.eq_ignore_ascii_case("like")),
        Value::CharArray(array) if array.rows == 1 && array.data().len() == 4 => array
            .data()
            .iter()
            .copied()
            .zip(b"like")
            .all(|(actual, expected)| {
                u8::try_from(actual.raw()).is_ok_and(|actual| actual.eq_ignore_ascii_case(expected))
            }),
        _ => false,
    }
}

fn cat_err(message: impl Into<String>) -> RuntimeError {
    cat_error(message)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CatCategory {
    Numeric,
    Logical,
    Complex,
    Char,
    String,
    Cell,
    Symbolic,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LikeDevice {
    Host,
    Gpu,
}

#[derive(Clone, Debug)]
struct LikeSpec {
    device: LikeDevice,
    category_hint: Option<CatCategory>,
    explicit: bool,
}

impl Default for LikeSpec {
    fn default() -> Self {
        Self {
            device: LikeDevice::Host,
            category_hint: None,
            explicit: false,
        }
    }
}

impl LikeSpec {
    fn from_prototype(proto: &Value) -> BuiltinResult<Self> {
        match proto {
            Value::GpuTensor(_) => Ok(Self {
                device: LikeDevice::Gpu,
                category_hint: Some(CatCategory::Numeric),
                explicit: true,
            }),
            Value::Tensor(_) | Value::Num(_) | Value::Int(_) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Numeric),
                explicit: true,
            }),
            Value::LogicalArray(_) | Value::Bool(_) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Logical),
                explicit: true,
            }),
            Value::ComplexTensor(_) | Value::Complex(_, _) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Complex),
                explicit: true,
            }),
            other => Err(cat_err(format!(
                "cat: unsupported prototype for 'like' ({other:?}); provide a numeric or gpuArray prototype"
            ))),
        }
    }

    fn ensure_device(&self, category: CatCategory) -> BuiltinResult<()> {
        if matches!(self.device, LikeDevice::Gpu)
            && !matches!(category, CatCategory::Numeric | CatCategory::Complex)
        {
            return Err(cat_err(
                "cat: GPU 'like' prototypes are only supported for numeric or complex inputs",
            ));
        }
        Ok(())
    }
}

fn extract_like(
    mut inputs: crate::call::BuiltinArguments,
) -> BuiltinResult<(crate::call::BuiltinArguments, LikeSpec)> {
    if inputs.len() >= 2 && is_like_name(&inputs[inputs.len() - 2]) {
        let prototype = inputs.last().expect("nonempty like suffix");
        if matches!(
            prototype,
            Value::CharArray(_) | Value::String(_) | Value::StringArray(_) | Value::Cell(_)
        ) {
            // Treat as data to avoid colliding with textual concatenation cases.
        } else if inputs.len() < 4 {
            // Removing the pair would leave fewer than two inputs; treat as data.
        } else {
            let spec = LikeSpec::from_prototype(prototype)?;
            inputs.pop();
            inputs.pop();
            return Ok((inputs, spec));
        }
    }
    Ok((inputs, LikeSpec::default()))
}

#[runtime_builtin(
    name = "cat",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::array::shape::cat"
)]
async fn cat_builtin(
    context: Option<crate::context::RuntimeContext>,
    dim: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if rest.len() < 2 {
        return Err(cat_err(CAT_ERROR_TOO_FEW_INPUTS.message));
    }
    let (inputs, like) = extract_like(rest)?;
    if inputs.len() < 2 {
        return Err(cat_err(CAT_ERROR_TOO_FEW_INPUTS.message));
    }
    if like.explicit {
        crate::compatibility::ensure_builtin_extension_enabled(&CAT_LIKE_EXTENSION, BUILTIN_NAME)?;
    }
    if matches!(dim, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CAT_RESIDENT_DIMENSION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let dim_index = match dim {
        Value::Int(_) | Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_) => {
            match tensor::dimension_from_value_async(&dim, "cat", false)
                .await
                .map_err(cat_err)?
            {
                Some(index) => index,
                None => {
                    return Err(cat_err(format!(
                        "cat: dimension must be numeric, got {:?}",
                        dim
                    )))
                }
            }
        }
        _ => {
            return Err(cat_err(format!(
                "cat: dimension must be numeric, got {:?}",
                dim
            )))
        }
    };
    if dim_index > MAX_CAT_DIMENSION {
        return Err(cat_err(format!(
            "cat: dimension must be <= {MAX_CAT_DIMENSION}"
        )));
    }
    let dim_zero = dim_index - 1;

    if inputs
        .iter()
        .any(|value| matches!(value, Value::Struct(_) | Value::StructArray(_)))
    {
        let has_nonempty_structure = inputs.iter().any(|value| {
            matches!(value, Value::Struct(_))
                || matches!(value, Value::StructArray(array) if !array.is_empty())
        });
        return crate::aggregate::structure::operations::concatenate_arguments(
            dim_zero,
            &inputs,
            has_nonempty_structure,
            context.as_ref(),
        )
        .map_err(|error| {
            crate::aggregate::structure::operations::remap(error, |message| {
                cat_err(format!("cat: {message}"))
            })
        });
    }

    if inputs.iter().any(|value| matches!(value, Value::Cell(_))) {
        let category = determine_category(inputs.view(), &like)?;
        debug_assert_eq!(category, CatCategory::Cell);
        return cat_cell_arrays(dim_zero, inputs);
    }

    if inputs.iter().any(|v| matches!(v, Value::GpuTensor(_))) {
        return cat_gpu_values(dim_zero, inputs, &like, context.as_ref()).await;
    }

    cat_host_values(dim_zero, inputs, &like, context.as_ref())
}

pub(crate) fn is_canonical_host_binding(binding: crate::builtin::RuntimeBuiltinBinding) -> bool {
    binding.identity.builtin.name == "cat"
        && binding.identity.variant == "default"
        && std::ptr::fn_addr_eq(
            binding.implementation,
            __rt_wrap_cat_builtin as crate::builtin::RuntimeBuiltinImplementation,
        )
        && binding.continuation_layout.is_some_and(|layout| {
            std::ptr::fn_addr_eq(
                layout,
                __rt_owned_layout_cat_builtin as crate::builtin::RuntimeBuiltinContinuationLayout,
            )
        })
}

/// The one synchronous host assembly tail shared by `cat` and matrix literals.
/// Callers have already selected the canonical host route and excluded cell,
/// struct, resident, and method-dispatched operands.
pub(crate) fn concatenate_host_values(dim_zero: usize, inputs: Vec<Value>) -> BuiltinResult<Value> {
    let inputs = BuiltinArguments::from_borrowed(&inputs, None)?;
    cat_host_values(dim_zero, inputs, &LikeSpec::default(), None)
}

fn cat_host_values(
    dim_zero: usize,
    inputs: BuiltinArguments,
    like: &LikeSpec,
    context: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let category = determine_category(inputs.view(), like)?;
    if let Some(context) = context {
        let policy = HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| managed::admission_error(context, cause))?;
        if policy == HostAllocationPolicy::Managed && like.device == LikeDevice::Host {
            match category {
                CatCategory::Numeric => return managed::numeric(context, dim_zero, inputs.view()),
                CatCategory::Logical => return managed::logical(context, dim_zero, inputs.view()),
                CatCategory::Complex => return managed::complex(context, dim_zero, inputs.view()),
                CatCategory::Char => return managed::character(context, dim_zero, inputs.view()),
                _ => {}
            }
        }
    }
    match category {
        CatCategory::String => cat_string_arrays(dim_zero, inputs),
        CatCategory::Char => cat_char_arrays(dim_zero, inputs),
        CatCategory::Cell => unreachable!("cell concatenation returned before residency dispatch"),
        CatCategory::Logical => cat_logical_arrays(dim_zero, inputs, like),
        CatCategory::Complex => cat_complex_arrays(dim_zero, inputs, like),
        CatCategory::Numeric => cat_numeric_tensors(dim_zero, inputs, like),
        CatCategory::Symbolic => symbolic::concatenate(dim_zero, inputs.iter()),
    }
}

fn determine_category(
    inputs: FunctionArgumentView<'_>,
    like: &LikeSpec,
) -> BuiltinResult<CatCategory> {
    let mut category = infer_category(inputs)?;
    if let Some(hint) = like.category_hint {
        category = match hint {
            CatCategory::Numeric => {
                if matches!(
                    category,
                    CatCategory::String
                        | CatCategory::Char
                        | CatCategory::Cell
                        | CatCategory::Complex
                        | CatCategory::Symbolic
                ) {
                    return Err(cat_err(
                        "cat: 'like' prototype class does not match the input classes",
                    ));
                }
                CatCategory::Numeric
            }
            CatCategory::Logical => {
                if !matches!(category, CatCategory::Logical) {
                    return Err(cat_err(
                        "cat: 'like' logical prototypes require logical inputs",
                    ));
                }
                CatCategory::Logical
            }
            CatCategory::Complex => {
                if matches!(
                    category,
                    CatCategory::String
                        | CatCategory::Char
                        | CatCategory::Cell
                        | CatCategory::Symbolic
                ) {
                    return Err(cat_err(
                        "cat: 'like' complex prototypes require numeric or complex inputs",
                    ));
                }
                CatCategory::Complex
            }
            CatCategory::Char | CatCategory::String | CatCategory::Cell | CatCategory::Symbolic => {
                return Err(cat_err(
                    "cat: 'like' prototypes for char, string, or cell arrays are not supported",
                ));
            }
        };
    }
    like.ensure_device(category)?;
    Ok(category)
}

fn infer_category(inputs: FunctionArgumentView<'_>) -> BuiltinResult<CatCategory> {
    // Direct cell concatenation encloses each complete non-cell operand in one
    // cell. Resolve this dominance before validating the enclosed value class.
    if inputs.iter().any(|value| matches!(value, Value::Cell(_))) {
        return Ok(CatCategory::Cell);
    }
    if inputs
        .iter()
        .any(|value| matches!(value, Value::Symbolic(_) | Value::SymbolicArray(_)))
    {
        return Ok(CatCategory::Symbolic);
    }

    let mut has_string = false;
    let mut has_char = false;
    let mut has_complex = false;
    let mut has_logical = false;
    let mut all_logical = true;

    for value in inputs {
        match value {
            Value::Tensor(_) | Value::Num(_) | Value::Int(_) => {
                all_logical = false;
            }
            Value::LogicalArray(_) | Value::Bool(_) => {
                has_logical = true;
            }
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                has_complex = true;
                all_logical = false;
            }
            Value::String(_) | Value::StringArray(_) => {
                has_string = true;
                all_logical = false;
            }
            Value::CharArray(_) => {
                has_char = true;
                all_logical = false;
            }
            Value::Cell(_) => unreachable!("cell dominance returned above"),
            Value::GpuTensor(_) => {
                return Err(cat_err(
                    "cat: gpuArray inputs must be concatenated using the GPU path",
                ));
            }
            other => {
                return Err(cat_err(format!(
                    "cat: unsupported input type for concatenation: {other:?}"
                )));
            }
        }
        if !matches!(value, Value::LogicalArray(_) | Value::Bool(_)) {
            all_logical = false;
        }
    }

    if has_string && has_complex {
        return Err(cat_err(
            "cat: mixed string and complex concatenation is not supported",
        ));
    }
    if has_char && (has_complex || has_logical) {
        return Err(cat_err("cat: cannot mix char arrays with other classes"));
    }
    if has_complex && (has_string || has_char) {
        return Err(cat_err(
            "cat: cannot mix complex arrays with textual or cell arrays",
        ));
    }

    if has_string {
        Ok(CatCategory::String)
    } else if has_char {
        Ok(CatCategory::Char)
    } else if has_complex {
        Ok(CatCategory::Complex)
    } else if all_logical {
        Ok(CatCategory::Logical)
    } else {
        Ok(CatCategory::Numeric)
    }
}

fn finalize_numeric_output(tensor: Tensor, like: &LikeSpec) -> BuiltinResult<Value> {
    finalize_numeric_output_with_provider(tensor, like, None)
}

fn finalize_numeric_output_with_provider(
    tensor: Tensor,
    like: &LikeSpec,
    provider_override: Option<&'static dyn AccelProvider>,
) -> BuiltinResult<Value> {
    like.ensure_device(CatCategory::Numeric)?;
    match like.device {
        LikeDevice::Host => Ok(tensor::tensor_into_value(tensor)),
        LikeDevice::Gpu => {
            let provider = provider_override
                .or_else(runmat_accelerate_api::provider)
                .ok_or_else(|| {
                    cat_err(
                        "cat: GPU output requested via 'like' but no acceleration provider is active",
                    )
                })?;
            let handle = upload_numeric_tensor(provider, &tensor)?;
            Ok(Value::GpuTensor(handle))
        }
    }
}

fn upload_numeric_tensor(
    provider: &dyn AccelProvider,
    tensor: &Tensor,
) -> BuiltinResult<runmat_accelerate_api::GpuTensorHandle> {
    gpu_helpers::upload_tensor(provider, tensor)
        .map_err(|err| cat_err(format!("cat: failed to upload concatenated tensor: {err}")))
}

fn cat_numeric_tensors(
    dim_zero: usize,
    values: BuiltinArguments,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    if let Some(target) = leftmost_integer_target(values.view()) {
        return cat_integer_tensors(target, dim_zero, values, like);
    }

    let mut tensors = Vec::with_capacity(values.len());
    for value in values.iter().cloned() {
        let tensor = tensor::value_into_tensor_for("cat", value).map_err(cat_err)?;
        tensors.push(tensor);
    }

    let tensor = concat_floating_tensors(dim_zero, tensors)?;
    finalize_numeric_output(tensor, like)
}

fn concat_floating_tensors(dim_zero: usize, tensors: Vec<Tensor>) -> BuiltinResult<Tensor> {
    let use_single = tensors
        .iter()
        .any(|tensor| tensor.numeric_dtype() == NumericDType::F32);
    let shapes: Vec<Vec<usize>> = tensors
        .iter()
        .map(|tensor| tensor.shape().to_vec())
        .collect();
    let storages: Vec<NumericStorage> = tensors
        .into_iter()
        .map(|tensor| {
            tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))
        })
        .collect::<BuiltinResult<_>>()?;

    if use_single {
        let buffers: Vec<Vec<f32>> = storages
            .into_iter()
            .map(|storage| match storage {
                NumericStorage::F64(values) => {
                    Ok(values.into_iter().map(|value| value as f32).collect())
                }
                NumericStorage::F32(values) => Ok(values),
                storage => Err(cat_err(format!(
                    "cat: internal floating concatenation received {} storage",
                    storage.class_name()
                ))),
            })
            .collect::<BuiltinResult<_>>()?;
        let data_refs: Vec<&[f32]> = buffers.iter().map(Vec::as_slice).collect();
        let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
        Tensor::from_numeric_storage(NumericStorage::F32(data), shape)
            .map_err(|error| cat_err(format!("cat: {error}")))
    } else {
        let buffers: Vec<Vec<f64>> = storages
            .into_iter()
            .map(|storage| match storage {
                NumericStorage::F64(values) => Ok(values),
                storage => Err(cat_err(format!(
                    "cat: internal double concatenation received {} storage",
                    storage.class_name()
                ))),
            })
            .collect::<BuiltinResult<_>>()?;
        let data_refs: Vec<&[f64]> = buffers.iter().map(Vec::as_slice).collect();
        let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
        Tensor::from_numeric_storage(NumericStorage::F64(data), shape)
            .map_err(|error| cat_err(format!("cat: {error}")))
    }
}

/// MATLAB concatenation adopts the class of the leftmost integer operand,
/// converting every other numeric or logical input to that class.
fn leftmost_integer_target(values: FunctionArgumentView<'_>) -> Option<IntegerClass> {
    let mut empty_integer_target = None;
    for value in values {
        match value {
            Value::Int(value) => return Some(IntegerClass::from_int_value(value)),
            Value::Tensor(tensor) => {
                if let Some(storage) = tensor.integer_storage() {
                    let target = IntegerClass::from_storage(storage);
                    if !is_empty_concat_shape(tensor.shape()) {
                        return Some(target);
                    }
                    empty_integer_target.get_or_insert(target);
                }
            }
            _ => {}
        }
    }
    empty_integer_target
}

fn cat_integer_tensors(
    target: IntegerClass,
    dim_zero: usize,
    values: BuiltinArguments,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    let tensor = build_integer_cat_tensor(target, dim_zero, values)?;
    finalize_numeric_output(tensor, like)
}

fn build_integer_cat_tensor(
    target: IntegerClass,
    dim_zero: usize,
    values: BuiltinArguments,
) -> BuiltinResult<Tensor> {
    let mut shapes = Vec::with_capacity(values.len());
    let mut value_buffers = Vec::with_capacity(values.len());

    for value in values.iter().cloned() {
        let (shape, values) = integer_concat_input(target, value)?;
        shapes.push(shape);
        value_buffers.push(values);
    }

    let data_refs: Vec<&[runmat_value::IntValue]> =
        value_buffers.iter().map(Vec::as_slice).collect();
    let (values, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    Tensor::new_integer(target.storage(values), shape)
        .map_err(|error| cat_err(format!("cat: {error}")))
}

fn integer_concat_input(
    target: IntegerClass,
    value: Value,
) -> BuiltinResult<(Vec<usize>, Vec<runmat_value::IntValue>)> {
    match value {
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let values = match tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))?
            {
                NumericStorage::F64(values) => values
                    .into_iter()
                    .map(|value| target.cast_scalar(value))
                    .collect(),
                NumericStorage::F32(values) => values
                    .into_iter()
                    .map(|value| target.cast_scalar(f64::from(value)))
                    .collect(),
                storage => integer_values(
                    storage
                        .into_integer_storage()
                        .expect("non-floating numeric storage is integer"),
                )
                .iter()
                .map(|value| target.cast_int(value))
                .collect(),
            };
            Ok((shape, values))
        }
        Value::Int(value) => Ok((vec![1, 1], vec![target.cast_int(&value)])),
        Value::Num(value) => Ok((vec![1, 1], vec![target.cast_scalar(value)])),
        Value::Bool(value) => Ok((
            vec![1, 1],
            vec![target.cast_scalar(if value { 1.0 } else { 0.0 })],
        )),
        Value::LogicalArray(array) => Ok((
            array.shape.to_vec(),
            array
                .data
                .iter()
                .map(|&value| target.cast_scalar(if value == 0 { 0.0 } else { 1.0 }))
                .collect(),
        )),
        other => Err(cat_err(format!(
            "cat: cannot concatenate integer arrays with {other:?}"
        ))),
    }
}

fn cat_logical_arrays(
    dim_zero: usize,
    values: BuiltinArguments,
    _like: &LikeSpec,
) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values.iter().cloned() {
        arrays.push(value_into_logical(value)?);
    }
    let shapes: Vec<&[usize]> = arrays.iter().map(|a| a.shape.as_slice()).collect();
    let data_refs: Vec<&[u8]> = arrays.iter().map(|a| a.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let logical = LogicalArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::LogicalArray(logical))
}

fn cat_complex_arrays(
    dim_zero: usize,
    values: BuiltinArguments,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    if values.iter().any(|v| matches!(v, Value::GpuTensor(_))) {
        return Err(cat_err(
            "cat: complex concatenation requires host arrays; convert gpuArray values first",
        ));
    }

    if let Some(target) = leftmost_complex_integer_target(values.view()) {
        return cat_typed_complex_integer_arrays(target, dim_zero, values, like);
    }

    let mut tensors = Vec::with_capacity(values.len());
    for value in values.iter().cloned() {
        let tensor = match value {
            Value::ComplexTensor(ct) => ct,
            Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cat_err(format!("cat: {e}")))?,
            other => {
                let real = tensor::value_into_tensor_for("cat", other).map_err(cat_err)?;
                tensor_to_complex(real)?
            }
        };
        tensors.push(tensor);
    }

    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape().to_vec()).collect();
    let output_dtype = if tensors
        .iter()
        .all(|tensor| tensor.numeric_dtype() == NumericDType::F32)
    {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let materialized: Vec<Vec<(f64, f64)>> =
        tensors.iter().map(ComplexTensor::materialize_f64).collect();
    let data_refs: Vec<&[(f64, f64)]> = materialized.iter().map(Vec::as_slice).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let tensor = ComplexTensor::from_f64_values_with_dtype(data, shape, output_dtype)
        .map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn leftmost_complex_integer_target(values: FunctionArgumentView<'_>) -> Option<IntegerClass> {
    let mut empty_target = None;
    for value in values {
        let Value::ComplexTensor(tensor) = value else {
            continue;
        };
        let Some(storage) = tensor.integer_storage() else {
            continue;
        };
        let target = IntegerClass::from_storage(storage.real());
        if !is_empty_concat_shape(tensor.shape()) {
            return Some(target);
        }
        empty_target.get_or_insert(target);
    }
    empty_target
}

fn cat_typed_complex_integer_arrays(
    target: IntegerClass,
    dim_zero: usize,
    values: BuiltinArguments,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    like.ensure_device(CatCategory::Complex)?;
    if matches!(like.device, LikeDevice::Gpu) {
        return Err(cat_err(
            "cat: the RunMat-only explicit GPU like form does not accept typed complex-integer output",
        ));
    }

    let mut shapes = Vec::with_capacity(values.len());
    let mut real_buffers = Vec::with_capacity(values.len());
    let mut imag_buffers = Vec::with_capacity(values.len());
    for value in values.iter().cloned() {
        let (shape, real, imag) = typed_complex_integer_concat_input(target, value)?;
        shapes.push(shape);
        real_buffers.push(real);
        imag_buffers.push(imag);
    }

    let real_refs: Vec<&[runmat_value::IntValue]> =
        real_buffers.iter().map(Vec::as_slice).collect();
    let imag_refs: Vec<&[runmat_value::IntValue]> =
        imag_buffers.iter().map(Vec::as_slice).collect();
    let (real, shape) = concat_column_major(dim_zero, &shapes, &real_refs, "cat")?;
    let (imag, imag_shape) = concat_column_major(dim_zero, &shapes, &imag_refs, "cat")?;
    debug_assert_eq!(shape, imag_shape);
    IntegerComplexStorage::new(target.storage(real), target.storage(imag))
        .and_then(|storage| ComplexTensor::new_integer(storage, shape))
        .map(Value::ComplexTensor)
        .map_err(|error| cat_err(format!("cat: {error}")))
}

fn typed_complex_integer_concat_input(
    target: IntegerClass,
    value: Value,
) -> BuiltinResult<(
    Vec<usize>,
    Vec<runmat_value::IntValue>,
    Vec<runmat_value::IntValue>,
)> {
    let zero = || target.cast_scalar(0.0);
    match value {
        Value::ComplexTensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return Ok((
                    tensor.shape().to_vec(),
                    integer_values(storage.real().clone())
                        .iter()
                        .map(|value| target.cast_int(value))
                        .collect(),
                    integer_values(storage.imag().clone())
                        .iter()
                        .map(|value| target.cast_int(value))
                        .collect(),
                ));
            }
            let shape = tensor.shape().to_vec();
            let values = tensor.materialize_f64();
            Ok((
                shape,
                values
                    .iter()
                    .map(|&(real, _)| target.cast_scalar(real))
                    .collect(),
                values
                    .iter()
                    .map(|&(_, imag)| target.cast_scalar(imag))
                    .collect(),
            ))
        }
        Value::Complex(real, imag) => Ok((
            vec![1, 1],
            vec![target.cast_scalar(real)],
            vec![target.cast_scalar(imag)],
        )),
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let real: Vec<runmat_value::IntValue> = match tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))?
            {
                NumericStorage::F64(values) => values
                    .into_iter()
                    .map(|value| target.cast_scalar(value))
                    .collect(),
                NumericStorage::F32(values) => values
                    .into_iter()
                    .map(|value| target.cast_scalar(f64::from(value)))
                    .collect(),
                storage => integer_values(
                    storage
                        .into_integer_storage()
                        .expect("non-floating numeric storage is integer"),
                )
                .iter()
                .map(|value| target.cast_int(value))
                .collect(),
            };
            let imag = vec![zero(); real.len()];
            Ok((shape, real, imag))
        }
        Value::Int(value) => Ok((vec![1, 1], vec![target.cast_int(&value)], vec![zero()])),
        Value::Num(value) => Ok((vec![1, 1], vec![target.cast_scalar(value)], vec![zero()])),
        Value::Bool(value) => Ok((
            vec![1, 1],
            vec![target.cast_scalar(if value { 1.0 } else { 0.0 })],
            vec![zero()],
        )),
        Value::LogicalArray(array) => {
            let real: Vec<_> = array
                .data
                .iter()
                .map(|&value| target.cast_scalar(if value == 0 { 0.0 } else { 1.0 }))
                .collect();
            let imag = vec![zero(); real.len()];
            Ok((array.shape.to_vec(), real, imag))
        }
        other => Err(cat_err(format!(
            "cat: cannot concatenate typed complex integer arrays with {other:?}"
        ))),
    }
}

fn cat_char_arrays(dim_zero: usize, values: BuiltinArguments) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values.iter().cloned() {
        arrays.push(char_array_from_value(value)?);
    }
    let shapes: Vec<Vec<usize>> = arrays
        .iter()
        .map(|array| match array.shape() {
            [] => vec![0, 0],
            [length] => vec![1, *length],
            shape => shape.to_vec(),
        })
        .collect();
    let buffers: Vec<Vec<runmat_value::Utf16CodeUnit>> =
        arrays.iter().map(CharArray::to_column_major).collect();
    let data_refs: Vec<&[runmat_value::Utf16CodeUnit]> =
        buffers.iter().map(Vec::as_slice).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    CharArray::from_column_major(data, shape)
        .map(Value::CharArray)
        .map_err(|error| cat_err(format!("cat: {error}")))
}

fn char_array_from_value(value: Value) -> BuiltinResult<CharArray> {
    match value {
        Value::CharArray(ca) => Ok(ca),
        Value::Num(n) => char_array_from_f64_with_prefix(n, "cat"),
        Value::Int(i) => char_array_from_f64_with_prefix(i.to_f64(), "cat"),
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let column_major = match tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))?
            {
                NumericStorage::F64(values) => values
                    .into_iter()
                    .map(char_from_numeric_code_point)
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericStorage::F32(values) => values
                    .into_iter()
                    .map(|value| char_from_numeric_code_point(f64::from(value)))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                storage => integer_values(
                    storage
                        .into_integer_storage()
                        .expect("non-floating numeric storage is integer"),
                )
                .into_iter()
                .map(|value| char_from_numeric_code_point(value.to_f64()))
                .collect::<BuiltinResult<Vec<_>>>()?,
            };
            CharArray::from_column_major(column_major, shape)
                .map_err(|error| cat_err(format!("cat: {error}")))
        }
        Value::Bool(_) | Value::LogicalArray(_) => Err(cat_err(
            "cat: character arrays cannot be concatenated with logical values",
        )),
        other => Err(cat_err(format!(
            "cat: expected char arrays or scalar code points, got {other:?}"
        ))),
    }
}

fn char_from_numeric_code_point(value: f64) -> BuiltinResult<runmat_value::Utf16CodeUnit> {
    char_from_f64_code_unit(value, "cat")
}

fn cat_string_arrays(dim_zero: usize, values: BuiltinArguments) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values.iter().cloned() {
        arrays.push(value_into_string_array(value)?);
    }
    let shapes: Vec<&[usize]> = arrays.iter().map(StringArray::shape).collect();
    let data_refs: Vec<&[runmat_value::StringElement]> =
        arrays.iter().map(StringArray::data).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let array = StringArray::from_elements_unbounded(data, shape)
        .map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::StringArray(array))
}

fn cat_cell_arrays(dim_zero: usize, values: BuiltinArguments) -> BuiltinResult<Value> {
    let mut shapes = Vec::with_capacity(values.len());
    let mut buffers = Vec::with_capacity(values.len());
    for value in values.iter().cloned() {
        let (shape, data) = match value {
            Value::Cell(cell) => {
                let data = cell.to_column_major();
                (cell.shape().to_vec(), data)
            }
            other => (vec![1, 1], vec![other]),
        };
        shapes.push(shape);
        buffers.push(data);
    }
    let data_refs: Vec<&[Value]> = buffers.iter().map(Vec::as_slice).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    CellArray::from_column_major(data, shape)
        .map(Value::Cell)
        .map_err(|error| cat_err(format!("cat: {error}")))
}

async fn cat_gpu_values(
    dim_zero: usize,
    values: BuiltinArguments,
    like: &LikeSpec,
    context: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let has_complex_data = values.iter().any(|value| match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => true,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        }
        _ => false,
    });
    let output_like = if like.explicit {
        like.clone()
    } else {
        LikeSpec {
            device: LikeDevice::Gpu,
            category_hint: Some(if has_complex_data {
                CatCategory::Complex
            } else {
                CatCategory::Numeric
            }),
            explicit: false,
        }
    };
    if let Some(hint) = like.category_hint {
        if (has_complex_data && !matches!(hint, CatCategory::Complex))
            || (!has_complex_data && !matches!(hint, CatCategory::Numeric))
        {
            return Err(cat_err(
                "cat: 'like' prototype class does not match gpuArray inputs",
            ));
        }
    }
    output_like.ensure_device(if has_complex_data {
        CatCategory::Complex
    } else {
        CatCategory::Numeric
    })?;

    let all_gpu = values
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)));
    let handles: Vec<_> = values
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect();

    let first_handle = handles
        .first()
        .map(|handle| (*handle).clone())
        .ok_or_else(|| cat_err("cat: no gpuArray inputs to concatenate"))?;
    let provider = runmat_accelerate_api::provider_for_handle(&first_handle)
        .ok_or_else(|| cat_err("cat: no acceleration provider is registered"))?;
    for handle in &handles {
        let Some(owner) = runmat_accelerate_api::provider_for_handle(handle) else {
            return Err(cat_err("cat: no acceleration provider is registered"));
        };
        if !runmat_accelerate_api::same_provider_instance(owner, provider) {
            return Err(cat_err(
                "cat: all gpuArray inputs must belong to the same acceleration provider",
            ));
        }
    }
    let has_integer_handle = handles
        .iter()
        .any(|handle| runmat_accelerate_api::handle_integer_type(handle).is_some());
    let has_empty_handle = handles.iter().any(|handle| handle.shape().contains(&0));
    let has_nonempty_handle = handles.iter().any(|handle| !handle.shape().contains(&0));
    let must_apply_runtime_empty_omission = has_empty_handle && has_nonempty_handle;

    // A provider may decline before effects, but an attempted execution may
    // not be replayed through gathering after allocation or submission.
    if all_gpu && !has_integer_handle && !must_apply_runtime_empty_omission {
        let expected_shape = gpu_cat_output_shape(dim_zero, &handles)?;
        let expected_storage = if has_complex_data {
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        } else {
            runmat_accelerate_api::GpuTensorStorage::Real
        };
        let expected_logical = handles
            .iter()
            .all(|handle| runmat_accelerate_api::handle_is_logical(handle));
        let expected_precision = if (has_complex_data
            && handles.iter().all(|handle| {
                runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32)
            }))
            || (!has_complex_data
                && handles.iter().any(|handle| {
                    runmat_accelerate_api::handle_precision(handle)
                        == Some(runmat_accelerate_api::ProviderPrecision::F32)
                })) {
            runmat_accelerate_api::ProviderPrecision::F32
        } else {
            runmat_accelerate_api::ProviderPrecision::F64
        };
        let owned_handles: Vec<_> = handles.iter().map(|handle| (*handle).clone()).collect();
        match provider.cat_optional(dim_zero + 1, &owned_handles).await {
            Ok(result) => {
                let same_owner =
                    runmat_accelerate_api::provider_for_handle(&result).is_some_and(|owner| {
                        runmat_accelerate_api::same_provider_instance(owner, provider)
                    });
                if !same_owner
                    || result.shape() != expected_shape
                    || result.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
                    || runmat_accelerate_api::handle_storage(&result) != expected_storage
                    || runmat_accelerate_api::handle_is_logical(&result) != expected_logical
                    || (!expected_logical
                        && runmat_accelerate_api::handle_precision(&result)
                            != Some(expected_precision))
                    || owned_handles
                        .iter()
                        .any(|input| result.shares_allocation_with(input))
                {
                    return Err(gpu_helpers::reject_gpu_output(
                        result,
                        cat_err("cat: provider returned an invalid output shape, class, owner, or alias"),
                    ));
                }
                let mut result = result;
                if let Err(error) =
                    gpu_helpers::propagate_output_provenance(&mut result, owned_handles.iter())
                {
                    return Err(gpu_helpers::reject_gpu_output(result, error));
                }
                return finalize_gpu_value(result, &output_like).await;
            }
            Err(error) if error.is_unsupported() => {}
            Err(error) => {
                return Err(build_runtime_error(format!(
                    "cat: provider execution failed: {error}"
                ))
                .with_builtin(BUILTIN_NAME)
                .with_source(error)
                .build());
            }
        }
    }

    let gathered_values = values
        .try_map_preserving_roots_async(values.len(), context, async |index| {
            let value = &values[index];
            match value {
                Value::GpuTensor(_) => gpu_helpers::gather_value_async(value).await,
                value => Ok(value.clone()),
            }
        })
        .await?;

    let category = determine_category(gathered_values.view(), &output_like)?;
    if matches!(category, CatCategory::Complex) {
        let host_like = LikeSpec {
            device: LikeDevice::Host,
            category_hint: Some(CatCategory::Complex),
            explicit: false,
        };
        let result = cat_complex_arrays(dim_zero, gathered_values, &host_like)?;
        if matches!(output_like.device, LikeDevice::Host) {
            return Ok(result);
        }
        return gpu_helpers::restore_class_preserving_value(&first_handle, result, BUILTIN_NAME)
            .map_err(|error| cat_err(error.message()));
    }
    if !matches!(category, CatCategory::Numeric) {
        return Err(cat_err("cat: gpuArray concatenation requires numeric data"));
    }
    if let Some(target) = leftmost_integer_target(gathered_values.view()) {
        let tensor = build_integer_cat_tensor(target, dim_zero, gathered_values)?;
        return finalize_numeric_output_with_provider(tensor, &output_like, Some(provider));
    }

    let mut tensors = Vec::with_capacity(gathered_values.len());
    for value in gathered_values.iter().cloned() {
        tensors.push(tensor::value_into_tensor_for("cat", value).map_err(cat_err)?);
    }
    let tensor = concat_floating_tensors(dim_zero, tensors)?;
    if matches!(output_like.device, LikeDevice::Host) {
        return Ok(tensor::tensor_into_value(tensor));
    }

    match upload_numeric_tensor(provider, &tensor) {
        Ok(handle) => Ok(Value::GpuTensor(handle)),
        Err(_) => Ok(tensor::tensor_into_value(tensor)),
    }
}

fn gpu_cat_output_shape(
    dim_zero: usize,
    handles: &[&runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Vec<usize>> {
    let rank = handles
        .iter()
        .map(|handle| handle.shape().len())
        .max()
        .unwrap_or(0)
        .max(dim_zero + 1);
    let mut output = vec![1usize; rank];
    let first = handles
        .first()
        .ok_or_else(|| cat_err("cat: no gpuArray inputs to concatenate"))?;
    for (axis, extent) in output.iter_mut().enumerate() {
        *extent = first.shape().get(axis).copied().unwrap_or(1);
    }
    let mut concatenated = 0usize;
    for (index, handle) in handles.iter().enumerate() {
        for (axis, &expected) in output.iter().enumerate() {
            if axis != dim_zero && handle.shape().get(axis).copied().unwrap_or(1) != expected {
                return Err(cat_err(format!(
                    "cat: dimension {} mismatch between input 1 and input {}",
                    axis + 1,
                    index + 1
                )));
            }
        }
        concatenated = concatenated
            .checked_add(handle.shape().get(dim_zero).copied().unwrap_or(1))
            .ok_or_else(|| cat_err("cat: concatenated dimension exceeds maximum size"))?;
    }
    output[dim_zero] = concatenated;
    if !output.contains(&0) {
        checked_product(&output)
            .ok_or_else(|| cat_err("cat: resulting array exceeds maximum size"))?;
    }
    Ok(normalize_shape(output, dim_zero))
}

fn concat_column_major<T: Clone, S: AsRef<[usize]>>(
    dim_zero: usize,
    shapes: &[S],
    data: &[&[T]],
    context: &str,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    concat_column_major_with_neutral(dim_zero, shapes, data, context, is_empty_concat_shape)
}

fn concat_column_major_with_neutral<T: Clone, S: AsRef<[usize]>>(
    dim_zero: usize,
    shapes: &[S],
    data: &[&[T]],
    context: &str,
    is_neutral: impl Fn(&[usize]) -> bool,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    if shapes.is_empty() {
        return Err(cat_err(format!("{context}: no inputs to concatenate")));
    }
    let rank = shapes
        .iter()
        .map(|s| s.as_ref().len())
        .max()
        .unwrap_or(1)
        .max(dim_zero + 1);

    let mut padded = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let mut current = shape.as_ref().to_vec();
        while current.len() < rank {
            current.push(1);
        }
        padded.push(current);
    }

    for (idx, (shape, slice)) in padded.iter().zip(data.iter()).enumerate() {
        let expected = checked_product(shape)
            .ok_or_else(|| cat_err(format!("{context}: input {} exceeds maximum size", idx + 1)))?;
        if expected != slice.len() {
            return Err(cat_err(format!(
                "{context}: input {} has {} elements but the shape multiplies to {}",
                idx + 1,
                slice.len(),
                expected
            )));
        }
    }

    // MATLAB omits empty operands when a nonempty partner remains. Preserve
    // ordinary geometry when every operand is empty.
    let has_non_neutral = padded.iter().any(|shape| !is_neutral(shape));
    let mut active_shapes: Vec<Vec<usize>> = Vec::with_capacity(padded.len());
    let mut active_data: Vec<&[T]> = Vec::with_capacity(data.len());
    if has_non_neutral {
        for (shape, slice) in padded.iter().zip(data.iter()) {
            if !is_neutral(shape) {
                active_shapes.push(shape.clone());
                active_data.push(*slice);
            }
        }
    } else {
        active_shapes = padded;
        active_data.extend(data.iter().copied());
    }

    for axis in 0..rank {
        if axis == dim_zero {
            continue;
        }
        let reference = active_shapes[0][axis];
        for (idx, shape) in active_shapes.iter().enumerate().skip(1) {
            if shape[axis] != reference {
                return Err(cat_err(format!(
                    "{context}: dimension {} mismatch between input 1 (size {}) and input {} (size {})",
                    axis + 1,
                    reference,
                    idx + 1,
                    shape[axis]
                )));
            }
        }
    }

    let mut output_shape = active_shapes[0].clone();
    let mut concat_dim = 0usize;
    for shape in &active_shapes {
        concat_dim = concat_dim.checked_add(shape[dim_zero]).ok_or_else(|| {
            cat_err(format!(
                "{context}: concatenated dimension exceeds maximum size"
            ))
        })?;
    }
    output_shape[dim_zero] = concat_dim;

    let total = match checked_product(&output_shape) {
        Some(total) => total,
        None => {
            return Err(cat_err(format!(
                "{context}: resulting array exceeds maximum size"
            )))
        }
    };
    if total == 0 {
        return Ok((Vec::new(), normalize_shape(output_shape, dim_zero)));
    }

    let inner = if dim_zero == 0 {
        1
    } else {
        output_shape[..dim_zero].iter().product()
    };
    let outer = if dim_zero + 1 >= rank {
        1
    } else {
        output_shape[dim_zero + 1..].iter().product()
    };

    let mut output = Vec::with_capacity(total);
    for outer_idx in 0..outer {
        for (shape, slice) in active_shapes.iter().zip(active_data.iter()) {
            let mid = shape[dim_zero];
            let chunk = mid * inner;
            if chunk == 0 {
                continue;
            }
            let offset = outer_idx * chunk;
            output.extend_from_slice(&slice[offset..offset + chunk]);
        }
    }

    Ok((output, normalize_shape(output_shape, dim_zero)))
}

fn is_empty_concat_shape(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn normalize_shape(mut shape: Vec<usize>, dim_zero: usize) -> Vec<usize> {
    let min_len = (dim_zero + 1).max(2).min(shape.len());
    while shape.len() > min_len && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn checked_product(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn value_into_logical(value: Value) -> BuiltinResult<LogicalArray> {
    match value {
        Value::LogicalArray(array) => Ok(array),
        Value::Bool(flag) => LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
            .map_err(|e| cat_err(format!("cat: {e}"))),
        other => Err(cat_err(format!(
            "cat: expected logical inputs, got {:?}",
            other
        ))),
    }
}

fn value_into_string_array(value: Value) -> BuiltinResult<StringArray> {
    match value {
        Value::StringArray(array) => Ok(array),
        Value::String(text) => StringArray::from_elements_unbounded(vec![text], vec![1, 1])
            .map_err(|e| cat_err(format!("cat: {e}"))),
        Value::CharArray(chars) => {
            let pages = if chars.shape.len() <= 2 {
                1
            } else {
                checked_product(&chars.shape[2..])
                    .ok_or_else(|| cat_err("cat: character page count exceeds maximum size"))?
            };
            let mut data = Vec::with_capacity(chars.rows * pages);
            for page in 0..pages {
                let page_offset = page * chars.rows * chars.cols;
                for row in 0..chars.rows {
                    let start = page_offset + row * chars.cols;
                    let end = start + chars.cols;
                    let text = runmat_value::decode_utf16_code_units(&chars.data()[start..end])
                        .map_err(|source| {
                            let mut error = cat_err(format!("cat: invalid string row: {source}"));
                            error.source = Some(Box::new(source));
                            error
                        })?;
                    data.push(text);
                }
            }
            let mut shape = vec![chars.rows, 1];
            shape.extend(chars.shape.iter().skip(2).copied());
            StringArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))
        }
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let data = match tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))?
            {
                NumericStorage::F64(values) => {
                    values.into_iter().map(|value| value.to_string()).collect()
                }
                NumericStorage::F32(values) => {
                    values.into_iter().map(|value| value.to_string()).collect()
                }
                storage => integer_values(
                    storage
                        .into_integer_storage()
                        .expect("non-floating numeric storage is integer"),
                )
                .into_iter()
                .map(|value| value.decimal_string())
                .collect(),
            };
            StringArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))
        }
        Value::Num(value) => StringArray::new(vec![value.to_string()], vec![1, 1])
            .map_err(|e| cat_err(format!("cat: {e}"))),
        Value::Int(value) => StringArray::new(vec![value.decimal_string()], vec![1, 1])
            .map_err(|e| cat_err(format!("cat: {e}"))),
        Value::Bool(value) => {
            StringArray::new(vec![if value { "1" } else { "0" }.to_string()], vec![1, 1])
                .map_err(|e| cat_err(format!("cat: {e}")))
        }
        Value::LogicalArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .map(|value| if value == 0 { "0" } else { "1" }.to_string())
                .collect(),
            array.shape.to_vec(),
        )
        .map_err(|e| cat_err(format!("cat: {e}"))),
        other => Err(cat_err(format!(
            "cat: expected string arrays, got {:?}",
            other
        ))),
    }
}

fn tensor_to_complex(tensor: Tensor) -> BuiltinResult<ComplexTensor> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| cat_err(format!("cat: {error}")))?;
    let data = match storage {
        NumericStorage::F64(values) => values.into_iter().map(|real| (real, 0.0)).collect(),
        NumericStorage::F32(values) => values
            .into_iter()
            .map(|real| (f64::from(real), 0.0))
            .collect(),
        storage => integer_values(
            storage
                .into_integer_storage()
                .expect("non-floating numeric storage is integer"),
        )
        .into_iter()
        .map(|real| (real.to_f64(), 0.0))
        .collect(),
    };
    ComplexTensor::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))
}

async fn finalize_gpu_value(
    handle: runmat_accelerate_api::GpuTensorHandle,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    if matches!(like.device, LikeDevice::Host) {
        let value = match gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await {
            Ok(value) => value,
            Err(error) => return Err(gpu_helpers::reject_gpu_output(handle, error)),
        };
        handle
            .release()
            .map_err(|error| cat_err(format!("cat: provider output retirement failed: {error}")))?;
        return Ok(value);
    }
    Ok(Value::GpuTensor(handle))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn cat_builtin(dim: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::cat_builtin(
            None,
            dim,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{
        HostIntegerDataView, HostIntegerTensorView, HostTensorView, IntegerElementType,
    };
    use runmat_value::{IntValue, IntegerStorage, StructArray, StructValue, Tensor};

    fn test_structure(value: f64) -> StructValue {
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("payload", Value::Num(value));
        structure.finish()
    }

    #[test]
    fn struct_cat_omits_empty_operands_when_nonempty_structure_remains() {
        let array = StructArray::new(vec![test_structure(1.0), test_structure(2.0)], vec![2, 1])
            .expect("structure array");
        let unrelated_empty =
            StructArray::empty(vec!["other".into()], vec![0, 4]).expect("empty structure array");
        let numeric_empty = Tensor::new(Vec::new(), vec![0, 0]).expect("numeric empty");
        let result = cat_builtin(
            Value::Num(1.0),
            vec![
                Value::StructArray(unrelated_empty),
                Value::Tensor(numeric_empty),
                Value::StructArray(array),
            ],
        )
        .expect("empty operands are neutral");
        let Value::StructArray(result) = result else {
            panic!("expected structure array");
        };
        assert_eq!(result.shape(), [2, 1]);
        assert_eq!(
            result
                .field_names()
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            ["payload"]
        );
    }

    #[test]
    fn cat_integer_capability_and_extension_metadata_is_explicit() {
        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name("cat").expect("catalogued cat");
        assert_eq!(builtin.integer_capabilities.len(), 4);
        assert_eq!(builtin.extensions.len(), 2);
        assert_eq!(
            builtin.integer_capabilities[0].overload,
            BuiltinIntegerOverloadKind::StructuralParameter
        );
        assert_eq!(
            builtin.integer_capabilities[1].overflow,
            BuiltinIntegerOverflowRule::Saturate
        );
        assert_eq!(
            builtin.integer_capabilities[2].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            builtin.integer_capabilities[3].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(
            builtin.integer_capabilities[3].backend,
            BuiltinIntegerBackendRule::GatherFallback
        );
        assert_eq!(builtin.extensions, &CAT_EXTENSIONS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_numeric_rows() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .expect("cat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[4, 2]);
                assert_eq!(
                    t.materialize_f64(),
                    vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn cat_single_and_double_preserves_single_storage() {
        let single = Tensor::from_f32(vec![1.25, -2.5], vec![1, 2]).expect("single input tensor");
        let double = Tensor::new(vec![3.75], vec![1, 1]).expect("double input tensor");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(single), Value::Tensor(double)],
        )
        .expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            output.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![1.25, -2.5, 3.75])
        );
    }

    #[test]
    fn cat_dimension_scalar_tensor_reads_typed_integer_storage_exactly() {
        let dim = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("dim");
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result =
            cat_builtin(Value::Tensor(dim), vec![Value::Tensor(a), Value::Tensor(b)]).expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(output.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cat_dimension_scalar_double_tensor_is_accepted() {
        let dim = Tensor::new(vec![2.0], vec![1, 1]).expect("dim");
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result =
            cat_builtin(Value::Tensor(dim), vec![Value::Tensor(a), Value::Tensor(b)]).expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(output.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cat_dimension_scalar_tensor_rejects_negative_typed_integer_storage() {
        let dim = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("dim");
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let err =
            cat_builtin(Value::Tensor(dim), vec![Value::Tensor(a), Value::Tensor(b)]).unwrap_err();
        assert!(err.message().contains("dimension"));
    }

    #[test]
    fn cat_dimension_tensor_rejects_non_scalar_and_oversized_typed_integer_storage() {
        let vector_dim = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("dim");
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let err = cat_builtin(
            Value::Tensor(vector_dim),
            vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        )
        .unwrap_err();
        assert!(!err.message().is_empty());

        let too_large =
            Tensor::new_integer(IntegerStorage::U64(vec![usize::MAX as u64]), vec![1, 1])
                .expect("dim");
        let err = cat_builtin(
            Value::Tensor(too_large),
            vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        )
        .unwrap_err();
        assert!(err.message().contains("dimension"));

        let err = cat_builtin(
            Value::Int(IntValue::U64(usize::MAX as u64)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .unwrap_err();
        assert!(err.message().contains("dimension"));
    }

    #[test]
    fn cat_preserves_exact_uint64_storage() {
        let left =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 4]), vec![1, 2]).expect("left");
        let right = Tensor::new_integer(IntegerStorage::U64(vec![5]), vec![1, 1]).expect("right");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(left), Value::Tensor(right)],
        )
        .expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 4, 5]))
        );
    }

    #[test]
    fn cat_uses_leftmost_integer_class_and_conversion_rules() {
        let left = Tensor::new_integer(IntegerStorage::I8(vec![12]), vec![1, 1]).expect("left");
        let single = Tensor::from_f32(vec![3.5], vec![1, 1]).expect("single");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![
                Value::Tensor(left),
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Tensor(single),
                Value::Bool(true),
            ],
        )
        .expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![12, i8::MAX, 4, 1]))
        );
    }

    #[test]
    fn cat_accepts_every_ordered_integer_class_pair_and_keeps_left_class() {
        let classes = vec![
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for left_storage in &classes {
            for right_storage in &classes {
                let left = Tensor::new_integer(left_storage.clone(), vec![1, 1]).expect("left");
                let right = Tensor::new_integer(right_storage.clone(), vec![1, 1]).expect("right");
                let result = cat_builtin(
                    Value::Int(IntValue::I32(2)),
                    vec![Value::Tensor(left), Value::Tensor(right)],
                )
                .expect("ordered integer pair");
                let Value::Tensor(output) = result else {
                    panic!("expected tensor");
                };
                let output_storage = output.integer_storage().expect("integer output");
                assert_eq!(
                    std::mem::discriminant(output_storage),
                    std::mem::discriminant(left_storage),
                    "left={left_storage:?}, right={right_storage:?}"
                );
                assert_eq!(output.shape(), &[1, 2]);
            }
        }
    }

    #[test]
    fn cat_selects_a_later_integer_class_after_double_input() {
        let right = Tensor::new_integer(IntegerStorage::I16(vec![8]), vec![1, 1]).expect("right");
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::Num(3.5), Value::Tensor(right)],
        )
        .expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape(), &[2, 1]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I16(vec![4, 8]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_dimension_mismatch_errors() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3, 1]).unwrap();
        let err = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .unwrap_err();
        assert!(err.message().contains("dimension 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_char_columns() {
        let left = CharArray::from_unicode_scalars("Run".chars().collect(), 1, 3).unwrap();
        let right = CharArray::from_unicode_scalars("Mat".chars().collect(), 1, 3).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::CharArray(left), Value::CharArray(right)],
        )
        .expect("cat");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 6);
                let text: String = runmat_value::decode_utf16_code_units(arr.data())
                    .expect("valid UTF-16 fixture");
                assert_eq!(text, "RunMat");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_string_and_char_mix_promotes_to_string_array() {
        let left = StringArray::new(vec!["wn = ".to_string()], vec![1, 1]).unwrap();
        let mid = CharArray::new_row("6");
        let right = StringArray::new(vec![" rad/s".to_string()], vec![1, 1]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![
                Value::StringArray(left),
                Value::CharArray(mid),
                Value::StringArray(right),
            ],
        )
        .expect("cat");
        match result {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![1, 3]);
                assert_eq!(
                    arr.data(),
                    vec!["wn = ".to_string(), "6".to_string(), " rad/s".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn cat_string_dominates_exact_numeric_and_logical_inputs() {
        let prefix = StringArray::new(vec!["id".to_string()], vec![1, 1]).unwrap();
        let integers = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("integer input");
        let logical = LogicalArray::new(vec![1], vec![1, 1]).expect("logical input");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![
                Value::StringArray(prefix),
                Value::Tensor(integers),
                Value::LogicalArray(logical),
            ],
        )
        .expect("mixed string cat");

        let Value::StringArray(output) = result else {
            panic!("expected string array");
        };
        assert_eq!(output.shape, vec![1, 4]);
        assert_eq!(
            output.data(),
            vec![
                "id".to_string(),
                "9007199254740993".to_string(),
                u64::MAX.to_string(),
                "1".to_string(),
            ]
        );
    }

    #[test]
    fn cat_cell_wraps_each_complete_noncell_operand() {
        let first = CellArray::new(vec![Value::String("head".into())], 1, 1).expect("cell");
        let matrix =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3, 4]), vec![2, 2]).expect("matrix");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Cell(first), Value::Tensor(matrix.clone())],
        )
        .expect("mixed cell cat");

        let Value::Cell(output) = result else {
            panic!("expected cell array");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(output.data()[0], Value::String("head".into()));
        assert_eq!(output.data()[1], Value::Tensor(matrix));
    }

    #[test]
    fn cat_char_and_logical_rejects_both_scalar_and_array_forms() {
        let chars = CharArray::new_row("a");
        let scalar_error = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::CharArray(chars.clone()), Value::Bool(true)],
        )
        .unwrap_err();
        assert!(scalar_error.message().contains("char"));

        let logical = LogicalArray::new(vec![1], vec![1, 1]).expect("logical");
        let array_error = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::CharArray(chars), Value::LogicalArray(logical)],
        )
        .unwrap_err();
        assert!(array_error.message().contains("char"));
    }

    #[test]
    fn cat_supports_nd_character_and_cell_arrays() {
        let left_chars =
            CharArray::from_unicode_scalars_with_shape("abcd".chars().collect(), vec![1, 2, 2])
                .expect("left");
        let right_chars =
            CharArray::from_unicode_scalars_with_shape("efgh".chars().collect(), vec![1, 2, 2])
                .expect("right");
        let char_result = cat_builtin(
            Value::Int(IntValue::I32(3)),
            vec![Value::CharArray(left_chars), Value::CharArray(right_chars)],
        )
        .expect("nd char cat");
        let Value::CharArray(chars) = char_result else {
            panic!("expected char array");
        };
        assert_eq!(chars.shape, vec![1, 2, 4]);
        assert_eq!(
            runmat_value::decode_utf16_code_units(chars.data()).unwrap(),
            "abcdefgh"
        );

        let left_cells = CellArray::new_with_shape(
            vec![Value::Int(IntValue::I8(1)), Value::Int(IntValue::I8(2))],
            vec![1, 1, 2],
        )
        .expect("left cells");
        let right_cells = CellArray::new_with_shape(
            vec![Value::Int(IntValue::I8(3)), Value::Int(IntValue::I8(4))],
            vec![1, 1, 2],
        )
        .expect("right cells");
        let cell_result = cat_builtin(
            Value::Int(IntValue::I32(3)),
            vec![Value::Cell(left_cells), Value::Cell(right_cells)],
        )
        .expect("nd cell cat");
        let Value::Cell(cells) = cell_result else {
            panic!("expected cell array");
        };
        assert_eq!(cells.shape, vec![1, 1, 4]);
        assert_eq!(
            cells.data(),
            vec![
                Value::Int(IntValue::I8(1)),
                Value::Int(IntValue::I8(2)),
                Value::Int(IntValue::I8(3)),
                Value::Int(IntValue::I8(4)),
            ]
        );
    }

    #[test]
    fn cat_omits_shaped_empty_geometry_and_uses_the_nonempty_integer_class() {
        let shaped_empty =
            Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 4]).expect("empty");
        let nonempty = Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 3, 4, 5, 6]), vec![2, 3])
            .expect("nonempty");
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::Tensor(shaped_empty), Value::Tensor(nonempty)],
        )
        .expect("empty omission");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U8(vec![1, 2, 3, 4, 5, 6]))
        );
    }

    #[test]
    fn cat_all_empty_inputs_keep_the_leftmost_integer_class() {
        let left =
            Tensor::new_integer(IntegerStorage::I16(Vec::new()), vec![0, 2]).expect("left empty");
        let right =
            Tensor::new_integer(IntegerStorage::U8(Vec::new()), vec![0, 3]).expect("right empty");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(left), Value::Tensor(right)],
        )
        .expect("all-empty concatenation");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape(), &[0, 5]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I16(Vec::new()))
        );
    }

    #[test]
    fn cat_complex_integer_empty_inputs_follow_the_same_class_selection_rule() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let empty = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I16(Vec::new()),
                IntegerStorage::I16(Vec::new()),
            )
            .expect("empty storage"),
            vec![1, 0],
        )
        .expect("empty complex integer");
        let value = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::U8(vec![7]), IntegerStorage::U8(vec![9]))
                .expect("value storage"),
            vec![1, 1],
        )
        .expect("complex integer value");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::ComplexTensor(empty), Value::ComplexTensor(value)],
        )
        .expect("complex empty omission");
        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(
                &IntegerComplexStorage::new(
                    IntegerStorage::U8(vec![7]),
                    IntegerStorage::U8(vec![9]),
                )
                .expect("expected storage")
            )
        );
    }

    #[test]
    fn cat_mixed_host_and_gpu_complex_integers_preserves_exact_resident_output() {
        test_support::with_test_provider(|provider| {
            let device = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![9_007_199_254_740_993]),
                    IntegerStorage::U64(vec![3]),
                )
                .expect("device storage"),
                vec![1, 1],
            )
            .expect("device value");
            let host = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![5]),
                )
                .expect("host storage"),
                vec![1, 1],
            )
            .expect("host value");
            let device = gpu_helpers::upload_complex_tensor(provider, &device).expect("upload");
            let Value::GpuTensor(output) = cat_builtin(
                Value::Int(IntValue::I32(2)),
                vec![Value::GpuTensor(device), Value::ComplexTensor(host)],
            )
            .expect("cat") else {
                panic!("expected resident complex integer");
            };
            let Value::ComplexTensor(gathered) =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output)))
                    .expect("gather")
            else {
                panic!("expected complex integer");
            };
            assert_eq!(gathered.shape(), &[1, 2]);
            let storage = gathered.integer_storage().expect("integer storage");
            assert_eq!(
                storage.real(),
                &IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX])
            );
            assert_eq!(storage.imag(), &IntegerStorage::U64(vec![3, 5]));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_logical_preserves_type() {
        let top = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let bottom = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::LogicalArray(top), Value::LogicalArray(bottom)],
        )
        .expect("cat");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![2, 3]);
                assert_eq!(arr.data, vec![1, 0, 0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let view_a = HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            };
            let view_b = HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            };
            let ha = provider.upload(&view_a).expect("upload a");
            let hb = provider.upload(&view_b).expect("upload b");
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![Value::GpuTensor(ha), Value::GpuTensor(hb)],
            )
            .expect("cat");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), &[4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_gpu_from_host_inputs() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &proto.materialize_f64(),
                shape: proto.shape(),
            };
            let proto_handle = provider.upload(&proto_view).expect("upload proto");

            let a = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![
                    Value::Tensor(a),
                    Value::Tensor(b),
                    Value::from("like"),
                    Value::GpuTensor(proto_handle),
                ],
            )
            .expect("cat with like");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), &[4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_gpu_from_host_integer_inputs_uploads_exact_integer_storage() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_handle = provider
                .upload(&HostTensorView {
                    data: &proto.materialize_f64(),
                    shape: proto.shape(),
                })
                .expect("upload proto");
            let left = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 4]), vec![1, 2])
                .expect("left");
            let right =
                Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                    .expect("right");

            let result = cat_builtin(
                Value::Int(IntValue::I32(2)),
                vec![
                    Value::Tensor(left),
                    Value::Tensor(right),
                    Value::from("like"),
                    Value::GpuTensor(proto_handle),
                ],
            )
            .expect("cat with integer gpu like");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.shape(), &[1, 3]);
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    u64::MAX,
                    4,
                    9_007_199_254_740_993,
                ]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_gpu_integer_inputs_fallback_concatenates_exact_storage() {
        test_support::with_test_provider(|provider| {
            let left_values = [u64::MAX, 4];
            let right_values = [9_007_199_254_740_993_u64];
            let left = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&left_values),
                    shape: &[1, 2],
                })
                .expect("upload left");
            let right = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&right_values),
                    shape: &[1, 1],
                })
                .expect("upload right");

            let result = cat_builtin(
                Value::Int(IntValue::I32(2)),
                vec![Value::GpuTensor(left), Value::GpuTensor(right)],
            )
            .expect("cat gpu integers");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpu integer output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.shape(), &[1, 3]);
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    u64::MAX,
                    4,
                    9_007_199_254_740_993,
                ]))
            );
        });
    }

    #[test]
    fn cat_mixed_host_and_gpu_integer_inputs_preserve_exact_storage() {
        test_support::with_test_provider(|provider| {
            let resident_values = [9_007_199_254_740_993_u64];
            let resident = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&resident_values),
                    shape: &[1, 1],
                })
                .expect("upload resident integer");
            let host = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("host integer");

            let result = cat_builtin(
                Value::Int(IntValue::I32(2)),
                vec![Value::GpuTensor(resident), Value::Tensor(host)],
            )
            .expect("mixed residency cat");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX,]))
            );
        });
    }

    #[test]
    fn cat_runmat_only_forms_gate_before_provider_or_data_conversion() {
        let left = Tensor::new(vec![1.0], vec![1, 1]).expect("left");
        let right = Tensor::new(vec![2.0], vec![1, 1]).expect("right");
        let like_error = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![
                Value::Tensor(left.clone()),
                Value::Tensor(right.clone()),
                Value::from("like"),
                Value::Num(0.0),
            ],
        )
        .expect_err("MATLAB mode rejects like form");
        assert_eq!(like_error.identifier(), CAT_LIKE_EXTENSION.error_identifier);

        let complex_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![2]))
                .expect("complex storage");
        let complex =
            ComplexTensor::new_integer(complex_storage, vec![1, 1]).expect("complex integer");
        let Value::ComplexTensor(complex_output) = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![
                Value::ComplexTensor(complex.clone()),
                Value::ComplexTensor(complex),
            ],
        )
        .expect("typed complex integer concatenation is documented") else {
            panic!("expected complex tensor");
        };
        assert_eq!(complex_output.shape(), &[2, 1]);

        test_support::with_test_provider(|provider| {
            let dim_values = [1_u16];
            let dim = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&dim_values),
                    shape: &[1, 1],
                })
                .expect("upload dimension");
            let dim_error = cat_builtin(
                Value::GpuTensor(dim.clone()),
                vec![Value::Tensor(left.clone()), Value::Tensor(right.clone())],
            )
            .expect_err("MATLAB mode rejects resident dimension");
            assert_eq!(
                dim_error.identifier(),
                CAT_RESIDENT_DIMENSION_EXTENSION.error_identifier
            );

            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let result = cat_builtin(
                Value::GpuTensor(dim),
                vec![Value::Tensor(left), Value::Tensor(right)],
            )
            .expect("RunMat mode accepts resident dimension");
            let Value::Tensor(output) = result else {
                panic!("expected host output");
            };
            assert_eq!(output.materialize_f64(), vec![1.0, 2.0]);
        });
    }

    #[test]
    fn cat_gpu_fallback_omits_shaped_empty_geometry() {
        test_support::with_test_provider(|provider| {
            let empty = provider
                .upload(&HostTensorView {
                    data: &[],
                    shape: &[0, 4],
                })
                .expect("upload empty");
            let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let nonempty = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &[2, 3],
                })
                .expect("upload nonempty");
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![Value::GpuTensor(empty), Value::GpuTensor(nonempty)],
            )
            .expect("gpu empty omission");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), &[2, 3]);
            assert_eq!(gathered.materialize_f64(), values);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_logical_mismatch_errors() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let proto = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let err = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
                Value::from("like"),
                Value::LogicalArray(proto),
            ],
        )
        .unwrap_err();
        assert!(err.message().contains("logical"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cat_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

        let cpu_result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        )
        .expect("cat cpu");
        let expected = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: a.shape(),
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: b.shape(),
        };
        let ha = provider.upload(&view_a).expect("upload a");
        let hb = provider.upload(&view_b).expect("upload b");
        let gpu_value = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::GpuTensor(ha.clone()), Value::GpuTensor(hb.clone())],
        )
        .expect("cat gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());

        let transposed = ha
            .with_view(
                runmat_accelerate_api::GpuTensorView::contiguous(vec![2, 2]).with_layout(
                    runmat_accelerate_api::GpuTensorLayout::Transpose(
                        runmat_accelerate_api::TransposeInfo {
                            base_rows: 2,
                            base_cols: 2,
                        },
                    ),
                ),
            )
            .expect("authenticated lazy transpose");
        let refused = block_on(provider.cat_optional(2, &[transposed.clone(), hb.clone()]))
            .expect_err("native copy must not reinterpret lazy transpose storage");
        assert!(refused.is_unsupported());
        assert!(!refused.is_after_effect());
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::GpuTensor(transposed), Value::GpuTensor(hb)],
        )
        .expect("cat fallback for lazy transpose");
        let gathered = test_support::gather(result).expect("gather lazy transpose result");
        assert_eq!(gathered.shape(), &[2, 4]);
        assert_eq!(
            gathered.materialize_f64(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 6.0, 8.0]
        );
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn cat_wgpu_mixed_residency_preserves_wide_integer_storage() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let resident_values = [9_007_199_254_740_993_u64];
        let resident = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&resident_values),
                shape: &[1, 1],
            })
            .expect("upload resident integer");
        let host = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("host integer");

        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::GpuTensor(resident), Value::Tensor(host)],
        )
        .expect("mixed WGPU cat");
        let Value::GpuTensor(handle) = result else {
            panic!("expected resident output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&handle),
            Some(IntegerElementType::U64)
        );
        let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX,]))
        );
    }
}
