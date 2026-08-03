//! MATLAB-compatible `mean` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexStorage, ComplexTensor, IntValue, IntegerStorage, NumericDType, NumericScalar,
    NumericStorage, Tensor, Type, Value,
};
const NAME: &str = "mean";

use runmat_builtins::ResolveContext;

fn mean_type(args: &[Type], ctx: &ResolveContext) -> Type {
    reduce_numeric_type(args, ctx)
}

use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use crate::builtins::common::arg_tokens::tokens_from_values;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    gpu_helpers,
    shape::{canonical_scalar_shape, is_scalar_shape, normalize_scalar_shape},
    tensor,
};
use crate::builtins::math::reduction::type_resolvers::reduce_numeric_type;
use crate::dispatcher;

const MEAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mean reduction result.",
}];

const MEAN_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const MEAN_INPUTS_A_DIM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Dimension selector or vector of dimensions.",
    },
];

const MEAN_INPUTS_A_ALL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "all",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"all\""),
        description: "Reduce across all dimensions.",
    },
];

const MEAN_INPUTS_A_MISSINGFLAG: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "missingflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includemissing\""),
        description: "Missing-data handling mode: \"includemissing\" or \"omitmissing\".",
    },
];

const MEAN_INPUTS_A_OUTTYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "outtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"default\""),
        description: "Output class specifier: \"double\", \"default\", or \"native\".",
    },
];

const MEAN_INPUTS_A_DIM_MISSINGFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Dimension selector or vector of dimensions.",
    },
    BuiltinParamDescriptor {
        name: "missingflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includemissing\""),
        description: "Missing-data handling mode: \"includemissing\" or \"omitmissing\".",
    },
];

const MEAN_INPUTS_A_MISSINGFLAG_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "missingflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includemissing\""),
        description: "Missing-data handling mode: \"includemissing\" or \"omitmissing\".",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Dimension selector or vector of dimensions.",
    },
];

const MEAN_INPUTS_A_WEIGHTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "Weights",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Weights\""),
        description: "Weighting-scheme name.",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonnegative single- or double-precision weighting scheme.",
    },
];

const MEAN_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "M = mean(A)",
        inputs: &MEAN_INPUTS_A,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, dim)",
        inputs: &MEAN_INPUTS_A_DIM,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, \"all\")",
        inputs: &MEAN_INPUTS_A_ALL,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, missingflag)",
        inputs: &MEAN_INPUTS_A_MISSINGFLAG,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, outtype)",
        inputs: &MEAN_INPUTS_A_OUTTYPE,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, dim, missingflag)",
        inputs: &MEAN_INPUTS_A_DIM_MISSINGFLAG,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, missingflag, dim)",
        inputs: &MEAN_INPUTS_A_MISSINGFLAG_DIM,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, vecdim)",
        inputs: &MEAN_INPUTS_A_DIM,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(___, Weights=W)",
        inputs: &MEAN_INPUTS_A_WEIGHTS,
        outputs: &MEAN_OUTPUT,
    },
];

const MEAN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEAN.INVALID_ARGUMENT",
    identifier: Some("RunMat:mean:InvalidArgument"),
    when: "Dimension, missing-data, weights, or output class argument grammar is invalid.",
    message: "mean: invalid argument",
};

const MEAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEAN.INVALID_INPUT",
    identifier: Some("RunMat:mean:InvalidInput"),
    when: "Input values cannot be converted to supported mean reduction domains.",
    message: "mean: invalid input",
};

const MEAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEAN.INTERNAL",
    identifier: Some("RunMat:mean:Internal"),
    when:
        "Reduction execution fails due to conversion, provider, allocation, or coercion operations.",
    message: "mean: internal reduction failure",
};

const MEAN_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MEAN_ERROR_INVALID_ARGUMENT,
    MEAN_ERROR_INVALID_INPUT,
    MEAN_ERROR_INTERNAL,
];

pub const MEAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MEAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MEAN_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::mean")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mean",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_mean_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_mean",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers can specialise mean reductions; omitnan currently falls back to the host.",
};

fn mean_descriptor_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn mean_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    mean_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn mean_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    mean_descriptor_error_with_detail(&MEAN_ERROR_INVALID_ARGUMENT, detail)
}

fn mean_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    mean_descriptor_error_with_detail(&MEAN_ERROR_INVALID_INPUT, detail)
}

fn mean_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    mean_descriptor_error_with_detail(&MEAN_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::mean")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mean",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion fallback currently gathers to host; future kernels will divide the accumulated sum by slice size.",
};

#[derive(Clone)]
struct ParsedArguments {
    axes: MeanAxes,
    nan_mode: ReductionNaN,
    output: OutputTemplate,
    weights: Option<MeanWeights>,
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Double,
    Native,
}

#[derive(Clone)]
struct MeanWeights {
    storage: MeanWeightStorage,
    shape: Vec<usize>,
}

#[derive(Clone)]
enum MeanWeightStorage {
    F64(Vec<f64>),
    F32(Vec<f32>),
}

impl MeanWeights {
    fn len(&self) -> usize {
        match &self.storage {
            MeanWeightStorage::F64(values) => values.len(),
            MeanWeightStorage::F32(values) => values.len(),
        }
    }

    fn f64_at(&self, index: usize) -> f64 {
        match &self.storage {
            MeanWeightStorage::F64(values) => values[index],
            MeanWeightStorage::F32(values) => f64::from(values[index]),
        }
    }

    fn f32_at(&self, index: usize) -> f32 {
        match &self.storage {
            MeanWeightStorage::F64(values) => values[index] as f32,
            MeanWeightStorage::F32(values) => values[index],
        }
    }
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

#[derive(Clone, Copy)]
enum InputClass {
    Double,
    Complex,
    Logical,
    Integer(IntClass),
    Bool,
}

#[derive(Clone, Copy)]
enum IntClass {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

struct InputMeta {
    class: InputClass,
    device: DevicePreference,
    numeric_dtype: Option<NumericDType>,
}

impl InputMeta {
    fn from_value(value: &Value) -> Self {
        let class = match value {
            Value::Complex(_, _) | Value::ComplexTensor(_) => InputClass::Complex,
            Value::LogicalArray(_) => InputClass::Logical,
            Value::Bool(_) => InputClass::Bool,
            Value::Int(i) => InputClass::Integer(IntClass::from_int_value(i)),
            Value::Tensor(tensor) => tensor
                .integer_storage()
                .map_or(InputClass::Double, |storage| {
                    InputClass::Integer(IntClass::from_integer_storage(storage))
                }),
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle) => {
                InputClass::Logical
            }
            Value::GpuTensor(handle) => {
                if let Some(element_type) = runmat_accelerate_api::handle_integer_type(handle) {
                    InputClass::Integer(IntClass::from_element_type(element_type))
                } else if matches!(
                    runmat_accelerate_api::handle_storage(handle),
                    runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                ) {
                    InputClass::Complex
                } else {
                    InputClass::Double
                }
            }
            _ => InputClass::Double,
        };
        let device = match value {
            Value::GpuTensor(_) => DevicePreference::Gpu,
            _ => DevicePreference::Host,
        };
        let numeric_dtype = numeric_dtype_from_value(value);
        Self {
            class,
            device,
            numeric_dtype,
        }
    }
}

fn numeric_dtype_from_value(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Tensor(t) => Some(t.numeric_dtype()),
        Value::ComplexTensor(t) => Some(t.numeric_dtype()),
        Value::Complex(_, _) => Some(NumericDType::F64),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle) {
                return Some(NumericDType::F64);
            }
            if let Some(element_type) = runmat_accelerate_api::handle_integer_type(handle) {
                return Some(IntClass::from_element_type(element_type).numeric_dtype());
            }
            let precision = runmat_accelerate_api::handle_precision(handle).or_else(|| {
                runmat_accelerate_api::provider_for_handle(handle)
                    .map(|provider| provider.precision())
            });
            precision.map(precision_to_dtype)
        }
        Value::Num(_) => Some(NumericDType::F64),
        Value::LogicalArray(_) => Some(NumericDType::F64),
        _ => None,
    }
}

fn precision_to_dtype(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

impl IntClass {
    fn from_integer_storage(storage: &IntegerStorage) -> Self {
        match storage {
            IntegerStorage::I8(_) => IntClass::I8,
            IntegerStorage::I16(_) => IntClass::I16,
            IntegerStorage::I32(_) => IntClass::I32,
            IntegerStorage::I64(_) => IntClass::I64,
            IntegerStorage::U8(_) => IntClass::U8,
            IntegerStorage::U16(_) => IntClass::U16,
            IntegerStorage::U32(_) => IntClass::U32,
            IntegerStorage::U64(_) => IntClass::U64,
        }
    }

    fn from_int_value(value: &IntValue) -> Self {
        match value {
            IntValue::I8(_) => IntClass::I8,
            IntValue::I16(_) => IntClass::I16,
            IntValue::I32(_) => IntClass::I32,
            IntValue::I64(_) => IntClass::I64,
            IntValue::U8(_) => IntClass::U8,
            IntValue::U16(_) => IntClass::U16,
            IntValue::U32(_) => IntClass::U32,
            IntValue::U64(_) => IntClass::U64,
        }
    }

    fn from_element_type(value: runmat_accelerate_api::IntegerElementType) -> Self {
        match value {
            runmat_accelerate_api::IntegerElementType::I8 => IntClass::I8,
            runmat_accelerate_api::IntegerElementType::I16 => IntClass::I16,
            runmat_accelerate_api::IntegerElementType::I32 => IntClass::I32,
            runmat_accelerate_api::IntegerElementType::I64 => IntClass::I64,
            runmat_accelerate_api::IntegerElementType::U8 => IntClass::U8,
            runmat_accelerate_api::IntegerElementType::U16 => IntClass::U16,
            runmat_accelerate_api::IntegerElementType::U32 => IntClass::U32,
            runmat_accelerate_api::IntegerElementType::U64 => IntClass::U64,
        }
    }

    fn numeric_dtype(self) -> NumericDType {
        match self {
            IntClass::I8 => NumericDType::I8,
            IntClass::I16 => NumericDType::I16,
            IntClass::I32 => NumericDType::I32,
            IntClass::I64 => NumericDType::I64,
            IntClass::U8 => NumericDType::U8,
            IntClass::U16 => NumericDType::U16,
            IntClass::U32 => NumericDType::U32,
            IntClass::U64 => NumericDType::U64,
        }
    }
}

#[derive(Clone, Debug)]
enum MeanAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[runtime_builtin(
    name = "mean",
    category = "math/reduction",
    summary = "Average elements of scalars, vectors, matrices, or N-D tensors.",
    keywords = "mean,average,reduction,gpu,omitnan",
    accel = "reduction",
    type_resolver(mean_type),
    descriptor(crate::builtins::math::reduction::mean::MEAN_DESCRIPTOR),
    builtin_path = "crate::builtins::math::reduction::mean"
)]
pub(crate) async fn mean_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    // Normalise argument order defensively:
    // If the primary 'value' is not data-like (e.g., 'all'), but a data-like
    // argument exists in 'rest', swap them so we interpret calls like
    // mean('all', X) as mean(X, 'all').
    let (value, rest) = normalise_mean_call_args(value, rest);

    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(mean_invalid_input(
            "operations involving complex numbers with integer types are not supported",
        ));
    }

    let input_meta = InputMeta::from_value(&value);
    let parsed = parse_arguments(&rest).await?;
    if matches!(parsed.output, OutputTemplate::Native) {
        if let Value::GpuTensor(handle) = &value {
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                return mean_native_integer_gpu(handle.clone(), &parsed).await;
            }
        }
        if let Some(result) = mean_native_integer(&value, &parsed)? {
            return Ok(result);
        }
    }
    let raw = match value {
        Value::GpuTensor(handle) => mean_gpu(handle, &parsed).await?,
        Value::Complex(re, im) => mean_host_complex_scalar(re, im, &parsed)?,
        Value::ComplexTensor(ct) => mean_host_complex_tensor(ct, &parsed)?,
        other => mean_host(other, &parsed)?,
    };
    apply_output_template(raw, &parsed.output, &input_meta).await
}

fn mean_native_integer(value: &Value, parsed: &ParsedArguments) -> BuiltinResult<Option<Value>> {
    let (storage, shape) = match value {
        Value::Int(value) => (
            crate::builtins::math::reduction::integer_native::storage_from_scalar(value),
            vec![1, 1],
        ),
        Value::Tensor(tensor) => {
            let Some(storage) = tensor.integer_storage() else {
                return Ok(None);
            };
            (storage.clone(), tensor.shape.clone())
        }
        _ => return Ok(None),
    };
    if let Some(weights) = parsed.weights.as_ref() {
        let dim = match &parsed.axes {
            MeanAxes::Default => default_dimension_from_shape(&shape),
            MeanAxes::Dim(dim) => *dim,
            MeanAxes::Vec(_) | MeanAxes::All => {
                return Err(mean_invalid_argument(
                    "mean: Weights cannot be combined with vecdim or 'all'",
                ));
            }
        };
        let result = weighted_integer_mean(&storage, &shape, dim, weights)?;
        return Ok(Some(tensor::tensor_into_value(result)));
    }
    let dims = resolve_native_dims(&shape, &parsed.axes)?;
    crate::builtins::math::reduction::integer_native::mean(&storage, &shape, &dims)
        .map(Some)
        .map_err(mean_internal_error)
}

async fn mean_native_integer_gpu(
    handle: GpuTensorHandle,
    parsed: &ParsedArguments,
) -> BuiltinResult<Value> {
    if let Some(weights) = parsed.weights.as_ref() {
        let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
            mean_internal_error("mean: native integer gpuArray requires an acceleration provider")
        })?;
        let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
        let storage = gathered
            .integer_storage()
            .cloned()
            .ok_or_else(|| mean_internal_error("mean: expected gathered integer storage"))?;
        let dim = match &parsed.axes {
            MeanAxes::Default => default_dimension_from_shape(&gathered.shape),
            MeanAxes::Dim(dim) => *dim,
            MeanAxes::Vec(_) | MeanAxes::All => {
                return Err(mean_invalid_argument(
                    "mean: Weights cannot be combined with vecdim or 'all'",
                ));
            }
        };
        let result = weighted_integer_mean(&storage, &gathered.shape, dim, weights)?;
        let uploaded = gpu_helpers::upload_tensor(provider, &result).map_err(|error| {
            mean_internal_error(format!(
                "mean: failed to upload weighted native result: {error}"
            ))
        })?;
        return Ok(Value::GpuTensor(uploaded));
    }
    let dims = resolve_native_dims(&handle.shape, &parsed.axes)?;
    if dims.is_empty() {
        return Ok(Value::GpuTensor(handle));
    }
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        mean_internal_error("mean: native integer gpuArray requires an acceleration provider")
    })?;
    let result = if dims.len() == handle.shape.len() {
        provider.reduce_integer_mean_native(&handle).await
    } else if dims.len() == 1 {
        provider
            .reduce_integer_mean_native_dim(&handle, dims[0])
            .await
    } else {
        provider
            .reduce_integer_mean_native_dims(&handle, &dims)
            .await
    }
    .map_err(|err| mean_internal_error(format!("mean: {err}")))?;
    Ok(Value::GpuTensor(result))
}

fn resolve_native_dims(shape: &[usize], axes: &MeanAxes) -> BuiltinResult<Vec<usize>> {
    let mut dims = match axes {
        MeanAxes::Default => vec![default_dimension_from_shape(shape).saturating_sub(1)],
        MeanAxes::Dim(dim) => vec![dim.saturating_sub(1)],
        MeanAxes::Vec(dims) => dims.iter().map(|dim| dim.saturating_sub(1)).collect(),
        MeanAxes::All => (0..shape.len()).collect(),
    };
    dims.retain(|&dim| dim < shape.len());
    dims.sort_unstable();
    dims.dedup();
    Ok(dims)
}

fn normalise_mean_call_args(value: Value, rest: Vec<Value>) -> (Value, Vec<Value>) {
    if is_data_like(&value) {
        return (value, rest);
    }
    if let Some(idx) = rest.iter().position(is_data_like) {
        let mut rest_mut = rest;
        let new_value = rest_mut.remove(idx);
        let mut new_rest = Vec::with_capacity(rest_mut.len() + 1);
        // Keep the original non-data 'value' (e.g., 'all') in rest so it can be parsed as a keyword
        new_rest.push(value);
        // Append the remaining rest args
        new_rest.extend(rest_mut);
        return (new_value, new_rest);
    }
    (value, rest)
}

fn is_data_like(v: &Value) -> bool {
    matches!(
        v,
        Value::Tensor(_)
            | Value::GpuTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::LogicalArray(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
    )
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<ParsedArguments> {
    let mut axes = MeanAxes::Default;
    let mut axes_set = false;
    let mut nan_mode = ReductionNaN::Include;
    let mut output = OutputTemplate::Default;
    let mut output_set = false;
    let mut weights = None;
    let tokens = tokens_from_values(args);

    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];
        if let Some(crate::builtins::common::arg_tokens::ArgToken::String(text)) = tokens.get(idx) {
            match text.as_str() {
                "weights" => {
                    if weights.is_some() {
                        return Err(mean_invalid_argument(
                            "mean: Weights may be specified only once",
                        ));
                    }
                    let value = args
                        .get(idx + 1)
                        .ok_or_else(|| mean_invalid_argument("mean: Weights requires a value"))?;
                    weights = Some(parse_weights(value).await?);
                    idx += 2;
                    continue;
                }
                "omitnan" | "omitmissing" => {
                    nan_mode = ReductionNaN::Omit;
                    idx += 1;
                    continue;
                }
                "includenan" | "includemissing" => {
                    nan_mode = ReductionNaN::Include;
                    idx += 1;
                    continue;
                }
                "all" => {
                    if axes_set && !matches!(axes, MeanAxes::Default) {
                        return Err(mean_invalid_argument(
                            "mean: 'all' cannot be combined with an explicit dimension",
                        ));
                    }
                    axes = MeanAxes::All;
                    axes_set = true;
                    idx += 1;
                    continue;
                }
                _ => {}
            }
        }
        if let Some(keyword) = keyword_of(arg) {
            match keyword.as_str() {
                "weights" => {
                    if weights.is_some() {
                        return Err(mean_invalid_argument(
                            "mean: Weights may be specified only once",
                        ));
                    }
                    let value = args
                        .get(idx + 1)
                        .ok_or_else(|| mean_invalid_argument("mean: Weights requires a value"))?;
                    weights = Some(parse_weights(value).await?);
                    idx += 2;
                    continue;
                }
                "omitnan" | "omitmissing" => {
                    nan_mode = ReductionNaN::Omit;
                    idx += 1;
                    continue;
                }
                "includenan" | "includemissing" => {
                    nan_mode = ReductionNaN::Include;
                    idx += 1;
                    continue;
                }
                "all" => {
                    if axes_set && !matches!(axes, MeanAxes::Default) {
                        return Err(mean_invalid_argument(
                            "mean: 'all' cannot be combined with an explicit dimension",
                        ));
                    }
                    axes = MeanAxes::All;
                    axes_set = true;
                    idx += 1;
                    continue;
                }
                "default" => {
                    if output_set {
                        return Err(mean_invalid_argument(
                            "mean: multiple output class specifications provided",
                        ));
                    }
                    output = OutputTemplate::Default;
                    output_set = true;
                    idx += 1;
                    continue;
                }
                "double" => {
                    if output_set {
                        return Err(mean_invalid_argument(
                            "mean: multiple output class specifications provided",
                        ));
                    }
                    output = OutputTemplate::Double;
                    output_set = true;
                    idx += 1;
                    continue;
                }
                "native" => {
                    if output_set {
                        return Err(mean_invalid_argument(
                            "mean: multiple output class specifications provided",
                        ));
                    }
                    output = OutputTemplate::Native;
                    output_set = true;
                    idx += 1;
                    continue;
                }
                _ => {}
            }
        }

        if !axes_set || matches!(axes, MeanAxes::Default) {
            if let Some(selection) = parse_axes(arg).await? {
                if matches!(selection, MeanAxes::All)
                    && axes_set
                    && !matches!(axes, MeanAxes::Default)
                {
                    return Err(mean_invalid_argument(
                        "mean: 'all' cannot be combined with an explicit dimension",
                    ));
                }
                axes = selection;
                axes_set = true;
                idx += 1;
                continue;
            }
        }

        if axes_set && !matches!(axes, MeanAxes::Default) {
            if let Some(selection) = parse_axes(arg).await? {
                if matches!(selection, MeanAxes::All) {
                    return Err(mean_invalid_argument(
                        "mean: 'all' cannot be combined with an explicit dimension",
                    ));
                }
                return Err(mean_invalid_argument(
                    "mean: multiple dimension specifications provided",
                ));
            }
        }

        return Err(mean_invalid_argument(format!(
            "mean: unrecognised argument {arg:?}"
        )));
    }

    if weights.is_some() && matches!(axes, MeanAxes::Vec(_) | MeanAxes::All) {
        return Err(mean_invalid_argument(
            "mean: Weights cannot be combined with vecdim or 'all'",
        ));
    }

    Ok(ParsedArguments {
        axes,
        nan_mode,
        output,
        weights,
    })
}

async fn parse_weights(value: &Value) -> BuiltinResult<MeanWeights> {
    let tensor = match value {
        Value::Num(value) => Tensor::new(vec![*value], vec![1, 1])
            .map_err(|error| mean_internal_error(format!("mean: {error}")))?,
        Value::Tensor(tensor) => tensor.clone(),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
            {
                return Err(mean_invalid_argument(
                    "mean: Weights must be single or double",
                ));
            }
            gpu_helpers::gather_tensor_async(handle).await?
        }
        _ => {
            return Err(mean_invalid_argument(
                "mean: Weights must be a single- or double-precision numeric array",
            ));
        }
    };
    let shape = tensor.shape.clone();
    let storage = match tensor.into_numeric_storage().map_err(mean_internal_error)? {
        NumericStorage::F64(values) => MeanWeightStorage::F64(values),
        NumericStorage::F32(values) => MeanWeightStorage::F32(values),
        _ => {
            return Err(mean_invalid_argument(
                "mean: Weights must be single or double",
            ));
        }
    };
    let weights = MeanWeights { storage, shape };
    for index in 0..weights.len() {
        let weight = weights.f64_at(index);
        if weight.is_nan() || weight < 0.0 {
            return Err(mean_invalid_argument(format!(
                "mean: Weights must contain nonnegative values (index {})",
                index + 1
            )));
        }
    }
    Ok(weights)
}

async fn parse_axes(value: &Value) -> BuiltinResult<Option<MeanAxes>> {
    if let Some(text) = value_as_str(value) {
        let lowered = text.trim().to_ascii_lowercase();
        return match lowered.as_str() {
            "all" => Ok(Some(MeanAxes::All)),
            "omitnan" | "includenan" | "omitmissing" | "includemissing" | "double" | "native"
            | "default" | "weights" => Ok(None),
            "" => Err(mean_invalid_argument(
                "mean: dimension string must not be empty",
            )),
            _ => Ok(None),
        };
    }

    let scalar_hint = match value {
        Value::Num(_) | Value::Int(_) => true,
        Value::Tensor(t) => tensor::is_scalar_tensor(t),
        Value::LogicalArray(logical) => logical.data.len() == 1,
        Value::GpuTensor(handle) => {
            is_scalar_shape(&handle.shape) || tensor::element_count(&handle.shape) == 1
        }
        _ => false,
    };

    let dims = match value {
        Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Int(_)
        | Value::Num(_)
        | Value::GpuTensor(_) => tensor::dims_from_value_async(value)
            .await
            .map_err(|err| map_dims_error(err, scalar_hint))?,
        Value::Bool(_) => {
            return Err(mean_invalid_argument("mean: dimension must be numeric"));
        }
        _ => return Ok(None),
    };

    let Some(mut dims) = dims else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Ok(Some(MeanAxes::Default));
    }
    if dims.len() == 1 {
        let dim = dims[0];
        if dim < 1 {
            return Err(mean_invalid_argument("mean: dimension must be >= 1"));
        }
        return Ok(Some(MeanAxes::Dim(dim)));
    }
    for dim in &mut dims {
        if *dim == 0 {
            *dim = 1;
        }
        if *dim < 1 {
            return Err(mean_invalid_argument(
                "mean: dimension entries must be >= 1",
            ));
        }
    }
    Ok(Some(MeanAxes::Vec(dims)))
}

fn value_as_str(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn map_dims_error(message: String, scalar: bool) -> RuntimeError {
    if message.contains("non-negative") {
        if scalar {
            return mean_invalid_argument("mean: dimension must be >= 1");
        }
        return mean_invalid_argument("mean: dimension entries must be >= 1");
    }
    if scalar {
        if message.contains("finite") {
            return mean_invalid_argument("mean: dimension must be finite");
        }
        if message.contains("integer") {
            return mean_invalid_argument("mean: dimension must be an integer");
        }
    }
    mean_invalid_argument("mean: dimension entries must be finite integers")
}

#[derive(Clone, Copy)]
enum WeightLayout {
    Vector,
    Full,
}

fn weighted_dimension(shape: &[usize], axes: &MeanAxes) -> BuiltinResult<usize> {
    match axes {
        MeanAxes::Default => Ok(default_dimension_from_shape(shape)),
        MeanAxes::Dim(dim) => Ok(*dim),
        MeanAxes::Vec(_) | MeanAxes::All => Err(mean_invalid_argument(
            "mean: Weights cannot be combined with vecdim or 'all'",
        )),
    }
}

fn weighted_computation_dtype(input: NumericDType, output: &OutputTemplate) -> NumericDType {
    match output {
        OutputTemplate::Double => NumericDType::F64,
        OutputTemplate::Default => {
            if input == NumericDType::F32 {
                NumericDType::F32
            } else {
                NumericDType::F64
            }
        }
        OutputTemplate::Native => input,
    }
}

fn validate_weight_shape(
    weights: &MeanWeights,
    input_shape: &[usize],
    reduce_len: usize,
) -> BuiltinResult<WeightLayout> {
    if is_vector_shape(&weights.shape) {
        if weights.len() != reduce_len {
            return Err(mean_invalid_argument(format!(
                "mean: vector Weights length {} must match operating dimension length {reduce_len}",
                weights.len()
            )));
        }
        return Ok(WeightLayout::Vector);
    }
    if !matlab_shape_equal(&weights.shape, input_shape) {
        return Err(mean_invalid_argument(format!(
            "mean: nonvector Weights shape {:?} must match input shape {:?}",
            weights.shape, input_shape
        )));
    }
    Ok(WeightLayout::Full)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] | [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        _ => false,
    }
}

fn matlab_shape_equal(left: &[usize], right: &[usize]) -> bool {
    let rank = left.len().max(right.len());
    (0..rank).all(|index| {
        left.get(index).copied().unwrap_or(1) == right.get(index).copied().unwrap_or(1)
    })
}

fn normalized_weight_f64(weight: f64, scale: f64) -> f64 {
    if scale.is_infinite() {
        if weight.is_infinite() {
            1.0
        } else {
            0.0
        }
    } else {
        weight / scale
    }
}

fn normalized_weight_f32(weight: f32, scale: f32) -> f32 {
    if scale.is_infinite() {
        if weight.is_infinite() {
            1.0
        } else {
            0.0
        }
    } else {
        weight / scale
    }
}

fn weighted_real_mean(
    tensor: &Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
    weights: &MeanWeights,
    dtype: NumericDType,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(mean_invalid_argument("mean: dimension must be >= 1"));
    }
    let reduce_len = if dim <= tensor.shape.len() {
        tensor.shape[dim - 1]
    } else {
        1
    };
    let layout = validate_weight_shape(weights, &tensor.shape, reduce_len)?;
    if dim > tensor.shape.len() {
        return Ok(tensor::coerce_tensor_dtype(tensor.clone(), dtype));
    }
    let output_shape = reduction_shape(&tensor.shape, dim).unwrap_or_else(|| tensor.shape.clone());
    if reduce_len == 0 || tensor.is_empty() {
        return match dtype {
            NumericDType::F32 => Tensor::from_f32(
                vec![f32::NAN; tensor::element_count(&output_shape)],
                output_shape,
            ),
            _ => Tensor::new(
                vec![f64::NAN; tensor::element_count(&output_shape)],
                output_shape,
            ),
        }
        .map_err(|error| mean_internal_error(format!("mean: {error}")));
    }

    match dtype {
        NumericDType::F32 => {
            weighted_real_mean_f32(tensor, dim, nan_mode, weights, layout, output_shape)
        }
        _ => weighted_real_mean_f64(tensor, dim, nan_mode, weights, layout, output_shape),
    }
}

fn weighted_real_mean_f64(
    tensor: &Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
    weights: &MeanWeights,
    layout: WeightLayout,
    output_shape: Vec<usize>,
) -> BuiltinResult<Tensor> {
    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let mut output = Vec::with_capacity(tensor::element_count(&output_shape));

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut candidates = Vec::with_capacity(reduce_len);
            let mut saw_missing = false;
            for k in 0..reduce_len {
                let index = before + k * stride_before + after * stride_before * reduce_len;
                let value = tensor
                    .numeric_value_at(index)
                    .expect("weighted mean index is in bounds");
                if numeric_scalar_is_nan(value) {
                    saw_missing = true;
                    if nan_mode == ReductionNaN::Include {
                        break;
                    }
                    continue;
                }
                let weight_index = match layout {
                    WeightLayout::Vector => k,
                    WeightLayout::Full => index,
                };
                candidates.push((value, weights.f64_at(weight_index)));
            }
            if saw_missing && nan_mode == ReductionNaN::Include {
                output.push(f64::NAN);
                continue;
            }
            let scale = candidates
                .iter()
                .map(|(_, weight)| *weight)
                .fold(0.0_f64, f64::max);
            if candidates.is_empty() || scale == 0.0 {
                output.push(f64::NAN);
                continue;
            }
            let anchor = candidates[0].0;
            let mut total = 0.0;
            let mut delta = 0.0;
            for (value, weight) in candidates {
                let weight = normalized_weight_f64(weight, scale);
                total += weight;
                delta += weight * numeric_scalar_difference_f64(value, anchor);
            }
            output.push(numeric_scalar_to_f64(anchor) + delta / total);
        }
    }
    Tensor::new(output, output_shape).map_err(|error| mean_internal_error(format!("mean: {error}")))
}

fn weighted_real_mean_f32(
    tensor: &Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
    weights: &MeanWeights,
    layout: WeightLayout,
    output_shape: Vec<usize>,
) -> BuiltinResult<Tensor> {
    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let mut output = Vec::with_capacity(tensor::element_count(&output_shape));

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut candidates = Vec::with_capacity(reduce_len);
            let mut saw_missing = false;
            for k in 0..reduce_len {
                let index = before + k * stride_before + after * stride_before * reduce_len;
                let value = tensor
                    .numeric_value_at(index)
                    .expect("weighted mean index is in bounds");
                let value = numeric_scalar_to_f32(value);
                if value.is_nan() {
                    saw_missing = true;
                    if nan_mode == ReductionNaN::Include {
                        break;
                    }
                    continue;
                }
                let weight_index = match layout {
                    WeightLayout::Vector => k,
                    WeightLayout::Full => index,
                };
                candidates.push((value, weights.f32_at(weight_index)));
            }
            if saw_missing && nan_mode == ReductionNaN::Include {
                output.push(f32::NAN);
                continue;
            }
            let scale = candidates
                .iter()
                .map(|(_, weight)| *weight)
                .fold(0.0_f32, f32::max);
            if candidates.is_empty() || scale == 0.0 {
                output.push(f32::NAN);
                continue;
            }
            let anchor = candidates[0].0;
            let mut total = 0.0_f32;
            let mut delta = 0.0_f32;
            for (value, weight) in candidates {
                let weight = normalized_weight_f32(weight, scale);
                total += weight;
                delta += weight * (value - anchor);
            }
            output.push(anchor + delta / total);
        }
    }
    Tensor::from_f32(output, output_shape)
        .map_err(|error| mean_internal_error(format!("mean: {error}")))
}

fn numeric_scalar_is_nan(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.is_nan(),
        NumericScalar::F32(value) => value.is_nan(),
        NumericScalar::I8(_)
        | NumericScalar::I16(_)
        | NumericScalar::I32(_)
        | NumericScalar::I64(_)
        | NumericScalar::U8(_)
        | NumericScalar::U16(_)
        | NumericScalar::U32(_)
        | NumericScalar::U64(_) => false,
    }
}

fn numeric_scalar_to_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        value => value
            .into_int_value()
            .expect("nonfloating numeric scalar is integer")
            .to_f64(),
    }
}

fn numeric_scalar_to_f32(value: NumericScalar) -> f32 {
    match value {
        NumericScalar::F64(value) => value as f32,
        NumericScalar::F32(value) => value,
        value => value
            .into_int_value()
            .expect("nonfloating numeric scalar is integer")
            .to_f64() as f32,
    }
}

fn numeric_scalar_difference_f64(value: NumericScalar, anchor: NumericScalar) -> f64 {
    match (value, anchor) {
        (NumericScalar::I8(value), NumericScalar::I8(anchor)) => {
            f64::from(value as i16 - anchor as i16)
        }
        (NumericScalar::I16(value), NumericScalar::I16(anchor)) => {
            f64::from(value as i32 - anchor as i32)
        }
        (NumericScalar::I32(value), NumericScalar::I32(anchor)) => {
            (value as i64 - anchor as i64) as f64
        }
        (NumericScalar::I64(value), NumericScalar::I64(anchor)) => {
            (value as i128 - anchor as i128) as f64
        }
        (NumericScalar::U8(value), NumericScalar::U8(anchor)) => {
            f64::from(value as i16 - anchor as i16)
        }
        (NumericScalar::U16(value), NumericScalar::U16(anchor)) => {
            f64::from(value as i32 - anchor as i32)
        }
        (NumericScalar::U32(value), NumericScalar::U32(anchor)) => {
            (value as i64 - anchor as i64) as f64
        }
        (NumericScalar::U64(value), NumericScalar::U64(anchor)) => {
            (value as i128 - anchor as i128) as f64
        }
        _ => numeric_scalar_to_f64(value) - numeric_scalar_to_f64(anchor),
    }
}

fn weighted_integer_mean(
    storage: &IntegerStorage,
    shape: &[usize],
    dim: usize,
    weights: &MeanWeights,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(mean_invalid_argument("mean: dimension must be >= 1"));
    }
    let reduce_len = if dim <= shape.len() {
        shape[dim - 1]
    } else {
        1
    };
    let layout = validate_weight_shape(weights, shape, reduce_len)?;
    if dim > shape.len() || reduce_len == 1 {
        return Tensor::new_integer(storage.clone(), shape.to_vec())
            .map_err(|error| mean_internal_error(format!("mean: {error}")));
    }
    let output_shape = reduction_shape(shape, dim).unwrap_or_else(|| shape.to_vec());
    if reduce_len == 0 || storage.is_empty() {
        return Tensor::new_integer(
            storage.zeros_like(tensor::element_count(&output_shape)),
            output_shape,
        )
        .map_err(|error| mean_internal_error(format!("mean: {error}")));
    }

    let dim_index = dim - 1;
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim..]);
    let exact = storage.exact_values();
    let mut output = Vec::with_capacity(tensor::element_count(&output_shape));
    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut candidates = Vec::with_capacity(reduce_len);
            for k in 0..reduce_len {
                let index = before + k * stride_before + after * stride_before * reduce_len;
                let weight_index = match layout {
                    WeightLayout::Vector => k,
                    WeightLayout::Full => index,
                };
                candidates.push((
                    int_value_to_i128(&exact[index]),
                    weights.f64_at(weight_index),
                ));
            }
            output.push(weighted_integer_value(
                &candidates,
                exact.first().expect("nonempty integer storage"),
            ));
        }
    }
    let output = storage
        .from_same_class_values(output)
        .map_err(mean_internal_error)?;
    Tensor::new_integer(output, output_shape)
        .map_err(|error| mean_internal_error(format!("mean: {error}")))
}

fn weighted_integer_value(candidates: &[(i128, f64)], prototype: &IntValue) -> IntValue {
    let scale = candidates
        .iter()
        .map(|(_, weight)| *weight)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return integer_value_from_i128_like(prototype, 0);
    }
    let anchor = candidates[0].0;
    let mut total = 0.0;
    let mut delta = 0.0;
    for &(value, weight) in candidates {
        let weight = normalized_weight_f64(weight, scale);
        total += weight;
        delta += weight * (value - anchor) as f64;
    }
    let delta = delta / total;
    let final_is_nonnegative = anchor as f64 + delta >= 0.0;
    let rounded_offset = if final_is_nonnegative {
        (delta + 0.5).floor()
    } else {
        (delta - 0.5).ceil()
    };
    let rounded = anchor.saturating_add(rounded_offset as i128);
    integer_value_from_i128_like(prototype, rounded)
}

fn int_value_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn integer_value_from_i128_like(prototype: &IntValue, value: i128) -> IntValue {
    match prototype {
        IntValue::I8(_) => IntValue::I8(value.clamp(i8::MIN as i128, i8::MAX as i128) as i8),
        IntValue::I16(_) => IntValue::I16(value.clamp(i16::MIN as i128, i16::MAX as i128) as i16),
        IntValue::I32(_) => IntValue::I32(value.clamp(i32::MIN as i128, i32::MAX as i128) as i32),
        IntValue::I64(_) => IntValue::I64(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64),
        IntValue::U8(_) => IntValue::U8(value.clamp(0, u8::MAX as i128) as u8),
        IntValue::U16(_) => IntValue::U16(value.clamp(0, u16::MAX as i128) as u16),
        IntValue::U32(_) => IntValue::U32(value.clamp(0, u32::MAX as i128) as u32),
        IntValue::U64(_) => IntValue::U64(value.clamp(0, u64::MAX as i128) as u64),
    }
}

fn weighted_complex_mean(
    tensor: &ComplexTensor,
    dim: usize,
    nan_mode: ReductionNaN,
    weights: &MeanWeights,
    dtype: NumericDType,
) -> BuiltinResult<ComplexTensor> {
    if dim == 0 {
        return Err(mean_invalid_argument("mean: dimension must be >= 1"));
    }
    let reduce_len = if dim <= tensor.shape.len() {
        tensor.shape[dim - 1]
    } else {
        1
    };
    let layout = validate_weight_shape(weights, &tensor.shape, reduce_len)?;
    if dim > tensor.shape.len() {
        return coerce_complex_tensor_dtype(tensor.clone(), dtype);
    }
    let output_shape = reduction_shape(&tensor.shape, dim).unwrap_or_else(|| tensor.shape.clone());
    if reduce_len == 0 || tensor.is_empty() {
        return match dtype {
            NumericDType::F32 => ComplexTensor::from_f32(
                vec![(f32::NAN, f32::NAN); tensor::element_count(&output_shape)],
                output_shape,
            ),
            _ => ComplexTensor::new(
                vec![(f64::NAN, f64::NAN); tensor::element_count(&output_shape)],
                output_shape,
            ),
        }
        .map_err(|error| mean_internal_error(format!("mean: {error}")));
    }
    match dtype {
        NumericDType::F32 => {
            weighted_complex_mean_f32(tensor, dim, nan_mode, weights, layout, output_shape)
        }
        _ => weighted_complex_mean_f64(tensor, dim, nan_mode, weights, layout, output_shape),
    }
}

fn weighted_complex_mean_f64(
    tensor: &ComplexTensor,
    dim: usize,
    nan_mode: ReductionNaN,
    weights: &MeanWeights,
    layout: WeightLayout,
    output_shape: Vec<usize>,
) -> BuiltinResult<ComplexTensor> {
    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let mut output = Vec::with_capacity(tensor::element_count(&output_shape));
    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut candidates = Vec::with_capacity(reduce_len);
            let mut saw_missing = false;
            for k in 0..reduce_len {
                let index = before + k * stride_before + after * stride_before * reduce_len;
                let (real, imag) = tensor
                    .numeric_value_at(index)
                    .expect("weighted complex mean index is in bounds");
                let real = numeric_scalar_to_f64(real);
                let imag = numeric_scalar_to_f64(imag);
                if real.is_nan() || imag.is_nan() {
                    saw_missing = true;
                    if nan_mode == ReductionNaN::Include {
                        break;
                    }
                    continue;
                }
                let weight_index = match layout {
                    WeightLayout::Vector => k,
                    WeightLayout::Full => index,
                };
                candidates.push(((real, imag), weights.f64_at(weight_index)));
            }
            if saw_missing && nan_mode == ReductionNaN::Include {
                output.push((f64::NAN, f64::NAN));
                continue;
            }
            let scale = candidates
                .iter()
                .map(|(_, weight)| *weight)
                .fold(0.0_f64, f64::max);
            if candidates.is_empty() || scale == 0.0 {
                output.push((f64::NAN, f64::NAN));
                continue;
            }
            let anchor = candidates[0].0;
            let mut total = 0.0;
            let mut delta_real = 0.0;
            let mut delta_imag = 0.0;
            for ((real, imag), weight) in candidates {
                let weight = normalized_weight_f64(weight, scale);
                total += weight;
                delta_real += weight * (real - anchor.0);
                delta_imag += weight * (imag - anchor.1);
            }
            output.push((anchor.0 + delta_real / total, anchor.1 + delta_imag / total));
        }
    }
    ComplexTensor::new(output, output_shape)
        .map_err(|error| mean_internal_error(format!("mean: {error}")))
}

fn weighted_complex_mean_f32(
    tensor: &ComplexTensor,
    dim: usize,
    nan_mode: ReductionNaN,
    weights: &MeanWeights,
    layout: WeightLayout,
    output_shape: Vec<usize>,
) -> BuiltinResult<ComplexTensor> {
    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let mut output = Vec::with_capacity(tensor::element_count(&output_shape));
    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut candidates = Vec::with_capacity(reduce_len);
            let mut saw_missing = false;
            for k in 0..reduce_len {
                let index = before + k * stride_before + after * stride_before * reduce_len;
                let (real, imag) = tensor
                    .numeric_value_at(index)
                    .expect("weighted complex mean index is in bounds");
                let real = numeric_scalar_to_f32(real);
                let imag = numeric_scalar_to_f32(imag);
                if real.is_nan() || imag.is_nan() {
                    saw_missing = true;
                    if nan_mode == ReductionNaN::Include {
                        break;
                    }
                    continue;
                }
                let weight_index = match layout {
                    WeightLayout::Vector => k,
                    WeightLayout::Full => index,
                };
                candidates.push(((real, imag), weights.f32_at(weight_index)));
            }
            if saw_missing && nan_mode == ReductionNaN::Include {
                output.push((f32::NAN, f32::NAN));
                continue;
            }
            let scale = candidates
                .iter()
                .map(|(_, weight)| *weight)
                .fold(0.0_f32, f32::max);
            if candidates.is_empty() || scale == 0.0 {
                output.push((f32::NAN, f32::NAN));
                continue;
            }
            let anchor = candidates[0].0;
            let mut total = 0.0_f32;
            let mut delta_real = 0.0_f32;
            let mut delta_imag = 0.0_f32;
            for ((real, imag), weight) in candidates {
                let weight = normalized_weight_f32(weight, scale);
                total += weight;
                delta_real += weight * (real - anchor.0);
                delta_imag += weight * (imag - anchor.1);
            }
            output.push((anchor.0 + delta_real / total, anchor.1 + delta_imag / total));
        }
    }
    ComplexTensor::from_f32(output, output_shape)
        .map_err(|error| mean_internal_error(format!("mean: {error}")))
}

fn coerce_complex_tensor_dtype(
    tensor: ComplexTensor,
    dtype: NumericDType,
) -> BuiltinResult<ComplexTensor> {
    let shape = tensor.shape.clone();
    match (tensor.into_complex_storage(), dtype) {
        (ComplexStorage::F64(values), NumericDType::F64) => {
            ComplexTensor::new(values, shape).map_err(mean_internal_error)
        }
        (ComplexStorage::F32(values), NumericDType::F32) => {
            ComplexTensor::from_f32(values, shape).map_err(mean_internal_error)
        }
        (ComplexStorage::F32(values), NumericDType::F64) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(real, imag)| (f64::from(real), f64::from(imag)))
                .collect(),
            shape,
        )
        .map_err(mean_internal_error),
        (ComplexStorage::F64(values), NumericDType::F32) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(real, imag)| (real as f32, imag as f32))
                .collect(),
            shape,
        )
        .map_err(mean_internal_error),
        (ComplexStorage::Integer(_), _) => Err(mean_invalid_input(
            "mean: complex integer input is not supported",
        )),
        (_, dtype) => Err(mean_invalid_argument(format!(
            "mean: complex output cannot use {}",
            dtype.class_name()
        ))),
    }
}

fn mean_host(value: Value, args: &ParsedArguments) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("mean", value).map_err(mean_invalid_input)?;
    let reduced = if let Some(weights) = args.weights.as_ref() {
        let dim = weighted_dimension(&tensor.shape, &args.axes)?;
        let dtype = weighted_computation_dtype(tensor.numeric_dtype(), &args.output);
        weighted_real_mean(&tensor, dim, args.nan_mode, weights, dtype)?
    } else {
        mean_tensor(tensor, args.axes.clone(), args.nan_mode)?
    };
    Ok(tensor::tensor_into_value(reduced))
}

fn mean_host_complex_scalar(re: f64, im: f64, args: &ParsedArguments) -> BuiltinResult<Value> {
    let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
        .map_err(|e| mean_internal_error(format!("mean: {e}")))?;
    mean_host_complex_tensor(tensor, args)
}

fn mean_host_complex_tensor(tensor: ComplexTensor, args: &ParsedArguments) -> BuiltinResult<Value> {
    let reduced = if let Some(weights) = args.weights.as_ref() {
        let dim = weighted_dimension(&tensor.shape, &args.axes)?;
        let dtype = weighted_computation_dtype(tensor.numeric_dtype(), &args.output);
        weighted_complex_mean(&tensor, dim, args.nan_mode, weights, dtype)?
    } else {
        mean_complex_tensor(tensor, args.axes.clone(), args.nan_mode)?
    };
    Ok(complex_tensor_into_value(reduced))
}

async fn mean_gpu(handle: GpuTensorHandle, args: &ParsedArguments) -> BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(weights) = args.weights.as_ref() {
        let provider = runmat_accelerate_api::provider_for_handle(&handle)
            .or_else(runmat_accelerate_api::provider)
            .ok_or_else(|| mean_internal_error("mean: GPU input has no owning provider"))?;
        if runmat_accelerate_api::handle_storage(&handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            let gathered = dispatcher::gather_if_needed_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|error| mean_internal_error(format!("mean: {error}")))?;
            let Value::ComplexTensor(gathered) = gathered else {
                return Err(mean_internal_error(
                    "mean: expected gathered complex GPU storage",
                ));
            };
            let dim = weighted_dimension(&gathered.shape, &args.axes)?;
            let input_dtype = numeric_dtype_from_value(&Value::GpuTensor(handle))
                .unwrap_or_else(|| gathered.numeric_dtype());
            let dtype = weighted_computation_dtype(input_dtype, &args.output);
            let reduced = weighted_complex_mean(&gathered, dim, args.nan_mode, weights, dtype)?;
            let uploaded =
                gpu_helpers::upload_complex_tensor(provider, &reduced).map_err(|error| {
                    mean_internal_error(format!(
                        "mean: failed to upload weighted complex GPU result: {error}"
                    ))
                })?;
            return Ok(Value::GpuTensor(uploaded));
        }
        let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
        let dim = weighted_dimension(&gathered.shape, &args.axes)?;
        let input_dtype = numeric_dtype_from_value(&Value::GpuTensor(handle.clone()))
            .unwrap_or_else(|| gathered.numeric_dtype());
        let dtype = weighted_computation_dtype(input_dtype, &args.output);
        let reduced = weighted_real_mean(&gathered, dim, args.nan_mode, weights, dtype)?;
        let uploaded = gpu_helpers::upload_tensor(provider, &reduced).map_err(|error| {
            mean_internal_error(format!(
                "mean: failed to upload weighted GPU result: {error}"
            ))
        })?;
        return Ok(Value::GpuTensor(uploaded));
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        // Include-NaN: use provider reduce_mean_* hooks
        if args.nan_mode == ReductionNaN::Include {
            if let Some(device_result) = mean_gpu_try(provider, &handle, &args.axes).await {
                return Ok(Value::GpuTensor(device_result));
            }
        } else {
            // Omit-NaN: compute fully on device via cleaned sum and non-NaN counts
            if let Some(device_result) = mean_gpu_omitnan(provider, &handle, &args.axes).await {
                return Ok(Value::GpuTensor(device_result));
            }
        }
    }

    let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
    let reduced = mean_tensor(gathered, args.axes.clone(), args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

async fn mean_gpu_try(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    axes: &MeanAxes,
) -> Option<GpuTensorHandle> {
    match axes {
        MeanAxes::Default => {
            if is_scalar_shape(&handle.shape) {
                return Some(handle.clone());
            }
            let dim = default_dimension_from_shape(&handle.shape);
            reduce_mean_dim_gpu(provider, handle.clone(), dim).await
        }
        MeanAxes::Dim(dim) => reduce_mean_dim_gpu(provider, handle.clone(), *dim).await,
        MeanAxes::Vec(dims) => {
            // Prefer provider N-D reduce if available
            let mut dims0: Vec<usize> = dims
                .iter()
                .filter_map(|&d| if d > 0 { Some(d - 1) } else { None })
                .collect();
            dims0.sort_unstable();
            dims0.dedup();
            if !dims0.is_empty() {
                if let Ok(out) = provider.reduce_mean_nd(handle, &dims0).await {
                    return Some(out);
                }
            }
            // Try fast permute+2D fallback
            if let Some(nd) = reduce_mean_vecdim_nd_gpu(provider, handle, dims).await {
                return Some(nd);
            }
            // Sequential per-dimension reductions
            let mut result = handle.clone();
            let mut dims_sorted = dims.clone();
            dims_sorted.sort_unstable();
            dims_sorted.dedup();
            for dim in dims_sorted {
                if is_scalar_shape(&result.shape) {
                    break;
                }
                result = reduce_mean_dim_gpu(provider, result, dim).await?;
            }
            Some(result)
        }
        MeanAxes::All => {
            if is_scalar_shape(&handle.shape) {
                return Some(handle.clone());
            }
            match provider.reduce_mean(handle).await {
                Ok(out) => Some(out),
                Err(err) => {
                    log::trace!("mean: provider reduce_mean fallback triggered: {err}");
                    let rank = handle.shape.len();
                    if is_scalar_shape(&handle.shape) || rank == 0 {
                        Some(handle.clone())
                    } else {
                        let dims: Vec<usize> = (1..=rank).collect();
                        if let Some(result) =
                            reduce_mean_vecdim_nd_gpu(provider, handle, &dims).await
                        {
                            Some(result)
                        } else {
                            let mut result = handle.clone();
                            for dim in 1..=rank {
                                match reduce_mean_dim_gpu(provider, result, dim).await {
                                    Some(updated) => result = updated,
                                    None => return None,
                                }
                            }
                            Some(result)
                        }
                    }
                }
            }
        }
    }
}

async fn reduce_mean_dim_gpu(
    provider: &dyn AccelProvider,
    handle: GpuTensorHandle,
    dim: usize,
) -> Option<GpuTensorHandle> {
    if dim == 0 {
        return None;
    }
    if handle.shape.len() < dim {
        return Some(handle);
    }
    provider
        .reduce_mean_dim(&handle, dim - 1)
        .await
        .map_err(|err| {
            log::trace!("mean: provider reduce_mean_dim fallback triggered: {err}");
            err
        })
        .ok()
}

/// Reduce mean across multiple (1-based) dimensions in a single device pass by
/// permuting reduce dims to the front, reshaping to 2-D, reducing rows, and
/// reshaping/permuting back to the original order with size-1 dims preserved.
// (N-D mean fast path omitted for now; sequential per-dimension GPU reductions used instead.)
async fn reduce_mean_vecdim_nd_gpu(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    dims_1based: &[usize],
) -> Option<GpuTensorHandle> {
    let rank = handle.shape.len();
    if rank == 0 || dims_1based.is_empty() {
        return Some(handle.clone());
    }
    // Convert to 0-based and filter in-bounds
    let mut reduce_dims: Vec<usize> = dims_1based
        .iter()
        .filter_map(|&d| {
            if d > 0 && d <= rank {
                Some(d - 1)
            } else {
                None
            }
        })
        .collect();
    if reduce_dims.is_empty() {
        return Some(handle.clone());
    }
    reduce_dims.sort_unstable();
    reduce_dims.dedup();
    // Kept dims
    let kept_dims: Vec<usize> = (0..rank).filter(|i| !reduce_dims.contains(i)).collect();
    // Permute reduced dims first
    let mut order: Vec<usize> = Vec::with_capacity(rank);
    order.extend_from_slice(&reduce_dims);
    order.extend_from_slice(&kept_dims);
    let permuted = provider.permute(handle, &order).ok()?;
    // Compute rows/cols
    let mut reduce_len: usize = 1;
    for &d in &reduce_dims {
        reduce_len = reduce_len.saturating_mul(handle.shape[d]);
    }
    let total_elems: usize = handle.shape.iter().copied().product();
    if reduce_len == 0 || total_elems == 0 {
        let _ = provider.free(&permuted);
        return provider.fill(&[1, 1], f64::NAN).ok();
    }
    let num_slices = total_elems / reduce_len;
    // Reshape permuted view to [rows, cols]
    let reshaped2d = provider
        .reshape(&permuted, &[reduce_len, num_slices])
        .ok()?;
    // Reduce along rows (dim 0) -> [1, num_slices]
    let reduced_rows = provider.reduce_mean_dim(&reshaped2d, 0).await.ok()?;
    let _ = provider.free(&reshaped2d);
    let _ = provider.free(&permuted);
    // Reshape to kept sizes (permuted order)
    let kept_sizes: Vec<usize> = kept_dims.iter().map(|&d| handle.shape[d]).collect();
    let kept_shape = if kept_sizes.is_empty() {
        vec![1, 1]
    } else {
        kept_sizes.clone()
    };
    let reshaped_kept = provider.reshape(&reduced_rows, &kept_shape).ok()?;
    let _ = provider.free(&reduced_rows);
    // Expand permuted shape by inserting ones for reduced axes
    let mut expanded_perm_shape: Vec<usize> = Vec::with_capacity(rank);
    expanded_perm_shape.extend(std::iter::repeat_n(1usize, reduce_dims.len()));
    expanded_perm_shape.extend_from_slice(&kept_sizes);
    let expanded = provider
        .reshape(&reshaped_kept, &expanded_perm_shape)
        .ok()?;
    let _ = provider.free(&reshaped_kept);
    // Inverse permute back to original axis order
    let mut inv_order = vec![0usize; rank];
    for (dst, &src) in order.iter().enumerate() {
        inv_order[src] = dst;
    }
    let out = provider.permute(&expanded, &inv_order).ok()?;
    let _ = provider.free(&expanded);
    Some(out)
}

async fn mean_gpu_omitnan(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    axes: &MeanAxes,
) -> Option<GpuTensorHandle> {
    // Early return for empty dim selection
    let dims_in_bounds: Vec<usize> = match axes {
        MeanAxes::Default => {
            if is_scalar_shape(&handle.shape) {
                return Some(handle.clone());
            }
            vec![default_dimension_from_shape(&handle.shape) - 1]
        }
        MeanAxes::Dim(d) => {
            if *d == 0 || *d > handle.shape.len() {
                return Some(handle.clone());
            }
            vec![*d - 1]
        }
        MeanAxes::Vec(v) => {
            let mut dims: Vec<usize> = v
                .iter()
                .filter_map(|&d| {
                    if d > 0 && d <= handle.shape.len() {
                        Some(d - 1)
                    } else {
                        None
                    }
                })
                .collect();
            dims.sort_unstable();
            dims.dedup();
            dims
        }
        MeanAxes::All => {
            if is_scalar_shape(&handle.shape) {
                return Some(handle.clone());
            }
            (0..handle.shape.len()).collect()
        }
    };

    if dims_in_bounds.is_empty() {
        return Some(handle.clone());
    }

    // Build cleaned values and not-NaN counts on device
    let cleaned = provider.map_nan_to_zero(handle).ok()?;
    let mask = provider.not_nan_mask(handle).ok()?;

    // Reduce cleaned (sum) and mask (count) along the requested dims
    let mut sum_h = cleaned.clone();
    let mut cnt_h = mask.clone();
    for &dim in &dims_in_bounds {
        sum_h = provider.reduce_sum_dim(&sum_h, dim).await.ok()?;
        cnt_h = provider.reduce_sum_dim(&cnt_h, dim).await.ok()?;
    }

    // mean = sum ./ count (0/0 -> NaN when all NaN)
    let out = provider.elem_div(&sum_h, &cnt_h).await.ok()?;

    // Free intermediates
    let _ = provider.free(&cleaned);
    let _ = provider.free(&mask);
    let _ = provider.free(&sum_h);
    let _ = provider.free(&cnt_h);

    Some(out)
}

fn mean_tensor(tensor: Tensor, axes: MeanAxes, nan_mode: ReductionNaN) -> BuiltinResult<Tensor> {
    match axes {
        MeanAxes::Default => {
            let dim = default_dimension(&tensor);
            reduce_tensor_mean_dim(&tensor, dim, nan_mode)
        }
        MeanAxes::Dim(dim) => reduce_tensor_mean_dim(&tensor, dim, nan_mode),
        MeanAxes::Vec(dims) => {
            let mut current = tensor;
            let mut dims_sorted = dims;
            dims_sorted.sort_unstable();
            dims_sorted.dedup();
            for dim in dims_sorted {
                current = reduce_tensor_mean_dim(&current, dim, nan_mode)?;
            }
            Ok(current)
        }
        MeanAxes::All => mean_tensor_all(&tensor, nan_mode),
    }
}

fn mean_tensor_all(tensor: &Tensor, nan_mode: ReductionNaN) -> BuiltinResult<Tensor> {
    if is_scalar_shape(&tensor.shape) {
        return reduce_tensor_mean_dim(tensor, 1, nan_mode);
    }
    let values = tensor::tensor_values_f64(tensor);
    let total_elems = tensor
        .shape
        .iter()
        .copied()
        .map(|dim| dim.max(1))
        .fold(1usize, |acc, dim| acc.saturating_mul(dim));
    if total_elems == 0 || values.is_empty() {
        return Tensor::new(vec![f64::NAN], vec![1, 1])
            .map_err(|e| mean_internal_error(format!("mean: {e}")));
    }
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut saw_nan = false;
    match nan_mode {
        ReductionNaN::Include => {
            for &value in &values {
                if value.is_nan() {
                    saw_nan = true;
                    break;
                }
                sum += value;
            }
            let result = if saw_nan {
                f64::NAN
            } else {
                sum / (total_elems as f64)
            };
            Tensor::new(vec![result], vec![1, 1])
                .map_err(|e| mean_internal_error(format!("mean: {e}")))
        }
        ReductionNaN::Omit => {
            for &value in &values {
                if value.is_nan() {
                    continue;
                }
                sum += value;
                count += 1;
            }
            let result = if count == 0 {
                f64::NAN
            } else {
                sum / (count as f64)
            };
            Tensor::new(vec![result], vec![1, 1])
                .map_err(|e| mean_internal_error(format!("mean: {e}")))
        }
    }
}

fn reduce_tensor_mean_dim(
    tensor: &Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(mean_internal_error("mean: dimension must be >= 1"));
    }

    if is_scalar_shape(&tensor.shape) {
        let value = tensor::tensor_values_f64(tensor)
            .into_iter()
            .next()
            .unwrap_or(f64::NAN);
        let result = match nan_mode {
            ReductionNaN::Include => value,
            ReductionNaN::Omit => {
                if value.is_nan() {
                    f64::NAN
                } else {
                    value
                }
            }
        };
        return Tensor::new(vec![result], vec![1, 1])
            .map_err(|e| mean_internal_error(format!("mean: {e}")));
    }

    let Some(output_shape) = reduction_shape(&tensor.shape, dim) else {
        return Ok(tensor.clone());
    };

    let values = tensor::tensor_values_f64(tensor);
    if values.is_empty() {
        let fill = vec![f64::NAN; tensor::element_count(&output_shape)];
        return Tensor::new(fill, output_shape)
            .map_err(|e| mean_internal_error(format!("mean: {e}")));
    }

    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let out_len = tensor::element_count(&output_shape);
    let mut output = vec![0.0f64; out_len];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut sum = 0.0;
            let mut count = 0usize;
            let mut saw_nan = false;

            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let value = values[idx];
                match nan_mode {
                    ReductionNaN::Include => {
                        if value.is_nan() {
                            saw_nan = true;
                            break;
                        }
                        sum += value;
                    }
                    ReductionNaN::Omit => {
                        if value.is_nan() {
                            continue;
                        }
                        sum += value;
                        count += 1;
                    }
                }
            }

            let out_idx = after * stride_before + before;
            output[out_idx] = match nan_mode {
                ReductionNaN::Include => {
                    if reduce_len == 0 || saw_nan {
                        f64::NAN
                    } else {
                        sum / (reduce_len as f64)
                    }
                }
                ReductionNaN::Omit => {
                    if count == 0 {
                        f64::NAN
                    } else {
                        sum / (count as f64)
                    }
                }
            };
        }
    }

    Tensor::new(output, output_shape).map_err(|e| mean_internal_error(format!("mean: {e}")))
}

fn mean_complex_tensor(
    tensor: ComplexTensor,
    axes: MeanAxes,
    nan_mode: ReductionNaN,
) -> BuiltinResult<ComplexTensor> {
    match axes {
        MeanAxes::Default => {
            let dim = default_dimension_from_shape(&tensor.shape);
            reduce_complex_tensor_mean_dim(&tensor, dim, nan_mode)
        }
        MeanAxes::Dim(dim) => reduce_complex_tensor_mean_dim(&tensor, dim, nan_mode),
        MeanAxes::Vec(mut dims) => {
            dims.sort_unstable();
            dims.dedup();
            let mut current = tensor;
            for dim in dims {
                current = reduce_complex_tensor_mean_dim(&current, dim, nan_mode)?;
            }
            Ok(current)
        }
        MeanAxes::All => {
            if is_scalar_shape(&tensor.shape) {
                Ok(tensor)
            } else {
                let mut current = tensor;
                let ndims = current.shape.len();
                for dim in 1..=ndims {
                    current = reduce_complex_tensor_mean_dim(&current, dim, nan_mode)?;
                }
                Ok(current)
            }
        }
    }
}

fn reduce_complex_tensor_mean_dim(
    tensor: &ComplexTensor,
    dim: usize,
    nan_mode: ReductionNaN,
) -> BuiltinResult<ComplexTensor> {
    if dim == 0 {
        return Err(mean_internal_error("mean: dimension must be >= 1"));
    }

    let shape = if is_scalar_shape(&tensor.shape) {
        normalize_scalar_shape(&tensor.shape)
    } else {
        tensor.shape.clone()
    };

    if is_scalar_shape(&shape) {
        let (re, im) = tensor
            .materialize_f64()
            .first()
            .copied()
            .unwrap_or((f64::NAN, f64::NAN));
        let result = match nan_mode {
            ReductionNaN::Include => (re, im),
            ReductionNaN::Omit => {
                if re.is_nan() || im.is_nan() {
                    (f64::NAN, f64::NAN)
                } else {
                    (re, im)
                }
            }
        };
        return ComplexTensor::new(vec![result], canonical_scalar_shape())
            .map_err(|e| mean_internal_error(format!("mean: {e}")));
    }

    let Some(output_shape) = reduction_shape(&shape, dim) else {
        return Ok(tensor.clone());
    };

    if tensor.materialize_f64().is_empty() {
        let fill = vec![(f64::NAN, f64::NAN); tensor::element_count(&output_shape)];
        return ComplexTensor::new(fill, output_shape)
            .map_err(|e| mean_internal_error(format!("mean: {e}")));
    }

    let dim_index = dim - 1;
    if dim_index >= shape.len() {
        return Ok(tensor.clone());
    }

    let reduce_len = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim..]);
    let out_len = tensor::element_count(&output_shape);
    let mut output = vec![(0.0f64, 0.0f64); out_len];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            let mut count = 0usize;
            let mut saw_nan = false;

            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let (re, im) = tensor.materialize_f64()[idx];
                let is_nan = re.is_nan() || im.is_nan();
                match nan_mode {
                    ReductionNaN::Include => {
                        if is_nan {
                            saw_nan = true;
                            break;
                        }
                        sum_re += re;
                        sum_im += im;
                    }
                    ReductionNaN::Omit => {
                        if is_nan {
                            continue;
                        }
                        sum_re += re;
                        sum_im += im;
                        count += 1;
                    }
                }
            }

            let out_idx = after * stride_before + before;
            output[out_idx] = match nan_mode {
                ReductionNaN::Include => {
                    if reduce_len == 0 || saw_nan {
                        (f64::NAN, f64::NAN)
                    } else {
                        (sum_re / (reduce_len as f64), sum_im / (reduce_len as f64))
                    }
                }
                ReductionNaN::Omit => {
                    if count == 0 {
                        (f64::NAN, f64::NAN)
                    } else {
                        (sum_re / (count as f64), sum_im / (count as f64))
                    }
                }
            };
        }
    }

    ComplexTensor::new(output, output_shape).map_err(|e| mean_internal_error(format!("mean: {e}")))
}

fn reduction_shape(shape: &[usize], dim: usize) -> Option<Vec<usize>> {
    if dim == 0 {
        return None;
    }
    if is_scalar_shape(shape) {
        if dim == 1 {
            return Some(canonical_scalar_shape());
        }
        return None;
    }
    if dim > shape.len() {
        return None;
    }
    let mut out = shape.to_vec();
    out[dim - 1] = 1;
    Some(out)
}

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, v| acc.saturating_mul(v))
}

fn default_dimension(tensor: &Tensor) -> usize {
    default_dimension_from_shape(&tensor.shape)
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

async fn apply_output_template(
    value: Value,
    template: &OutputTemplate,
    meta: &InputMeta,
) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => {
            let value = if meta.numeric_dtype == Some(NumericDType::F32) {
                coerce_value_to_dtype(value, NumericDType::F32).await?
            } else {
                value
            };
            ensure_device(value, meta.device).await
        }
        OutputTemplate::Double => {
            let value = coerce_value_to_dtype(value, NumericDType::F64).await?;
            ensure_device(value, meta.device).await
        }
        OutputTemplate::Native => {
            let value = apply_native_template(value, meta).await?;
            ensure_device(value, meta.device).await
        }
    }
}

async fn apply_native_template(value: Value, meta: &InputMeta) -> BuiltinResult<Value> {
    match meta.class {
        InputClass::Integer(class) => coerce_value_to_dtype(value, class.numeric_dtype()).await,
        _ => {
            if let Some(dtype) = meta.numeric_dtype {
                coerce_value_to_dtype(value, dtype).await
            } else {
                Ok(value)
            }
        }
    }
}

async fn coerce_value_to_dtype(value: Value, dtype: NumericDType) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            if numeric_dtype_from_value(&Value::GpuTensor(handle.clone())) == Some(dtype) {
                return Ok(Value::GpuTensor(handle));
            }
            let gathered = dispatcher::gather_if_needed_async(&Value::GpuTensor(handle))
                .await
                .map_err(|error| mean_internal_error(format!("mean: {error}")))?;
            coerce_host_value_to_dtype(gathered, dtype)
        }
        other => coerce_host_value_to_dtype(other, dtype),
    }
}

fn coerce_host_value_to_dtype(value: Value, dtype: NumericDType) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            let tensor = tensor::coerce_tensor_dtype(tensor, dtype);
            Ok(tensor::tensor_into_value(tensor))
        }
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], vec![1, 1])
                .map_err(|error| mean_internal_error(format!("mean: {error}")))?;
            let tensor = tensor::coerce_tensor_dtype(tensor, dtype);
            Ok(tensor::tensor_into_value(tensor))
        }
        Value::Int(value) => {
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|error| mean_internal_error(format!("mean: {error}")))?;
            let tensor = tensor::coerce_tensor_dtype(tensor, dtype);
            Ok(tensor::tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|error| mean_internal_error(format!("mean: {error}")))?;
            let tensor = tensor::coerce_tensor_dtype(tensor, dtype);
            Ok(tensor::tensor_into_value(tensor))
        }
        Value::Bool(value) => {
            coerce_host_value_to_dtype(Value::Num(if value { 1.0 } else { 0.0 }), dtype)
        }
        Value::ComplexTensor(tensor) => {
            coerce_complex_tensor_dtype(tensor, dtype).map(complex_tensor_into_value)
        }
        Value::Complex(real, imag) => match dtype {
            NumericDType::F64 => Ok(Value::Complex(real, imag)),
            NumericDType::F32 => {
                ComplexTensor::from_f32(vec![(real as f32, imag as f32)], canonical_scalar_shape())
                    .map(complex_tensor_into_value)
                    .map_err(mean_internal_error)
            }
            _ => Err(mean_invalid_argument(
                "mean: complex output must be single or double",
            )),
        },
        other => Ok(other),
    }
}

async fn ensure_device(value: Value, device: DevicePreference) -> BuiltinResult<Value> {
    match device {
        DevicePreference::Host => match value {
            Value::GpuTensor(handle) => {
                dispatcher::gather_if_needed_async(&Value::GpuTensor(handle))
                    .await
                    .map_err(|error| mean_internal_error(format!("mean: {error}")))
            }
            _ => Ok(value),
        },
        DevicePreference::Gpu => match value {
            Value::GpuTensor(_) => Ok(value),
            Value::Tensor(t) => upload_tensor(t),
            Value::Num(n) => {
                let tensor = Tensor::new(vec![n], vec![1, 1])
                    .map_err(|e| mean_internal_error(format!("mean: {e}")))?;
                upload_tensor(tensor)
            }
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical).map_err(mean_invalid_input)?;
                upload_tensor(tensor)
            }
            Value::ComplexTensor(tensor) => upload_complex_tensor(tensor),
            Value::Complex(real, imag) => {
                let tensor = ComplexTensor::new(vec![(real, imag)], canonical_scalar_shape())
                    .map_err(mean_internal_error)?;
                upload_complex_tensor(tensor)
            }
            other => Err(mean_invalid_input(format!(
                "mean: cannot place value {other:?} on the GPU"
            ))),
        },
    }
}

fn upload_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(mean_internal_error(
            "mean: no acceleration provider available to honour GPU output",
        ));
    };
    let handle = gpu_helpers::upload_tensor(provider, &tensor)
        .map_err(|e| mean_internal_error(format!("mean: failed to upload GPU result: {e}")))?;
    Ok(Value::GpuTensor(handle))
}

fn upload_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(mean_internal_error(
            "mean: no acceleration provider available to honour complex GPU output",
        ));
    };
    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|error| {
        mean_internal_error(format!(
            "mean: failed to upload complex GPU result: {error}"
        ))
    })?;
    Ok(Value::GpuTensor(handle))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage};

    fn mean_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::mean_builtin(value, rest))
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn mean_type_reduces_first_dim() {
        let out = mean_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn mean_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = MEAN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"M = mean(A)"));
        assert!(labels.contains(&"M = mean(A, dim)"));
        assert!(labels.contains(&"M = mean(A, \"all\")"));
        assert!(labels.contains(&"M = mean(A, missingflag)"));
        assert!(labels.contains(&"M = mean(A, outtype)"));
        assert!(labels.contains(&"M = mean(A, dim, missingflag)"));
        assert!(labels.contains(&"M = mean(A, missingflag, dim)"));
        assert!(!labels.iter().any(|label| label.contains("like")));
        assert!(labels.contains(&"M = mean(A, vecdim)"));
        assert!(labels.contains(&"M = mean(___, Weights=W)"));
        assert_eq!(MEAN_INPUTS_A_OUTTYPE[1].default, Some("\"default\""));
        assert_eq!(
            MEAN_INPUTS_A_MISSINGFLAG[1].default,
            Some("\"includemissing\"")
        );
    }

    #[test]
    fn mean_descriptor_errors_have_stable_codes() {
        assert!(MEAN_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == MEAN_ERROR_INVALID_ARGUMENT.code));
        assert!(MEAN_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == MEAN_ERROR_INVALID_INPUT.code));
        assert!(MEAN_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == MEAN_ERROR_INTERNAL.code));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_scalar_num() {
        let result = mean_builtin(Value::Num(6.0), Vec::new()).expect("mean");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = mean_builtin(Value::Tensor(tensor), Vec::new()).expect("mean");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(values(&out), vec![2.5, 3.5, 4.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn mean_weights_vector_and_full_size_follow_operating_dimension() {
        let input = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let vector_weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
        let vector = mean_builtin(
            Value::Tensor(input.clone()),
            vec![
                Value::Int(IntValue::I32(1)),
                Value::from("Weights"),
                Value::Tensor(vector_weights),
            ],
        )
        .expect("weighted vector mean");
        let Value::Tensor(vector) = vector else {
            panic!("expected tensor weighted vector result");
        };
        assert_eq!(vector.shape, vec![1, 2]);
        assert_eq!(values(&vector), vec![2.5, 3.5]);

        let full_weights = Tensor::new(vec![1.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let full = mean_builtin(
            Value::Tensor(input),
            vec![
                Value::Int(IntValue::I32(1)),
                Value::from("Weights"),
                Value::Tensor(full_weights),
            ],
        )
        .expect("weighted full-size mean");
        let Value::Tensor(full) = full else {
            panic!("expected tensor weighted full-size result");
        };
        assert_eq!(full.shape, vec![1, 2]);
        assert!((values(&full)[0] - 2.5).abs() < 1e-12);
        assert!((values(&full)[1] - (8.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn mean_weights_support_generic_missing_flags() {
        let input = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, 100.0, 3.0], vec![3, 1]).unwrap();
        let included = mean_builtin(
            Value::Tensor(input.clone()),
            vec![
                Value::from("Weights"),
                Value::Tensor(weights.clone()),
                Value::from("includemissing"),
            ],
        )
        .expect("included weighted mean");
        assert!(matches!(included, Value::Num(value) if value.is_nan()));

        let omitted = mean_builtin(
            Value::Tensor(input),
            vec![
                Value::from("omitmissing"),
                Value::from("Weights"),
                Value::Tensor(weights),
            ],
        )
        .expect("omitted weighted mean");
        assert!(matches!(omitted, Value::Num(value) if (value - 2.5).abs() < 1e-12));
    }

    #[test]
    fn mean_weights_preserve_single_default_and_honor_explicit_double() {
        let input = Tensor::from_f32(vec![1.0, 3.0], vec![2, 1]).unwrap();
        let weights = Tensor::from_f32(vec![1.0, 3.0], vec![2, 1]).unwrap();
        let default = mean_builtin(
            Value::Tensor(input.clone()),
            vec![Value::from("Weights"), Value::Tensor(weights.clone())],
        )
        .expect("single weighted mean");
        let Value::Tensor(default) = default else {
            panic!("expected typed single scalar tensor");
        };
        assert_eq!(default.numeric_dtype(), NumericDType::F32);
        assert_eq!(values(&default), vec![2.5]);

        let double = mean_builtin(
            Value::Tensor(input),
            vec![
                Value::from("Weights"),
                Value::Tensor(weights),
                Value::from("double"),
            ],
        )
        .expect("double weighted mean");
        assert!(matches!(double, Value::Num(value) if (value - 2.5).abs() < 1e-12));
    }

    #[test]
    fn mean_weights_default_integer_output_is_double() {
        let input = Tensor::new_integer(IntegerStorage::I16(vec![1, 3]), vec![2, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
        let result = mean_builtin(
            Value::Tensor(input),
            vec![Value::from("Weights"), Value::Tensor(weights)],
        )
        .expect("weighted integer mean");
        assert!(matches!(result, Value::Num(value) if (value - 2.5).abs() < 1e-12));
    }

    #[test]
    fn mean_weights_logical_default_and_native_are_double() {
        let input =
            runmat_builtins::LogicalArray::new(vec![0, 1], vec![2, 1]).expect("logical input");
        let weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
        for suffix in [Vec::new(), vec![Value::from("native")]] {
            let mut args = vec![Value::from("Weights"), Value::Tensor(weights.clone())];
            args.extend(suffix);
            let result = mean_builtin(Value::LogicalArray(input.clone()), args)
                .expect("weighted logical mean");
            assert!(matches!(result, Value::Num(value) if (value - 0.75).abs() < 1e-12));
        }
    }

    #[test]
    fn mean_weights_native_preserves_every_integer_class() {
        let cases = [
            (IntegerStorage::I8(vec![1, 3]), IntegerStorage::I8(vec![3])),
            (
                IntegerStorage::I16(vec![1, 3]),
                IntegerStorage::I16(vec![3]),
            ),
            (
                IntegerStorage::I32(vec![1, 3]),
                IntegerStorage::I32(vec![3]),
            ),
            (
                IntegerStorage::I64(vec![1, 3]),
                IntegerStorage::I64(vec![3]),
            ),
            (IntegerStorage::U8(vec![1, 3]), IntegerStorage::U8(vec![3])),
            (
                IntegerStorage::U16(vec![1, 3]),
                IntegerStorage::U16(vec![3]),
            ),
            (
                IntegerStorage::U32(vec![1, 3]),
                IntegerStorage::U32(vec![3]),
            ),
            (
                IntegerStorage::U64(vec![1, 3]),
                IntegerStorage::U64(vec![3]),
            ),
        ];
        for (input, expected) in cases {
            let input = Tensor::new_integer(input, vec![2, 1]).unwrap();
            let weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let result = mean_builtin(
                Value::Tensor(input),
                vec![
                    Value::from("Weights"),
                    Value::Tensor(weights),
                    Value::from("native"),
                ],
            )
            .expect("native weighted integer mean");
            assert_eq!(
                result,
                Value::Int(expected.value_at(0).expect("expected scalar storage"))
            );
        }
    }

    #[test]
    fn mean_weights_native_uint64_is_exact_above_flintmax() {
        let wide = 1_u64 << 63;
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![wide + 1, wide + 3]), vec![2, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
        let result = mean_builtin(
            Value::Tensor(input),
            vec![
                Value::from("Weights"),
                Value::Tensor(weights),
                Value::from("native"),
            ],
        )
        .expect("wide native weighted mean");
        assert_eq!(result, Value::Int(IntValue::U64(wide + 3)));
    }

    #[test]
    fn mean_weights_complex_single_preserves_precision() {
        let input = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, 6.0)], vec![2, 1]).unwrap();
        let weights = Tensor::from_f32(vec![1.0, 3.0], vec![2, 1]).unwrap();
        let result = mean_builtin(
            Value::ComplexTensor(input),
            vec![Value::from("Weights"), Value::Tensor(weights)],
        )
        .expect("complex single weighted mean");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(result.materialize_f64(), vec![(2.5, 5.0)]);
    }

    #[test]
    fn mean_weights_reject_invalid_grammar_values_and_shapes() {
        let input = Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap());
        let valid = Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap());
        let invalid_cases = [
            vec![Value::from("Weights")],
            vec![
                Value::from("Weights"),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 3]), vec![2, 1]).unwrap(),
                ),
            ],
            vec![
                Value::from("Weights"),
                Value::Tensor(Tensor::new(vec![1.0, -1.0], vec![2, 1]).unwrap()),
            ],
            vec![
                Value::from("Weights"),
                Value::Tensor(Tensor::new(vec![1.0, f64::NAN], vec![2, 1]).unwrap()),
            ],
            vec![
                Value::from("Weights"),
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            ],
            vec![Value::from("all"), Value::from("Weights"), valid.clone()],
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                Value::from("Weights"),
                valid.clone(),
            ],
            vec![
                Value::from("Weights"),
                valid.clone(),
                Value::from("Weights"),
                valid,
            ],
        ];
        for args in invalid_cases {
            let error = mean_builtin(input.clone(), args).expect_err("invalid weighted mean");
            assert_eq!(error.identifier(), MEAN_ERROR_INVALID_ARGUMENT.identifier);
        }
    }

    #[test]
    fn mean_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![0, 2, 4, 6]), vec![2, 2]).expect("tensor");

        let result = mean_builtin(Value::Tensor(tensor), Vec::new()).expect("mean");

        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(values(&out), vec![1.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn mean_all_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![0, 2, 4, 6]), vec![2, 2]).expect("tensor");

        let result = mean_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("mean");

        assert_eq!(result, Value::Num(3.0));
    }

    #[test]
    fn mean_vecdim_reads_typed_integer_dimensions_and_values_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 3, 5, 7]), vec![2, 2]).expect("tensor");
        let dims = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).expect("dims");

        let result = mean_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("mean");

        assert_eq!(result, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("mean");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(values(&out), vec![2.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_with_omit_nan_default_dimension() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("mean");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_with_omit_nan_all_nan_returns_nan() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("mean");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_with_include_nan_propagates_nan() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::from("includenan")]).expect("mean");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = tensor.clone();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(5))]).expect("mean");
        match result {
            Value::Tensor(out) => assert_eq!(out, original),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_integer_scalar() {
        let value = Value::Int(IntValue::I16(42));
        let result = mean_builtin(value, vec![Value::from("native")]).expect("mean");
        assert_eq!(result, Value::Int(IntValue::I16(42)));
    }

    #[test]
    fn mean_native_template_reads_typed_integer_scalar_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![42]), vec![1, 1]).expect("tensor");
        let meta = InputMeta {
            class: InputClass::Integer(IntClass::U16),
            device: DevicePreference::Host,
            numeric_dtype: None,
        };

        let result =
            block_on(apply_native_template(Value::Tensor(tensor), &meta)).expect("native template");

        assert_eq!(result, Value::Int(IntValue::U16(42)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_uint64_uses_exact_storage_and_rounds_halves_up() {
        let input = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX, u64::MAX, 0, 0]),
            vec![2, 2],
        )
        .expect("integer tensor");
        let result = mean_builtin(
            Value::Tensor(input),
            vec![Value::from("all"), Value::from("native")],
        )
        .expect("native mean");
        assert_eq!(result, Value::Int(IntValue::U64(1_u64 << 63)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_integer_default_dimension_preserves_typed_tensor() {
        let input = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U16(vec![1, 2, 4, 5]),
            vec![2, 2],
        )
        .expect("integer tensor");
        let result =
            mean_builtin(Value::Tensor(input), vec![Value::from("native")]).expect("native mean");
        let Value::Tensor(output) = result else {
            panic!("expected typed tensor result");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U16(vec![2, 5]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_integer_empty_reduction_covers_every_class_and_shape() {
        for storage in [
            IntegerStorage::I8(Vec::new()),
            IntegerStorage::I16(Vec::new()),
            IntegerStorage::I32(Vec::new()),
            IntegerStorage::I64(Vec::new()),
            IntegerStorage::U8(Vec::new()),
            IntegerStorage::U16(Vec::new()),
            IntegerStorage::U32(Vec::new()),
            IntegerStorage::U64(Vec::new()),
        ] {
            let default = mean_builtin(
                Value::Tensor(
                    Tensor::new_integer(storage.clone(), vec![0, 3]).expect("integer tensor"),
                ),
                vec![Value::from("native")],
            )
            .expect("default native mean");
            assert_eq!(
                default,
                Value::Tensor(
                    Tensor::new_integer(storage.zeros_like(3), vec![1, 3])
                        .expect("default expected output")
                )
            );

            let second_dimension = mean_builtin(
                Value::Tensor(
                    Tensor::new_integer(storage.clone(), vec![0, 3]).expect("integer tensor"),
                ),
                vec![Value::Num(2.0), Value::from("native")],
            )
            .expect("dimension-two native mean");
            assert_eq!(
                second_dimension,
                Value::Tensor(
                    Tensor::new_integer(storage.clone(), vec![0, 1])
                        .expect("dimension-two expected output")
                )
            );

            let all = mean_builtin(
                Value::Tensor(
                    Tensor::new_integer(storage.clone(), vec![0, 3]).expect("integer tensor"),
                ),
                vec![Value::from("all"), Value::from("native")],
            )
            .expect("all native mean");
            assert_eq!(
                all,
                Value::Int(storage.cast_exact_assignment(&IntValue::I8(0)))
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_int64_rounds_negative_halves_away_from_zero() {
        let input = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I64(vec![-2, -1, 0, 1]),
            vec![2, 2],
        )
        .expect("integer tensor");
        let result = mean_builtin(
            Value::Tensor(input),
            vec![Value::from("all"), Value::from("native")],
        )
        .expect("native mean");
        assert_eq!(result, Value::Int(IntValue::I64(-1)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_vecdim_reduces_once_before_rounding() {
        let input = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I8(vec![0, 0, 0, 5]),
            vec![2, 2],
        )
        .expect("integer tensor");
        let dimensions = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("dimensions");
        let result = mean_builtin(
            Value::Tensor(input),
            vec![Value::Tensor(dimensions), Value::from("native")],
        )
        .expect("native mean");
        assert_eq!(result, Value::Int(IntValue::I8(1)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_rejects_undocumented_like_option() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = mean_builtin(
            Value::Tensor(tensor),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect_err("expected error");
        assert_eq!(err.identifier(), MEAN_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_invalid_dim_identifier_is_descriptor_backed() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = mean_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
            .expect_err("mean");
        assert_eq!(err.identifier(), MEAN_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_dimension_with_omit_nan() {
        let tensor =
            Tensor::new(vec![1.0, f64::NAN, 3.0, 4.0], vec![2, 2]).expect("tensor construction");
        let result = mean_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::from("omitnan")],
        )
        .expect("mean");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(values(&out), vec![1.0, 3.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_all_dimension_reduces_to_scalar() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = mean_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("mean");
        match result {
            Value::Num(v) => assert!((v - 2.5).abs() < 1e-12),
            Value::Tensor(t) => {
                assert_eq!(values(&t), vec![2.5]);
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_all_keyword_first_arg_swapped_ok() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let a = mean_builtin(Value::Tensor(tensor.clone()), vec![Value::from("all")]).unwrap();
        // Provide 'all' as the first argument (char/string), then the tensor
        let b = mean_builtin(Value::from("all"), vec![Value::Tensor(tensor)]).unwrap();
        match (a, b) {
            (Value::Num(x), Value::Num(y)) => assert!((x - y).abs() < 1e-12),
            (Value::Tensor(tx), Value::Tensor(ty)) => {
                assert_eq!(tx.shape, ty.shape);
                for (x, y) in values(&tx).iter().zip(values(&ty).iter()) {
                    assert!((x - y).abs() < 1e-12);
                }
            }
            (ax, bx) => panic!("shape mismatch a={ax:?} b={bx:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_all_with_omit_nan() {
        let tensor = Tensor::new(vec![f64::NAN, 2.0, 4.0, f64::NAN], vec![2, 2]).expect("tensor");
        let result = mean_builtin(
            Value::Tensor(tensor),
            vec![Value::from("all"), Value::from("omitnan")],
        )
        .expect("mean");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_all_matches_sequential_for_nd_tensor() {
        let data: Vec<f64> = (1..=24).map(|v| v as f64).collect();
        let tensor = Tensor::new(data, vec![2, 3, 4]).expect("tensor");
        let fused =
            mean_builtin(Value::Tensor(tensor.clone()), vec![Value::from("all")]).expect("mean");
        let sequential = mean_builtin(
            mean_builtin(Value::Tensor(tensor.clone()), vec![Value::Num(1.0)]).expect("mean"),
            vec![Value::Num(2.0)],
        )
        .and_then(|v| mean_builtin(v, vec![Value::Num(3.0)]))
        .expect("mean");
        assert_eq!(fused, sequential);
        if let Value::Num(v) = fused {
            assert!((v - 12.5).abs() < 1e-12);
        } else if let Value::Tensor(t) = fused {
            assert_eq!(values(&t), vec![12.5]);
        } else {
            panic!("unexpected result {fused:?}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_vector_dimensions_match_sequential() {
        let tensor =
            Tensor::new(vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let combined = mean_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(dims.clone())],
        )
        .expect("mean");
        let first = mean_builtin(Value::Tensor(tensor), vec![Value::Num(1.0)]).expect("mean");
        let sequential = mean_builtin(first, vec![Value::Num(3.0)]).expect("mean");
        assert_eq!(combined, sequential);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_complex_scalar_passthrough() {
        let result = mean_builtin(Value::Complex(2.0, -3.0), Vec::new()).expect("mean");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!((im + 3.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_complex_matrix_along_rows() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (5.0, -1.0), (2.0, 2.0), (6.0, -2.0)],
            vec![2, 2],
        )
        .unwrap();
        let result =
            mean_builtin(Value::ComplexTensor(tensor), vec![Value::Num(1.0)]).expect("mean");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                let expected = [(3.0, 0.0), (4.0, 0.0)];
                for (got, exp) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < 1e-12);
                    assert!((got.1 - exp.1).abs() < 1e-12);
                }
            }
            Value::Complex(re, im) => {
                panic!("expected tensor result, got scalar {re}+{im}i");
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_complex_omit_nan_returns_nan() {
        let tensor =
            ComplexTensor::new(vec![(f64::NAN, 0.0), (1.0, f64::NAN)], vec![2, 1]).unwrap();
        let result = mean_builtin(
            Value::ComplexTensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::from("omitnan")],
        )
        .expect("mean");
        match result {
            Value::Complex(re, im) => {
                assert!(re.is_nan());
                assert!(im.is_nan());
            }
            Value::ComplexTensor(out) => {
                let (re, im) = out.materialize_f64()[0];
                assert!(re.is_nan());
                assert!(im.is_nan());
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = mean_builtin(Value::GpuTensor(handle), Vec::new()).expect("mean");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(values(&gathered), vec![2.5, 3.5, 4.5]);
        });
    }

    #[test]
    fn mean_weights_gpu_input_preserves_residency() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let input = gpu_helpers::upload_tensor(provider, &input).expect("upload input");
            let weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let result = mean_builtin(
                Value::GpuTensor(input),
                vec![Value::from("Weights"), Value::Tensor(weights)],
            )
            .expect("weighted GPU mean");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident weighted mean");
            };
            let gathered =
                test_support::gather(Value::GpuTensor(handle)).expect("gather weighted mean");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(values(&gathered), vec![2.5, 3.5]);
        });
    }

    #[test]
    fn mean_weights_logical_gpu_result_is_numeric_and_resident() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let input = gpu_helpers::upload_tensor(provider, &input).expect("upload logical input");
            let input = gpu_helpers::logical_gpu_value(input);
            let weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let result = mean_builtin(input, vec![Value::from("Weights"), Value::Tensor(weights)])
                .expect("weighted logical GPU mean");
            let Value::GpuTensor(result) = result else {
                panic!("expected resident numeric result");
            };
            assert!(!runmat_accelerate_api::handle_is_logical(&result));
            let gathered =
                test_support::gather(Value::GpuTensor(result)).expect("gather logical mean");
            assert_eq!(values(&gathered), vec![0.75]);
        });
    }

    #[test]
    fn mean_weights_complex_gpu_input_preserves_class_and_residency() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 6.0)], vec![2, 1]).unwrap();
            let input = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload input");
            let weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let result = mean_builtin(
                Value::GpuTensor(input),
                vec![Value::from("Weights"), Value::Tensor(weights)],
            )
            .expect("weighted complex GPU mean");
            let Value::GpuTensor(result) = result else {
                panic!("expected resident weighted complex mean");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&result),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(dispatcher::gather_if_needed_async(&Value::GpuTensor(
                result,
            )))
            .expect("gather weighted complex mean");
            let Value::ComplexTensor(gathered) = gathered else {
                panic!("expected gathered complex tensor");
            };
            assert_eq!(gathered.materialize_f64(), vec![(2.5, 5.0)]);
        });
    }

    #[test]
    fn mean_weights_accept_resident_gpu_weights() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let weights = gpu_helpers::upload_tensor(provider, &weights).expect("upload weights");
            let result = mean_builtin(
                Value::Tensor(input),
                vec![Value::from("Weights"), Value::GpuTensor(weights)],
            )
            .expect("mean with resident weights");
            assert!(matches!(result, Value::Num(value) if (value - 2.5).abs() < 1e-12));
        });
    }

    #[test]
    fn mean_weights_native_integer_gpu_is_exact_and_resident() {
        test_support::with_test_provider(|provider| {
            let wide = 1_u64 << 63;
            let input = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[wide + 1, wide + 3]),
                    shape: &[2, 1],
                })
                .expect("upload native integer");
            let weights = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let result = mean_builtin(
                Value::GpuTensor(input),
                vec![
                    Value::from("Weights"),
                    Value::Tensor(weights),
                    Value::from("native"),
                ],
            )
            .expect("native weighted GPU mean");
            let Value::GpuTensor(result) = result else {
                panic!("expected resident native weighted mean");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&result),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(
                block_on(provider.download_integer(&result))
                    .expect("download native weighted mean")
                    .data,
                runmat_accelerate_api::HostIntegerDataOwned::U64(vec![wide + 3])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_gpu_omit_nan_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![2, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result =
                mean_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("mean");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(values(&gathered), vec![2.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_gpu_all_dimension_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result =
                mean_builtin(Value::GpuTensor(handle), vec![Value::from("all")]).expect("mean");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(values(&gathered), vec![2.5]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_gpu_vector_dimensions_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], vec![2, 2, 2]).unwrap();
            let cpu_dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
            let cpu_result = mean_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::Tensor(cpu_dims.clone())],
            )
            .expect("mean cpu");

            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let gpu_result = mean_builtin(Value::GpuTensor(handle), vec![Value::Tensor(cpu_dims)])
                .expect("mean gpu");

            let cpu_tensor = match cpu_result {
                Value::Tensor(t) => t,
                Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
                other => panic!("unexpected cpu result {other:?}"),
            };
            let gpu_tensor = test_support::gather(gpu_result).expect("gather");
            assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
            for (a, b) in values(&gpu_tensor).iter().zip(values(&cpu_tensor).iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_integer_gpu_preserves_exact_class_and_residency() {
        test_support::with_test_provider(|provider| {
            let large = 1_u64 << 63;
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[
                        large + 1,
                        large + 3,
                        7,
                        9,
                    ]),
                    shape: &[2, 2],
                })
                .expect("upload native integer");
            let result = mean_builtin(
                Value::GpuTensor(handle),
                vec![Value::Int(IntValue::I32(1)), Value::from("native")],
            )
            .expect("native mean");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident GPU tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&out),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(
                block_on(provider.download_integer(&out))
                    .expect("download native integer mean")
                    .data,
                runmat_accelerate_api::HostIntegerDataOwned::U64(vec![large + 2, 8])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_empty_integer_gpu_returns_resident_typed_zeros() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U16(&[]),
                    shape: &[0, 3],
                })
                .expect("upload empty native integer");
            let result = mean_builtin(Value::GpuTensor(handle), vec![Value::from("native")])
                .expect("native empty mean");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident GPU tensor");
            };
            assert_eq!(out.shape, vec![1, 3]);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&out),
                Some(runmat_accelerate_api::IntegerElementType::U16)
            );
            assert_eq!(
                block_on(provider.download_integer(&out))
                    .expect("download native integer mean")
                    .data,
                runmat_accelerate_api::HostIntegerDataOwned::U16(vec![0, 0, 0])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_integer_gpu_vecdim_rounds_once() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::I16(&[1, 1, 1, 3]),
                    shape: &[2, 2],
                })
                .expect("upload native integer");
            let dims = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("dims");
            let result = mean_builtin(
                Value::GpuTensor(handle),
                vec![Value::Tensor(dims), Value::from("native")],
            )
            .expect("native vecdim mean");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident GPU tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&out),
                Some(runmat_accelerate_api::IntegerElementType::I16)
            );
            assert_eq!(out.shape, vec![1, 1]);
            assert_eq!(
                block_on(provider.download_integer(&out))
                    .expect("download native integer vecdim mean")
                    .data,
                runmat_accelerate_api::HostIntegerDataOwned::I16(vec![2])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_nested_dim2_then_dim3_host_matches_vecdim() {
        let t = Tensor::new((0..(2 * 3 * 4)).map(|i| i as f64).collect(), vec![2, 3, 4]).unwrap();
        let vecdim = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let a = mean_builtin(Value::Tensor(t.clone()), vec![Value::Tensor(vecdim)]).unwrap();
        let b1 = mean_builtin(Value::Tensor(t), vec![Value::Num(2.0)]).unwrap();
        let b2 = mean_builtin(b1, vec![Value::Num(3.0)]).unwrap();
        assert_eq!(a, b2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn mean_wgpu_dim1_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![1.0, 4.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let args = ParsedArguments {
            axes: MeanAxes::Dim(1),
            nan_mode: ReductionNaN::Include,
            output: OutputTemplate::Double,
            weights: None,
        };
        let cpu = mean_host(Value::Tensor(t.clone()), &args).unwrap();
        let provider = runmat_accelerate_api::provider().unwrap();
        let h = gpu_helpers::upload_tensor(provider, &t).unwrap();
        let gpu = block_on(mean_gpu(h, &args)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                for (a, b) in values(&gt).iter().zip(values(&ct).iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
