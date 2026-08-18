//! MATLAB-compatible `mode` builtin for RunMat.
//!
//! Computes the most frequent value along a dimension. NaNs are ignored
//! unless all values are NaN, in which case the result is NaN with
//! frequency 0 and an empty tied set. When several values share the same
//! maximum frequency, the smallest value is returned in `M` while `C`
//! contains the entire sorted tied set per slice.

use std::cmp::Ordering;
use std::collections::HashMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, IntValue, IntegerStorage, LogicalArray, NumericDType,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::builtins::stats::type_resolvers::mode_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "mode";

const MODE_OUTPUT_M: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Most frequent value along the selected dimension.",
}];

const MODE_OUTPUT_MF: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Most frequent value along the selected dimension.",
    },
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequency counts for each reported mode.",
    },
];

const MODE_OUTPUT_MFC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Most frequent value along the selected dimension.",
    },
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequency counts for each reported mode.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array containing all tied modal values per slice.",
    },
];

const MODE_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input data array.",
}];

const MODE_INPUTS_X_DIM_OR_ALL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data array.",
    },
    BuiltinParamDescriptor {
        name: "dim_or_vecdim_or_all",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reduction axis, vector of axes, or 'all'.",
    },
];

const MODE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "M = mode(X)",
        inputs: &MODE_INPUTS_X,
        outputs: &MODE_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = mode(X, dim_or_vecdim_or_all)",
        inputs: &MODE_INPUTS_X_DIM_OR_ALL,
        outputs: &MODE_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F] = mode(X)",
        inputs: &MODE_INPUTS_X,
        outputs: &MODE_OUTPUT_MF,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F] = mode(X, dim_or_vecdim_or_all)",
        inputs: &MODE_INPUTS_X_DIM_OR_ALL,
        outputs: &MODE_OUTPUT_MF,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F, C] = mode(X)",
        inputs: &MODE_INPUTS_X,
        outputs: &MODE_OUTPUT_MFC,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F, C] = mode(X, dim_or_vecdim_or_all)",
        inputs: &MODE_INPUTS_X_DIM_OR_ALL,
        outputs: &MODE_OUTPUT_MFC,
    },
];

const MODE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MODE.INVALID_ARGUMENT",
    identifier: Some("RunMat:mode:InvalidArgument"),
    when: "Arguments are malformed, duplicated, or unrecognised.",
    message: "mode: invalid argument",
};

const MODE_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MODE.INVALID_DIMENSION",
    identifier: Some("RunMat:mode:InvalidDimension"),
    when: "Dimension argument is zero or negative.",
    message: "mode: dimension must be >= 1",
};

const MODE_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MODE.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:mode:ComplexUnsupported"),
    when: "Input data is complex-valued.",
    message: "mode: complex inputs are not supported; gather real data first",
};

const MODE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MODE.INTERNAL",
    identifier: Some("RunMat:mode:Internal"),
    when: "Internal conversion/allocation/shape handling fails.",
    message: "mode: internal operation failed",
};

const MODE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    MODE_ERROR_INVALID_ARGUMENT,
    MODE_ERROR_INVALID_DIMENSION,
    MODE_ERROR_COMPLEX_UNSUPPORTED,
    MODE_ERROR_INTERNAL,
];

pub const MODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MODE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MODE_ERRORS,
};

const MODE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Numeric arrays accept every real integer class; modal values and every tied-value cell preserve the exact input class.",
    },
    BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Positive integer dimensions are parsed exactly from typed integer or integer-valued floating controls.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[M, F, C] = mode(A, dim_or_vecdim_or_all)",
        inputs: &MODE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "M and the sorted tied vectors in C preserve exact same-class integer storage, F is always double, first/smallest tie rules are exact above flintmax, and resident integer inputs gather then re-upload typed value outputs.",
    }];

fn mode_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn mode_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    mode_error_with(error, error.message)
}

fn mode_internal_error(message: impl Into<String>) -> RuntimeError {
    mode_error_with(&MODE_ERROR_INTERNAL, message)
}

fn mode_type_resolver(args: &[Type], ctx: &ResolveContext) -> Type {
    mode_type(args, ctx)
}

#[runtime_builtin(
    name = "mode",
    category = "stats/summary",
    summary = "Most frequent value along a dimension with MATLAB-compatible tie semantics.",
    keywords = "mode,frequency,statistics,reduction,ties",
    type_resolver(mode_type_resolver),
    descriptor(crate::builtins::stats::summary::mode::MODE_DESCRIPTOR),
    integer_capabilities(crate::builtins::stats::summary::mode::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::mode"
)]
async fn mode_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let parsed = parse_arguments(&rest).await?;
    let gpu_provider = match &value {
        Value::GpuTensor(handle) => runmat_accelerate_api::provider_for_handle(handle)
            .or_else(runmat_accelerate_api::provider),
        _ => None,
    };
    let value = crate::gather_if_needed_async(&value).await?;
    let output_class = OutputClass::from_value(&value);
    let eval = mode_evaluate(value, parsed, output_class)?;
    let result = if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            Value::OutputList(Vec::new())
        } else if out_count == 1 {
            Value::OutputList(vec![eval.into_values_value()?])
        } else if out_count == 2 {
            let (values, freq) = eval.into_pair()?;
            Value::OutputList(vec![values, freq])
        } else {
            let (values, freq, cells) = eval.into_triple()?;
            crate::output_count::output_list_with_padding(out_count, vec![values, freq, cells])
        }
    } else {
        eval.into_values_value()?
    };
    if let Some(provider) = gpu_provider {
        upload_mode_value(provider, result)
    } else {
        Ok(result)
    }
}

fn upload_mode_value(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    value: Value,
) -> BuiltinResult<Value> {
    let upload_tensor = |tensor: Tensor, logical: bool| -> BuiltinResult<Value> {
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .map_err(|error| mode_internal_error(format!("mode: GPU upload failed: {error}")))?;
        if logical {
            runmat_accelerate_api::set_handle_logical(&handle, true);
        }
        Ok(Value::GpuTensor(handle))
    };
    match value {
        gpu @ Value::GpuTensor(_) => Ok(gpu),
        Value::Tensor(tensor) => upload_tensor(tensor, false),
        Value::Num(number) => upload_tensor(
            Tensor::new(vec![number], vec![1, 1])
                .map_err(|error| mode_internal_error(format!("mode: {error}")))?,
            false,
        ),
        Value::Int(integer) => upload_tensor(
            Tensor::new_integer(
                crate::builtins::math::reduction::integer_native::storage_from_scalar(&integer),
                vec![1, 1],
            )
            .map_err(|error| mode_internal_error(format!("mode: {error}")))?,
            false,
        ),
        Value::Bool(logical) => upload_tensor(
            Tensor::new(vec![if logical { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|error| mode_internal_error(format!("mode: {error}")))?,
            true,
        ),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(mode_internal_error)?;
            upload_tensor(tensor, true)
        }
        Value::Cell(cell) => {
            let shape = cell.shape.clone();
            let data = cell
                .data
                .into_iter()
                .map(|entry| upload_mode_value(provider, entry))
                .collect::<BuiltinResult<Vec<_>>>()?;
            runmat_builtins::CellArray::new_with_shape(data, shape)
                .map(Value::Cell)
                .map_err(mode_internal_error)
        }
        Value::OutputList(values) => values
            .into_iter()
            .map(|entry| upload_mode_value(provider, entry))
            .collect::<BuiltinResult<Vec<_>>>()
            .map(Value::OutputList),
        other => Ok(other),
    }
}

#[derive(Clone, Debug)]
enum ModeAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Clone, Debug)]
struct ParsedArguments {
    axes: ModeAxes,
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<ParsedArguments> {
    let mut axes = ModeAxes::Default;
    let mut axes_set = false;

    for arg in args {
        if axes_set {
            return Err(mode_error_with(
                &MODE_ERROR_INVALID_ARGUMENT,
                format!("mode: unexpected extra argument {arg:?}"),
            ));
        }
        if let Some(selection) = parse_axes(arg).await? {
            axes = selection;
            axes_set = true;
            continue;
        }
        return Err(mode_error_with(
            &MODE_ERROR_INVALID_ARGUMENT,
            format!("mode: unrecognised argument {arg:?}"),
        ));
    }

    Ok(ParsedArguments { axes })
}

async fn parse_axes(value: &Value) -> BuiltinResult<Option<ModeAxes>> {
    if let Some(text) = value_as_str(value) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(mode_error_with(
                &MODE_ERROR_INVALID_ARGUMENT,
                "mode: dimension string must not be empty",
            ));
        }
        let lowered = trimmed.to_ascii_lowercase();
        return match lowered.as_str() {
            "all" => Ok(Some(ModeAxes::All)),
            other => Err(mode_error_with(
                &MODE_ERROR_INVALID_ARGUMENT,
                format!("mode: unrecognised argument '{other}'"),
            )),
        };
    }

    let (scalar_hint, is_empty) = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => (true, false),
        Value::Tensor(tensor) => (tensor::is_scalar_tensor(tensor), tensor.is_empty()),
        Value::LogicalArray(logical) => (logical.data.len() == 1, logical.data.is_empty()),
        Value::GpuTensor(handle) => {
            let len = tensor::element_count(&handle.shape);
            (len == 1, len == 0)
        }
        _ => return Ok(None),
    };
    if is_empty {
        return Ok(Some(ModeAxes::Default));
    }
    let dims = tensor::dims_from_value_async(value)
        .await
        .map_err(|message| mode_dimension_error(message, scalar_hint))?;
    let Some(dims) = dims else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Ok(Some(ModeAxes::Default));
    }
    for &dim in &dims {
        if dim < 1 {
            return Err(mode_error_with(
                &MODE_ERROR_INVALID_DIMENSION,
                if scalar_hint {
                    "mode: dimension must be >= 1"
                } else {
                    "mode: dimension entries must be >= 1"
                },
            ));
        }
    }
    if dims.len() == 1 {
        Ok(Some(ModeAxes::Dim(dims[0])))
    } else {
        Ok(Some(ModeAxes::Vec(dims)))
    }
}

fn mode_dimension_error(message: String, scalar: bool) -> RuntimeError {
    let detail = if message.contains("finite") {
        if scalar {
            "mode: dimension must be finite"
        } else {
            "mode: dimension entries must be finite integers"
        }
    } else if message.contains("integer") {
        if scalar {
            "mode: dimension must be an integer"
        } else {
            "mode: dimension entries must be integers"
        }
    } else if message.contains("non-negative") {
        if scalar {
            "mode: dimension must be >= 1"
        } else {
            "mode: dimension entries must be >= 1"
        }
    } else {
        return mode_error_with(&MODE_ERROR_INVALID_DIMENSION, message);
    };
    mode_error_with(&MODE_ERROR_INVALID_DIMENSION, detail)
}

fn value_as_str(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

/// Evaluation artifact produced by `mode` carrying values, frequencies, and tied sets.
#[derive(Debug)]
pub struct ModeEvaluation {
    /// Most frequent value per slice; NaN when the slice was entirely NaN/empty.
    values: Tensor,
    /// Frequency of the mode per slice; 0.0 when the slice was entirely NaN/empty.
    freq: Tensor,
    /// One sorted tied-set column vector per slice, flattened in column-major order.
    ties: ModeTies,
    /// Shape of the M / F tensors (also the cell array shape for C).
    output_shape: Vec<usize>,
    /// MATLAB class to preserve for `M` and the tied values in `C`.
    output_class: OutputClass,
}

#[derive(Debug)]
enum ModeTies {
    Floating(Vec<Vec<f64>>),
    Integer(Vec<IntegerStorage>),
}

impl ModeEvaluation {
    fn empty(output_shape: Vec<usize>, output_class: OutputClass) -> BuiltinResult<Self> {
        let len = tensor::element_count(&output_shape);
        let values = Tensor::new(vec![f64::NAN; len], output_shape.clone())
            .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
        let freq = Tensor::new(vec![0.0; len], output_shape.clone())
            .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
        let ties = ModeTies::Floating(vec![Vec::new(); len]);
        Ok(Self {
            values,
            freq,
            ties,
            output_shape,
            output_class,
        })
    }

    fn into_values_value(self) -> BuiltinResult<Value> {
        tensor_into_class_value(self.values, self.output_class)
    }

    fn into_pair(self) -> BuiltinResult<(Value, Value)> {
        let ModeEvaluation {
            values,
            freq,
            output_class,
            ..
        } = self;
        Ok((
            tensor_into_class_value(values, output_class)?,
            tensor::tensor_into_value(freq),
        ))
    }

    fn into_triple(self) -> BuiltinResult<(Value, Value, Value)> {
        let ModeEvaluation {
            values,
            freq,
            ties,
            output_shape,
            output_class,
        } = self;
        let cell = ties_to_cell(ties, &output_shape, output_class)?;
        Ok((
            tensor_into_class_value(values, output_class)?,
            tensor::tensor_into_value(freq),
            cell,
        ))
    }
}

fn ties_to_cell(
    ties: ModeTies,
    output_shape: &[usize],
    output_class: OutputClass,
) -> BuiltinResult<Value> {
    let cell_shape = if output_shape.is_empty() {
        vec![1, 1]
    } else {
        output_shape.to_vec()
    };
    let cell_values = match ties {
        ModeTies::Floating(ties) => {
            let mut cell_values = Vec::with_capacity(ties.len());
            for entry in ties {
                let rows = entry.len();
                let tensor = Tensor::new(entry, vec![rows, 1]).map_err(|e| {
                    mode_internal_error(format!("mode: cell construction failed: {e}"))
                })?;
                cell_values.push(tensor_into_class_array_value(tensor, output_class)?);
            }
            cell_values
        }
        ModeTies::Integer(ties) => ties
            .into_iter()
            .map(|storage| {
                let rows = storage.len();
                Tensor::new_integer(storage, vec![rows, 1])
                    .map(Value::Tensor)
                    .map_err(|e| {
                        mode_internal_error(format!("mode: cell construction failed: {e}"))
                    })
            })
            .collect::<BuiltinResult<Vec<_>>>()?,
    };
    crate::make_cell_with_shape(cell_values, cell_shape).map_err(mode_internal_error)
}

fn mode_evaluate(
    value: Value,
    args: ParsedArguments,
    output_class: OutputClass,
) -> BuiltinResult<ModeEvaluation> {
    let tensor = materialize_tensor(value)?;
    if matches!(&args.axes, ModeAxes::Default) && tensor.shape == [0, 0] {
        return ModeEvaluation::empty(vec![1, 1], output_class);
    }
    if let Some(storage) = tensor.integer_storage().cloned() {
        return match args.axes {
            ModeAxes::Default => {
                let dim = default_dimension_from_shape(&tensor.shape);
                reduce_integer_along_dim(tensor, storage, dim)
            }
            ModeAxes::Dim(dim) => reduce_integer_along_dim(tensor, storage, dim),
            ModeAxes::Vec(dims) => reduce_integer_along_dims(tensor, storage, dims),
            ModeAxes::All => reduce_integer_all(tensor, storage),
        };
    }
    match args.axes {
        ModeAxes::Default => {
            let dim = default_dimension_from_shape(&tensor.shape);
            reduce_along_dim(tensor, dim, output_class)
        }
        ModeAxes::Dim(dim) => reduce_along_dim(tensor, dim, output_class),
        ModeAxes::Vec(dims) => reduce_along_dims(tensor, dims, output_class),
        ModeAxes::All => reduce_all(tensor, output_class),
    }
}

fn materialize_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            Err(mode_error(&MODE_ERROR_COMPLEX_UNSUPPORTED))
        }
        other => tensor::value_into_tensor_for(NAME, other)
            .map_err(|e| mode_error_with(&MODE_ERROR_INVALID_ARGUMENT, e)),
    }
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn reduce_all(tensor: Tensor, output_class: OutputClass) -> BuiltinResult<ModeEvaluation> {
    let output_shape = vec![1usize, 1];
    let values = tensor::tensor_into_values_f64(tensor);
    if values.is_empty() {
        return ModeEvaluation::empty(output_shape, output_class);
    }
    let scalar = scalar_mode(&values);
    finalize_single_slice(scalar, output_shape, output_class)
}

fn finalize_single_slice(
    scalar: ScalarMode,
    output_shape: Vec<usize>,
    output_class: OutputClass,
) -> BuiltinResult<ModeEvaluation> {
    let values = Tensor::new(vec![scalar.value], output_shape.clone())
        .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
    let freq = Tensor::new(vec![scalar.frequency], output_shape.clone())
        .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
    let ties = ModeTies::Floating(vec![scalar.ties]);
    Ok(ModeEvaluation {
        values,
        freq,
        ties,
        output_shape,
        output_class,
    })
}

fn reduce_along_dim(
    tensor: Tensor,
    dim: usize,
    output_class: OutputClass,
) -> BuiltinResult<ModeEvaluation> {
    if dim == 0 {
        return Err(mode_error(&MODE_ERROR_INVALID_DIMENSION));
    }
    let shape = tensor.shape.clone();
    let data = tensor::tensor_into_values_f64(tensor);

    if shape.is_empty() {
        let scalar_value = data.first().copied().unwrap_or(f64::NAN);
        let output_shape = vec![1usize, 1];
        if scalar_value.is_nan() {
            return ModeEvaluation::empty(output_shape, output_class);
        }
        let scalar = ScalarMode {
            value: scalar_value,
            frequency: 1.0,
            ties: vec![scalar_value],
        };
        return finalize_single_slice(scalar, output_shape, output_class);
    }

    if dim > shape.len() {
        // Reducing along a trailing singleton: every slice has one element.
        let output_shape = shape.clone();
        let len = tensor::element_count(&output_shape);
        let mut values = Vec::with_capacity(len);
        let mut freq = Vec::with_capacity(len);
        let mut ties = Vec::with_capacity(len);
        for &v in &data {
            if v.is_nan() {
                values.push(f64::NAN);
                freq.push(0.0);
                ties.push(Vec::new());
            } else {
                values.push(v);
                freq.push(1.0);
                ties.push(vec![v]);
            }
        }
        let values_tensor = Tensor::new(values, output_shape.clone())
            .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
        let freq_tensor = Tensor::new(freq, output_shape.clone())
            .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
        return Ok(ModeEvaluation {
            values: values_tensor,
            freq: freq_tensor,
            ties: ModeTies::Floating(ties),
            output_shape,
            output_class,
        });
    }

    let dim_index = dim - 1;
    let reduce_len = shape[dim_index];
    let mut output_shape = shape.clone();
    output_shape[dim_index] = 1;

    if reduce_len == 0 || data.is_empty() {
        return ModeEvaluation::empty(output_shape, output_class);
    }

    let stride_before = dim_product(&shape[..dim_index])?;
    let stride_after = dim_product(&shape[dim_index + 1..])?;
    let output_len = stride_before
        .checked_mul(stride_after)
        .ok_or_else(|| mode_internal_error("mode: output size overflow"))?;

    let mut values = vec![0.0f64; output_len];
    let mut freq = vec![0.0f64; output_len];
    let mut ties: Vec<Vec<f64>> = vec![Vec::new(); output_len];
    let mut slice = Vec::with_capacity(reduce_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            slice.clear();
            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                slice.push(data[idx]);
            }
            let scalar = scalar_mode(&slice);
            let out_idx = before + after * stride_before;
            values[out_idx] = scalar.value;
            freq[out_idx] = scalar.frequency;
            ties[out_idx] = scalar.ties;
        }
    }

    let values_tensor = Tensor::new(values, output_shape.clone())
        .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
    let freq_tensor = Tensor::new(freq, output_shape.clone())
        .map_err(|e| mode_internal_error(format!("mode: {e}")))?;

    Ok(ModeEvaluation {
        values: values_tensor,
        freq: freq_tensor,
        ties: ModeTies::Floating(ties),
        output_shape,
        output_class,
    })
}

fn reduce_along_dims(
    tensor: Tensor,
    dims: Vec<usize>,
    output_class: OutputClass,
) -> BuiltinResult<ModeEvaluation> {
    let shape = tensor.shape.clone();
    let (output_shape, reduce_mask) = vecdim_plan(&shape, dims)?;
    if !reduce_mask.iter().any(|reduced| *reduced) {
        return reduce_along_dim(tensor, shape.len() + 1, output_class);
    }
    let data = tensor::tensor_into_values_f64(tensor);
    if data.is_empty() {
        return ModeEvaluation::empty(output_shape, output_class);
    }
    let output_len = tensor::element_count(&output_shape);
    let mut slices = vec![Vec::new(); output_len];
    for (linear, value) in data.into_iter().enumerate() {
        let output_index = vecdim_output_index(linear, &shape, &reduce_mask);
        slices[output_index].push(value);
    }
    let mut values = Vec::with_capacity(output_len);
    let mut freq = Vec::with_capacity(output_len);
    let mut ties = Vec::with_capacity(output_len);
    for slice in slices {
        let scalar = scalar_mode(&slice);
        values.push(scalar.value);
        freq.push(scalar.frequency);
        ties.push(scalar.ties);
    }
    let values = Tensor::new(values, output_shape.clone())
        .map_err(|error| mode_internal_error(format!("mode: {error}")))?;
    let freq = Tensor::new(freq, output_shape.clone())
        .map_err(|error| mode_internal_error(format!("mode: {error}")))?;
    Ok(ModeEvaluation {
        values,
        freq,
        ties: ModeTies::Floating(ties),
        output_shape,
        output_class,
    })
}

fn reduce_integer_all(_tensor: Tensor, storage: IntegerStorage) -> BuiltinResult<ModeEvaluation> {
    let output_shape = vec![1usize, 1];
    if storage.is_empty() {
        return ModeEvaluation::empty(output_shape, OutputClass::Double);
    }
    let scalar = integer_scalar_mode(&storage.exact_values());
    integer_single_slice(scalar, output_shape, &storage)
}

fn reduce_integer_along_dim(
    tensor: Tensor,
    storage: IntegerStorage,
    dim: usize,
) -> BuiltinResult<ModeEvaluation> {
    if dim == 0 {
        return Err(mode_error(&MODE_ERROR_INVALID_DIMENSION));
    }
    if tensor.shape.is_empty() || dim > tensor.shape.len() {
        return integer_identity_mode(tensor, storage);
    }

    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let mut output_shape = tensor.shape.clone();
    output_shape[dim_index] = 1;
    if reduce_len == 0 || storage.is_empty() {
        return ModeEvaluation::empty(output_shape, OutputClass::Double);
    }

    let stride_before = dim_product(&tensor.shape[..dim_index])?;
    let stride_after = dim_product(&tensor.shape[dim_index + 1..])?;
    let output_len = stride_before
        .checked_mul(stride_after)
        .ok_or_else(|| mode_internal_error("mode: output size overflow"))?;
    let exact = storage.exact_values();
    let mut values = Vec::with_capacity(output_len);
    let mut freq = Vec::with_capacity(output_len);
    let mut ties = Vec::with_capacity(output_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut slice = Vec::with_capacity(reduce_len);
            for k in 0..reduce_len {
                let index = before + k * stride_before + after * stride_before * reduce_len;
                slice.push(exact[index].clone());
            }
            let scalar = integer_scalar_mode(&slice);
            values.push(scalar.value);
            freq.push(scalar.frequency);
            ties.push(
                storage
                    .from_same_class_values(scalar.ties)
                    .map_err(mode_internal_error)?,
            );
        }
    }

    let values = Tensor::new_integer(
        storage
            .from_same_class_values(values)
            .map_err(mode_internal_error)?,
        output_shape.clone(),
    )
    .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
    let freq = Tensor::new(freq, output_shape.clone())
        .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
    Ok(ModeEvaluation {
        values,
        freq,
        ties: ModeTies::Integer(ties),
        output_shape,
        output_class: OutputClass::Double,
    })
}

fn reduce_integer_along_dims(
    tensor: Tensor,
    storage: IntegerStorage,
    dims: Vec<usize>,
) -> BuiltinResult<ModeEvaluation> {
    let (output_shape, reduce_mask) = vecdim_plan(&tensor.shape, dims)?;
    if !reduce_mask.iter().any(|reduced| *reduced) {
        return integer_identity_mode(tensor, storage);
    }
    if storage.is_empty() {
        return ModeEvaluation::empty(output_shape, OutputClass::Double);
    }
    let output_len = tensor::element_count(&output_shape);
    let mut slices = vec![Vec::new(); output_len];
    for (linear, value) in storage.exact_values().into_iter().enumerate() {
        let output_index = vecdim_output_index(linear, &tensor.shape, &reduce_mask);
        slices[output_index].push(value);
    }
    let mut values = Vec::with_capacity(output_len);
    let mut freq = Vec::with_capacity(output_len);
    let mut ties = Vec::with_capacity(output_len);
    for slice in slices {
        let scalar = integer_scalar_mode(&slice);
        values.push(scalar.value);
        freq.push(scalar.frequency);
        ties.push(
            storage
                .from_same_class_values(scalar.ties)
                .map_err(mode_internal_error)?,
        );
    }
    let values = Tensor::new_integer(
        storage
            .from_same_class_values(values)
            .map_err(mode_internal_error)?,
        output_shape.clone(),
    )
    .map_err(|error| mode_internal_error(format!("mode: {error}")))?;
    let freq = Tensor::new(freq, output_shape.clone())
        .map_err(|error| mode_internal_error(format!("mode: {error}")))?;
    Ok(ModeEvaluation {
        values,
        freq,
        ties: ModeTies::Integer(ties),
        output_shape,
        output_class: OutputClass::Double,
    })
}

fn vecdim_plan(shape: &[usize], mut dims: Vec<usize>) -> BuiltinResult<(Vec<usize>, Vec<bool>)> {
    dims.sort_unstable();
    dims.dedup();
    if dims.iter().any(|dim| *dim == 0) {
        return Err(mode_error(&MODE_ERROR_INVALID_DIMENSION));
    }
    let mut output_shape = shape.to_vec();
    let mut reduce_mask = vec![false; shape.len()];
    for dim in dims {
        let index = dim - 1;
        if index < shape.len() {
            output_shape[index] = 1;
            reduce_mask[index] = true;
        }
    }
    Ok((output_shape, reduce_mask))
}

fn vecdim_output_index(linear: usize, shape: &[usize], reduce_mask: &[bool]) -> usize {
    let mut remainder = linear;
    let mut output_index = 0usize;
    let mut output_stride = 1usize;
    for (dimension, &extent) in shape.iter().enumerate() {
        let coordinate = if extent == 0 { 0 } else { remainder % extent };
        if extent != 0 {
            remainder /= extent;
        }
        if !reduce_mask[dimension] {
            output_index += coordinate * output_stride;
            output_stride *= extent;
        }
    }
    output_index
}

fn integer_identity_mode(tensor: Tensor, storage: IntegerStorage) -> BuiltinResult<ModeEvaluation> {
    let output_shape = tensor.shape.clone();
    let values = Tensor::new_integer(storage.clone(), output_shape.clone())
        .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
    let freq = Tensor::new(vec![1.0; storage.len()], output_shape.clone())
        .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
    let ties = storage
        .exact_values()
        .into_iter()
        .map(|value| storage.from_same_class_values(vec![value]))
        .collect::<Result<Vec<_>, _>>()
        .map_err(mode_internal_error)?;
    Ok(ModeEvaluation {
        values,
        freq,
        ties: ModeTies::Integer(ties),
        output_shape,
        output_class: OutputClass::Double,
    })
}

fn integer_single_slice(
    scalar: IntegerScalarMode,
    output_shape: Vec<usize>,
    prototype: &IntegerStorage,
) -> BuiltinResult<ModeEvaluation> {
    let values = Tensor::new_integer(
        prototype
            .from_same_class_values(vec![scalar.value])
            .map_err(mode_internal_error)?,
        output_shape.clone(),
    )
    .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
    let freq = Tensor::new(vec![scalar.frequency], output_shape.clone())
        .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
    let ties = prototype
        .from_same_class_values(scalar.ties)
        .map_err(mode_internal_error)?;
    Ok(ModeEvaluation {
        values,
        freq,
        ties: ModeTies::Integer(vec![ties]),
        output_shape,
        output_class: OutputClass::Double,
    })
}

#[derive(Debug)]
struct IntegerScalarMode {
    value: IntValue,
    frequency: f64,
    ties: Vec<IntValue>,
}

fn integer_scalar_mode(values: &[IntValue]) -> IntegerScalarMode {
    let mut sorted = values.to_vec();
    sorted.sort_by(compare_same_class_integer);
    let mut highest_count = 0usize;
    let mut ties = Vec::new();
    let mut start = 0usize;
    while start < sorted.len() {
        let mut end = start + 1;
        while end < sorted.len()
            && compare_same_class_integer(&sorted[start], &sorted[end]) == Ordering::Equal
        {
            end += 1;
        }
        let count = end - start;
        if count > highest_count {
            highest_count = count;
            ties.clear();
            ties.push(sorted[start].clone());
        } else if count == highest_count {
            ties.push(sorted[start].clone());
        }
        start = end;
    }
    IntegerScalarMode {
        value: ties[0].clone(),
        frequency: highest_count as f64,
        ties,
    }
}

fn compare_same_class_integer(left: &IntValue, right: &IntValue) -> Ordering {
    match (left, right) {
        (IntValue::I8(a), IntValue::I8(b)) => a.cmp(b),
        (IntValue::I16(a), IntValue::I16(b)) => a.cmp(b),
        (IntValue::I32(a), IntValue::I32(b)) => a.cmp(b),
        (IntValue::I64(a), IntValue::I64(b)) => a.cmp(b),
        (IntValue::U8(a), IntValue::U8(b)) => a.cmp(b),
        (IntValue::U16(a), IntValue::U16(b)) => a.cmp(b),
        (IntValue::U32(a), IntValue::U32(b)) => a.cmp(b),
        (IntValue::U64(a), IntValue::U64(b)) => a.cmp(b),
        _ => unreachable!("integer storage supplies one homogeneous class"),
    }
}

#[derive(Debug, Clone, Copy)]
enum OutputClass {
    Double,
    Single,
    UInt8,
    UInt16,
    UInt32,
    Logical,
    Int(IntKind),
}

#[derive(Debug, Clone, Copy)]
enum IntKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl OutputClass {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Tensor(tensor) => match tensor.numeric_dtype() {
                NumericDType::F64 => OutputClass::Double,
                NumericDType::F32 => OutputClass::Single,
                NumericDType::I8 => OutputClass::Int(IntKind::I8),
                NumericDType::I16 => OutputClass::Int(IntKind::I16),
                NumericDType::I32 => OutputClass::Int(IntKind::I32),
                NumericDType::I64 => OutputClass::Int(IntKind::I64),
                NumericDType::U8 => OutputClass::UInt8,
                NumericDType::U16 => OutputClass::UInt16,
                NumericDType::U32 => OutputClass::UInt32,
                NumericDType::U64 => OutputClass::Int(IntKind::U64),
            },
            Value::LogicalArray(_) | Value::Bool(_) => OutputClass::Logical,
            Value::Int(value) => OutputClass::Int(IntKind::from_int_value(value)),
            _ => OutputClass::Double,
        }
    }
}

impl IntKind {
    fn from_int_value(value: &IntValue) -> Self {
        match value {
            IntValue::I8(_) => IntKind::I8,
            IntValue::I16(_) => IntKind::I16,
            IntValue::I32(_) => IntKind::I32,
            IntValue::I64(_) => IntKind::I64,
            IntValue::U8(_) => IntKind::U8,
            IntValue::U16(_) => IntKind::U16,
            IntValue::U32(_) => IntKind::U32,
            IntValue::U64(_) => IntKind::U64,
        }
    }

    fn to_value(self, value: f64) -> Value {
        match self {
            IntKind::I8 => Value::Int(IntValue::I8(value.round() as i8)),
            IntKind::I16 => Value::Int(IntValue::I16(value.round() as i16)),
            IntKind::I32 => Value::Int(IntValue::I32(value.round() as i32)),
            IntKind::I64 => Value::Int(IntValue::I64(value.round() as i64)),
            IntKind::U8 => Value::Int(IntValue::U8(value.round() as u8)),
            IntKind::U16 => Value::Int(IntValue::U16(value.round() as u16)),
            IntKind::U32 => Value::Int(IntValue::U32(value.round() as u32)),
            IntKind::U64 => Value::Int(IntValue::U64(value.round() as u64)),
        }
    }

    fn storage_from_f64_values(self, values: &[f64]) -> IntegerStorage {
        match self {
            IntKind::I8 => IntegerStorage::I8(
                values
                    .iter()
                    .map(|value| value.round().clamp(i8::MIN as f64, i8::MAX as f64) as i8)
                    .collect(),
            ),
            IntKind::I16 => IntegerStorage::I16(
                values
                    .iter()
                    .map(|value| value.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
                    .collect(),
            ),
            IntKind::I32 => IntegerStorage::I32(
                values
                    .iter()
                    .map(|value| value.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
                    .collect(),
            ),
            IntKind::I64 => IntegerStorage::I64(
                values
                    .iter()
                    .map(|value| value.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64)
                    .collect(),
            ),
            IntKind::U8 => IntegerStorage::U8(
                values
                    .iter()
                    .map(|value| value.round().clamp(0.0, u8::MAX as f64) as u8)
                    .collect(),
            ),
            IntKind::U16 => IntegerStorage::U16(
                values
                    .iter()
                    .map(|value| value.round().clamp(0.0, u16::MAX as f64) as u16)
                    .collect(),
            ),
            IntKind::U32 => IntegerStorage::U32(
                values
                    .iter()
                    .map(|value| value.round().clamp(0.0, u32::MAX as f64) as u32)
                    .collect(),
            ),
            IntKind::U64 => IntegerStorage::U64(
                values
                    .iter()
                    .map(|value| value.round().clamp(0.0, u64::MAX as f64) as u64)
                    .collect(),
            ),
        }
    }

    fn value_from_int_value(self, value: &IntValue) -> IntValue {
        match self {
            IntKind::I8 => IntValue::I8(value.to_i64().clamp(i8::MIN as i64, i8::MAX as i64) as i8),
            IntKind::I16 => {
                IntValue::I16(value.to_i64().clamp(i16::MIN as i64, i16::MAX as i64) as i16)
            }
            IntKind::I32 => {
                IntValue::I32(value.to_i64().clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            }
            IntKind::I64 => IntValue::I64(value.to_i64()),
            IntKind::U8 => IntValue::U8(value.try_to_u64().unwrap_or(0).min(u8::MAX as u64) as u8),
            IntKind::U16 => {
                IntValue::U16(value.try_to_u64().unwrap_or(0).min(u16::MAX as u64) as u16)
            }
            IntKind::U32 => {
                IntValue::U32(value.try_to_u64().unwrap_or(0).min(u32::MAX as u64) as u32)
            }
            IntKind::U64 => IntValue::U64(value.try_to_u64().unwrap_or(0)),
        }
    }

    fn storage_from_int_values(self, values: Vec<IntValue>) -> IntegerStorage {
        match self {
            IntKind::I8 => IntegerStorage::I8(
                values
                    .into_iter()
                    .map(|value| match self.value_from_int_value(&value) {
                        IntValue::I8(value) => value,
                        _ => unreachable!("int kind creates matching int8 values"),
                    })
                    .collect(),
            ),
            IntKind::I16 => IntegerStorage::I16(
                values
                    .into_iter()
                    .map(|value| match self.value_from_int_value(&value) {
                        IntValue::I16(value) => value,
                        _ => unreachable!("int kind creates matching int16 values"),
                    })
                    .collect(),
            ),
            IntKind::I32 => IntegerStorage::I32(
                values
                    .into_iter()
                    .map(|value| match self.value_from_int_value(&value) {
                        IntValue::I32(value) => value,
                        _ => unreachable!("int kind creates matching int32 values"),
                    })
                    .collect(),
            ),
            IntKind::I64 => IntegerStorage::I64(
                values
                    .into_iter()
                    .map(|value| match self.value_from_int_value(&value) {
                        IntValue::I64(value) => value,
                        _ => unreachable!("int kind creates matching int64 values"),
                    })
                    .collect(),
            ),
            IntKind::U8 => IntegerStorage::U8(
                values
                    .into_iter()
                    .map(|value| match self.value_from_int_value(&value) {
                        IntValue::U8(value) => value,
                        _ => unreachable!("int kind creates matching uint8 values"),
                    })
                    .collect(),
            ),
            IntKind::U16 => IntegerStorage::U16(
                values
                    .into_iter()
                    .map(|value| match self.value_from_int_value(&value) {
                        IntValue::U16(value) => value,
                        _ => unreachable!("int kind creates matching uint16 values"),
                    })
                    .collect(),
            ),
            IntKind::U32 => IntegerStorage::U32(
                values
                    .into_iter()
                    .map(|value| match self.value_from_int_value(&value) {
                        IntValue::U32(value) => value,
                        _ => unreachable!("int kind creates matching uint32 values"),
                    })
                    .collect(),
            ),
            IntKind::U64 => IntegerStorage::U64(
                values
                    .into_iter()
                    .map(|value| match self.value_from_int_value(&value) {
                        IntValue::U64(value) => value,
                        _ => unreachable!("int kind creates matching uint64 values"),
                    })
                    .collect(),
            ),
        }
    }
}

fn tensor_into_class_value(tensor: Tensor, class: OutputClass) -> BuiltinResult<Value> {
    if matches!(class, OutputClass::Double) {
        return Ok(tensor::tensor_into_value(tensor));
    }
    let contains_nan = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .any(|value| value.is_nan());
    match class {
        OutputClass::Double => unreachable!("double handled above"),
        OutputClass::Single => {
            let shape = tensor.shape.clone();
            let values = tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(|value| value as f32)
                .collect();
            Tensor::from_f32(values, shape)
                .map(Value::Tensor)
                .map_err(mode_internal_error)
        }
        OutputClass::UInt8 => {
            if contains_nan {
                return Ok(tensor::tensor_into_value(tensor));
            }
            tensor_into_integer_class_value(tensor, IntKind::U8)
        }
        OutputClass::UInt16 => {
            if contains_nan {
                return Ok(tensor::tensor_into_value(tensor));
            }
            tensor_into_integer_class_value(tensor, IntKind::U16)
        }
        OutputClass::UInt32 => {
            if contains_nan {
                return Ok(tensor::tensor_into_value(tensor));
            }
            tensor_into_integer_class_value(tensor, IntKind::U32)
        }
        OutputClass::Logical => {
            if contains_nan {
                return Ok(tensor::tensor_into_value(tensor));
            }
            let shape = tensor.shape.clone();
            let data: Vec<u8> = tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(|value| if value != 0.0 { 1 } else { 0 })
                .collect();
            if data.len() == 1 {
                Ok(Value::Bool(data[0] != 0))
            } else {
                LogicalArray::new(data, shape)
                    .map(Value::LogicalArray)
                    .map_err(mode_internal_error)
            }
        }
        OutputClass::Int(kind) => {
            if contains_nan {
                return Ok(tensor::tensor_into_value(tensor));
            }
            tensor_into_integer_class_value(tensor, kind)
        }
    }
}

fn tensor_into_class_array_value(tensor: Tensor, class: OutputClass) -> BuiltinResult<Value> {
    if matches!(class, OutputClass::Double) {
        return Ok(Value::Tensor(tensor));
    }
    let contains_nan = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .any(|value| value.is_nan());
    match class {
        OutputClass::Double => unreachable!("double handled above"),
        OutputClass::Single => {
            let shape = tensor.shape.clone();
            let values = tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(|value| value as f32)
                .collect();
            Tensor::from_f32(values, shape)
                .map(Value::Tensor)
                .map_err(mode_internal_error)
        }
        OutputClass::UInt8 => {
            if contains_nan {
                return Ok(Value::Tensor(tensor));
            }
            tensor_into_integer_class_array_value(tensor, IntKind::U8)
        }
        OutputClass::UInt16 => {
            if contains_nan {
                return Ok(Value::Tensor(tensor));
            }
            tensor_into_integer_class_array_value(tensor, IntKind::U16)
        }
        OutputClass::UInt32 => {
            if contains_nan {
                return Ok(Value::Tensor(tensor));
            }
            tensor_into_integer_class_array_value(tensor, IntKind::U32)
        }
        OutputClass::Logical => {
            if contains_nan {
                return Ok(Value::Tensor(tensor));
            }
            let shape = tensor.shape.clone();
            let data: Vec<u8> = tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(|value| if value != 0.0 { 1 } else { 0 })
                .collect();
            LogicalArray::new(data, shape)
                .map(Value::LogicalArray)
                .map_err(mode_internal_error)
        }
        OutputClass::Int(kind) => {
            if contains_nan {
                return Ok(Value::Tensor(tensor));
            }
            tensor_into_integer_class_array_value(tensor, kind)
        }
    }
}

fn tensor_into_integer_class_value(tensor: Tensor, kind: IntKind) -> BuiltinResult<Value> {
    if tensor::is_scalar_tensor(&tensor) {
        if let Some(storage) = tensor.integer_storage() {
            return Ok(Value::Int(
                kind.value_from_int_value(
                    &storage
                        .value_at(0)
                        .expect("one-element integer tensor storage"),
                ),
            ));
        }
        Ok(kind.to_value(tensor::tensor_value_f64(&tensor, 0)))
    } else {
        tensor_into_integer_class_array_value(tensor, kind)
    }
}

fn tensor_into_integer_class_array_value(tensor: Tensor, kind: IntKind) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = if let Some(storage) = tensor.integer_storage() {
        kind.storage_from_int_values(storage.exact_values())
    } else {
        kind.storage_from_f64_values(&tensor::tensor_into_values_f64(tensor))
    };
    Tensor::new_integer(storage, shape)
        .map(Value::Tensor)
        .map_err(mode_internal_error)
}

fn dim_product(dims: &[usize]) -> BuiltinResult<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
        .ok_or_else(|| mode_internal_error("mode: output size overflow"))
}

#[derive(Debug, Clone)]
struct ScalarMode {
    value: f64,
    frequency: f64,
    ties: Vec<f64>,
}

fn scalar_mode(values: &[f64]) -> ScalarMode {
    let mut counts: HashMap<u64, (f64, usize)> = HashMap::new();
    for &v in values {
        if v.is_nan() {
            continue;
        }
        let key = canonical_bits(v);
        counts
            .entry(key)
            .and_modify(|(_, c)| *c += 1)
            .or_insert((v, 1));
    }

    if counts.is_empty() {
        return ScalarMode {
            value: f64::NAN,
            frequency: 0.0,
            ties: Vec::new(),
        };
    }

    let max_count = counts.values().map(|(_, c)| *c).max().unwrap_or(0);
    let mut tied: Vec<f64> = counts
        .values()
        .filter_map(|(v, c)| if *c == max_count { Some(*v) } else { None })
        .collect();
    tied.sort_by(|a, b| compare_f64(*a, *b));

    let smallest = tied[0];
    ScalarMode {
        value: smallest,
        frequency: max_count as f64,
        ties: tied,
    }
}

fn canonical_bits(value: f64) -> u64 {
    // Treat +0.0 and -0.0 as the same key so MATLAB-equivalent counting is preserved.
    if value == 0.0 {
        0u64
    } else {
        value.to_bits()
    }
}

fn compare_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, NumericDType, Tensor, Value};

    fn mode_call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::mode_builtin(value, rest))
    }

    fn mode_outputs(value: Value, rest: Vec<Value>, out_count: usize) -> BuiltinResult<Vec<Value>> {
        let _guard = crate::output_count::push_output_count(Some(out_count));
        let result = mode_call(value, rest)?;
        match result {
            Value::OutputList(list) => Ok(list),
            other => Ok(vec![other]),
        }
    }

    fn expect_tensor(value: &Value) -> &Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mode_type_resolver_reduces_first_dim() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let out = mode_type_resolver(&[ty], &ResolveContext::new(Vec::new()));
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn mode_descriptor_advertises_vecdim_forms() {
        assert!(MODE_DESCRIPTOR
            .signatures
            .iter()
            .any(|signature| signature.label == "M = mode(X, dim_or_vecdim_or_all)"));
        assert!(MODE_DESCRIPTOR
            .signatures
            .iter()
            .flat_map(|signature| signature.inputs)
            .any(|input| input.name == "dim_or_vecdim_or_all"));
    }

    #[test]
    fn mode_scalar_returns_self() {
        let result = mode_call(Value::Num(7.0), Vec::new()).expect("mode");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn mode_vector_simple_majority() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], vec![7, 1]).unwrap();
        let result = mode_call(Value::Tensor(tensor), Vec::new()).expect("mode");
        assert_eq!(result, Value::Num(3.0));
    }

    #[test]
    fn mode_preserves_native_single_values_and_ties() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0, 2.0, 3.0], vec![4, 1]).expect("single tensor");
        let outputs = mode_outputs(Value::Tensor(tensor), Vec::new(), 3).expect("single mode");
        let values = expect_tensor(&outputs[0]);
        assert_eq!(values.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            values.clone().into_numeric_storage().expect("mode storage"),
            runmat_builtins::NumericStorage::F32(vec![2.0])
        );
        let Value::Cell(ties) = &outputs[2] else {
            panic!("expected ties cell");
        };
        let Value::Tensor(tie_values) = &ties.data[0] else {
            panic!("expected tie tensor");
        };
        assert_eq!(tie_values.numeric_dtype(), NumericDType::F32);
    }

    #[test]
    fn mode_ties_return_smallest_with_sorted_set() {
        let tensor = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![1, 4]).unwrap();
        let outputs = mode_outputs(Value::Tensor(tensor), Vec::new(), 3).expect("mode");
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0], Value::Num(1.0));
        assert_eq!(outputs[1], Value::Num(2.0));
        match &outputs[2] {
            Value::Cell(cell) => {
                assert_eq!(cell.shape, vec![1, 1]);
                assert_eq!(cell.data.len(), 1);
                let entry = &cell.data[0];
                match entry {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![2, 1]);
                        assert_eq!(t.materialize_f64(), vec![1.0, 2.0]);
                    }
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn mode_matrix_default_dimension_columnwise() {
        let tensor = Tensor::new(
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let result = mode_call(Value::Tensor(tensor), Vec::new()).expect("mode");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.materialize_f64(), vec![2.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mode_matrix_along_dim_two() {
        let tensor = Tensor::new(
            vec![1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 1.0, 4.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let result =
            mode_call(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("mode");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mode_dimension_argument_reads_typed_integer_scalar_storage() {
        let tensor = Tensor::new(
            vec![1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 1.0, 4.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let dim = Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1]).unwrap();

        let result = mode_call(Value::Tensor(tensor), vec![Value::Tensor(dim)]).expect("mode");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mode_all_reduces_across_all_elements() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 2.0, 3.0, 2.0], vec![2, 3]).unwrap();
        let outputs =
            mode_outputs(Value::Tensor(tensor), vec![Value::from("all")], 3).expect("mode");
        assert_eq!(outputs[0], Value::Num(2.0));
        assert_eq!(outputs[1], Value::Num(3.0));
        match &outputs[2] {
            Value::Cell(cell) => {
                assert_eq!(cell.shape, vec![1, 1]);
                let entry = &cell.data[0];
                match entry {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![1, 1]);
                        assert_eq!(t.materialize_f64(), vec![2.0]);
                    }
                    other => panic!("expected tensor in cell, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn mode_vecdim_reduces_combined_slices_with_ties_and_frequencies() {
        let input = Tensor::new(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], vec![2, 2, 2])
            .expect("input");
        let dims = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2])
            .expect("dimension vector");
        let outputs =
            mode_outputs(Value::Tensor(input), vec![Value::Tensor(dims)], 3).expect("mode");
        let values = expect_tensor(&outputs[0]);
        assert_eq!(values.shape, vec![1, 1, 2]);
        assert_eq!(values.materialize_f64(), vec![1.0, 3.0]);
        let frequency = expect_tensor(&outputs[1]);
        assert_eq!(frequency.shape, vec![1, 1, 2]);
        assert_eq!(frequency.materialize_f64(), vec![2.0, 3.0]);
        let Value::Cell(ties) = &outputs[2] else {
            panic!("expected tied-value cell");
        };
        assert_eq!(ties.shape, vec![1, 1, 2]);
        assert_eq!(
            expect_tensor(&ties.data[0]).materialize_f64(),
            vec![1.0, 2.0]
        );
        assert_eq!(expect_tensor(&ties.data[1]).materialize_f64(), vec![3.0]);
    }

    #[test]
    fn mode_vecdim_preserves_exact_wide_integer_values_and_ties() {
        let wide = u64::MAX - 1;
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![
                wide,
                wide,
                wide - 1,
                wide - 1,
                wide,
                wide,
                wide,
                wide - 1,
            ]),
            vec![2, 2, 2],
        )
        .expect("input");
        let dims = Tensor::new_integer(IntegerStorage::U8(vec![2, 1, 2]), vec![1, 3])
            .expect("dimension vector");
        let outputs =
            mode_outputs(Value::Tensor(input), vec![Value::Tensor(dims)], 3).expect("mode");
        assert_eq!(
            expect_tensor(&outputs[0]).integer_storage(),
            Some(&IntegerStorage::U64(vec![wide - 1, wide]))
        );
        assert_eq!(expect_tensor(&outputs[1]).materialize_f64(), vec![2.0, 3.0]);
        let Value::Cell(ties) = &outputs[2] else {
            panic!("expected tied-value cell");
        };
        assert_eq!(
            expect_tensor(&ties.data[0]).integer_storage(),
            Some(&IntegerStorage::U64(vec![wide - 1, wide]))
        );
        assert_eq!(
            expect_tensor(&ties.data[1]).integer_storage(),
            Some(&IntegerStorage::U64(vec![wide]))
        );
    }

    #[test]
    fn mode_skips_nans_in_majority() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 2.0, 2.0, f64::NAN], vec![5, 1]).unwrap();
        let outputs = mode_outputs(Value::Tensor(tensor), Vec::new(), 2).expect("mode");
        assert_eq!(outputs[0], Value::Num(2.0));
        assert_eq!(outputs[1], Value::Num(2.0));
    }

    #[test]
    fn mode_all_nan_input_returns_nan() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN, f64::NAN], vec![3, 1]).unwrap();
        let outputs = mode_outputs(Value::Tensor(tensor), Vec::new(), 3).expect("mode");
        match &outputs[0] {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
        assert_eq!(outputs[1], Value::Num(0.0));
        match &outputs[2] {
            Value::Cell(cell) => {
                assert_eq!(cell.shape, vec![1, 1]);
                let entry = &cell.data[0];
                match entry {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![0, 1]);
                        assert!(t.materialize_f64().is_empty());
                    }
                    other => panic!("expected empty tensor in cell, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn mode_logical_input_preserves_scalar_class() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0, 1], vec![5, 1]).unwrap();
        let outputs = mode_outputs(Value::LogicalArray(logical), Vec::new(), 2).expect("mode");
        assert_eq!(outputs[0], Value::Bool(true));
        assert_eq!(outputs[1], Value::Num(3.0));
    }

    #[test]
    fn mode_integer_input_works() {
        let result = mode_call(Value::Int(IntValue::I32(5)), Vec::new()).expect("mode");
        assert_eq!(result, Value::Int(IntValue::I32(5)));
    }

    #[test]
    fn mode_uint16_tensor_preserves_value_class() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![9, 10, 2, 10]), vec![1, 4]).unwrap();
        let outputs = mode_outputs(Value::Tensor(tensor), Vec::new(), 3).expect("mode");
        assert_eq!(outputs[0], Value::Int(IntValue::U16(10)));
        assert_eq!(outputs[1], Value::Num(2.0));
        match &outputs[2] {
            Value::Cell(cell) => {
                let entry = &cell.data[0];
                match entry {
                    Value::Tensor(t) => {
                        assert_eq!(t.numeric_dtype(), NumericDType::U16);
                        assert_eq!(t.shape, vec![1, 1]);
                        assert_eq!(t.integer_storage(), Some(&IntegerStorage::U16(vec![10])));
                    }
                    other => panic!("expected uint16 tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn mode_integer_outputs_preserve_authoritative_typed_storage_exactly() {
        let unsigned =
            Tensor::new_integer(IntegerStorage::U16(vec![2, 2, 1, 3, 3, 1]), vec![3, 2]).unwrap();
        let outputs = mode_outputs(Value::Tensor(unsigned), Vec::new(), 3).expect("mode");
        match &outputs[0] {
            Value::Tensor(values) => {
                assert_eq!(values.shape, vec![1, 2]);
                assert_eq!(
                    values.integer_storage(),
                    Some(&IntegerStorage::U16(vec![2, 3]))
                );
            }
            other => panic!("expected uint16 tensor, got {other:?}"),
        }
        match &outputs[2] {
            Value::Cell(cell) => {
                assert_eq!(
                    expect_tensor(&cell.data[0]).integer_storage(),
                    Some(&IntegerStorage::U16(vec![2]))
                );
                assert_eq!(
                    expect_tensor(&cell.data[1]).integer_storage(),
                    Some(&IntegerStorage::U16(vec![3]))
                );
            }
            other => panic!("expected tied-value cell array, got {other:?}"),
        }

        let signed =
            Tensor::new_integer(IntegerStorage::I16(vec![-5, -5, 3, -2, -2, 1]), vec![3, 2])
                .unwrap();
        let outputs = mode_outputs(Value::Tensor(signed), Vec::new(), 3).expect("mode");
        match &outputs[0] {
            Value::Tensor(values) => assert_eq!(
                values.integer_storage(),
                Some(&IntegerStorage::I16(vec![-5, -2]))
            ),
            other => panic!("expected int16 tensor, got {other:?}"),
        }
        match &outputs[2] {
            Value::Cell(cell) => assert_eq!(
                expect_tensor(&cell.data[1]).integer_storage(),
                Some(&IntegerStorage::I16(vec![-2]))
            ),
            other => panic!("expected tied-value cell array, got {other:?}"),
        }
    }

    #[test]
    fn mode_preserves_all_exact_integer_classes_without_float_rounding() {
        let cases = [
            (IntegerStorage::I8(vec![-4, -4, 3]), IntValue::I8(-4)),
            (
                IntegerStorage::I16(vec![-400, -400, 3]),
                IntValue::I16(-400),
            ),
            (
                IntegerStorage::I32(vec![-40_000, -40_000, 3]),
                IntValue::I32(-40_000),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN + 1, i64::MIN + 1, 3]),
                IntValue::I64(i64::MIN + 1),
            ),
            (IntegerStorage::U8(vec![4, 4, 3]), IntValue::U8(4)),
            (IntegerStorage::U16(vec![400, 400, 3]), IntValue::U16(400)),
            (
                IntegerStorage::U32(vec![40_000, 40_000, 3]),
                IntValue::U32(40_000),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX - 1, 3]),
                IntValue::U64(u64::MAX - 1),
            ),
        ];

        for (storage, expected) in cases {
            let input = Tensor::new_integer(storage, vec![1, 3]).unwrap();
            let result = mode_call(Value::Tensor(input), Vec::new()).expect("mode");
            assert_eq!(result, Value::Int(expected));
        }
    }

    #[test]
    fn mode_integer_class_materialization_reads_exact_storage() {
        let wide = u64::MAX - 1;
        let scalar = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        assert_eq!(
            tensor_into_integer_class_value(scalar, IntKind::U64).expect("scalar"),
            Value::Int(IntValue::U64(wide))
        );

        let vector =
            Tensor::new_integer(IntegerStorage::U64(vec![wide, wide - 1]), vec![1, 2]).unwrap();
        let result = tensor_into_integer_class_array_value(vector, IntKind::U64).expect("vector");
        assert_eq!(
            expect_tensor(&result).integer_storage(),
            Some(&IntegerStorage::U64(vec![wide, wide - 1]))
        );
    }

    #[test]
    fn mode_exact_integer_dimension_all_and_ties_preserve_storage() {
        let wide = u64::MAX - 3;
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![wide, wide - 1, wide, wide, wide - 1, wide - 1]),
            vec![2, 3],
        )
        .unwrap();
        let outputs = mode_outputs(Value::Tensor(input), vec![Value::Int(IntValue::I32(2))], 3)
            .expect("mode");
        match &outputs[0] {
            Value::Tensor(values) => {
                assert_eq!(values.shape, vec![2, 1]);
                assert_eq!(
                    values.integer_storage(),
                    Some(&IntegerStorage::U64(vec![wide, wide - 1]))
                );
            }
            other => panic!("expected exact integer tensor, got {other:?}"),
        }
        assert_eq!(
            outputs[1],
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![2, 1]).unwrap())
        );
        match &outputs[2] {
            Value::Cell(cell) => {
                assert_eq!(cell.shape, vec![2, 1]);
                assert_eq!(
                    expect_tensor(&cell.data[0]).integer_storage(),
                    Some(&IntegerStorage::U64(vec![wide]))
                );
                assert_eq!(
                    expect_tensor(&cell.data[1]).integer_storage(),
                    Some(&IntegerStorage::U64(vec![wide - 1]))
                );
            }
            other => panic!("expected tied-value cell array, got {other:?}"),
        }

        let all_input = Tensor::new_integer(
            IntegerStorage::I64(vec![i64::MIN + 2, i64::MIN + 2, 8, 8]),
            vec![2, 2],
        )
        .unwrap();
        let all_outputs =
            mode_outputs(Value::Tensor(all_input), vec![Value::from("all")], 3).expect("mode all");
        assert_eq!(all_outputs[0], Value::Int(IntValue::I64(i64::MIN + 2)));
        assert_eq!(all_outputs[1], Value::Num(2.0));
        match &all_outputs[2] {
            Value::Cell(cell) => assert_eq!(
                expect_tensor(&cell.data[0]).integer_storage(),
                Some(&IntegerStorage::I64(vec![i64::MIN + 2, 8]))
            ),
            other => panic!("expected tied-value cell array, got {other:?}"),
        }
    }

    #[test]
    fn mode_exact_integer_trailing_dimension_preserves_each_value() {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1]),
            vec![2, 1],
        )
        .unwrap();
        let outputs = mode_outputs(Value::Tensor(input), vec![Value::Int(IntValue::I32(5))], 3)
            .expect("mode");
        match &outputs[0] {
            Value::Tensor(values) => assert_eq!(
                values.integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1]))
            ),
            other => panic!("expected exact integer tensor, got {other:?}"),
        }
        match &outputs[2] {
            Value::Cell(cell) => assert_eq!(
                expect_tensor(&cell.data[1]).integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX - 1]))
            ),
            other => panic!("expected tied-value cell array, got {other:?}"),
        }
    }

    #[test]
    fn mode_logical_input_preserves_logical_class() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0, 1], vec![5, 1]).unwrap();
        let outputs = mode_outputs(Value::LogicalArray(logical), Vec::new(), 3).expect("mode");
        assert_eq!(outputs[0], Value::Bool(true));
        assert_eq!(outputs[1], Value::Num(3.0));
        match &outputs[2] {
            Value::Cell(cell) => {
                let entry = &cell.data[0];
                match entry {
                    Value::LogicalArray(array) => {
                        assert_eq!(array.shape, vec![1, 1]);
                        assert_eq!(array.data, vec![1]);
                    }
                    other => panic!("expected logical array inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn mode_empty_input_returns_nan_frequency_zero() {
        let tensor = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let outputs = mode_outputs(Value::Tensor(tensor), Vec::new(), 2).expect("mode");
        match &outputs[0] {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
        assert_eq!(outputs[1], Value::Num(0.0));
    }

    #[test]
    fn mode_empty_integer_inputs_return_documented_double_nan_and_zero_frequency() {
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
            let tensor = Tensor::new_integer(storage, vec![0, 0]).expect("typed empty");
            let outputs = mode_outputs(Value::Tensor(tensor), Vec::new(), 2).expect("mode");
            assert!(
                matches!(&outputs[0], Value::Num(value) if value.is_nan()),
                "unexpected empty integer mode output: {:?}",
                outputs[0]
            );
            assert_eq!(outputs[1], Value::Num(0.0));
        }
    }

    #[test]
    fn mode_rejects_unknown_string_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = mode_call(Value::Tensor(tensor), vec![Value::from("flat")]).unwrap_err();
        assert_eq!(err.identifier(), MODE_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn mode_rejects_negative_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = mode_call(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))]).unwrap_err();
        assert_eq!(err.identifier(), MODE_ERROR_INVALID_DIMENSION.identifier);
    }

    #[test]
    fn mode_gpu_fallback_preserves_residency_for_values_and_frequency() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let source = Tensor::new(vec![1.0, 2.0, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &source.materialize_f64(),
                shape: &source.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let outputs = mode_outputs(Value::GpuTensor(handle), Vec::new(), 2).expect("mode");
            assert_eq!(
                test_support::gather(outputs[0].clone())
                    .expect("gather mode")
                    .materialize_f64(),
                vec![2.0]
            );
            assert_eq!(
                test_support::gather(outputs[1].clone())
                    .expect("gather frequency")
                    .materialize_f64(),
                vec![2.0]
            );

            let logical_source = Tensor::new(vec![0.0, 1.0, 1.0], vec![3, 1]).unwrap();
            let logical_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &logical_source.materialize_f64(),
                    shape: &logical_source.shape,
                })
                .expect("upload logical");
            runmat_accelerate_api::set_handle_logical(&logical_handle, true);
            let logical_outputs =
                mode_outputs(Value::GpuTensor(logical_handle), Vec::new(), 2).expect("mode");
            let Value::GpuTensor(logical_mode) = &logical_outputs[0] else {
                panic!("expected resident logical mode");
            };
            assert!(runmat_accelerate_api::handle_is_logical(logical_mode));
            assert_eq!(
                test_support::gather(logical_outputs[0].clone())
                    .expect("gather logical mode")
                    .materialize_f64(),
                vec![1.0]
            );
            assert_eq!(
                test_support::gather(logical_outputs[1].clone())
                    .expect("gather logical frequency")
                    .materialize_f64(),
                vec![2.0]
            );
        });
    }

    #[test]
    fn mode_integer_gpu_fallback_preserves_exact_values_frequency_and_ties() {
        use crate::builtins::common::{gpu_helpers, test_support};

        test_support::with_test_provider(|provider| {
            let wide = u64::MAX - 2;
            let source = Tensor::new_integer(
                IntegerStorage::U64(vec![wide, wide - 1, wide, wide - 1]),
                vec![4, 1],
            )
            .expect("integer source");
            let handle = gpu_helpers::upload_tensor(provider, &source).expect("integer upload");
            let outputs =
                mode_outputs(Value::GpuTensor(handle), Vec::new(), 3).expect("integer mode");

            let Value::GpuTensor(mode_handle) = &outputs[0] else {
                panic!("expected resident integer mode");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(mode_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(
                test_support::gather(outputs[0].clone())
                    .expect("gather mode")
                    .integer_storage(),
                Some(&IntegerStorage::U64(vec![wide - 1]))
            );
            assert_eq!(
                test_support::gather(outputs[1].clone())
                    .expect("gather frequency")
                    .materialize_f64(),
                vec![2.0]
            );
            let Value::Cell(ties) = &outputs[2] else {
                panic!("expected tied-value cell");
            };
            let Value::GpuTensor(tie_handle) = &ties.data[0] else {
                panic!("expected resident integer tied values");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(tie_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(
                test_support::gather(ties.data[0].clone())
                    .expect("gather ties")
                    .integer_storage(),
                Some(&IntegerStorage::U64(vec![wide - 1, wide]))
            );
        });
    }

    #[test]
    fn mode_dim_beyond_ndims_preserves_input_with_unit_frequency() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let outputs = mode_outputs(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(5))], 2)
            .expect("mode");
        match &outputs[0] {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor of frequencies, got {other:?}"),
        }
    }
}
