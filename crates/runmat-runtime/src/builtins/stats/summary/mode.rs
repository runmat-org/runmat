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
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, LogicalArray, NumericDType, ResolveContext, Tensor, Type, Value,
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
        name: "dim_or_all",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reduction axis (positive integer) or 'all'.",
    },
];

const MODE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "M = mode(X)",
        inputs: &MODE_INPUTS_X,
        outputs: &MODE_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = mode(X, dim_or_all)",
        inputs: &MODE_INPUTS_X_DIM_OR_ALL,
        outputs: &MODE_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F] = mode(X)",
        inputs: &MODE_INPUTS_X,
        outputs: &MODE_OUTPUT_MF,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F] = mode(X, dim_or_all)",
        inputs: &MODE_INPUTS_X_DIM_OR_ALL,
        outputs: &MODE_OUTPUT_MF,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F, C] = mode(X)",
        inputs: &MODE_INPUTS_X,
        outputs: &MODE_OUTPUT_MFC,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F, C] = mode(X, dim_or_all)",
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

const MODE_ERROR_GPU_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MODE.GPU_UNSUPPORTED",
    identifier: Some("RunMat:mode:GpuUnsupported"),
    when: "Input is GPU-resident and mode requires host data.",
    message: "mode: GPU tensors must be gathered to the host before mode can be computed",
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

const MODE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    MODE_ERROR_INVALID_ARGUMENT,
    MODE_ERROR_INVALID_DIMENSION,
    MODE_ERROR_GPU_UNSUPPORTED,
    MODE_ERROR_COMPLEX_UNSUPPORTED,
    MODE_ERROR_INTERNAL,
];

pub const MODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MODE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MODE_ERRORS,
};

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
    builtin_path = "crate::builtins::stats::summary::mode"
)]
async fn mode_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let parsed = parse_arguments(&rest).await?;
    let output_class = OutputClass::from_value(&value);
    let eval = mode_evaluate(value, parsed, output_class)?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.into_values_value()?]));
        }
        if out_count == 2 {
            let (values, freq) = eval.into_pair()?;
            return Ok(Value::OutputList(vec![values, freq]));
        }
        let (values, freq, cells) = eval.into_triple()?;
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![values, freq, cells],
        ));
    }
    eval.into_values_value()
}

#[derive(Clone, Debug)]
enum ModeAxes {
    Default,
    Dim(usize),
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

    let dim = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            tensor::dimension_from_value_async(value, NAME, false)
                .await
                .map_err(|e| mode_error_with(&MODE_ERROR_INVALID_DIMENSION, e))?
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            tensor::dimension_from_value_async(value, NAME, false)
                .await
                .map_err(|e| mode_error_with(&MODE_ERROR_INVALID_DIMENSION, e))?
        }
        Value::LogicalArray(la) if la.data.len() == 1 => {
            tensor::dimension_from_value_async(value, NAME, false)
                .await
                .map_err(|e| mode_error_with(&MODE_ERROR_INVALID_DIMENSION, e))?
        }
        _ => return Ok(None),
    };

    let Some(dim) = dim else {
        return Ok(None);
    };
    if dim < 1 {
        return Err(mode_error(&MODE_ERROR_INVALID_DIMENSION));
    }
    Ok(Some(ModeAxes::Dim(dim)))
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
    ties: Vec<Vec<f64>>,
    /// Shape of the M / F tensors (also the cell array shape for C).
    output_shape: Vec<usize>,
    /// MATLAB class to preserve for `M` and the tied values in `C`.
    output_class: OutputClass,
}

impl ModeEvaluation {
    fn empty(output_shape: Vec<usize>, output_class: OutputClass) -> BuiltinResult<Self> {
        let len = tensor::element_count(&output_shape);
        let values = Tensor::new(vec![f64::NAN; len], output_shape.clone())
            .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
        let freq = Tensor::new(vec![0.0; len], output_shape.clone())
            .map_err(|e| mode_internal_error(format!("mode: {e}")))?;
        let ties = vec![Vec::new(); len];
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
    ties: Vec<Vec<f64>>,
    output_shape: &[usize],
    output_class: OutputClass,
) -> BuiltinResult<Value> {
    let cell_shape = if output_shape.is_empty() {
        vec![1, 1]
    } else {
        output_shape.to_vec()
    };
    let mut cell_values: Vec<Value> = Vec::with_capacity(ties.len());
    for entry in ties {
        let rows = entry.len();
        let tensor = Tensor::new(entry, vec![rows, 1])
            .map_err(|e| mode_internal_error(format!("mode: cell construction failed: {e}")))?;
        cell_values.push(tensor_into_class_array_value(tensor, output_class)?);
    }
    crate::make_cell_with_shape(cell_values, cell_shape).map_err(mode_internal_error)
}

fn mode_evaluate(
    value: Value,
    args: ParsedArguments,
    output_class: OutputClass,
) -> BuiltinResult<ModeEvaluation> {
    let tensor = materialize_tensor(value)?;
    match args.axes {
        ModeAxes::Default => {
            let dim = default_dimension_from_shape(&tensor.shape);
            reduce_along_dim(tensor, dim, output_class)
        }
        ModeAxes::Dim(dim) => reduce_along_dim(tensor, dim, output_class),
        ModeAxes::All => reduce_all(tensor, output_class),
    }
}

fn materialize_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(_) => Err(mode_error(&MODE_ERROR_GPU_UNSUPPORTED)),
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
    if tensor.data.is_empty() {
        return ModeEvaluation::empty(output_shape, output_class);
    }
    let scalar = scalar_mode(&tensor.data);
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
    let ties = vec![scalar.ties];
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

    if tensor.shape.is_empty() {
        let scalar_value = tensor.data.first().copied().unwrap_or(f64::NAN);
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

    if dim > tensor.shape.len() {
        // Reducing along a trailing singleton: every slice has one element.
        let output_shape = tensor.shape.clone();
        let len = tensor::element_count(&output_shape);
        let mut values = Vec::with_capacity(len);
        let mut freq = Vec::with_capacity(len);
        let mut ties = Vec::with_capacity(len);
        for &v in &tensor.data {
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
            ties,
            output_shape,
            output_class,
        });
    }

    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let mut output_shape = tensor.shape.clone();
    output_shape[dim_index] = 1;

    if reduce_len == 0 || tensor.data.is_empty() {
        return ModeEvaluation::empty(output_shape, output_class);
    }

    let stride_before = dim_product(&tensor.shape[..dim_index])?;
    let stride_after = dim_product(&tensor.shape[dim_index + 1..])?;
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
                slice.push(tensor.data[idx]);
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
        ties,
        output_shape,
        output_class,
    })
}

#[derive(Debug, Clone, Copy)]
enum OutputClass {
    Double,
    Single,
    UInt8,
    UInt16,
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
            Value::Tensor(tensor) => match tensor.dtype {
                NumericDType::F64 => OutputClass::Double,
                NumericDType::F32 => OutputClass::Single,
                NumericDType::U8 => OutputClass::UInt8,
                NumericDType::U16 => OutputClass::UInt16,
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
}

fn tensor_into_class_value(mut tensor: Tensor, class: OutputClass) -> BuiltinResult<Value> {
    let contains_nan = tensor.data.iter().any(|value| value.is_nan());
    match class {
        OutputClass::Double => Ok(tensor::tensor_into_value(tensor)),
        OutputClass::Single => {
            for value in &mut tensor.data {
                *value = (*value as f32) as f64;
            }
            tensor.dtype = NumericDType::F32;
            Ok(Value::Tensor(tensor))
        }
        OutputClass::UInt8 => {
            if contains_nan {
                return Ok(tensor::tensor_into_value(tensor));
            }
            for value in &mut tensor.data {
                *value = value.round().clamp(0.0, u8::MAX as f64);
            }
            tensor.dtype = NumericDType::U8;
            if tensor.data.len() == 1 {
                Ok(Value::Int(IntValue::U8(tensor.data[0] as u8)))
            } else {
                Ok(Value::Tensor(tensor))
            }
        }
        OutputClass::UInt16 => {
            if contains_nan {
                return Ok(tensor::tensor_into_value(tensor));
            }
            for value in &mut tensor.data {
                *value = value.round().clamp(0.0, u16::MAX as f64);
            }
            tensor.dtype = NumericDType::U16;
            if tensor.data.len() == 1 {
                Ok(Value::Int(IntValue::U16(tensor.data[0] as u16)))
            } else {
                Ok(Value::Tensor(tensor))
            }
        }
        OutputClass::Logical => {
            if contains_nan {
                return Ok(tensor::tensor_into_value(tensor));
            }
            let data: Vec<u8> = tensor
                .data
                .iter()
                .map(|value| if *value != 0.0 { 1 } else { 0 })
                .collect();
            if data.len() == 1 {
                Ok(Value::Bool(data[0] != 0))
            } else {
                LogicalArray::new(data, tensor.shape)
                    .map(Value::LogicalArray)
                    .map_err(mode_internal_error)
            }
        }
        OutputClass::Int(kind) => {
            if contains_nan {
                return Ok(tensor::tensor_into_value(tensor));
            }
            if tensor.data.len() == 1 {
                Ok(kind.to_value(tensor.data[0]))
            } else {
                Ok(tensor::tensor_into_value(tensor))
            }
        }
    }
}

fn tensor_into_class_array_value(mut tensor: Tensor, class: OutputClass) -> BuiltinResult<Value> {
    let contains_nan = tensor.data.iter().any(|value| value.is_nan());
    match class {
        OutputClass::Double => Ok(Value::Tensor(tensor)),
        OutputClass::Single => {
            for value in &mut tensor.data {
                *value = (*value as f32) as f64;
            }
            tensor.dtype = NumericDType::F32;
            Ok(Value::Tensor(tensor))
        }
        OutputClass::UInt8 => {
            if contains_nan {
                return Ok(Value::Tensor(tensor));
            }
            for value in &mut tensor.data {
                *value = value.round().clamp(0.0, u8::MAX as f64);
            }
            tensor.dtype = NumericDType::U8;
            Ok(Value::Tensor(tensor))
        }
        OutputClass::UInt16 => {
            if contains_nan {
                return Ok(Value::Tensor(tensor));
            }
            for value in &mut tensor.data {
                *value = value.round().clamp(0.0, u16::MAX as f64);
            }
            tensor.dtype = NumericDType::U16;
            Ok(Value::Tensor(tensor))
        }
        OutputClass::Logical => {
            if contains_nan {
                return Ok(Value::Tensor(tensor));
            }
            let data: Vec<u8> = tensor
                .data
                .iter()
                .map(|value| if *value != 0.0 { 1 } else { 0 })
                .collect();
            LogicalArray::new(data, tensor.shape)
                .map(Value::LogicalArray)
                .map_err(mode_internal_error)
        }
        OutputClass::Int(kind) => {
            if contains_nan || tensor.data.len() != 1 {
                return Ok(Value::Tensor(tensor));
            }
            Ok(kind.to_value(tensor.data[0]))
        }
    }
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
                        assert_eq!(t.data, vec![1.0, 2.0]);
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
                assert_eq!(t.data, vec![2.0, 3.0, 4.0]);
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
                assert_eq!(t.data, vec![1.0, 2.0, 1.0]);
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
                        assert_eq!(t.data, vec![2.0]);
                    }
                    other => panic!("expected tensor in cell, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
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
                        assert!(t.data.is_empty());
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
        let mut tensor = Tensor::new(vec![9.0, 10.0, 2.0, 10.0], vec![1, 4]).unwrap();
        tensor.dtype = NumericDType::U16;
        let outputs = mode_outputs(Value::Tensor(tensor), Vec::new(), 3).expect("mode");
        assert_eq!(outputs[0], Value::Int(IntValue::U16(10)));
        assert_eq!(outputs[1], Value::Num(2.0));
        match &outputs[2] {
            Value::Cell(cell) => {
                let entry = &cell.data[0];
                match entry {
                    Value::Tensor(t) => {
                        assert_eq!(t.dtype, NumericDType::U16);
                        assert_eq!(t.shape, vec![1, 1]);
                        assert_eq!(t.data, vec![10.0]);
                    }
                    other => panic!("expected uint16 tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
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
    fn mode_rejects_gpu_input_without_gather() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let source = Tensor::new(vec![1.0, 2.0, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &source.data,
                shape: &source.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let err = mode_call(Value::GpuTensor(handle), Vec::new()).unwrap_err();
            assert_eq!(err.identifier(), MODE_ERROR_GPU_UNSUPPORTED.identifier);
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
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.data, vec![1.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor of frequencies, got {other:?}"),
        }
    }
}
