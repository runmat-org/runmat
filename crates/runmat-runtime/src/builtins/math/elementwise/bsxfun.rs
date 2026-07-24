//! MATLAB-compatible `bsxfun` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, LogicalArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::user_functions::resolve_semantic_function_by_name;
use crate::{
    build_runtime_error, call_feval_async_with_outputs, gather_if_needed_async, BuiltinResult,
    RuntimeError,
};

const BUILTIN_NAME: &str = "bsxfun";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::bsxfun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("host-binary-map"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Executes arbitrary binary callbacks on the host after gathering gpuArray inputs; use direct elementwise operators for provider-native GPU execution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::bsxfun")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Callback execution happens through feval; fusion planners should treat bsxfun as a fusion barrier.",
};

const BSXFUN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise callback result after singleton expansion.",
}];

const BSXFUN_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Binary function handle or callable text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left scalar or array input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right scalar or array input.",
    },
];

const BSXFUN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = bsxfun(fun, A, B)",
    inputs: &BSXFUN_INPUTS,
    outputs: &BSXFUN_OUTPUT,
}];

const BSXFUN_ERROR_INVALID_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.INVALID_FUNCTION",
    identifier: Some("RunMat:bsxfun:InvalidFunction"),
    when: "The first input cannot be used as a binary function.",
    message: "bsxfun: first input must be a function handle or callable name",
};

const BSXFUN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.INVALID_INPUT",
    identifier: Some("RunMat:bsxfun:InvalidInput"),
    when: "Input arrays use unsupported types for bsxfun scalar expansion.",
    message: "bsxfun: inputs must be numeric, logical, complex, or character arrays",
};

const BSXFUN_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.SIZE_MISMATCH",
    identifier: Some("RunMat:bsxfun:SizeMismatch"),
    when: "Input arrays are not compatible for singleton expansion.",
    message: "bsxfun: input sizes are not compatible for singleton expansion",
};

const BSXFUN_ERROR_FUNCTION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.FUNCTION_ERROR",
    identifier: Some("RunMat:bsxfun:FunctionError"),
    when: "The binary function errors or does not return scalar-uniform values.",
    message: "bsxfun: callback execution error",
};

const BSXFUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.INTERNAL",
    identifier: Some("RunMat:bsxfun:Internal"),
    when: "Internal output allocation or GPU gather fails.",
    message: "bsxfun: internal error",
};

const BSXFUN_ERRORS: [BuiltinErrorDescriptor; 5] = [
    BSXFUN_ERROR_INVALID_FUNCTION,
    BSXFUN_ERROR_INVALID_INPUT,
    BSXFUN_ERROR_SIZE_MISMATCH,
    BSXFUN_ERROR_FUNCTION_ERROR,
    BSXFUN_ERROR_INTERNAL,
];

pub const BSXFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BSXFUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BSXFUN_ERRORS,
};

#[runtime_builtin(
    name = "bsxfun",
    category = "math/elementwise",
    summary = "Apply a binary function with singleton expansion.",
    keywords = "bsxfun,binary singleton expansion,implicit expansion,function handle",
    accel = "host",
    descriptor(crate::builtins::math::elementwise::bsxfun::BSXFUN_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::bsxfun"
)]
async fn bsxfun_builtin(function: Value, left: Value, right: Value) -> BuiltinResult<Value> {
    validate_function(&function)?;
    let output_hint = CallbackOutputHint::for_function(&function);
    let left = gather_if_needed_async(&left)
        .await
        .map_err(|flow| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, flow.to_string()))?;
    let right = gather_if_needed_async(&right)
        .await
        .map_err(|flow| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, flow.to_string()))?;
    let left = ArrayInput::from_value(left)?;
    let right = ArrayInput::from_value(right)?;
    let plan = BsxfunBroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_SIZE_MISMATCH, err))?;

    let mut collector = UniformCollector::Pending;
    for (_, left_idx, right_idx) in plan.iter() {
        let args = [left.value_at(left_idx)?, right.value_at(right_idx)?];
        let mut value = call_feval_async_with_outputs(function.clone(), &args, 1)
            .await
            .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_FUNCTION_ERROR, err.message()))?;
        value = gather_if_needed_async(&value)
            .await
            .map_err(|flow| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, flow.to_string()))?;
        if output_hint == CallbackOutputHint::Logical {
            value = logicalize_callback_output(value)?;
        }
        collector.push(&value)?;
    }

    collector.finish(plan.output_shape(), output_hint)
}

fn validate_function(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_) => Ok(()),
        other => Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_INVALID_FUNCTION,
            format!("got {other:?}"),
        )),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CallbackOutputHint {
    None,
    Logical,
}

impl CallbackOutputHint {
    fn for_function(value: &Value) -> Self {
        let Some(name) = callback_function_name(value) else {
            return Self::None;
        };
        if resolve_semantic_function_by_name(&name).is_some() {
            return Self::None;
        }
        match name.as_str() {
            "eq" | "ne" | "gt" | "ge" | "lt" | "le" | "and" | "or" | "xor" => Self::Logical,
            _ => Self::None,
        }
    }
}

fn callback_function_name(value: &Value) -> Option<String> {
    match value {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => normalize_callback_name(name),
        Value::String(name) => normalize_callback_name(name),
        Value::StringArray(array) if array.data.len() == 1 => {
            normalize_callback_name(array.data.first()?)
        }
        Value::CharArray(array) if array.rows == 1 => {
            let name: String = array.data.iter().collect();
            normalize_callback_name(&name)
        }
        _ => None,
    }
}

fn normalize_callback_name(name: &str) -> Option<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.strip_prefix('@').unwrap_or(trimmed).to_string())
}

fn logicalize_callback_output(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => Ok(value),
        Value::Num(value) => Ok(Value::Bool(value != 0.0)),
        Value::Int(value) => Ok(Value::Bool(value.to_f64() != 0.0)),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(Value::Bool(tensor.data[0] != 0.0)),
        other => Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            format!("logical callback must return scalar logical values (got {other:?})"),
        )),
    }
}

struct ArrayInput {
    data: ArrayData,
    shape: Vec<usize>,
}

impl ArrayInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        let data = ArrayData::from_value(value)?;
        let shape = data.shape_vec();
        Ok(Self { data, shape })
    }

    fn value_at(&self, idx: usize) -> BuiltinResult<Value> {
        self.data.value_at(idx)
    }
}

enum ArrayData {
    Tensor(Tensor),
    Logical(LogicalArray),
    Complex(ComplexTensor),
    Char(CharArray),
    Scalar(Value),
}

impl ArrayData {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => Ok(Self::Tensor(tensor)),
            Value::LogicalArray(array) => Ok(Self::Logical(array)),
            Value::ComplexTensor(tensor) => Ok(Self::Complex(tensor)),
            Value::CharArray(array) => Ok(Self::Char(array)),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => {
                Ok(Self::Scalar(value))
            }
            other => Err(bsxfun_error_with_detail(
                &BSXFUN_ERROR_INVALID_INPUT,
                format!("unsupported input type {other:?}"),
            )),
        }
    }

    fn shape_vec(&self) -> Vec<usize> {
        match self {
            Self::Tensor(tensor) => normalize_shape(&tensor.shape),
            Self::Logical(array) => normalize_shape(&array.shape),
            Self::Complex(tensor) => normalize_shape(&tensor.shape),
            Self::Char(array) => vec![array.rows, array.cols],
            Self::Scalar(_) => vec![1, 1],
        }
    }

    fn value_at(&self, idx: usize) -> BuiltinResult<Value> {
        match self {
            Self::Tensor(tensor) => tensor
                .data
                .get(idx)
                .copied()
                .map(Value::Num)
                .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL)),
            Self::Logical(array) => array
                .data
                .get(idx)
                .map(|bit| Value::Bool(*bit != 0))
                .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL)),
            Self::Complex(tensor) => tensor
                .data
                .get(idx)
                .copied()
                .map(|(re, im)| Value::Complex(re, im))
                .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL)),
            Self::Char(array) => char_value_at(array, idx),
            Self::Scalar(value) => Ok(value.clone()),
        }
    }
}

fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn char_value_at(array: &CharArray, idx: usize) -> BuiltinResult<Value> {
    if array.rows == 0 || array.cols == 0 {
        let empty = CharArray::new(Vec::new(), 0, 0)
            .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err))?;
        return Ok(Value::CharArray(empty));
    }
    let row = idx % array.rows;
    let col = idx / array.rows;
    let data_idx = row * array.cols + col;
    let ch = array
        .data
        .get(data_idx)
        .copied()
        .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL))?;
    let scalar = CharArray::new(vec![ch], 1, 1)
        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err))?;
    Ok(Value::CharArray(scalar))
}

struct BsxfunBroadcastPlan {
    output_shape: Vec<usize>,
    left_shape: Vec<usize>,
    right_shape: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
    len: usize,
}

impl BsxfunBroadcastPlan {
    fn new(left_shape: &[usize], right_shape: &[usize]) -> Result<Self, String> {
        let rank = left_shape.len().max(right_shape.len());
        let left_shape = extend_shape_trailing(left_shape, rank);
        let right_shape = extend_shape_trailing(right_shape, rank);
        let mut output_shape = Vec::with_capacity(rank);

        for dim in 0..rank {
            let left = left_shape[dim];
            let right = right_shape[dim];
            if left == right {
                output_shape.push(left);
            } else if left == 1 {
                output_shape.push(right);
            } else if right == 1 {
                output_shape.push(left);
            } else {
                return Err(format!(
                    "broadcast: non-singleton dimension mismatch (dimension {}: {} vs {})",
                    dim + 1,
                    left,
                    right
                ));
            }
        }

        let len = checked_element_count(&output_shape)?;
        let left_strides = checked_strides(&left_shape)?;
        let right_strides = checked_strides(&right_shape)?;

        Ok(Self {
            output_shape,
            left_shape,
            right_shape,
            left_strides,
            right_strides,
            len,
        })
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        (0..self.len).map(|idx| {
            (
                idx,
                broadcast_input_index(
                    idx,
                    &self.output_shape,
                    &self.left_shape,
                    &self.left_strides,
                ),
                broadcast_input_index(
                    idx,
                    &self.output_shape,
                    &self.right_shape,
                    &self.right_strides,
                ),
            )
        })
    }

    fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
}

fn extend_shape_trailing(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut extended = Vec::with_capacity(rank);
    extended.extend_from_slice(shape);
    extended.resize(rank, 1);
    extended
}

fn checked_element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().copied().try_fold(1usize, |acc, extent| {
        acc.checked_mul(extent)
            .ok_or_else(|| "broadcast: output size exceeds platform limits".to_string())
    })
}

fn checked_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| "broadcast: input size exceeds platform limits".to_string())?;
    }
    Ok(strides)
}

fn broadcast_input_index(
    output_idx: usize,
    output_shape: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
) -> usize {
    let mut remaining = output_idx;
    let mut offset = 0usize;
    for dim in 0..output_shape.len() {
        let extent = output_shape[dim];
        let coord = remaining % extent;
        remaining /= extent;

        if input_shape[dim] > 1 {
            offset += coord * input_strides[dim];
        }
    }
    offset
}

enum UniformCollector {
    Pending,
    Double(Vec<f64>),
    Logical(Vec<u8>),
    Complex(Vec<(f64, f64)>),
    Char(Vec<char>),
}

impl UniformCollector {
    fn push(&mut self, value: &Value) -> BuiltinResult<()> {
        match self {
            Self::Pending => match classify_value(value)? {
                ClassifiedValue::Logical(bit) => {
                    *self = Self::Logical(vec![u8::from(bit)]);
                    Ok(())
                }
                ClassifiedValue::Double(value) => {
                    *self = Self::Double(vec![value]);
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    *self = Self::Complex(vec![value]);
                    Ok(())
                }
                ClassifiedValue::Char(value) => {
                    *self = Self::Char(vec![value]);
                    Ok(())
                }
            },
            Self::Double(values) => match classify_value(value)? {
                ClassifiedValue::Logical(bit) => {
                    values.push(if bit { 1.0 } else { 0.0 });
                    Ok(())
                }
                ClassifiedValue::Double(value) => {
                    values.push(value);
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    let mut promoted: Vec<(f64, f64)> =
                        values.iter().map(|&real| (real, 0.0)).collect();
                    promoted.push(value);
                    *self = Self::Complex(promoted);
                    Ok(())
                }
                ClassifiedValue::Char(ch) => {
                    values.push(ch as u32 as f64);
                    Ok(())
                }
            },
            Self::Logical(bits) => match classify_value(value)? {
                ClassifiedValue::Logical(bit) => {
                    bits.push(u8::from(bit));
                    Ok(())
                }
                ClassifiedValue::Double(value) => {
                    let mut promoted: Vec<f64> = bits
                        .iter()
                        .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                        .collect();
                    promoted.push(value);
                    *self = Self::Double(promoted);
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    let mut promoted: Vec<(f64, f64)> = bits
                        .iter()
                        .map(|&bit| (if bit != 0 { 1.0 } else { 0.0 }, 0.0))
                        .collect();
                    promoted.push(value);
                    *self = Self::Complex(promoted);
                    Ok(())
                }
                ClassifiedValue::Char(ch) => {
                    let mut promoted: Vec<f64> = bits
                        .iter()
                        .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                        .collect();
                    promoted.push(ch as u32 as f64);
                    *self = Self::Double(promoted);
                    Ok(())
                }
            },
            Self::Complex(values) => match classify_value(value)? {
                ClassifiedValue::Logical(bit) => {
                    values.push((if bit { 1.0 } else { 0.0 }, 0.0));
                    Ok(())
                }
                ClassifiedValue::Double(value) => {
                    values.push((value, 0.0));
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    values.push(value);
                    Ok(())
                }
                ClassifiedValue::Char(ch) => {
                    values.push((ch as u32 as f64, 0.0));
                    Ok(())
                }
            },
            Self::Char(chars) => match classify_value(value)? {
                ClassifiedValue::Char(ch) => {
                    chars.push(ch);
                    Ok(())
                }
                ClassifiedValue::Logical(bit) => {
                    let mut promoted: Vec<f64> = chars.iter().map(|&ch| ch as u32 as f64).collect();
                    promoted.push(if bit { 1.0 } else { 0.0 });
                    *self = Self::Double(promoted);
                    Ok(())
                }
                ClassifiedValue::Double(value) => {
                    let mut promoted: Vec<f64> = chars.iter().map(|&ch| ch as u32 as f64).collect();
                    promoted.push(value);
                    *self = Self::Double(promoted);
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    let mut promoted: Vec<(f64, f64)> =
                        chars.iter().map(|&ch| (ch as u32 as f64, 0.0)).collect();
                    promoted.push(value);
                    *self = Self::Complex(promoted);
                    Ok(())
                }
            },
        }
    }

    fn finish(self, shape: &[usize], output_hint: CallbackOutputHint) -> BuiltinResult<Value> {
        match self {
            Self::Pending => {
                let len = shape.iter().copied().product();
                match output_hint {
                    CallbackOutputHint::Logical => LogicalArray::new(vec![0; len], shape.to_vec())
                        .map(Value::LogicalArray)
                        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
                    CallbackOutputHint::None => Tensor::new(vec![0.0; len], shape.to_vec())
                        .map(Value::Tensor)
                        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
                }
            }
            Self::Double(values) => Tensor::new(values, shape.to_vec())
                .map(Value::Tensor)
                .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
            Self::Logical(bits) => LogicalArray::new(bits, shape.to_vec())
                .map(Value::LogicalArray)
                .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
            Self::Complex(values) => ComplexTensor::new(values, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
            Self::Char(chars) => finish_char_output(chars, shape),
        }
    }
}

enum ClassifiedValue {
    Logical(bool),
    Double(f64),
    Complex((f64, f64)),
    Char(char),
}

fn classify_value(value: &Value) -> BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(value) => Ok(ClassifiedValue::Logical(*value)),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(ClassifiedValue::Logical(array.data[0] != 0))
        }
        Value::Num(value) => Ok(ClassifiedValue::Double(*value)),
        Value::Int(value) => Ok(ClassifiedValue::Double(value.to_f64())),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            Ok(ClassifiedValue::Double(tensor.data[0]))
        }
        Value::Complex(re, im) => Ok(ClassifiedValue::Complex((*re, *im))),
        Value::ComplexTensor(tensor) if tensor.data.len() == 1 => {
            Ok(ClassifiedValue::Complex(tensor.data[0]))
        }
        Value::CharArray(array) if array.rows * array.cols == 1 => {
            Ok(ClassifiedValue::Char(array.data.first().copied().unwrap_or('\0')))
        }
        other => Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            format!("callback must return scalar numeric, logical, complex, or character values (got {other:?})"),
        )),
    }
}

fn finish_char_output(chars: Vec<char>, shape: &[usize]) -> BuiltinResult<Value> {
    let normalized_shape = normalize_shape(shape);
    if normalized_shape.len() > 2 {
        return Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            "character callback outputs must form a 2-D char array",
        ));
    }
    let rows = normalized_shape.first().copied().unwrap_or(1);
    let cols = normalized_shape.get(1).copied().unwrap_or(1);
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        bsxfun_error_with_detail(
            &BSXFUN_ERROR_INTERNAL,
            "character output size exceeds platform limits",
        )
    })?;
    if expected != chars.len() {
        return Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            "callback returned the wrong number of characters",
        ));
    }

    let mut row_major = vec!['\0'; expected];
    for col in 0..cols {
        for row in 0..rows {
            let col_major_idx = row + col * rows;
            let row_major_idx = row * cols + col;
            row_major[row_major_idx] = chars[col_major_idx];
        }
    }

    CharArray::new(row_major, rows, cols)
        .map(Value::CharArray)
        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err))
}

fn bsxfun_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn bsxfun_error_with_detail(
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::Arc;

    fn call(function: Value, left: Value, right: Value) -> BuiltinResult<Value> {
        block_on(bsxfun_builtin(function, left, right))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_plus_expands_row_and_column_vectors() {
        let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap();
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(column),
            Value::Tensor(row),
        )
        .expect("bsxfun plus");

        let Value::Tensor(tensor) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(tensor.shape, vec![3, 2]);
        assert_eq!(tensor.data, vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_nd_expansion_uses_matlab_trailing_dimensions() {
        let left =
            Tensor::new((1..=24).map(|value| value as f64).collect(), vec![2, 3, 4]).unwrap();
        let right = Tensor::new(vec![100.0, 200.0, 300.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(left),
            Value::Tensor(right),
        )
        .expect("bsxfun nd trailing expansion");

        let Value::Tensor(tensor) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_eq!(tensor.data[0], 101.0);
        assert_eq!(tensor.data[1], 102.0);
        assert_eq!(tensor.data[2], 203.0);
        assert_eq!(tensor.data[6], 107.0);
        assert_eq!(tensor.data[8], 209.0);
        assert_eq!(tensor.data[23], 324.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_rejects_front_padded_nd_shapes() {
        let left =
            Tensor::new((1..=24).map(|value| value as f64).collect(), vec![2, 3, 4]).unwrap();
        let right = Tensor::new((1..=12).map(|value| value as f64).collect(), vec![3, 4]).unwrap();
        let err = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(left),
            Value::Tensor(right),
        )
        .expect_err("expected MATLAB trailing-dimension mismatch");
        assert_eq!(err.identifier(), BSXFUN_ERROR_SIZE_MISMATCH.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_gt_collects_logical_outputs() {
        let column = Tensor::new(vec![8.0, 17.0, 20.0, 24.0], vec![4, 1]).unwrap();
        let row = Tensor::new(vec![0.0, 10.0, 21.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("gt".to_string()),
            Value::Tensor(column),
            Value::Tensor(row),
        )
        .expect("bsxfun gt");

        let Value::LogicalArray(array) = result else {
            panic!("expected logical result, got {result:?}");
        };
        assert_eq!(array.shape, vec![4, 3]);
        assert_eq!(array.data, vec![1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_does_not_logicalize_shadowed_callback_names() {
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "gt").then_some(17)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, _args, _requested_outputs| Box::pin(async move { Ok(Value::Num(2.0)) }),
        )));
        let result = call(
            Value::FunctionHandle("gt".to_string()),
            Value::Num(10.0),
            Value::Num(1.0),
        )
        .expect("bsxfun shadowed gt");

        let Value::Tensor(tensor) = result else {
            panic!("expected numeric result, got {result:?}");
        };
        assert_eq!(tensor.shape, vec![1, 1]);
        assert_eq!(tensor.data, vec![2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_handles_complex_callback_results() {
        let left = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -1.0)], vec![2, 1]).unwrap();
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::ComplexTensor(left),
            Value::Num(10.0),
        )
        .expect("bsxfun complex");

        let Value::ComplexTensor(tensor) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(tensor.data, vec![(11.0, 2.0), (13.0, -1.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_invokes_bound_function_for_each_broadcasted_pair() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let args = args.to_vec();
                Box::pin(async move {
                    let a = match &args[0] {
                        Value::Num(value) => *value,
                        _ => 0.0,
                    };
                    let b = match &args[1] {
                        Value::Num(value) => *value,
                        _ => 0.0,
                    };
                    Ok(Value::Num(a + 2.0 * b))
                })
            },
        )));
        let column = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
        let result = call(
            Value::BoundFunctionHandle {
                name: "scaled_add".to_string(),
                function: 7,
            },
            Value::Tensor(column),
            Value::Tensor(row),
        )
        .expect("bsxfun closure");

        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.data, vec![21.0, 22.0, 41.0, 42.0, 61.0, 62.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_singleton_dimension_can_expand_to_zero() {
        let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(empty),
            Value::Tensor(row),
        )
        .expect("bsxfun empty");

        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![0, 3]);
        assert!(tensor.data.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_empty_logical_callback_preserves_logical_type() {
        let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("gt".to_string()),
            Value::Tensor(empty),
            Value::Tensor(row),
        )
        .expect("bsxfun empty logical");

        let Value::LogicalArray(array) = result else {
            panic!("expected logical result, got {result:?}");
        };
        assert_eq!(array.shape, vec![0, 3]);
        assert!(array.data.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_rejects_incompatible_sizes() {
        let left = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(left),
            Value::Tensor(right),
        )
        .expect_err("expected size mismatch");
        assert_eq!(err.identifier(), BSXFUN_ERROR_SIZE_MISMATCH.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_rejects_non_scalar_callback_output() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, _args, _requested_outputs| {
                Box::pin(async move {
                    Ok(Value::Tensor(
                        Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
                    ))
                })
            },
        )));
        let err = call(
            Value::BoundFunctionHandle {
                name: "bad".to_string(),
                function: 9,
            },
            Value::Num(1.0),
            Value::Num(2.0),
        )
        .expect_err("expected callback output error");
        assert_eq!(err.identifier(), BSXFUN_ERROR_FUNCTION_ERROR.identifier);
    }
}
