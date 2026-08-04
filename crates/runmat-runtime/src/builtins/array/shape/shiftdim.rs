//! MATLAB-compatible `shiftdim` with native-class and GPU-residency preservation.

use crate::builtins::array::shape::permute::{
    permute_char_array, permute_complex_tensor, permute_generic, permute_logical_array,
    permute_string_array, permute_tensor,
};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ComplexTensor, LiteralValue, LogicalArray, NumericScalar, ResolveContext,
    StringArray, Type, Value,
};
use runmat_macros::runtime_builtin;

const NAME: &str = "shiftdim";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::shiftdim")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("shiftdim"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("permute"), ProviderHook::Custom("reshape")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Positive shifts use the provider permutation hook; negative and leading-singleton shifts use metadata-only reshape.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::shiftdim")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "shiftdim is a shape operation and becomes a layout boundary only when a positive cyclic shift reorders data.",
};

const OUTPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array with dimensions shifted.",
};

const OUTPUT_M: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of leading singleton dimensions removed.",
};

const OUTPUTS_B: [BuiltinParamDescriptor; 1] = [OUTPUT_B];
const OUTPUTS_B_M: [BuiltinParamDescriptor; 2] = [OUTPUT_B, OUTPUT_M];

const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
};

const INPUT_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real integer dimension shift.",
};

const INPUTS_A: [BuiltinParamDescriptor; 1] = [INPUT_A];
const INPUTS_A_N: [BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_N];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "B = shiftdim(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, m] = shiftdim(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_B_M,
    },
    BuiltinSignatureDescriptor {
        label: "B = shiftdim(A, n)",
        inputs: &INPUTS_A_N,
        outputs: &OUTPUTS_B,
    },
];

const ERROR_INVALID_SHIFT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHIFTDIM.INVALID_SHIFT",
    identifier: Some("RunMat:shiftdim:InvalidShift"),
    when: "n is not a finite real integer scalar representable as a platform shift.",
    message: "shiftdim: n must be a finite real integer scalar",
};

const ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHIFTDIM.TOO_MANY_INPUTS",
    identifier: Some("RunMat:shiftdim:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "shiftdim: too many input arguments",
};

const ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHIFTDIM.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:shiftdim:TooManyOutputs"),
    when: "More than two outputs are requested, or m is requested when n is supplied.",
    message: "shiftdim: invalid number of output arguments",
};

const ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHIFTDIM.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:shiftdim:UnsupportedInput"),
    when: "A has a type or representation that cannot express the requested shifted shape.",
    message: "shiftdim: unsupported input type or shape",
};

const ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERROR_INVALID_SHIFT,
    ERROR_TOO_MANY_INPUTS,
    ERROR_TOO_MANY_OUTPUTS,
    ERROR_UNSUPPORTED_INPUT,
];

pub const SHIFTDIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn shiftdim_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn effective_shape(shape: &[usize]) -> Vec<usize> {
    let mut shape = match shape {
        [] => vec![1, 1],
        [length] => vec![1, *length],
        _ => shape.to_vec(),
    };
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn canonical_shape(mut shape: Vec<usize>) -> Vec<usize> {
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    if shape.is_empty() {
        vec![1, 1]
    } else if shape.len() == 1 {
        shape.push(1);
        shape
    } else {
        shape
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ShiftPlan {
    RemoveLeading {
        shape: Vec<usize>,
        removed: usize,
    },
    Prepend {
        shape: Vec<usize>,
    },
    Rotate {
        order: Vec<usize>,
        permuted_shape: Vec<usize>,
        shape: Vec<usize>,
    },
}

fn make_plan(shape: &[usize], shift: Option<isize>) -> Result<ShiftPlan, RuntimeError> {
    let shape = effective_shape(shape);
    match shift {
        None => {
            let removed = shape
                .iter()
                .position(|dimension| *dimension != 1)
                .unwrap_or(0);
            let output = if removed == 0 {
                shape
            } else {
                canonical_shape(shape[removed..].to_vec())
            };
            Ok(ShiftPlan::RemoveLeading {
                shape: output,
                removed,
            })
        }
        Some(n) if n < 0 => {
            let count = n
                .checked_abs()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    shiftdim_error(
                        &ERROR_INVALID_SHIFT,
                        "shiftdim: negative n is too large to represent",
                    )
                })?;
            let capacity = shape.len().checked_add(count).ok_or_else(|| {
                shiftdim_error(
                    &ERROR_INVALID_SHIFT,
                    "shiftdim: shifted rank exceeds platform limits",
                )
            })?;
            let mut output = Vec::new();
            output.try_reserve_exact(capacity).map_err(|_| {
                shiftdim_error(
                    &ERROR_INVALID_SHIFT,
                    "shiftdim: shifted rank exceeds available memory",
                )
            })?;
            output.resize(count, 1);
            output.extend(shape);
            Ok(ShiftPlan::Prepend { shape: output })
        }
        Some(n) => {
            let amount = usize::try_from(n)
                .map_err(|_| shiftdim_error(&ERROR_INVALID_SHIFT, ERROR_INVALID_SHIFT.message))?
                % shape.len();
            let order: Vec<usize> = (0..shape.len())
                .map(|index| (index + amount) % shape.len() + 1)
                .collect();
            let permuted_shape: Vec<usize> = order
                .iter()
                .map(|dimension| shape[*dimension - 1])
                .collect();
            let output = canonical_shape(permuted_shape.clone());
            Ok(ShiftPlan::Rotate {
                order,
                permuted_shape,
                shape: output,
            })
        }
    }
}

fn shifted_type(args: &[Type], context: &ResolveContext) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    let shift = if args.len() == 1 {
        Some(None)
    } else {
        match context.literal_args.get(1) {
            Some(LiteralValue::Number(value))
                if value.is_finite()
                    && value.fract() == 0.0
                    && *value >= isize::MIN as f64
                    && *value < 9_223_372_036_854_775_808.0 =>
            {
                Some(Some(*value as isize))
            }
            _ => None,
        }
    };
    let map_shape = |shape: &[Option<usize>]| -> Option<Vec<Option<usize>>> {
        let concrete: Option<Vec<usize>> = shape.iter().copied().collect();
        let concrete = concrete?;
        let plan = make_plan(&concrete, shift?).ok()?;
        Some(
            match plan {
                ShiftPlan::RemoveLeading { shape, .. }
                | ShiftPlan::Prepend { shape }
                | ShiftPlan::Rotate { shape, .. } => shape,
            }
            .into_iter()
            .map(Some)
            .collect(),
        )
    };
    match input {
        Type::Tensor { shape } => Type::Tensor {
            shape: shape.as_deref().and_then(map_shape),
        },
        Type::Logical { shape } => Type::Logical {
            shape: shape.as_deref().and_then(map_shape),
        },
        Type::Num | Type::Int | Type::Bool => input.clone(),
        Type::Cell {
            element_type,
            length,
        } => Type::Cell {
            element_type: element_type.clone(),
            length: *length,
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runtime_builtin(
    name = "shiftdim",
    category = "array/shape",
    summary = "Shift array dimensions while preserving native class and GPU residency.",
    keywords = "shiftdim,dimensions,singleton,permute,gpu",
    accel = "custom",
    type_resolver(shifted_type),
    descriptor(crate::builtins::array::shape::shiftdim::SHIFTDIM_DESCRIPTOR),
    builtin_path = "crate::builtins::array::shape::shiftdim"
)]
async fn shiftdim_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(shiftdim_error(
            &ERROR_TOO_MANY_INPUTS,
            ERROR_TOO_MANY_INPUTS.message,
        ));
    }
    let explicit_shift = if let Some(value) = rest.first() {
        Some(parse_shift(value)?)
    } else {
        None
    };
    let requested = crate::output_count::current_output_count();
    if matches!(requested, Some(count) if count > 2)
        || (explicit_shift.is_some() && matches!(requested, Some(count) if count > 1))
    {
        return Err(shiftdim_error(
            &ERROR_TOO_MANY_OUTPUTS,
            ERROR_TOO_MANY_OUTPUTS.message,
        ));
    }
    let shape = value_shape(&value)?;
    let plan = make_plan(&shape, explicit_shift)?;
    let removed = match &plan {
        ShiftPlan::RemoveLeading { removed, .. } => *removed,
        _ => 0,
    };
    let shifted = apply_plan(value, &plan).await?;
    match requested {
        None => Ok(shifted),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![shifted])),
        Some(2) => Ok(Value::OutputList(vec![shifted, Value::Num(removed as f64)])),
        Some(_) => unreachable!("output count validated above"),
    }
}

fn parse_shift(value: &Value) -> Result<isize, RuntimeError> {
    let scalar = match value {
        Value::Int(value) => {
            return value.try_to_isize().ok_or_else(|| {
                shiftdim_error(
                    &ERROR_INVALID_SHIFT,
                    "shiftdim: n is outside the supported signed shift range",
                )
            })
        }
        Value::Num(value) => NumericScalar::F64(*value),
        Value::Bool(value) => NumericScalar::U8(u8::from(*value)),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .numeric_value_at(0)
            .ok_or_else(|| shiftdim_error(&ERROR_INVALID_SHIFT, ERROR_INVALID_SHIFT.message))?,
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            NumericScalar::U8(logical.data[0])
        }
        _ => {
            return Err(shiftdim_error(
                &ERROR_INVALID_SHIFT,
                ERROR_INVALID_SHIFT.message,
            ))
        }
    };
    numeric_scalar_to_isize(scalar)
        .ok_or_else(|| shiftdim_error(&ERROR_INVALID_SHIFT, ERROR_INVALID_SHIFT.message))
}

fn numeric_scalar_to_isize(value: NumericScalar) -> Option<isize> {
    match value {
        NumericScalar::F64(value) => floating_to_isize(value),
        NumericScalar::F32(value) => floating_to_isize(f64::from(value)),
        other => other.into_int_value()?.try_to_isize(),
    }
}

fn floating_to_isize(value: f64) -> Option<isize> {
    if !value.is_finite()
        || value.fract() != 0.0
        || value < isize::MIN as f64
        || value >= 9_223_372_036_854_775_808.0
    {
        None
    } else {
        Some(value as isize)
    }
}

fn value_shape(value: &Value) -> Result<Vec<usize>, RuntimeError> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.shape.clone()),
        Value::ComplexTensor(tensor) => Ok(tensor.shape.clone()),
        Value::LogicalArray(array) => Ok(array.shape.clone()),
        Value::StringArray(array) => Ok(array.shape.clone()),
        Value::CharArray(array) => Ok(vec![array.rows, array.cols]),
        Value::Cell(array) => Ok(array.shape.clone()),
        Value::GpuTensor(handle) => Ok(handle.shape.clone()),
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::String(_)
        | Value::Struct(_) => Ok(vec![1, 1]),
        Value::SparseTensor(array) => Ok(vec![array.rows, array.cols]),
        other => Err(shiftdim_error(
            &ERROR_UNSUPPORTED_INPUT,
            format!("shiftdim: unsupported input type {other:?}"),
        )),
    }
}

async fn apply_plan(value: Value, plan: &ShiftPlan) -> crate::BuiltinResult<Value> {
    match plan {
        ShiftPlan::RemoveLeading { shape, .. } | ShiftPlan::Prepend { shape } => {
            reshape_value(value, shape)
        }
        ShiftPlan::Rotate {
            order,
            permuted_shape,
            shape,
        } => {
            let raw_shape = value_shape(&value)?;
            let source_shape = effective_shape(&raw_shape);
            let value = if raw_shape == source_shape {
                value
            } else {
                reshape_value(value, &source_shape)?
            };
            let value = permute_value(value, order).await?;
            if shape == permuted_shape {
                Ok(value)
            } else {
                reshape_value(value, shape)
            }
        }
    }
}

async fn permute_value(value: Value, order: &[usize]) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => permute_tensor(NAME, tensor, order).map(Value::Tensor),
        Value::ComplexTensor(tensor) => {
            permute_complex_tensor(NAME, tensor, order).map(Value::ComplexTensor)
        }
        Value::LogicalArray(array) => {
            permute_logical_array(NAME, array, order).map(Value::LogicalArray)
        }
        Value::StringArray(array) => {
            permute_string_array(NAME, array, order).map(Value::StringArray)
        }
        Value::CharArray(array) => permute_char_array(NAME, array, order).map(Value::CharArray),
        Value::Cell(array) => {
            let (data, shape) = permute_generic(NAME, &array.data, &array.shape, order)?;
            CellArray::new_with_shape(data, shape)
                .map(Value::Cell)
                .map_err(|error| shiftdim_error(&ERROR_UNSUPPORTED_INPUT, error))
        }
        Value::GpuTensor(handle) => permute_gpu(handle, order),
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::String(_)
        | Value::Struct(_) => Ok(value),
        Value::SparseTensor(_) => Err(shiftdim_error(
            &ERROR_UNSUPPORTED_INPUT,
            "shiftdim: sparse arrays currently support only unshifted 2-D shapes",
        )),
        other => Err(shiftdim_error(
            &ERROR_UNSUPPORTED_INPUT,
            format!("shiftdim: unsupported input type {other:?}"),
        )),
    }
}

fn reshape_value(value: Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => tensor
            .reshape(shape.to_vec())
            .map(Value::Tensor)
            .map_err(|error| shiftdim_error(&ERROR_UNSUPPORTED_INPUT, error)),
        Value::ComplexTensor(tensor) => {
            ComplexTensor::from_complex_storage(tensor.into_complex_storage(), shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|error| shiftdim_error(&ERROR_UNSUPPORTED_INPUT, error))
        }
        Value::LogicalArray(array) => LogicalArray::new(array.data, shape.to_vec())
            .map(Value::LogicalArray)
            .map_err(|error| shiftdim_error(&ERROR_UNSUPPORTED_INPUT, error)),
        Value::StringArray(array) => StringArray::new(array.data, shape.to_vec())
            .map(Value::StringArray)
            .map_err(|error| shiftdim_error(&ERROR_UNSUPPORTED_INPUT, error)),
        Value::Cell(array) => CellArray::new_with_shape(array.data, shape.to_vec())
            .map(Value::Cell)
            .map_err(|error| shiftdim_error(&ERROR_UNSUPPORTED_INPUT, error)),
        Value::CharArray(array) => reshape_char(array, shape).map(Value::CharArray),
        Value::GpuTensor(handle) => reshape_gpu(handle, shape),
        Value::SparseTensor(array) if shape == [array.rows, array.cols] => {
            Ok(Value::SparseTensor(array))
        }
        Value::SparseTensor(_) => Err(shiftdim_error(
            &ERROR_UNSUPPORTED_INPUT,
            "shiftdim: sparse arrays currently support only their existing 2-D shape",
        )),
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::String(_)
        | Value::Struct(_)
            if shape.iter().all(|dimension| *dimension == 1) =>
        {
            Ok(value)
        }
        other => Err(shiftdim_error(
            &ERROR_UNSUPPORTED_INPUT,
            format!("shiftdim: input representation cannot express shape {shape:?}: {other:?}"),
        )),
    }
}

fn reshape_char(array: CharArray, shape: &[usize]) -> crate::BuiltinResult<CharArray> {
    if shape.len() != 2 {
        return Err(shiftdim_error(
            &ERROR_UNSUPPORTED_INPUT,
            "shiftdim: char arrays currently support at most two dimensions",
        ));
    }
    CharArray::new(array.data, shape[0], shape[1])
        .map_err(|error| shiftdim_error(&ERROR_UNSUPPORTED_INPUT, error))
}

fn wrap_gpu(handle: GpuTensorHandle, complex: bool, logical: bool) -> Value {
    if complex {
        gpu_helpers::complex_gpu_value(handle)
    } else if logical {
        gpu_helpers::logical_gpu_value(handle)
    } else {
        gpu_helpers::resident_gpu_value(handle)
    }
}

fn permute_gpu(handle: GpuTensorHandle, order: &[usize]) -> crate::BuiltinResult<Value> {
    let complex =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;
    if complex && runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        return Err(shiftdim_error(
            &ERROR_UNSUPPORTED_INPUT,
            "shiftdim: positive shifts of complex integer GPU arrays require typed complex device storage support",
        ));
    }
    let logical = runmat_accelerate_api::handle_is_logical(&handle);
    let provider = runmat_accelerate_api::provider_for_handle(&handle)
        .or_else(runmat_accelerate_api::provider)
        .ok_or_else(|| {
            shiftdim_error(
                &ERROR_UNSUPPORTED_INPUT,
                "shiftdim: no acceleration provider owns the GPU input",
            )
        })?;
    let order: Vec<usize> = order.iter().map(|dimension| dimension - 1).collect();
    let output = provider.permute(&handle, &order).map_err(|error| {
        shiftdim_error(
            &ERROR_UNSUPPORTED_INPUT,
            format!("shiftdim: GPU permutation failed: {error}"),
        )
    })?;
    Ok(wrap_gpu(output, complex, logical))
}

fn reshape_gpu(handle: GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    let complex =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;
    let logical = runmat_accelerate_api::handle_is_logical(&handle);
    if complex && runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let mut output = handle;
        output.shape = shape.to_vec();
        return Ok(wrap_gpu(output, true, false));
    }
    let output = if let Some(provider) =
        runmat_accelerate_api::provider_for_handle(&handle).or_else(runmat_accelerate_api::provider)
    {
        provider.reshape(&handle, shape).map_err(|error| {
            shiftdim_error(
                &ERROR_UNSUPPORTED_INPUT,
                format!("shiftdim: GPU reshape failed: {error}"),
            )
        })?
    } else {
        let mut handle = handle;
        handle.shape = shape.to_vec();
        handle
    };
    Ok(wrap_gpu(output, complex, logical))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        ComplexStorage, IntValue, IntegerComplexStorage, IntegerStorage, NumericDType, Tensor,
    };

    fn call(value: Value, shift: Option<Value>) -> crate::BuiltinResult<Value> {
        block_on(shiftdim_builtin(value, shift.into_iter().collect()))
    }

    fn tensor_shape(value: &Value) -> &[usize] {
        match value {
            Value::Tensor(tensor) => &tensor.shape,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn plan_matches_positive_negative_and_default_semantics() {
        assert_eq!(
            make_plan(&[4, 2, 3, 5], Some(2)).unwrap(),
            ShiftPlan::Rotate {
                order: vec![3, 4, 1, 2],
                permuted_shape: vec![3, 5, 4, 2],
                shape: vec![3, 5, 4, 2],
            }
        );
        assert_eq!(
            make_plan(&[4, 2, 3, 5], Some(-2)).unwrap(),
            ShiftPlan::Prepend {
                shape: vec![1, 1, 4, 2, 3, 5],
            }
        );
        assert_eq!(
            make_plan(&[1, 1, 3, 2, 4], None).unwrap(),
            ShiftPlan::RemoveLeading {
                shape: vec![3, 2, 4],
                removed: 2,
            }
        );
        assert_eq!(
            make_plan(&[1, 7], None).unwrap(),
            ShiftPlan::RemoveLeading {
                shape: vec![7, 1],
                removed: 1,
            }
        );
        assert_eq!(
            make_plan(&[1, 1, 1], None).unwrap(),
            ShiftPlan::RemoveLeading {
                shape: vec![1, 1],
                removed: 0,
            }
        );
        assert_eq!(
            make_plan(&[7], Some(1)).unwrap(),
            ShiftPlan::Rotate {
                order: vec![2, 1],
                permuted_shape: vec![7, 1],
                shape: vec![7, 1],
            }
        );
    }

    #[test]
    fn positive_shift_reorders_column_major_data_and_wraps_large_n() {
        let input = Tensor::new((1..=24).map(f64::from).collect(), vec![2, 3, 4]).unwrap();
        let output = call(Value::Tensor(input), Some(Value::Num(4.0))).unwrap();
        let Value::Tensor(output) = output else {
            panic!("expected tensor")
        };
        assert_eq!(output.shape, vec![3, 4, 2]);
        assert_eq!(
            output.materialize_f64(),
            vec![
                1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 2.0, 4.0, 6.0,
                8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
            ]
        );
    }

    #[test]
    fn negative_shift_and_default_shift_only_change_shape_metadata() {
        let input = Tensor::from_numeric_storage(
            runmat_builtins::NumericStorage::F32(vec![1.0, 2.0, 3.0, 4.0]),
            vec![1, 1, 2, 2],
        )
        .unwrap();
        let output = call(Value::Tensor(input), None).unwrap();
        assert_eq!(tensor_shape(&output), &[2, 2]);
        let Value::Tensor(output) = output else {
            unreachable!()
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        let output = call(Value::Tensor(output), Some(Value::Int(IntValue::I8(-3)))).unwrap();
        assert_eq!(tensor_shape(&output), &[1, 1, 1, 2, 2]);
    }

    #[test]
    fn empty_dimensions_shift_without_synthesizing_elements() {
        let input = Tensor::new(Vec::new(), vec![1, 0, 2]).unwrap();
        let Value::Tensor(output) =
            call(Value::Tensor(input.clone()), Some(Value::Num(1.0))).unwrap()
        else {
            panic!("expected tensor")
        };
        assert_eq!(output.shape, vec![0, 2]);
        assert!(output.is_empty());

        let Value::Tensor(output) = call(Value::Tensor(input), Some(Value::Num(-2.0))).unwrap()
        else {
            panic!("expected tensor")
        };
        assert_eq!(output.shape, vec![1, 1, 1, 0, 2]);
        assert!(output.is_empty());
    }

    macro_rules! integer_case {
        ($variant:ident, $ty:ty) => {{
            let storage = IntegerStorage::$variant(vec![1 as $ty, 2 as $ty, 3 as $ty, 4 as $ty]);
            let class_name = storage.class_name();
            let input = Tensor::new_integer(storage, vec![1, 2, 2]).unwrap();
            let output = call(Value::Tensor(input), Some(Value::Num(1.0))).unwrap();
            let Value::Tensor(output) = output else {
                panic!("expected tensor")
            };
            assert_eq!(output.shape, vec![2, 2]);
            assert_eq!(output.integer_storage().unwrap().class_name(), class_name);
        }};
    }

    #[test]
    fn all_eight_integer_classes_preserve_native_storage() {
        integer_case!(I8, i8);
        integer_case!(I16, i16);
        integer_case!(I32, i32);
        integer_case!(I64, i64);
        integer_case!(U8, u8);
        integer_case!(U16, u16);
        integer_case!(U32, u32);
        integer_case!(U64, u64);
    }

    #[test]
    fn uint64_and_complex_integer_values_remain_exact() {
        let wide = vec![9_007_199_254_740_993, u64::MAX, 1, 2];
        let input = Tensor::new_integer(IntegerStorage::U64(wide.clone()), vec![1, 2, 2]).unwrap();
        let Value::Tensor(output) =
            call(Value::Tensor(input), Some(Value::Int(IntValue::U8(1)))).unwrap()
        else {
            panic!("expected tensor")
        };
        assert_eq!(output.integer_storage(), Some(&IntegerStorage::U64(wide)));

        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
        )
        .unwrap();
        let input = ComplexTensor::new_integer(storage.clone(), vec![1, 2, 1]).unwrap();
        let Value::ComplexTensor(output) =
            call(Value::ComplexTensor(input), Some(Value::Num(1.0))).unwrap()
        else {
            panic!("expected complex tensor")
        };
        assert_eq!(output.integer_storage(), Some(&storage));
        assert_eq!(output.shape, vec![2, 1]);
    }

    #[test]
    fn floating_complex_single_preserves_component_class() {
        let input = ComplexTensor::from_f32(vec![(1.0, -1.0), (2.0, -2.0)], vec![1, 2, 1]).unwrap();
        let Value::ComplexTensor(output) =
            call(Value::ComplexTensor(input), Some(Value::Num(1.0))).unwrap()
        else {
            panic!("expected complex tensor")
        };
        assert!(matches!(output.complex_storage(), ComplexStorage::F32(_)));
        assert_eq!(output.shape, vec![2, 1]);
    }

    #[test]
    fn two_output_form_reports_removed_leading_dimensions() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 1, 3]).unwrap();
        let output = call(Value::Tensor(input), None).unwrap();
        let Value::OutputList(values) = output else {
            panic!("expected output list")
        };
        assert_eq!(values.len(), 2);
        assert_eq!(tensor_shape(&values[0]), &[3, 1]);
        assert_eq!(values[1], Value::Num(2.0));
    }

    #[test]
    fn output_and_shift_validation_are_stable() {
        for invalid in [
            Value::Num(1.5),
            Value::Num(f64::NAN),
            Value::Num(f64::INFINITY),
            Value::Int(IntValue::U64(u64::MAX)),
            Value::String("1".into()),
            Value::Complex(1.0, 0.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ] {
            let error = call(Value::Num(1.0), Some(invalid)).unwrap_err();
            assert_eq!(
                error.identifier.as_deref(),
                Some("RunMat:shiftdim:InvalidShift")
            );
        }
        let _guard = crate::output_count::push_output_count(Some(2));
        let error = call(Value::Num(1.0), Some(Value::Num(1.0))).unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:shiftdim:TooManyOutputs")
        );
    }

    #[test]
    fn logical_string_cell_and_char_arrays_follow_shape_contract() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![1, 2, 2]).unwrap();
        let Value::LogicalArray(logical) =
            call(Value::LogicalArray(logical), Some(Value::Num(1.0))).unwrap()
        else {
            panic!("expected logical")
        };
        assert_eq!(logical.shape, vec![2, 2]);

        let strings = StringArray::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![1, 2, 2],
        )
        .unwrap();
        let Value::StringArray(strings) =
            call(Value::StringArray(strings), Some(Value::Num(1.0))).unwrap()
        else {
            panic!("expected strings")
        };
        assert_eq!(strings.shape, vec![2, 2]);

        let cell = CellArray::new_with_shape(
            vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
            ],
            vec![1, 2, 2],
        )
        .unwrap();
        let Value::Cell(cell) = call(Value::Cell(cell), Some(Value::Num(1.0))).unwrap() else {
            panic!("expected cell")
        };
        assert_eq!(cell.shape, vec![2, 2]);

        let chars = CharArray::new_row("abc");
        let Value::CharArray(chars) = call(Value::CharArray(chars), None).unwrap() else {
            panic!("expected chars")
        };
        assert_eq!((chars.rows, chars.cols), (3, 1));
        let error = call(
            Value::CharArray(CharArray::new_row("abc")),
            Some(Value::Num(-1.0)),
        )
        .unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:shiftdim:UnsupportedInput")
        );
    }

    #[test]
    fn type_resolver_tracks_literal_shifted_shapes() {
        let input = Type::Tensor {
            shape: Some(vec![Some(1), Some(2), Some(3)]),
        };
        assert_eq!(
            shifted_type(
                std::slice::from_ref(&input),
                &ResolveContext::new(vec![LiteralValue::Unknown])
            ),
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
        assert_eq!(
            shifted_type(
                &[input, Type::Num],
                &ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(-2.0)])
            ),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1), Some(1), Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn descriptor_exposes_both_output_forms_and_gpu_hooks() {
        assert_eq!(SHIFTDIM_DESCRIPTOR.signatures.len(), 3);
        assert_eq!(
            SHIFTDIM_DESCRIPTOR.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );
        assert_eq!(GPU_SPEC.provider_hooks.len(), 2);
    }

    #[test]
    fn simple_provider_keeps_integer_gpu_values_resident_and_exact() {
        runmat_accelerate::initialize_acceleration_provider();
        runmat_accelerate::simple_provider::register_inprocess_provider();
        let provider = runmat_accelerate_api::provider().expect("provider");
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX, 3, 4, 5, 6]),
            vec![1, 2, 3],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let Value::GpuTensor(output) =
            call(Value::GpuTensor(handle), Some(Value::Num(1.0))).unwrap()
        else {
            panic!("expected GPU tensor")
        };
        assert_eq!(output.shape, vec![2, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        assert!(runmat_accelerate::fusion_residency::is_resident(&output));
        let gathered = block_on(gpu_helpers::gather_tensor_async(&output)).unwrap();
        assert_eq!(gathered.integer_storage(), tensor.integer_storage());

        let Value::GpuTensor(output) =
            call(Value::GpuTensor(output), Some(Value::Num(-2.0))).unwrap()
        else {
            panic!("expected GPU tensor")
        };
        assert_eq!(output.shape, vec![1, 1, 2, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        assert!(runmat_accelerate::fusion_residency::is_resident(&output));
    }

    #[test]
    fn simple_provider_keeps_complex_gpu_values_resident() {
        runmat_accelerate::initialize_acceleration_provider();
        runmat_accelerate::simple_provider::register_inprocess_provider();
        let provider = runmat_accelerate_api::provider().expect("provider");
        let tensor = ComplexTensor::new(
            vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
            vec![1, 2, 2],
        )
        .unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).unwrap();
        let Value::GpuTensor(output) =
            call(Value::GpuTensor(handle), Some(Value::Num(1.0))).unwrap()
        else {
            panic!("expected GPU tensor")
        };
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            GpuTensorStorage::ComplexInterleaved
        );
        assert!(runmat_accelerate::fusion_residency::is_resident(&output));
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output))).unwrap();
        assert!(matches!(
            gathered,
            Value::ComplexTensor(value)
                if value.shape == vec![2, 2]
                    && value.materialize_f64() == tensor.materialize_f64()
        ));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_positive_integer_shift_preserves_class_when_adapter_is_available() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor =
            Tensor::new_integer(IntegerStorage::I32(vec![1, 2, 3, 4]), vec![1, 2, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let Value::GpuTensor(output) =
            call(Value::GpuTensor(handle), Some(Value::Num(1.0))).unwrap()
        else {
            panic!("expected GPU tensor")
        };
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::I32)
        );
        let Value::GpuTensor(output) =
            call(Value::GpuTensor(output), Some(Value::Num(-1.0))).unwrap()
        else {
            panic!("expected GPU tensor")
        };
        assert_eq!(output.shape, vec![1, 2, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::I32)
        );
    }
}
