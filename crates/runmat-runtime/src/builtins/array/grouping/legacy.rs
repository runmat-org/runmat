//! MATLAB-compatible array grouping, binning, and grouped-apply builtins.

use std::collections::BTreeMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, IntValue, IntegerStorage, LogicalArray, NumericDType, SparseTensor, Tensor, Value,
};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::table::select_rows;
use crate::{
    build_runtime_error, call_feval_async_with_outputs, gather_if_needed_async, BuiltinResult,
    RuntimeError,
};

const MAX_MATERIALIZED_ELEMENTS: usize = 50_000_000;

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROUPING.INVALID_INPUT",
    identifier: Some("RunMat:grouping:InvalidInput"),
    when: "Inputs are malformed, have incompatible lengths, or request unsupported grouped output.",
    message: "grouping builtin: invalid input",
};

const ERROR_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROUPING.CALLBACK",
    identifier: Some("RunMat:grouping:CallbackFailed"),
    when: "A grouped callback fails or returns incompatible outputs.",
    message: "grouping builtin: callback failed",
};

const ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROUPING.TOO_LARGE",
    identifier: Some("RunMat:grouping:TooLarge"),
    when: "The requested dense or combinatorial output exceeds RunMat's materialization limit.",
    message: "grouping builtin: output is too large",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INVALID_INPUT, ERROR_CALLBACK, ERROR_TOO_LARGE];

const ACCUMARRAY_IND_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ind",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive one-based output subscripts.",
};
const ACCUMARRAY_DATA_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar or vector data to accumulate.",
};
const ACCUMARRAY_SIZE_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("[]"),
    description: "Positive output-size vector or [].",
};
const ACCUMARRAY_FUN_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("@sum"),
    description: "Scalar-returning group function or [].",
};
const ACCUMARRAY_FILL_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fillval",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("0"),
    description: "Scalar fill matching the group-function output class or [].",
};
const ACCUMARRAY_SPARSE_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "issparse",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("false"),
    description: "Logical or numeric scalar 0 or 1 selecting sparse output.",
};
const ACCUMARRAY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Accumulated full or sparse array.",
}];
const ACCUMARRAY_INPUTS_2: [BuiltinParamDescriptor; 2] =
    [ACCUMARRAY_IND_PARAM, ACCUMARRAY_DATA_PARAM];
const ACCUMARRAY_INPUTS_3: [BuiltinParamDescriptor; 3] = [
    ACCUMARRAY_IND_PARAM,
    ACCUMARRAY_DATA_PARAM,
    ACCUMARRAY_SIZE_PARAM,
];
const ACCUMARRAY_INPUTS_4: [BuiltinParamDescriptor; 4] = [
    ACCUMARRAY_IND_PARAM,
    ACCUMARRAY_DATA_PARAM,
    ACCUMARRAY_SIZE_PARAM,
    ACCUMARRAY_FUN_PARAM,
];
const ACCUMARRAY_INPUTS_5: [BuiltinParamDescriptor; 5] = [
    ACCUMARRAY_IND_PARAM,
    ACCUMARRAY_DATA_PARAM,
    ACCUMARRAY_SIZE_PARAM,
    ACCUMARRAY_FUN_PARAM,
    ACCUMARRAY_FILL_PARAM,
];
const ACCUMARRAY_INPUTS_6: [BuiltinParamDescriptor; 6] = [
    ACCUMARRAY_IND_PARAM,
    ACCUMARRAY_DATA_PARAM,
    ACCUMARRAY_SIZE_PARAM,
    ACCUMARRAY_FUN_PARAM,
    ACCUMARRAY_FILL_PARAM,
    ACCUMARRAY_SPARSE_PARAM,
];
const ACCUMARRAY_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data)",
        inputs: &ACCUMARRAY_INPUTS_2,
        outputs: &ACCUMARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data, sz)",
        inputs: &ACCUMARRAY_INPUTS_3,
        outputs: &ACCUMARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data, sz, fun)",
        inputs: &ACCUMARRAY_INPUTS_4,
        outputs: &ACCUMARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data, sz, fun, fillval)",
        inputs: &ACCUMARRAY_INPUTS_5,
        outputs: &ACCUMARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind, data, sz, fun, fillval, issparse)",
        inputs: &ACCUMARRAY_INPUTS_6,
        outputs: &ACCUMARRAY_OUTPUT,
    },
];
pub const ACCUMARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ACCUMARRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ACCUMARRAY_STRUCTURAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "ind",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Positive one-based subscripts accept every integer class or exactly integral floating values and are parsed without an f64 intermediary.",
    },
    BuiltinIntegerInputCapability {
        name: "sz",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Explicit positive output dimensions accept every integer class or exactly integral floating values within platform and materialization bounds.",
    },
];

const ACCUMARRAY_DEFAULT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "data",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Scalar or vector data accepts every integer class; scalar data expands in its native class before groups are formed.",
    },
    BuiltinIntegerInputCapability {
        name: "fillval",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The default integer-data sum returns double, so its fill value must also be double rather than a typed integer.",
    },
];

const ACCUMARRAY_CUSTOM_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "data",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Scalar or vector data accepts every integer class and reaches the group computation as a native-class column vector.",
    },
    BuiltinIntegerInputCapability {
        name: "fillval",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer fill value is valid when it has the same class as the scalar returned by the group computation.",
    },
];

const ACCUMARRAY_SPARSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "data",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Sparse accumarray output requires double input data; typed-integer data is rejected before accumulation.",
    },
    BuiltinIntegerInputCapability {
        name: "fillval",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Sparse accumarray output requires a double zero or omitted fill; typed-integer fills are rejected.",
    },
    BuiltinIntegerInputCapability {
        name: "issparse",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The sparse selector accepts logical or numeric scalar 0 or 1 in every integer class and rejects every other numeric value.",
    },
];

pub const ACCUMARRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = accumarray(integer_ind, data, integer_sz)",
        inputs: &ACCUMARRAY_STRUCTURAL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Subscripts and explicit sizes remain exact through bounds checks and column-major linearization; oversized or unmaterializable results reject deterministically.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = accumarray(ind, integer_data, sz, [])",
        inputs: &ACCUMARRAY_DEFAULT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "The default sum follows sum's default integer rule and returns double; interactive GPU accumarray does not admit integer data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = accumarray(ind, integer_data, sz, fun, integer_fillval)",
        inputs: &ACCUMARRAY_CUSTOM_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Each group reaches fun as a native-class column vector; B has the class returned by fun, and any integer fill must match that class exactly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = accumarray(ind, integer_data, sz, fun, integer_fillval, true)",
        inputs: &ACCUMARRAY_SPARSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The documented sparse form is double-only for both data and group results and permits only an omitted or double-zero fill.",
    },
];

fn grouping_error(message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("array_grouping");
    if let Some(identifier) = ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn callback_error(message: impl Into<String>, source: Option<RuntimeError>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("array_grouping");
    if let Some(identifier) = ERROR_CALLBACK.identifier {
        builder = builder.with_identifier(identifier);
    }
    if let Some(source) = source {
        builder = builder.with_source(source);
    }
    builder.build()
}

fn too_large_error(message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("array_grouping");
    if let Some(identifier) = ERROR_TOO_LARGE.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "accumarray",
    execution_stack = "process",
    category = "array/grouping",
    summary = "Accumulate values into an array by subscript groups.",
    keywords = "accumarray,accumulate,groups,sum,sparse",
    accel = "cpu",
    descriptor(crate::builtins::array::grouping::ACCUMARRAY_DESCRIPTOR),
    integer_capabilities(crate::builtins::array::grouping::ACCUMARRAY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::grouping"
)]
pub(crate) async fn accumarray_builtin(
    subs: Value,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if matches!(&data, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
    {
        return Err(grouping_error(
            "accumarray: GPU input data must be logical, single, or double",
        ));
    }
    if matches!(
        rest.get(2),
        Some(Value::GpuTensor(handle))
            if runmat_accelerate_api::handle_integer_type(handle).is_some()
    ) {
        return Err(grouping_error(
            "accumarray: GPU fill value must be logical, single, or double",
        ));
    }
    let subs = gather_if_needed_async(&subs).await?;
    let data = gather_if_needed_async(&data).await?;
    let rest = gather_values(rest).await?;
    accumarray_impl(subs, data, rest).await
}

async fn accumarray_impl(subs: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let index_rows = accumarray_subscripts(subs)?;
    let rows = index_rows.len();
    let data = accumarray_data_column(data, rows)?;
    let output_shape = if let Some(size_value) = rest.first() {
        if is_empty_value(size_value) {
            infer_accumarray_shape(&index_rows)
        } else {
            parse_positive_size_vector(size_value, "accumarray")?
        }
    } else {
        infer_accumarray_shape(&index_rows)
    };
    let fun = rest.get(1).filter(|value| !is_empty_value(value)).cloned();
    let fill = rest.get(2).filter(|value| !is_empty_value(value)).cloned();
    let issparse = rest
        .get(3)
        .map(|value| binary_bool_scalar(value, "accumarray issparse"))
        .transpose()?
        .unwrap_or(false);
    let output_len = checked_element_count(&output_shape, "accumarray")?;
    if output_len > MAX_MATERIALIZED_ELEMENTS {
        return Err(too_large_error("accumarray: output is too large"));
    }
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, subs) in index_rows.iter().enumerate() {
        let lin = subscript_to_linear(subs, &output_shape)?;
        buckets.entry(lin).or_default().push(idx);
    }
    if let Some(fun) = fun {
        return accumarray_callback_output(
            data,
            buckets,
            output_shape,
            output_len,
            fun,
            fill,
            issparse,
        )
        .await;
    }
    if issparse && !is_double_numeric_value(&data) {
        return Err(grouping_error(
            "accumarray: sparse output requires double input data",
        ));
    }
    if issparse && fill.as_ref().is_some_and(is_integer_numeric_value) {
        return Err(grouping_error(
            "accumarray: sparse output requires a double zero fill value",
        ));
    }
    if fill.as_ref().is_some_and(is_integer_numeric_value) {
        return Err(grouping_error(
            "accumarray: fill value class must match the double default sum output",
        ));
    }
    let data_values = accumarray_data_values(data, rows)?;
    let fill_num = match fill.as_ref() {
        None => 0.0,
        Some(value) => numeric_scalar(value, "accumarray fill value")?,
    };
    if issparse && fill_num != 0.0 {
        return Err(grouping_error(
            "accumarray: sparse output requires a zero fill value",
        ));
    }
    let mut data_out = vec![fill_num; output_len];
    for (lin, indices) in buckets {
        data_out[lin] = indices.iter().map(|index| data_values[*index]).sum();
    }
    accumarray_numeric_output(data_out, output_shape, issparse)
}

fn accumarray_data_column(data: Value, rows: usize) -> BuiltinResult<Value> {
    match data {
        Value::Num(value) => Tensor::new(vec![value; rows], vec![rows, 1])
            .map(Value::Tensor)
            .map_err(grouping_error),
        Value::Int(value) => {
            let prototype = IntegerStorage::from_scalar(value);
            Tensor::new_integer(repeat_integer_scalar(&prototype, rows)?, vec![rows, 1])
                .map(Value::Tensor)
                .map_err(grouping_error)
        }
        Value::Bool(value) => LogicalArray::new(vec![u8::from(value); rows], vec![rows, 1])
            .map(Value::LogicalArray)
            .map_err(grouping_error),
        Value::Tensor(tensor) => {
            let len = tensor_utils::tensor_element_len(&tensor);
            if len != 1 && len != rows {
                return Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ));
            }
            if len == rows {
                return tensor
                    .reshape(vec![rows, 1])
                    .map(Value::Tensor)
                    .map_err(grouping_error);
            }
            if let Some(storage) = tensor.integer_storage() {
                return Tensor::new_integer(repeat_integer_scalar(storage, rows)?, vec![rows, 1])
                    .map(Value::Tensor)
                    .map_err(grouping_error);
            }
            Tensor::new_with_dtype(
                vec![tensor_utils::tensor_value_f64(&tensor, 0); rows],
                vec![rows, 1],
                tensor.numeric_dtype(),
            )
            .map(Value::Tensor)
            .map_err(grouping_error)
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 && array.data.len() != rows {
                return Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ));
            }
            let values = if array.data.len() == 1 {
                vec![array.data[0]; rows]
            } else {
                array.data.into_vec()
            };
            LogicalArray::new(values, vec![rows, 1])
                .map(Value::LogicalArray)
                .map_err(grouping_error)
        }
        other => Err(grouping_error(format!(
            "accumarray: unsupported data input {other:?}"
        ))),
    }
}

fn repeat_integer_scalar(storage: &IntegerStorage, len: usize) -> BuiltinResult<IntegerStorage> {
    let value = storage
        .value_at(0)
        .ok_or_else(|| grouping_error("accumarray: scalar integer data is empty"))?;
    storage
        .from_exact_values_like(vec![value; len])
        .map_err(grouping_error)
}

async fn accumarray_callback_output(
    data: Value,
    buckets: BTreeMap<usize, Vec<usize>>,
    output_shape: Vec<usize>,
    output_len: usize,
    fun: Value,
    fill: Option<Value>,
    issparse: bool,
) -> BuiltinResult<Value> {
    if issparse && !is_double_numeric_value(&data) {
        return Err(grouping_error(
            "accumarray: sparse output requires double input data",
        ));
    }
    if issparse && fill.as_ref().is_some_and(is_integer_numeric_value) {
        return Err(grouping_error(
            "accumarray: sparse output requires a double zero fill value",
        ));
    }
    let mut computed = BTreeMap::new();
    for (linear, rows) in buckets {
        let group = select_rows(&data, &rows)?;
        let result = apply_accumarray_callback(fun.clone(), group).await?;
        computed.insert(linear, accumarray_scalar_result(result)?);
    }
    let prototype = computed.values().next();
    let fill = match fill {
        Some(value) => accumarray_scalar_result(value)?,
        None => prototype
            .map(accumarray_zero_like)
            .transpose()?
            .unwrap_or(Value::Num(0.0)),
    };
    if issparse && value_as_numeric_scalar(&fill) != Some(0.0) {
        return Err(grouping_error(
            "accumarray: sparse output requires a zero fill value",
        ));
    }
    let mut values = vec![fill; output_len];
    for (linear, value) in computed {
        values[linear] = value;
    }
    accumarray_collect_results(values, output_shape, issparse)
}

fn accumarray_scalar_result(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(&tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .value_at(0)
                    .map(Value::Int)
                    .ok_or_else(|| grouping_error("accumarray: empty scalar result"));
            }
            Ok(Value::Tensor(tensor))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(Value::Bool(array.data[0] != 0)),
        Value::Cell(cell) if cell.data.len() == 1 => Ok(cell.data[0].clone()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(value),
        other => Err(grouping_error(format!(
            "accumarray: group function must return a scalar, got {other:?}"
        ))),
    }
}

fn accumarray_zero_like(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Int(value) => IntegerStorage::from_scalar(value.clone())
            .zeros_like(1)
            .value_at(0)
            .map(Value::Int)
            .ok_or_else(|| grouping_error("accumarray: could not construct integer fill")),
        Value::Num(_) => Ok(Value::Num(0.0)),
        Value::Bool(_) => Ok(Value::Bool(false)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Tensor::new_with_dtype(vec![0.0], vec![1, 1], tensor.numeric_dtype())
                .map(Value::Tensor)
                .map_err(grouping_error)
        }
        other => Err(grouping_error(format!(
            "accumarray: explicit fill value required for group result {other:?}"
        ))),
    }
}

fn accumarray_collect_results(
    values: Vec<Value>,
    shape: Vec<usize>,
    issparse: bool,
) -> BuiltinResult<Value> {
    let integer_values = values
        .iter()
        .map(exact_integer_scalar)
        .collect::<Option<Vec<_>>>();
    if let Some(exact) = integer_values {
        if issparse {
            return Err(grouping_error(
                "accumarray: sparse output requires double group results",
            ));
        }
        let prototype = IntegerStorage::from_scalar(
            exact
                .first()
                .cloned()
                .ok_or_else(|| grouping_error("accumarray: empty integer result"))?,
        );
        let storage = prototype
            .from_exact_values_like(exact)
            .map_err(|_| grouping_error("accumarray: fill value class must match group output"))?;
        return Tensor::new_integer(storage, shape)
            .map(Value::Tensor)
            .map_err(grouping_error);
    }
    if values
        .iter()
        .any(|value| exact_integer_scalar(value).is_some())
    {
        return Err(grouping_error(
            "accumarray: fill value class must match group output",
        ));
    }
    if values.iter().all(|value| matches!(value, Value::Bool(_))) {
        if issparse {
            return Err(grouping_error(
                "accumarray: sparse output requires double scalar group results",
            ));
        }
        return LogicalArray::new(
            values
                .into_iter()
                .map(|value| match value {
                    Value::Bool(flag) => u8::from(flag),
                    _ => unreachable!("checked above"),
                })
                .collect(),
            shape,
        )
        .map(Value::LogicalArray)
        .map_err(grouping_error);
    }
    let floating_dtype = values
        .iter()
        .map(floating_scalar_dtype)
        .collect::<Option<Vec<_>>>()
        .and_then(|dtypes| {
            let first = dtypes.first().copied()?;
            dtypes.iter().all(|dtype| *dtype == first).then_some(first)
        });
    if floating_dtype.is_none()
        && values
            .iter()
            .any(|value| floating_scalar_dtype(value).is_some())
    {
        return Err(grouping_error(
            "accumarray: fill value class must match group output",
        ));
    }
    if let Some(dtype) = floating_dtype {
        if issparse && dtype != NumericDType::F64 {
            return Err(grouping_error(
                "accumarray: sparse output requires double scalar group results",
            ));
        }
        let data = values
            .iter()
            .map(|value| value_as_numeric_scalar(value).unwrap())
            .collect::<Vec<_>>();
        if issparse {
            return accumarray_numeric_output(data, shape, true);
        }
        return Tensor::new_with_dtype(data, shape, dtype)
            .map(Value::Tensor)
            .map_err(grouping_error);
    }
    if values
        .iter()
        .all(|value| value_as_numeric_scalar(value).is_some())
    {
        let data = values
            .iter()
            .map(|value| value_as_numeric_scalar(value).unwrap())
            .collect();
        return accumarray_numeric_output(data, shape, issparse);
    }
    if issparse {
        return Err(grouping_error(
            "accumarray: sparse output requires double scalar group results",
        ));
    }
    CellArray::new_with_shape(values, shape)
        .map(Value::Cell)
        .map_err(grouping_error)
}

fn floating_scalar_dtype(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Num(_) => Some(NumericDType::F64),
        Value::Tensor(tensor)
            if tensor_utils::is_scalar_tensor(tensor) && tensor.integer_storage().is_none() =>
        {
            Some(tensor.numeric_dtype())
        }
        _ => None,
    }
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            tensor.integer_storage()?.value_at(0)
        }
        _ => None,
    }
}

fn is_integer_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn is_double_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Num(_))
        || matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64)
}

fn accumarray_subscripts(subs: Value) -> BuiltinResult<Vec<Vec<usize>>> {
    match subs {
        Value::Int(value) => Ok(vec![vec![positive_integer_value(
            &value,
            "accumarray subscript",
        )?]]),
        Value::Num(value) => Ok(vec![vec![positive_integer(value, "accumarray subscript")?]]),
        Value::Tensor(tensor) => {
            if tensor_utils::tensor_element_len(&tensor) == 0 {
                return Ok(Vec::new());
            }
            if let Some(storage) = tensor.integer_storage() {
                if tensor.cols() <= 1 || tensor.rows() == 1 {
                    return storage
                        .exact_values()
                        .into_iter()
                        .map(|value| {
                            Ok(vec![positive_integer_value(
                                &value,
                                "accumarray subscript",
                            )?])
                        })
                        .collect();
                }
                let mut out = Vec::with_capacity(tensor.rows());
                for row in 0..tensor.rows() {
                    let mut subs = Vec::with_capacity(tensor.cols());
                    for col in 0..tensor.cols() {
                        let index = row + col * tensor.rows();
                        let value = storage.value_at(index).ok_or_else(|| {
                            grouping_error("accumarray: integer subscript index out of bounds")
                        })?;
                        subs.push(positive_integer_value(&value, "accumarray subscript")?);
                    }
                    out.push(subs);
                }
                return Ok(out);
            }
            if tensor.cols() <= 1 || tensor.rows() == 1 {
                tensor_utils::tensor_into_values_f64(tensor)
                    .into_iter()
                    .map(|value| Ok(vec![positive_integer(value, "accumarray subscript")?]))
                    .collect()
            } else {
                let mut out = Vec::with_capacity(tensor.rows());
                for row in 0..tensor.rows() {
                    let mut subs = Vec::with_capacity(tensor.cols());
                    for col in 0..tensor.cols() {
                        subs.push(positive_integer(
                            tensor.get2(row, col).map_err(grouping_error)?,
                            "accumarray subscript",
                        )?);
                    }
                    out.push(subs);
                }
                Ok(out)
            }
        }
        Value::Cell(cell) => {
            let mut columns = Vec::with_capacity(cell.data.len());
            for value in cell.data {
                let column = accumarray_subscripts(value)?;
                if column.iter().any(|row| row.len() != 1) {
                    return Err(grouping_error(
                        "accumarray: cell subscript entries must be vectors",
                    ));
                }
                columns.push(column.into_iter().map(|row| row[0]).collect::<Vec<_>>());
            }
            let rows = columns.first().map(Vec::len).unwrap_or(0);
            for column in &columns {
                if column.len() != rows {
                    return Err(grouping_error(
                        "accumarray: cell subscript vectors must have equal length",
                    ));
                }
            }
            let mut out = Vec::with_capacity(rows);
            for row in 0..rows {
                out.push(columns.iter().map(|column| column[row]).collect());
            }
            Ok(out)
        }
        other => Err(grouping_error(format!(
            "accumarray: unsupported subscript input {other:?}"
        ))),
    }
}

fn accumarray_data_values(data: Value, rows: usize) -> BuiltinResult<Vec<f64>> {
    match data {
        Value::Num(value) => Ok(vec![value; rows]),
        Value::Int(value) => Ok(vec![value.to_f64(); rows]),
        Value::Bool(value) => Ok(vec![if value { 1.0 } else { 0.0 }; rows]),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let values = integer_storage_to_f64_vec(storage);
                if values.len() == 1 {
                    return Ok(vec![values[0]; rows]);
                } else if values.len() == rows {
                    return Ok(values);
                }
                return Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ));
            }
            let len = tensor_utils::tensor_element_len(&tensor);
            if len == 1 {
                Ok(vec![tensor_utils::tensor_value_f64(&tensor, 0); rows])
            } else if len == rows {
                Ok(tensor_utils::tensor_into_values_f64(tensor))
            } else {
                Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ))
            }
        }
        Value::LogicalArray(array) => {
            if array.data.len() == 1 {
                Ok(vec![if array.data[0] != 0 { 1.0 } else { 0.0 }; rows])
            } else if array.data.len() == rows {
                Ok(array
                    .data
                    .into_iter()
                    .map(|flag| if flag != 0 { 1.0 } else { 0.0 })
                    .collect())
            } else {
                Err(grouping_error(
                    "accumarray: data must be scalar or match subscript row count",
                ))
            }
        }
        other => Err(grouping_error(format!(
            "accumarray: unsupported data input {other:?}"
        ))),
    }
}

fn infer_accumarray_shape(index_rows: &[Vec<usize>]) -> Vec<usize> {
    let dims = index_rows.first().map(Vec::len).unwrap_or(1).max(1);
    let mut shape = vec![0usize; dims];
    for row in index_rows {
        for (dim, idx) in row.iter().enumerate() {
            shape[dim] = shape[dim].max(*idx);
        }
    }
    if dims == 1 {
        vec![shape[0], 1]
    } else {
        shape
    }
}

fn subscript_to_linear(subs: &[usize], shape: &[usize]) -> BuiltinResult<usize> {
    if subs.len() > shape.len() {
        return Err(grouping_error("accumarray: too many subscript dimensions"));
    }
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (dim, &size) in shape.iter().enumerate() {
        let sub = subs.get(dim).copied().unwrap_or(1);
        if sub == 0 || sub > size {
            return Err(grouping_error("accumarray: subscript exceeds output size"));
        }
        linear = linear
            .checked_add(
                (sub - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| too_large_error("accumarray: output linear index overflow"))?,
            )
            .ok_or_else(|| too_large_error("accumarray: output linear index overflow"))?;
        stride = stride
            .checked_mul(size)
            .ok_or_else(|| too_large_error("accumarray: output size overflow"))?;
    }
    Ok(linear)
}

async fn apply_accumarray_callback(func: Value, values: Value) -> BuiltinResult<Value> {
    let result = call_feval_async_with_outputs(func, &[values], 1)
        .await
        .map_err(|err| callback_error("accumarray: callback failed", Some(err)))?;
    match result {
        Value::OutputList(mut values) if values.len() == 1 => Ok(values.remove(0)),
        other => Ok(other),
    }
}

fn accumarray_numeric_output(
    data: Vec<f64>,
    shape: Vec<usize>,
    issparse: bool,
) -> BuiltinResult<Value> {
    if issparse {
        let (rows, cols) = shape_to_rows_cols(&shape)?;
        if shape.len() > 2 {
            return Err(grouping_error(
                "accumarray: sparse output is only supported for 2-D results",
            ));
        }
        let mut col_ptrs = Vec::with_capacity(cols + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        col_ptrs.push(0);
        for col in 0..cols {
            for row in 0..rows {
                let value = data[row + col * rows];
                if value != 0.0 {
                    row_indices.push(row);
                    values.push(value);
                }
            }
            col_ptrs.push(values.len());
        }
        return SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
            .map(Value::SparseTensor)
            .map_err(grouping_error);
    }
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(grouping_error)
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather_if_needed_async(&value).await?);
    }
    Ok(out)
}

fn numeric_values(value: &Value, context: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        Value::Bool(value) => Ok(vec![if *value { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => Ok(tensor_utils::tensor_values_f64(tensor)),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|flag| if *flag != 0 { 1.0 } else { 0.0 })
            .collect()),
        Value::SparseTensor(sparse) => sparse
            .to_dense()
            .map(tensor_utils::tensor_into_values_f64)
            .map_err(grouping_error),
        other => Err(grouping_error(format!(
            "{context}: expected numeric input, got {other:?}"
        ))),
    }
}

fn parse_positive_size_vector(value: &Value, context: &str) -> BuiltinResult<Vec<usize>> {
    let dims = match value {
        Value::Int(value) => vec![positive_integer_value(value, context)?],
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                storage
                    .exact_values()
                    .iter()
                    .map(|value| positive_integer_value(value, context))
                    .collect::<BuiltinResult<Vec<_>>>()?
            } else {
                numeric_values(value, context)?
                    .into_iter()
                    .map(|value| positive_integer(value, context))
                    .collect::<BuiltinResult<Vec<_>>>()?
            }
        }
        _ => numeric_values(value, context)?
            .into_iter()
            .map(|value| positive_integer(value, context))
            .collect::<BuiltinResult<Vec<_>>>()?,
    };
    if dims.is_empty() {
        return Err(grouping_error(format!(
            "{context}: size vector must not be empty"
        )));
    }
    Ok(if dims.len() == 1 {
        vec![dims[0], 1]
    } else {
        dims
    })
}

fn shape_to_rows_cols(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    let rows = shape.first().copied().unwrap_or(0);
    let cols = if shape.len() <= 1 {
        1
    } else if shape.len() == 2 {
        shape[1]
    } else {
        shape[1..].iter().product()
    };
    Ok((rows, cols))
}

fn checked_element_count(shape: &[usize], context: &str) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| too_large_error(format!("{context}: output size overflow")))
    })
}

fn binary_bool_scalar(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(value) if *value == 0.0 => Ok(false),
        Value::Num(value) if *value == 1.0 => Ok(true),
        Value::Int(value) if value.is_zero() => Ok(false),
        Value::Int(value) if value.to_f64() == 1.0 => Ok(true),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(grouping_error(format!(
            "{context}: expected logical or numeric 0 or 1, got {other:?}"
        ))),
    }
}

fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    value_as_numeric_scalar(value)
        .ok_or_else(|| grouping_error(format!("{context}: expected numeric scalar")))
}

fn value_as_numeric_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Some(if array.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => None,
    }
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor_utils::tensor_element_len(tensor) == 0,
        Value::StringArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::CharArray(chars) => chars.data.is_empty(),
        _ => false,
    }
}

fn positive_integer(value: f64, context: &str) -> BuiltinResult<usize> {
    if let Some(value) = positive_platform_usize(value) {
        Ok(value)
    } else {
        Err(grouping_error(format!(
            "{context}: expected positive integer"
        )))
    }
}

fn positive_integer_value(value: &IntValue, context: &str) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|value| *value > 0)
        .ok_or_else(|| grouping_error(format!("{context}: expected positive integer")))
}

#[cfg(test)]
fn nonnegative_integer(value: f64, context: &str) -> BuiltinResult<usize> {
    if let Some(value) = nonnegative_platform_usize(value) {
        Ok(value)
    } else {
        Err(grouping_error(format!(
            "{context}: expected nonnegative integer"
        )))
    }
}

fn integer_storage_to_f64_vec(storage: &IntegerStorage) -> Vec<f64> {
    storage
        .exact_values()
        .iter()
        .map(IntValue::to_f64)
        .collect()
}

fn positive_platform_usize(value: f64) -> Option<usize> {
    nonnegative_platform_usize(value).filter(|value| *value > 0)
}

fn nonnegative_platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage};

    #[test]
    fn grouping_dimension_parsers_reject_fractional_and_out_of_range_doubles() {
        assert!(positive_integer(1.5, "test").is_err());
        assert!(nonnegative_integer(1.5, "test").is_err());
        assert!(positive_integer(usize::MAX as f64 + 1.0, "test").is_err());
        assert!(nonnegative_integer(usize::MAX as f64 + 1.0, "test").is_err());
    }

    #[test]
    fn accumarray_sums_vector_and_matrix_subscripts() {
        let out = block_on(accumarray_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 3.0, 4.0, 2.0, 4.0, 1.0], vec![6, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6, 1]).unwrap()),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![7.0, 4.0, 2.0, 8.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let out = block_on(accumarray_builtin(
            Value::Tensor(
                Tensor::new(
                    vec![1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
                    vec![6, 2],
                )
                .unwrap(),
            ),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6, 1]).unwrap()),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![4, 2]);
                assert_eq!(
                    tensor.materialize_f64(),
                    vec![5.0, 0.0, 0.0, 6.0, 0.0, 7.0, 3.0, 0.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn accumarray_integer_metadata_covers_structural_default_callback_and_sparse_forms() {
        assert_eq!(ACCUMARRAY_DESCRIPTOR.signatures.len(), 5);
        assert_eq!(ACCUMARRAY_DESCRIPTOR.output_mode, BuiltinOutputMode::Fixed);
        assert_eq!(ACCUMARRAY_INTEGER_CAPABILITIES.len(), 4);
        assert!(ACCUMARRAY_INTEGER_CAPABILITIES
            .iter()
            .any(|capability| capability.computation_domain
                == BuiltinIntegerComputationDomain::Structural));
        assert!(ACCUMARRAY_INTEGER_CAPABILITIES
            .iter()
            .any(|capability| capability.output_class == BuiltinIntegerOutputClassRule::Double));
        assert!(ACCUMARRAY_INTEGER_CAPABILITIES.iter().any(|capability| {
            capability
                .inputs
                .iter()
                .any(|input| input.availability == BuiltinIntegerInputAvailability::Rejected)
        }));
    }

    #[test]
    fn accumarray_accepts_exact_integer_subscripts_data_and_size_vectors() {
        let subs_tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 3, 4, 2]), vec![4, 1]).unwrap();
        let subs = Value::Tensor(subs_tensor);
        let data_tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![10, 20, 30, 40]), vec![4, 1]).unwrap();
        let data = Value::Tensor(data_tensor);
        let size_tensor = Tensor::new_integer(IntegerStorage::U8(vec![4, 1]), vec![1, 2]).unwrap();
        let size = Value::Tensor(size_tensor);

        let out = block_on(accumarray_builtin(subs, data, vec![size])).unwrap();

        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![4, 1]);
                assert_eq!(tensor.materialize_f64(), vec![10.0, 40.0, 20.0, 30.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn grouping_numeric_scalar_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1]).unwrap();

        assert_eq!(
            value_as_numeric_scalar(&Value::Tensor(tensor)),
            Some(2026.0)
        );
    }

    #[test]
    fn accumarray_empty_subscripts_read_typed_integer_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 1]).unwrap();

        assert_eq!(
            accumarray_subscripts(Value::Tensor(tensor)).unwrap(),
            Vec::<Vec<usize>>::new()
        );
    }

    #[test]
    fn accumarray_data_values_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).unwrap();

        assert_eq!(
            accumarray_data_values(Value::Tensor(tensor), 3).unwrap(),
            vec![7.0; 3]
        );
    }

    #[test]
    fn accumarray_rejects_negative_exact_integer_subscripts_and_sizes() {
        let subs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![1, -1]), vec![2, 1]).unwrap(),
        );
        let data = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap());
        let err = block_on(accumarray_builtin(subs, data, Vec::new()))
            .expect_err("negative subscript should fail");
        assert!(err.message.contains("expected positive integer"));

        let subs =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap());
        let data = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap());
        let size = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![2, -1]), vec![1, 2]).unwrap(),
        );
        let err = block_on(accumarray_builtin(subs, data, vec![size]))
            .expect_err("negative size should fail");
        assert!(err.message.contains("expected positive integer"));
    }

    fn all_integer_triplets() -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(vec![4, 2, 9]),
            IntegerStorage::I16(vec![4, 2, 9]),
            IntegerStorage::I32(vec![4, 2, 9]),
            IntegerStorage::I64(vec![4, 2, 9]),
            IntegerStorage::U8(vec![4, 2, 9]),
            IntegerStorage::U16(vec![4, 2, 9]),
            IntegerStorage::U32(vec![4, 2, 9]),
            IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1, 9]),
        ]
    }

    #[test]
    fn accumarray_default_sum_accepts_all_integer_data_classes_and_returns_double() {
        for storage in all_integer_triplets() {
            let data = Tensor::new_integer(storage, vec![3, 1]).unwrap();
            let out = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 2]), vec![3, 1]).unwrap(),
                ),
                Value::Tensor(data),
                Vec::new(),
            ))
            .unwrap();
            let Value::Tensor(out) = out else {
                panic!("expected dense numeric output");
            };
            assert!(out.integer_storage().is_none());
            assert_eq!(out.shape, vec![2, 1]);
        }
    }

    #[test]
    fn accumarray_default_integer_sum_rejects_typed_integer_fill() {
        let err = block_on(accumarray_builtin(
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap()),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![4, 9]), vec![2, 1]).unwrap(),
            ),
            vec![
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![3, 1]), vec![1, 2]).unwrap(),
                ),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Int(IntValue::I16(0)),
            ],
        ))
        .expect_err("default integer sum requires double fill");
        assert!(err.message.contains("double default sum output"));
    }

    #[test]
    fn accumarray_custom_callback_preserves_native_single_result_class() {
        let data =
            Tensor::new_with_dtype(vec![4.0, 2.0, 9.0], vec![3, 1], NumericDType::F32).unwrap();
        let out = block_on(accumarray_builtin(
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 2]), vec![3, 1]).unwrap(),
            ),
            Value::Tensor(data),
            vec![
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![3, 1]), vec![1, 2]).unwrap(),
                ),
                Value::FunctionHandle("min".into()),
            ],
        ))
        .unwrap();
        let Value::Tensor(out) = out else {
            panic!("expected native-single tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        assert_eq!(out.materialize_f64(), vec![2.0, 9.0, 0.0]);
    }

    #[test]
    fn accumarray_custom_min_preserves_all_integer_classes_and_exact_fill() {
        for storage in all_integer_triplets() {
            let second = storage.value_at(1).unwrap();
            let third = storage.value_at(2).unwrap();
            let fill = storage.zeros_like(1).value_at(0).unwrap();
            let expected = storage
                .from_exact_values_like(vec![second, third, fill.clone()])
                .unwrap();
            let out = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 2]), vec![3, 1]).unwrap(),
                ),
                Value::Tensor(Tensor::new_integer(storage, vec![3, 1]).unwrap()),
                vec![
                    Value::Tensor(
                        Tensor::new_integer(IntegerStorage::U8(vec![3, 1]), vec![1, 2]).unwrap(),
                    ),
                    Value::FunctionHandle("min".into()),
                    Value::Int(fill),
                ],
            ))
            .unwrap();
            let Value::Tensor(out) = out else {
                panic!("expected typed integer output");
            };
            assert_eq!(out.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn accumarray_accepts_all_integer_subscript_and_size_classes_exactly() {
        let subs = vec![
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];
        let sizes = vec![
            IntegerStorage::I8(vec![3, 1]),
            IntegerStorage::I16(vec![3, 1]),
            IntegerStorage::I32(vec![3, 1]),
            IntegerStorage::I64(vec![3, 1]),
            IntegerStorage::U8(vec![3, 1]),
            IntegerStorage::U16(vec![3, 1]),
            IntegerStorage::U32(vec![3, 1]),
            IntegerStorage::U64(vec![3, 1]),
        ];
        for (subs, size) in subs.into_iter().zip(sizes) {
            let out = block_on(accumarray_builtin(
                Value::Tensor(Tensor::new_integer(subs, vec![2, 1]).unwrap()),
                Value::Num(1.0),
                vec![Value::Tensor(
                    Tensor::new_integer(size, vec![1, 2]).unwrap(),
                )],
            ))
            .unwrap();
            let Value::Tensor(out) = out else {
                panic!("expected dense output");
            };
            assert_eq!(out.shape, vec![3, 1]);
            assert_eq!(out.materialize_f64(), vec![1.0, 1.0, 0.0]);
        }
    }

    #[test]
    fn accumarray_accepts_all_integer_scalar_subscript_classes() {
        for storage in all_integer_triplets() {
            let subscript = storage
                .from_exact_values_like(vec![storage.cast_f64_assignment(1.0)])
                .unwrap()
                .value_at(0)
                .unwrap();
            let out = block_on(accumarray_builtin(
                Value::Int(subscript),
                Value::Int(storage.value_at(0).unwrap()),
                Vec::new(),
            ))
            .unwrap();
            let Value::Tensor(out) = out else {
                panic!("expected dense output");
            };
            assert_eq!(out.shape, vec![1, 1]);
            assert!(out.integer_storage().is_none());
        }
    }

    #[test]
    fn accumarray_sparse_rejects_all_integer_data_classes_and_nonbinary_controls() {
        let empty = || Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        for storage in all_integer_triplets() {
            let err = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 2]), vec![3, 1]).unwrap(),
                ),
                Value::Tensor(Tensor::new_integer(storage, vec![3, 1]).unwrap()),
                vec![empty(), empty(), empty(), Value::Bool(true)],
            ))
            .expect_err("sparse integer data must reject");
            assert!(err.message.contains("requires double input data"));
        }

        for storage in all_integer_triplets() {
            let control = storage
                .from_exact_values_like(vec![storage.cast_f64_assignment(2.0)])
                .unwrap();
            let err = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap(),
                ),
                Value::Num(1.0),
                vec![
                    empty(),
                    empty(),
                    empty(),
                    Value::Int(control.value_at(0).unwrap()),
                ],
            ))
            .expect_err("issparse must be binary");
            assert!(err.message.contains("expected logical or numeric 0 or 1"));
        }

        let err = block_on(accumarray_builtin(
            Value::Num(1.0),
            Value::Tensor(
                Tensor::new_with_dtype(vec![1.0], vec![1, 1], NumericDType::F32).unwrap(),
            ),
            vec![empty(), empty(), empty(), Value::Bool(true)],
        ))
        .expect_err("sparse single data must reject");
        assert!(err.message.contains("requires double input data"));
    }

    #[test]
    fn accumarray_rejects_resident_integer_data_before_gather() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let error = block_on(accumarray_builtin(
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap(),
                ),
                Value::GpuTensor(handle.clone()),
                Vec::new(),
            ))
            .expect_err("resident integer data must reject");
            assert!(error.message.contains("GPU input data must be"));
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn accumarray_supports_callbacks_and_sparse_output() {
        let out = block_on(accumarray_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![4, 1]).unwrap()),
            vec![
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::FunctionHandle("mean".into()),
            ],
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 6.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let sparse = block_on(accumarray_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
            Value::Num(1.0),
            vec![
                Value::Tensor(Tensor::new(vec![4.0, 1.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Bool(true),
            ],
        ))
        .unwrap();
        match sparse {
            Value::SparseTensor(st) => {
                assert_eq!(st.rows, 4);
                assert_eq!(st.cols, 1);
                assert_eq!(st.nnz(), 2);
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }
}
