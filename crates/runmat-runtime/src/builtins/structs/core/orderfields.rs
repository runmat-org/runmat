//! MATLAB-compatible `orderfields` builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::structs::type_resolvers::orderfields_type;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::orderfields")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "orderfields",
    op_kind: GpuOpKind::Custom("orderfields"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only metadata manipulation; struct values that live on the GPU remain resident.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::orderfields")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "orderfields",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Reordering fields is a metadata operation and does not participate in fusion planning.",
};

const BUILTIN_NAME: &str = "orderfields";

const ORDERFIELDS_OUTPUT_ORDERED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Struct or struct array with reordered fields.",
}];

const ORDERFIELDS_OUTPUT_PERM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Struct or struct array with reordered fields.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Permutation vector mapping ordered fields to original positions.",
    },
];

const ORDERFIELDS_INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input struct or struct array.",
}];

const ORDERFIELDS_INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input struct or struct array.",
    },
    BuiltinParamDescriptor {
        name: "order",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Reference struct, field-name list, or permutation vector.",
    },
];

const ORDERFIELDS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "S = orderfields(S)",
        inputs: &ORDERFIELDS_INPUTS_ONE,
        outputs: &ORDERFIELDS_OUTPUT_ORDERED,
    },
    BuiltinSignatureDescriptor {
        label: "S = orderfields(S, order)",
        inputs: &ORDERFIELDS_INPUTS_TWO,
        outputs: &ORDERFIELDS_OUTPUT_ORDERED,
    },
    BuiltinSignatureDescriptor {
        label: "[S,P] = orderfields(S)",
        inputs: &ORDERFIELDS_INPUTS_ONE,
        outputs: &ORDERFIELDS_OUTPUT_PERM,
    },
    BuiltinSignatureDescriptor {
        label: "[S,P] = orderfields(S, order)",
        inputs: &ORDERFIELDS_INPUTS_TWO,
        outputs: &ORDERFIELDS_OUTPUT_PERM,
    },
];

const ORDERFIELDS_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.TOO_MANY_INPUTS",
    identifier: Some("orderfields:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "orderfields: expected at most two input arguments",
};

const ORDERFIELDS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INVALID_INPUT",
    identifier: Some("orderfields:InvalidInput"),
    when: "First argument is not a struct or struct array.",
    message: "orderfields: first argument must be a struct or struct array",
};

const ORDERFIELDS_ERROR_EMPTY_STRUCT_ARRAY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.EMPTY_STRUCT_ARRAY",
    identifier: Some("orderfields:EmptyStructArray"),
    when: "Empty struct array is asked to adopt a non-empty reference order.",
    message: "orderfields: empty struct arrays cannot adopt a non-empty reference order",
};

const ORDERFIELDS_ERROR_NO_FIELDS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.NO_FIELDS",
    identifier: Some("orderfields:NoFields"),
    when: "Ordering is requested for a struct array with no fields.",
    message: "orderfields: struct array has no fields to reorder",
};

const ORDERFIELDS_ERROR_INVALID_STRUCT_ARRAY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INVALID_STRUCT_ARRAY",
    identifier: Some("orderfields:InvalidStructArray"),
    when: "A struct-array element is not a struct.",
    message: "orderfields: struct array element is not a struct",
};

const ORDERFIELDS_ERROR_INVALID_STRUCT_CONTENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INVALID_STRUCT_CONTENTS",
    identifier: Some("orderfields:InvalidStructContents"),
    when: "Struct-array contents are not all structs.",
    message: "orderfields: expected struct array contents to be structs",
};

const ORDERFIELDS_ERROR_REBUILD_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.REBUILD_FAILED",
    identifier: Some("orderfields:RebuildFailed"),
    when: "Rebuilding reordered struct array failed.",
    message: "orderfields: failed to rebuild struct array",
};

const ORDERFIELDS_ERROR_INVALID_REFERENCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INVALID_REFERENCE",
    identifier: Some("orderfields:InvalidReference"),
    when: "Reference struct-array contains non-struct values.",
    message: "orderfields: reference struct array element is not a struct",
};

const ORDERFIELDS_ERROR_INVALID_NAME_LIST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INVALID_NAME_LIST",
    identifier: Some("orderfields:InvalidFieldNameList"),
    when: "Name-list entries are not scalar strings or character vectors.",
    message: "orderfields: cell array elements must be a string or character vector",
};

const ORDERFIELDS_ERROR_EMPTY_FIELD_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.EMPTY_FIELD_NAME",
    identifier: Some("orderfields:EmptyFieldName"),
    when: "Requested field-name list contains an empty name.",
    message: "orderfields: field names must be nonempty",
};

const ORDERFIELDS_ERROR_INVALID_PERMUTATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INVALID_PERMUTATION",
    identifier: Some("orderfields:InvalidPermutation"),
    when: "Permutation vector does not include each field exactly once.",
    message: "orderfields: index vector must permute every field exactly once",
};

const ORDERFIELDS_ERROR_INDEX_NOT_INTEGER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INDEX_NOT_INTEGER",
    identifier: Some("orderfields:IndexNotInteger"),
    when: "Permutation vector contains non-integer entries.",
    message: "orderfields: index vector must contain integers",
};

const ORDERFIELDS_ERROR_INDEX_OUT_OF_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INDEX_OUT_OF_RANGE",
    identifier: Some("orderfields:IndexOutOfRange"),
    when: "Permutation vector index is outside valid field range.",
    message: "orderfields: index vector element out of range",
};

const ORDERFIELDS_ERROR_INDEX_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INDEX_DUPLICATE",
    identifier: Some("orderfields:DuplicateIndex"),
    when: "Permutation vector contains duplicate positions.",
    message: "orderfields: index vector contains duplicate positions",
};

const ORDERFIELDS_ERROR_FIELD_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.FIELD_MISMATCH",
    identifier: Some("orderfields:FieldMismatch"),
    when: "Requested field set does not exactly match struct fields.",
    message: "orderfields: field names must match the struct exactly",
};

const ORDERFIELDS_ERROR_UNKNOWN_FIELD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.UNKNOWN_FIELD",
    identifier: Some("orderfields:UnknownField"),
    when: "Requested order references an unknown field.",
    message: "orderfields: unknown field in requested order",
};

const ORDERFIELDS_ERROR_DUPLICATE_FIELD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.DUPLICATE_FIELD",
    identifier: Some("orderfields:DuplicateField"),
    when: "Requested field order includes duplicate names.",
    message: "orderfields: duplicate field in requested order",
};

const ORDERFIELDS_ERROR_MISSING_FIELD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.MISSING_FIELD",
    identifier: Some("orderfields:MissingField"),
    when: "A field from the requested order is missing on the struct.",
    message: "orderfields: field does not exist on the struct",
};

const ORDERFIELDS_ERROR_INVALID_ORDER_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INVALID_ORDER_ARGUMENT",
    identifier: Some("orderfields:InvalidOrderArgument"),
    when: "Second argument is not a supported order descriptor.",
    message: "orderfields: unrecognised ordering argument",
};

const ORDERFIELDS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDERFIELDS.INTERNAL",
    identifier: Some("orderfields:InternalError"),
    when: "Internal permutation tensor construction failed.",
    message: "orderfields: internal error",
};

const ORDERFIELDS_ERRORS: [BuiltinErrorDescriptor; 20] = [
    ORDERFIELDS_ERROR_TOO_MANY_INPUTS,
    ORDERFIELDS_ERROR_INVALID_INPUT,
    ORDERFIELDS_ERROR_EMPTY_STRUCT_ARRAY,
    ORDERFIELDS_ERROR_NO_FIELDS,
    ORDERFIELDS_ERROR_INVALID_STRUCT_ARRAY,
    ORDERFIELDS_ERROR_INVALID_STRUCT_CONTENTS,
    ORDERFIELDS_ERROR_REBUILD_FAILED,
    ORDERFIELDS_ERROR_INVALID_REFERENCE,
    ORDERFIELDS_ERROR_INVALID_NAME_LIST,
    ORDERFIELDS_ERROR_EMPTY_FIELD_NAME,
    ORDERFIELDS_ERROR_INVALID_PERMUTATION,
    ORDERFIELDS_ERROR_INDEX_NOT_INTEGER,
    ORDERFIELDS_ERROR_INDEX_OUT_OF_RANGE,
    ORDERFIELDS_ERROR_INDEX_DUPLICATE,
    ORDERFIELDS_ERROR_FIELD_MISMATCH,
    ORDERFIELDS_ERROR_UNKNOWN_FIELD,
    ORDERFIELDS_ERROR_DUPLICATE_FIELD,
    ORDERFIELDS_ERROR_MISSING_FIELD,
    ORDERFIELDS_ERROR_INVALID_ORDER_ARGUMENT,
    ORDERFIELDS_ERROR_INTERNAL,
];

pub const ORDERFIELDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ORDERFIELDS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ORDERFIELDS_ERRORS,
};

fn orderfields_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    orderfields_error_with_message(error.message, error)
}

fn orderfields_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "orderfields",
    category = "structs/core",
    summary = "Reorder structure field definitions alphabetically or using a supplied order.",
    keywords = "orderfields,struct,reorder fields,alphabetical,struct array",
    type_resolver(orderfields_type),
    descriptor(crate::builtins::structs::core::orderfields::ORDERFIELDS_DESCRIPTOR),
    builtin_path = "crate::builtins::structs::core::orderfields"
)]
async fn orderfields_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let eval = evaluate(value, &rest)?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let (ordered, permutation) = eval.into_values();
        let mut outputs = vec![ordered];
        if out_count >= 2 {
            outputs.push(permutation);
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
    Ok(eval.into_ordered_value())
}

/// Evaluate the `orderfields` builtin once and expose both outputs.
pub fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<OrderFieldsEvaluation> {
    if rest.len() > 1 {
        return Err(orderfields_error(&ORDERFIELDS_ERROR_TOO_MANY_INPUTS));
    }
    let order_arg = rest.first();

    match value {
        Value::Struct(struct_value) => {
            let original: Vec<String> = struct_value.field_names().cloned().collect();
            let order = resolve_order(&struct_value, order_arg)?;
            let permutation = permutation_from(&original, &order)?;
            let permutation = permutation_tensor(permutation)?;
            let reordered = reorder_struct(&struct_value, &order)?;
            Ok(OrderFieldsEvaluation::new(
                Value::Struct(reordered),
                permutation,
            ))
        }
        Value::Cell(cell) => {
            if cell.data.is_empty() {
                let permutation = permutation_tensor(Vec::new())?;
                if let Some(arg) = order_arg {
                    if let Some(reference) = extract_reference_struct(arg)? {
                        if reference.fields.is_empty() {
                            return Ok(OrderFieldsEvaluation::new(Value::Cell(cell), permutation));
                        } else {
                            return Err(orderfields_error(&ORDERFIELDS_ERROR_EMPTY_STRUCT_ARRAY));
                        }
                    }
                    if let Some(names) = extract_name_list(arg)? {
                        if names.is_empty() {
                            return Ok(OrderFieldsEvaluation::new(Value::Cell(cell), permutation));
                        }
                        return Err(orderfields_error(&ORDERFIELDS_ERROR_NO_FIELDS));
                    }
                    if let Value::Tensor(tensor) = arg {
                        if tensor.data.is_empty() {
                            return Ok(OrderFieldsEvaluation::new(Value::Cell(cell), permutation));
                        }
                        return Err(orderfields_error(&ORDERFIELDS_ERROR_NO_FIELDS));
                    }
                    return Err(orderfields_error(&ORDERFIELDS_ERROR_NO_FIELDS));
                }
                return Ok(OrderFieldsEvaluation::new(Value::Cell(cell), permutation));
            }
            let first = extract_struct_from_cell(&cell, 0)?;
            let original: Vec<String> = first.field_names().cloned().collect();
            let order = resolve_order(&first, order_arg)?;
            let permutation = permutation_from(&original, &order)?;
            let permutation = permutation_tensor(permutation)?;
            let reordered = reorder_struct_array(&cell, &order)?;
            Ok(OrderFieldsEvaluation::new(
                Value::Cell(reordered),
                permutation,
            ))
        }
        other => Err(orderfields_error_with_message(
            format!(
                "{} (got {other:?})",
                ORDERFIELDS_ERROR_INVALID_INPUT.message
            ),
            &ORDERFIELDS_ERROR_INVALID_INPUT,
        )),
    }
}

pub struct OrderFieldsEvaluation {
    ordered: Value,
    permutation: Tensor,
}

impl OrderFieldsEvaluation {
    fn new(ordered: Value, permutation: Tensor) -> Self {
        Self {
            ordered,
            permutation,
        }
    }

    pub fn into_ordered_value(self) -> Value {
        self.ordered
    }

    pub fn permutation_value(&self) -> Value {
        tensor::tensor_into_value(self.permutation.clone())
    }

    pub fn into_values(self) -> (Value, Value) {
        let perm = tensor::tensor_into_value(self.permutation);
        (self.ordered, perm)
    }
}

fn reorder_struct_array(array: &CellArray, order: &[String]) -> BuiltinResult<CellArray> {
    let mut reordered_elems = Vec::with_capacity(array.data.len());
    for (index, handle) in array.data.iter().enumerate() {
        let value = unsafe { &*handle.as_raw() };
        let Value::Struct(st) = value else {
            return Err(orderfields_error_with_message(
                format!(
                    "{} (element {} is not a struct)",
                    ORDERFIELDS_ERROR_INVALID_STRUCT_ARRAY.message,
                    index + 1
                ),
                &ORDERFIELDS_ERROR_INVALID_STRUCT_ARRAY,
            ));
        };
        ensure_same_field_set(order, st)?;
        let reordered = reorder_struct(st, order)?;
        reordered_elems.push(Value::Struct(reordered));
    }
    CellArray::new_with_shape(reordered_elems, array.shape.clone()).map_err(|e| {
        orderfields_error_with_message(
            format!("{}: {e}", ORDERFIELDS_ERROR_REBUILD_FAILED.message),
            &ORDERFIELDS_ERROR_REBUILD_FAILED,
        )
    })
}

fn reorder_struct(struct_value: &StructValue, order: &[String]) -> BuiltinResult<StructValue> {
    let mut reordered = StructValue::new();
    for name in order {
        let value = struct_value
            .fields
            .get(name)
            .ok_or_else(|| missing_field(name))?
            .clone();
        reordered.fields.insert(name.clone(), value);
    }
    Ok(reordered)
}

fn resolve_order(
    struct_value: &StructValue,
    order_arg: Option<&Value>,
) -> BuiltinResult<Vec<String>> {
    let mut current: Vec<String> = struct_value.field_names().cloned().collect();
    if let Some(arg) = order_arg {
        if let Some(reference) = extract_reference_struct(arg)? {
            let reference_names: Vec<String> = reference.field_names().cloned().collect();
            ensure_same_field_set(&reference_names, struct_value)?;
            return Ok(reference_names);
        }

        if let Some(names) = extract_name_list(arg)? {
            ensure_same_field_set(&names, struct_value)?;
            return Ok(names);
        }

        if let Some(permutation) = extract_indices(&current, arg)? {
            return Ok(permutation);
        }

        return Err(orderfields_error(&ORDERFIELDS_ERROR_INVALID_ORDER_ARGUMENT));
    }

    sort_field_names(&mut current);
    Ok(current)
}

fn permutation_from(original: &[String], order: &[String]) -> BuiltinResult<Vec<f64>> {
    let mut index_map = HashMap::with_capacity(original.len());
    for (idx, name) in original.iter().enumerate() {
        index_map.insert(name.as_str(), idx);
    }
    let mut indices = Vec::with_capacity(order.len());
    for name in order {
        let Some(position) = index_map.get(name.as_str()) else {
            return Err(missing_field(name));
        };
        indices.push((*position as f64) + 1.0);
    }
    Ok(indices)
}

fn permutation_tensor(indices: Vec<f64>) -> BuiltinResult<Tensor> {
    let rows = indices.len();
    let shape = vec![rows, 1];
    Tensor::new(indices, shape).map_err(|e| {
        orderfields_error_with_message(
            format!("{}: {e}", ORDERFIELDS_ERROR_INTERNAL.message),
            &ORDERFIELDS_ERROR_INTERNAL,
        )
    })
}

fn sort_field_names(names: &mut [String]) {
    names.sort_by(|a, b| {
        let lower_a = a.to_ascii_lowercase();
        let lower_b = b.to_ascii_lowercase();
        match lower_a.cmp(&lower_b) {
            Ordering::Equal => a.cmp(b),
            other => other,
        }
    });
}

fn extract_reference_struct(value: &Value) -> BuiltinResult<Option<StructValue>> {
    match value {
        Value::Struct(st) => Ok(Some(st.clone())),
        Value::Cell(cell) => {
            let mut first: Option<StructValue> = None;
            for (index, handle) in cell.data.iter().enumerate() {
                let value = unsafe { &*handle.as_raw() };
                if let Value::Struct(st) = value {
                    if first.is_none() {
                        first = Some(st.clone());
                    }
                } else if first.is_some() {
                    return Err(orderfields_error_with_message(
                        format!(
                            "{} (element {} is not a struct)",
                            ORDERFIELDS_ERROR_INVALID_REFERENCE.message,
                            index + 1
                        ),
                        &ORDERFIELDS_ERROR_INVALID_REFERENCE,
                    ));
                } else {
                    return Ok(None);
                }
            }
            Ok(first)
        }
        _ => Ok(None),
    }
}

fn extract_name_list(arg: &Value) -> BuiltinResult<Option<Vec<String>>> {
    match arg {
        Value::Cell(cell) => {
            let mut names = Vec::with_capacity(cell.data.len());
            for (index, handle) in cell.data.iter().enumerate() {
                let value = unsafe { &*handle.as_raw() };
                let text = scalar_string(value).ok_or_else(|| {
                    orderfields_error_with_message(
                        format!(
                            "{} (cell array element {})",
                            ORDERFIELDS_ERROR_INVALID_NAME_LIST.message,
                            index + 1
                        ),
                        &ORDERFIELDS_ERROR_INVALID_NAME_LIST,
                    )
                })?;
                if text.is_empty() {
                    return Err(orderfields_error(&ORDERFIELDS_ERROR_EMPTY_FIELD_NAME));
                }
                names.push(text);
            }
            Ok(Some(names))
        }
        Value::StringArray(sa) => Ok(Some(sa.data.clone())),
        Value::CharArray(ca) => {
            if ca.rows == 0 {
                return Ok(Some(Vec::new()));
            }
            let mut names = Vec::with_capacity(ca.rows);
            for row in 0..ca.rows {
                let start = row * ca.cols;
                let end = start + ca.cols;
                let mut text: String = ca.data[start..end].iter().collect();
                while text.ends_with(' ') {
                    text.pop();
                }
                if text.is_empty() {
                    return Err(orderfields_error(&ORDERFIELDS_ERROR_EMPTY_FIELD_NAME));
                }
                names.push(text);
            }
            Ok(Some(names))
        }
        _ => Ok(None),
    }
}

fn extract_indices(current: &[String], arg: &Value) -> BuiltinResult<Option<Vec<String>>> {
    let Value::Tensor(tensor) = arg else {
        return Ok(None);
    };
    if tensor.data.is_empty() && current.is_empty() {
        return Ok(Some(Vec::new()));
    }
    if tensor.data.len() != current.len() {
        return Err(orderfields_error(&ORDERFIELDS_ERROR_INVALID_PERMUTATION));
    }
    let mut seen = HashSet::with_capacity(current.len());
    let mut order = Vec::with_capacity(current.len());
    for value in &tensor.data {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err(orderfields_error(&ORDERFIELDS_ERROR_INDEX_NOT_INTEGER));
        }
        let idx = *value as isize;
        if idx < 1 || idx as usize > current.len() {
            return Err(orderfields_error(&ORDERFIELDS_ERROR_INDEX_OUT_OF_RANGE));
        }
        let zero_based = (idx as usize) - 1;
        if !seen.insert(zero_based) {
            return Err(orderfields_error(&ORDERFIELDS_ERROR_INDEX_DUPLICATE));
        }
        order.push(current[zero_based].clone());
    }
    Ok(Some(order))
}

fn ensure_same_field_set(order: &[String], original: &StructValue) -> BuiltinResult<()> {
    if order.len() != original.fields.len() {
        return Err(orderfields_error(&ORDERFIELDS_ERROR_FIELD_MISMATCH));
    }
    let mut seen = HashSet::with_capacity(order.len());
    let original_set: HashSet<&str> = original.field_names().map(|s| s.as_str()).collect();
    for name in order {
        if !original_set.contains(name.as_str()) {
            return Err(orderfields_error_with_message(
                format!(
                    "{} ('{name}' in requested order)",
                    ORDERFIELDS_ERROR_UNKNOWN_FIELD.message
                ),
                &ORDERFIELDS_ERROR_UNKNOWN_FIELD,
            ));
        }
        if !seen.insert(name.as_str()) {
            return Err(orderfields_error_with_message(
                format!(
                    "{} ('{name}' in requested order)",
                    ORDERFIELDS_ERROR_DUPLICATE_FIELD.message
                ),
                &ORDERFIELDS_ERROR_DUPLICATE_FIELD,
            ));
        }
    }
    Ok(())
}

fn extract_struct_from_cell(cell: &CellArray, index: usize) -> BuiltinResult<StructValue> {
    let value = unsafe { &*cell.data[index].as_raw() };
    match value {
        Value::Struct(st) => Ok(st.clone()),
        other => Err(orderfields_error_with_message(
            format!(
                "{} (found {other:?})",
                ORDERFIELDS_ERROR_INVALID_STRUCT_CONTENTS.message
            ),
            &ORDERFIELDS_ERROR_INVALID_STRUCT_CONTENTS,
        )),
    }
}

fn scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let mut text: String = ca.data.iter().collect();
            while text.ends_with(' ') {
                text.pop();
            }
            Some(text)
        }
        _ => None,
    }
}

fn missing_field(name: &str) -> RuntimeError {
    orderfields_error_with_message(
        format!("{} ('{name}')", ORDERFIELDS_ERROR_MISSING_FIELD.message),
        &ORDERFIELDS_ERROR_MISSING_FIELD,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CellArray, CharArray, StringArray, Tensor};

    fn run_orderfields(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::orderfields_builtin(value, rest))
    }

    fn assert_error_identifier(error: RuntimeError, expected: &str) {
        assert_eq!(error.identifier(), Some(expected));
    }

    fn field_order(struct_value: &StructValue) -> Vec<String> {
        struct_value.field_names().cloned().collect()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_sorts_alphabetically() {
        let mut st = StructValue::new();
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("gamma".to_string(), Value::Num(3.0));

        let result = run_orderfields(Value::Struct(st), Vec::new()).expect("orderfields");
        let Value::Struct(sorted) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&sorted),
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_cell_name_list() {
        let mut st = StructValue::new();
        st.fields.insert("a".to_string(), Value::Num(1.0));
        st.fields.insert("b".to_string(), Value::Num(2.0));
        st.fields.insert("c".to_string(), Value::Num(3.0));
        let names = CellArray::new(
            vec![Value::from("c"), Value::from("a"), Value::from("b")],
            1,
            3,
        )
        .expect("cell");

        let reordered =
            run_orderfields(Value::Struct(st), vec![Value::Cell(names)]).expect("orderfields");
        let Value::Struct(result) = reordered else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&result),
            vec!["c".to_string(), "a".to_string(), "b".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_string_array_names() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        st.fields.insert("gamma".to_string(), Value::Num(3.0));

        let strings = StringArray::new(
            vec!["gamma".into(), "alpha".into(), "beta".into()],
            vec![1, 3],
        )
        .expect("string array");

        let result = run_orderfields(Value::Struct(st), vec![Value::StringArray(strings)])
            .expect("orderfields");
        let Value::Struct(sorted) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&sorted),
            vec!["gamma".to_string(), "alpha".to_string(), "beta".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_char_array_names() {
        let mut st = StructValue::new();
        st.fields.insert("cat".to_string(), Value::Num(1.0));
        st.fields.insert("ant".to_string(), Value::Num(2.0));
        st.fields.insert("bat".to_string(), Value::Num(3.0));

        let data = vec!['b', 'a', 't', 'c', 'a', 't', 'a', 'n', 't'];
        let char_array = CharArray::new(data, 3, 3).expect("char array");

        let result =
            run_orderfields(Value::Struct(st), vec![Value::CharArray(char_array)]).expect("order");
        let Value::Struct(sorted) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&sorted),
            vec!["bat".to_string(), "cat".to_string(), "ant".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_reference_struct() {
        let mut source = StructValue::new();
        source.fields.insert("y".to_string(), Value::Num(2.0));
        source.fields.insert("x".to_string(), Value::Num(1.0));

        let mut reference = StructValue::new();
        reference.fields.insert("x".to_string(), Value::Num(0.0));
        reference.fields.insert("y".to_string(), Value::Num(0.0));

        let result = run_orderfields(
            Value::Struct(source),
            vec![Value::Struct(reference.clone())],
        )
        .expect("orderfields");
        let Value::Struct(reordered) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&reordered),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_with_index_vector() {
        let mut st = StructValue::new();
        st.fields.insert("first".to_string(), Value::Num(1.0));
        st.fields.insert("second".to_string(), Value::Num(2.0));
        st.fields.insert("third".to_string(), Value::Num(3.0));

        let permutation = Tensor::new(vec![3.0, 1.0, 2.0], vec![1, 3]).expect("tensor permutation");
        let result = run_orderfields(Value::Struct(st), vec![Value::Tensor(permutation)])
            .expect("orderfields");
        let Value::Struct(reordered) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&reordered),
            vec![
                "third".to_string(),
                "first".to_string(),
                "second".to_string()
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn index_vector_must_be_integers() {
        let mut st = StructValue::new();
        st.fields.insert("one".to_string(), Value::Num(1.0));
        st.fields.insert("two".to_string(), Value::Num(2.0));

        let permutation = Tensor::new(vec![1.0, 1.5], vec![1, 2]).expect("tensor");
        let err = run_orderfields(Value::Struct(st), vec![Value::Tensor(permutation)]).unwrap_err();
        assert_error_identifier(err, ORDERFIELDS_ERROR_INDEX_NOT_INTEGER.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permutation_vector_matches_original_positions() {
        let mut st = StructValue::new();
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("gamma".to_string(), Value::Num(3.0));

        let eval = evaluate(Value::Struct(st), &[]).expect("evaluate");
        let perm = eval.permutation_value();
        match perm {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 1.0, 3.0]),
            other => panic!("expected tensor permutation, got {other:?}"),
        }
        let Value::Struct(ordered) = eval.into_ordered_value() else {
            panic!("expected struct result");
        };
        assert_eq!(
            field_order(&ordered),
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reorder_struct_array() {
        let mut first = StructValue::new();
        first.fields.insert("b".to_string(), Value::Num(1.0));
        first.fields.insert("a".to_string(), Value::Num(2.0));
        let mut second = StructValue::new();
        second.fields.insert("b".to_string(), Value::Num(3.0));
        second.fields.insert("a".to_string(), Value::Num(4.0));
        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .expect("struct array");
        let names =
            CellArray::new(vec![Value::from("a"), Value::from("b")], 1, 2).expect("cell names");

        let result =
            run_orderfields(Value::Cell(array), vec![Value::Cell(names)]).expect("orderfields");
        let Value::Cell(reordered) = result else {
            panic!("expected cell array");
        };
        for handle in &reordered.data {
            let Value::Struct(st) = (unsafe { &*handle.as_raw() }) else {
                panic!("expected struct element");
            };
            assert_eq!(field_order(st), vec!["a".to_string(), "b".to_string()]);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_array_permutation_reuses_order() {
        let mut first = StructValue::new();
        first.fields.insert("z".to_string(), Value::Num(1.0));
        first.fields.insert("x".to_string(), Value::Num(2.0));
        first.fields.insert("y".to_string(), Value::Num(3.0));

        let mut second = StructValue::new();
        second.fields.insert("z".to_string(), Value::Num(4.0));
        second.fields.insert("x".to_string(), Value::Num(5.0));
        second.fields.insert("y".to_string(), Value::Num(6.0));

        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .expect("struct array");

        let eval = evaluate(Value::Cell(array), &[]).expect("evaluate");
        let perm = eval.permutation_value();
        match perm {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0, 1.0]),
            other => panic!("expected tensor permutation, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unknown_field() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        let err = run_orderfields(
            Value::Struct(st),
            vec![Value::Cell(
                CellArray::new(vec![Value::from("beta"), Value::from("gamma")], 1, 2)
                    .expect("cell"),
            )],
        )
        .unwrap_err();
        assert_error_identifier(err, ORDERFIELDS_ERROR_UNKNOWN_FIELD.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn duplicate_field_names_rejected() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("beta".to_string(), Value::Num(2.0));

        let names =
            CellArray::new(vec![Value::from("alpha"), Value::from("alpha")], 1, 2).expect("cell");
        let err = run_orderfields(Value::Struct(st), vec![Value::Cell(names)]).unwrap_err();
        assert_error_identifier(err, ORDERFIELDS_ERROR_DUPLICATE_FIELD.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reference_struct_mismatch_errors() {
        let mut source = StructValue::new();
        source.fields.insert("x".to_string(), Value::Num(1.0));
        source.fields.insert("y".to_string(), Value::Num(2.0));

        let mut reference = StructValue::new();
        reference.fields.insert("x".to_string(), Value::Num(0.0));

        let err =
            run_orderfields(Value::Struct(source), vec![Value::Struct(reference)]).unwrap_err();
        assert_error_identifier(err, ORDERFIELDS_ERROR_FIELD_MISMATCH.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_order_argument_type_errors() {
        let mut st = StructValue::new();
        st.fields.insert("x".to_string(), Value::Num(1.0));

        let err = run_orderfields(Value::Struct(st), vec![Value::Num(1.0)]).unwrap_err();
        assert_error_identifier(
            err,
            ORDERFIELDS_ERROR_INVALID_ORDER_ARGUMENT.identifier.unwrap(),
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_struct_array_nonempty_reference_errors() {
        let empty = CellArray::new(Vec::new(), 0, 0).expect("empty struct array");
        let mut reference = StructValue::new();
        reference
            .fields
            .insert("field".to_string(), Value::Num(1.0));

        let err = run_orderfields(Value::Cell(empty), vec![Value::Struct(reference)]).unwrap_err();
        assert_error_identifier(
            err,
            ORDERFIELDS_ERROR_EMPTY_STRUCT_ARRAY.identifier.unwrap(),
        );
    }
}
