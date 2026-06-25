//! MATLAB-compatible `getfield` builtin with struct array and object support.

use crate::builtins::common::indexing::perform_indexing;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::structs::type_resolvers::getfield_type;
use crate::make_cell_with_shape;
use crate::{
    build_runtime_error, call_builtin_async, gather_if_needed_async, object_property_getter_name,
    BuiltinResult, RuntimeError,
};
use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ComplexTensor, HandleRef, Listener, LogicalArray, MException,
    ObjectInstance, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::getfield")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "getfield",
    op_kind: GpuOpKind::Custom("getfield"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Pure metadata operation; acceleration providers do not participate.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::getfield")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "getfield",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion barrier because it inspects metadata on the host.",
};

const BUILTIN_NAME: &str = "getfield";
const GETFIELD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected field/property value.",
}];

const GETFIELD_INPUTS_SCALAR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Struct, struct array, object, or supported metadata container.",
    },
    BuiltinParamDescriptor {
        name: "field",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Field/property name.",
    },
];

const GETFIELD_INPUTS_NESTED: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Struct, struct array, object, or supported metadata container.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Alternating field names and optional index-selector cells `{...}` for nested access.",
    },
];

const GETFIELD_INPUTS_LEADING_INDEX: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Struct array or supported indexable container.",
    },
    BuiltinParamDescriptor {
        name: "index_selector",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Leading index selector in a cell array, e.g. `{2}` or `{end}`.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Alternating field names and optional index-selector cells `{...}` for nested access.",
    },
];

const GETFIELD_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "value = getfield(S, field)",
        inputs: &GETFIELD_INPUTS_SCALAR,
        outputs: &GETFIELD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = getfield(S, field_or_index, ...)",
        inputs: &GETFIELD_INPUTS_NESTED,
        outputs: &GETFIELD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = getfield(S, {idx0}, field_or_index, ...)",
        inputs: &GETFIELD_INPUTS_LEADING_INDEX,
        outputs: &GETFIELD_OUTPUT,
    },
];

const GETFIELD_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:getfield:NotEnoughInputs"),
    when: "No field-name/path arguments are supplied.",
    message: "getfield: expected at least one field name",
};

const GETFIELD_ERROR_FIELD_EXPECTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.FIELD_EXPECTED",
    identifier: Some("RunMat:getfield:FieldExpected"),
    when: "Field name is missing after indices or argument parsing.",
    message: "getfield: expected field name arguments",
};

const GETFIELD_ERROR_INDEX_SELECTOR_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INDEX_SELECTOR_TYPE",
    identifier: Some("RunMat:getfield:IndexSelectorType"),
    when: "Index selector is not provided as a cell array.",
    message: "getfield: indices must be provided in a cell array",
};

const GETFIELD_ERROR_INDEX_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INDEX_INVALID",
    identifier: Some("RunMat:getfield:InvalidIndex"),
    when: "Index components are malformed, empty, unsupported, or not positive integers.",
    message: "getfield: invalid index element",
};

const GETFIELD_ERROR_FIELD_NAME_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.FIELD_NAME_TYPE",
    identifier: Some("RunMat:getfield:FieldNameType"),
    when: "Field name is not a string scalar or 1-by-N char vector.",
    message: "getfield: expected field name",
};

const GETFIELD_ERROR_INDEX_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INDEX_SHAPE",
    identifier: Some("RunMat:getfield:IndexShape"),
    when: "Indexing rank/shape is unsupported for the targeted value.",
    message: "getfield: unsupported index shape for target value",
};

const GETFIELD_ERROR_NON_STRUCT_REFERENCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.NON_STRUCT_REFERENCE",
    identifier: Some("RunMat:getfield:NonStructReference"),
    when: "Field lookup is attempted on a non-struct/non-object container.",
    message: "Struct contents reference from a non-struct array object.",
};

const GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INDEX_OUT_OF_BOUNDS",
    identifier: Some("RunMat:getfield:IndexOutOfBounds"),
    when: "Resolved index is outside the bounds of the targeted value.",
    message: "Index exceeds the number of array elements.",
};

const GETFIELD_ERROR_MISSING_FIELD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.MISSING_FIELD",
    identifier: Some("RunMat:getfield:MissingField"),
    when: "Requested field does not exist on struct/exception/listener.",
    message: "Reference to non-existent field",
};

const GETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.PROPERTY_PRIVATE_ACCESS",
    identifier: Some("RunMat:PropertyPrivateAccess"),
    when: "Property exists but is private/inaccessible from this context.",
    message: "You cannot get this property from the current context.",
};

const GETFIELD_ERROR_OBJECT_PROPERTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.OBJECT_PROPERTY",
    identifier: Some("RunMat:getfield:ObjectProperty"),
    when: "Object property access is invalid (static-through-instance, unknown, or non-public).",
    message: "getfield: invalid object property access",
};

const GETFIELD_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INVALID_HANDLE",
    identifier: Some("RunMat:getfield:InvalidHandle"),
    when: "Handle object is invalid/deleted.",
    message: "Invalid or deleted handle object",
};

const GETFIELD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETFIELD.INTERNAL",
    identifier: Some("RunMat:getfield:InternalError"),
    when: "Internal value conversion or output assembly failed.",
    message: "getfield: internal error",
};

const GETFIELD_ERRORS: [BuiltinErrorDescriptor; 13] = [
    GETFIELD_ERROR_NOT_ENOUGH_INPUTS,
    GETFIELD_ERROR_FIELD_EXPECTED,
    GETFIELD_ERROR_INDEX_SELECTOR_TYPE,
    GETFIELD_ERROR_INDEX_INVALID,
    GETFIELD_ERROR_FIELD_NAME_TYPE,
    GETFIELD_ERROR_INDEX_SHAPE,
    GETFIELD_ERROR_NON_STRUCT_REFERENCE,
    GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
    GETFIELD_ERROR_MISSING_FIELD,
    GETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS,
    GETFIELD_ERROR_OBJECT_PROPERTY,
    GETFIELD_ERROR_INVALID_HANDLE,
    GETFIELD_ERROR_INTERNAL,
];

pub const GETFIELD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GETFIELD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GETFIELD_ERRORS,
};

fn getfield_flow(message: impl Into<String>) -> RuntimeError {
    getfield_error_with_message(message, &GETFIELD_ERROR_INTERNAL)
}

fn getfield_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    getfield_error_with_message(error.message, error)
}

fn getfield_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn getfield_private_access(message: impl Into<String>) -> RuntimeError {
    getfield_error_with_message(message, &GETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS)
}

fn remap_getfield_flow(err: RuntimeError, prefix: Option<&str>) -> RuntimeError {
    let mut message = err.message().to_string();
    if let Some(prefix) = prefix {
        if !message.starts_with(prefix) {
            message = format!("{prefix}{message}");
        }
    }
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

fn is_undefined_function(err: &RuntimeError) -> bool {
    err.identifier() == Some(crate::IDENT_UNDEFINED_FUNCTION)
}

#[runtime_builtin(
    name = "getfield",
    category = "structs/core",
    summary = "Access struct or object fields.",
    keywords = "getfield,struct,object,field access",
    type_resolver(getfield_type),
    descriptor(crate::builtins::structs::core::getfield::GETFIELD_DESCRIPTOR),
    builtin_path = "crate::builtins::structs::core::getfield"
)]
async fn getfield_builtin(base: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = parse_arguments(rest)?;

    let mut current = base;
    if let Some(index) = parsed.leading_index {
        current = apply_indices(current, &index).await?;
    }

    for step in parsed.fields {
        current = get_field_value(current, &step.name).await?;
        if let Some(index) = step.index {
            current = apply_indices(current, &index).await?;
        }
    }

    Ok(current)
}

#[derive(Default)]
struct ParsedArguments {
    leading_index: Option<IndexSelector>,
    fields: Vec<FieldStep>,
}

struct FieldStep {
    name: String,
    index: Option<IndexSelector>,
}

#[derive(Clone)]
struct IndexSelector {
    components: Vec<IndexComponent>,
}

#[derive(Clone)]
enum IndexComponent {
    Scalar(usize),
    Vector(Vec<usize>, Vec<usize>),
    End,
}

fn parse_arguments(mut rest: Vec<Value>) -> BuiltinResult<ParsedArguments> {
    if rest.is_empty() {
        return Err(getfield_error(&GETFIELD_ERROR_NOT_ENOUGH_INPUTS));
    }

    let mut parsed = ParsedArguments::default();
    if let Some(first) = rest.first() {
        if is_index_selector(first) {
            let value = rest.remove(0);
            parsed.leading_index = Some(parse_index_selector(value)?);
        }
    }

    if rest.is_empty() {
        return Err(getfield_error_with_message(
            "getfield: expected field name after indices",
            &GETFIELD_ERROR_FIELD_EXPECTED,
        ));
    }

    let mut iter = rest.into_iter().peekable();
    while let Some(arg) = iter.next() {
        let field_name = parse_field_name(arg)?;
        let mut step = FieldStep {
            name: field_name,
            index: None,
        };
        if let Some(next) = iter.peek() {
            if is_index_selector(next) {
                let selector = iter.next().unwrap();
                step.index = Some(parse_index_selector(selector)?);
            }
        }
        parsed.fields.push(step);
    }

    if parsed.fields.is_empty() {
        return Err(getfield_error(&GETFIELD_ERROR_FIELD_EXPECTED));
    }

    Ok(parsed)
}

fn is_index_selector(value: &Value) -> bool {
    matches!(value, Value::Cell(_))
}

fn parse_index_selector(value: Value) -> BuiltinResult<IndexSelector> {
    let Value::Cell(cell) = value else {
        return Err(getfield_error(&GETFIELD_ERROR_INDEX_SELECTOR_TYPE));
    };

    let mut components = Vec::with_capacity(cell.data.len());
    for handle in &cell.data {
        let entry = handle;
        components.push(parse_index_component(entry)?);
    }

    Ok(IndexSelector { components })
}

fn parse_index_component(value: &Value) -> BuiltinResult<IndexComponent> {
    match value {
        Value::CharArray(ca) => {
            let text: String = ca.data.iter().collect();
            parse_index_text(text.trim())
        }
        Value::String(s) => parse_index_text(s.trim()),
        Value::StringArray(sa) if sa.data.len() == 1 => parse_index_text(sa.data[0].trim()),
        Value::Tensor(tensor) if tensor.data.len() > 1 => {
            let indices = tensor
                .data
                .iter()
                .map(|&value| parse_positive_integer(value))
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(IndexComponent::Vector(indices, tensor.shape.clone()))
        }
        _ => {
            let idx = parse_positive_scalar(value).map_err(|err| {
                getfield_error_with_message(
                    format!("getfield: invalid index element ({})", err.message()),
                    &GETFIELD_ERROR_INDEX_INVALID,
                )
            })?;
            Ok(IndexComponent::Scalar(idx))
        }
    }
}

fn parse_index_text(text: &str) -> BuiltinResult<IndexComponent> {
    if text.eq_ignore_ascii_case("end") {
        return Ok(IndexComponent::End);
    }
    if text == ":" {
        return Err(getfield_error_with_message(
            "getfield: ':' indexing is not currently supported",
            &GETFIELD_ERROR_INDEX_INVALID,
        ));
    }
    if text.is_empty() {
        return Err(getfield_error_with_message(
            "getfield: index elements must not be empty",
            &GETFIELD_ERROR_INDEX_INVALID,
        ));
    }
    if let Ok(value) = text.parse::<usize>() {
        if value == 0 {
            return Err(getfield_error_with_message(
                "getfield: index must be >= 1",
                &GETFIELD_ERROR_INDEX_INVALID,
            ));
        }
        return Ok(IndexComponent::Scalar(value));
    }
    Err(getfield_error_with_message(
        format!("getfield: invalid index element '{}'", text),
        &GETFIELD_ERROR_INDEX_INVALID,
    ))
}

fn parse_positive_scalar(value: &Value) -> BuiltinResult<usize> {
    let number = match value {
        Value::Int(i) => i.to_i64() as f64,
        Value::Num(n) => *n,
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        _ => {
            let repr = format!("{value:?}");
            return Err(getfield_error_with_message(
                format!("expected positive integer index, got {repr}"),
                &GETFIELD_ERROR_INDEX_INVALID,
            ));
        }
    };

    parse_positive_integer(number)
}

fn parse_positive_integer(number: f64) -> BuiltinResult<usize> {
    if !number.is_finite() {
        return Err(getfield_error_with_message(
            "index must be a finite number",
            &GETFIELD_ERROR_INDEX_INVALID,
        ));
    }
    if number.fract() != 0.0 {
        return Err(getfield_error_with_message(
            "index must be an integer",
            &GETFIELD_ERROR_INDEX_INVALID,
        ));
    }
    if number <= 0.0 {
        return Err(getfield_error_with_message(
            "index must be >= 1",
            &GETFIELD_ERROR_INDEX_INVALID,
        ));
    }
    if number > usize::MAX as f64 {
        return Err(getfield_error_with_message(
            "index exceeds platform limits",
            &GETFIELD_ERROR_INDEX_INVALID,
        ));
    }
    Ok(number as usize)
}

fn parse_field_name(value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(getfield_error_with_message(
                    "getfield: field names must be scalar string arrays or character vectors",
                    &GETFIELD_ERROR_FIELD_NAME_TYPE,
                ))
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                Ok(ca.data.iter().collect())
            } else {
                Err(getfield_error_with_message(
                    "getfield: field names must be 1-by-N character vectors",
                    &GETFIELD_ERROR_FIELD_NAME_TYPE,
                ))
            }
        }
        other => Err(getfield_error_with_message(
            format!("getfield: expected field name, got {other:?}"),
            &GETFIELD_ERROR_FIELD_NAME_TYPE,
        )),
    }
}

async fn apply_indices(value: Value, selector: &IndexSelector) -> BuiltinResult<Value> {
    if selector.components.is_empty() {
        return Err(getfield_error_with_message(
            "getfield: index cell must contain at least one element",
            &GETFIELD_ERROR_INDEX_SELECTOR_TYPE,
        ));
    }

    let value = match value {
        Value::GpuTensor(handle) => gather_if_needed_async(&Value::GpuTensor(handle))
            .await
            .map_err(|flow| remap_getfield_flow(flow, Some("getfield: ")))?,
        other => other,
    };

    if let Some(indexed) = apply_vector_index(&value, selector)? {
        return Ok(indexed);
    }

    let resolved = resolve_indices(&value, selector)?;
    let resolved_f64: Vec<f64> = resolved.iter().map(|&idx| idx as f64).collect();

    match &value {
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)
                .map_err(|e| getfield_flow(format!("getfield: {e}")))?;
            let scratch = Value::Tensor(tensor);
            let indexed = perform_indexing(&scratch, &resolved_f64)
                .await
                .map_err(|err| remap_getfield_flow(err, Some("getfield: ")))?;
            match indexed {
                Value::Num(n) => Ok(Value::Bool(n != 0.0)),
                Value::Tensor(t) => {
                    let bits: Vec<u8> = t
                        .data
                        .iter()
                        .map(|&v| if v != 0.0 { 1 } else { 0 })
                        .collect();
                    let logical = LogicalArray::new(bits, t.shape.clone())
                        .map_err(|e| getfield_flow(format!("getfield: {e}")))?;
                    Ok(Value::LogicalArray(logical))
                }
                other => Ok(other),
            }
        }
        Value::CharArray(array) => index_char_array(array, &resolved),
        Value::ComplexTensor(tensor) => index_complex_tensor(tensor, &resolved),
        Value::Tensor(_)
        | Value::StringArray(_)
        | Value::Cell(_)
        | Value::Num(_)
        | Value::Int(_) => perform_indexing(&value, &resolved_f64)
            .await
            .map_err(|err| remap_getfield_flow(err, Some("getfield: "))),
        Value::Bool(_) => {
            if resolved.len() == 1 && resolved[0] == 1 {
                Ok(value)
            } else {
                Err(getfield_error(&GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS))
            }
        }
        _ => Err(getfield_error(&GETFIELD_ERROR_NON_STRUCT_REFERENCE)),
    }
}

fn apply_vector_index(value: &Value, selector: &IndexSelector) -> BuiltinResult<Option<Value>> {
    let [IndexComponent::Vector(indices, shape)] = selector.components.as_slice() else {
        return Ok(None);
    };
    match value {
        Value::Tensor(tensor) => {
            let mut data = Vec::with_capacity(indices.len());
            for &index in indices {
                if index < 1 || index > tensor.data.len() {
                    return Err(getfield_error(&GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS));
                }
                data.push(tensor.data[index - 1]);
            }
            let tensor = Tensor::new(data, shape.clone())
                .map_err(|e| getfield_flow(format!("getfield: {e}")))?;
            Ok(Some(Value::Tensor(tensor)))
        }
        _ => Ok(None),
    }
}

fn resolve_indices(value: &Value, selector: &IndexSelector) -> BuiltinResult<Vec<usize>> {
    let dims = selector.components.len();
    let mut resolved = Vec::with_capacity(dims);
    for (dim_idx, component) in selector.components.iter().enumerate() {
        let index = match component {
            IndexComponent::Scalar(idx) => *idx,
            IndexComponent::Vector(_, _) => {
                return Err(getfield_error_with_message(
                    "getfield: vector indices are only supported for one-dimensional indexing",
                    &GETFIELD_ERROR_INDEX_SHAPE,
                ))
            }
            IndexComponent::End => dimension_length(value, dims, dim_idx)?,
        };
        resolved.push(index);
    }
    Ok(resolved)
}

fn dimension_length(value: &Value, dims: usize, dim_idx: usize) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) => tensor_dimension_length(tensor, dims, dim_idx),
        Value::Cell(cell) => cell_dimension_length(cell, dims, dim_idx),
        Value::StringArray(sa) => string_array_dimension_length(sa, dims, dim_idx),
        Value::LogicalArray(logical) => logical_array_dimension_length(logical, dims, dim_idx),
        Value::CharArray(array) => char_array_dimension_length(array, dims, dim_idx),
        Value::ComplexTensor(tensor) => complex_tensor_dimension_length(tensor, dims, dim_idx),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            if dims == 1 {
                Ok(1)
            } else {
                Err(getfield_error_with_message(
                    "getfield: indexing with more than one dimension is not supported for scalars",
                    &GETFIELD_ERROR_INDEX_SHAPE,
                ))
            }
        }
        _ => Err(getfield_error(&GETFIELD_ERROR_NON_STRUCT_REFERENCE)),
    }
}

fn tensor_dimension_length(tensor: &Tensor, dims: usize, dim_idx: usize) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = tensor.data.len();
        if total == 0 {
            return Err(getfield_error_with_message(
                "Index exceeds the number of array elements (0).",
                &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_error_with_message(
            "getfield: indexing with more than two indices is not supported yet",
            &GETFIELD_ERROR_INDEX_SHAPE,
        ));
    }
    let len = if dim_idx == 0 {
        tensor.rows()
    } else {
        tensor.cols()
    };
    if len == 0 {
        return Err(getfield_error_with_message(
            "Index exceeds the number of array elements (0).",
            &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
        ));
    }
    Ok(len)
}

fn cell_dimension_length(cell: &CellArray, dims: usize, dim_idx: usize) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = cell.data.len();
        if total == 0 {
            return Err(getfield_error_with_message(
                "Index exceeds the number of array elements (0).",
                &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_error_with_message(
            "getfield: indexing with more than two indices is not supported yet",
            &GETFIELD_ERROR_INDEX_SHAPE,
        ));
    }
    let len = if dim_idx == 0 { cell.rows } else { cell.cols };
    if len == 0 {
        return Err(getfield_error_with_message(
            "Index exceeds the number of array elements (0).",
            &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
        ));
    }
    Ok(len)
}

fn string_array_dimension_length(
    array: &runmat_builtins::StringArray,
    dims: usize,
    dim_idx: usize,
) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = array.data.len();
        if total == 0 {
            return Err(getfield_error_with_message(
                "Index exceeds the number of array elements (0).",
                &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_error_with_message(
            "getfield: indexing with more than two indices is not supported yet",
            &GETFIELD_ERROR_INDEX_SHAPE,
        ));
    }
    let len = if dim_idx == 0 {
        array.rows()
    } else {
        array.cols()
    };
    if len == 0 {
        return Err(getfield_error_with_message(
            "Index exceeds the number of array elements (0).",
            &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
        ));
    }
    Ok(len)
}

fn logical_array_dimension_length(
    logical: &LogicalArray,
    dims: usize,
    dim_idx: usize,
) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = logical.data.len();
        if total == 0 {
            return Err(getfield_error_with_message(
                "Index exceeds the number of array elements (0).",
                &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_error_with_message(
            "getfield: indexing with more than two indices is not supported yet",
            &GETFIELD_ERROR_INDEX_SHAPE,
        ));
    }
    let len = if dim_idx == 0 {
        logical.shape.first().copied().unwrap_or(logical.data.len())
    } else {
        logical.shape.get(1).copied().unwrap_or(1)
    };
    if len == 0 {
        return Err(getfield_error_with_message(
            "Index exceeds the number of array elements (0).",
            &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
        ));
    }
    Ok(len)
}

fn char_array_dimension_length(
    array: &CharArray,
    dims: usize,
    dim_idx: usize,
) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = array.rows * array.cols;
        if total == 0 {
            return Err(getfield_error_with_message(
                "Index exceeds the number of array elements (0).",
                &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_error_with_message(
            "getfield: indexing with more than two indices is not supported yet",
            &GETFIELD_ERROR_INDEX_SHAPE,
        ));
    }
    let len = if dim_idx == 0 { array.rows } else { array.cols };
    if len == 0 {
        return Err(getfield_error_with_message(
            "Index exceeds the number of array elements (0).",
            &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
        ));
    }
    Ok(len)
}

fn complex_tensor_dimension_length(
    tensor: &ComplexTensor,
    dims: usize,
    dim_idx: usize,
) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = tensor.data.len();
        if total == 0 {
            return Err(getfield_error_with_message(
                "Index exceeds the number of array elements (0).",
                &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_error_with_message(
            "getfield: indexing with more than two indices is not supported yet",
            &GETFIELD_ERROR_INDEX_SHAPE,
        ));
    }
    let len = if dim_idx == 0 {
        tensor.rows
    } else {
        tensor.cols
    };
    if len == 0 {
        return Err(getfield_error_with_message(
            "Index exceeds the number of array elements (0).",
            &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
        ));
    }
    Ok(len)
}

fn index_char_array(array: &CharArray, indices: &[usize]) -> BuiltinResult<Value> {
    if indices.is_empty() {
        return Err(getfield_error_with_message(
            "getfield: at least one index is required for char arrays",
            &GETFIELD_ERROR_INDEX_INVALID,
        ));
    }
    if indices.len() == 1 {
        let total = array.rows * array.cols;
        let idx = indices[0];
        if idx == 0 || idx > total {
            return Err(getfield_error(&GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS));
        }
        let linear = idx - 1;
        let rows = array.rows.max(1);
        let col = linear / rows;
        let row = linear % rows;
        let pos = row * array.cols + col;
        let ch = array
            .data
            .get(pos)
            .copied()
            .ok_or_else(|| getfield_error(&GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS))?;
        let out =
            CharArray::new(vec![ch], 1, 1).map_err(|e| getfield_flow(format!("getfield: {e}")))?;
        return Ok(Value::CharArray(out));
    }
    if indices.len() == 2 {
        let row = indices[0];
        let col = indices[1];
        if row == 0 || row > array.rows || col == 0 || col > array.cols {
            return Err(getfield_error(&GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS));
        }
        let pos = (row - 1) * array.cols + (col - 1);
        let ch = array
            .data
            .get(pos)
            .copied()
            .ok_or_else(|| getfield_error(&GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS))?;
        let out =
            CharArray::new(vec![ch], 1, 1).map_err(|e| getfield_flow(format!("getfield: {e}")))?;
        return Ok(Value::CharArray(out));
    }
    Err(getfield_error_with_message(
        "getfield: indexing with more than two indices is not supported for char arrays",
        &GETFIELD_ERROR_INDEX_SHAPE,
    ))
}

fn index_complex_tensor(tensor: &ComplexTensor, indices: &[usize]) -> BuiltinResult<Value> {
    if indices.is_empty() {
        return Err(getfield_error_with_message(
            "getfield: at least one index is required for complex tensors",
            &GETFIELD_ERROR_INDEX_INVALID,
        ));
    }
    if indices.len() == 1 {
        let total = tensor.data.len();
        let idx = indices[0];
        if idx == 0 || idx > total {
            return Err(getfield_error(&GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS));
        }
        let (re, im) = tensor.data[idx - 1];
        return Ok(Value::Complex(re, im));
    }
    if indices.len() == 2 {
        let row = indices[0];
        let col = indices[1];
        if row == 0 || row > tensor.rows || col == 0 || col > tensor.cols {
            return Err(getfield_error(&GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS));
        }
        let pos = (row - 1) + (col - 1) * tensor.rows;
        let (re, im) = tensor
            .data
            .get(pos)
            .copied()
            .ok_or_else(|| getfield_error(&GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS))?;
        return Ok(Value::Complex(re, im));
    }
    Err(getfield_error_with_message(
        "getfield: indexing with more than two indices is not supported for complex tensors",
        &GETFIELD_ERROR_INDEX_SHAPE,
    ))
}

#[async_recursion::async_recursion(?Send)]
async fn get_field_value(value: Value, name: &str) -> BuiltinResult<Value> {
    match value {
        Value::Struct(st) => get_struct_field(&st, name),
        Value::Object(obj) => get_object_field(&obj, name).await,
        Value::HandleObject(handle) => get_handle_field(&handle, name).await,
        Value::Listener(listener) => get_listener_field(&listener, name),
        Value::MException(ex) => get_exception_field(&ex, name),
        Value::Cell(cell) if is_struct_array(&cell) => {
            if cell.data.is_empty() {
                return Err(getfield_error_with_message(
                    "Struct contents reference from an empty struct array.",
                    &GETFIELD_ERROR_NON_STRUCT_REFERENCE,
                ));
            }
            // Default to first element when no index is specified
            let first_entry = &cell.data[0];
            match first_entry {
                Value::Struct(st) => get_struct_field(st, name),
                _ => Err(getfield_error(&GETFIELD_ERROR_NON_STRUCT_REFERENCE)),
            }
        }
        _ => Err(getfield_error(&GETFIELD_ERROR_NON_STRUCT_REFERENCE)),
    }
}

fn get_struct_field(struct_value: &StructValue, name: &str) -> BuiltinResult<Value> {
    struct_value.fields.get(name).cloned().ok_or_else(|| {
        getfield_error_with_message(
            format!("Reference to non-existent field '{}'.", name),
            &GETFIELD_ERROR_MISSING_FIELD,
        )
    })
}

async fn get_object_field(obj: &ObjectInstance, name: &str) -> BuiltinResult<Value> {
    if let Some((prop, _owner)) = runmat_builtins::lookup_property(&obj.class_name, name) {
        if prop.is_static {
            return Err(getfield_error_with_message(
                format!(
                    "You cannot access the static property '{}' through an instance of class '{}'.",
                    name, obj.class_name
                ),
                &GETFIELD_ERROR_OBJECT_PROPERTY,
            ));
        }
        if prop.get_access == Access::Private {
            return Err(getfield_private_access(format!(
                "You cannot get the '{}' property of '{}' class.",
                name, obj.class_name
            )));
        }
        if prop.is_dependent {
            let getter = object_property_getter_name(name);
            match call_builtin_async(&getter, &[Value::Object(obj.clone())]).await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if !is_undefined_function(&err) {
                        return Err(remap_getfield_flow(err, None));
                    }
                }
            }
            if let Some(val) = obj.properties.get(&format!("{name}_backing")) {
                return Ok(val.clone());
            }
        }
    }

    if let Some(value) = obj.properties.get(name) {
        return Ok(value.clone());
    }

    if let Some((prop, _owner)) = runmat_builtins::lookup_property(&obj.class_name, name) {
        if prop.get_access == Access::Private {
            return Err(getfield_private_access(format!(
                "You cannot get the '{}' property of '{}' class.",
                name, obj.class_name
            )));
        }
        return Err(getfield_error_with_message(
            format!(
                "No public property '{}' for class '{}'.",
                name, obj.class_name
            ),
            &GETFIELD_ERROR_OBJECT_PROPERTY,
        ));
    }

    Err(getfield_error_with_message(
        format!("Undefined property '{}' for class {}", name, obj.class_name),
        &GETFIELD_ERROR_OBJECT_PROPERTY,
    ))
}

#[async_recursion::async_recursion(?Send)]
async fn get_handle_field(handle: &HandleRef, name: &str) -> BuiltinResult<Value> {
    if !crate::is_handle_valid(handle) {
        return Err(getfield_error_with_message(
            format!("Invalid or deleted handle object '{}'.", handle.class_name),
            &GETFIELD_ERROR_INVALID_HANDLE,
        ));
    }
    let target = runmat_gc::gc_clone_value(&handle.target).map_err(|e| {
        getfield_error_with_message(
            format!("getfield: invalid handle target: {e}"),
            &GETFIELD_ERROR_INVALID_HANDLE,
        )
    })?;
    get_field_value(target, name).await
}

fn get_listener_field(listener: &Listener, name: &str) -> BuiltinResult<Value> {
    match name {
        "Enabled" | "enabled" => Ok(Value::Bool(listener.enabled)),
        "Valid" | "valid" => Ok(Value::Bool(listener.valid)),
        "EventName" | "event_name" => Ok(Value::String(listener.event_name.clone())),
        "Callback" | "callback" => {
            if !listener.valid {
                return Err(getfield_error_with_message(
                    "getfield: listener is invalid or deleted",
                    &GETFIELD_ERROR_INVALID_HANDLE,
                ));
            }
            let value = runmat_gc::gc_clone_value(&listener.callback).map_err(|e| {
                getfield_error_with_message(
                    format!("getfield: invalid listener callback: {e}"),
                    &GETFIELD_ERROR_INVALID_HANDLE,
                )
            })?;
            Ok(value)
        }
        "Target" | "target" => {
            if !listener.valid {
                return Err(getfield_error_with_message(
                    "getfield: listener is invalid or deleted",
                    &GETFIELD_ERROR_INVALID_HANDLE,
                ));
            }
            let value = runmat_gc::gc_clone_value(&listener.target).map_err(|e| {
                getfield_error_with_message(
                    format!("getfield: invalid listener target: {e}"),
                    &GETFIELD_ERROR_INVALID_HANDLE,
                )
            })?;
            Ok(value)
        }
        "Id" | "id" => Ok(Value::Int(runmat_builtins::IntValue::U64(listener.id))),
        other => Err(getfield_error_with_message(
            format!("getfield: unknown field '{}' on listener object", other),
            &GETFIELD_ERROR_MISSING_FIELD,
        )),
    }
}

fn get_exception_field(exception: &MException, name: &str) -> BuiltinResult<Value> {
    match name {
        "message" => Ok(Value::String(exception.message.clone())),
        "identifier" => Ok(Value::String(exception.identifier.clone())),
        "stack" => exception_stack_to_value(&exception.stack),
        other => Err(getfield_error_with_message(
            format!("Reference to non-existent field '{}'.", other),
            &GETFIELD_ERROR_MISSING_FIELD,
        )),
    }
}

fn exception_stack_to_value(stack: &[String]) -> BuiltinResult<Value> {
    if stack.is_empty() {
        return make_cell_with_shape(Vec::new(), vec![0, 1])
            .map_err(|e| getfield_flow(format!("getfield: {e}")));
    }
    let mut values = Vec::with_capacity(stack.len());
    for frame in stack {
        values.push(Value::String(frame.clone()));
    }
    make_cell_with_shape(values, vec![stack.len(), 1])
        .map_err(|e| getfield_flow(format!("getfield: {e}")))
}

fn is_struct_array(cell: &CellArray) -> bool {
    cell.data
        .iter()
        .all(|handle| matches!(handle, Value::Struct(_)))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{
        Access, CellArray, CharArray, ClassDef, ComplexTensor, HandleRef, IntValue, Listener,
        MException, ObjectInstance, PropertyDef, StructValue,
    };

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_getfield(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(getfield_builtin(base, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_scalar_struct() {
        let mut st = StructValue::new();
        st.fields.insert("answer".to_string(), Value::Num(42.0));
        let value = run_getfield(Value::Struct(st), vec![Value::from("answer")]).expect("getfield");
        assert_eq!(value, Value::Num(42.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_nested_structs() {
        let mut inner = StructValue::new();
        inner.fields.insert("depth".to_string(), Value::Num(3.0));
        let mut outer = StructValue::new();
        outer
            .fields
            .insert("inner".to_string(), Value::Struct(inner));
        let result = run_getfield(
            Value::Struct(outer),
            vec![Value::from("inner"), Value::from("depth")],
        )
        .expect("nested getfield");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_struct_array_element() {
        let mut first = StructValue::new();
        first.fields.insert("name".to_string(), Value::from("Ada"));
        let mut second = StructValue::new();
        second
            .fields
            .insert("name".to_string(), Value::from("Grace"));
        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .unwrap();
        let index =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
        let result = run_getfield(
            Value::Cell(array),
            vec![Value::Cell(index), Value::from("name")],
        )
        .expect("struct array element");
        assert_eq!(result, Value::from("Grace"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_object_property() {
        let mut obj = ObjectInstance::new("TestClass".to_string());
        obj.properties.insert("value".to_string(), Value::Num(7.0));
        let result = run_getfield(Value::Object(obj), vec![Value::from("value")]).expect("object");
        assert_eq!(result, Value::Num(7.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_missing_field_errors() {
        let st = StructValue::new();
        let err = error_message(
            run_getfield(Value::Struct(st), vec![Value::from("missing")]).unwrap_err(),
        );
        assert!(err.contains("Reference to non-existent field 'missing'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_exception_fields() {
        let ex = MException::new("RunMat:Test".to_string(), "failure".to_string());
        let msg = run_getfield(Value::MException(ex.clone()), vec![Value::from("message")])
            .expect("message");
        assert_eq!(msg, Value::String("failure".to_string()));
        let ident = run_getfield(Value::MException(ex), vec![Value::from("identifier")])
            .expect("identifier");
        assert_eq!(ident, Value::String("RunMat:Test".to_string()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_exception_stack_cell() {
        let mut ex = MException::new("RunMat:Test".to_string(), "failure".to_string());
        ex.stack.push("demo.m:5".to_string());
        ex.stack.push("main.m:1".to_string());
        let stack = run_getfield(Value::MException(ex), vec![Value::from("stack")]).expect("stack");
        let Value::Cell(cell) = stack else {
            panic!("expected cell array");
        };
        assert_eq!(cell.rows, 2);
        assert_eq!(cell.cols, 1);
        let first = cell.data[0].clone();
        assert_eq!(first, Value::String("demo.m:5".to_string()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn indexing_missing_field_name_fails() {
        let mut outer = StructValue::new();
        outer.fields.insert("inner".to_string(), Value::Num(1.0));
        let index =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(1))], vec![1, 1]).unwrap();
        let err = error_message(
            run_getfield(Value::Struct(outer), vec![Value::Cell(index)]).unwrap_err(),
        );
        assert!(err.contains("expected field name"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_supports_end_index() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let mut st = StructValue::new();
        st.fields
            .insert("values".to_string(), Value::Tensor(tensor));
        let idx_cell =
            CellArray::new(vec![Value::CharArray(CharArray::new_row("end"))], 1, 1).unwrap();
        let result = run_getfield(
            Value::Struct(st),
            vec![Value::from("values"), Value::Cell(idx_cell)],
        )
        .expect("end index");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_struct_array_defaults_to_first() {
        let mut first = StructValue::new();
        first.fields.insert("name".to_string(), Value::from("Ada"));
        let mut second = StructValue::new();
        second
            .fields
            .insert("name".to_string(), Value::from("Grace"));
        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .unwrap();
        let result =
            run_getfield(Value::Cell(array), vec![Value::from("name")]).expect("default index");
        assert_eq!(result, Value::from("Ada"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_char_array_single_element() {
        let chars = CharArray::new_row("Ada");
        let mut st = StructValue::new();
        st.fields
            .insert("name".to_string(), Value::CharArray(chars));
        let index =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
        let result = run_getfield(
            Value::Struct(st),
            vec![Value::from("name"), Value::Cell(index)],
        )
        .expect("char indexing");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 1);
                assert_eq!(ca.data, vec!['d']);
            }
            other => panic!("expected 1x1 CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_complex_tensor_index() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2, 1]).expect("complex tensor");
        let mut st = StructValue::new();
        st.fields
            .insert("vals".to_string(), Value::ComplexTensor(tensor));
        let index =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
        let result = run_getfield(
            Value::Struct(st),
            vec![Value::from("vals"), Value::Cell(index)],
        )
        .expect("complex index");
        assert_eq!(result, Value::Complex(3.0, 4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_dependent_property_invokes_getter() {
        let class_name = "runmat.unittest.GetfieldDependent";
        let mut def = ClassDef {
            name: class_name.to_string(),
            parent: None,
            properties: std::collections::HashMap::new(),
            methods: std::collections::HashMap::new(),
        };
        def.properties.insert(
            "p".to_string(),
            PropertyDef {
                name: "p".to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: true,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
        runmat_builtins::register_class(def);

        let mut obj = ObjectInstance::new(class_name.to_string());
        obj.properties
            .insert("p_backing".to_string(), Value::Num(42.0));

        let result = run_getfield(Value::Object(obj), vec![Value::from("p")]).expect("dependent");
        assert_eq!(result, Value::Num(42.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_inherited_dependent_property_uses_parent_metadata() {
        let parent_name = "runmat.unittest.GetfieldDependentParent";
        let child_name = "runmat.unittest.GetfieldDependentChild";

        let mut parent = ClassDef {
            name: parent_name.to_string(),
            parent: None,
            properties: std::collections::HashMap::new(),
            methods: std::collections::HashMap::new(),
        };
        parent.properties.insert(
            "p".to_string(),
            PropertyDef {
                name: "p".to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: true,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
        runmat_builtins::register_class(parent);

        runmat_builtins::register_class(ClassDef {
            name: child_name.to_string(),
            parent: Some(parent_name.to_string()),
            properties: std::collections::HashMap::new(),
            methods: std::collections::HashMap::new(),
        });

        let mut obj = ObjectInstance::new(child_name.to_string());
        obj.properties
            .insert("p_backing".to_string(), Value::Num(17.0));

        let result =
            run_getfield(Value::Object(obj), vec![Value::from("p")]).expect("inherited dependent");
        assert_eq!(result, Value::Num(17.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_invalid_handle_errors() {
        let target = runmat_gc::gc_allocate(Value::Num(1.0)).expect("gc allocate target");
        let handle = HandleRef {
            class_name: "Demo".to_string(),
            target,
            valid: false,
        };
        let err = error_message(
            run_getfield(Value::HandleObject(handle), vec![Value::from("x")]).unwrap_err(),
        );
        assert!(err.contains("Invalid or deleted handle object 'Demo'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_listener_fields_resolved() {
        let target = runmat_gc::gc_allocate(Value::Num(7.0)).expect("gc allocate target");
        let callback = runmat_gc::gc_allocate(Value::FunctionHandle("cb".to_string()))
            .expect("gc allocate callback");
        let listener = Listener {
            id: 9,
            target,
            target_class_name: "EventTarget".to_string(),
            event_name: "tick".to_string(),
            callback,
            enabled: true,
            valid: true,
        };
        let enabled = run_getfield(
            Value::Listener(listener.clone()),
            vec![Value::from("Enabled")],
        )
        .expect("enabled");
        assert_eq!(enabled, Value::Bool(true));
        let event_name = run_getfield(
            Value::Listener(listener.clone()),
            vec![Value::from("EventName")],
        )
        .expect("event name");
        assert_eq!(event_name, Value::String("tick".to_string()));
        let callback = run_getfield(Value::Listener(listener), vec![Value::from("Callback")])
            .expect("callback");
        assert!(matches!(callback, Value::FunctionHandle(_)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_invalid_listener_rejects_rooted_fields() {
        let target = runmat_gc::gc_allocate(Value::Num(7.0)).expect("gc allocate target");
        let callback = runmat_gc::gc_allocate(Value::FunctionHandle("cb".to_string()))
            .expect("gc allocate callback");
        let listener = Listener {
            id: 10,
            target,
            target_class_name: "EventTarget".to_string(),
            event_name: "tick".to_string(),
            callback,
            enabled: false,
            valid: false,
        };

        let err = error_message(
            run_getfield(
                Value::Listener(listener.clone()),
                vec![Value::from("Callback")],
            )
            .unwrap_err(),
        );
        assert!(err.contains("listener is invalid or deleted"));

        let err = error_message(
            run_getfield(Value::Listener(listener), vec![Value::from("Target")]).unwrap_err(),
        );
        assert!(err.contains("listener is invalid or deleted"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn getfield_gpu_tensor_indexing() {
        let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let mut st = StructValue::new();
        st.fields
            .insert("values".to_string(), Value::GpuTensor(handle.clone()));

        let direct = run_getfield(Value::Struct(st.clone()), vec![Value::from("values")])
            .expect("direct gpu field");
        match direct {
            Value::GpuTensor(out) => assert_eq!(out.buffer_id, handle.buffer_id),
            other => panic!("expected gpu tensor, got {other:?}"),
        }

        let idx_cell =
            CellArray::new(vec![Value::CharArray(CharArray::new_row("end"))], 1, 1).unwrap();
        let indexed = run_getfield(
            Value::Struct(st),
            vec![Value::from("values"), Value::Cell(idx_cell)],
        )
        .expect("gpu indexed field");
        match indexed {
            Value::Num(v) => assert_eq!(v, 3.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_undefined_detection_requires_identifier() {
        let with_identifier = build_runtime_error("missing")
            .with_identifier(crate::IDENT_UNDEFINED_FUNCTION)
            .build();
        assert!(is_undefined_function(&with_identifier));

        let message_only =
            build_runtime_error(format!("{} message only", crate::IDENT_UNDEFINED_FUNCTION))
                .build();
        assert!(
            !is_undefined_function(&message_only),
            "message-only undefined markers should not trigger getter fallback"
        );
    }
}
