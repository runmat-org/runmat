use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericScalar, Value,
};
use runmat_macros::runtime_builtin;
use std::sync::OnceLock;

pub(crate) const NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD: &str = "numArgumentsFromSubscript";
pub(crate) const INDEXING_CONTEXT_CLASS: &str = "matlab.indexing.IndexingContext";
pub(crate) const LEGACY_INDEXING_CONTEXT_CLASS: &str = "matlab.mixin.util.IndexingContext";
const INDEXING_CONTEXT_STATEMENT: &str = "Statement";
const INDEXING_CONTEXT_EXPRESSION: &str = "Expression";
const INDEXING_CONTEXT_ASSIGNMENT: &str = "Assignment";
const ENUM_MEMBER_PROPERTY: &str = "__enum_member__";

static INDEXING_CONTEXT_REGISTERED: OnceLock<()> = OnceLock::new();

const SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed value.",
}];

const SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token ('()', '{}', '.').",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
];

const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = subsref(obj, kind, payload)",
    inputs: &SUBSREF_INPUTS,
    outputs: &SUBSREF_OUTPUT,
}];

const SUBSREF_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSREF.RECEIVER_INVALID",
    identifier: Some("RunMat:InvalidObjectDispatch"),
    when: "Receiver is not an object or handle object.",
    message: "subsref: requires object receiver",
};

const SUBSREF_ERROR_METHOD_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSREF.METHOD_MISSING",
    identifier: Some("RunMat:MissingSubsref"),
    when: "Target class does not implement subsref.",
    message: "subsref: class does not define subsref for indexing operation",
};

const SUBSREF_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SUBSREF_ERROR_RECEIVER_INVALID, SUBSREF_ERROR_METHOD_MISSING];

pub const SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUBSREF_ERRORS,
};

const SUBSASGN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated object value.",
}];

const SUBSASGN_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token ('()', '{}', '.').",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];

const SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = subsasgn(obj, kind, payload, rhs)",
    inputs: &SUBSASGN_INPUTS,
    outputs: &SUBSASGN_OUTPUT,
}];

const SUBSASGN_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSASGN.RECEIVER_INVALID",
    identifier: Some("RunMat:InvalidObjectDispatch"),
    when: "Receiver is not an object or handle object.",
    message: "subsasgn: requires object receiver",
};

const SUBSASGN_ERROR_METHOD_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBSASGN.METHOD_MISSING",
    identifier: Some("RunMat:MissingSubsasgn"),
    when: "Target class does not implement subsasgn.",
    message: "subsasgn: class does not define subsasgn for indexed assignment",
};

const SUBSASGN_ERRORS: [BuiltinErrorDescriptor; 2] = [
    SUBSASGN_ERROR_RECEIVER_INVALID,
    SUBSASGN_ERROR_METHOD_MISSING,
];

pub const SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUBSASGN_ERRORS,
};

const NUM_ARGUMENTS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of expected subsref outputs or subsasgn inputs.",
}];

const NUM_ARGUMENTS_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target of the indexing expression.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Substruct-compatible indexing descriptor.",
    },
    BuiltinParamDescriptor {
        name: "indexingContext",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "IndexingContext enum object or compatible text token.",
    },
];

const NUM_ARGUMENTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "n = numArgumentsFromSubscript(A, S, indexingContext)",
    inputs: &NUM_ARGUMENTS_INPUTS,
    outputs: &NUM_ARGUMENTS_OUTPUT,
}];

const NUM_ARGUMENTS_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM_ARGUMENTS_FROM_SUBSCRIPT.INVALID_ARGUMENT",
    identifier: Some("RunMat:numArgumentsFromSubscript:InvalidArgument"),
    when: "Inputs do not match the documented target, substruct, and indexing-context form.",
    message: "numArgumentsFromSubscript: invalid argument",
};

const NUM_ARGUMENTS_ERROR_SUBSTRUCT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM_ARGUMENTS_FROM_SUBSCRIPT.INVALID_SUBSTRUCT",
    identifier: Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct"),
    when: "The substruct-compatible descriptor is missing type/subs fields or has invalid values.",
    message: "numArgumentsFromSubscript: invalid substruct",
};

const NUM_ARGUMENTS_ERROR_CONTEXT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM_ARGUMENTS_FROM_SUBSCRIPT.INVALID_CONTEXT",
    identifier: Some("RunMat:numArgumentsFromSubscript:InvalidContext"),
    when: "The indexing context is not Statement, Expression, or Assignment.",
    message: "numArgumentsFromSubscript: invalid indexing context",
};

const NUM_ARGUMENTS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    NUM_ARGUMENTS_ERROR_ARGUMENT,
    NUM_ARGUMENTS_ERROR_SUBSTRUCT,
    NUM_ARGUMENTS_ERROR_CONTEXT,
];

pub const NUM_ARGUMENTS_FROM_SUBSCRIPT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NUM_ARGUMENTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NUM_ARGUMENTS_ERRORS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndexingContext {
    Statement,
    Expression,
    Assignment,
}

#[derive(Clone)]
struct SubscriptLevel {
    kind: String,
    subs: Value,
}

pub(crate) fn ensure_indexing_context_classes_registered() {
    INDEXING_CONTEXT_REGISTERED.get_or_init(|| {
        register_indexing_context_class(INDEXING_CONTEXT_CLASS);
        register_indexing_context_class(LEGACY_INDEXING_CONTEXT_CLASS);
    });
}

fn register_indexing_context_class(name: &str) {
    runmat_builtins::register_class(runmat_builtins::ClassDef {
        name: name.to_string(),
        parent: None,
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    });
    runmat_builtins::register_class_enumerations(
        name,
        [
            INDEXING_CONTEXT_STATEMENT.to_string(),
            INDEXING_CONTEXT_EXPRESSION.to_string(),
            INDEXING_CONTEXT_ASSIGNMENT.to_string(),
        ],
    );
}

fn num_args_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    crate::runtime_descriptor_error_with_detail(
        NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD,
        descriptor,
        detail.into(),
    )
}

fn parse_indexing_context(value: &Value) -> crate::BuiltinResult<IndexingContext> {
    let text = match value {
        Value::Object(object)
            if object.class_name == INDEXING_CONTEXT_CLASS
                || object.class_name == LEGACY_INDEXING_CONTEXT_CLASS =>
        {
            match object.properties.get(ENUM_MEMBER_PROPERTY) {
                Some(Value::String(member)) => member.clone(),
                Some(Value::CharArray(chars)) => chars.data.iter().collect(),
                _ => {
                    return Err(num_args_error(
                        &NUM_ARGUMENTS_ERROR_CONTEXT,
                        "indexing context enum object is missing its member name",
                    ))
                }
            }
        }
        other => String::try_from(other).map_err(|err| {
            num_args_error(
                &NUM_ARGUMENTS_ERROR_CONTEXT,
                format!("indexing context must be an IndexingContext enum or text: {err}"),
            )
        })?,
    };
    match text
        .trim()
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "statement" => Ok(IndexingContext::Statement),
        "expression" => Ok(IndexingContext::Expression),
        "assignment" => Ok(IndexingContext::Assignment),
        _ => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_CONTEXT,
            format!("unsupported indexing context '{text}'"),
        )),
    }
}

fn parse_subscript_levels(value: &Value) -> crate::BuiltinResult<Vec<SubscriptLevel>> {
    match value {
        Value::Struct(struct_value) => Ok(vec![parse_subscript_level(struct_value)?]),
        Value::Cell(cell) => {
            let mut levels = Vec::with_capacity(cell.data.len());
            for item in &cell.data {
                let Value::Struct(struct_value) = item else {
                    return Err(num_args_error(
                        &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                        "struct array elements must be scalar structs",
                    ));
                };
                levels.push(parse_subscript_level(struct_value)?);
            }
            if levels.is_empty() {
                return Err(num_args_error(
                    &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                    "substruct array must not be empty",
                ));
            }
            Ok(levels)
        }
        other => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            format!("S must be a struct or struct-array cell, got {other:?}"),
        )),
    }
}

fn parse_subscript_level(
    struct_value: &runmat_builtins::StructValue,
) -> crate::BuiltinResult<SubscriptLevel> {
    let kind_value = struct_value.fields.get("type").ok_or_else(|| {
        num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "substruct level is missing field 'type'",
        )
    })?;
    let kind = String::try_from(kind_value).map_err(|err| {
        num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            format!("substruct type must be text: {err}"),
        )
    })?;
    let subs = struct_value.fields.get("subs").cloned().ok_or_else(|| {
        num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "substruct level is missing field 'subs'",
        )
    })?;
    let kind = kind.trim().to_string();
    if !matches!(
        kind.as_str(),
        crate::OBJECT_INDEX_PAREN | crate::OBJECT_INDEX_BRACE | crate::OBJECT_INDEX_MEMBER
    ) {
        return Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            format!("unsupported substruct type '{kind}'"),
        ));
    }
    Ok(SubscriptLevel { kind, subs })
}

fn brace_selection_count(target: &Value, subs: &Value) -> crate::BuiltinResult<usize> {
    let Value::Cell(subscripts) = subs else {
        return Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "brace or parentheses subscripts must be stored in a cell array",
        ));
    };
    if subscripts.data.is_empty() {
        return Ok(0);
    }
    if subscripts.data.len() == 1 {
        return single_subscript_count(target, &subscripts.data[0], 0, true);
    }
    let mut count = 1usize;
    for (dim, subscript) in subscripts.data.iter().enumerate() {
        count = count
            .checked_mul(single_subscript_count(target, subscript, dim, false)?)
            .ok_or_else(|| {
                num_args_error(
                    &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                    "subscript selection count overflowed",
                )
            })?;
    }
    Ok(count)
}

fn single_subscript_count(
    target: &Value,
    subscript: &Value,
    dim: usize,
    linear_indexing: bool,
) -> crate::BuiltinResult<usize> {
    match subscript {
        Value::String(value) if value.trim() == ":" => {
            target_colon_extent(target, dim, linear_indexing)
        }
        Value::CharArray(chars) if chars.data.iter().collect::<String>().trim() == ":" => {
            target_colon_extent(target, dim, linear_indexing)
        }
        Value::StringArray(array) if array.data.len() == 1 && array.data[0].trim() == ":" => {
            target_colon_extent(target, dim, linear_indexing)
        }
        Value::Bool(flag) => Ok(usize::from(*flag)),
        Value::Num(value) => {
            validate_positive_integer_subscript(*value)?;
            Ok(1)
        }
        Value::Int(value) => {
            if value.try_to_u64().is_some_and(|value| value >= 1) {
                Ok(1)
            } else {
                Err(num_args_error(
                    &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                    "numeric subscripts must be finite positive integers",
                ))
            }
        }
        Value::Tensor(tensor) => {
            for index in 0..tensor.len() {
                let value = tensor.numeric_value_at(index).ok_or_else(|| {
                    num_args_error(
                        &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                        "numeric subscript storage is invalid",
                    )
                })?;
                match value {
                    NumericScalar::F64(value) => validate_positive_integer_subscript(value)?,
                    NumericScalar::F32(value) => {
                        validate_positive_integer_subscript(f64::from(value))?
                    }
                    value => {
                        if value
                            .into_int_value()
                            .and_then(|value| value.try_to_u64())
                            .is_none_or(|value| value < 1)
                        {
                            return Err(num_args_error(
                                &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                                "numeric subscripts must be finite positive integers",
                            ));
                        }
                    }
                }
            }
            checked_shape_element_count(&tensor.shape)
        }
        Value::LogicalArray(array) => Ok(array.data.iter().filter(|&&bit| bit != 0).count()),
        Value::StringArray(_) => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "string-array subscripts are only supported for ':'",
        )),
        Value::ComplexTensor(_) => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "complex subscripts are not valid cell-index selectors",
        )),
        Value::Cell(_) => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "nested cell subscripts are not valid cell-index selectors",
        )),
        Value::GpuTensor(_) => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "gpuArray subscripts cannot be validated without gathering",
        )),
        other => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            format!("unsupported subscript value {other:?}"),
        )),
    }
}

fn validate_positive_integer_subscript(value: f64) -> crate::BuiltinResult<()> {
    if value.is_finite() && value >= 1.0 && (value.round() - value).abs() <= f64::EPSILON {
        Ok(())
    } else {
        Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "numeric subscripts must be finite positive integers",
        ))
    }
}

fn checked_shape_element_count(shape: &[usize]) -> crate::BuiltinResult<usize> {
    let mut count = 1usize;
    for &dim in shape {
        count = count.checked_mul(dim).ok_or_else(|| {
            num_args_error(
                &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                "shape element count overflowed",
            )
        })?;
    }
    Ok(count)
}

fn target_colon_extent(
    target: &Value,
    dim: usize,
    linear_indexing: bool,
) -> crate::BuiltinResult<usize> {
    if linear_indexing {
        default_target_numel(target)
    } else {
        Ok(target_shape(target)
            .and_then(|shape| {
                if shape.is_empty() {
                    None
                } else {
                    Some(shape.get(dim).copied().unwrap_or(1))
                }
            })
            .unwrap_or(1))
    }
}

fn target_shape(target: &Value) -> Option<Vec<usize>> {
    match target {
        Value::Cell(cell) => Some(cell.shape.clone()),
        Value::Tensor(tensor) => Some(tensor.shape.clone()),
        Value::ComplexTensor(tensor) => Some(tensor.shape.clone()),
        Value::LogicalArray(array) => Some(array.shape.clone()),
        Value::StringArray(array) => Some(array.shape.clone()),
        Value::SparseTensor(sparse) => Some(sparse.shape()),
        Value::CharArray(chars) => Some(vec![chars.rows, chars.cols]),
        Value::GpuTensor(handle) => Some(handle.shape.clone()),
        Value::Object(_) | Value::HandleObject(_) => Some(vec![1, 1]),
        _ => None,
    }
}

fn default_target_numel(target: &Value) -> crate::BuiltinResult<usize> {
    match target_shape(target) {
        Some(shape) => checked_shape_element_count(&shape),
        None => Ok(1),
    }
}

fn default_num_arguments(
    target: &Value,
    levels: &[SubscriptLevel],
    _context: IndexingContext,
) -> crate::BuiltinResult<usize> {
    let Some(first) = levels.first() else {
        return Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "substruct must include at least one indexing level",
        ));
    };
    if first.kind == crate::OBJECT_INDEX_BRACE {
        return brace_selection_count(target, &first.subs);
    }
    Ok(1)
}

async fn dispatch_num_arguments_overload(
    class_name: String,
    target: Value,
    subscript: Value,
    indexing_context: Value,
) -> crate::BuiltinResult<Option<Value>> {
    let args = vec![target, subscript, indexing_context];
    if let Some((method, owner)) =
        runmat_builtins::lookup_method(&class_name, NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD)
    {
        let owner_member = format!("{owner}.{NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD}");
        let mut candidates = vec![method.function_name];
        if !candidates
            .iter()
            .any(|candidate| candidate == &owner_member)
        {
            candidates.push(owner_member);
        }
        let mut undefined = None;
        for candidate in candidates {
            let (identity, fallback_policy) = crate::callable_identity_for_handle_name(&candidate);
            match crate::dispatch_callable_with_policy(identity, fallback_policy, args.clone(), 1)
                .await
            {
                Ok(value) => return Ok(Some(value)),
                Err(err) if crate::is_undefined_function_error(&err) => {
                    undefined = Some(err);
                }
                Err(err) => return Err(err),
            }
        }
        return Err(undefined.unwrap_or_else(|| {
            crate::runtime_descriptor_error_with_detail(
                NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD,
                &NUM_ARGUMENTS_ERROR_ARGUMENT,
                "registered method did not resolve to a callable implementation",
            )
        }));
    }

    match crate::dispatch_object_external_member(
        class_name,
        NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD,
        args,
        1,
    )
    .await
    {
        Ok(value) => Ok(Some(value)),
        Err(err) if crate::is_undefined_function_error(&err) => Ok(None),
        Err(err) => Err(err),
    }
}

pub(crate) async fn dispatch_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> crate::BuiltinResult<Value> {
    match obj {
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let class_name = crate::object_receiver_class_name(&receiver).ok_or_else(|| {
                crate::runtime_descriptor_error("subsref", &SUBSREF_ERROR_RECEIVER_INVALID)
            })?;
            let dispatch_receiver = receiver.clone();
            let dispatch_kind = kind.clone();
            let dispatch_payload = payload.clone();
            match crate::dispatch_object_external_member(
                class_name,
                crate::OBJECT_SUBSREF_METHOD,
                vec![
                    dispatch_receiver,
                    Value::String(dispatch_kind),
                    dispatch_payload,
                ],
                crate::current_requested_outputs(),
            )
            .await
            {
                Ok(value) => Ok(value),
                Err(err) if crate::is_undefined_function_error(&err) => {
                    if kind == crate::OBJECT_INDEX_MEMBER {
                        let field = match payload {
                            Value::String(field) => Some(field),
                            Value::CharArray(ca) => Some(ca.data.iter().collect::<String>()),
                            _ => None,
                        };
                        if let Some(field) = field {
                            return crate::call_builtin_async_with_outputs(
                                "getfield",
                                &[receiver, Value::String(field)],
                                crate::current_requested_outputs(),
                            )
                            .await;
                        }
                    }
                    Err(crate::runtime_descriptor_error(
                        "subsref",
                        &SUBSREF_ERROR_METHOD_MISSING,
                    ))
                }
                Err(err) => Err(err),
            }
        }
        other => Err(crate::runtime_descriptor_error_with_detail(
            "subsref",
            &SUBSREF_ERROR_RECEIVER_INVALID,
            format!("receiver must be object, got {other:?}"),
        )),
    }
}

pub(crate) async fn dispatch_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    match obj {
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let class_name = crate::object_receiver_class_name(&receiver).ok_or_else(|| {
                crate::runtime_descriptor_error("subsasgn", &SUBSASGN_ERROR_RECEIVER_INVALID)
            })?;
            let dispatch_receiver = receiver.clone();
            let dispatch_kind = kind.clone();
            let dispatch_payload = payload.clone();
            let dispatch_rhs = rhs.clone();
            match crate::dispatch_object_external_member(
                class_name,
                crate::OBJECT_SUBSASGN_METHOD,
                vec![
                    dispatch_receiver,
                    Value::String(dispatch_kind),
                    dispatch_payload,
                    dispatch_rhs,
                ],
                crate::current_requested_outputs(),
            )
            .await
            {
                Ok(value) => Ok(value),
                Err(err) if crate::is_undefined_function_error(&err) => {
                    if kind == crate::OBJECT_INDEX_MEMBER {
                        let field = match payload {
                            Value::String(field) => Some(field),
                            Value::CharArray(ca) => Some(ca.data.iter().collect::<String>()),
                            _ => None,
                        };
                        if let Some(field) = field {
                            return crate::call_builtin_async_with_outputs(
                                "setfield",
                                &[receiver, Value::String(field), rhs],
                                crate::current_requested_outputs(),
                            )
                            .await;
                        }
                    }
                    Err(crate::runtime_descriptor_error(
                        "subsasgn",
                        &SUBSASGN_ERROR_METHOD_MISSING,
                    ))
                }
                Err(err) => Err(err),
            }
        }
        other => Err(crate::runtime_descriptor_error_with_detail(
            "subsasgn",
            &SUBSASGN_ERROR_RECEIVER_INVALID,
            format!("receiver must be object, got {other:?}"),
        )),
    }
}

#[runtime_builtin(
    name = "subsref",
    category = "introspection",
    summary = "Dispatch overloaded object indexing reads.",
    keywords = "subsref,indexing,classdef,object",
    descriptor(crate::builtins::introspection::object_indexing::SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::object_indexing"
)]
pub async fn subsref_builtin(
    obj: Value,
    kind: String,
    payload: Value,
) -> crate::BuiltinResult<Value> {
    dispatch_subsref(obj, kind, payload).await
}

#[runtime_builtin(
    name = "subsasgn",
    category = "introspection",
    summary = "Dispatch overloaded object indexing writes.",
    keywords = "subsasgn,indexing,classdef,object",
    descriptor(crate::builtins::introspection::object_indexing::SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::object_indexing"
)]
pub async fn subsasgn_builtin(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    dispatch_subsasgn(obj, kind, payload, rhs).await
}

#[runtime_builtin(
    name = "numArgumentsFromSubscript",
    category = "introspection",
    summary = "Return the number of arguments implied by a substruct indexing expression.",
    keywords = "numArgumentsFromSubscript,substruct,subsref,subsasgn,indexing,classdef,object",
    descriptor(
        crate::builtins::introspection::object_indexing::NUM_ARGUMENTS_FROM_SUBSCRIPT_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::introspection::object_indexing"
)]
pub async fn num_arguments_from_subscript_builtin(
    target: Value,
    subscript: Value,
    indexing_context: Value,
) -> crate::BuiltinResult<Value> {
    ensure_indexing_context_classes_registered();
    let context = parse_indexing_context(&indexing_context)?;
    if let Some(class_name) = crate::object_receiver_class_name(&target) {
        if let Some(value) = dispatch_num_arguments_overload(
            class_name,
            target.clone(),
            subscript.clone(),
            indexing_context.clone(),
        )
        .await?
        {
            return Ok(value);
        }
    }
    let levels = parse_subscript_levels(&subscript)?;
    Ok(Value::Num(
        default_num_arguments(&target, &levels, context)? as f64,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        Access, CellArray, ClassDef, IntValue, MethodDef, ObjectInstance, StructValue, Tensor,
    };
    use std::collections::HashMap;

    fn substruct(kind: &str, subs: Value) -> Value {
        let mut st = StructValue::new();
        st.insert("type", Value::String(kind.to_string()));
        st.insert("subs", subs);
        Value::Struct(st)
    }

    fn cell(values: Vec<Value>) -> Value {
        let cols = values.len();
        Value::Cell(CellArray::new(values, 1, cols).expect("cell"))
    }

    fn call(target: Value, subscript: Value, context: Value) -> Value {
        block_on(num_arguments_from_subscript_builtin(
            target, subscript, context,
        ))
        .expect("numArgumentsFromSubscript")
    }

    fn call_err(target: Value, subscript: Value, context: Value) -> crate::RuntimeError {
        block_on(num_arguments_from_subscript_builtin(
            target, subscript, context,
        ))
        .expect_err("numArgumentsFromSubscript should fail")
    }

    #[test]
    fn typed_scalar_subscripts_do_not_round_or_saturate() {
        assert_eq!(
            single_subscript_count(
                &Value::Num(0.0),
                &Value::Int(IntValue::U64(u64::MAX)),
                0,
                true
            )
            .expect("positive uint64 subscript"),
            1
        );
        assert!(
            single_subscript_count(&Value::Num(0.0), &Value::Int(IntValue::I64(-1)), 0, true)
                .is_err()
        );
    }

    #[test]
    fn default_cell_brace_indexing_counts_selected_outputs() {
        let target = cell(vec![
            Value::String("one".to_string()),
            Value::Num(2.0),
            Value::String("three".to_string()),
        ]);
        let indices = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("indices");
        let descriptor = substruct(
            crate::OBJECT_INDEX_BRACE,
            cell(vec![Value::Tensor(indices)]),
        );
        assert_eq!(
            call(target, descriptor, Value::String("Statement".to_string())),
            Value::Num(2.0)
        );
    }

    #[test]
    fn default_cell_brace_colon_counts_target_elements_without_gather() {
        let target = cell(vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)]);
        let descriptor = substruct(
            crate::OBJECT_INDEX_BRACE,
            cell(vec![Value::String(":".to_string())]),
        );
        assert_eq!(
            call(
                target,
                descriptor,
                Value::String("matlab.indexing.IndexingContext.Expression".to_string()),
            ),
            Value::Num(3.0)
        );
    }

    #[test]
    fn default_cell_brace_multidimensional_colons_use_shape_extents() {
        let target = Value::Cell(
            CellArray::new_with_shape(
                vec![
                    Value::Num(1.0),
                    Value::Num(2.0),
                    Value::Num(3.0),
                    Value::Num(4.0),
                    Value::Num(5.0),
                    Value::Num(6.0),
                ],
                vec![2, 3],
            )
            .expect("cell"),
        );
        let descriptor = substruct(
            crate::OBJECT_INDEX_BRACE,
            cell(vec![
                Value::String(":".to_string()),
                Value::String(":".to_string()),
            ]),
        );
        assert_eq!(
            call(target, descriptor, Value::String("Statement".to_string())),
            Value::Num(6.0)
        );
    }

    #[test]
    fn default_cell_brace_empty_logical_selection_can_return_zero() {
        let target = cell(vec![Value::Num(1.0), Value::Num(2.0)]);
        let descriptor = substruct(crate::OBJECT_INDEX_BRACE, cell(vec![Value::Bool(false)]));
        assert_eq!(
            call(target, descriptor, Value::String("Statement".to_string())),
            Value::Num(0.0)
        );
    }

    #[test]
    fn default_gpu_brace_colon_counts_shape_without_gather() {
        let target = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 3],
            device_id: 0,
            buffer_id: 99,
        });
        let descriptor = substruct(
            crate::OBJECT_INDEX_BRACE,
            cell(vec![
                Value::String(":".to_string()),
                Value::String(":".to_string()),
            ]),
        );
        assert_eq!(
            call(target, descriptor, Value::String("Statement".to_string())),
            Value::Num(6.0)
        );
    }

    #[test]
    fn default_brace_rejects_invalid_numeric_subscripts() {
        let target = cell(vec![Value::Num(1.0), Value::Num(2.0)]);
        for selector in [
            Value::Num(0.0),
            Value::Num(f64::NAN),
            Value::Num(1.5),
            Value::Tensor(Tensor::new(vec![1.0, 1.25], vec![1, 2]).expect("indices")),
        ] {
            let descriptor = substruct(crate::OBJECT_INDEX_BRACE, cell(vec![selector]));
            let err = call_err(
                target.clone(),
                descriptor,
                Value::String("Statement".to_string()),
            );
            assert_eq!(
                err.identifier(),
                Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct")
            );
        }
    }

    #[test]
    fn default_brace_rejects_unvalidated_subscript_containers() {
        let target = cell(vec![Value::Num(1.0), Value::Num(2.0)]);
        let selector = cell(vec![Value::Num(1.0)]);
        let descriptor = substruct(crate::OBJECT_INDEX_BRACE, cell(vec![selector]));
        let err = call_err(target, descriptor, Value::String("Statement".to_string()));
        assert_eq!(
            err.identifier(),
            Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct")
        );
    }

    #[test]
    fn default_brace_shape_overflow_returns_substruct_error() {
        let target = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![usize::MAX, 2],
            device_id: 0,
            buffer_id: 99,
        });
        let descriptor = substruct(
            crate::OBJECT_INDEX_BRACE,
            cell(vec![Value::String(":".to_string())]),
        );
        let err = call_err(target, descriptor, Value::String("Statement".to_string()));
        assert_eq!(
            err.identifier(),
            Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct")
        );
    }

    #[test]
    fn enum_context_object_is_accepted() {
        ensure_indexing_context_classes_registered();
        let target = cell(vec![Value::Num(1.0)]);
        let descriptor = substruct(crate::OBJECT_INDEX_PAREN, cell(vec![Value::Num(1.0)]));
        let mut context = ObjectInstance::new(INDEXING_CONTEXT_CLASS.to_string());
        context.properties.insert(
            ENUM_MEMBER_PROPERTY.to_string(),
            Value::String("Assignment".to_string()),
        );
        assert_eq!(
            call(target, descriptor, Value::Object(context)),
            Value::Num(1.0)
        );
    }

    #[test]
    fn object_overload_dispatches_num_arguments_from_subscript_method() {
        block_on(crate::register_test_classes_builtin()).expect("test classes");
        let mut object = ObjectInstance::new("OverIdx".to_string());
        object
            .properties
            .insert("nargs".to_string(), Value::Num(5.0));
        let descriptor = substruct(
            crate::OBJECT_INDEX_MEMBER,
            Value::String("field".to_string()),
        );
        assert_eq!(
            call(
                Value::Object(object),
                descriptor,
                Value::String("Expression".to_string()),
            ),
            Value::Num(5.0)
        );
    }

    #[test]
    fn object_overload_uses_inherited_remapped_method_metadata() {
        block_on(crate::register_test_classes_builtin()).expect("test classes");
        let base = "NumArgsFromSubscriptBase";
        let child = "NumArgsFromSubscriptChild";
        let mut methods = HashMap::new();
        methods.insert(
            NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD.to_string(),
            MethodDef {
                name: NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: format!("OverIdx.{NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD}"),
                implicit_class_argument: None,
            },
        );
        runmat_builtins::register_class(ClassDef {
            name: base.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods,
        });
        runmat_builtins::register_class(ClassDef {
            name: child.to_string(),
            parent: Some(base.to_string()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let mut object = ObjectInstance::new(child.to_string());
        object
            .properties
            .insert("nargs".to_string(), Value::Num(7.0));
        let descriptor = substruct(
            crate::OBJECT_INDEX_MEMBER,
            Value::String("field".to_string()),
        );
        assert_eq!(
            call(
                Value::Object(object),
                descriptor,
                Value::String("Expression".to_string()),
            ),
            Value::Num(7.0)
        );
    }

    #[test]
    fn invalid_context_has_stable_identifier() {
        let target = cell(vec![Value::Num(1.0)]);
        let descriptor = substruct(crate::OBJECT_INDEX_PAREN, cell(vec![Value::Num(1.0)]));
        let err = block_on(num_arguments_from_subscript_builtin(
            target,
            descriptor,
            Value::String("Loop".to_string()),
        ))
        .expect_err("invalid context should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:numArgumentsFromSubscript:InvalidContext")
        );
    }
}
