use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

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
