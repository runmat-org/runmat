use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const BUILTIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result from the builtin implementation.",
}];

const BUILTIN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Builtin function name.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Builtin function arguments.",
    },
];

const BUILTIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[out] = builtin(name, varargin)",
    inputs: &BUILTIN_INPUTS,
    outputs: &BUILTIN_OUTPUT,
}];

pub const BUILTIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUILTIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

fn parse_substruct_type_and_subs(arg: &Value) -> crate::BuiltinResult<(String, Value)> {
    let Value::Struct(st) = arg else {
        return Err(crate::build_runtime_error(
            "builtin: substruct argument must be a scalar struct",
        )
        .with_builtin("builtin")
        .build());
    };
    let kind_value = st.fields.get("type").ok_or_else(|| {
        crate::build_runtime_error("builtin: substruct argument missing field 'type'")
            .with_builtin("builtin")
            .build()
    })?;
    let kind = String::try_from(kind_value).map_err(|err| {
        crate::build_runtime_error(format!("builtin subsref/subsasgn: {err}"))
            .with_builtin("builtin")
            .build()
    })?;
    let subs = st.fields.get("subs").cloned().ok_or_else(|| {
        crate::build_runtime_error("builtin: substruct argument missing field 'subs'")
            .with_builtin("builtin")
            .build()
    })?;
    Ok((kind, subs))
}

fn builtin_member_field_name(subs: &Value, op: &str) -> crate::BuiltinResult<String> {
    String::try_from(subs).map_err(|err| {
        crate::build_runtime_error(format!("builtin {op}: {err}"))
            .with_builtin("builtin")
            .build()
    })
}

async fn dispatch_builtin_subsref(
    rest: &[Value],
    requested_outputs: usize,
) -> crate::BuiltinResult<Value> {
    if rest.len() != 2 {
        return crate::call_builtin_async_with_outputs("subsref", rest, requested_outputs).await;
    }
    let (kind, subs) = parse_substruct_type_and_subs(&rest[1])?;
    if kind != crate::OBJECT_INDEX_MEMBER {
        return crate::call_builtin_async_with_outputs("subsref", rest, requested_outputs).await;
    }
    let field = builtin_member_field_name(&subs, "subsref")?;
    match crate::call_builtin_async_with_outputs(
        "getfield",
        &[rest[0].clone(), Value::String(field.clone())],
        requested_outputs,
    )
    .await
    {
        Ok(value) => Ok(value),
        Err(err) if err.identifier() == Some("RunMat:getfield:ObjectProperty") => {
            crate::call_builtin_async_with_outputs(
                "call_method",
                &[rest[0].clone(), Value::String(field)],
                requested_outputs,
            )
            .await
        }
        Err(err) => Err(err),
    }
}

async fn dispatch_builtin_subsasgn(
    rest: &[Value],
    requested_outputs: usize,
) -> crate::BuiltinResult<Value> {
    if rest.len() != 3 {
        return crate::call_builtin_async_with_outputs("subsasgn", rest, requested_outputs).await;
    }
    let (kind, subs) = parse_substruct_type_and_subs(&rest[1])?;
    if kind != crate::OBJECT_INDEX_MEMBER {
        return crate::call_builtin_async_with_outputs("subsasgn", rest, requested_outputs).await;
    }
    let field = builtin_member_field_name(&subs, "subsasgn")?;
    crate::call_builtin_async_with_outputs(
        "setfield",
        &[rest[0].clone(), Value::String(field), rest[2].clone()],
        requested_outputs,
    )
    .await
}

#[runtime_builtin(
    name = "builtin",
    category = "introspection",
    summary = "Invoke a builtin implementation by name.",
    keywords = "builtin,dispatch,subsref,subsasgn",
    descriptor(crate::builtins::introspection::builtin::BUILTIN_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::builtin"
)]
pub async fn builtin_builtin(name: String, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);

    if name.eq_ignore_ascii_case("subsref") {
        return dispatch_builtin_subsref(&rest, requested_outputs).await;
    }

    if name.eq_ignore_ascii_case("subsasgn") {
        return dispatch_builtin_subsasgn(&rest, requested_outputs).await;
    }

    crate::call_builtin_async_with_outputs(&name, &rest, requested_outputs).await
}
