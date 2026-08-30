//! Java interoperability builtins backed by the session foreign runtime.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

const JAVA_ADAPTER: &str = "java";

const TEXT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Java class, method, feature, or classpath name.",
};
const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Java object or RunMat value used by the operation.",
};
const VARIADIC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Arguments supplied to the Java operation.",
};
const RESULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result of the Java operation.",
};

const TEXT_INPUT: [BuiltinParamDescriptor; 1] = [TEXT];
const VALUE_INPUT: [BuiltinParamDescriptor; 1] = [VALUE];
const VARIADIC_INPUT: [BuiltinParamDescriptor; 1] = [VARIADIC];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [RESULT];
const TEXT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "result = java_operation(name)",
    inputs: &TEXT_INPUT,
    outputs: &OUTPUTS,
}];
const VALUE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "result = java_operation(value)",
    inputs: &VALUE_INPUT,
    outputs: &OUTPUTS,
}];
const VARIADIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "result = java_operation(arguments)",
    inputs: &VARIADIC_INPUT,
    outputs: &OUTPUTS,
}];

const ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JAVA.UNAVAILABLE",
    identifier: Some("RunMat:Foreign:UnsupportedOnWasm"),
    when: "The current host has no native Java runtime capability.",
    message: "Java interoperability is unavailable on this host.",
};
const ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JAVA.INVALID_CALL",
    identifier: Some("RunMat:Foreign:InvalidCall"),
    when: "Arguments do not satisfy the Java interoperability contract.",
    message: "Invalid Java interoperability operation.",
};
const ERROR_HOST_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JAVA.HOST_UNAVAILABLE",
    identifier: Some("RunMat:Foreign:HostUnavailable"),
    when: "A Java operation is invoked without an active runtime context.",
    message: "Java operation has no active runtime context.",
};
const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_UNAVAILABLE, ERROR_INVALID, ERROR_HOST_UNAVAILABLE];
const TEXT_DESCRIPTOR: BuiltinDescriptor = descriptor(&TEXT_SIGNATURES);
const VALUE_DESCRIPTOR: BuiltinDescriptor = descriptor(&VALUE_SIGNATURES);
const VARIADIC_DESCRIPTOR: BuiltinDescriptor = descriptor(&VARIADIC_SIGNATURES);

const fn descriptor(signatures: &'static [BuiltinSignatureDescriptor]) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    }
}

const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Fixed-width integer scalars retain exact signed classes or use a checked Java widening conversion.",
};

async fn invoke_java(operation: &str, arguments: Vec<Value>) -> BuiltinResult<Value> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (operation, arguments);
        Err(java_error(&ERROR_UNAVAILABLE, ERROR_UNAVAILABLE.message))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let context = crate::context::legacy::active()
            .ok_or_else(|| java_error(&ERROR_HOST_UNAVAILABLE, ERROR_HOST_UNAVAILABLE.message))?;
        let service = context
            .service_ports()
            .require_foreign(operation)
            .map_err(|error| error.into_runtime_error())?
            .clone();
        context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: JAVA_ADAPTER.into(),
                    symbol: operation.into(),
                    arguments,
                    requested_outputs: 1,
                },
            ))
            .await
    }
}

#[runtime_builtin(
    name = "javaObject",
    execution_stack = "process",
    category = "interop/java",
    summary = "Construct an object from a Java class.",
    keywords = "javaObject,java,jvm,constructor",
    descriptor(crate::builtins::interop::java::VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_object_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke_java("construct", arguments).await
}

#[runtime_builtin(
    name = "javaObjectEDT",
    execution_stack = "process",
    category = "interop/java",
    summary = "Construct a Java object on the Desktop event-dispatch thread.",
    keywords = "javaObjectEDT,java,jvm,desktop,edt,constructor",
    descriptor(crate::builtins::interop::java::VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_object_edt_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke_java("construct_edt", arguments).await
}

#[runtime_builtin(
    name = "javaMethod",
    execution_stack = "process",
    category = "interop/java",
    summary = "Invoke a static or instance Java method.",
    keywords = "javaMethod,java,jvm,method",
    descriptor(crate::builtins::interop::java::VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_method_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    java_method_dispatch(arguments, "invoke_member", "call_static", "javaMethod").await
}

#[runtime_builtin(
    name = "javaMethodEDT",
    execution_stack = "process",
    category = "interop/java",
    summary = "Invoke a Java method on the Desktop event-dispatch thread.",
    keywords = "javaMethodEDT,java,jvm,desktop,edt,method",
    descriptor(crate::builtins::interop::java::VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_method_edt_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    java_method_dispatch(
        arguments,
        "invoke_member_edt",
        "call_static_edt",
        "javaMethodEDT",
    )
    .await
}

async fn java_method_dispatch(
    mut arguments: Vec<Value>,
    member_operation: &str,
    static_operation: &str,
    builtin: &str,
) -> BuiltinResult<Value> {
    if arguments.len() < 2 {
        return Err(invalid_call(format!(
            "{builtin} expects a method and class or object"
        )));
    }
    let method = arguments.remove(0);
    let receiver = arguments.remove(0);
    match receiver {
        Value::Foreign(reference) if reference.type_identity.family == JAVA_ADAPTER => {
            let mut call = vec![Value::Foreign(reference), method];
            call.extend(arguments);
            invoke_java(member_operation, call).await
        }
        class => {
            let mut call = vec![class, method];
            call.extend(arguments);
            invoke_java(static_operation, call).await
        }
    }
}

#[runtime_builtin(
    name = "javaArray",
    execution_stack = "process",
    category = "interop/java",
    summary = "Create a Java object array with the requested dimensions.",
    keywords = "javaArray,java,jvm,array",
    descriptor(crate::builtins::interop::java::VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_array_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke_java("new_array", arguments).await
}

#[runtime_builtin(
    name = "javaaddpath",
    execution_stack = "process",
    category = "interop/java",
    summary = "Add an entry to the current session's dynamic Java classpath.",
    keywords = "javaaddpath,java,jvm,classpath",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::interop::java::VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_add_path_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    let mut entries = text_entries(arguments, "javaaddpath")?;
    let at_end = entries.last().is_some_and(|entry| entry == "-end");
    if at_end {
        entries.pop();
    }
    if entries.is_empty() {
        return Err(invalid_call("javaaddpath requires a classpath entry"));
    }
    if !at_end {
        entries.reverse();
    }
    for entry in entries {
        invoke_java(
            "add_classpath",
            vec![
                Value::String(entry),
                Value::String(if at_end { "end" } else { "begin" }.into()),
            ],
        )
        .await?;
    }
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "javarmpath",
    execution_stack = "process",
    category = "interop/java",
    summary = "Remove an entry from the current session's dynamic Java classpath.",
    keywords = "javarmpath,java,jvm,classpath",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::interop::java::VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_remove_path_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    for entry in text_entries(arguments, "javarmpath")? {
        invoke_java("remove_classpath", vec![Value::String(entry)]).await?;
    }
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "javaclasspath",
    execution_stack = "process",
    category = "interop/java",
    summary = "Return the current session's Java classpath.",
    keywords = "javaclasspath,java,jvm,classpath",
    descriptor(crate::builtins::interop::java::VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_class_path_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    if arguments.is_empty() {
        return invoke_java("classpath", vec![Value::String("dynamic".into())]).await;
    }
    if arguments.len() == 1 {
        if let Ok(option) = String::try_from(&arguments[0]) {
            let layer = match option.to_ascii_lowercase().as_str() {
                "-dynamic" => Some("dynamic"),
                "-static" => Some("static"),
                "-all" => Some("all"),
                "-v0" | "-v1" => return Ok(Value::OutputList(Vec::new())),
                _ => None,
            };
            if let Some(layer) = layer {
                return invoke_java("classpath", vec![Value::String(layer.into())]).await;
            }
        }
    }
    let entries = text_entries(arguments, "javaclasspath")?
        .into_iter()
        .map(Value::String)
        .collect();
    invoke_java("set_classpath", entries).await?;
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "jenv",
    category = "interop/java",
    summary = "Inspect the Java runtime environment for this session.",
    keywords = "jenv,java,jvm,environment",
    descriptor(crate::builtins::interop::java::VARIADIC_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn jenv_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    if arguments.is_empty() {
        invoke_java("status", Vec::new()).await
    } else {
        invoke_java("configure", arguments).await
    }
}

#[runtime_builtin(
    name = "usejava",
    category = "interop/java",
    summary = "Test whether a Java runtime feature is available.",
    keywords = "usejava,java,jvm,awt,swing,desktop",
    descriptor(crate::builtins::interop::java::TEXT_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
async fn use_java_builtin(feature: Value) -> BuiltinResult<Value> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = feature;
        Ok(Value::Bool(false))
    }
    #[cfg(not(target_arch = "wasm32"))]
    invoke_java("usejava", vec![feature]).await
}

#[runtime_builtin(
    name = "isjava",
    category = "interop/java",
    summary = "Test whether a value is a live Java object.",
    keywords = "isjava,java,jvm,object",
    descriptor(crate::builtins::interop::java::VALUE_DESCRIPTOR),
    integer_audit(crate::builtins::interop::java::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::java"
)]
fn is_java_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Foreign(reference)
            if reference.type_identity.family == JAVA_ADAPTER
    )))
}

fn text_entries(arguments: Vec<Value>, builtin: &str) -> BuiltinResult<Vec<String>> {
    let mut entries = Vec::new();
    for value in arguments {
        match value {
            Value::Cell(cell) => {
                for value in cell.data {
                    entries
                        .push(String::try_from(&value).map_err(|_| {
                            invalid_call(format!("{builtin} entries must be text"))
                        })?);
                }
            }
            value => entries.push(
                String::try_from(&value)
                    .map_err(|_| invalid_call(format!("{builtin} entries must be text")))?,
            ),
        }
    }
    if entries.is_empty() {
        return Err(invalid_call(format!(
            "{builtin} requires a classpath entry"
        )));
    }
    Ok(entries)
}

fn invalid_call(message: impl Into<String>) -> crate::RuntimeError {
    java_error(&ERROR_INVALID, message)
}

fn java_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("java");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
