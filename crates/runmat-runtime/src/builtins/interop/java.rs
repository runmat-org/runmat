//! Java interoperability builtins backed by the session foreign runtime.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

mod dispatch;
use dispatch::{invalid_call, invoke_java, text_entries, JavaOperation};

const JAVA_ADAPTER: &str = "java";

#[runtime_builtin(
    name = "javaObject",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_object_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    invoke_java(context, JavaOperation::Construct, arguments).await
}

#[runtime_builtin(
    name = "javaObjectEDT",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_object_edt_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    invoke_java(context, JavaOperation::ConstructEdt, arguments).await
}

#[runtime_builtin(
    name = "javaMethod",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_method_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    java_method_dispatch(
        context,
        arguments,
        JavaOperation::InvokeMember,
        JavaOperation::CallStatic,
        "javaMethod",
    )
    .await
}

#[runtime_builtin(
    name = "javaMethodEDT",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_method_edt_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    java_method_dispatch(
        context,
        arguments,
        JavaOperation::InvokeMemberEdt,
        JavaOperation::CallStaticEdt,
        "javaMethodEDT",
    )
    .await
}

async fn java_method_dispatch(
    context: Option<crate::context::RuntimeContext>,
    mut arguments: Vec<Value>,
    member_operation: JavaOperation,
    static_operation: JavaOperation,
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
            invoke_java(context, member_operation, call).await
        }
        class => {
            let mut call = vec![class, method];
            call.extend(arguments);
            invoke_java(context, static_operation, call).await
        }
    }
}

#[runtime_builtin(
    name = "javaArray",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_array_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    invoke_java(context, JavaOperation::NewArray, arguments).await
}

#[runtime_builtin(
    name = "javaaddpath",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_add_path_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
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
            context.clone(),
            JavaOperation::AddClasspath,
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_remove_path_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    for entry in text_entries(arguments, "javarmpath")? {
        invoke_java(
            context.clone(),
            JavaOperation::RemoveClasspath,
            vec![Value::String(entry)],
        )
        .await?;
    }
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "javaclasspath",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_class_path_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    if arguments.is_empty() {
        return invoke_java(
            context,
            JavaOperation::Classpath,
            vec![Value::String("dynamic".into())],
        )
        .await;
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
                return invoke_java(
                    context,
                    JavaOperation::Classpath,
                    vec![Value::String(layer.into())],
                )
                .await;
            }
        }
    }
    let entries = text_entries(arguments, "javaclasspath")?
        .into_iter()
        .map(Value::String)
        .collect();
    invoke_java(context, JavaOperation::SetClasspath, entries).await?;
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "jenv",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn jenv_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    if arguments.is_empty() {
        invoke_java(context, JavaOperation::Status, Vec::new()).await
    } else {
        invoke_java(context, JavaOperation::Configure, arguments).await
    }
}

#[runtime_builtin(
    name = "usejava",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn use_java_builtin(
    context: Option<crate::context::RuntimeContext>,
    feature: Value,
) -> BuiltinResult<Value> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (context, feature);
        Ok(Value::Bool(false))
    }
    #[cfg(not(target_arch = "wasm32"))]
    invoke_java(context, JavaOperation::UseJava, vec![feature]).await
}

#[runtime_builtin(
    name = "isjava",
    binding_variant = "default",
    builtin_path = "crate::builtins::interop::java"
)]
fn is_java_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Foreign(reference)
            if reference.type_identity.family == JAVA_ADAPTER
    )))
}

#[cfg(test)]
#[path = "java/tests.rs"]
mod tests;
