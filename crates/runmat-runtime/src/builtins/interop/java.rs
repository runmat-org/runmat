//! Java interoperability builtins backed by the session foreign runtime.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::execution::RootedValue;
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
) -> BuiltinResult<RootedValue> {
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
) -> BuiltinResult<RootedValue> {
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
) -> BuiltinResult<RootedValue> {
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
) -> BuiltinResult<RootedValue> {
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
) -> BuiltinResult<RootedValue> {
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
) -> BuiltinResult<RootedValue> {
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
) -> BuiltinResult<RootedValue> {
    let mut entries = text_entries(arguments, "javaaddpath")?;
    let position = entries
        .last()
        .and_then(|entry| runmat_builtins::JavaInsertionPosition::from_public_terminal(entry))
        .unwrap_or(runmat_builtins::JavaInsertionPosition::Begin);
    if position == runmat_builtins::JavaInsertionPosition::End {
        entries.pop();
    }
    if entries.is_empty() {
        return Err(invalid_call("javaaddpath requires a classpath entry"));
    }
    if position == runmat_builtins::JavaInsertionPosition::Begin {
        entries.reverse();
    }
    for entry in entries {
        invoke_java(
            context.clone(),
            JavaOperation::AddClasspath,
            vec![
                Value::String(entry),
                Value::String(position.wire_name().into()),
            ],
        )
        .await?;
    }
    RootedValue::new(Value::output_list(Vec::new())).map_err(|error| error.into_runtime_error())
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
) -> BuiltinResult<RootedValue> {
    for entry in text_entries(arguments, "javarmpath")? {
        invoke_java(
            context.clone(),
            JavaOperation::RemoveClasspath,
            vec![Value::String(entry)],
        )
        .await?;
    }
    RootedValue::new(Value::output_list(Vec::new())).map_err(|error| error.into_runtime_error())
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
) -> BuiltinResult<RootedValue> {
    if arguments.is_empty() {
        return invoke_java(
            context,
            JavaOperation::Classpath,
            vec![Value::String(
                runmat_builtins::JavaClasspathQuery::Dynamic
                    .wire_name()
                    .into(),
            )],
        )
        .await;
    }
    if arguments.len() == 1 {
        if let Some(option) =
            crate::builtins::common::validation::value_text_scalar_string(&arguments[0])
        {
            match runmat_builtins::JavaClasspathControl::parse_public(&option) {
                runmat_builtins::JavaClasspathControl::Query(layer) => {
                    return invoke_java(
                        context,
                        JavaOperation::Classpath,
                        vec![Value::String(layer.wire_name().into())],
                    )
                    .await;
                }
                runmat_builtins::JavaClasspathControl::Verbosity => {
                    return RootedValue::new(Value::output_list(Vec::new()))
                        .map_err(|error| error.into_runtime_error());
                }
                runmat_builtins::JavaClasspathControl::Path => {}
            }
        }
    }
    let entries = text_entries(arguments, "javaclasspath")?
        .into_iter()
        .map(Value::String)
        .collect();
    invoke_java(context, JavaOperation::SetClasspath, entries).await?;
    RootedValue::new(Value::output_list(Vec::new())).map_err(|error| error.into_runtime_error())
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
) -> BuiltinResult<RootedValue> {
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
) -> BuiltinResult<RootedValue> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = context;
        let name = crate::builtins::common::validation::value_text_scalar_string(&feature)
            .ok_or_else(|| invalid_call("usejava feature must be text"))?;
        if runmat_builtins::JavaFeature::parse(&name).is_none() {
            return Err(invalid_call(
                "usejava feature must be jvm, desktop, awt, or swing",
            ));
        }
        RootedValue::new(Value::Bool(false)).map_err(|error| error.into_runtime_error())
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
