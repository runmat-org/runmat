//! Java interoperability builtins backed by the session foreign runtime.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::execution::RootedValue;
use crate::BuiltinResult;

mod dispatch;
use dispatch::{
    invalid_call, invoke_java, invoke_java_prepared, text_entries, text_scalar_eq, JavaOperation,
};

const JAVA_ADAPTER: &str = "java";

#[runtime_builtin(
    name = "javaObject",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_object_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
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
    arguments: crate::call::BuiltinArguments,
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
    arguments: crate::call::BuiltinArguments,
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
    arguments: crate::call::BuiltinArguments,
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
    arguments: crate::call::BuiltinArguments,
    member_operation: JavaOperation,
    static_operation: JavaOperation,
    builtin: &str,
) -> BuiltinResult<RootedValue> {
    if arguments.len() < 2 {
        return Err(invalid_call(format!(
            "{builtin} expects a method and class or object"
        )));
    }
    let receiver = arguments.get(1).expect("validated argument prefix");
    let operation = match receiver {
        Value::Foreign(reference) if reference.type_identity.family == JAVA_ADAPTER => {
            member_operation
        }
        _ => static_operation,
    };
    let count = arguments.len();
    invoke_java_prepared(context, operation, move |runtime| {
        arguments.try_map_preserving_roots_in_context(count, runtime, |slot, source| {
            let source_index = match slot {
                0 => 1,
                1 => 0,
                index => index,
            };
            Ok(source
                .get(source_index)
                .expect("validated Java argument slot")
                .clone())
        })
    })
    .await
}

#[runtime_builtin(
    name = "javaArray",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::java"
)]
async fn java_array_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entries = text_entries(arguments, "javaaddpath")?;
    let position = entries
        .len()
        .checked_sub(1)
        .and_then(|index| entries.get(index))
        .filter(|entry| text_scalar_eq(entry, "-end", false))
        .map(|_| runmat_builtins::JavaInsertionPosition::End)
        .unwrap_or(runmat_builtins::JavaInsertionPosition::Begin);
    let count =
        entries.len() - usize::from(position == runmat_builtins::JavaInsertionPosition::End);
    if count == 0 {
        return Err(invalid_call("javaaddpath requires a classpath entry"));
    }
    for logical_index in 0..count {
        let entry_index = if position == runmat_builtins::JavaInsertionPosition::Begin {
            count - logical_index - 1
        } else {
            logical_index
        };
        invoke_java_prepared(context.clone(), JavaOperation::AddClasspath, |runtime| {
            entries.entry_arguments(entry_index, Some(position.wire_name()), runtime)
        })
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entries = text_entries(arguments, "javarmpath")?;
    for index in 0..entries.len() {
        invoke_java_prepared(context.clone(), JavaOperation::RemoveClasspath, |runtime| {
            entries.entry_arguments(index, None, runtime)
        })
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    if arguments.is_empty() {
        return invoke_java_prepared(context, JavaOperation::Classpath, |runtime| {
            wire_arguments(
                runtime,
                &[runmat_builtins::JavaClasspathQuery::Dynamic.wire_name()],
            )
        })
        .await;
    }
    if arguments.len() == 1 {
        let value = arguments.first().expect("one Java classpath argument");
        let control = if text_scalar_eq(value, "-dynamic", true) {
            Some(runmat_builtins::JavaClasspathControl::Query(
                runmat_builtins::JavaClasspathQuery::Dynamic,
            ))
        } else if text_scalar_eq(value, "-static", true) {
            Some(runmat_builtins::JavaClasspathControl::Query(
                runmat_builtins::JavaClasspathQuery::Static,
            ))
        } else if text_scalar_eq(value, "-all", true) {
            Some(runmat_builtins::JavaClasspathControl::Query(
                runmat_builtins::JavaClasspathQuery::All,
            ))
        } else if text_scalar_eq(value, "-v0", true) || text_scalar_eq(value, "-v1", true) {
            Some(runmat_builtins::JavaClasspathControl::Verbosity)
        } else {
            None
        };
        if let Some(control) = control {
            match control {
                runmat_builtins::JavaClasspathControl::Query(layer) => {
                    return invoke_java_prepared(context, JavaOperation::Classpath, |runtime| {
                        wire_arguments(runtime, &[layer.wire_name()])
                    })
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
    let entries = text_entries(arguments, "javaclasspath")?;
    invoke_java_prepared(context, JavaOperation::SetClasspath, move |runtime| {
        entries.all_arguments(runtime)
    })
    .await?;
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    if arguments.is_empty() {
        invoke_java(context, JavaOperation::Status, arguments).await
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
    invoke_java_prepared(context, JavaOperation::UseJava, move |runtime| {
        crate::call::BuiltinArguments::from_borrowed(std::slice::from_ref(&feature), Some(runtime))
    })
    .await
}

fn wire_arguments(
    runtime: &crate::context::RuntimeContext,
    values: &[&str],
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let values = crate::execution::RootedValues::try_initialize_in_context(
        values.len(),
        runtime,
        |index| {
            crate::call::argument_text::copy_str(values[index], Some(runtime))
                .map(|text| Value::String(text.into()))
        },
    )?;
    Ok(values.into())
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
