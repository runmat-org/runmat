//! Shared-library interoperability builtins backed by the session foreign service.

use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{
    NativeLoadOption, CALLLIB_DESCRIPTOR, LIBFUNCTIONS_DESCRIPTOR, LIBISLOADED_DESCRIPTOR,
    LIBPOINTER_DESCRIPTOR, LIBSTRUCT_DESCRIPTOR, LOADLIBRARY_DESCRIPTOR, NATIVE_FFI_INTEGER_AUDIT,
    SETDATATYPE_DESCRIPTOR, UNLOADLIBRARY_DESCRIPTOR,
};
use runmat_builtins::{
    NATIVE_FFI_ERROR_HOST_UNAVAILABLE as ERROR_HOST_UNAVAILABLE,
    NATIVE_FFI_ERROR_INVALID as ERROR_INVALID, NATIVE_FFI_ERROR_UNAVAILABLE as ERROR_UNAVAILABLE,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

async fn invoke_native(
    context: Option<crate::context::RuntimeContext>,
    operation: crate::foreign::NativeFfiOperation,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<Value> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (context, operation, arguments, requested_outputs);
        Err(native_error(
            &ERROR_UNAVAILABLE,
            "native_ffi",
            ERROR_UNAVAILABLE.message,
        ))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let context = context.ok_or_else(|| {
            native_error(
                &ERROR_HOST_UNAVAILABLE,
                "native_ffi",
                ERROR_HOST_UNAVAILABLE.message,
            )
        })?;
        let service = context
            .service_ports()
            .require_foreign(operation.symbol())
            .map_err(|error| error.into_runtime_error())?
            .clone();
        let sequence = context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: "native-ffi".into(),
                    symbol: operation.symbol().into(),
                    arguments,
                    requested_outputs,
                },
            ))
            .await?;
        Ok(crate::call::arguments::project_legacy_builtin_value_abi(
            sequence,
        ))
    }
}

#[runtime_builtin(
    name = "loadlibrary",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn loadlibrary_builtin(
    context: Option<crate::context::RuntimeContext>,
    mut arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    if arguments.is_empty() {
        return Err(invalid_builtin_call("loadlibrary expects a library path"));
    }
    if arguments.len() == 1 {
        let library = String::try_from(&arguments[0]).map_err(|error| {
            native_error(
                &ERROR_INVALID,
                "loadlibrary",
                format!("loadlibrary: expected a library name: {error}"),
            )
        })?;
        let header = std::path::PathBuf::from(library).with_extension("h");
        arguments.push(Value::String(header.display().to_string()));
    }
    text_argument(&arguments[1], "loadlibrary header")?;
    let mut alias = None;
    let mut preparation_options = Vec::new();
    if arguments.len() > 2 {
        if !(arguments.len() - 2).is_multiple_of(2) {
            return Err(invalid_builtin_call(
                "loadlibrary options must be supplied as name-value pairs",
            ));
        }
        for pair in arguments[2..].chunks_exact(2) {
            let name = String::try_from(&pair[0])
                .map_err(|_| invalid_builtin_call("loadlibrary option names must be text"))?;
            match NativeLoadOption::parse(&name) {
                Some(NativeLoadOption::Alias) => {
                    alias = Some(Value::String(String::try_from(&pair[1]).map_err(|_| {
                        invalid_builtin_call("loadlibrary alias must be text")
                    })?));
                }
                Some(NativeLoadOption::IncludePath) => {
                    preparation_options.push(Value::String(
                        NativeLoadOption::IncludePath.canonical_name().into(),
                    ));
                    preparation_options.push(Value::String(String::try_from(&pair[1]).map_err(
                        |_| invalid_builtin_call("loadlibrary include path must be text"),
                    )?));
                }
                Some(NativeLoadOption::AddHeader) => {
                    preparation_options.push(Value::String(
                        NativeLoadOption::AddHeader.canonical_name().into(),
                    ));
                    preparation_options.push(Value::String(String::try_from(&pair[1]).map_err(
                        |_| invalid_builtin_call("loadlibrary additional header must be text"),
                    )?));
                }
                None => {
                    return Err(invalid_builtin_call(format!(
                        "loadlibrary option `{}` is not available in this interface path",
                        name.to_ascii_lowercase()
                    )))
                }
            }
        }
        arguments.truncate(2);
    }
    if let Some(alias) = alias {
        arguments.push(alias);
    } else {
        let library = String::try_from(&arguments[0])
            .map_err(|_| invalid_builtin_call("loadlibrary library name must be text"))?;
        let alias = std::path::PathBuf::from(library)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| invalid_builtin_call("loadlibrary path has no valid file name"))?
            .to_string();
        arguments.push(Value::String(alias));
    }
    arguments.extend(preparation_options);
    let report = invoke_native(
        context,
        crate::foreign::NativeFfiOperation::LoadReport,
        arguments,
        2,
    )
    .await?;
    let Value::OutputList(mut report) = report else {
        return Err(invalid_builtin_call(
            "native library loader returned an invalid preparation report",
        ));
    };
    if report.len() != 2 {
        return Err(invalid_builtin_call(
            "native library loader returned an incomplete preparation report",
        ));
    }
    let requested = crate::output_count::current_output_count().unwrap_or(0);
    Ok(match requested {
        0 => Value::OutputList(Vec::new()),
        1 => report.remove(0),
        _ => Value::OutputList(report),
    })
}

#[runtime_builtin(
    name = "calllib",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn calllib_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    if arguments.len() < 2 {
        return Err(invalid_builtin_call(
            "calllib requires a library alias and function name",
        ));
    }
    text_argument(&arguments[0], "calllib library alias")?;
    text_argument(&arguments[1], "calllib function name")?;
    let requested = crate::output_count::current_output_count().unwrap_or(1);
    invoke_native(
        context,
        crate::foreign::NativeFfiOperation::Call,
        arguments,
        requested,
    )
    .await
}

#[runtime_builtin(
    name = "unloadlibrary",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn unloadlibrary_builtin(
    context: Option<crate::context::RuntimeContext>,
    alias: Value,
) -> BuiltinResult<Value> {
    text_argument(&alias, "unloadlibrary alias")?;
    invoke_native(
        context,
        crate::foreign::NativeFfiOperation::Unload,
        vec![alias],
        1,
    )
    .await?;
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "libisloaded",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libisloaded_builtin(
    context: Option<crate::context::RuntimeContext>,
    alias: Value,
) -> BuiltinResult<Value> {
    text_argument(&alias, "libisloaded alias")?;
    invoke_native(
        context,
        crate::foreign::NativeFfiOperation::IsLoaded,
        vec![alias],
        1,
    )
    .await
}

#[runtime_builtin(
    name = "libfunctions",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libfunctions_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    if !(1..=2).contains(&arguments.len()) {
        return Err(invalid_builtin_call(
            "libfunctions expects an alias and optional -full flag",
        ));
    }
    text_argument(&arguments[0], "libfunctions alias")?;
    if let Some(option) = arguments.get(1) {
        let option = text_argument(option, "libfunctions option")?;
        if !option.eq_ignore_ascii_case("-full") {
            return Err(invalid_builtin_call(format!(
                "unsupported libfunctions option `{option}`; expected -full"
            )));
        }
    }
    invoke_native(
        context,
        crate::foreign::NativeFfiOperation::Functions,
        arguments,
        1,
    )
    .await
}

#[runtime_builtin(
    name = "libpointer",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libpointer_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    if arguments.len() > 2 {
        return Err(invalid_builtin_call(
            "libpointer expects an optional type and value",
        ));
    }
    if let Some(pointer_type) = arguments.first() {
        text_argument(pointer_type, "libpointer type")?;
    }
    invoke_native(
        context,
        crate::foreign::NativeFfiOperation::Pointer,
        arguments,
        1,
    )
    .await
}

#[runtime_builtin(
    name = "setdatatype",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn setdatatype_builtin(
    context: Option<crate::context::RuntimeContext>,
    mut arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    if arguments.len() < 3 {
        return Err(invalid_builtin_call(
            "setdatatype expects a pointer, type, and one or more dimensions",
        ));
    }
    if !matches!(arguments.first(), Some(Value::Foreign(_))) {
        return Err(invalid_builtin_call(
            "setdatatype requires a native pointer handle",
        ));
    }
    text_argument(&arguments[1], "setdatatype pointer type")?;
    arguments.insert(1, Value::String("setdatatype".into()));
    invoke_native(
        context,
        crate::foreign::NativeFfiOperation::InvokeMember,
        arguments,
        1,
    )
    .await?;
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "libstruct",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libstruct_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    if !(1..=2).contains(&arguments.len()) {
        return Err(invalid_builtin_call(
            "libstruct expects a type and optional scalar structure",
        ));
    }
    text_argument(&arguments[0], "libstruct type")?;
    if arguments
        .get(1)
        .is_some_and(|value| !matches!(value, Value::Struct(_)))
    {
        return Err(invalid_builtin_call(
            "native structure initialization requires a scalar structure value",
        ));
    }
    invoke_native(
        context,
        crate::foreign::NativeFfiOperation::Structure,
        arguments,
        1,
    )
    .await
}

fn invalid_builtin_call(message: impl Into<String>) -> crate::RuntimeError {
    native_error(&ERROR_INVALID, "native_ffi", message)
}

fn text_argument(value: &Value, label: &str) -> BuiltinResult<String> {
    String::try_from(value).map_err(|_| invalid_builtin_call(format!("{label} must be text")))
}

fn native_error(
    descriptor: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests;
