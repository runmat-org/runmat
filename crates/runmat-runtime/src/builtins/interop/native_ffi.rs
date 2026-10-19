//! Shared-library interoperability builtins backed by the session foreign service.

#[cfg(not(target_arch = "wasm32"))]
use runmat_builtins::NATIVE_FFI_ERROR_HOST_UNAVAILABLE as ERROR_HOST_UNAVAILABLE;
use runmat_builtins::NATIVE_FFI_ERROR_INVALID as ERROR_INVALID;
#[cfg(target_arch = "wasm32")]
use runmat_builtins::NATIVE_FFI_ERROR_UNAVAILABLE as ERROR_UNAVAILABLE;
use runmat_builtins::{
    BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinParamArity, NativeFfiOperation,
};
pub use runmat_builtins::{
    NativeLoadOption, CALLLIB_DESCRIPTOR, LIBFUNCTIONS_DESCRIPTOR, LIBISLOADED_DESCRIPTOR,
    LIBPOINTER_DESCRIPTOR, LIBSTRUCT_DESCRIPTOR, LOADLIBRARY_DESCRIPTOR, SETDATATYPE_DESCRIPTOR,
    UNLOADLIBRARY_DESCRIPTOR,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::execution::RootedValue;
use crate::{build_runtime_error, BuiltinResult};

async fn invoke_native(
    context: Option<crate::context::RuntimeContext>,
    operation: NativeFfiOperation,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<RootedValue> {
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
            .await
            .into_invoked_result()?;
        sequence.try_map_owned(
            |sequence| {
                Ok::<_, crate::RuntimeError>(
                    crate::call::arguments::project_legacy_builtin_value_abi(sequence),
                )
            },
            |error| error.into_runtime_error(),
        )
    }
}

#[runtime_builtin(
    name = "loadlibrary",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn loadlibrary_builtin(
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    mut arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&LOADLIBRARY_DESCRIPTOR, "loadlibrary", requested_outputs)?;
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
    let report = invoke_native(context, NativeFfiOperation::LoadReport, arguments, 2).await?;
    let requested = requested_outputs.unwrap_or(0);
    report.try_map_owned(
        |report| {
            let Value::OutputList(report) = report else {
                return Err(invalid_builtin_call(
                    "native library loader returned an invalid preparation report",
                ));
            };
            if report.len() != 2 {
                return Err(invalid_builtin_call(
                    "native library loader returned an incomplete preparation report",
                ));
            }
            Ok(match requested {
                0 => Value::output_list(Vec::new()),
                1 => crate::sequence::into_first_output(report)?
                    .expect("validated preparation report"),
                _ => Value::OutputList(report),
            })
        },
        |error| error.into_runtime_error(),
    )
}

#[runtime_builtin(
    name = "calllib",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn calllib_builtin(
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    if arguments.len() < 2 {
        return Err(invalid_builtin_call(
            "calllib requires a library alias and function name",
        ));
    }
    text_argument(&arguments[0], "calllib library alias")?;
    text_argument(&arguments[1], "calllib function name")?;
    validate_output_count(&CALLLIB_DESCRIPTOR, "calllib", requested_outputs)?;
    let requested = requested_outputs.unwrap_or(1);
    invoke_native(context, NativeFfiOperation::Call, arguments, requested).await
}

#[runtime_builtin(
    name = "unloadlibrary",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn unloadlibrary_builtin(
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    alias: Value,
) -> BuiltinResult<RootedValue> {
    validate_output_count(
        &UNLOADLIBRARY_DESCRIPTOR,
        "unloadlibrary",
        requested_outputs,
    )?;
    text_argument(&alias, "unloadlibrary alias")?;
    invoke_native(context, NativeFfiOperation::Unload, vec![alias], 1).await?;
    RootedValue::new(Value::output_list(Vec::new())).map_err(|error| error.into_runtime_error())
}

#[runtime_builtin(
    name = "libisloaded",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libisloaded_builtin(
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    alias: Value,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&LIBISLOADED_DESCRIPTOR, "libisloaded", requested_outputs)?;
    text_argument(&alias, "libisloaded alias")?;
    invoke_native(context, NativeFfiOperation::IsLoaded, vec![alias], 1).await
}

#[runtime_builtin(
    name = "libfunctions",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libfunctions_builtin(
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&LIBFUNCTIONS_DESCRIPTOR, "libfunctions", requested_outputs)?;
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
    invoke_native(context, NativeFfiOperation::Functions, arguments, 1).await
}

#[runtime_builtin(
    name = "libpointer",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libpointer_builtin(
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&LIBPOINTER_DESCRIPTOR, "libpointer", requested_outputs)?;
    if arguments.len() > 2 {
        return Err(invalid_builtin_call(
            "libpointer expects an optional type and value",
        ));
    }
    if let Some(pointer_type) = arguments.first() {
        text_argument(pointer_type, "libpointer type")?;
    }
    invoke_native(context, NativeFfiOperation::Pointer, arguments, 1).await
}

#[runtime_builtin(
    name = "setdatatype",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn setdatatype_builtin(
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    mut arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&SETDATATYPE_DESCRIPTOR, "setdatatype", requested_outputs)?;
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
    invoke_native(context, NativeFfiOperation::InvokeMember, arguments, 1).await?;
    RootedValue::new(Value::output_list(Vec::new())).map_err(|error| error.into_runtime_error())
}

#[runtime_builtin(
    name = "libstruct",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libstruct_builtin(
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&LIBSTRUCT_DESCRIPTOR, "libstruct", requested_outputs)?;
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
    invoke_native(context, NativeFfiOperation::Structure, arguments, 1).await
}

fn validate_output_count(
    descriptor: &BuiltinDescriptor,
    builtin: &'static str,
    requested: Option<usize>,
) -> BuiltinResult<()> {
    let dynamic = descriptor.signatures.iter().any(|signature| {
        signature
            .outputs
            .iter()
            .any(|output| output.arity == BuiltinParamArity::Variadic)
    });
    let maximum = descriptor
        .signatures
        .iter()
        .map(|signature| signature.outputs.len())
        .max()
        .unwrap_or(0);
    if !dynamic && requested.is_some_and(|count| count > maximum) {
        return Err(native_error(
            &ERROR_INVALID,
            builtin,
            format!("{builtin}: too many output arguments"),
        ));
    }
    Ok(())
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
