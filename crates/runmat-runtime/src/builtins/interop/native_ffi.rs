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
    NativeBuildOption, NativeLoadOption, CALLLIB_DESCRIPTOR, CLIBGEN_BUILD_INTERFACE_DESCRIPTOR,
    LIBFUNCTIONS_DESCRIPTOR, LIBISLOADED_DESCRIPTOR, LIBPOINTER_DESCRIPTOR, LIBSTRUCT_DESCRIPTOR,
    LOADLIBRARY_DESCRIPTOR, SETDATATYPE_DESCRIPTOR, UNLOADLIBRARY_DESCRIPTOR,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::execution::RootedValue;
use crate::{build_runtime_error, BuiltinResult};

async fn invoke_native(
    context: Option<crate::context::RuntimeContext>,
    operation: NativeFfiOperation,
    arguments: crate::call::BuiltinArguments,
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
                    arguments: arguments.into_function_arguments(),
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
    name = "clibgen.buildInterface",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn clibgen_build_interface_builtin(
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    validate_output_count(
        &CLIBGEN_BUILD_INTERFACE_DESCRIPTOR,
        "clibgen.buildInterface",
        requested_outputs,
    )?;
    if arguments.len() < 3 || arguments.len().is_multiple_of(2) {
        return Err(invalid_builtin_call(
            "clibgen.buildInterface requires a header and name-value pairs including Libraries",
        ));
    }
    validate_text_argument(&arguments[0], "clibgen.buildInterface header")?;
    let mut libraries = false;
    for index in (1..arguments.len()).step_by(2) {
        let name = text_argument(
            &arguments[index],
            "buildInterface option names",
            context.as_ref(),
        )?;
        let option = NativeBuildOption::parse(&name).ok_or_else(|| {
            invalid_builtin_call("unsupported C ABI interface preparation option")
        })?;
        validate_text_argument(&arguments[index + 1], "buildInterface option value")?;
        libraries |= option == NativeBuildOption::Libraries;
    }
    if !libraries {
        return Err(invalid_builtin_call(
            "buildInterface requires the Libraries option",
        ));
    }
    invoke_native(context, NativeFfiOperation::BuildInterface, arguments, 1).await
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&LOADLIBRARY_DESCRIPTOR, "loadlibrary", requested_outputs)?;
    if arguments.is_empty() {
        return Err(invalid_builtin_call("loadlibrary expects a library path"));
    }
    if arguments.len() > 2 && !(arguments.len() - 2).is_multiple_of(2) {
        return Err(invalid_builtin_call(
            "loadlibrary options must be supplied as name-value pairs",
        ));
    }
    if arguments.len() > 1 {
        validate_text_argument(&arguments[1], "loadlibrary header")?;
    }
    let mut alias_index = None;
    let mut preparation_count = 0usize;
    for (offset, pair) in arguments
        .view()
        .range(2.min(arguments.len())..)?
        .chunks_exact(2)
        .enumerate()
    {
        let name = text_argument(&pair[0], "loadlibrary option names", context.as_ref())?;
        match NativeLoadOption::parse(&name) {
            Some(NativeLoadOption::Alias) => {
                validate_text_argument(&pair[1], "loadlibrary alias")?;
                alias_index = Some(2 + offset * 2 + 1);
            }
            Some(NativeLoadOption::IncludePath) => {
                validate_text_argument(&pair[1], "loadlibrary include path")?;
                preparation_count += 2;
            }
            Some(NativeLoadOption::AddHeader) => {
                validate_text_argument(&pair[1], "loadlibrary additional header")?;
                preparation_count += 2;
            }
            None => {
                return Err(invalid_builtin_call(format!(
                    "loadlibrary option `{}` is not available in this interface path",
                    name.to_ascii_lowercase()
                )))
            }
        }
    }
    let library = if arguments.len() == 1 || alias_index.is_none() {
        Some(text_argument(
            &arguments[0],
            "loadlibrary library name",
            context.as_ref(),
        )?)
    } else {
        None
    };
    let runtime = transform_runtime(context.as_ref())?;
    let header = if arguments.len() == 1 {
        Some(default_header(
            library.as_ref().expect("default header library"),
            runtime,
        )?)
    } else {
        None
    };
    let alias = if alias_index.is_none() {
        let library = library.as_ref().expect("default alias library");
        let stem = std::path::Path::new(library.as_str())
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| invalid_builtin_call("loadlibrary path has no valid file name"))?;
        Some(crate::call::argument_text::copy_str(stem, Some(runtime))?)
    } else {
        None
    };
    let count = 3usize.checked_add(preparation_count).ok_or_else(|| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "RunMat:Call:Arguments",
            "native argument extent overflow",
            crate::context::host_allocation::HostAllocationError::Overflow,
        )
    })?;
    // The cursor advances once over original options; there is no selection
    // Vec or repeated per-final-slot search through the option tail.
    let mut cursor = 2usize;
    let mut pending_value = None;
    let arguments = arguments.try_map_preserving_roots_in_context(
        count,
        runtime,
        |slot, source| match slot {
            0 => Ok(source[0].clone()),
            1 => Ok(header
                .as_ref()
                .map(|text| Value::String(runmat_value::StringElement::Text(text.clone())))
                .unwrap_or_else(|| source[1].clone())),
            2 => match alias_index {
                Some(index) => text_argument(&source[index], "loadlibrary alias", Some(runtime))
                    .map(|text| Value::String(text.into())),
                None => Ok(Value::String(runmat_value::StringElement::Text(
                    alias.as_ref().expect("prepared alias").clone(),
                ))),
            },
            _ => {
                if let Some(index) = pending_value.take() {
                    return text_argument(
                        &source[index],
                        "loadlibrary preparation value",
                        Some(runtime),
                    )
                    .map(|text| Value::String(text.into()));
                }
                while cursor < source.len() {
                    let index = cursor;
                    cursor += 2;
                    let name =
                        text_argument(&source[index], "loadlibrary option names", Some(runtime))?;
                    let option = NativeLoadOption::parse(&name).expect("validated option");
                    if matches!(option, NativeLoadOption::Alias) {
                        continue;
                    }
                    pending_value = Some(index + 1);
                    return crate::call::argument_text::copy_str(
                        option.canonical_name(),
                        Some(runtime),
                    )
                    .map(|text| Value::String(text.into()));
                }
                Err(invalid_builtin_call(
                    "loadlibrary preparation source ended early",
                ))
            }
        },
    )?;
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    if arguments.len() < 2 {
        return Err(invalid_builtin_call(
            "calllib requires a library alias and function name",
        ));
    }
    text_argument(&arguments[0], "calllib library alias", context.as_ref())?;
    text_argument(&arguments[1], "calllib function name", context.as_ref())?;
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
    text_argument(&alias, "unloadlibrary alias", context.as_ref())?;
    let arguments = crate::call::BuiltinArguments::from_borrowed(
        std::slice::from_ref(&alias),
        context.as_ref(),
    )?;
    invoke_native(context, NativeFfiOperation::Unload, arguments, 1).await?;
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
    text_argument(&alias, "libisloaded alias", context.as_ref())?;
    let arguments = crate::call::BuiltinArguments::from_borrowed(
        std::slice::from_ref(&alias),
        context.as_ref(),
    )?;
    invoke_native(context, NativeFfiOperation::IsLoaded, arguments, 1).await
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&LIBFUNCTIONS_DESCRIPTOR, "libfunctions", requested_outputs)?;
    if !(1..=2).contains(&arguments.len()) {
        return Err(invalid_builtin_call(
            "libfunctions expects an alias and optional -full flag",
        ));
    }
    text_argument(&arguments[0], "libfunctions alias", context.as_ref())?;
    if let Some(option) = arguments.get(1) {
        let option = text_argument(option, "libfunctions option", context.as_ref())?;
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&LIBPOINTER_DESCRIPTOR, "libpointer", requested_outputs)?;
    if arguments.len() > 2 {
        return Err(invalid_builtin_call(
            "libpointer expects an optional type and value",
        ));
    }
    if let Some(pointer_type) = arguments.first() {
        text_argument(pointer_type, "libpointer type", context.as_ref())?;
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
    arguments: crate::call::BuiltinArguments,
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
    text_argument(&arguments[1], "setdatatype pointer type", context.as_ref())?;
    let runtime = transform_runtime(context.as_ref())?;
    let operation = crate::call::argument_text::copy_str("setdatatype", Some(runtime))?;
    let operation = RootedValue::in_context(
        Value::String(runmat_value::StringElement::Text(operation)),
        runtime,
    )?;
    let arguments = arguments.try_insert_in_context(1, operation, runtime)?;
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    validate_output_count(&LIBSTRUCT_DESCRIPTOR, "libstruct", requested_outputs)?;
    if !(1..=2).contains(&arguments.len()) {
        return Err(invalid_builtin_call(
            "libstruct expects a type and optional scalar structure",
        ));
    }
    text_argument(&arguments[0], "libstruct type", context.as_ref())?;
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

fn validate_text_argument(value: &Value, label: &str) -> BuiltinResult<()> {
    let supported = match value {
        Value::String(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::StringArray(array) => array.data().len() == 1,
        Value::CharArray(array) => array.rows == 1 && array.shape.len() <= 2,
        _ => false,
    };
    if supported {
        Ok(())
    } else {
        Err(invalid_builtin_call(format!("{label} must be text")))
    }
}

fn text_argument(
    value: &Value,
    label: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<runmat_value::Utf8Text> {
    validate_text_argument(value, label)?;
    crate::call::argument_text::convert(value, runtime)
}

fn transform_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<&crate::context::RuntimeContext> {
    runtime.ok_or_else(|| {
        #[cfg(not(target_arch = "wasm32"))]
        {
            native_error(
                &ERROR_HOST_UNAVAILABLE,
                "native_ffi",
                ERROR_HOST_UNAVAILABLE.message,
            )
        }
        #[cfg(target_arch = "wasm32")]
        {
            native_error(&ERROR_UNAVAILABLE, "native_ffi", ERROR_UNAVAILABLE.message)
        }
    })
}

fn default_header(
    library: &runmat_value::Utf8Text,
    runtime: &crate::context::RuntimeContext,
) -> BuiltinResult<runmat_value::Utf8Text> {
    let path = std::path::Path::new(library.as_str());
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(library.clone());
    };
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("UTF-8 file name has a UTF-8 stem");
    let end = library
        .rfind(file_name)
        .expect("file name is borrowed from path")
        + stem.len();
    crate::call::argument_text::copy_parts(&[&library[..end], ".h"], Some(runtime))
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

#[cfg(test)]
#[test]
fn exact_default_header_preserves_native_path_extension_semantics() {
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    for path in [
        "libfoo.so",
        "dir/λ.tar.gz",
        ".hidden",
        "foo/",
        "foo/.",
        "..",
        "/",
        "file.",
    ] {
        let source: runmat_value::Utf8Text = path.into();
        let expected = std::path::Path::new(path).with_extension("h");
        assert_eq!(
            default_header(&source, &runtime).unwrap().as_str(),
            expected.to_str().unwrap()
        );
    }
}
