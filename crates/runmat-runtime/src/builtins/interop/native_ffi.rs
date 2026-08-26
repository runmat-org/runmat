//! Shared-library interoperability builtins backed by the session foreign service.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

const TEXT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Library, header, alias, function, or pointer type name.",
};
const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value passed through the normalized native ABI contract.",
};
const VARIADIC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Arguments required by this shared-library operation.",
};
const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Native return value followed by declared output parameters.",
};

const LOAD_INPUTS: [BuiltinParamDescriptor; 1] = [VARIADIC];
const CALL_INPUTS: [BuiltinParamDescriptor; 1] = [VARIADIC];
const ONE_TEXT_INPUT: [BuiltinParamDescriptor; 1] = [TEXT];
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const POINTER_TYPE_INPUT: [BuiltinParamDescriptor; 1] = [TEXT];
const POINTER_INPUTS: [BuiltinParamDescriptor; 2] = [TEXT, VALUE];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT];

const LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[notfound, warnings] = loadlibrary(library, header, options)",
    inputs: &LOAD_INPUTS,
    outputs: &OUTPUTS,
}];
const CALL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[outputs] = calllib(library, function, arguments)",
    inputs: &CALL_INPUTS,
    outputs: &OUTPUTS,
}];
const ONE_TEXT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "result = shared_library_operation(name)",
    inputs: &ONE_TEXT_INPUT,
    outputs: &OUTPUTS,
}];
const FUNCTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "names = libfunctions(library)",
        inputs: &ONE_TEXT_INPUT,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "signatures = libfunctions(library, '-full')",
        inputs: &LOAD_INPUTS,
        outputs: &OUTPUTS,
    },
];
const POINTER_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "pointer = libpointer()",
        inputs: &NO_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pointer = libpointer(type)",
        inputs: &POINTER_TYPE_INPUT,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pointer = libpointer(type, value)",
        inputs: &POINTER_INPUTS,
        outputs: &OUTPUTS,
    },
];
const STRUCTURE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "value = libstruct(type)",
        inputs: &POINTER_TYPE_INPUT,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "value = libstruct(type, structure)",
        inputs: &POINTER_INPUTS,
        outputs: &OUTPUTS,
    },
];
const SETDATATYPE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "setdatatype(pointer, type, dimensions)",
    inputs: &LOAD_INPUTS,
    outputs: &[],
}];

const ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NATIVE_FFI.UNAVAILABLE",
    identifier: Some("RunMat:Foreign:UnsupportedOnWasm"),
    when: "The current host cannot load a native shared library.",
    message: "Native shared libraries are unavailable on this host.",
};
const ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NATIVE_FFI.INVALID_CALL",
    identifier: Some("RunMat:Foreign:InvalidCall"),
    when: "Arguments do not satisfy the normalized shared-library contract.",
    message: "Invalid shared-library operation.",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_UNAVAILABLE, ERROR_INVALID];

const LOAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const CALL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CALL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const ONE_TEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ONE_TEXT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const FUNCTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FUNCTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const POINTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POINTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const STRUCTURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRUCTURE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const SETDATATYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SETDATATYPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed integer values retain native storage and are range-checked against the declared C ABI type.",
};

async fn invoke_native(
    operation: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<Value> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (operation, arguments, requested_outputs);
        return Err(build_runtime_error(ERROR_UNAVAILABLE.message)
            .with_builtin("native_ffi")
            .with_identifier("RunMat:Foreign:UnsupportedOnWasm")
            .build());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let context = crate::context::legacy::active().ok_or_else(|| {
            build_runtime_error("native shared-library operation has no active runtime context")
                .with_builtin("native_ffi")
                .with_identifier("RunMat:Foreign:HostUnavailable")
                .build()
        })?;
        let service = context
            .service_ports()
            .require_foreign(operation)
            .map_err(|error| error.into_runtime_error())?
            .clone();
        context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: "native-ffi".into(),
                    symbol: operation.into(),
                    arguments,
                    requested_outputs,
                },
            ))
            .await
    }
}

#[runtime_builtin(
    name = "loadlibrary",
    category = "interop/native",
    summary = "Load a C shared library and prepare its declared interface.",
    keywords = "loadlibrary,shared library,c,ffi,native",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::interop::native_ffi::LOAD_DESCRIPTOR),
    integer_audit(crate::builtins::interop::native_ffi::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn loadlibrary_builtin(mut arguments: Vec<Value>) -> BuiltinResult<Value> {
    if arguments.len() == 1 {
        let library = String::try_from(&arguments[0]).map_err(|error| {
            build_runtime_error(format!("loadlibrary: expected a library name: {error}"))
                .with_builtin("loadlibrary")
                .with_identifier("RunMat:Foreign:InvalidCall")
                .build()
        })?;
        let header = std::path::PathBuf::from(library).with_extension("h");
        arguments.push(Value::String(header.display().to_string()));
    }
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
            match name.to_ascii_lowercase().as_str() {
                "alias" => {
                    alias = Some(Value::String(String::try_from(&pair[1]).map_err(|_| {
                        invalid_builtin_call("loadlibrary alias must be text")
                    })?));
                }
                "includepath" => {
                    preparation_options.push(Value::String("includepath".into()));
                    preparation_options.push(Value::String(String::try_from(&pair[1]).map_err(
                        |_| invalid_builtin_call("loadlibrary include path must be text"),
                    )?));
                }
                "addheader" => {
                    preparation_options.push(Value::String("addheader".into()));
                    preparation_options.push(Value::String(String::try_from(&pair[1]).map_err(
                        |_| invalid_builtin_call("loadlibrary additional header must be text"),
                    )?));
                }
                unsupported => {
                    return Err(invalid_builtin_call(format!(
                        "loadlibrary option `{unsupported}` is not available in this interface path"
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
    let report = invoke_native("load_report", arguments, 2).await?;
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
    category = "interop/native",
    summary = "Call a function in a loaded C shared library.",
    keywords = "calllib,shared library,c,ffi,native",
    descriptor(crate::builtins::interop::native_ffi::CALL_DESCRIPTOR),
    integer_audit(crate::builtins::interop::native_ffi::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn calllib_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count().unwrap_or(1);
    invoke_native("call", arguments, requested).await
}

#[runtime_builtin(
    name = "unloadlibrary",
    category = "interop/native",
    summary = "Unload a named C shared library from the current session.",
    keywords = "unloadlibrary,shared library,c,ffi,native",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::interop::native_ffi::ONE_TEXT_DESCRIPTOR),
    integer_audit(crate::builtins::interop::native_ffi::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn unloadlibrary_builtin(alias: Value) -> BuiltinResult<Value> {
    invoke_native("unload", vec![alias], 1).await?;
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "libisloaded",
    category = "interop/native",
    summary = "Test whether a named C shared library is loaded in this session.",
    keywords = "libisloaded,shared library,c,ffi,native",
    descriptor(crate::builtins::interop::native_ffi::ONE_TEXT_DESCRIPTOR),
    integer_audit(crate::builtins::interop::native_ffi::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libisloaded_builtin(alias: Value) -> BuiltinResult<Value> {
    invoke_native("is_loaded", vec![alias], 1).await
}

#[runtime_builtin(
    name = "libfunctions",
    category = "interop/native",
    summary = "List functions declared by a loaded C shared library interface.",
    keywords = "libfunctions,shared library,c,ffi,native",
    descriptor(crate::builtins::interop::native_ffi::FUNCTIONS_DESCRIPTOR),
    integer_audit(crate::builtins::interop::native_ffi::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libfunctions_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke_native("functions", arguments, 1).await
}

#[runtime_builtin(
    name = "libpointer",
    category = "interop/native",
    summary = "Create session-owned typed storage for a native pointer argument.",
    keywords = "libpointer,pointer,shared library,c,ffi,native",
    descriptor(crate::builtins::interop::native_ffi::POINTER_DESCRIPTOR),
    integer_audit(crate::builtins::interop::native_ffi::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libpointer_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke_native("pointer", arguments, 1).await
}

#[runtime_builtin(
    name = "setdatatype",
    category = "interop/native",
    summary = "Declare the type and dimensions of an opaque native pointer.",
    keywords = "setdatatype,libpointer,pointer,shared library,c,ffi,native",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::interop::native_ffi::SETDATATYPE_DESCRIPTOR),
    integer_audit(crate::builtins::interop::native_ffi::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn setdatatype_builtin(mut arguments: Vec<Value>) -> BuiltinResult<Value> {
    if arguments.len() < 3 {
        return Err(invalid_builtin_call(
            "setdatatype expects a pointer, type, and one or more dimensions",
        ));
    }
    arguments.insert(1, Value::String("setdatatype".into()));
    invoke_native("invoke_member", arguments, 1).await?;
    Ok(Value::OutputList(Vec::new()))
}

#[runtime_builtin(
    name = "libstruct",
    category = "interop/native",
    summary = "Create session-owned storage for a structure declared by a loaded library.",
    keywords = "libstruct,structure,shared library,c,ffi,native",
    descriptor(crate::builtins::interop::native_ffi::STRUCTURE_DESCRIPTOR),
    integer_audit(crate::builtins::interop::native_ffi::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::native_ffi"
)]
async fn libstruct_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke_native("structure", arguments, 1).await
}

fn invalid_builtin_call(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("native_ffi")
        .with_identifier("RunMat:Foreign:InvalidCall")
        .build()
}
