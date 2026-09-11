//! Python interoperability builtins backed by the session foreign runtime.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, ObjectInstance, Value};

use crate::{build_runtime_error, BuiltinResult};

const PYTHON_ADAPTER: &str = "python";
const PYTHON_ENVIRONMENT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("py.PythonEnvironment");

const ARGUMENTS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Arguments supplied to the Python operation.",
};
const RESULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result of the Python operation.",
};
const INPUTS: [BuiltinParamDescriptor; 1] = [ARGUMENTS];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [RESULT];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "result = python_operation(arguments)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.UNAVAILABLE",
    identifier: Some("RunMat:Foreign:UnsupportedOnWasm"),
    when: "The current host has no explicitly provided CPython capability.",
    message: "Python interoperability is unavailable on this host.",
};
const ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.INVALID_CALL",
    identifier: Some("RunMat:Foreign:InvalidCall"),
    when: "Arguments do not satisfy the Python interoperability contract.",
    message: "Invalid Python interoperability operation.",
};
const ERROR_HOST_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.HOST_UNAVAILABLE",
    identifier: Some("RunMat:Foreign:HostUnavailable"),
    when: "A Python operation is invoked without an active runtime context.",
    message: "Python operation has no active runtime context.",
};
const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_UNAVAILABLE, ERROR_INVALID, ERROR_HOST_UNAVAILABLE];
const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Fixed-width integer scalars and arrays retain their exact signed or unsigned Python representation.",
};

async fn invoke_python(operation: &str, arguments: Vec<Value>) -> BuiltinResult<Value> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (operation, arguments);
        Err(python_error(&ERROR_UNAVAILABLE, ERROR_UNAVAILABLE.message))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let context = crate::context::legacy::active()
            .ok_or_else(|| python_error(&ERROR_HOST_UNAVAILABLE, ERROR_HOST_UNAVAILABLE.message))?;
        let service = context
            .service_ports()
            .require_foreign(operation)
            .map_err(|error| error.into_runtime_error())?
            .clone();
        let sequence = context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: PYTHON_ADAPTER.into(),
                    symbol: operation.into(),
                    arguments,
                    requested_outputs: crate::current_requested_outputs(),
                },
            ))
            .await?;
        Ok(crate::call::arguments::project_legacy_builtin_value_abi(
            sequence,
        ))
    }
}

#[runtime_builtin(
    name = "pyenv",
    execution_stack = "process",
    category = "interop/python",
    summary = "Inspect or configure the Python runtime for this session.",
    keywords = "pyenv,python,environment,interpreter",
    descriptor(crate::builtins::interop::python::DESCRIPTOR),
    integer_audit(crate::builtins::interop::python::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::python"
)]
async fn pyenv_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    if arguments.is_empty() {
        invoke_python("status", arguments).await
    } else {
        invoke_python("configure", arguments).await
    }
}

#[runtime_builtin(
    name = "pyargs",
    category = "interop/python",
    summary = "Create a Python keyword-argument bundle.",
    keywords = "pyargs,python,keyword,arguments",
    descriptor(crate::builtins::interop::python::DESCRIPTOR),
    integer_audit(crate::builtins::interop::python::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::python"
)]
fn pyargs_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    if !arguments.len().is_multiple_of(2) {
        return Err(invalid_call("pyargs requires name-value pairs"));
    }
    let mut names = Vec::with_capacity(arguments.len() / 2);
    let mut values = Vec::with_capacity(arguments.len() / 2);
    let mut arguments = arguments.into_iter();
    while let Some(name) = arguments.next() {
        let name =
            String::try_from(&name).map_err(|_| invalid_call("pyargs names must be text"))?;
        if name.is_empty() {
            return Err(invalid_call("pyargs names must be non-empty"));
        }
        names.push(Value::String(name));
        values.push(arguments.next().expect("even argument count"));
    }
    let length = names.len();
    let mut bundle = ObjectInstance::new(runmat_types::standard::PYTHON_ARGUMENTS);
    bundle.properties.insert(
        "Names".into(),
        Value::Cell(CellArray::new(names, 1, length).map_err(invalid_call)?),
    );
    bundle.properties.insert(
        "Values".into(),
        Value::Cell(CellArray::new(values, 1, length).map_err(invalid_call)?),
    );
    Ok(Value::Object(bundle))
}

#[runtime_builtin(
    name = "pyrun",
    execution_stack = "process",
    category = "interop/python",
    summary = "Execute Python code in the session's persistent Python workspace.",
    keywords = "pyrun,python,code,workspace",
    descriptor(crate::builtins::interop::python::DESCRIPTOR),
    integer_audit(crate::builtins::interop::python::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::python"
)]
async fn pyrun_builtin(mut arguments: Vec<Value>) -> BuiltinResult<Value> {
    if arguments.is_empty() {
        return Err(invalid_call("pyrun requires Python source code"));
    }
    let code = arguments.remove(0);
    let outputs = requested_output_names(&mut arguments, "pyrun")?;
    let mut call = vec![code, outputs];
    call.extend(arguments);
    invoke_python("pyrun", call).await
}

#[runtime_builtin(
    name = "pyrunfile",
    execution_stack = "process",
    category = "interop/python",
    summary = "Execute a Python file in an isolated script workspace.",
    keywords = "pyrunfile,python,file,script",
    descriptor(crate::builtins::interop::python::DESCRIPTOR),
    integer_audit(crate::builtins::interop::python::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::python"
)]
async fn pyrunfile_builtin(mut arguments: Vec<Value>) -> BuiltinResult<Value> {
    if arguments.is_empty() {
        return Err(invalid_call("pyrunfile requires a Python file"));
    }
    let file = arguments.remove(0);
    let outputs = requested_output_names(&mut arguments, "pyrunfile")?;
    let mut call = vec![file, outputs];
    call.extend(arguments);
    invoke_python("pyrunfile", call).await
}

#[runtime_builtin(
    name = "terminate",
    execution_stack = "process",
    category = "interop/python",
    summary = "Terminate an out-of-process Python environment.",
    keywords = "terminate,python,environment,process",
    descriptor(crate::builtins::interop::python::DESCRIPTOR),
    integer_audit(crate::builtins::interop::python::INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::python"
)]
async fn terminate_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    if arguments.len() != 1
        || !matches!(
            arguments.first(),
            Some(Value::Object(environment))
                if environment.class_name.is(PYTHON_ENVIRONMENT_CLASS)
        )
    {
        return Err(invalid_call(
            "terminate expects the PythonEnvironment returned by pyenv",
        ));
    }
    invoke_python("terminate", Vec::new()).await
}

fn requested_output_names(arguments: &mut Vec<Value>, builtin: &str) -> BuiltinResult<Value> {
    let requested = crate::current_requested_outputs();
    if requested == 0 {
        return CellArray::new(Vec::new(), 0, 0)
            .map(Value::Cell)
            .map_err(invalid_call);
    }
    if arguments.is_empty() {
        return Err(invalid_call(format!(
            "{builtin} requires Python output names when outputs are requested"
        )));
    }
    Ok(arguments.remove(0))
}

fn invalid_call(message: impl Into<String>) -> crate::RuntimeError {
    python_error(&ERROR_INVALID, message)
}

fn python_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("python");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
