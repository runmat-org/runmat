//! Python interoperability builtins backed by the session foreign runtime.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, ObjectInstance, Value};

use crate::execution::RootedValue;
use crate::{build_runtime_error, BuiltinResult};

#[derive(Clone, Copy)]
enum PythonBuiltinOperation {
    Status,
    Configure,
    Run,
    RunFile,
    Terminate,
}

impl PythonBuiltinOperation {
    #[cfg(not(target_arch = "wasm32"))]
    const fn symbol(self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Configure => "configure",
            Self::Run => "pyrun",
            Self::RunFile => "pyrunfile",
            Self::Terminate => "terminate",
        }
    }
}
pub use runmat_builtins::{
    PYARGS_DESCRIPTOR, PYENV_DESCRIPTOR, PYRUNFILE_DESCRIPTOR, PYRUN_DESCRIPTOR,
    PYTHON_ENVIRONMENT_CLASS, PYTHON_ERROR_HOST_UNAVAILABLE, PYTHON_ERROR_INVALID,
    PYTHON_ERROR_UNAVAILABLE, TERMINATE_DESCRIPTOR,
};

async fn invoke_python(
    context: Option<crate::context::RuntimeContext>,
    operation: PythonBuiltinOperation,
    arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (operation, arguments);
        let _ = context;
        Err(python_error(
            &PYTHON_ERROR_UNAVAILABLE,
            PYTHON_ERROR_UNAVAILABLE.message,
        ))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let symbol = operation.symbol();
        let context = context.ok_or_else(|| {
            python_error(
                &PYTHON_ERROR_HOST_UNAVAILABLE,
                PYTHON_ERROR_HOST_UNAVAILABLE.message,
            )
        })?;
        let service = context
            .service_ports()
            .require_foreign(symbol)
            .map_err(|error| error.into_runtime_error())?
            .clone();
        let arguments = crate::execution::RootedValues::in_context(arguments, &context)?;
        let sequence = context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: runmat_python::PYTHON_ADAPTER_ID.into(),
                    symbol: symbol.into(),
                    arguments: crate::call::RootedFunctionArguments::ordinary(arguments),
                    requested_outputs: crate::current_requested_outputs(),
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
    name = "pyenv",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::python"
)]
async fn pyenv_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    if arguments.is_empty() {
        invoke_python(context, PythonBuiltinOperation::Status, arguments).await
    } else {
        invoke_python(context, PythonBuiltinOperation::Configure, arguments).await
    }
}

#[runtime_builtin(
    name = "pyargs",
    binding_variant = "default",
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
        let name = crate::builtins::common::validation::value_text_scalar_string(&name)
            .ok_or_else(|| invalid_call("pyargs names must be text"))?;
        if name.is_empty() {
            return Err(invalid_call("pyargs names must be non-empty"));
        }
        names.push(Value::String(name.into()));
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::python"
)]
async fn pyrun_builtin(
    context: Option<crate::context::RuntimeContext>,
    mut arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    if arguments.is_empty() {
        return Err(invalid_call("pyrun requires Python source code"));
    }
    let code = arguments.remove(0);
    let outputs = requested_output_names(&mut arguments, "pyrun")?;
    let mut call = vec![code, outputs];
    call.extend(arguments);
    invoke_python(context, PythonBuiltinOperation::Run, call).await
}

#[runtime_builtin(
    name = "pyrunfile",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::python"
)]
async fn pyrunfile_builtin(
    context: Option<crate::context::RuntimeContext>,
    mut arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
    if arguments.is_empty() {
        return Err(invalid_call("pyrunfile requires a Python file"));
    }
    let file = arguments.remove(0);
    let outputs = requested_output_names(&mut arguments, "pyrunfile")?;
    let mut call = vec![file, outputs];
    call.extend(arguments);
    invoke_python(context, PythonBuiltinOperation::RunFile, call).await
}

#[runtime_builtin(
    name = "terminate",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::python"
)]
async fn terminate_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<RootedValue> {
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
    // The foreign adapter validates the environment's private session ticket.
    // Keep the exact caller object; public class checking alone is insufficient.
    invoke_python(context, PythonBuiltinOperation::Terminate, arguments).await
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
    python_error(&PYTHON_ERROR_INVALID, message)
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

#[cfg(test)]
#[path = "python/tests.rs"]
mod tests;
