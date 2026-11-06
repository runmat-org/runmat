//! Design values computed by `.m` functions, run natively in a RunMat session.
//!
//! [`call_with_m_values`] calls a design tool with a value host set
//! ([`runmat_cad_wasm::runtime::values_host`]): a part whose dimension calls an `.m` function in the
//! workspace (`base = camBaseRadius(lift = input.lift, …) as m`) regenerates with the value the
//! function returns. Each call runs in a fresh session, so no state carries between calls.

use crate::{call, Files, ToolError};
use runmat_cad::value::m_functions::MRunner;
use runmat_cad_wasm::runtime::values_host::{with_value_host, ValueHost};
use runmat_core::abi::{DiagnosticSeverity, ExecutionRequest, HostExecutionPolicy, SourceInput};
use runmat_core::RunMatSession;
use runmat_value::Value;
use serde_json::Value as Json;

/// Runs `.m` functions in a RunMat session: the call, then the function's file and its folder's
/// other functions as local functions of one script.
pub struct SessionRunner;

impl MRunner for SessionRunner {
    fn call(&self, sources: &[(String, String)], function: &str, arguments: &[f64]) -> Result<Vec<f64>, String> {
        let args: Vec<String> = arguments.iter().map(|a| format!("{a:e}")).collect();
        let mut script = format!("{function}({})\n", args.join(", "));
        for (_, text) in sources {
            script.push('\n');
            script.push_str(text);
            if !text.trim_end().ends_with("end") {
                script.push_str("\nend");
            }
            script.push('\n');
        }
        let mut session = RunMatSession::with_options(false, false).map_err(|e| e.to_string())?;
        let request = ExecutionRequest::for_source(
            SourceInput::Text {
                name: format!("{function}.m"),
                text: script,
            },
            session.compat_mode(),
            HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        let response = futures::executor::block_on(session.execute_request(request));
        let rooted = response.result.map_err(|e| e.to_string())?;
        // Read the result while it is still rooted.
        let outcome = rooted.value();
        if let Some(d) = outcome.diagnostics.iter().find(|d| d.severity == DiagnosticSeverity::Error) {
            return Err(d.message.clone());
        }
        numbers(outcome.flow.durable_workspace_value())
    }

    fn runtime_version(&self) -> String {
        env!("CARGO_PKG_VERSION").into()
    }
}

fn numbers(value: Option<&Value>) -> Result<Vec<f64>, String> {
    match value {
        Some(Value::Num(v)) => Ok(vec![*v]),
        Some(Value::Bool(b)) => Ok(vec![f64::from(u8::from(*b))]),
        Some(Value::Tensor(t)) => Ok(t.materialize_f64()),
        Some(other) => Err(format!("returned {}, not a number", runmat_core::matlab_class_name(other))),
        None => Err("returned no value".into()),
    }
}

/// Runs `f` with `.m` design values served from `files` by a [`SessionRunner`]: parts regenerate
/// with their computed values, and circuits compile with theirs
/// ([`crate::circuit_values::circuit_options`]).
pub fn with_m_values<R>(files: &Files, f: impl FnOnce() -> R) -> R {
    let host = ValueHost {
        files: files.clone(),
        runner: Box::new(SessionRunner),
    };
    with_value_host(host, f)
}

/// [`call`] with `.m` design values served from `files` by a [`SessionRunner`].
pub fn call_with_m_values(tool: &str, input: Json, files: &Files) -> Result<Json, ToolError> {
    with_m_values(files, || call(tool, input, files))
}
