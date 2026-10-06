//! MATLAB-compatible `load` builtin for RunMat.

use std::collections::HashMap;
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use regex::Regex;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, StructValue, Value};

use super::authority::MatFsPath;
use crate::builtins::introspection::object_serialization::restore_value_from_mat_load;
use crate::context::host_effect::HostEffectResourceLivenessLease;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use runmat_builtins::{
    LOAD_ERROR_FILENAME, LOAD_ERROR_INVALID_ARGUMENT, LOAD_ERROR_INVALID_OPTION, LOAD_ERROR_IO,
    LOAD_ERROR_OUTPUT_COUNT, LOAD_ERROR_SELECTION, LOAD_ERROR_WORKSPACE,
};

mod admission;
use crate::context::host_effect::{HostEffectReceipt, MatDecodeOutput};
#[runtime_builtin(
    name = "load",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::load"
)]
async fn load_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let requested_outputs = match crate::output_count::current_output_scope() {
        crate::output_count::InvocationOutputScope::Requested(count) => Some(count),
        crate::output_count::InvocationOutputScope::Implicit => None,
        crate::output_count::InvocationOutputScope::Unscoped => {
            crate::output_context::requested_output_count()
        }
    };
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(load_error_with(
            &LOAD_ERROR_OUTPUT_COUNT,
            "load supports at most one output argument",
        ));
    }
    let mut workspace_target = if requested_outputs == Some(0) {
        Some(LoadWorkspaceTarget::capture(runtime.as_ref())?)
    } else {
        None
    };
    let execution = evaluate_with_context(
        &args,
        runtime.as_ref(),
        if requested_outputs == Some(0) {
            MatDecodeOutput::Workspace
        } else {
            MatDecodeOutput::Struct
        },
    )
    .await?;
    let LoadExecution { eval, receipt } = execution;

    // The same invocation-owned output request remains authoritative across
    // the awaited decode/restoration work.
    let result = (|| {
        if requested_outputs == Some(0) {
            let target = workspace_target.as_mut().ok_or_else(|| {
                load_error_with(
                    &LOAD_ERROR_WORKSPACE,
                    "load: statement-form workspace target was not retained",
                )
            })?;
            target.assign_all(eval.variables(), receipt.as_ref())?;
            Ok(Value::OutputList(Vec::new()))
        } else {
            if let Some(receipt) = receipt.as_ref() {
                receipt.checkpoint()?;
            }
            Ok(eval.first_output())
        }
    })()
    .and_then(|value| {
        if let Some(receipt) = receipt.as_ref() {
            receipt.checkpoint()?;
        }
        Ok(value)
    });
    admission::finish(receipt, &result);
    result
}

#[derive(Clone, Debug)]
pub struct LoadEval {
    variables: Vec<(String, Value)>,
}

impl LoadEval {
    pub fn first_output(&self) -> Value {
        let mut st = StructValue::new();
        for (name, value) in &self.variables {
            st.fields.insert(name.clone(), value.clone());
        }
        Value::Struct(st)
    }

    pub fn variables(&self) -> &[(String, Value)] {
        &self.variables
    }

    pub fn into_variables(self) -> Vec<(String, Value)> {
        self.variables
    }
}

struct LoadRequest {
    variables: Vec<String>,
    regex_patterns: Vec<Regex>,
    regex_tokens: Vec<String>,
}

struct LoadExecution {
    eval: LoadEval,
    receipt: Option<HostEffectReceipt>,
}

enum LoadWorkspaceTarget {
    Canonical(crate::workspace::BaseWorkspaceProjection),
    Service(Rc<dyn crate::context::RuntimeWorkspaceService>),
    Facade,
}

impl LoadWorkspaceTarget {
    fn capture(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<Self> {
        let Some(service) = runtime.and_then(|runtime| runtime.service_ports().workspace()) else {
            return Ok(Self::Facade);
        };
        if let Some(owner) = service.base_workspace_owner() {
            return owner
                .begin_projection()
                .map(Self::Canonical)
                .map_err(crate::workspace::base_workspace_error_to_runtime)
                .map_err(load_workspace_flow);
        }
        Ok(Self::Service(service.clone()))
    }

    fn assign_all(
        &mut self,
        variables: &[(String, Value)],
        receipt: Option<&HostEffectReceipt>,
    ) -> BuiltinResult<()> {
        match self {
            Self::Canonical(projection) => {
                for (name, value) in variables {
                    checkpoint(receipt)?;
                    projection
                        .assign(name, value.clone())
                        .map_err(crate::workspace::base_workspace_error_to_runtime)
                        .map_err(load_workspace_flow)?;
                }
                checkpoint(receipt)?;
                projection
                    .publish()
                    .map(|_| ())
                    .map_err(crate::workspace::base_workspace_error_to_runtime)
                    .map_err(load_workspace_flow)
            }
            Self::Service(service) => {
                for (name, value) in variables {
                    checkpoint(receipt)?;
                    service
                        .assign(name, value.clone())
                        .map_err(load_workspace_flow)?;
                }
                Ok(())
            }
            Self::Facade => {
                for (name, value) in variables {
                    checkpoint(receipt)?;
                    crate::workspace::try_assign(name, value.clone())
                        .map_err(load_workspace_flow)?;
                }
                Ok(())
            }
        }
    }
}

fn checkpoint(receipt: Option<&HostEffectReceipt>) -> BuiltinResult<()> {
    if let Some(receipt) = receipt {
        receipt.checkpoint()?;
    }
    Ok(())
}

const BUILTIN_NAME: &str = "load";

fn load_error(message: impl Into<String>) -> RuntimeError {
    load_error_with(&LOAD_ERROR_INVALID_ARGUMENT, message)
}

fn load_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn load_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn load_workspace_flow(mut error: RuntimeError) -> RuntimeError {
    error.context = error.context.with_builtin(BUILTIN_NAME);
    error
}

pub async fn evaluate(args: &[Value]) -> BuiltinResult<LoadEval> {
    evaluate_with_context(args, None, MatDecodeOutput::Struct)
        .await
        .map(|execution| execution.eval)
}

async fn evaluate_with_context(
    args: &[Value],
    runtime: Option<&crate::context::RuntimeContext>,
    output: MatDecodeOutput,
) -> BuiltinResult<LoadExecution> {
    if args.iter().any(crate::dispatcher::value_contains_gpu) {
        return Err(load_error_with(
            &LOAD_ERROR_INVALID_ARGUMENT,
            "load: filename, options, and variable selectors must be host text values",
        ));
    }
    preflight_known_options(args)?;
    // Filename/options/selectors are text controls. A numeric resident handle
    // cannot become an admitted text control by downloading it first.
    let host_args = args.to_vec();

    let invocation = parse_invocation(&host_args).await?;

    let mut path_value = if let Some(path) = invocation.path_value {
        path
    } else {
        Value::from("matlab.mat")
    };

    if invocation.path_was_default {
        if let Ok(override_path) = std::env::var("RUNMAT_LOAD_DEFAULT_PATH") {
            path_value = Value::from(override_path);
        }
    }

    let mut regex_patterns = Vec::with_capacity(invocation.regex_tokens.len());
    for pattern in &invocation.regex_tokens {
        let regex = Regex::new(pattern).map_err(|err| {
            load_error_with_source(
                &LOAD_ERROR_INVALID_OPTION,
                format!("load: invalid regular expression '{pattern}': {err}"),
                err,
            )
        })?;
        regex_patterns.push(regex);
    }

    let request = LoadRequest {
        variables: invocation.variables,
        regex_patterns,
        regex_tokens: invocation.regex_tokens,
    };
    let path = normalise_path(&path_value)?;
    let source = MatFsPath::capture(path).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to resolve MAT path: {err}"),
            err,
        )
    })?;
    let receipt = admission::select(runtime, &source, &request, output)?;
    let result = async {
        let entries = read_mat_file_on_with_resource_liveness(
            &source,
            runtime,
            receipt
                .as_ref()
                .map(|selected| selected.resource_liveness_lease()),
        )
        .await?;
        let selected = select_variables(&entries, &request)?;
        Ok(LoadEval {
            variables: selected,
        })
    }
    .await;
    match result {
        Ok(eval) => Ok(LoadExecution { eval, receipt }),
        Err(error) => {
            admission::finish_error(receipt, &error);
            Err(error)
        }
    }
}

mod controls;
use controls::{normalise_path, parse_invocation, preflight_known_options, select_variables};

mod reader;
pub use reader::decode_workspace_from_mat_bytes;
#[cfg(test)]
pub(crate) use reader::read_mat_file;
pub(crate) use reader::{
    read_mat_file_for_builtin, read_mat_file_for_builtin_on_with_resource_liveness,
    read_mat_file_on_with_resource_liveness,
};

mod decoder;
#[cfg(test)]
use decoder::mat_array_to_value;
use decoder::read_mat_reader;

#[cfg(test)]
#[path = "load/tests.rs"]
pub(crate) mod tests;
