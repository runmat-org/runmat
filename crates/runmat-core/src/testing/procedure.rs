use runmat_test::descriptor::{ProcedureDescriptor, ProcedureKind};
use runmat_value::Value;

use crate::{ExecutableUnit, ProcedureInvocation};

pub(super) fn invocation_for(
    runtime: &runmat_runtime::context::RuntimeContext,
    unit: &ExecutableUnit,
    procedure: &ProcedureDescriptor,
    parameters: &[runmat_test::descriptor::ParameterDescriptor],
) -> Result<ProcedureInvocation, runmat_runtime::RuntimeError> {
    if procedure.kind == ProcedureKind::ScriptSection {
        return Ok(ProcedureInvocation::entrypoint());
    }
    let function_name = function_name(procedure).map_err(projection_error)?;
    let mut arguments = Vec::new();
    let input_count = unit.procedure_input_count(&function_name).ok_or_else(|| {
        projection_error(format!(
            "compiled procedure '{function_name}' is unavailable"
        ))
    })?;
    if input_count > parameters.len() {
        runmat_runtime::testing::ensure_testing_classes();
        let class_name = match procedure.kind {
            ProcedureKind::Method | ProcedureKind::Fixture | ProcedureKind::Teardown => {
                class_name(procedure)
            }
            ProcedureKind::Function
            | ProcedureKind::SuiteFactory
            | ProcedureKind::ScriptSection => None,
        };
        let name =
            runmat_runtime::call::argument_text::copy_str(&procedure.display_name, Some(runtime))?;
        let fields = [("Name", Value::String(name.into()))];
        let test_case = runmat_runtime::testing::object_from_fields_in_context(
            runtime,
            class_name
                .as_deref()
                .unwrap_or("matlab.unittest.FunctionTestCase"),
            fields.iter().map(|(key, value)| (*key, value)),
        )?;
        // The projected TestCase contains only the admitted scalar text above,
        // not newly allocated GC identities.
        arguments.push(test_case.with_owned(|value| value));
    }
    for parameter in parameters {
        arguments.push(super::value::from_json(&parameter.value).map_err(projection_error)?);
    }
    Ok(ProcedureInvocation::function(function_name, arguments))
}

pub(crate) fn function_name(procedure: &ProcedureDescriptor) -> Result<String, String> {
    let segments = semantic_segments(procedure);
    match procedure.kind {
        ProcedureKind::Function | ProcedureKind::SuiteFactory => segments
            .last()
            .map(|name| (*name).to_owned())
            .ok_or_else(|| "function semantic path is empty".into()),
        ProcedureKind::Method => {
            if segments.len() < 2 {
                return Err("method semantic path has no class owner".into());
            }
            Ok(format!(
                "{}.{}",
                segments[segments.len() - 2],
                segments[segments.len() - 1]
            ))
        }
        ProcedureKind::Fixture | ProcedureKind::Teardown => {
            let class = segments
                .first()
                .ok_or_else(|| "fixture semantic path has no class owner".to_string())?;
            Ok(format!("{class}.{}", procedure.display_name))
        }
        ProcedureKind::ScriptSection => Err("script sections use the executable entrypoint".into()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn method_name(procedure: &ProcedureDescriptor) -> Result<&str, String> {
    if procedure.kind != ProcedureKind::Method {
        return Err("test procedure is not a method".into());
    }
    let segments = semantic_segments(procedure);
    if segments.len() < 2 {
        return Err("method semantic path has no class owner".into());
    }
    Ok(segments[segments.len() - 1])
}

fn semantic_segments(procedure: &ProcedureDescriptor) -> Vec<&str> {
    let item = procedure
        .semantic_path
        .rsplit_once("::")
        .map_or(procedure.semantic_path.as_str(), |(_, item)| item);
    item.split('/').collect()
}

fn class_name(procedure: &ProcedureDescriptor) -> Option<String> {
    semantic_segments(procedure)
        .first()
        .map(|name| (*name).to_owned())
}
fn projection_error(message: impl Into<String>) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(message)
        .with_identifier("RunMat:Testing:ProcedureProjection")
        .build()
}
