use runmat_test::descriptor::{ProcedureDescriptor, ProcedureKind};
use runmat_value::{ObjectInstance, Value};

use crate::{ExecutableUnit, ProcedureInvocation};

pub(super) fn invocation_for(
    unit: &ExecutableUnit,
    procedure: &ProcedureDescriptor,
    parameters: &[runmat_test::descriptor::ParameterDescriptor],
) -> Result<ProcedureInvocation, String> {
    if procedure.kind == ProcedureKind::ScriptSection {
        return Ok(ProcedureInvocation::entrypoint());
    }
    let function_name = function_name(procedure)?;
    let mut arguments = Vec::new();
    let input_count = unit
        .procedure_input_count(&function_name)
        .ok_or_else(|| format!("compiled procedure '{function_name}' is unavailable"))?;
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
        let mut test_case = ObjectInstance::new(
            class_name.unwrap_or_else(|| "matlab.unittest.FunctionTestCase".into()),
        );
        test_case
            .properties
            .insert("Name".into(), Value::String(procedure.display_name.clone()));
        arguments.push(Value::Object(test_case));
    }
    for parameter in parameters {
        arguments.push(super::value::from_json(&parameter.value)?);
    }
    Ok(ProcedureInvocation::function(function_name, arguments))
}

fn function_name(procedure: &ProcedureDescriptor) -> Result<String, String> {
    let item = procedure
        .semantic_path
        .rsplit_once("::")
        .map_or(procedure.semantic_path.as_str(), |(_, item)| item);
    let segments = item.split('/').collect::<Vec<_>>();
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

fn class_name(procedure: &ProcedureDescriptor) -> Option<String> {
    let item = procedure
        .semantic_path
        .rsplit_once("::")
        .map_or(procedure.semantic_path.as_str(), |(_, item)| item);
    item.split('/').next().map(str::to_owned)
}
