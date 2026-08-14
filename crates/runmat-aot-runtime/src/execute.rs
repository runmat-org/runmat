use std::{collections::BTreeMap, rc::Rc};

use runmat_execution::ExecutableEntrypointKind;
use runmat_jit::execute::NativeWorkspaceInput;
use runmat_types::ProgramPointId;

use crate::AotProcessInput;

pub fn execute(input: AotProcessInput) -> Result<(), String> {
    let assembly: runmat_native_codegen::NativeAssembly = serde_json::from_slice(&input.native_ir)
        .map_err(|error| format!("standalone Native IR is invalid: {error}"))?;
    assembly
        .verify()
        .map_err(|error| format!("standalone Native IR failed verification: {error}"))?;
    let _: runmat_vm::FunctionRegistry = serde_json::from_slice(&input.program)
        .map_err(|error| format!("standalone program registry is invalid: {error}"))?;
    let resume_points: BTreeMap<ProgramPointId, u64> = serde_json::from_slice(&input.resume_points)
        .map_err(|error| format!("standalone resume-point map is invalid: {error}"))?;
    let entrypoint = assembly.executable_identity.entrypoint_function;
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entrypoint)
        .ok_or_else(|| "standalone entrypoint is absent from Native IR".to_string())?;
    if !function.abi.fixed_inputs.is_empty() || function.abi.varargin.is_some() {
        return Err("standalone entrypoint requires unsupported command-line inputs".into());
    }
    let requested_outputs = function.abi.fixed_outputs.len();
    let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_program_revision(Some(assembly.program.clone()));
    let executor = runmat_jit::GenericExecutor::from_static_entrypoints(
        assembly.clone(),
        BTreeMap::from([(entrypoint, input.entrypoint)]),
        Some(input.program),
        resume_points,
    )
    .map_err(|error| format!("standalone native host initialization failed: {error}"))?;

    let execution = futures::executor::block_on(async {
        if assembly.executable_identity.entrypoint_kind == ExecutableEntrypointKind::Script {
            let local_names = function
                .locals
                .iter()
                .filter_map(|local| local.binding.zip(local.name.clone()))
                .collect();
            executor
                .invoke_workspace_async(
                    entrypoint,
                    NativeWorkspaceInput {
                        local_names,
                        ..NativeWorkspaceInput::default()
                    },
                    requested_outputs,
                    runtime,
                )
                .await
        } else {
            executor
                .invoke_async(entrypoint, Vec::new(), requested_outputs, runtime)
                .await
        }
    })
    .map_err(|error| format!("standalone native execution failed: {error}"))?;
    if let Some(expression) = execution.expression {
        crate::output::value(&expression);
    } else {
        for output in execution.outputs {
            crate::output::value(&output);
        }
    }
    Ok(())
}
