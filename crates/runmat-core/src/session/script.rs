//! Fresh script outputs use the VM's actual assignment inventory, never slot order.
use super::*;

impl RunMatSession {
    pub(super) async fn invoke_script_interpreted(
        &mut self,
        unit: &crate::ExecutableUnit,
        requested_outputs: usize,
    ) -> std::result::Result<Value, RuntimeError> {
        let names = unit
            .bytecode()
            .var_names
            .iter()
            .map(|(slot, name)| (name.clone(), *slot))
            .collect();
        let _workspace = runmat_vm::push_pending_workspace(names, Default::default());
        let mut variables = vec![Value::Num(0.0); unit.bytecode().var_count];
        let result = runmat_vm::interpret_with_vars_in_context(
            unit.bytecode(),
            &mut variables,
            Some(&unit.source().relative_path),
            self.runtime_context
                .clone()
                .with_program_revision(Some(unit.revision().program_revision.clone())),
        )
        .await;
        let workspace = runmat_vm::take_updated_workspace_state();
        let _ = runmat_vm::take_updated_workspace_assigned_report();
        let runmat_vm::InterpreterOutcome::Completed(values) = result?;
        if requested_outputs == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let slot = workspace.as_ref().and_then(|workspace| {
            workspace
                .assigned
                .contains("ans")
                .then(|| workspace.names.get("ans").copied())
                .flatten()
        });
        slot.and_then(|slot| values.get(slot).cloned())
            .ok_or_else(|| {
                runmat_runtime::build_runtime_error(
                    "script did not produce the requested ans value",
                )
                .with_identifier("RunMat:ExecutableOutput")
                .build()
            })
    }
}
