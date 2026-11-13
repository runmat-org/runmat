//! Declaration visibility in the selected VM frame is separate from a store
//! to session state or a language-visible assignment report.

use super::*;

pub(crate) fn prepare_workspace_persistent_slot(
    selected: runmat_runtime::context::RuntimeContextLocalId,
    index: usize,
    context: &runmat_runtime::context::RuntimeContext,
) -> Result<Option<super::frame_assignment::PreparedFrameAssignment>, runmat_runtime::RuntimeError>
{
    let owner = WorkspaceOwner::Runtime(selected);
    if current_owner() != owner {
        return Ok(None);
    }
    let Some(frame) = workspace_frame_identity() else {
        return Ok(None);
    };
    super::frame_assignment::prepare_workspace_frame_import(
        frame,
        Some(context),
        index,
        &Value::Num(0.0),
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::{context::RuntimeContext, execution::RuntimeExecutionService};

    #[test]
    fn persistent_declaration_defines_only_its_selected_frame_slot_without_a_store() {
        runmat_gc::gc_test_context(|| {
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            let other = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            let (runtime, activation) =
                crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
            let frame = activation.frame();
            futures::executor::block_on(activation.scope(runtime.scope(async {
                let mut vars =
                    crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(99.0)]);
                let _frame = set_transient_workspace_state(
                    [("p".into(), 3), ("untouched".into(), 0)]
                        .into_iter()
                        .collect(),
                    HashSet::new(),
                    &mut vars,
                    FrameGlobalBindings::default(),
                );
                let empty =
                    Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap());
                assert!(
                    prepare_workspace_persistent_slot(other.local_identity(), 3, &other)
                        .unwrap()
                        .is_none()
                );
                assert_eq!(workspace_slot_assigned(3), Some(false));
                let prepared =
                    prepare_workspace_persistent_slot(runtime.local_identity(), 3, &runtime)
                        .unwrap()
                        .expect("selected frame");
                prepared.checkpoint_before_publication().unwrap();
                prepared.commit_owned_value(empty.clone());
                assert_eq!(workspace_slot_assigned(3), Some(true));
                assert_eq!(workspace_lookup("p"), Some(empty));
                assert_eq!(workspace_slot_assigned(0), Some(false));
                with_stack(current_owner(), |stack| {
                    let state = &stack.last().unwrap().state;
                    assert!(state.assigned_ids_this_execution.is_empty());
                    assert!(state.assigned_names_this_execution.is_empty());
                });
                assert_eq!(
                    runtime
                        .session_variable_authority()
                        .persistent_named_value("f", "p")
                        .unwrap(),
                    None
                );
            })));
            let _ = take_workspace_completion(frame);
        });
    }
}
