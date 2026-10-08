//! Declaration visibility in the selected VM frame is separate from a store
//! to session state or a language-visible assignment report.

use super::*;

pub(crate) fn declare_workspace_persistent_slot(
    selected: runmat_runtime::context::RuntimeContextLocalId,
    index: usize,
    value: &Value,
) {
    let owner = WorkspaceOwner::Runtime(selected);
    if current_owner() != owner {
        return;
    }
    with_active_frame_mut(owner, |frame| {
        let ws = &mut frame.state;
        let Some(name) = ws
            .slot_lifecycle
            .get(&index)
            .map(|slot| slot.name().to_owned())
        else {
            return;
        };
        ws.assigned.insert(name.clone());
        mark_slot_assigned(ws, index, name);
        if ws.values.len() <= index {
            ws.values.resize(index + 1, Value::Num(0.0));
        }
        ws.values[index] = value.clone();
    });
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
                let mut vars = vec![Value::Num(99.0)];
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
                declare_workspace_persistent_slot(other.local_identity(), 3, &empty);
                assert_eq!(workspace_slot_assigned(3), Some(false));
                declare_workspace_persistent_slot(runtime.local_identity(), 3, &empty);
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
