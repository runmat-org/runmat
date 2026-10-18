use super::*;
use crate::bytecode::program::{LexicalBindingKind, LexicalBindingSlot};
use runmat_types::BindingId;
use std::error::Error;

#[test]
fn committed_workspace_remove_retires_the_owned_lexical_cell() {
    runmat_gc::gc_test_context(|| {
        reset_thread_state_for_tests();
        let (environment, _roots) =
            runmat_runtime::call::lexical::allocate_environment(None).unwrap();
        let binding = BindingId(13);
        assert!(
            runmat_runtime::call::lexical::reserve_owned_binding(&environment, binding).unwrap()
        );
        runmat_runtime::call::lexical::write_owned_binding(&environment, binding, Value::Num(7.0))
            .unwrap();
        let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(7.0)]);
        let _frame = set_workspace_state(
            HashMap::from([("shared".to_string(), 0)]),
            HashSet::from(["shared".to_string()]),
            &mut vars,
            FrameGlobalBindings::default(),
        );
        attach_lexical_workspace_bindings(
            Some(&environment),
            &[LexicalBindingSlot {
                binding,
                slot: 0,
                kind: LexicalBindingKind::Owned,
            }],
        )
        .unwrap();
        workspace_assign("shared", Value::Num(8.0)).unwrap();
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, binding).unwrap(),
            Some(Value::Num(8.0))
        );
        workspace_remove("shared").unwrap();
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, binding).unwrap(),
            None
        );
        assert_eq!(workspace_slot_assigned(0), Some(false));
        runmat_runtime::call::lexical::write_owned_binding(&environment, binding, Value::Num(11.0))
            .unwrap();
        refresh_workspace_shared_slot(0, &Value::Num(11.0)).unwrap();
        assert_eq!(workspace_lookup("shared"), Some(Value::Num(11.0)));
        drop(_frame);
        reset_thread_state_for_tests();
    });
}

#[test]
fn clearing_an_anonymous_transit_snapshot_does_not_clear_its_ancestor() {
    runmat_gc::gc_test_context(|| {
        reset_thread_state_for_tests();
        let (environment, _roots) =
            runmat_runtime::call::lexical::allocate_environment(None).unwrap();
        let binding = BindingId(17);
        assert!(
            runmat_runtime::call::lexical::reserve_owned_binding(&environment, binding).unwrap()
        );
        runmat_runtime::call::lexical::write_owned_binding(&environment, binding, Value::Num(9.0))
            .unwrap();
        let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(3.0)]);
        let _frame = set_workspace_state(
            HashMap::from([("shared".to_string(), 0)]),
            HashSet::from(["shared".to_string()]),
            &mut vars,
            FrameGlobalBindings::default(),
        );
        attach_lexical_workspace_bindings(
            Some(&environment),
            &[LexicalBindingSlot {
                binding,
                slot: 0,
                kind: LexicalBindingKind::Transit,
            }],
        )
        .unwrap();
        workspace_assign("shared", Value::Num(4.0)).unwrap();
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, binding).unwrap(),
            Some(Value::Num(9.0))
        );
        workspace_remove("shared").unwrap();
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, binding).unwrap(),
            Some(Value::Num(9.0))
        );
        drop(_frame);
        reset_thread_state_for_tests();
    });
}

#[test]
fn transactional_completion_publishes_only_after_commit_and_retires_the_owned_cell() {
    fn completion(
        runtime: &runmat_runtime::context::RuntimeContext,
        value: Option<Value>,
    ) -> (WorkspaceValueSnapshot, WorkspaceAssignedReport) {
        let gate = runmat_runtime::context::host_allocation::FrameCollectionGate::unbounded();
        let _installed = gate.install_unbounded().expect("test collection gate");
        let allocator = gate.allocator();
        let mut names = frame_metadata::empty_map(allocator.clone());
        names.insert(WorkspaceName::unbounded("shared"), 0);
        let mut assigned = frame_metadata::empty_set(allocator.clone());
        let mut values = FrameVec::new_in(allocator.clone());
        values
            .try_reserve_exact(1)
            .expect("test completion slot backing");
        let is_assigned = value.is_some();
        values.push(match value {
            Some(value) => crate::interpreter::vars::InterpreterSlot::Retained(
                runmat_runtime::workspace::WorkspaceSlotReadLease::prepare_owned(value, runtime)
                    .expect("retain completed lexical value"),
            ),
            None => crate::interpreter::vars::InterpreterSlot::Owned(Value::Num(0.0)),
        });
        let mut vars = crate::interpreter::vars::InterpreterVars::from_retained(values);
        drop(
            vars.promote_shared(None, None)
                .expect("completion uses the same shared control as a production frame"),
        );
        let mut report = WorkspaceAssignedReport {
            ids: frame_metadata::empty_set(allocator.clone()),
            names: frame_metadata::empty_set(allocator.clone()),
            removed_ids: frame_metadata::empty_set(allocator.clone()),
            removed_names: frame_metadata::empty_set(allocator),
        };
        if is_assigned {
            assigned.insert(WorkspaceName::unbounded("shared"));
            report.names.insert(WorkspaceName::unbounded("shared"));
            report.ids.insert(0);
        } else {
            report
                .removed_names
                .insert(WorkspaceName::unbounded("shared"));
            report.removed_ids.insert(0);
        }
        (
            WorkspaceValueSnapshot {
                vars,
                names,
                assigned,
                global_aliases: FrameGlobalBindings::default(),
                collection_gate: gate,
            },
            report,
        )
    }

    runmat_gc::gc_test_context(|| {
        reset_thread_state_for_tests();
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (environment, _roots) =
            runmat_runtime::call::lexical::allocate_environment(None).unwrap();
        let binding = BindingId(18);
        assert!(
            runmat_runtime::call::lexical::reserve_owned_binding(&environment, binding).unwrap()
        );
        runmat_runtime::call::lexical::write_owned_binding(&environment, binding, Value::Num(7.0))
            .unwrap();
        let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(7.0)]);
        let _frame = set_workspace_state(
            HashMap::from([("shared".to_string(), 0)]),
            HashSet::from(["shared".to_string()]),
            &mut vars,
            FrameGlobalBindings::default(),
        );
        attach_lexical_workspace_bindings(
            Some(&environment),
            &[LexicalBindingSlot {
                binding,
                slot: 0,
                kind: LexicalBindingKind::Owned,
            }],
        )
        .unwrap();
        let frame = workspace_frame_identity().expect("active frame");

        let (snapshot, report) = completion(&runtime, Some(Value::Num(9.0)));
        let rejected =
            target_completion::PreparedTargetCompletion::prepare(frame, None, snapshot, report)
                .unwrap();
        rejected.checkpoint_before_publication().unwrap();
        drop(rejected);
        assert_eq!(workspace_lookup("shared"), Some(Value::Num(7.0)));
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, binding).unwrap(),
            Some(Value::Num(7.0))
        );

        let (snapshot, report) = completion(&runtime, Some(Value::Num(9.0)));
        let committed =
            target_completion::PreparedTargetCompletion::prepare(frame, None, snapshot, report)
                .unwrap();
        committed.checkpoint_before_publication().unwrap();
        committed.commit().unwrap();
        reconcile_workspace_vars(&mut vars);
        assert_eq!(*vars.get(0).unwrap(), Value::Num(9.0));
        assert_eq!(workspace_lookup("shared"), Some(Value::Num(9.0)));
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, binding).unwrap(),
            Some(Value::Num(9.0))
        );

        let (snapshot, report) = completion(&runtime, None);
        let retired =
            target_completion::PreparedTargetCompletion::prepare(frame, None, snapshot, report)
                .unwrap();
        retired.checkpoint_before_publication().unwrap();
        retired.commit().unwrap();
        assert_eq!(workspace_lookup("shared"), None);
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, binding).unwrap(),
            None
        );
        drop(_frame);
        reset_thread_state_for_tests();
    });
}

#[test]
fn committed_vm_assignment_retains_the_lexical_cause_without_replaying_the_write() {
    runmat_gc::gc_test_context(|| {
        reset_thread_state_for_tests();
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        runtime.set_error_namespace("Acme");
        let (runtime, activation) =
            crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
        futures::executor::block_on(activation.scope(runtime.scope(async {
            let (environment, _roots) =
                runmat_runtime::call::lexical::allocate_environment(None).unwrap();
            let binding = BindingId(19);
            let mut vars =
                crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(7.0)]);
            let _frame = set_workspace_state(
                HashMap::from([("shared".to_string(), 0)]),
                HashSet::from(["shared".to_string()]),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            attach_lexical_workspace_bindings(
                Some(&environment),
                &[LexicalBindingSlot {
                    binding,
                    slot: 0,
                    kind: LexicalBindingKind::Owned,
                }],
            )
            .unwrap();
            // The unreserved Owned cell fails only after the frame publication.
            let error = workspace_assign("shared", Value::Num(8.0)).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("Acme:Workspace:LexicalPublication")
            );
            assert_eq!(
                error.gpu_gather_retry,
                runmat_runtime::runtime_error::GpuGatherRetry::Never
            );
            let cause = error
                .source()
                .and_then(|source| source.downcast_ref::<runmat_runtime::RuntimeError>())
                .expect("original lexical RuntimeError is retained");
            assert_eq!(cause.identifier(), Some("Acme:LexicalEnvironment"));
            assert_eq!(workspace_lookup("shared"), Some(Value::Num(8.0)));
            assert_eq!(workspace_slot_assigned(0), Some(true));
            drop(_frame);
        })));
        reset_thread_state_for_tests();
    });
}

#[test]
fn assignin_atomically_updates_the_caller_frame_and_owned_lexical_binding() {
    runmat_gc::gc_test_context(|| {
        reset_thread_state_for_tests();
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        runtime.set_error_namespace("Acme");
        let (runtime, activation) =
            crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
        futures::executor::block_on(activation.scope(runtime.scope(async {
            let (environment, _roots) =
                runmat_runtime::call::lexical::allocate_environment(None).unwrap();
            let binding = BindingId(23);
            assert!(
                runmat_runtime::call::lexical::reserve_owned_binding(&environment, binding)
                    .unwrap()
            );
            runmat_runtime::call::lexical::write_owned_binding(
                &environment,
                binding,
                Value::Num(7.0),
            )
            .unwrap();
            let mut vars =
                crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(7.0)]);
            let _frame = set_workspace_state(
                HashMap::from([("shared".to_string(), 0)]),
                HashSet::from(["shared".to_string()]),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            attach_lexical_workspace_bindings(
                Some(&environment),
                &[LexicalBindingSlot {
                    binding,
                    slot: 0,
                    kind: LexicalBindingKind::Owned,
                }],
            )
            .unwrap();
            let (child_runtime, child_activation) =
                crate::runtime::workspace_service::VmWorkspaceService::install(
                    runtime.clone(),
                    None,
                );
            let invocation_runtime = child_runtime.clone();
            let stack = child_activation
                .scope(child_runtime.scope(async move {
                    let mut child_vars =
                        crate::interpreter::vars::InterpreterVars::from_owned(Vec::new());
                    let _child_frame = set_workspace_state(
                        HashMap::new(),
                        HashSet::new(),
                        &mut child_vars,
                        FrameGlobalBindings::default(),
                    );
                    let mut stack =
                        crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
                            Value::String("caller".into()),
                            Value::String("shared".into()),
                            Value::Num(8.0),
                        ]);
                    crate::call::builtins::vm_dynamic_workspace_builtin(
                        &invocation_runtime,
                        &mut stack,
                        "assignin",
                        3,
                        0,
                        &crate::bytecode::FunctionRegistry::default(),
                        None,
                    )
                    .await
                    .unwrap();
                    stack
                }))
                .await;
            assert_eq!(workspace_lookup("shared"), Some(Value::Num(8.0)));
            assert_eq!(
                runmat_runtime::call::lexical::read_binding(&environment, binding).unwrap(),
                Some(Value::Num(8.0))
            );
            assert!(stack.is_empty(), "assignin arguments are consumed once");
            drop(_frame);
        })));
        reset_thread_state_for_tests();
    });
}
