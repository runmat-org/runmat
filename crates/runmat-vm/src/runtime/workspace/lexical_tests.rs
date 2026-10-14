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
        assert!(environment.reserve_owned_binding(binding));
        runmat_runtime::call::lexical::write_owned_binding(&environment, binding, Value::Num(7.0))
            .unwrap();
        let mut vars = vec![Value::Num(7.0)];
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
        assert!(environment.reserve_owned_binding(binding));
        runmat_runtime::call::lexical::write_owned_binding(&environment, binding, Value::Num(9.0))
            .unwrap();
        let mut vars = vec![Value::Num(3.0)];
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
            let mut vars = vec![Value::Num(7.0)];
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
fn assignin_keeps_the_post_commit_lexical_cause_below_its_public_error() {
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
            let mut vars = vec![Value::Num(7.0)];
            let _frame = set_workspace_state(
                HashMap::from([("shared".to_string(), 0)]),
                HashSet::from(["shared".to_string()]),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            attach_lexical_workspace_bindings(
                Some(&environment),
                &[LexicalBindingSlot {
                    binding: BindingId(23),
                    slot: 0,
                    kind: LexicalBindingKind::Owned,
                }],
            )
            .unwrap();
            let mut stack = vec![
                Value::String("base".into()),
                Value::String("shared".into()),
                Value::Num(8.0),
            ];
            let error = crate::call::builtins::vm_dynamic_workspace_builtin(
                &runtime,
                &mut stack,
                "assignin",
                3,
                0,
                &crate::bytecode::FunctionRegistry::default(),
                None,
            )
            .await
            .unwrap_err();
            assert_eq!(error.identifier(), Some("Acme:DynamicWorkspaceUnavailable"));
            assert_eq!(
                error.gpu_gather_retry,
                runmat_runtime::runtime_error::GpuGatherRetry::Never
            );
            let publication = error
                .source()
                .and_then(|source| source.downcast_ref::<runmat_runtime::RuntimeError>())
                .expect("typed post-commit publication error");
            assert_eq!(
                publication.identifier(),
                Some("Acme:Workspace:LexicalPublication")
            );
            let cause = publication
                .source()
                .and_then(|source| source.downcast_ref::<runmat_runtime::RuntimeError>())
                .expect("original lexical error");
            assert_eq!(cause.identifier(), Some("Acme:LexicalEnvironment"));
            assert_eq!(workspace_lookup("shared"), Some(Value::Num(8.0)));
            assert!(stack.is_empty(), "assignin arguments are consumed once");
            drop(_frame);
        })));
        reset_thread_state_for_tests();
    });
}
