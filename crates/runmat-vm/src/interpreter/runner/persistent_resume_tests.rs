use super::*;
use runmat_runtime::{context::RuntimeContext, execution::RuntimeExecutionService};
use std::rc::Rc;

#[test]
fn foreign_session_or_function_persistent_resume_fails_before_first_read_or_store() {
    runmat_gc::gc_test_context(|| {
        let origin = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let foreign = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let authority = origin.session_variable_authority();
        authority
            .store_persistent_named_and_slot("f", "p", 0, Value::Num(7.0))
            .unwrap();
        let continuation = runmat_runtime::workspace::session::PersistentBindingContinuation {
            name: "p".into(),
            function: "f".into(),
            owner: authority.clone(),
            invalidation: authority.persistent_invalidation_revision("f").unwrap(),
            invalidated: false,
        };
        let binding = crate::interpreter::PersistentBinding::from_native_continuation(
            continuation.clone(),
            &authority,
            "f",
        )
        .unwrap();
        assert!(
            crate::interpreter::PersistentBinding::from_native_continuation(
                continuation.clone(),
                &foreign.session_variable_authority(),
                "f",
            )
            .is_err()
        );
        assert!(
            crate::interpreter::PersistentBinding::from_native_continuation(
                continuation,
                &authority,
                "g",
            )
            .is_err()
        );

        let bytecode = Bytecode::with_instructions(
            vec![Instr::LoadVar(0), Instr::StoreVar(0), Instr::Return],
            1,
        );
        for (runtime, function) in [(foreign.clone(), "f"), (origin.clone(), "g")] {
            let resume = InterpreterResumeState {
                pc: 0,
                completion_boundary: None,
                vars: crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(7.0)]),
                assigned_slots: std::collections::HashSet::from([0]),
                supplied_inputs: 0,
                requested_outputs: 0,
                missing_input_slots: Default::default(),
                global_aliases: Default::default(),
                persistent_aliases: HashMap::from([(0, binding.clone())]),
                side_effect_epoch: 0,
            };
            let error = futures::executor::block_on(interpret_resume_in_context(
                &bytecode,
                resume,
                Some(function),
                runtime,
            ))
            .expect_err("foreign persistent continuation must be rejected pre-effect");
            assert_eq!(
                error.identifier(),
                Some("RunMat:Workspace:PersistentResumeOwner")
            );
        }
        assert_eq!(
            authority.persistent_named_value("f", "p").unwrap(),
            Some(Value::Num(7.0))
        );
        assert_eq!(
            foreign
                .session_variable_authority()
                .persistent_named_value("f", "p")
                .unwrap(),
            None
        );
    });
}
