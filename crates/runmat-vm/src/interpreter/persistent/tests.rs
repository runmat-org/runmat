use super::*;
use runmat_runtime::{context::RuntimeContext, execution::RuntimeExecutionService};
use std::rc::Rc;

#[test]
fn unchanged_revision_skips_canonical_value_copy_but_recursive_write_refreshes() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let authority = runtime.session_variable_authority();
        authority
            .store_persistent_named_and_slot("f", "x", 0, Value::Num(1.0))
            .unwrap();
        let mut aliases = PersistentAliases::new();
        let mut vars = InterpreterVars::from_owned(vec![Value::Num(0.0)]);
        declare(
            &mut aliases,
            &mut vars,
            &authority,
            "f",
            0,
            PersistentAlias::Named("x".into()),
        )
        .unwrap();
        let _ = vars.replace_owned(0, Value::Num(99.0));
        refresh(&aliases, &mut vars, &authority, "f", 0).unwrap();
        assert_eq!(
            *vars.get(0).unwrap(),
            Value::Num(99.0),
            "unchanged binding is not copied"
        );
        authority
            .store_persistent_named_and_slot("f", "x", 0, Value::Num(3.0))
            .unwrap();
        refresh(&aliases, &mut vars, &authority, "f", 0).unwrap();
        assert_eq!(
            *vars.get(0).unwrap(),
            Value::Num(3.0),
            "nested write refreshes alias"
        );
    });
}

#[test]
fn cleared_binding_cannot_be_republished_by_an_older_frame() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let authority = runtime.session_variable_authority();
        authority
            .store_persistent_named_and_slot("f", "x", 0, Value::Num(7.0))
            .unwrap();
        let mut aliases = PersistentAliases::new();
        let mut vars = InterpreterVars::from_owned(vec![Value::Num(0.0)]);
        declare(
            &mut aliases,
            &mut vars,
            &authority,
            "f",
            0,
            PersistentAlias::Named("x".into()),
        )
        .unwrap();
        let before_clear = authority.persistent_revision().unwrap();
        authority.clear_persistent_function("f").unwrap();
        let after_clear = authority.persistent_revision().unwrap();
        assert_ne!(before_clear, after_clear);
        refresh(&aliases, &mut vars, &authority, "f", 0).unwrap();
        assert_eq!(*vars.get(0).unwrap(), empty_workspace_value());
        finish_success(&aliases, &mut vars, &authority, "f").unwrap();
        assert_eq!(authority.persistent_named_value("f", "x").unwrap(), None);
        assert_eq!(authority.persistent_slot_value("f", 0).unwrap(), None);
        assert_eq!(*vars.get(0).unwrap(), empty_workspace_value());
        authority
            .store_persistent_named_and_slot("f", "x", 0, Value::Num(3.0))
            .unwrap();
        assert_ne!(authority.persistent_revision().unwrap(), before_clear);
        refresh(&aliases, &mut vars, &authority, "f", 0).unwrap();
        assert_eq!(*vars.get(0).unwrap(), Value::Num(3.0));
    });
}

#[test]
fn clear_before_first_write_cannot_initialize_from_stale_frame() {
    runmat_gc::gc_test_context(|| {
        for alias in [
            PersistentAlias::Named("x".into()),
            PersistentAlias::LegacySlot,
        ] {
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            let authority = runtime.session_variable_authority();
            let mut aliases = PersistentAliases::new();
            let mut vars = InterpreterVars::from_owned(vec![Value::Num(0.0)]);
            declare(&mut aliases, &mut vars, &authority, "f", 0, alias).unwrap();
            let declaration = authority.persistent_invalidation_revision("f").unwrap();
            authority.clear_persistent_function("f").unwrap();
            assert_ne!(
                authority.persistent_invalidation_revision("f").unwrap(),
                declaration
            );
            // No read between clear and successful old-frame completion.
            finish_success(&aliases, &mut vars, &authority, "f").unwrap();
            assert_eq!(*vars.get(0).unwrap(), empty_workspace_value());
            assert_eq!(authority.persistent_slot_value("f", 0).unwrap(), None);
            assert_eq!(authority.persistent_named_value("f", "x").unwrap(), None);

            // An actual assignment after clear is a new publication, not a
            // replay of the declaration's stale local.
            let _ = vars.replace_owned(0, Value::Num(9.0));
            publish_store(&aliases, &vars, &authority, "f", 0).unwrap();
            finish_success(&aliases, &mut vars, &authority, "f").unwrap();
            assert_eq!(
                authority.persistent_slot_value("f", 0).unwrap(),
                Some(Value::Num(9.0))
            );
        }
    });
}

#[test]
fn native_continuation_keeps_preclear_generation_across_vm_resume() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let authority = runtime.session_variable_authority();
        authority
            .store_persistent_named("f", "x", Value::Num(7.0))
            .unwrap();
        let saved = runmat_runtime::workspace::session::PersistentBindingContinuation {
            name: "x".into(),
            function: "f".into(),
            owner: authority.clone(),
            invalidation: authority.persistent_invalidation_revision("f").unwrap(),
            invalidated: false,
        };
        let old_ssa_operand = Value::Num(7.0);
        authority.clear_persistent_function("f").unwrap();
        let mut aliases = PersistentAliases::new();
        aliases.insert(
            0,
            PersistentBinding::from_native_continuation(saved, &authority, "f").unwrap(),
        );
        let mut vars = InterpreterVars::from_owned(vec![Value::Num(7.0)]);
        let mut first_read = InterpreterVars::from_owned(vars.clone_values_for_test());
        let read_aliases = aliases.clone();
        refresh(&read_aliases, &mut first_read, &authority, "f", 0).unwrap();
        assert_eq!(first_read.get(0).as_deref(), Some(&empty_workspace_value()));
        finish_success(&aliases, &mut vars, &authority, "f").unwrap();
        assert_eq!(*vars.get(0).unwrap(), empty_workspace_value());
        assert_eq!(old_ssa_operand, Value::Num(7.0));
        assert_eq!(authority.persistent_named_value("f", "x").unwrap(), None);

        let _ = vars.replace_owned(0, Value::Num(9.0));
        publish_store(&aliases, &vars, &authority, "f", 0).unwrap();
        finish_success(&aliases, &mut vars, &authority, "f").unwrap();
        assert_eq!(
            authority.persistent_named_value("f", "x").unwrap(),
            Some(Value::Num(9.0))
        );
    });
}

#[test]
fn named_store_publishes_exact_function_and_legacy_slot_then_refreshes_only_local() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let _scope = runtime.enter();
        let authority = runtime.session_variable_authority();
        let mut aliases = PersistentAliases::new();
        let mut vars = InterpreterVars::from_owned(vec![Value::Num(0.0)]);
        declare(
            &mut aliases,
            &mut vars,
            &authority,
            "f",
            0,
            PersistentAlias::Named("x".into()),
        )
        .unwrap();
        let _ = vars.replace_owned(0, Value::Num(0.0));
        publish_store(&aliases, &vars, &authority, "f", 0).unwrap();
        assert_eq!(
            authority.persistent_slot_value("f", 0).unwrap(),
            Some(Value::Num(0.0))
        );
        assert_eq!(
            authority.persistent_named_value("f", "x").unwrap(),
            Some(Value::Num(0.0))
        );
        assert_eq!(authority.persistent_named_value("g", "x").unwrap(), None);

        let evaluated_stack_operand = vars.get(0).unwrap().clone();
        authority
            .store_persistent_named_and_slot("f", "x", 0, Value::Num(3.0))
            .unwrap();
        refresh(&aliases, &mut vars, &authority, "f", 0).unwrap();
        assert_eq!(*vars.get(0).unwrap(), Value::Num(3.0));
        assert_eq!(evaluated_stack_operand, Value::Num(0.0));
        authority
            .store_persistent_named_and_slot("f", "x", 0, Value::Num(5.0))
            .unwrap();
        finish_success(&aliases, &mut vars, &authority, "f").unwrap();
        assert_eq!(*vars.get(0).unwrap(), Value::Num(5.0));
        assert_eq!(
            authority.persistent_named_value("f", "x").unwrap(),
            Some(Value::Num(5.0))
        );
    });
}

#[test]
fn legacy_slot_and_named_fallback_keep_selected_context_authority() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let other = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let authority = runtime.session_variable_authority();
        authority
            .store_persistent_slot("f", 2, Value::Num(7.0))
            .unwrap();
        let _other_scope = other.enter();
        let mut legacy = PersistentAliases::new();
        let mut vars = InterpreterVars::from_owned(vec![Value::Num(0.0); 3]);
        declare(
            &mut legacy,
            &mut vars,
            &authority,
            "f",
            2,
            PersistentAlias::LegacySlot,
        )
        .unwrap();
        assert_eq!(*vars.get(2).unwrap(), Value::Num(7.0));
        let mut named = PersistentAliases::new();
        declare(
            &mut named,
            &mut vars,
            &authority,
            "f",
            2,
            PersistentAlias::Named("x".into()),
        )
        .unwrap();
        assert_eq!(*vars.get(2).unwrap(), Value::Num(7.0));
        finish_success(&named, &mut vars, &authority, "f").unwrap();
        assert_eq!(
            authority.persistent_named_value("f", "x").unwrap(),
            Some(Value::Num(7.0))
        );
        assert_eq!(
            other
                .session_variable_authority()
                .persistent_slot_value("f", 2)
                .unwrap(),
            None
        );
    });
}

#[test]
fn no_completion_replay_after_newer_recursive_or_failed_frame_effect() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let authority = runtime.session_variable_authority();
        let mut aliases = PersistentAliases::new();
        let mut vars = InterpreterVars::from_owned(vec![Value::Num(1.0)]);
        declare(
            &mut aliases,
            &mut vars,
            &authority,
            "f",
            0,
            PersistentAlias::Named("x".into()),
        )
        .unwrap();
        let _ = vars.replace_owned(0, Value::Num(1.0));
        publish_store(&aliases, &vars, &authority, "f", 0).unwrap();
        authority
            .store_persistent_named_and_slot("f", "x", 0, Value::Num(2.0))
            .unwrap();
        // Error/cancellation path does not call finish_success. A subsequent
        // successful frame may finalize but can only refresh newer canonical 2.
        assert_eq!(
            authority.persistent_named_value("f", "x").unwrap(),
            Some(Value::Num(2.0))
        );
        finish_success(&aliases, &mut vars, &authority, "f").unwrap();
        assert_eq!(*vars.get(0).unwrap(), Value::Num(2.0));
        assert_eq!(
            authority.persistent_named_value("f", "x").unwrap(),
            Some(Value::Num(2.0))
        );
    });
}
