use super::*;
use runmat_runtime::{context::RuntimeContext, execution::RuntimeExecutionService};
use std::rc::Rc;

#[test]
fn inherited_alias_uses_ancestor_named_store_without_child_slot_fallback() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let authority = runtime.session_variable_authority();
        authority
            .store_persistent_named("outer", "P", Value::Num(5.0))
            .unwrap();
        authority
            .store_persistent_slot("outer", 9, Value::Num(33.0))
            .unwrap();

        let mut aliases = PersistentAliases::new();
        let mut child_vars = InterpreterVars::from_owned(vec![Value::Num(0.0); 10]);
        declare(
            &mut aliases,
            &mut child_vars,
            &authority,
            "outer",
            9,
            PersistentAlias::InheritedNamed("P".into()),
        )
        .unwrap();
        assert_eq!(*child_vars.get(9).unwrap(), Value::Num(5.0));

        let _ = child_vars.replace_owned(9, Value::Num(7.0));
        publish_store(&aliases, &child_vars, &authority, "child", 9).unwrap();
        assert_eq!(
            authority.persistent_named_value("outer", "P").unwrap(),
            Some(Value::Num(7.0))
        );
        assert_eq!(
            authority.persistent_slot_value("outer", 9).unwrap(),
            Some(Value::Num(33.0)),
            "the child slot must not become an ancestor legacy alias"
        );

        authority
            .store_persistent_named("outer", "P", Value::Num(8.0))
            .unwrap();
        refresh(&aliases, &mut child_vars, &authority, "child", 9).unwrap();
        assert_eq!(*child_vars.get(9).unwrap(), Value::Num(8.0));

        authority.clear_persistent_function("outer").unwrap();
        finish_success(&aliases, &mut child_vars, &authority, "child").unwrap();
        assert_eq!(
            authority.persistent_named_value("outer", "P").unwrap(),
            None,
            "a retired child must not replay its captured value after clear"
        );
    });
}
