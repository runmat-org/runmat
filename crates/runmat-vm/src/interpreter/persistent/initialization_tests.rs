use super::*;
use runmat_runtime::{context::RuntimeContext, execution::RuntimeExecutionService};
use std::rc::Rc;

#[test]
fn absent_persistent_is_empty_without_publishing_a_frame_placeholder() {
    runmat_gc::gc_test_context(|| {
        for alias in [
            PersistentAlias::Named("x".into()),
            PersistentAlias::LegacySlot,
        ] {
            for slot in [0, 3] {
                let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
                let authority = runtime.session_variable_authority();
                let mut aliases = PersistentAliases::new();
                let mut vars = InterpreterVars::from_owned(vec![Value::Num(99.0)]);
                let evaluated = vars.get(0).unwrap().clone();
                declare(
                    &mut aliases,
                    &mut vars,
                    &authority,
                    "f",
                    slot,
                    alias.clone(),
                )
                .unwrap();
                assert_eq!(*vars.get(slot).unwrap(), empty_workspace_value());
                assert_eq!(evaluated, Value::Num(99.0));
                assert_eq!(authority.persistent_slot_value("f", slot).unwrap(), None);
                assert_eq!(authority.persistent_named_value("f", "x").unwrap(), None);

                let _ = vars.replace_owned(slot, Value::Num(17.0));
                publish_store(&aliases, &vars, &authority, "f", slot).unwrap();
                finish_success(&aliases, &mut vars, &authority, "f").unwrap();
                assert_eq!(*vars.get(slot).unwrap(), Value::Num(17.0));
                assert_eq!(
                    authority.persistent_slot_value("f", slot).unwrap(),
                    Some(Value::Num(17.0))
                );
            }
        }
    });
}
