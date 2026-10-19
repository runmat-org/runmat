use super::*;
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::rc::Rc;

#[test]
fn session_variable_state_is_context_isolated() {
    let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));

    block_on(first.scope(async {
        store_global_named("answer", Value::Num(42.0)).unwrap();
        assert_eq!(global_value("answer"), Some(Value::Num(42.0)));
    }));
    block_on(second.scope(async {
        assert_eq!(global_value("answer"), None);
    }));
}

#[test]
fn semantic_names_bridge_executor_specific_global_and_persistent_layouts() {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    block_on(context.scope(async {
        store_global_named("answer", Value::Num(42.0)).unwrap();
        assert_eq!(global_value("answer"), Some(Value::Num(42.0)));

        store_persistent_named("counter", "calls", Value::Num(3.0));
        assert_eq!(
            persistent_named_value("counter", "calls"),
            Some(Value::Num(3.0))
        );
    }));
}

#[test]
fn global_revision_changes_for_each_canonical_write_without_aba() {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    block_on(context.scope(async {
        let initial = global_revision();
        store_global_named("answer", Value::Num(1.0)).unwrap();
        let first = global_revision();
        assert_ne!(initial, first);
        assert_eq!(first, global_revision());
        store_global_named("answer", Value::Num(2.0)).unwrap();
        let second = global_revision();
        assert_ne!(first, second);
        store_global_slot(4, Value::Num(2.0)).unwrap();
        let third = global_revision();
        assert_ne!(second, third);
        assert_ne!(initial, third);
        assert_eq!(global_value("answer"), Some(Value::Num(2.0)));
    }));
}

#[test]
fn user_global_with_legacy_slot_spelling_stays_named_and_disjoint() {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    block_on(context.scope(async {
        store_global_named("var_4", Value::Num(11.0)).unwrap();
        store_global_named("other", Value::Num(2.0)).unwrap();
        assert_eq!(global_names(), vec!["other", "var_4"]);

        let before_binding = global_revision();
        store_global_slot(4, Value::Num(2.0)).unwrap();
        assert_ne!(global_revision(), before_binding);
        assert_eq!(global_slot_value(4), Some(Value::Num(2.0)));
        assert_eq!(global_value("var_4"), Some(Value::Num(11.0)));

        store_global_slot(4, Value::Num(3.0)).unwrap();
        store_global_named("other", Value::Num(3.0)).unwrap();
        assert_eq!(global_slot_value(4), Some(Value::Num(3.0)));
        assert_eq!(global_value("other"), Some(Value::Num(3.0)));
        assert_eq!(global_value("var_4"), Some(Value::Num(11.0)));
        assert_eq!(global_names(), vec!["other", "var_4"]);

        store_global_slot(4, Value::Num(7.0)).unwrap();
        assert_eq!(global_slot_value(4), Some(Value::Num(7.0)));
        assert_eq!(global_value("other"), Some(Value::Num(3.0)));
        let roots = roots();
        assert!(roots.contains(&Value::Num(11.0)));
        assert!(roots.contains(&Value::Num(7.0)));
        let mut visited = Vec::new();
        assert_eq!(
            visit_roots(&mut |value| {
                visited.push(value.clone());
                Some(())
            }),
            Some(())
        );
        assert!(visited.contains(&Value::Num(11.0)));
        assert!(visited.contains(&Value::Num(7.0)));
    }));
}

#[test]
fn prepared_global_write_reserves_until_commit_or_drop() {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let authority = context.session_variable_authority();
    authority
        .store_global_with_context(&context, "answer", Value::Num(1.0))
        .unwrap();
    let before = authority.global_revision().unwrap();
    let pending = authority
        .prepare_store_global_with_context(&context, "answer", Value::Num(2.0))
        .unwrap();
    assert_eq!(
        authority.global_value("answer").unwrap(),
        Some(Value::Num(1.0))
    );
    assert!(context
        .fork_parallel_lab(crate::context::RuntimeServicePorts::default())
        .is_err());
    assert!(authority
        .store_global_with_context(&context, "other", Value::Num(3.0))
        .is_err());
    assert_eq!(authority.global_revision().unwrap(), before);
    drop(pending);
    assert!(context
        .fork_parallel_lab(crate::context::RuntimeServicePorts::default())
        .is_ok());
    let pending = authority
        .prepare_store_global_with_context(&context, "answer", Value::Num(2.0))
        .unwrap();
    pending.checkpoint(&context).unwrap();
    pending.commit();
    assert_eq!(
        authority.global_value("answer").unwrap(),
        Some(Value::Num(2.0))
    );
    assert_ne!(authority.global_revision().unwrap(), before);

    // An occupied update in a full admitted hash table must replace its
    // value without asking hashbrown for an unquoted resize at commit.
    authority
        .store_global_with_context(&context, "second", Value::Num(3.0))
        .unwrap();
    authority
        .store_global_with_context(&context, "third", Value::Num(4.0))
        .unwrap();
    authority
        .store_global_with_context(&context, "answer", Value::Num(5.0))
        .unwrap();
    assert_eq!(
        authority.global_value("answer").unwrap(),
        Some(Value::Num(5.0))
    );

    let cancellation = CancellationSource::new();
    let current = context
        .clone()
        .with_cancellation_token(cancellation.token());
    let declaration = authority
        .prepare_declare_global_with_context(&current, "answer")
        .unwrap();
    let rebound = current
        .clone()
        .with_cancellation_token(CancellationSource::new().token());
    assert_eq!(
        declaration.checkpoint(&rebound).unwrap_err().identifier(),
        Some("RunMat:parallel:InfrastructureFailure")
    );
    cancellation.cancel(CancellationReason::User);
    assert_eq!(
        declaration.checkpoint(&current).unwrap_err().identifier(),
        Some("RunMat:Cancelled")
    );
    drop(declaration);
    assert_eq!(
        authority.global_value("answer").unwrap(),
        Some(Value::Num(5.0))
    );
}

#[test]
fn redeclaration_preserves_existing_named_and_slot_values() {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let authority = context.session_variable_authority();
    let tensor = runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let payload = tensor.as_f64_slice().unwrap().as_ptr();
    authority
        .store_global_with_context(&context, "state", Value::Tensor(tensor))
        .unwrap();
    authority
        .store_global_slot_with_context(&context, 4, Value::Num(11.0))
        .unwrap();
    let before_redeclaration = authority.global_revision().unwrap();

    let named = authority
        .prepare_declare_global_with_context(&context, "state")
        .unwrap();
    let Value::Tensor(current) = named.value() else {
        panic!("redeclaration changed the existing global class");
    };
    assert_eq!(current.as_f64_slice().unwrap().as_ptr(), payload);
    named.checkpoint(&context).unwrap();
    named.commit();
    assert_eq!(authority.global_revision().unwrap(), before_redeclaration);
    let Value::Tensor(current) = authority.global_value("state").unwrap().unwrap() else {
        panic!("redeclaration replaced the existing global");
    };
    assert_eq!(current.as_f64_slice().unwrap().as_ptr(), payload);

    let slot = authority
        .prepare_declare_global_slot_with_context(&context, 4)
        .unwrap();
    assert_eq!(slot.value(), &Value::Num(11.0));
    slot.checkpoint(&context).unwrap();
    slot.commit();
    assert_eq!(authority.global_revision().unwrap(), before_redeclaration);
    assert_eq!(authority.global_slot(4).unwrap(), Some(Value::Num(11.0)));

    let first = authority
        .prepare_declare_global_with_context(&context, "fresh")
        .unwrap();
    let Value::Tensor(empty) = first.value() else {
        panic!("a first declaration did not create canonical empty");
    };
    assert_eq!(empty.shape(), &[0, 0]);
    first.checkpoint(&context).unwrap();
    first.commit();
    assert_ne!(authority.global_revision().unwrap(), before_redeclaration);
}

#[test]
fn transient_global_value_is_rejected_before_publication() {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let authority = context.session_variable_authority();
    let before = authority.global_revision().unwrap();
    let error = authority
        .store_global_with_context(&context, "bad", Value::output_list(vec![Value::Num(1.0)]))
        .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:TransientSequenceNotStorable")
    );
    assert_eq!(authority.global_value("bad").unwrap(), None);
    assert_eq!(authority.global_revision().unwrap(), before);
    authority
        .store_global_with_context(&context, "good", Value::Num(1.0))
        .unwrap();
}
