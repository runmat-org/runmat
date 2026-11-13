//! Public dynamic ingress must not mistake unattempted lookup for absence.
use crate::call::BuiltinArguments;
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

#[test]
fn unattempted_name_lookup_is_deferred_even_for_a_known_builtin() {
    struct Unattempted;
    impl crate::call::descriptor::FunctionNameResolver for Unattempted {
        fn resolve_function(&self, _: &str) -> Option<runmat_types::FunctionId> {
            None
        }
    }
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let name = runmat_value::Utf8Text::new("logical".to_owned());
    let pointer = name.as_str().as_ptr();
    let target =
        crate::execution::RootedValue::in_context(Value::String(name.into()), &runtime).unwrap();
    let selected = crate::call::descriptor::RepeatableCallable::select_rooted(
        Some(&runtime),
        target,
        &Unattempted,
    )
    .unwrap();
    let crate::call::descriptor::CallableTarget::RetainedFevalForward(value) =
        selected.selected_target()
    else {
        panic!("unattempted lookup cannot prove a builtin binding")
    };
    let Value::String(name) = value.value() else {
        panic!("same admitted text")
    };
    assert_eq!(name.as_str().expect("nonmissing text").as_ptr(), pointer);
}

#[test]
fn public_feval_uses_original_user_shadow_and_preserves_zero_outputs_and_errors() {
    futures::executor::block_on(async {
        let original = crate::user_functions::test_support::exact_fixture_runtime();
        let foreign = crate::user_functions::test_support::exact_fixture_runtime();
        let _original_names = crate::user_functions::test_support::install_exact_named_forwarders(
            &original,
            &[("logical", 29401)],
        )
        .unwrap();
        let _foreign_names = crate::user_functions::test_support::install_exact_named_forwarders(
            &foreign,
            &[("logical", 29402)],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let fail = Arc::new(AtomicBool::new(false));
        let _invoker = {
            let _entry = original.enter();
            let calls = Arc::clone(&calls);
            let fail = Arc::clone(&fail);
            crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
                move |function, args, outputs| {
                    assert_eq!(function, 29401, "the original runtime owns name lookup");
                    assert_eq!(args, &[Value::Num(7.)]);
                    assert!(outputs <= 1);
                    calls.fetch_add(1, Ordering::SeqCst);
                    let fail = fail.load(Ordering::SeqCst);
                    Box::pin(async move {
                        if fail {
                            Err(crate::runtime_error::semantic_error(
                                "NamedProbeFailure",
                                "the selected user function failed",
                            ))
                        } else {
                            crate::user_functions::test_support::rooted_single_value_sequence(
                                Value::Num(42.),
                            )
                        }
                    })
                },
            )))
            .unwrap()
        };
        let _foreign_invoker = {
            let _entry = foreign.enter();
            crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, _, _| {
                panic!("foreign ambient name authority must not be used")
            })))
            .unwrap()
        };
        for target in [
            Value::from("logical"),
            Value::function_handle("logical".into()),
            Value::CharArray(runmat_value::CharArray::new_row("logical")),
        ] {
            for outputs in [0, 1] {
                let arguments =
                    BuiltinArguments::from_borrowed(&[Value::Num(7.)], Some(&original)).unwrap();
                let _foreign = foreign.enter();
                let result = crate::builtins::introspection::feval::feval_builtin_registered(
                    Some(original.clone()),
                    Some(outputs),
                    target.clone(),
                    arguments,
                )
                .await
                .unwrap();
                if outputs == 0 {
                    assert_eq!(
                        result.len(),
                        0,
                        "zero outputs still execute the user function"
                    );
                } else {
                    assert_eq!(result.try_into_single().unwrap().value(), &Value::Num(42.));
                }
            }
        }
        assert_eq!(calls.load(Ordering::SeqCst), 6);
        fail.store(true, Ordering::SeqCst);
        for outputs in [0, 1] {
            let arguments =
                BuiltinArguments::from_borrowed(&[Value::Num(7.)], Some(&original)).unwrap();
            let _foreign = foreign.enter();
            let error = crate::builtins::introspection::feval::feval_builtin_registered(
                Some(original.clone()),
                Some(outputs),
                Value::from("logical"),
                arguments,
            )
            .await
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:NamedProbeFailure"));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 8);

        // The supported None ingress keeps its existing active-runtime lookup.
        fail.store(false, Ordering::SeqCst);
        let _entry = original.enter();
        let arguments =
            BuiltinArguments::from_borrowed(&[Value::Num(7.)], Some(&original)).unwrap();
        let result = crate::builtins::introspection::feval::feval_builtin_registered(
            None,
            Some(1),
            Value::from("logical"),
            arguments,
        )
        .await
        .unwrap();
        assert_eq!(result.try_into_single().unwrap().value(), &Value::Num(42.));
        assert_eq!(calls.load(Ordering::SeqCst), 9);
    });
}
