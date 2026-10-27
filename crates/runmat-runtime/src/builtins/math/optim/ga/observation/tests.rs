use super::*;
use runmat_value::{CellArray, ValueSequence};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn structure(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    for (key, field) in fields {
        value.insert(key, field);
    }
    Value::Struct(value.finish())
}

#[test]
fn interrupt_keeps_original_options_and_all_observers_run_after_stop() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("ga_first_observer", 29111), ("ga_second_observer", 29112)],
        )
        .unwrap();
        let called = Arc::new(AtomicUsize::new(0));
        let counted = called.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                assert_eq!(width, 3);
                assert_eq!(args.len(), 3);
                assert_eq!(
                    crate::builtins::common::validation::value_text_scalar_string(&args[2])
                        .as_deref(),
                    Some("interrupt")
                );
                counted.fetch_add(1, Ordering::SeqCst);
                let state = structure([(
                    "StopFlag",
                    Value::from(if id == 29111 { "first-stop" } else { "" }),
                )]);
                // Neither options nor optchanged has its ordinary shape. Both
                // are documented inactive during an AL interrupt event.
                let values = vec![state, Value::Num(42.), Value::from("not-a-boolean")];
                Box::pin(async move {
                    crate::execution::RootedValueSequence::new(
                        ValueSequence::comma_separated(values)
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                    )
                    .map_err(|source| source.into_runtime_error())
                })
            },
        )))
        .unwrap();
        let mut cooperation =
            ScalarCooperation::for_builtin(super::super::IDENTITY, Some(&runtime));
        let controls = structure([
            (
                "OutputFcn",
                Value::Cell(
                    CellArray::new(
                        vec![
                            Value::FunctionHandle("ga_first_observer".into()),
                            Value::FunctionHandle("ga_second_observer".into()),
                        ],
                        1,
                        2,
                    )
                    .unwrap(),
                ),
            ),
            ("Marker", Value::Num(7.)),
        ]);
        let observers = Observers {
            output: vec![
                super::super::callbacks::fixture::target(&runtime, &_bindings, "ga_first_observer"),
                super::super::callbacks::fixture::target(
                    &runtime,
                    &_bindings,
                    "ga_second_observer",
                ),
            ],
            plot: Vec::new(),
        };
        let state = RootedValue::in_optional_context(
            structure([("StopFlag", Value::from(""))]),
            Some(&runtime),
        )
        .unwrap();
        let controls = RootedValue::in_optional_context(controls, Some(&runtime)).unwrap();
        let foreign = crate::user_functions::test_support::exact_fixture_runtime();
        let _foreign = foreign.enter();
        let result = observers
            .notify(
                state,
                controls,
                OptimGeneticObserverState::Interrupt,
                true,
                None,
                &mut cooperation,
            )
            .await
            .unwrap();
        assert_eq!(called.load(Ordering::SeqCst), 2);
        assert!(
            result.stopped,
            "later clearing StopFlag cannot erase earlier request"
        );
        assert!(!result.options_changed);
        let Value::Struct(options) = result.controls.value() else {
            panic!("original options");
        };
        assert!(matches!(options.fields.get("Marker"), Some(Value::Num(7.))));
    });
}
