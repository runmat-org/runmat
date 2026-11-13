use super::*;
use std::sync::Mutex;
#[test]
fn nelder_search_reuses_seed_and_original_counts_options_observers_and_errors() {
    for mode in 0..5 {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("pattern_objective", 29401), ("nelder_observer", 29402)],
        )
        .unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let measured = seen.clone();
        let phases = Arc::new(Mutex::new(Vec::new()));
        let observed = phases.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                assert_eq!(width, 1);
                if id == 29402 {
                    let Value::Struct(metadata) = &args[1] else {
                        panic!("actual NM metadata")
                    };
                    let count = numbers(&metadata.fields["funccount"])[0];
                    let Value::String(runmat_value::StringElement::Text(state)) = &args[2] else {
                        panic!("actual phase")
                    };
                    observed.lock().unwrap().push((state.to_string(), count));
                    return Box::pin(std::future::ready(returned(vec![Value::Bool(mode == 4)])));
                }
                assert_eq!(id, 29401);
                let x = numbers(&args[0])[0];
                let mut seen = measured.lock().unwrap();
                seen.push(x);
                let ordinal = seen.len();
                drop(seen);
                let value = if mode == 1 {
                    array(vec![(x - 1.).powi(2)], vec![1, 1])
                } else {
                    Value::Num((x - 1.).powi(2))
                };
                Box::pin(async move {
                    if mode == 3 && ordinal == 2 {
                        Err(crate::build_runtime_error("original NM objective")
                            .with_identifier("OriginalNM:Failure")
                            .build())
                    } else {
                        returned(vec![value])
                    }
                })
            },
        )))
        .unwrap();
        let local = structure([
            ("Display", Value::from("off")),
            ("MaxFunEvals", Value::Num(1.)),
            ("MaxIter", Value::Num(10.)),
            ("OutputFcn", Value::FunctionHandle("nelder_observer".into())),
        ]);
        let search = Value::Cell(
            runmat_value::CellArray::new(
                vec![Value::from("searchneldermead"), Value::Num(1.), local],
                1,
                3,
            )
            .unwrap(),
        );
        let options = structure([
            ("Display", Value::from("off")),
            ("SearchFcn", search),
            ("MaxIterations", Value::Num(1.)),
            ("MaxFunctionEvaluations", Value::Num(10.)),
            ("UseVectorized", Value::Bool(mode == 1)),
        ]);
        let (lower, upper) = if mode == 2 {
            (Value::Num(0.), Value::Num(1.))
        } else if mode == 1 {
            (Value::Num(f64::NEG_INFINITY), Value::Num(f64::INFINITY))
        } else {
            (empty(), empty())
        };
        let _width = crate::output_count::push_output_count(Some(4)).unwrap();
        let output = block_on(registered(
            Some(runtime),
            arguments(vec![
                Value::FunctionHandle("pattern_objective".into()),
                Value::Num(0.),
                empty(),
                empty(),
                empty(),
                empty(),
                lower,
                upper,
                empty(),
                options,
            ]),
        ));
        if mode == 2 {
            assert_eq!(
                output.unwrap_err().identifier(),
                Some("RunMat:patternsearch:InvalidArgument")
            );
            assert!(seen.lock().unwrap().is_empty());
            continue;
        }
        if mode == 3 {
            assert_eq!(output.unwrap_err().identifier(), Some("OriginalNM:Failure"));
            assert_eq!(*seen.lock().unwrap(), [0., 0.00025]);
            continue;
        }
        let output = output.unwrap();
        let x = numbers(&output.as_slice()[0])[0];
        assert_eq!(numbers(&output.as_slice()[1]), [(x - 1.).powi(2)]);
        if mode == 4 {
            assert_eq!(x, 1.);
            assert_eq!(*seen.lock().unwrap(), [0., 1.]);
            assert_eq!(
                *phases.lock().unwrap(),
                [("init".into(), 0.), ("done".into(), 0.)]
            );
        } else {
            assert_eq!(x, 0.00025);
            assert_eq!(*seen.lock().unwrap(), [0., 0.00025]);
            assert_eq!(
                *phases.lock().unwrap(),
                [
                    ("init".into(), 0.),
                    ("iter".into(), 1.),
                    ("done".into(), 1.)
                ]
            );
        }
    }
}
