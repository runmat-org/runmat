use super::*;
use std::sync::{Arc, Mutex};

#[test]
fn real_milp_event_groups_stop_error_cancel_and_complete_without_incumbent_replay() {
    for mode in 0..5 {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = context().with_cancellation_token(cancellation.token());
        runtime.set_error_namespace("MilpInvocation");
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("milp_first", 2130), ("milp_second", 2131)],
        )
        .unwrap();
        let _entry = runtime.enter();
        let events = Arc::new(Mutex::new(Vec::<(usize, String, Vec<f64>)>::new()));
        let seen = Arc::clone(&events);
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, requested| {
                assert_eq!(requested, 1);
                assert!(id == 2130 || id == 2131);
                let state = crate::builtins::common::validation::value_text_scalar_string(&args[2])
                    .unwrap();
                let x = values(&args[0]);
                let Value::Struct(metadata) = &args[1] else {
                    panic!("MILP event metadata");
                };
                assert_eq!(metadata.fields.len(), 6);
                for name in [
                    "dualbound",
                    "fval",
                    "numfeaspoints",
                    "numnodes",
                    "relativegap",
                    "time",
                ] {
                    assert!(metadata.fields.contains_key(name));
                }
                assert!(!metadata.fields.contains_key("phase"));
                assert!(!metadata.fields.contains_key("gradient"));
                if state != "iter" {
                    assert!(x.is_empty(), "init/done cannot replay a feasible discovery");
                }
                if !x.is_empty() {
                    assert_eq!(x.len(), 2);
                    assert!(x.iter().all(|v| *v == 0.0 || *v == 1.0));
                    assert!(x.iter().sum::<f64>() <= 1.0);
                    assert_eq!(
                        values(&metadata.fields["fval"]),
                        vec![-x.iter().sum::<f64>()]
                    );
                    assert!(values(&metadata.fields["numfeaspoints"])[0] >= 1.0);
                }
                assert!(values(&metadata.fields["time"])[0] >= 0.0);
                seen.lock().unwrap().push((id, state.clone(), x.clone()));
                let response = if id == 2130 && mode == 2 {
                    Err(crate::build_runtime_error("original MILP observer error")
                        .with_identifier("Original:MilpObserver")
                        .with_source(std::io::Error::other("owned observer cause"))
                        .build())
                } else if id == 2130 && mode == 3 {
                    cancellation.cancel(runmat_execution::CancellationReason::Quota);
                    Ok(Value::Bool(false))
                } else {
                    Ok(Value::Bool(
                        id == 2130
                            && ((mode == 0 && state == "init")
                                || (mode == 1 && state == "iter" && !x.is_empty())),
                    ))
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(response?)
                })
            },
        )))
        .unwrap();
        let callbacks = Value::Cell(
            runmat_value::CellArray::new(
                vec![
                    Value::FunctionHandle("milp_first".into()),
                    Value::FunctionHandle("milp_second".into()),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let _count = crate::output_count::push_output_count(Some(4)).unwrap();
        let result = block_on(execute(
            Some(&runtime),
            crate::builtins::common::test_support::builtin_arguments(binary_problem(options([(
                "OutputFcn",
                callbacks,
            )]))),
        ));
        let observed = events.lock().unwrap();
        if mode == 2 {
            let error = result.unwrap_err();
            assert_eq!(error.identifier(), Some("Original:MilpObserver"));
            assert!(std::error::Error::source(&error)
                .unwrap()
                .is::<std::io::Error>());
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(observed.len(), 1);
        } else if mode == 3 {
            let error = result.unwrap_err();
            assert_eq!(error.identifier(), Some("MilpInvocation:Cancelled"));
            assert!(crate::runtime_error::error_has_control_cancellation_origin(
                &error
            ));
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(observed.len(), 1);
        } else {
            let Value::OutputList(out) = result.unwrap() else {
                panic!("four outputs");
            };
            assert_eq!(values(&out[2]), vec![if mode == 4 { 1.0 } else { -1.0 }]);
            assert_eq!(observed.first().unwrap().1, "init");
            assert_eq!(observed.last().unwrap().1, "done");
            for pair in observed.chunks_exact(2) {
                assert_eq!(
                    (pair[0].0, pair[1].0),
                    (2130, 2131),
                    "later callbacks run after an earlier true flag"
                );
                assert_eq!(pair[0].1, pair[1].1);
            }
            if mode == 0 {
                assert_eq!(values(field(&out[3], "numnodes")), vec![0.0]);
            } else {
                assert!(observed
                    .iter()
                    .any(|(_, state, x)| state == "iter" && !x.is_empty()));
            }
        }
    }
}

#[test]
fn accepted_candidates_and_improving_observer_count_are_distinct_real_certificates() {
    let runtime = context();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::INTLINPROG_CATALOG_ENTRY.identity,
        Some(&runtime),
    );
    block_on(async {
        let (problem, mut controls) = parse::parse(
            crate::builtins::common::test_support::builtin_arguments(binary_problem(options([]))),
            &runtime,
            &mut cooperation,
        )
        .await
        .unwrap();
        let workspace = model::Workspace::new(problem, &mut cooperation)
            .await
            .unwrap();
        let mut best = None;
        for (x, improved) in [
            ([0.0, 0.0], true),
            ([1.0, 0.0], true),
            ([0.0, 1.0], false),
            ([1.0, 0.0], false),
        ] {
            let point = proofs::incumbent(
                &workspace.view(),
                &workspace.integers,
                &workspace.semis,
                &x,
                &mut cooperation,
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(
                engine::accept(point, &mut best, &mut controls).unwrap(),
                improved
            );
        }
        assert_eq!(controls.feasible_points, 3);
        assert_eq!(controls.improving_points, 2);
        assert_eq!(publication::number(&best.unwrap().objective), -1.0);
    });
}
