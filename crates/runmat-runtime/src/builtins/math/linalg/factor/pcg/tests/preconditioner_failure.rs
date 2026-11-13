use super::*;

fn matrix(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(values, shape).unwrap())
}

#[test]
fn singular_first_or_second_factor_returns_best_initial_iterate_under_both_policies() {
    for managed in [false, true] {
        for second in [false, true] {
            let runtime =
                RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
            let runtime = if managed {
                runtime.with_service_ports(RuntimeServicePorts::default().with_parallel(Rc::new(
                    ManagedParallel(ManagedHostEffectLedger::new(1 << 20)),
                )))
            } else {
                runtime
            };
            let mut options = vec![Value::Num(1e-12), Value::Num(2.0)];
            if second {
                options.push(matrix(Vec::new(), vec![0, 0]));
            }
            options.push(matrix(vec![0.0; 4], vec![2, 2]));
            let request = super::super::request::parse(
                matrix(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]),
                matrix(vec![1.0, 2.0], vec![2, 1]),
                crate::builtins::common::test_support::builtin_arguments(options),
            )
            .unwrap();
            let Value::OutputList(outputs) = futures::executor::block_on(
                super::super::execution::execute(request, Some(&runtime), Some(5)),
            )
            .unwrap() else {
                panic!("five requested outputs");
            };
            assert_eq!(outputs.as_slice()[1], Value::Num(2.0));
            assert_eq!(outputs.as_slice()[2], Value::Num(1.0));
            assert_eq!(outputs.as_slice()[3], Value::Num(0.0));
            let Value::Tensor(solution) = &outputs.as_slice()[0] else {
                panic!("solution");
            };
            assert_eq!(solution.as_f64_slice().unwrap(), &[0.0, 0.0]);
            let Value::Tensor(history) = &outputs.as_slice()[4] else {
                panic!("history");
            };
            assert_eq!(history.len(), 1);
            assert!((history.as_f64_slice().unwrap()[0] - 5.0f64.sqrt()).abs() < 1e-12);
        }
    }
}

#[test]
fn failed_inverse_action_retains_best_iterate_but_language_errors_stay_terminal() {
    for language_error in [false, true] {
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
        let calls = Rc::new(std::cell::Cell::new(0usize));
        let observed = Rc::clone(&calls);
        runtime.set_bound_function_invoker(Some(Rc::new(
            move |_runtime, _binding, arguments, _requested, _environment| {
                let call = observed.get() + 1;
                observed.set(call);
                let Value::Tensor(input) = &arguments.supplied()[0] else {
                    panic!("input");
                };
                let values = if call == 1 {
                    input.as_f64_slice().unwrap().to_vec()
                } else {
                    vec![f64::NAN; 2]
                };
                crate::execution::admit_executable_sequence_with(
                    &_runtime.clone(),
                    move || async move {
                        if language_error && call == 2 {
                            return Err(crate::build_runtime_error("reader failed")
                                .with_identifier("RunMat:Test:InverseAction")
                                .build());
                        }
                        crate::user_functions::test_support::rooted_single_value_sequence(matrix(
                            values,
                            vec![2, 1],
                        ))
                    },
                )
            },
        )));
        let target = crate::user_functions::test_support::exact_local_handle(
            &runtime,
            "pcg_inverse_action",
            9200,
        );
        let request = super::super::request::parse(
            matrix(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]),
            matrix(vec![1.0, 2.0], vec![2, 1]),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1e-12),
                Value::Num(2.0),
                target,
            ]),
        )
        .unwrap();
        let result = futures::executor::block_on(super::super::execution::execute(
            request,
            Some(&runtime),
            Some(5),
        ));
        assert_eq!(calls.get(), 2, "completed callbacks are not replayed");
        if language_error {
            assert_eq!(
                result.unwrap_err().identifier(),
                Some("RunMat:Test:InverseAction")
            );
        } else {
            let Value::OutputList(outputs) = result.unwrap() else {
                panic!("outputs");
            };
            assert_eq!(outputs.as_slice()[1], Value::Num(2.0));
            assert_eq!(outputs.as_slice()[3], Value::Num(1.0));
            let Value::Num(relative) = outputs.as_slice()[2] else {
                panic!("relative residual");
            };
            assert!((relative - 0.25).abs() < 1e-12);
            let Value::Tensor(solution) = &outputs.as_slice()[0] else {
                panic!("solution");
            };
            assert_eq!(solution.as_f64_slice().unwrap(), &[0.25, 0.5]);
            let Value::Tensor(history) = &outputs.as_slice()[4] else {
                panic!("history");
            };
            assert_eq!(history.len(), 2);
        }
    }
}
