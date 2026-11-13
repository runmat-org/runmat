use super::*;
#[test]
fn rbf_search_keeps_actual_counts_shapes_equality_and_original_failures() {
    for mode in 0..3 {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("pattern_objective", 29411)],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let measured = calls.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                assert_eq!((id, width), (29411, 1));
                let x = numbers(&args[0]);
                let axis = if mode == 1 {
                    evaluation::PointAxis::Columns
                } else {
                    evaluation::PointAxis::Rows
                };
                let points = if mode == 1 {
                    let Value::Tensor(value) = &args[0] else {
                        panic!("vectorized original column carrier")
                    };
                    axis.point_count(value.shape(), 2).unwrap()
                } else {
                    1
                };
                let ordinal = measured.fetch_add(points, Ordering::SeqCst);
                let values = (0..points)
                    .map(|point| {
                        let a = x[axis.offset(point, 0, points, 2)];
                        let b = x[axis.offset(point, 1, points, 2)];
                        assert!((a - b).abs() <= 1e-6);
                        (a - 1.2).powi(2) + (b - 1.2).powi(2)
                    })
                    .collect::<Vec<_>>();
                Box::pin(async move {
                    if mode == 2 && ordinal > 0 {
                        return Err(crate::build_runtime_error("original RBF objective")
                            .with_identifier("OriginalRBF:Failure")
                            .build());
                    }
                    returned(vec![if mode == 1 {
                        array(values, axis.result_shape(points).to_vec())
                    } else {
                        Value::Num(values[0])
                    }])
                })
            },
        )))
        .unwrap();
        let options = structure([
            ("Display", Value::from("off")),
            ("SearchFcn", Value::from("rbfsurrogate")),
            ("ScaleMesh", Value::Bool(false)),
            ("MaxIterations", Value::Num(30.)),
            ("MaxFunctionEvaluations", Value::Num(100.)),
            ("UseVectorized", Value::Bool(mode == 1)),
            ("UseCompletePoll", Value::Bool(mode == 1)),
        ]);
        let _width = crate::output_count::push_output_count(Some(4)).unwrap();
        let output = block_on(registered(
            Some(runtime),
            arguments(vec![
                Value::FunctionHandle("pattern_objective".into()),
                array(
                    vec![0., 0.],
                    if mode == 1 { vec![2, 1] } else { vec![1, 2] },
                ),
                empty(),
                empty(),
                array(vec![1., -1.], vec![1, 2]),
                Value::Num(0.),
                array(vec![0., 0.], vec![1, 2]),
                array(vec![3., 3.], vec![1, 2]),
                empty(),
                options,
            ]),
        ));
        if mode == 2 {
            assert_eq!(
                output.unwrap_err().identifier(),
                Some("OriginalRBF:Failure")
            );
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            continue;
        }
        let output = output.unwrap();
        let x = numbers(&output.as_slice()[0]);
        assert!((x[0] - x[1]).abs() <= 1e-6);
        let f = (x[0] - 1.2).powi(2) + (x[1] - 1.2).powi(2);
        assert!(f < 1e-3);
        assert!((numbers(&output.as_slice()[1])[0] - f).abs() < 1e-12);
        let Value::Struct(metadata) = &output.as_slice()[3] else {
            panic!("original Pattern output")
        };
        assert_eq!(
            numbers(&metadata.fields["funccount"])[0] as usize,
            calls.load(Ordering::SeqCst)
        );
    }
}
