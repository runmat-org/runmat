use super::*;

#[test]
fn animatedline_gathers_proven_numeric_style_and_parent_positions() {
    let _guard = setup();
    test_support::with_test_provider(|provider| {
        for (name, values) in [
            ("LineWidth", vec![2.]),
            ("MarkerSize", vec![8.]),
            ("Color", vec![1., 0., 0.]),
            ("MarkerEdgeColor", vec![0., 1., 0.]),
            ("MarkerFaceColor", vec![0., 0., 1.]),
        ] {
            let tensor = Tensor::new(values.clone(), vec![1, values.len()]).unwrap();
            let resident = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let host = block_on(animatedline_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("Marker"),
                    Value::from("o"),
                    Value::from(name),
                    Value::Tensor(tensor),
                ]),
            ))
            .unwrap();
            let gpu = block_on(animatedline_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("Marker"),
                    Value::from("o"),
                    Value::from(name),
                    Value::GpuTensor(resident.clone()),
                ]),
            ))
            .unwrap();
            assert_eq!(
                get_builtin(vec![gpu, Value::from(name)]).unwrap(),
                get_builtin(vec![host, Value::from(name)]).unwrap(),
                "{name}"
            );
            // Gathering and barrier bookkeeping must not free borrowed input.
            assert!(block_on(crate::gather_if_needed_async(&Value::GpuTensor(resident))).is_ok());
        }
        let axes = crate::builtins::plotting::axes::axes_builtin(vec![]).unwrap();
        let tensor = Tensor::new(vec![axes], vec![1, 1]).unwrap();
        let resident = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        for args in [
            vec![Value::GpuTensor(resident.clone())],
            vec![Value::from("Parent"), Value::GpuTensor(resident.clone())],
        ] {
            let line = block_on(animatedline_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .unwrap();
            assert_eq!(
                get_builtin(vec![line, Value::from("Parent")]).unwrap(),
                Value::Num(axes)
            );
        }
    });
}

#[test]
fn animatedline_pair_syntax_is_checked_without_materialization() {
    // The parser's syntax stage does not inspect coordinate storage/provider.
    let error = super::super::arguments::validate_pair_syntax(&[
        vector(&[1., 2.]),
        vector(&[3., 4.]),
        Value::from("Color"),
    ])
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:animatedline:InvalidArgument")
    );
    assert!(error.message().contains("pairs"));
}
