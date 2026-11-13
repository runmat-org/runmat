use super::*;

#[test]
fn typed_alias_and_options_preflight_before_nested_provider_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .unwrap();
        let input =
            || Value::Cell(CellArray::new(vec![Value::GpuTensor(resident.clone())], 1, 1).unwrap());
        provider.reset_telemetry();
        for options in [
            vec![
                Value::String("Direction".into()),
                Value::String("symmetric".into()),
            ],
            vec![
                Value::String("Direction".into()),
                Value::String("left".into()),
                Value::String("PaddingDirection".into()),
                Value::String("right".into()),
            ],
            vec![
                Value::String("Length".into()),
                Value::Int(IntValue::I64(-1)),
            ],
        ] {
            let mut rest = vec![Value::Num(2.0)];
            rest.extend(options);
            let error = block_on(padsequences_builtin(
                input(),
                crate::builtins::common::test_support::builtin_arguments(rest),
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }
        let error = block_on(padsequences_builtin(
            input(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U64(u64::MAX),
            )]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let output = block_on(padsequences_builtin(
            input(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::String("DiReCtIoN".into()),
                Value::String("LEFT".into()),
                Value::String("UniformOutput".into()),
                Value::String("off".into()),
            ]),
        ))
        .unwrap();
        let Value::Cell(output) = output else {
            panic!("UniformOutput=off must retain cells")
        };
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 1);
        let Value::Tensor(sequence) = &output.data()[0] else {
            panic!("numeric sequence")
        };
        assert_eq!(sequence.materialize_f64(), vec![1.0, 2.0]);
        assert!(provider.telemetry_snapshot().download_bytes > 0);

        let overwritten = block_on(padsequences_builtin(
            input(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::String("UniformOutput".into()),
                Value::String("invalid".into()),
                Value::String("uniformoutput".into()),
                Value::String("off".into()),
            ]),
        ))
        .unwrap();
        assert!(matches!(overwritten, Value::Cell(_)));
    });
}
