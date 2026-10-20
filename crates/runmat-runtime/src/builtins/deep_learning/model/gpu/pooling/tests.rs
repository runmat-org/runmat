use super::*;
use runmat_accelerate_api::HostTensorView;

#[test]
fn reshaped_alias_cannot_masquerade_as_nontrivial_gpu_permutation() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[1., 2., 3., 4., 5., 6.],
                shape: &[2, 3],
            })
            .unwrap();
        let source_axes = AxisPlan::new(
            runmat_builtins::DeepLearningAxisLayout::parse("CB", 2).unwrap(),
            input.shape(),
        )
        .unwrap();
        let planned = DeepLearningPool1DRoute {
            reduction: DeepLearningPool1DRequest { axis_zero_based: 0 },
            result_axis: DeepLearningPool1DResultAxis::KeepSingleton,
            source_shape: vec![2, 3],
            target_shape: vec![3, 2],
            target_to_source: vec![1, 0],
        };
        let alias = presentation_alias_policy(&source_axes, &planned);
        assert_eq!(alias, GpuOutputAliasPolicy::RequireDistinctAllocation);
        let forged = input
            .with_view(input.view().clone().with_shape(vec![3, 2]))
            .unwrap();
        assert!(validate(
            forged,
            &input,
            &input,
            provider,
            &[3, 2],
            runmat_accelerate_api::handle_precision(&input),
            alias,
            "forward",
        )
        .is_err());
    });
}

#[test]
fn singleton_axis_permutations_accept_the_same_column_major_allocation() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        for (format, source_shape, order, target_shape) in [
            ("CB", vec![1, 2], vec![1, 0], vec![2, 1]),
            ("CBT", vec![2, 1, 3], vec![1, 0, 2], vec![1, 2, 3]),
        ] {
            let values = (1..=source_shape.iter().product::<usize>())
                .map(|value| value as f64)
                .collect::<Vec<_>>();
            let original_input = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &source_shape,
                })
                .unwrap();
            let reduced = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &source_shape,
                })
                .unwrap();
            let source_axes = AxisPlan::new(
                runmat_builtins::DeepLearningAxisLayout::parse(format, source_shape.len()).unwrap(),
                &source_shape,
            )
            .unwrap();
            let planned = DeepLearningPool1DRoute {
                reduction: DeepLearningPool1DRequest { axis_zero_based: 0 },
                result_axis: DeepLearningPool1DResultAxis::KeepSingleton,
                source_shape,
                target_shape: target_shape.clone(),
                target_to_source: order,
            };
            let alias = presentation_alias_policy(&source_axes, &planned);
            assert_eq!(alias, GpuOutputAliasPolicy::AllowInput);
            let view = reduced
                .with_view(reduced.view().clone().with_shape(target_shape.clone()))
                .unwrap();
            let accepted = validate(
                view,
                &reduced,
                &original_input,
                provider,
                &target_shape,
                runmat_accelerate_api::handle_precision(&reduced),
                alias,
                "forward",
            )
            .unwrap();
            assert!(accepted.shares_allocation_with(&reduced));
            assert!(!accepted.shares_allocation_with(&original_input));
            let gathered =
                crate::builtins::common::test_support::gather(Value::GpuTensor(accepted)).unwrap();
            assert_eq!(gathered.materialize_f64(), values);
        }
    });
}

#[test]
fn presentation_cannot_substitute_the_original_borrowed_allocation() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        // A size-one pool keeps the element count unchanged. A malicious
        // provider could return the original input as a differently
        // shaped view instead of permuting the separately reduced result.
        let values = [1., 2., 3., 4., 5., 6.];
        let original_input = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &[2, 3, 1],
            })
            .unwrap();
        let reduced = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &[2, 3, 1],
            })
            .unwrap();
        let source_axes = AxisPlan::new(
            runmat_builtins::DeepLearningAxisLayout::parse("CBT", 3).unwrap(),
            &[2, 3, 1],
        )
        .unwrap();
        let route = DeepLearningPool1DRoute {
            reduction: DeepLearningPool1DRequest { axis_zero_based: 2 },
            result_axis: DeepLearningPool1DResultAxis::KeepSingleton,
            source_shape: vec![2, 3, 1],
            target_shape: vec![3, 2, 1],
            target_to_source: vec![1, 0, 2],
        };
        let alias = presentation_alias_policy(&source_axes, &route);
        assert_eq!(alias, GpuOutputAliasPolicy::RequireDistinctAllocation);
        let forged = original_input
            .with_view(original_input.view().clone().with_shape(vec![3, 2, 1]))
            .unwrap();
        assert!(!forged.shares_allocation_with(&reduced));
        let error = validate(
            forged,
            &reduced,
            &original_input,
            provider,
            &route.target_shape,
            runmat_accelerate_api::handle_precision(&reduced),
            alias,
            "forward",
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert!(original_input.allocation_lease().validate_live().is_ok());
        assert!(reduced.allocation_lease().validate_live().is_ok());
    });
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires an actual compatible WGPU adapter"]
fn public_forward_and_predict_keep_pooling_resident_with_correct_axes() {
    use futures::executor::block_on;
    use runmat_accelerate_api::{handle_provenance, AccelProvider, GpuHandleProvenance};
    use runmat_value::CellArray;

    let _guard = crate::builtins::common::test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("WGPU network pooling qualification requires a compatible adapter");
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let sequence = block_on(
        crate::builtins::deep_learning::layers::sequence_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ),
    )
    .unwrap();
    let pool = block_on(
        crate::builtins::deep_learning::layers::global_average_pooling_1d_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ),
    )
    .unwrap();
    let network = block_on(super::super::dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![sequence, pool], 2, 1).unwrap(),
    )]))
    .unwrap();
    let values = (1..=24).map(f64::from).collect::<Vec<_>>();
    let input = provider
        .upload(&HostTensorView {
            data: &values,
            shape: &[2, 3, 4],
        })
        .unwrap()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let formatted = crate::builtins::deep_learning::object(
        "dlarray",
        vec![
            ("Data", Value::GpuTensor(input.clone())),
            ("Format", Value::String("CBT".into())),
            ("Labels", Value::String("CBT".into())),
        ],
    )
    .expect("fixture object");
    provider.reset_telemetry();
    let Value::Object(forward) = block_on(super::super::forward_builtin(
        network.value().clone(),
        formatted,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("formatted forward output expected")
    };
    let Some(Value::GpuTensor(forward_data)) = forward.properties().get("Data") else {
        panic!("resident forward data expected")
    };
    assert_eq!(forward_data.shape(), &[2, 3]);
    assert_eq!(
        forward.properties().get("Format"),
        Some(&Value::String("CB".into()))
    );
    assert_eq!(forward_data.device_id(), input.device_id());
    assert_eq!(
        runmat_accelerate_api::handle_precision(forward_data),
        runmat_accelerate_api::handle_precision(&input)
    );
    assert_eq!(
        handle_provenance(forward_data),
        Some(GpuHandleProvenance::Explicit)
    );
    assert!(!forward_data.shares_allocation_with(&input));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

    let Value::Object(network_object) = network.value() else {
        panic!("network object expected")
    };
    let predicted = block_on(super::super::predict_deep_learning_object(
        network_object.clone(),
        Value::GpuTensor(input.clone()),
        vec![],
    ))
    .unwrap();
    let Value::GpuTensor(predicted_data) = &predicted[0] else {
        panic!("raw predict must return resident scores")
    };
    assert_eq!(predicted_data.shape(), &[3, 2]);
    assert_eq!(predicted_data.device_id(), input.device_id());
    assert_eq!(
        runmat_accelerate_api::handle_precision(predicted_data),
        runmat_accelerate_api::handle_precision(&input)
    );
    assert_eq!(
        handle_provenance(predicted_data),
        Some(GpuHandleProvenance::Explicit)
    );
    assert!(!predicted_data.shares_allocation_with(&input));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

    let forward_values = block_on(provider.download(forward_data)).unwrap();
    let predicted_values = block_on(provider.download(predicted_data)).unwrap();
    assert_eq!(forward_values.data.len(), 6);
    assert_eq!(predicted_values.data.len(), 6);
    let tolerance = if provider.precision() == ProviderPrecision::F32 {
        2e-5
    } else {
        2e-12
    };
    for (actual, expected) in forward_values
        .data
        .iter()
        .zip([10., 11., 12., 13., 14., 15.])
    {
        assert!((actual - expected).abs() <= tolerance);
    }
    for (actual, expected) in predicted_values
        .data
        .iter()
        .zip([10., 12., 14., 11., 13., 15.])
    {
        assert!((actual - expected).abs() <= tolerance);
    }
}
