use super::*;
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use futures::executor::block_on;
use runmat_accelerate_api::{handle_is_explicit, GpuHandleProvenance, HostTensorView};

fn construct(kind: DeepLearningLayerKind, options: Vec<Value>) -> Value {
    use DeepLearningLayerKind as K;
    block_on(async {
        match kind {
            K::FeatureInput => {
                feature_input_layer_builtin(
                    Value::Num(2.0),
                    crate::builtins::common::test_support::builtin_arguments(options),
                )
                .await
            }
            K::ClassificationOutput => {
                classification_layer_builtin(
                    crate::builtins::common::test_support::builtin_arguments(options),
                )
                .await
            }
            K::FullyConnected => {
                fully_connected_layer_builtin(
                    None,
                    Value::Num(2.0),
                    crate::builtins::common::test_support::builtin_arguments(options),
                )
                .await
            }
            K::Lstm => {
                lstm_layer_builtin(
                    Value::Num(2.0),
                    crate::builtins::common::test_support::builtin_arguments(options),
                )
                .await
            }
            K::Bilstm => {
                bilstm_layer_builtin(
                    Value::Num(2.0),
                    crate::builtins::common::test_support::builtin_arguments(options),
                )
                .await
            }
            K::Convolution1D => {
                convolution_1d_layer_builtin(
                    Value::Num(3.0),
                    Value::Num(2.0),
                    crate::builtins::common::test_support::builtin_arguments(options),
                )
                .await
            }
            K::LayerNormalization => {
                layer_normalization_layer_builtin(
                    crate::builtins::common::test_support::builtin_arguments(options),
                )
                .await
            }
            _ => unreachable!("only data-bearing layer constructors in this fixture"),
        }
    })
    .unwrap()
}

#[test]
fn resident_learnables_and_recurrent_state_are_not_materialized_by_construction() {
    use DeepLearningLayerKind as K;
    use DeepLearningLayerOption as O;
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = provider
            .upload(&HostTensorView {
                data: &[2.0, 5.0],
                shape: &[2, 1],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        for (kind, options) in [
            (
                K::FeatureInput,
                &[O::Mean, O::StandardDeviation, O::Min, O::Max][..],
            ),
            (K::ClassificationOutput, &[O::ClassWeights][..]),
            (
                K::FullyConnected,
                &[O::Weights, O::Bias, O::InputLearnables, O::OutputLearnables][..],
            ),
            (
                K::Lstm,
                &[
                    O::InputWeights,
                    O::RecurrentWeights,
                    O::Bias,
                    O::CellState,
                    O::HiddenState,
                ][..],
            ),
            (
                K::Bilstm,
                &[
                    O::InputWeights,
                    O::RecurrentWeights,
                    O::Bias,
                    O::CellState,
                    O::HiddenState,
                ][..],
            ),
            (K::Convolution1D, &[O::Weights, O::Bias][..]),
            (K::LayerNormalization, &[O::Scale, O::Offset][..]),
        ] {
            for option in options {
                provider.reset_telemetry();
                let Value::Object(layer) = construct(
                    kind,
                    vec![
                        Value::String(option.canonical_name().into()),
                        Value::GpuTensor(source.clone()),
                    ],
                ) else {
                    panic!("layer object expected")
                };
                let Some(Value::GpuTensor(stored)) =
                    layer.properties().get(option.canonical_name())
                else {
                    panic!("{kind:?}/{option:?} lost resident storage")
                };
                assert!(stored.shares_allocation_with(&source));
                assert_eq!(stored.view(), source.view());
                assert_eq!(stored.physical(), source.physical());
                assert!(handle_is_explicit(stored));
                stored.allocation_lease().validate_live().unwrap();
                assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
                assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
            }
        }
        source.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn initializer_closure_captures_keep_their_resident_lease_without_invocation() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = provider
            .upload(&HostTensorView {
                data: &[2.0, 5.0],
                shape: &[2, 1],
            })
            .unwrap();
        let callback = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "not_invoked_during_construction".into(),
                binding: None,
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: vec![Value::GpuTensor(source.clone())].into(),
            },
        ));
        provider.reset_telemetry();
        let Value::Object(layer) = construct(
            DeepLearningLayerKind::FullyConnected,
            vec![
                Value::String(
                    DeepLearningLayerOption::WeightsInitializer
                        .canonical_name()
                        .into(),
                ),
                callback.clone(),
            ],
        ) else {
            panic!("layer expected")
        };
        let stored = layer
            .properties()
            .get(DeepLearningLayerOption::WeightsInitializer.canonical_name())
            .unwrap();
        assert_eq!(stored, &callback);
        source.allocation_lease().validate_live().unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    });
}

#[test]
fn downloaded_invalid_control_rejects_before_publishing_a_layer() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let epsilon = provider
            .upload(&HostTensorView {
                data: &[0.002],
                shape: &[1, 1],
            })
            .unwrap();
        let factor = provider
            .upload(&HostTensorView {
                data: &[-1.0],
                shape: &[1, 1],
            })
            .unwrap();
        provider.reset_telemetry();
        let error = block_on(layer_normalization_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(DeepLearningLayerOption::Epsilon.canonical_name().into()),
                Value::GpuTensor(epsilon.clone()),
                Value::String(
                    DeepLearningLayerOption::ScaleLearnRateFactor
                        .canonical_name()
                        .into(),
                ),
                Value::GpuTensor(factor.clone()),
            ]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert!(provider.telemetry_snapshot().download_bytes > 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        epsilon.allocation_lease().validate_live().unwrap();
        factor.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn resident_structural_controls_still_materialize_at_the_control_boundary() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let epsilon = provider
            .upload(&HostTensorView {
                data: &[0.002],
                shape: &[1, 1],
            })
            .unwrap();
        provider.reset_telemetry();
        let Value::Object(layer) = block_on(layer_normalization_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(DeepLearningLayerOption::Epsilon.canonical_name().into()),
                Value::GpuTensor(epsilon.clone()),
            ]),
        ))
        .unwrap() else {
            panic!("normalization layer expected")
        };
        let control = layer
            .properties()
            .get(DeepLearningLayerOption::Epsilon.canonical_name())
            .unwrap();
        assert!(!matches!(control, Value::GpuTensor(_)));
        assert_eq!(numeric_scalar(control, "test", "Epsilon").unwrap(), 0.002);
        assert!(provider.telemetry_snapshot().download_bytes > 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        epsilon.allocation_lease().validate_live().unwrap();
    });
}
