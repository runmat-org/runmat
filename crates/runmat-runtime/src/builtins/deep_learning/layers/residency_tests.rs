use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{handle_is_explicit, GpuHandleProvenance, HostTensorView};

fn construct(kind: DeepLearningLayerKind, options: Vec<Value>) -> Value {
    use DeepLearningLayerKind as K;
    block_on(async {
        match kind {
            K::FullyConnected => fully_connected_layer_builtin(Value::Num(2.0), options).await,
            K::Lstm => lstm_layer_builtin(Value::Num(2.0), options).await,
            K::Bilstm => bilstm_layer_builtin(Value::Num(2.0), options).await,
            K::Convolution1D => {
                convolution_1d_layer_builtin(Value::Num(3.0), Value::Num(2.0), options).await
            }
            K::LayerNormalization => layer_normalization_layer_builtin(options).await,
            _ => unreachable!("only learnable layer constructors in this fixture"),
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
            (K::FullyConnected, &[O::Weights, O::Bias][..]),
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
                let Some(Value::GpuTensor(stored)) = layer.properties.get(option.canonical_name())
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
fn resident_structural_controls_still_materialize_at_the_control_boundary() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let epsilon = provider
            .upload(&HostTensorView {
                data: &[0.002],
                shape: &[1, 1],
            })
            .unwrap();
        provider.reset_telemetry();
        let Value::Object(layer) = block_on(layer_normalization_layer_builtin(vec![
            Value::String(DeepLearningLayerOption::Epsilon.canonical_name().into()),
            Value::GpuTensor(epsilon.clone()),
        ]))
        .unwrap() else {
            panic!("normalization layer expected")
        };
        let control = layer
            .properties
            .get(DeepLearningLayerOption::Epsilon.canonical_name())
            .unwrap();
        assert!(!matches!(control, Value::GpuTensor(_)));
        assert_eq!(numeric_scalar(control, "test", "Epsilon").unwrap(), 0.002);
        assert!(provider.telemetry_snapshot().download_bytes > 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        epsilon.allocation_lease().validate_live().unwrap();
    });
}
