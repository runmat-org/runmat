use futures::executor::block_on;
use runmat_builtins::{
    DeepLearningAxisLabel, DeepLearningLayerKind, DeepLearningLayerOption, LayerOptionDefault,
};
use runmat_value::{CellArray, Tensor, Value};

use super::HostNetworkPlan;
use crate::builtins::deep_learning::{layers::*, model::dlnetwork_builtin};

fn network(layers: Vec<Value>) -> crate::execution::RootedValue {
    let len = layers.len();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(layers, len, 1).expect("layer array"),
    )]))
    .expect("initialized network")
}

#[test]
fn public_conv_default_has_valid_geometry_and_explicit_same_remains_distinct() {
    let input = Value::Tensor(Tensor::from_f32(vec![1.0; 2 * 3 * 5], vec![2, 3, 5]).unwrap());
    for (options, expected_time, expected_padding) in [
        (Vec::new(), 3, Value::Num(0.0)),
        (
            vec![
                Value::String("Padding".into()),
                Value::String("same".into()),
            ],
            5,
            Value::String("same".into()),
        ),
    ] {
        let input_layer = block_on(sequence_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("sequence input layer");
        let convolution = block_on(convolution_1d_layer_builtin(
            Value::Num(3.0),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(options),
        ))
        .expect("convolution layer");
        let Value::Object(layer) = &convolution else {
            panic!("convolution object expected")
        };
        assert_eq!(layer.properties().get("Padding"), Some(&expected_padding));
        let network_owner = network(vec![input_layer, convolution]);
        let Value::Object(network) = network_owner.value() else {
            panic!("network object expected")
        };
        let plan = HostNetworkPlan::admit(network, &input, "forward")
            .expect("public convolution must admit geometry");
        assert_eq!(
            plan.output_axes().axis_size(DeepLearningAxisLabel::Time),
            Some(expected_time)
        );
        assert_eq!(
            plan.output_axes().axis_size(DeepLearningAxisLabel::Channel),
            Some(2)
        );
        assert_eq!(
            plan.output_axes().axis_size(DeepLearningAxisLabel::Batch),
            Some(3)
        );
    }
    assert_eq!(
        DeepLearningLayerOption::Padding.default_for(DeepLearningLayerKind::Convolution1D),
        Some(LayerOptionDefault::Number(0.0))
    );
}

#[test]
fn public_recurrent_defaults_preserve_time_axis_for_both_directions() {
    let input = Value::Tensor(Tensor::from_f32(vec![1.0; 3 * 2 * 4], vec![3, 2, 4]).unwrap());
    for (kind, channels) in [
        (DeepLearningLayerKind::Lstm, 2),
        (DeepLearningLayerKind::Bilstm, 4),
    ] {
        for (options, expected_time, expected_mode) in [
            (Vec::new(), Some(4), "sequence"),
            (
                vec![
                    Value::String("OutputMode".into()),
                    Value::String("last".into()),
                ],
                None,
                "last",
            ),
        ] {
            let input_layer = block_on(sequence_input_layer_builtin(
                Value::Num(3.0),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("sequence input layer");
            let recurrent = match kind {
                DeepLearningLayerKind::Lstm => block_on(lstm_layer_builtin(
                    Value::Num(2.0),
                    crate::builtins::common::test_support::builtin_arguments(options),
                )),
                DeepLearningLayerKind::Bilstm => block_on(bilstm_layer_builtin(
                    Value::Num(2.0),
                    crate::builtins::common::test_support::builtin_arguments(options),
                )),
                _ => unreachable!("fixture uses recurrent layer kind"),
            }
            .expect("recurrent layer");
            let Value::Object(layer) = &recurrent else {
                panic!("recurrent object expected")
            };
            assert_eq!(
                layer.properties().get("OutputMode"),
                Some(&Value::String(expected_mode.into()))
            );
            let network_owner = network(vec![input_layer, recurrent]);
            let Value::Object(network) = network_owner.value() else {
                panic!("network object expected")
            };
            let plan = HostNetworkPlan::admit(network, &input, "forward")
                .expect("public recurrent layer must admit geometry");
            assert_eq!(
                plan.output_axes().axis_size(DeepLearningAxisLabel::Time),
                expected_time
            );
            assert_eq!(
                plan.output_axes().axis_size(DeepLearningAxisLabel::Channel),
                Some(channels)
            );
            assert_eq!(
                plan.output_axes().axis_size(DeepLearningAxisLabel::Batch),
                Some(2)
            );
        }
        assert_eq!(
            DeepLearningLayerOption::OutputMode.default_for(kind),
            Some(LayerOptionDefault::Text("sequence"))
        );
    }
}
