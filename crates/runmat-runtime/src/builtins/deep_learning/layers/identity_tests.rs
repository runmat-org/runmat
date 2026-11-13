use super::*;
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use futures::executor::block_on;
use runmat_builtins::{BILSTM_LAYER_CLASS, LSTM_LAYER_CLASS};

fn assert_class(value: Value, expected: runmat_types::StaticClassIdentity) {
    let Value::Object(layer) = value else {
        panic!("layer constructor must return an object");
    };
    assert!(layer.class_name().is(expected));
}

#[test]
fn every_constructor_uses_the_catalog_owned_layer_class_identity() {
    let size = || Value::Num(2.0);
    assert_class(
        block_on(
            crate::builtins::deep_learning::tests::fully_connected_layer_fixture(size(), vec![]),
        )
        .unwrap(),
        FULLY_CONNECTED_LAYER_CLASS,
    );
    assert_class(
        block_on(feature_input_layer_builtin(
            size(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        FEATURE_INPUT_LAYER_CLASS,
    );
    assert_class(
        block_on(sequence_input_layer_builtin(
            size(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        SEQUENCE_INPUT_LAYER_CLASS,
    );
    assert_class(
        block_on(relu_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        RELU_LAYER_CLASS,
    );
    assert_class(
        block_on(elu_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        ELU_LAYER_CLASS,
    );
    assert_class(
        block_on(softmax_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        SOFTMAX_LAYER_CLASS,
    );
    assert_class(
        block_on(classification_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        CLASSIFICATION_OUTPUT_LAYER_CLASS,
    );
    assert_class(
        block_on(regression_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        REGRESSION_OUTPUT_LAYER_CLASS,
    );
    assert_class(
        block_on(layer_normalization_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        LAYER_NORMALIZATION_LAYER_CLASS,
    );
    assert_class(
        block_on(global_average_pooling_1d_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        GLOBAL_AVERAGE_POOLING_1D_LAYER_CLASS,
    );
    assert_class(
        block_on(lstm_layer_builtin(
            size(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        LSTM_LAYER_CLASS,
    );
    assert_class(
        block_on(bilstm_layer_builtin(
            size(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        BILSTM_LAYER_CLASS,
    );
    assert_class(
        block_on(convolution_1d_layer_builtin(
            size(),
            size(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        CONVOLUTION_1D_LAYER_CLASS,
    );
}

#[test]
fn layer_options_are_canonical_and_reject_duplicates_and_cross_layer_names() {
    let sequence = block_on(sequence_input_layer_builtin(
        Value::Num(3.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("minlength".into()),
            Value::Num(2.0),
        ]),
    ))
    .expect("recognized sequence option");
    let Value::Object(sequence) = sequence else {
        panic!("sequence layer")
    };
    assert_eq!(
        sequence.properties().get("MinLength"),
        Some(&Value::Num(2.0))
    );
    assert!(!sequence.properties().contains_key("minlength"));

    for rest in [
        vec![
            Value::String("Padding".into()),
            Value::String("same".into()),
        ],
        vec![Value::String("invented".into()), Value::Num(1.0)],
        vec![
            Value::String("Name".into()),
            Value::String("a".into()),
            Value::String("name".into()),
            Value::String("b".into()),
        ],
    ] {
        let error = block_on(relu_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect_err("invalid ReLU option");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    }
}

#[test]
fn malformed_controls_reject_before_resident_option_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .expect("resident weights");
        provider.reset_telemetry();
        let error = block_on(fully_connected_layer_builtin(
            None,
            Value::Num(2.0),
            vec![
                Value::String("Weights".into()),
                Value::GpuTensor(resident),
                Value::String("OperationDimension".into()),
                Value::String("invalid".into()),
            ],
        ))
        .expect_err("invalid operation dimension");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn fully_connected_positional_operation_dimension_is_not_an_option_name() {
    let layer = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(4.0),
        vec![
            Value::String("spatial-channel".into()),
            Value::String("Name".into()),
            Value::String("dense".into()),
        ],
    ))
    .expect("documented positional dimension");
    let Value::Object(layer) = layer else {
        panic!("fully connected layer")
    };
    assert_eq!(
        layer.properties().get("OperationDimension"),
        Some(&Value::String("spatial-channel".into()))
    );
    assert_eq!(
        layer.properties().get("Name"),
        Some(&Value::String("dense".into()))
    );

    let upper = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(4.0),
        vec![Value::String("SPATIAL-CHANNEL".into())],
    ))
    .expect("case-insensitive positional dimension");
    let Value::Object(upper) = upper else {
        panic!("fully connected layer")
    };
    assert_eq!(
        upper.properties().get("OperationDimension"),
        Some(&Value::String("SPATIAL-CHANNEL".into()))
    );

    let duplicate = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(4.0),
        vec![
            Value::Num(2.0),
            Value::String("OperationDimension".into()),
            Value::Num(3.0),
        ],
    ))
    .expect_err("positional and named dimension conflict");
    assert_eq!(
        duplicate.identifier(),
        Some("RunMat:deepLearning:InvalidInput")
    );
}

#[test]
fn layer_normalization_axes_and_epsilon_are_validated_at_construction() {
    let layer = block_on(layer_normalization_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("OperationDimension".into()),
            Value::String("batch-excluded".into()),
            Value::String("Epsilon".into()),
            Value::Num(1.0e-4),
        ]),
    ))
    .expect("recognized normalization controls");
    let Value::Object(layer) = layer else {
        panic!("normalization layer")
    };
    assert_eq!(
        layer.properties().get("OperationDimension"),
        Some(&Value::String("batch-excluded".into()))
    );

    for options in [
        vec![
            Value::String("OperationDimension".into()),
            Value::String("not-an-axis".into()),
        ],
        vec![Value::String("Epsilon".into()), Value::Num(0.0)],
    ] {
        let error = block_on(layer_normalization_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(options),
        ))
        .expect_err("invalid normalization control");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    }
}
