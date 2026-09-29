use super::*;
use futures::executor::block_on;

fn assert_class(value: Value, expected: runmat_types::StaticClassIdentity) {
    let Value::Object(layer) = value else {
        panic!("layer constructor must return an object");
    };
    assert!(layer.class_name.is(expected));
}

#[test]
fn every_constructor_uses_the_catalog_owned_layer_class_identity() {
    let size = || Value::Num(2.0);
    assert_class(
        block_on(fully_connected_layer_builtin(size(), vec![])).unwrap(),
        FULLY_CONNECTED_LAYER_CLASS,
    );
    assert_class(
        block_on(feature_input_layer_builtin(size(), vec![])).unwrap(),
        FEATURE_INPUT_LAYER_CLASS,
    );
    assert_class(
        block_on(sequence_input_layer_builtin(size(), vec![])).unwrap(),
        SEQUENCE_INPUT_LAYER_CLASS,
    );
    assert_class(
        block_on(relu_layer_builtin(vec![])).unwrap(),
        RELU_LAYER_CLASS,
    );
    assert_class(
        block_on(elu_layer_builtin(vec![])).unwrap(),
        ELU_LAYER_CLASS,
    );
    assert_class(
        block_on(softmax_layer_builtin(vec![])).unwrap(),
        SOFTMAX_LAYER_CLASS,
    );
    assert_class(
        block_on(classification_layer_builtin(vec![])).unwrap(),
        CLASSIFICATION_OUTPUT_LAYER_CLASS,
    );
    assert_class(
        block_on(regression_layer_builtin(vec![])).unwrap(),
        REGRESSION_OUTPUT_LAYER_CLASS,
    );
    assert_class(
        block_on(layer_normalization_layer_builtin(vec![])).unwrap(),
        LAYER_NORMALIZATION_LAYER_CLASS,
    );
    assert_class(
        block_on(global_average_pooling_1d_layer_builtin(vec![])).unwrap(),
        GLOBAL_AVERAGE_POOLING_1D_LAYER_CLASS,
    );
    assert_class(
        block_on(lstm_layer_builtin(size(), vec![])).unwrap(),
        LSTM_LAYER_CLASS,
    );
    assert_class(
        block_on(bilstm_layer_builtin(size(), vec![])).unwrap(),
        BILSTM_LAYER_CLASS,
    );
    assert_class(
        block_on(convolution_1d_layer_builtin(size(), size(), vec![])).unwrap(),
        CONVOLUTION_1D_LAYER_CLASS,
    );
}
