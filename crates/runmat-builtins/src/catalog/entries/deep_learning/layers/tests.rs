use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StaticClassIdentity, ValueFact,
    ValueKindFact,
};

fn infer(
    entry: &BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

fn numeric() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn character() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Character)
}

#[test]
fn every_layer_has_its_own_runtime_class_and_catalog_documentation() {
    for (entry, count, class) in [
        (
            &FULLY_CONNECTED_CATALOG_ENTRY,
            1,
            "nnet.cnn.layer.FullyConnectedLayer",
        ),
        (
            &FEATURE_INPUT_CATALOG_ENTRY,
            1,
            "nnet.cnn.layer.FeatureInputLayer",
        ),
        (
            &SEQUENCE_INPUT_CATALOG_ENTRY,
            1,
            "nnet.cnn.layer.SequenceInputLayer",
        ),
        (&RELU_CATALOG_ENTRY, 0, "nnet.cnn.layer.ReLULayer"),
        (&ELU_CATALOG_ENTRY, 0, "nnet.cnn.layer.ELULayer"),
        (&SOFTMAX_CATALOG_ENTRY, 0, "nnet.cnn.layer.SoftmaxLayer"),
        (
            &CLASSIFICATION_CATALOG_ENTRY,
            0,
            "nnet.cnn.layer.ClassificationOutputLayer",
        ),
        (
            &REGRESSION_CATALOG_ENTRY,
            0,
            "nnet.cnn.layer.RegressionOutputLayer",
        ),
        (
            &LAYER_NORM_CATALOG_ENTRY,
            0,
            "nnet.cnn.layer.LayerNormalizationLayer",
        ),
        (
            &GLOBAL_POOL_CATALOG_ENTRY,
            0,
            "nnet.cnn.layer.GlobalAveragePooling1DLayer",
        ),
        (&LSTM_CATALOG_ENTRY, 1, "nnet.cnn.layer.LSTMLayer"),
        (&BILSTM_CATALOG_ENTRY, 1, "nnet.cnn.layer.BiLSTMLayer"),
        (
            &CONVOLUTION_CATALOG_ENTRY,
            2,
            "nnet.cnn.layer.Convolution1DLayer",
        ),
    ] {
        let result = infer(entry, vec![numeric(); count], vec![]);
        assert!(
            result.diagnostics.is_empty(),
            "{}: {:?}",
            entry.identity.name,
            result.diagnostics
        );
        let ValueKindFact::Object(ObjectFact {
            runtime_class,
            properties,
            properties_complete,
            ..
        }) = &result.outputs[0].kind
        else {
            panic!("{} must return a layer object", entry.identity.name)
        };
        assert_eq!(
            runtime_class.as_ref(),
            Some(&StaticClassIdentity::new(class).owned())
        );
        assert!(!properties_complete);
        assert!(properties.contains_key("Name"));
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert!(!entry.documentation.examples.is_empty());
        assert_eq!(
            entry.descriptor.errors[0].identifier,
            Some("RunMat:deepLearning:InvalidInput")
        );
    }
}

#[test]
fn invalid_sizes_and_known_invalid_options_are_diagnosed() {
    let wrong_size = infer(
        &FULLY_CONNECTED_CATALOG_ENTRY,
        vec![numeric()],
        vec![LiteralValue::Number(0.0)],
    );
    assert!(!wrong_size.diagnostics.is_empty());
    let negative = infer(
        &CONVOLUTION_CATALOG_ENTRY,
        vec![numeric(), numeric()],
        vec![LiteralValue::Number(3.0), LiteralValue::Number(-2.0)],
    );
    assert!(!negative.diagnostics.is_empty());
    let malformed = infer(
        &RELU_CATALOG_ENTRY,
        vec![character()],
        vec![LiteralValue::Character("Name".into())],
    );
    assert!(!malformed.diagnostics.is_empty());
    let invalid_name = infer(
        &RELU_CATALOG_ENTRY,
        vec![character(), numeric()],
        vec![
            LiteralValue::Character("Name".into()),
            LiteralValue::Number(1.0),
        ],
    );
    assert!(!invalid_name.diagnostics.is_empty());
    let valid = infer(
        &RELU_CATALOG_ENTRY,
        vec![character(), character()],
        vec![
            LiteralValue::Character("Name".into()),
            LiteralValue::Character("relu".into()),
        ],
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
}

#[test]
fn feature_vector_extension_and_integer_control_forms_are_declared() {
    assert_eq!(FEATURE_INPUT_CATALOG_ENTRY.extensions.len(), 1);
    assert_eq!(
        FEATURE_INPUT_CATALOG_ENTRY.extensions[0].id,
        FEATURE_INPUT_VECTOR_EXTENSION.id
    );
    assert_eq!(FEATURE_INPUT_CATALOG_ENTRY.integer_capabilities.len(), 2);
    assert_eq!(FC_INTEGER_CAPABILITIES.len(), 1);
    for entry in [
        &FULLY_CONNECTED_CATALOG_ENTRY,
        &FEATURE_INPUT_CATALOG_ENTRY,
        &SEQUENCE_INPUT_CATALOG_ENTRY,
        &LSTM_CATALOG_ENTRY,
        &BILSTM_CATALOG_ENTRY,
        &CONVOLUTION_CATALOG_ENTRY,
    ] {
        assert!(
            !entry.integer_capabilities.is_empty(),
            "{}",
            entry.identity.name
        );
    }
    let inexact = infer(
        &FULLY_CONNECTED_CATALOG_ENTRY,
        vec![numeric()],
        vec![LiteralValue::Integer {
            text: "9007199254740993".into(),
            class: NumericClass::UInt64,
        }],
    );
    assert!(!inexact.diagnostics.is_empty());
    let aligned = infer(
        &FULLY_CONNECTED_CATALOG_ENTRY,
        vec![numeric()],
        vec![LiteralValue::Integer {
            text: "18014398509481984".into(),
            class: NumericClass::UInt64,
        }],
    );
    assert!(aligned.diagnostics.is_empty());
}
