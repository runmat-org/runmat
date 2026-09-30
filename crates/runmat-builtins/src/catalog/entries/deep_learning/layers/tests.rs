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
fn known_layer_options_have_canonical_static_facts_and_reject_unknown_keys() {
    let accepted = infer(
        &SEQUENCE_INPUT_CATALOG_ENTRY,
        vec![numeric(), character(), numeric()],
        vec![
            LiteralValue::Number(3.0),
            LiteralValue::Character("MinLength".into()),
            LiteralValue::Number(2.0),
        ],
    );
    assert!(
        accepted.diagnostics.is_empty(),
        "{:?}",
        accepted.diagnostics
    );
    let ValueKindFact::Object(object) = &accepted.outputs[0].kind else {
        panic!("layer")
    };
    assert_eq!(
        object.properties["MinLength"].numeric().unwrap().class,
        NumericClass::Double
    );

    let rejected = infer(
        &RELU_CATALOG_ENTRY,
        vec![character(), numeric()],
        vec![
            LiteralValue::Character("Padding".into()),
            LiteralValue::Number(1.0),
        ],
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-LAYER-OPTION-UNKNOWN"));

    let duplicate = infer(
        &RELU_CATALOG_ENTRY,
        vec![character(), character(), character(), character()],
        vec![
            LiteralValue::Character("Name".into()),
            LiteralValue::Character("one".into()),
            LiteralValue::Character("name".into()),
            LiteralValue::Character("two".into()),
        ],
    );
    assert!(duplicate
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-LAYER-OPTION-DUPLICATE"));
}

#[test]
fn recurrent_gate_factor_facts_admit_only_scalar_or_documented_row_geometry() {
    let factor = |shape| ValueFact { shape, ..numeric() };
    for (entry, width) in [(&LSTM_CATALOG_ENTRY, 4), (&BILSTM_CATALOG_ENTRY, 8)] {
        let arguments = |shape| vec![numeric(), character(), factor(shape)];
        let literals = vec![
            LiteralValue::Number(2.0),
            LiteralValue::Character("InputWeightsLearnRateFactor".into()),
            LiteralValue::Unknown,
        ];
        let row = infer(
            entry,
            arguments(ShapeFact::from(vec![Some(1), Some(width)])),
            literals.clone(),
        );
        assert!(row.diagnostics.is_empty(), "{:?}", row.diagnostics);
        let column = infer(
            entry,
            arguments(ShapeFact::from(vec![Some(width), Some(1)])),
            literals,
        );
        assert!(column
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-DL-LAYER-OPTION-VALUE"));
    }
}

#[test]
fn recurrent_gate_factor_facts_reject_known_negative_or_nonfinite_values() {
    for (entry, width) in [(&LSTM_CATALOG_ENTRY, 4), (&BILSTM_CATALOG_ENTRY, 8)] {
        let arguments = |shape| vec![numeric(), character(), ValueFact { shape, ..numeric() }];
        let infer_factor = |shape, literal| {
            infer(
                entry,
                arguments(shape),
                vec![
                    LiteralValue::Number(2.0),
                    LiteralValue::Character("InputWeightsLearnRateFactor".into()),
                    literal,
                ],
            )
        };
        for invalid in [
            LiteralValue::Number(-0.5),
            LiteralValue::Number(f64::NAN),
            LiteralValue::Number(f64::INFINITY),
            LiteralValue::Real {
                text: "-0.25".into(),
                class: NumericClass::Single,
            },
            LiteralValue::Integer {
                text: "-1".into(),
                class: NumericClass::Int64,
            },
        ] {
            let result = infer_factor(ShapeFact::Scalar, invalid);
            assert!(result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "RM-DL-LAYER-OPTION-VALUE"));
        }
        let mut values = vec![LiteralValue::Number(0.0); width];
        values[width - 1] = LiteralValue::Number(f64::NEG_INFINITY);
        let result = infer_factor(
            ShapeFact::from(vec![Some(1), Some(width)]),
            LiteralValue::Vector(values),
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-DL-LAYER-OPTION-VALUE"));
        let matrix = infer_factor(
            ShapeFact::from(vec![Some(1), Some(width)]),
            LiteralValue::Matrix(vec![vec![LiteralValue::Number(-1.0); width]]),
        );
        assert!(matrix
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-DL-LAYER-OPTION-VALUE"));
        let conservative = infer_factor(ShapeFact::Unknown, LiteralValue::Unknown);
        assert!(
            conservative.diagnostics.is_empty(),
            "{:?}",
            conservative.diagnostics
        );
        let valid = infer_factor(ShapeFact::Scalar, LiteralValue::Number(-0.0));
        assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
        let valid_row = infer_factor(
            ShapeFact::from(vec![Some(1), Some(width)]),
            LiteralValue::Vector(vec![LiteralValue::Number(0.25); width]),
        );
        assert!(
            valid_row.diagnostics.is_empty(),
            "{:?}",
            valid_row.diagnostics
        );
    }
}

#[test]
fn fully_connected_positional_dimension_has_a_canonical_fact() {
    let result = infer(
        &FULLY_CONNECTED_CATALOG_ENTRY,
        vec![numeric(), character(), character(), character()],
        vec![
            LiteralValue::Number(4.0),
            LiteralValue::Character("spatial-channel".into()),
            LiteralValue::Character("Name".into()),
            LiteralValue::Character("dense".into()),
        ],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("layer")
    };
    assert_eq!(
        object.properties["OperationDimension"].kind,
        ValueKindFact::Character
    );
    assert_eq!(FULLY_CONNECTED_DESCRIPTOR.signatures.len(), 2);
}

#[test]
fn layer_normalization_axis_uses_the_closed_typed_choices() {
    let accepted = infer(
        &LAYER_NORM_CATALOG_ENTRY,
        vec![character(), character()],
        vec![
            LiteralValue::Character("OperationDimension".into()),
            LiteralValue::Character("batch-excluded".into()),
        ],
    );
    assert!(
        accepted.diagnostics.is_empty(),
        "{:?}",
        accepted.diagnostics
    );

    let rejected = infer(
        &LAYER_NORM_CATALOG_ENTRY,
        vec![character(), character()],
        vec![
            LiteralValue::Character("OperationDimension".into()),
            LiteralValue::Character("unrecognized-axis".into()),
        ],
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-LAYER-OPTION-VALUE"));
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
