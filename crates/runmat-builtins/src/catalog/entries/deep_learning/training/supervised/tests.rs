use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StaticClassIdentity, StorageFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn object(name: &str) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(runmat_types::ClassIdentity::new(name).unwrap()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }))
}

fn predictors() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), Some(2)]),
        StorageFact::Dense,
    )
}

fn infer(
    entry: &'static BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: usize,
) -> CallInference {
    crate::infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn network_class_and_two_result_structure_are_inferred() {
    let result = infer(
        &TRAIN_NETWORK_CATALOG_ENTRY,
        vec![
            predictors(),
            predictors(),
            object("nnet.cnn.LayerGraph"),
            object("nnet.cnn.TrainingOptionsSGDM"),
        ],
        vec![],
        2,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(network) = &result.outputs[0].kind else {
        panic!("expected network fact")
    };
    assert_eq!(
        network.runtime_class,
        Some(StaticClassIdentity::new("DAGNetwork").owned())
    );
    let ValueKindFact::Struct(info) = &result.outputs[1].kind else {
        panic!("expected training information fact")
    };
    assert!(info.fields.contains_key("TrainingLoss"));
    let loss_history = &info.fields["TrainingLoss"];
    assert_eq!(loss_history.shape.known_dims(), Some(vec![None, Some(1)]));
    assert_eq!(loss_history.storage, StorageFact::Dense);
    assert!(matches!(
        loss_history.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    ));

    let series = infer(
        &TRAIN_NETWORK_CATALOG_ENTRY,
        vec![
            predictors(),
            predictors(),
            object("nnet.cnn.layer.FeatureInputLayer"),
            object("nnet.cnn.TrainingOptionsADAM"),
        ],
        vec![],
        1,
    );
    let ValueKindFact::Object(network) = &series.outputs[0].kind else {
        panic!("expected network fact")
    };
    assert_eq!(
        network.runtime_class,
        Some(StaticClassIdentity::new("SeriesNetwork").owned())
    );
}

#[test]
fn known_wrong_options_loss_and_output_count_are_diagnosed() {
    let trainnet_args = vec![
        predictors(),
        predictors(),
        object("dlnetwork"),
        ValueFact::scalar(ValueKindFact::String),
        object("table"),
    ];
    let result = infer(
        &TRAINNET_CATALOG_ENTRY,
        trainnet_args,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("not-a-loss".into()),
            LiteralValue::Unknown,
        ],
        3,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|error| error.code == "RM-DL-SUPERVISED-OPTIONS"));
    assert!(result
        .diagnostics
        .iter()
        .any(|error| error.code == "RM-DL-TRAINNET-LOSS"));
    assert!(result
        .diagnostics
        .iter()
        .any(|error| error.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn unknown_layer_role_does_not_claim_series_network() {
    let result = infer(
        &TRAIN_NETWORK_CATALOG_ENTRY,
        vec![
            predictors(),
            predictors(),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            object("nnet.cnn.TrainingOptionsRMSProp"),
        ],
        vec![],
        1,
    );
    let ValueKindFact::Object(network) = &result.outputs[0].kind else {
        panic!("expected network fact")
    };
    assert!(network.runtime_class.is_none());
}

#[test]
fn supervised_calls_accept_only_the_three_typed_training_option_classes() {
    for solver in crate::TrainingSolver::ALL {
        let class = solver.class_identity().display_name();
        let result = infer(
            &TRAIN_NETWORK_CATALOG_ENTRY,
            vec![
                predictors(),
                predictors(),
                object("nnet.cnn.layer.FeatureInputLayer"),
                object(class),
            ],
            vec![],
            1,
        );
        assert!(
            result
                .diagnostics
                .iter()
                .all(|diagnostic| diagnostic.code != "RM-DL-SUPERVISED-OPTIONS"),
            "{solver:?}: {:?}",
            result.diagnostics
        );
    }
    let legacy = infer(
        &TRAIN_NETWORK_CATALOG_ENTRY,
        vec![
            predictors(),
            predictors(),
            object("nnet.cnn.layer.FeatureInputLayer"),
            object("nnet.cnn.TrainingOptions"),
        ],
        vec![],
        1,
    );
    assert!(legacy
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-SUPERVISED-OPTIONS"));
}
