use super::*;
use runmat_types::{
    CallRequest, CellFact, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact,
    StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn model(class: StaticClassIdentity) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn predictors(class: NumericClass) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(2)]),
        StorageFact::Dense,
    )
}

fn infer(model: ValueFact, input: ValueFact, outputs: usize) -> runmat_types::CallInference {
    super::inference::infer(
        &CallRequest {
            arguments: vec![model, input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &PREDICT_CATALOG_ENTRY,
    )
}

fn dlarray(data: Option<ValueFact>, properties_complete: bool) -> ValueFact {
    let mut properties = BTreeMap::new();
    if let Some(data) = data {
        properties.insert(crate::DLARRAY_DATA_PROPERTY.into(), data);
    }
    model_with_properties(
        runmat_types::standard::DL_ARRAY,
        properties,
        properties_complete,
    )
}

fn model_with_properties(
    class: StaticClassIdentity,
    properties: BTreeMap<String, ValueFact>,
    properties_complete: bool,
) -> ValueFact {
    let mut result = model(class);
    let ValueKindFact::Object(object) = &mut result.kind else {
        unreachable!()
    };
    object.properties = properties;
    object.properties_complete = properties_complete;
    result
}

fn network_with_layer(layer: crate::DeepLearningLayerKind) -> ValueFact {
    let layer = model(layer.class());
    model_with_properties(
        runmat_types::standard::DL_NETWORK,
        BTreeMap::from([(
            "Layers".into(),
            ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(layer.clone()),
                    elements: vec![layer],
                    elements_complete: true,
                }),
                ShapeFact::from(vec![Some(1), Some(1)]),
                StorageFact::Dense,
            ),
        )]),
        false,
    )
}

#[test]
fn typed_statistical_model_identity_controls_output_arity() {
    let linear = infer(
        model(runmat_types::standard::LINEAR_MODEL),
        predictors(NumericClass::Double),
        2,
    );
    assert!(linear.diagnostics.is_empty());
    assert_eq!(linear.outputs.len(), 2);
    assert_eq!(
        linear.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        linear.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );

    let tree = infer(
        model(runmat_types::standard::CLASSIFICATION_TREE),
        predictors(NumericClass::Double),
        4,
    );
    assert!(tree.diagnostics.is_empty());
    assert_eq!(tree.outputs.len(), 4);
    assert!(matches!(tree.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(
        tree.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );

    let classifier = infer(
        model(runmat_types::standard::CLASSIFICATION_LINEAR),
        predictors(NumericClass::Double),
        3,
    );
    assert!(!classifier.diagnostics.is_empty());
}

#[test]
fn deep_learning_predict_retains_single_precision_without_claiming_shape() {
    let network = model(runmat_types::standard::DL_NETWORK);
    let single = infer(network.clone(), predictors(NumericClass::Single), 2);
    assert!(single.diagnostics.is_empty());
    assert_eq!(single.outputs.len(), 2);
    assert_eq!(
        single.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert!(matches!(single.outputs[0].shape, ShapeFact::Unknown));
    let integer = infer(network, predictors(NumericClass::UInt16), 1);
    assert_eq!(
        integer.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn unsupported_model_is_diagnosed_without_inventing_a_concrete_output() {
    let result = infer(
        model(StaticClassIdentity::new("UnrelatedModel")),
        predictors(NumericClass::Double),
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn unknown_or_missing_network_data_does_not_become_proven_double() {
    let network = model(runmat_types::standard::DL_NETWORK);
    for input in [
        ValueFact::unknown(DynamicReason::RuntimeValue),
        dlarray(None, false),
        dlarray(Some(ValueFact::unknown(DynamicReason::RuntimeValue)), true),
    ] {
        let result = infer(network.clone(), input, 1);
        let data = match &result.outputs[0].kind {
            ValueKindFact::Object(object) => object.properties.get(crate::DLARRAY_DATA_PROPERTY),
            _ => Some(&result.outputs[0]),
        };
        assert!(data.is_some_and(|data| data.numeric().is_none()));
    }
    let missing = infer(network, dlarray(None, true), 1);
    assert!(missing.outputs[0].numeric().is_none());
}

#[test]
fn network_domain_and_residency_follow_canonical_forward_inference() {
    let mut complex = predictors(NumericClass::Single);
    complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Complex,
    });
    let supported = infer(
        network_with_layer(crate::DeepLearningLayerKind::FeatureInput),
        complex.clone(),
        1,
    );
    assert_eq!(
        supported.outputs[0].numeric().map(|number| number.domain),
        Some(NumericDomain::Complex)
    );
    assert_eq!(supported.outputs[0].residency, ResidencyFact::Host);
    let unknown = infer(
        model(runmat_types::standard::DL_NETWORK),
        complex.clone(),
        1,
    );
    assert!(unknown.outputs[0].numeric().is_none());
    let unsupported = infer(
        network_with_layer(crate::DeepLearningLayerKind::Relu),
        complex,
        1,
    );
    assert!(!unsupported.diagnostics.is_empty());
    assert!(unsupported.outputs[0].numeric().is_none());

    let mut resident = predictors(NumericClass::Single);
    resident.residency = ResidencyFact::Device { provider: None };
    let wrapped = infer(
        model(runmat_types::standard::DL_NETWORK),
        dlarray(Some(resident), true),
        2,
    );
    assert_eq!(wrapped.outputs.len(), 2);
    assert_eq!(wrapped.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Object(object) = &wrapped.outputs[0].kind else {
        panic!("expected dlarray wrapper")
    };
    let data = &object.properties[crate::DLARRAY_DATA_PROPERTY];
    assert_eq!(
        data.numeric().map(|number| number.class),
        Some(NumericClass::Single)
    );
    assert_eq!(data.residency, ResidencyFact::Unknown);
    assert!(matches!(data.shape, ShapeFact::Unknown));
    assert_eq!(wrapped.outputs[0].storage, StorageFact::Opaque);
    assert_eq!(data.storage, StorageFact::Dense);

    let mut resident_complex = predictors(NumericClass::Single);
    resident_complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Complex,
    });
    resident_complex.residency = ResidencyFact::Device { provider: None };
    let rejected = infer(
        model(runmat_types::standard::DL_NETWORK),
        dlarray(Some(resident_complex), true),
        1,
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-PREDICT-RESIDENT-CLASS"));
    let mut sparse = predictors(NumericClass::Double);
    sparse.storage = StorageFact::Sparse;
    let rejected = infer(model(runmat_types::standard::DL_NETWORK), sparse, 1);
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-PREDICT-STORAGE"));
}

#[test]
fn known_raw_network_rank_survives_predict_options_without_forward_arity_diagnostics() {
    let request = CallRequest {
        arguments: vec![
            network_with_layer(crate::DeepLearningLayerKind::FeatureInput),
            predictors(NumericClass::Single),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inferred = super::inference::infer(&request, &PREDICT_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Ranked { rank: 2 });
    assert_eq!(
        inferred.outputs[0].numeric().map(|number| number.class),
        Some(NumericClass::Single)
    );
}
