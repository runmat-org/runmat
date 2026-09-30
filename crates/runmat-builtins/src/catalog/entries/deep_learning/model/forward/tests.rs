use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn infer(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        &FORWARD_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

fn network() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DL_NETWORK.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn numeric(class: NumericClass) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(4)]),
        StorageFact::Dense,
    )
}

#[test]
fn forward_infers_observation_rows_and_floating_output_class() {
    for (input, output) in [
        (NumericClass::Double, NumericClass::Double),
        (NumericClass::Single, NumericClass::Single),
        (NumericClass::UInt16, NumericClass::Double),
    ] {
        let inferred = infer(vec![network(), numeric(input)]);
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].numeric().map(|fact| fact.class),
            Some(output)
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(3), None])
        );
        assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
    }
}

#[test]
fn forward_uses_known_fully_connected_weight_geometry_without_guessing_values() {
    let mut net = network();
    let ValueKindFact::Object(object) = &mut net.kind else {
        unreachable!()
    };
    let fc = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(
                runmat_types::StaticClassIdentity::new("nnet.cnn.layer.FullyConnectedLayer")
                    .owned(),
            ),
            properties: BTreeMap::from([(
                "Weights".into(),
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::from(vec![Some(7), Some(4)]),
                    StorageFact::Dense,
                ),
            )]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let input_layer = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(
                runmat_types::StaticClassIdentity::new("nnet.cnn.layer.FeatureInputLayer").owned(),
            ),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    object.properties.insert(
        "Layers".into(),
        ValueFact::proven(
            ValueKindFact::Cell(runmat_types::CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: vec![input_layer, fc],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        ),
    );
    let inferred = infer(vec![net, numeric(NumericClass::Single)]);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(7)])
    );
}

#[test]
fn dlarray_output_preserves_wrapper_and_data_residency() {
    let mut data = numeric(NumericClass::Single);
    data.residency = ResidencyFact::Device { provider: None };
    let wrapper = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DL_ARRAY.owned()),
            properties: BTreeMap::from([(crate::DLARRAY_DATA_PROPERTY.into(), data)]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::from(vec![Some(3), Some(4)]),
        StorageFact::Opaque,
    );
    let inferred = infer(vec![network(), wrapper]);
    assert!(inferred.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("dlarray expected")
    };
    assert_eq!(
        object.runtime_class,
        Some(runmat_types::standard::DL_ARRAY.owned())
    );
    assert_eq!(
        object.properties[crate::DLARRAY_DATA_PROPERTY]
            .numeric()
            .map(|fact| fact.class),
        Some(NumericClass::Single)
    );
    assert!(matches!(
        object.properties[crate::DLARRAY_DATA_PROPERTY].residency,
        ResidencyFact::Device { .. }
    ));
}

#[test]
fn invalid_network_data_and_extra_options_are_diagnosed() {
    let bad = ValueFact::scalar(ValueKindFact::String);
    assert!(!infer(vec![bad.clone(), numeric(NumericClass::Double)])
        .diagnostics
        .is_empty());
    assert!(!infer(vec![network(), bad.clone()]).diagnostics.is_empty());
    assert!(!infer(vec![network(), numeric(NumericClass::Double), bad])
        .diagnostics
        .is_empty());
    assert!(infer(vec![
        network(),
        ValueFact::unknown(DynamicReason::RuntimeValue)
    ])
    .diagnostics
    .is_empty());
}
