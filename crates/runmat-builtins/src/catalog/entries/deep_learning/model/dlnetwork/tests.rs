use super::*;
use runmat_types::{
    CallRequest, CellFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact,
    StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        &DLNETWORK_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

fn layer() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("nnet.cnn.layer.ReLULayer").owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn static_network_identity_and_layer_geometry_follow_known_collection() {
    let result = infer(
        vec![ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(layer()),
                elements: vec![layer(), layer()],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        )],
        vec![LiteralValue::Unknown],
    );
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(network) = &result.outputs[0].kind else {
        panic!("network object expected")
    };
    assert_eq!(
        network.runtime_class,
        Some(runmat_types::standard::DL_NETWORK.owned())
    );
    assert_eq!(
        network.properties["Layers"].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    let ValueKindFact::Cell(layers) = &network.properties["Layers"].kind else {
        panic!("cell fact expected")
    };
    assert_eq!(layers.elements.len(), 2);
    assert!(!network.properties_complete);
}

#[test]
fn malformed_known_layers_and_options_are_diagnosed() {
    let number = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let bad_cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(number.clone()),
            elements: vec![number.clone()],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    assert!(!infer(vec![bad_cell], vec![LiteralValue::Unknown])
        .diagnostics
        .is_empty());
    assert!(!infer(vec![number], vec![LiteralValue::Unknown])
        .diagnostics
        .is_empty());
    assert!(!infer(
        vec![
            layer(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(DynamicReason::RuntimeValue)
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("InputNames".into()),
            LiteralValue::Unknown
        ],
    )
    .diagnostics
    .is_empty());
}

#[test]
fn unknown_object_identity_does_not_imply_one_layer() {
    let mut source = layer();
    let ValueKindFact::Object(object) = &mut source.kind else {
        unreachable!()
    };
    object.runtime_class = None;
    let result = infer(vec![source], vec![LiteralValue::Unknown]);
    let ValueKindFact::Object(network) = &result.outputs[0].kind else {
        panic!("network object expected")
    };
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        network.properties["Layers"].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    let ValueKindFact::Cell(layers) = &network.properties["Layers"].kind else {
        panic!("cell fact expected")
    };
    assert!(!layers.elements_complete);
    assert!(layers.elements.is_empty());
}
