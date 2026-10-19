use super::*;

pub(super) fn complex_network() -> ValueFact {
    network_with_layers(vec![
        layer(
            crate::DeepLearningLayerKind::SequenceInput,
            BTreeMap::from([(
                "InputSize".into(),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            )]),
        ),
        layer(
            crate::DeepLearningLayerKind::GlobalAveragePooling1D,
            BTreeMap::new(),
        ),
    ])
}

pub(super) fn complex_data(class: NumericClass) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(4), Some(3), Some(8)]),
        StorageFact::Dense,
    )
}

pub(super) fn wrapped(data: ValueFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DL_ARRAY.owned()),
            properties: BTreeMap::from([(crate::DLARRAY_DATA_PROPERTY.into(), data)]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Unknown,
        StorageFact::Opaque,
    )
}

pub(super) fn public_infer(
    caller: NetworkInferenceCaller,
    net: ValueFact,
    input: ValueFact,
) -> runmat_types::CallInference {
    let name = match caller {
        NetworkInferenceCaller::Forward => "forward",
        NetworkInferenceCaller::Predict => "predict",
    };
    let entry = crate::builtin_catalog_entry_by_name(name).expect("public network caller");
    crate::infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![net, input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}
