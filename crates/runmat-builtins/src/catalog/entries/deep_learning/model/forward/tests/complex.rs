use super::*;

#[test]
fn complex_host_pooling_preserves_precision_domain_and_known_raw_axes() {
    use crate::DeepLearningLayerKind as K;
    let sequence = layer(
        K::SequenceInput,
        BTreeMap::from([(
            "InputSize".into(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        )]),
    );
    let network = network_with_layers(vec![
        sequence,
        layer(K::GlobalAveragePooling1D, BTreeMap::new()),
    ]);
    for class in [NumericClass::Single, NumericClass::Double] {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            }),
            ShapeFact::from(vec![Some(4), Some(3), Some(8)]),
            StorageFact::Dense,
        );
        let inferred = infer(vec![network.clone(), input]);
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].numeric(),
            Some(NumericFact {
                class,
                domain: NumericDomain::Complex,
            })
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(3)])
        );
        assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
    }
}

#[test]
fn complex_unknown_topology_stays_dynamic_and_real_only_layer_is_diagnosed() {
    use crate::DeepLearningLayerKind as K;
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(3), Some(4)]),
        StorageFact::Dense,
    );
    let unknown = infer(vec![network(), input.clone()]);
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);

    let known_real_only = network_with_layers(vec![
        layer(K::FeatureInput, BTreeMap::new()),
        layer(K::FullyConnected, BTreeMap::new()),
    ]);
    let rejected = infer(vec![known_real_only, input]);
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-FORWARD-COMPLEX-LAYER"));
    assert!(!rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-FORWARD-DATA"));
}

#[test]
fn complex_dlarray_with_unknown_layers_retains_wrapper_without_false_data_error() {
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        StorageFact::Dense,
    );
    let wrapper = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DL_ARRAY.owned()),
            properties: BTreeMap::from([(crate::DLARRAY_DATA_PROPERTY.into(), data)]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        StorageFact::Opaque,
    );
    let inferred = infer(vec![network(), wrapper]);
    assert!(inferred.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("dlarray wrapper expected")
    };
    assert_eq!(
        object.runtime_class,
        Some(runmat_types::standard::DL_ARRAY.owned())
    );
    assert_eq!(
        object.properties[crate::DLARRAY_DATA_PROPERTY].kind,
        ValueKindFact::Unknown
    );
}

#[test]
fn known_complex_dlarray_route_retains_complex_class_but_not_unproven_format_shape() {
    use crate::DeepLearningLayerKind as K;
    let network = network_with_layers(vec![
        layer(
            K::SequenceInput,
            BTreeMap::from([(
                "InputSize".into(),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            )]),
        ),
        layer(K::GlobalAveragePooling1D, BTreeMap::new()),
    ]);
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(4), Some(3), Some(8)]),
        StorageFact::Dense,
    );
    let wrapper = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DL_ARRAY.owned()),
            properties: BTreeMap::from([(crate::DLARRAY_DATA_PROPERTY.into(), data)]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::from(vec![Some(4), Some(3), Some(8)]),
        StorageFact::Opaque,
    );
    let inferred = infer(vec![network, wrapper]);
    assert!(inferred.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("dlarray wrapper expected")
    };
    assert_eq!(
        object.properties[crate::DLARRAY_DATA_PROPERTY].numeric(),
        Some(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(
        object.properties[crate::DLARRAY_DATA_PROPERTY].shape,
        ShapeFact::Unknown
    );
}
