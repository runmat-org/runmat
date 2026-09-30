use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn infer(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    infer_outputs(arguments, 1)
}

fn infer_outputs(arguments: Vec<ValueFact>, outputs: usize) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        &FORWARD_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

fn feature_network() -> ValueFact {
    let mut network = network();
    let ValueKindFact::Object(object) = &mut network.kind else {
        unreachable!()
    };
    let layer = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(crate::FEATURE_INPUT_LAYER_CLASS.owned()),
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
                element: Box::new(layer.clone()),
                elements: vec![layer],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(1), Some(1)]),
            StorageFact::Dense,
        ),
    );
    network
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

fn layer(kind: crate::DeepLearningLayerKind, properties: BTreeMap<String, ValueFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(kind.class().owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn network_with_layers(layers: Vec<ValueFact>) -> ValueFact {
    let mut result = network();
    let ValueKindFact::Object(object) = &mut result.kind else {
        unreachable!()
    };
    let count = layers.len();
    object.properties.insert(
        "Layers".into(),
        ValueFact::proven(
            ValueKindFact::Cell(runmat_types::CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: layers,
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(count), Some(1)]),
            StorageFact::Dense,
        ),
    );
    result
}

fn shaped_single(dims: Vec<Option<usize>>) -> ValueFact {
    let mut value = numeric(NumericClass::Single);
    value.shape = ShapeFact::from(dims);
    value
}

#[test]
fn forward_infers_raw_feature_axes_and_floating_output_class() {
    for (input, output) in [
        (NumericClass::Double, NumericClass::Double),
        (NumericClass::Single, NumericClass::Single),
        (NumericClass::UInt16, NumericClass::Double),
    ] {
        let inferred = infer(vec![feature_network(), numeric(input)]);
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].numeric().map(|fact| fact.class),
            Some(output)
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(4)])
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
                        class: NumericClass::UInt16,
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
    let inferred = infer(vec![net.clone(), numeric(NumericClass::Single)]);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(7)])
    );
    assert_eq!(
        inferred.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::Single)
    );
    let ValueKindFact::Object(object) = &mut net.kind else {
        unreachable!()
    };
    let ValueKindFact::Cell(layers) = &mut object.properties.get_mut("Layers").unwrap().kind else {
        unreachable!()
    };
    let ValueKindFact::Object(layer) = &mut layers.elements.last_mut().unwrap().kind else {
        unreachable!()
    };
    layer.runtime_class = None;
    let inferred = infer(vec![net, numeric(NumericClass::Single)]);
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn rank_one_dense_input_is_one_observation_not_one_observation_per_feature() {
    let mut input = numeric(NumericClass::Single);
    input.shape = ShapeFact::from(vec![Some(4)]);
    let inferred = infer(vec![feature_network(), input]);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
}

#[test]
fn dlarray_output_preserves_wrapper_without_guessing_format_or_provider_route() {
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
    assert_eq!(
        object.properties[crate::DLARRAY_DATA_PROPERTY].residency,
        ResidencyFact::Unknown
    );
    assert_eq!(
        object.properties[crate::DLARRAY_DATA_PROPERTY].shape,
        ShapeFact::Unknown
    );
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

#[test]
fn forward_integer_forms_distinguish_predictors_from_learnables() {
    assert_eq!(FORWARD_CATALOG_ENTRY.integer_capabilities.len(), 2);
    assert_eq!(
        FORWARD_CATALOG_ENTRY.integer_capabilities[0].output_class,
        BuiltinIntegerOutputClassRule::Double
    );
    assert_eq!(
        FORWARD_CATALOG_ENTRY.integer_capabilities[1].output_class,
        BuiltinIntegerOutputClassRule::FunctionSpecific
    );
    assert_eq!(
        FORWARD_CATALOG_ENTRY.integer_capabilities[1].backend,
        BuiltinIntegerBackendRule::FunctionSpecific
    );
}

#[test]
fn forward_diagnoses_known_sparse_and_unsupported_resident_dlarray_classes() {
    let mut sparse = numeric(NumericClass::Single);
    sparse.storage = StorageFact::Sparse;
    assert!(infer(vec![network(), sparse])
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-FORWARD-STORAGE"));

    for kind in [
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Logical,
    ] {
        let mut data = ValueFact::proven(
            kind,
            ShapeFact::from(vec![Some(3), Some(4)]),
            StorageFact::Dense,
        );
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
        assert!(infer(vec![network(), wrapper])
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-DL-FORWARD-RESIDENT-CLASS"));
    }
}

#[test]
fn raw_sequence_axes_and_temporal_pooling_preserve_channel_batch_order() {
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
    let input = shaped_single(vec![Some(4), Some(3), Some(8)]); // C B T
    let net = network_with_layers(vec![
        sequence,
        layer(K::GlobalAveragePooling1D, BTreeMap::new()),
    ]);
    let inferred = infer(vec![net, input]);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(3)])
    );
}

#[test]
fn spatial_convolution_forgets_unknown_extent_then_pool_sets_spatial_singleton() {
    use crate::DeepLearningLayerKind as K;
    let spatial_size = shaped_single(vec![Some(1), Some(2)]);
    let sequence = layer(
        K::SequenceInput,
        BTreeMap::from([("InputSize".into(), spatial_size)]),
    );
    let weights = shaped_single(vec![Some(3), Some(4), Some(6)]);
    let convolution = layer(
        K::Convolution1D,
        BTreeMap::from([("Weights".into(), weights)]),
    );
    let input = shaped_single(vec![Some(9), Some(4), Some(2), Some(7)]); // S C B T
    let net = network_with_layers(vec![sequence, convolution.clone()]);
    assert_eq!(
        infer(vec![net, input.clone()]).outputs[0].shape,
        ShapeFact::from(vec![None, Some(6), Some(2), Some(7)])
    );
    let pooled = network_with_layers(vec![
        layer(
            K::SequenceInput,
            BTreeMap::from([("InputSize".into(), shaped_single(vec![Some(1), Some(2)]))]),
        ),
        convolution,
        layer(K::GlobalAveragePooling1D, BTreeMap::new()),
    ]);
    assert_eq!(
        infer(vec![pooled, input]).outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(6), Some(2), Some(7)])
    );
}

#[test]
fn recurrent_output_mode_and_unknown_dlarray_format_do_not_claim_shape() {
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
    let net = network_with_layers(vec![sequence, layer(K::Lstm, BTreeMap::new())]);
    assert_eq!(
        infer(vec![net, shaped_single(vec![Some(4), Some(2), Some(8)])]).outputs[0].shape,
        ShapeFact::Unknown
    );
    let raw = shaped_single(vec![Some(2), Some(3)]);
    let wrapped = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DL_ARRAY.owned()),
            properties: BTreeMap::from([
                (crate::DLARRAY_DATA_PROPERTY.into(), raw),
                ("Format".into(), ValueFact::scalar(ValueKindFact::String)),
            ]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Opaque,
    );
    let inferred = infer(vec![feature_network(), wrapped]);
    let ValueKindFact::Object(output) = &inferred.outputs[0].kind else {
        panic!("dlarray output")
    };
    assert_eq!(
        output.properties[crate::DLARRAY_DATA_PROPERTY].shape,
        ShapeFact::Unknown
    );
}

#[test]
fn second_requested_output_is_state_table_and_third_is_rejected() {
    let inferred = infer_outputs(vec![feature_network(), numeric(NumericClass::Single)], 2);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs.len(), 2);
    let ValueKindFact::Object(state) = &inferred.outputs[1].kind else {
        panic!("State table")
    };
    assert_eq!(
        state.runtime_class,
        Some(runmat_types::standard::TABLE.owned())
    );
    assert!(state.properties.contains_key("__table_variables"));
    let too_many = infer_outputs(vec![feature_network(), numeric(NumericClass::Single)], 3);
    assert!(!too_many.diagnostics.is_empty());
}

#[test]
fn known_invalid_feature_rank_is_diagnosed() {
    let invalid = infer(vec![
        feature_network(),
        shaped_single(vec![Some(2), Some(3), Some(4)]),
    ]);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-FORWARD-SHAPE"));
}

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
