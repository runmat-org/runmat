use super::*;

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
