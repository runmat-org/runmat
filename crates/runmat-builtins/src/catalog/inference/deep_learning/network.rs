use crate::DeepLearningLayerKind;
use runmat_types::{ObjectFact, ValueFact, ValueKindFact};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ComplexLayerSupport {
    HostOnly,
    Unsupported,
    Unknown,
}

pub(super) fn complex_layer_support(
    layers: Option<&[(DeepLearningLayerKind, &ObjectFact)]>,
) -> ComplexLayerSupport {
    let Some(layers) = layers.filter(|layers| !layers.is_empty()) else {
        return ComplexLayerSupport::Unknown;
    };
    if layers.iter().all(|(kind, _)| {
        kind.is_input()
            || *kind == DeepLearningLayerKind::GlobalAveragePooling1D
            || kind.is_output()
    }) {
        ComplexLayerSupport::HostOnly
    } else {
        ComplexLayerSupport::Unsupported
    }
}

/// Only a complete, ordered layer collection can support dependent geometry.
/// An unknown class is not silently treated as a channel-preserving layer.
pub(super) fn known_layers(
    network: Option<&ValueFact>,
) -> Option<Vec<(DeepLearningLayerKind, &ObjectFact)>> {
    let ValueKindFact::Object(network) = &network?.kind else {
        return None;
    };
    let layers_fact = network.properties.get("Layers")?;
    let ValueKindFact::Cell(layers) = &layers_fact.kind else {
        return None;
    };
    if !layers.elements_complete
        || layers_fact
            .shape
            .element_count()
            .is_some_and(|count| count != layers.elements.len())
    {
        return None;
    }
    layers
        .elements
        .iter()
        .map(|value| {
            let ValueKindFact::Object(layer) = &value.kind else {
                return None;
            };
            let kind = DeepLearningLayerKind::from_identity(layer.runtime_class.as_ref()?)?;
            Some((kind, layer))
        })
        .collect()
}

pub(super) fn known_channel_extent(
    kind: DeepLearningLayerKind,
    layer: &ObjectFact,
) -> Option<usize> {
    if !matches!(
        kind,
        DeepLearningLayerKind::FullyConnected | DeepLearningLayerKind::Convolution1D
    ) {
        return None;
    }
    let dims = layer.properties.get("Weights")?.shape.known_dims()?;
    match kind {
        DeepLearningLayerKind::FullyConnected => dims.first().copied().flatten(),
        DeepLearningLayerKind::Convolution1D => dims.get(2).copied().flatten(),
        _ => None,
    }
}
