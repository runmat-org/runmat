use crate::{
    DeepLearningAxisLabel as Axis, DeepLearningAxisLayout, DeepLearningInputKind,
    DeepLearningLayerKind,
};
use runmat_types::{ObjectFact, ShapeFact, ValueFact};

use super::network::known_channel_extent;

fn input_kind(layer: &ObjectFact, kind: DeepLearningLayerKind) -> Option<DeepLearningInputKind> {
    match kind {
        DeepLearningLayerKind::FeatureInput => Some(DeepLearningInputKind::Feature),
        DeepLearningLayerKind::SequenceInput => {
            // The size payload itself is not retained by ValueFact. Its
            // structural cardinality still distinguishes scalar vector
            // sequence input from a two-element spatial-channel size.
            let size = layer.properties.get("InputSize")?;
            if size.shape.element_count() == Some(2) {
                Some(DeepLearningInputKind::SpatialSequence1d)
            } else if size.shape.element_count() == Some(1) {
                Some(DeepLearningInputKind::VectorSequence)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn invalid_raw_geometry(
    input: &ValueFact,
    layers: Option<&[(DeepLearningLayerKind, &ObjectFact)]>,
) -> bool {
    let Some((first_kind, first_layer)) = layers.and_then(|layers| layers.first()).copied() else {
        return false;
    };
    let (Some(kind), Some(rank)) = (input_kind(first_layer, first_kind), input.shape.rank()) else {
        return false;
    };
    DeepLearningAxisLayout::for_input("", kind, rank).is_err()
}

/// A formatted dlarray's `Format` is currently only a String kind fact: its
/// literal labels do not survive into ObjectFact. Do not infer BC from shape.
pub(super) fn output_shape(
    input: &ValueFact,
    formatted: bool,
    layers: Option<&[(DeepLearningLayerKind, &ObjectFact)]>,
) -> ShapeFact {
    if formatted {
        return ShapeFact::Unknown;
    }
    let Some(layers) = layers else {
        return ShapeFact::Unknown;
    };
    let Some((first_kind, first_layer)) = layers.first().copied() else {
        return ShapeFact::Unknown;
    };
    let Some(kind) = input_kind(first_layer, first_kind) else {
        return ShapeFact::Unknown;
    };
    let Some(mut dims) = input.shape.known_dims() else {
        return ShapeFact::Unknown;
    };
    let Ok(mut axes) = DeepLearningAxisLayout::for_input("", kind, dims.len()) else {
        return ShapeFact::Unknown;
    };
    let rank_one_feature = kind == DeepLearningInputKind::Feature && dims.len() == 1;
    for (layer_kind, layer) in layers.iter().skip(1) {
        match layer_kind {
            DeepLearningLayerKind::FullyConnected | DeepLearningLayerKind::Convolution1D => {
                let Some(channel_axis) = axes.index(Axis::Channel) else {
                    return ShapeFact::Unknown;
                };
                dims[channel_axis] = known_channel_extent(*layer_kind, layer);
                if *layer_kind == DeepLearningLayerKind::Convolution1D {
                    let Some(spatial_axis) = axes.index(Axis::Spatial) else {
                        return ShapeFact::Unknown;
                    };
                    // Filter/stride/padding literals are not retained by the
                    // current object fact. Preserve rank, not spatial extent.
                    dims[spatial_axis] = None;
                }
            }
            DeepLearningLayerKind::GlobalAveragePooling1D => {
                if let Some(spatial_axis) = axes.index(Axis::Spatial) {
                    dims[spatial_axis] = Some(1);
                } else if let Some(time_axis) = axes.index(Axis::Time) {
                    if axes.index(Axis::Batch).is_some() {
                        dims.remove(time_axis);
                        let mut labels = axes.axes().to_vec();
                        labels.remove(time_axis);
                        let Ok(reduced) = DeepLearningAxisLayout::from_axes(&labels) else {
                            return ShapeFact::Unknown;
                        };
                        axes = reduced;
                    } else {
                        dims[time_axis] = Some(1);
                        let mut labels = axes.axes().to_vec();
                        labels[time_axis] = Axis::Unspecified;
                        let Ok(reduced) = DeepLearningAxisLayout::from_axes(&labels) else {
                            return ShapeFact::Unknown;
                        };
                        axes = reduced;
                    }
                } else {
                    return ShapeFact::Unknown;
                }
            }
            DeepLearningLayerKind::Lstm | DeepLearningLayerKind::Bilstm => {
                // OutputMode can remove time, and its literal value is not
                // represented in ObjectFact. Neither rank nor width is proven.
                return ShapeFact::Unknown;
            }
            DeepLearningLayerKind::FeatureInput | DeepLearningLayerKind::SequenceInput => {
                return ShapeFact::Unknown;
            }
            DeepLearningLayerKind::LayerNormalization
            | DeepLearningLayerKind::Relu
            | DeepLearningLayerKind::Elu
            | DeepLearningLayerKind::Softmax
            | DeepLearningLayerKind::ClassificationOutput
            | DeepLearningLayerKind::RegressionOutput => {}
        }
    }
    if rank_one_feature {
        ShapeFact::from(vec![Some(1), dims[0]])
    } else {
        ShapeFact::from(dims)
    }
}
