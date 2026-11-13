//! Shared static facts for sequential network evaluation. Runtime execution
//! and each public builtin's arity/options remain with their owning entry.

mod caller;
mod input;
mod network;
mod output;
mod shape;

pub(in crate::catalog) use caller::NetworkInferenceCaller;
pub(in crate::catalog) use output::infer_network_output;

use runmat_types::{standard, ValueFact, ValueKindFact};

pub(in crate::catalog) fn network_possible(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => object.runtime_class.as_ref().is_none_or(|class| {
            [
                standard::DL_NETWORK,
                standard::SERIES_NETWORK,
                standard::DAG_NETWORK,
            ]
            .iter()
            .any(|expected| class.is(*expected))
        }),
        _ => false,
    }
}

pub(in crate::catalog) fn known_state_rows(network: Option<&ValueFact>) -> Option<usize> {
    let layers = network::known_layers(network)?;
    layers
        .iter()
        .all(|(kind, _)| {
            !matches!(
                kind,
                crate::DeepLearningLayerKind::Lstm | crate::DeepLearningLayerKind::Bilstm
            )
        })
        .then_some(0)
}
