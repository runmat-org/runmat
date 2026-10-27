//! Persistent attribution for selected native observation entities.
use crate::results::field::RecoveredFieldEntityDomain;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationEntity {
    Node(StableDigest),
    Element(StableDigest),
    IntegrationPoint(StableDigest),
    OrientedEdge(StableDigest),
    OrientedFace(StableDigest),
    Connector(StableDigest),
    Point {
        identity: StableDigest,
        #[serde(with = "runmat_types::wire_integer")]
        ordinal: u64,
        position_m: [f64; 3],
        source_element: StableDigest,
    },
}

pub(super) const fn observation_element_entity(
    domain: RecoveredFieldEntityDomain,
    identity: StableDigest,
) -> ObservationEntity {
    match domain {
        RecoveredFieldEntityDomain::VolumeIntegrationPoints => {
            ObservationEntity::IntegrationPoint(identity)
        }
        RecoveredFieldEntityDomain::OrientedEdges => ObservationEntity::OrientedEdge(identity),
        RecoveredFieldEntityDomain::OrientedFaces => ObservationEntity::OrientedFace(identity),
        _ => ObservationEntity::Element(identity),
    }
}
