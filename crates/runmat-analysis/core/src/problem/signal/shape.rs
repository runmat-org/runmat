//! Shared signal-shape cardinality and concrete topology resolution.
use serde::{Deserialize, Serialize};

mod physical_signal;

/// Authored entity domains whose ordering can be supplied by one canonical solver mesh.
/// Quadrature and other physics-owned domains require separate explicit contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredSignalFieldLocation {
    Node,
    Cell,
    Element,
    Face,
    Edge,
}

impl From<AuthoredSignalFieldLocation> for crate::FieldLocation {
    fn from(location: AuthoredSignalFieldLocation) -> Self {
        match location {
            AuthoredSignalFieldLocation::Node => Self::Node,
            AuthoredSignalFieldLocation::Cell => Self::Cell,
            AuthoredSignalFieldLocation::Element => Self::Element,
            AuthoredSignalFieldLocation::Face => Self::Face,
            AuthoredSignalFieldLocation::Edge => Self::Edge,
        }
    }
}

/// Unresolved ownership, not a topology digest or permission to infer entity ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AuthoredSignalTopologySelector {
    StudySolverMesh {
        location: AuthoredSignalFieldLocation,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SignalShape {
    Scalar,
    Vector3,
    Tensor(#[serde(with = "runmat_types::wire_integer::vec")] Vec<usize>),
    TopologyField {
        topology_id: String,
        location: crate::FieldLocation,
        #[serde(with = "runmat_types::wire_integer")]
        component_count: usize,
    },
    /// An authored generator declaration. It has no numerical cardinality until
    /// an authenticated resolver creates a distinct mesh-bound `TopologyField`.
    DeferredTopologyField {
        selector: AuthoredSignalTopologySelector,
        #[serde(with = "runmat_types::wire_integer")]
        component_count: usize,
    },
}

impl SignalShape {
    pub fn component_extents(&self) -> Option<Vec<usize>> {
        match self {
            Self::Scalar => Some(Vec::new()),
            Self::Vector3 => Some(vec![3]),
            Self::Tensor(shape) if !shape.is_empty() => Some(shape.clone()),
            Self::TopologyField {
                topology_id,
                component_count,
                ..
            } if !topology_id.trim().is_empty() && *component_count > 0 => {
                Some(vec![*component_count])
            }
            Self::DeferredTopologyField {
                component_count, ..
            } if *component_count > 0 => Some(vec![*component_count]),
            _ => None,
        }
    }

    /// Number of canonical values carried by one sample after topology resolution.
    pub fn realized_component_count(&self, topology_entity_count: Option<usize>) -> Option<usize> {
        match self {
            Self::Scalar if topology_entity_count.is_none() => Some(1),
            Self::Vector3 if topology_entity_count.is_none() => Some(3),
            Self::Tensor(extents)
                if topology_entity_count.is_none()
                    && !extents.is_empty()
                    && extents.iter().all(|extent| *extent > 0) =>
            {
                extents
                    .iter()
                    .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
            }
            Self::TopologyField {
                topology_id,
                component_count,
                ..
            } if !topology_id.trim().is_empty() && *component_count > 0 => topology_entity_count
                .filter(|entities| *entities > 0)
                .and_then(|entities| entities.checked_mul(*component_count)),
            _ => None,
        }
    }

    /// Declared physical width of a field, independent of entity resolution.
    pub fn topology_component_count(&self) -> Option<usize> {
        match self {
            Self::TopologyField {
                component_count, ..
            }
            | Self::DeferredTopologyField {
                component_count, ..
            } => Some(*component_count),
            _ => None,
        }
    }
}
