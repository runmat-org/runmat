//! Shared signal-shape cardinality and concrete topology resolution.
use super::SignalShape;

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
}
