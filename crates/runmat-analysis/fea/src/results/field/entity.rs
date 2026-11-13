//! Recovered-field entity domains and complete storage cardinalities.
use super::RecoveredFieldError;
use runmat_analysis_core::FieldLocation;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveredFieldEntityDomain {
    Nodes,
    VolumeElements,
    VolumeIntegrationPoints,
    OrientedEdges,
    /// All canonical owner-oriented volume faces, including internal faces.
    OrientedFaces,
    BeamElements,
    ShellElements,
    Connectors,
    ConstraintGroups,
}

impl RecoveredFieldEntityDomain {
    pub(super) const fn accepts_location(self, location: FieldLocation) -> bool {
        match self {
            Self::Nodes => matches!(location, FieldLocation::Node),
            Self::OrientedEdges => matches!(location, FieldLocation::Edge),
            Self::OrientedFaces => matches!(location, FieldLocation::Face),
            Self::VolumeElements => {
                matches!(location, FieldLocation::Element | FieldLocation::Cell)
            }
            Self::BeamElements | Self::ShellElements => {
                matches!(location, FieldLocation::Element)
            }
            Self::VolumeIntegrationPoints => matches!(location, FieldLocation::IntegrationPoint),
            Self::Connectors => matches!(location, FieldLocation::Connector),
            Self::ConstraintGroups => matches!(location, FieldLocation::Constraint),
        }
    }
}

/// What a field support is resolved against: the mesh projection and, when the topology has
/// them, the owner and neighbor projection elements of its owner-oriented faces.
#[derive(Clone, Copy)]
pub struct SupportTopology<'a> {
    pub projection: &'a crate::topology::AnalysisProjection,
    pub face_elements: Option<&'a [(usize, Option<usize>)]>,
}

/// The entities of its domain a field carries values for.
///
/// `Complete` covers the whole domain. `Regions` covers the volume elements of the listed mesh
/// regions, the nodes of those elements, their integration points and the faces they bound: for
/// example the fluid cells of a conjugate model. `entities` is the resolved count, checked
/// against the topology wherever the field meets it; values are stored in ascending global
/// entity order.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveredFieldSupport {
    #[default]
    Complete,
    Regions {
        regions: Vec<runmat_meshing_core::PersistentEntityId>,
        entities: usize,
    },
}

impl RecoveredFieldSupport {
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }

    pub fn regions(
        mut regions: Vec<runmat_meshing_core::PersistentEntityId>,
        entities: usize,
    ) -> Result<Self, RecoveredFieldError> {
        regions.sort();
        regions.dedup();
        if regions.is_empty() || entities == 0 {
            return Err(RecoveredFieldError::InvalidArtifact(
                "a region-supported field needs regions and entities".into(),
            ));
        }
        Ok(Self::Regions { regions, entities })
    }

    pub(super) fn validate(
        &self,
        domain: RecoveredFieldEntityDomain,
    ) -> Result<(), RecoveredFieldError> {
        match self {
            Self::Complete => Ok(()),
            Self::Regions { regions, entities } => {
                if regions.is_empty()
                    || *entities == 0
                    || regions.windows(2).any(|pair| pair[0] >= pair[1])
                    || !matches!(
                        domain,
                        RecoveredFieldEntityDomain::Nodes
                            | RecoveredFieldEntityDomain::VolumeElements
                            | RecoveredFieldEntityDomain::VolumeIntegrationPoints
                            | RecoveredFieldEntityDomain::OrientedFaces
                    )
                {
                    return Err(RecoveredFieldError::InvalidArtifact(
                        "field support regions are empty, unsorted or outside a volume domain"
                            .into(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Stored entity count: the complete domain or the resolved support.
    pub fn stored_entities(&self, complete: usize) -> usize {
        match self {
            Self::Complete => complete,
            Self::Regions { entities, .. } => *entities,
        }
    }

    /// Ascending global entity indices of the support on a topology (`None` when complete).
    pub fn resolve(
        &self,
        domain: RecoveredFieldEntityDomain,
        topology: SupportTopology<'_>,
    ) -> Result<Option<Vec<usize>>, RecoveredFieldError> {
        let projection = topology.projection;
        let Self::Regions { regions, entities } = self else {
            return Ok(None);
        };
        let inside = |element: &crate::topology::ProjectedElement| {
            regions.binary_search(&element.region_id).is_ok()
        };
        let indices: Vec<usize> = match domain {
            RecoveredFieldEntityDomain::VolumeElements => projection
                .elements
                .iter()
                .enumerate()
                .filter(|(_, element)| inside(element))
                .map(|(index, _)| index)
                .collect(),
            RecoveredFieldEntityDomain::Nodes => {
                let mut covered = vec![false; projection.nodes.len()];
                for element in projection.elements.iter().filter(|e| inside(e)) {
                    for &node in &element.node_indices {
                        covered[node] = true;
                    }
                }
                covered
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c)
                    .map(|(index, _)| index)
                    .collect()
            }
            RecoveredFieldEntityDomain::VolumeIntegrationPoints => {
                const POINTS: usize =
                    crate::topology::quadrature::TETRAHEDRON_FOUR_POINT_RULE.len();
                projection
                    .elements
                    .iter()
                    .enumerate()
                    .filter(|(_, element)| inside(element))
                    .flat_map(|(index, _)| (0..POINTS).map(move |point| index * POINTS + point))
                    .collect()
            }
            RecoveredFieldEntityDomain::OrientedFaces => {
                let faces = topology.face_elements.ok_or_else(|| {
                    RecoveredFieldError::InvalidArtifact(
                        "face support needs the topology's oriented faces".into(),
                    )
                })?;
                let inside_index = |index: usize| inside(&projection.elements[index]);
                faces
                    .iter()
                    .enumerate()
                    .filter(|(_, (owner, neighbor))| {
                        inside_index(*owner) || neighbor.is_some_and(inside_index)
                    })
                    .map(|(index, _)| index)
                    .collect()
            }
            _ => {
                return Err(RecoveredFieldError::InvalidArtifact(
                    "field support resolves only volume and face domains".into(),
                ))
            }
        };
        if indices.len() != *entities {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field support count differs from its regions on this topology".into(),
            ));
        }
        Ok(Some(indices))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredFieldEntityCounts {
    pub nodes: usize,
    pub volume_elements: usize,
    pub volume_integration_points: usize,
    pub oriented_edges: usize,
    pub oriented_faces: usize,
    pub beam_elements: usize,
    pub shell_elements: usize,
    pub connectors: usize,
    pub constraint_groups: usize,
}

impl RecoveredFieldEntityCounts {
    pub const fn entity_count(self, domain: RecoveredFieldEntityDomain) -> usize {
        match domain {
            RecoveredFieldEntityDomain::Nodes => self.nodes,
            RecoveredFieldEntityDomain::VolumeElements => self.volume_elements,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints => self.volume_integration_points,
            RecoveredFieldEntityDomain::OrientedEdges => self.oriented_edges,
            RecoveredFieldEntityDomain::OrientedFaces => self.oriented_faces,
            RecoveredFieldEntityDomain::BeamElements => self.beam_elements,
            RecoveredFieldEntityDomain::ShellElements => self.shell_elements,
            RecoveredFieldEntityDomain::Connectors => self.connectors,
            RecoveredFieldEntityDomain::ConstraintGroups => self.constraint_groups,
        }
    }

    pub(super) fn validate(self) -> Result<(), RecoveredFieldError> {
        if self.nodes == 0
            || self
                .volume_elements
                .checked_add(self.beam_elements)
                .and_then(|count| count.checked_add(self.shell_elements))
                .is_none_or(|count| count == 0)
        {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field topology requires nodes and at least one element".into(),
            ));
        }
        Ok(())
    }
}
