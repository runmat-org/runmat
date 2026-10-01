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
            Self::VolumeElements | Self::BeamElements | Self::ShellElements => {
                matches!(location, FieldLocation::Element)
            }
            Self::VolumeIntegrationPoints => matches!(location, FieldLocation::IntegrationPoint),
            Self::Connectors => matches!(location, FieldLocation::Connector),
            Self::ConstraintGroups => matches!(location, FieldLocation::Constraint),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredFieldEntityCounts {
    pub nodes: usize,
    pub volume_elements: usize,
    pub volume_integration_points: usize,
    pub oriented_edges: usize,
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
