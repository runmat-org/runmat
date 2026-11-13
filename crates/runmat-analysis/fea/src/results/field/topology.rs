use runmat_analysis_core::{AnalysisModel, StructuralElementKind};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::physics::structural::topology::{
    node_space::CanonicalStructuralNodeSpace, CanonicalStructuralTopologyIdentity,
};

use super::{RecoveredFieldEntityCounts, RecoveredFieldError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StructuralRecoveredFieldTopology {
    pub identity: CanonicalStructuralTopologyIdentity,
    pub entity_counts: RecoveredFieldEntityCounts,
    /// Exact number of solid element-to-node incidences in the admitted topology.
    pub solid_node_incidence_count: usize,
}

impl StructuralRecoveredFieldTopology {
    pub fn derive(
        model: &AnalysisModel,
        mesh: Option<&SolverMeshArtifact>,
    ) -> Result<Self, RecoveredFieldError> {
        if let Some(mesh) = mesh {
            mesh.validate_canonical()
                .map_err(|error| RecoveredFieldError::InvalidArtifact(error.to_string()))?;
        }
        let structural = model.structural.as_ref();
        let has_explicit = structural.is_some_and(|structural| !structural.elements.is_empty());
        let (identity, nodes) = match (mesh, structural.filter(|_| has_explicit)) {
            (Some(mesh), Some(structural))
                if !mesh.topology.volume_elements.is_empty()
                    || !mesh.resolved_request.dimension.is_solid() =>
            {
                let topology = CanonicalStructuralNodeSpace::derive(mesh, structural)
                    .map_err(|error| RecoveredFieldError::InvalidArtifact(error.to_string()))?;
                (topology.identity, topology.nodes.len())
            }
            (None, Some(structural)) => {
                let topology = CanonicalStructuralNodeSpace::derive_explicit(structural)
                    .map_err(|error| RecoveredFieldError::InvalidArtifact(error.to_string()))?;
                (topology.identity, topology.nodes.len())
            }
            (Some(mesh), _) => (
                CanonicalStructuralTopologyIdentity::Solid {
                    mesh_digest: mesh.canonical_digest,
                },
                mesh.topology.nodes.len(),
            ),
            (None, None) => {
                return Err(RecoveredFieldError::InvalidArtifact(
                    "structural field topology requires solid or explicit elements".into(),
                ))
            }
        };
        let (beam_elements, shell_elements) = structural
            .map(|structural| {
                structural
                    .elements
                    .iter()
                    .fold((0, 0), |counts, element| match element.kind {
                        StructuralElementKind::Beam(_) => (counts.0 + 1, counts.1),
                        StructuralElementKind::Shell(_) => (counts.0, counts.1 + 1),
                    })
            })
            .unwrap_or_default();
        let entity_counts = RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes,
            volume_elements: mesh
                .map(|mesh| mesh.topology.volume_elements.len())
                .unwrap_or(0),
            volume_integration_points: mesh
                .map(|mesh| mesh.topology.volume_elements.len().saturating_mul(4))
                .unwrap_or(0),
            oriented_edges: 0,
            beam_elements,
            shell_elements,
            connectors: model.connectors.len(),
            constraint_groups: 0,
        };
        entity_counts.validate()?;
        let solid_node_incidence_count = mesh
            .map(|mesh| {
                mesh.topology
                    .volume_elements
                    .iter()
                    .map(|element| element.node_ids.len())
                    .sum()
            })
            .unwrap_or(0);
        Ok(Self {
            identity,
            entity_counts,
            solid_node_incidence_count,
        })
    }
}
