use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, StructuralElementKind};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use crate::{
    physics::structural::topology::{
        explicit_element_identity, node_space::CanonicalStructuralNodeSpace,
        CanonicalStructuralTopologyIdentity,
    },
    results::field::RecoveredFieldEntityDomain,
    topology::AnalysisProjection,
};

use super::ObservationError;

#[derive(Clone, Debug)]
pub(crate) struct StructuralObservationNode {
    pub identity: StableDigest,
    pub coordinates_m: [f64; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct StructuralObservationElement {
    pub identity: StableDigest,
    pub region_id: String,
    pub centroid_m: [f64; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct StructuralObservationTopology {
    pub identity: CanonicalStructuralTopologyIdentity,
    topology_digest: Digest,
    pub nodes: Vec<StructuralObservationNode>,
    pub solid_elements: Vec<StructuralObservationElement>,
    pub beam_elements: Vec<StructuralObservationElement>,
    pub shell_elements: Vec<StructuralObservationElement>,
    pub solid_projection: Option<AnalysisProjection>,
    node_selectors: BTreeMap<u64, Vec<usize>>,
}

impl StructuralObservationTopology {
    pub fn from_projection(projection: &AnalysisProjection) -> Self {
        let node_selectors = projection
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.node_id, vec![index]))
            .collect();
        Self {
            identity: CanonicalStructuralTopologyIdentity::Solid {
                mesh_digest: projection.mesh_digest,
            },
            topology_digest: projection.projection_digest,
            nodes: projection
                .nodes
                .iter()
                .map(|node| StructuralObservationNode {
                    identity: node.stable_identity,
                    coordinates_m: node.coordinates_m,
                })
                .collect(),
            solid_elements: projection
                .elements
                .iter()
                .map(|element| StructuralObservationElement {
                    identity: element.stable_identity,
                    region_id: String::new(),
                    centroid_m: element.centroid_m,
                })
                .collect(),
            beam_elements: Vec::new(),
            shell_elements: Vec::new(),
            solid_projection: Some(projection.clone()),
            node_selectors,
        }
    }

    pub fn derive(
        model: &AnalysisModel,
        mesh: Option<&SolverMeshArtifact>,
    ) -> Result<Self, ObservationError> {
        let structural = model.structural.as_ref();
        let has_explicit = structural.is_some_and(|model| !model.elements.is_empty());
        let projection = mesh
            .map(AnalysisProjection::derive)
            .transpose()
            .map_err(|error| ObservationError::Invalid(error.to_string()))?;
        if !has_explicit {
            let mut topology = projection.as_ref().map(Self::from_projection).ok_or(
                ObservationError::Unsupported(
                    "structural observations require solid or explicit topology",
                ),
            )?;
            topology.topology_digest = topology.identity.discretization_digest();
            return Ok(topology);
        }
        let Some(structural) = structural else {
            return Err(ObservationError::Invalid(
                "explicit topology disappeared during observation derivation".into(),
            ));
        };
        let node_space = match mesh {
            Some(mesh) => CanonicalStructuralNodeSpace::derive(mesh, structural),
            None => CanonicalStructuralNodeSpace::derive_explicit(structural),
        }
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
        let explicit_digest = match node_space.identity {
            CanonicalStructuralTopologyIdentity::Explicit {
                structural_topology_digest,
            }
            | CanonicalStructuralTopologyIdentity::ConformalMixed {
                structural_topology_digest,
                ..
            } => structural_topology_digest,
            CanonicalStructuralTopologyIdentity::Solid { .. } => {
                return Err(ObservationError::Identity(
                    "explicit node space produced a solid-only identity",
                ))
            }
        };
        let mut node_selectors = BTreeMap::<u64, Vec<usize>>::new();
        if let Some(projection) = &projection {
            for (index, node) in projection.nodes.iter().enumerate() {
                node_selectors.entry(node.node_id).or_default().push(index);
            }
        }
        for (explicit_index, node) in structural.nodes.iter().enumerate() {
            let canonical_index = node_space.explicit_node_indices[explicit_index];
            node_selectors
                .entry(u64::from(node.node_id))
                .or_default()
                .push(canonical_index);
        }
        for indices in node_selectors.values_mut() {
            indices.sort_unstable();
            indices.dedup();
        }
        let mut topology = Self {
            identity: node_space.identity,
            topology_digest: node_space.identity.discretization_digest(),
            nodes: node_space
                .nodes
                .iter()
                .map(|node| StructuralObservationNode {
                    identity: node.identity,
                    coordinates_m: node.coordinates_m,
                })
                .collect(),
            solid_elements: projection
                .as_ref()
                .map(|projection| {
                    projection
                        .elements
                        .iter()
                        .map(|element| StructuralObservationElement {
                            identity: element.stable_identity,
                            region_id: String::new(),
                            centroid_m: element.centroid_m,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            beam_elements: Vec::new(),
            shell_elements: Vec::new(),
            solid_projection: projection,
            node_selectors,
        };
        for element in &structural.elements {
            let explicit_nodes = match &element.kind {
                StructuralElementKind::Beam(beam) => beam.node_ids.as_slice(),
                StructuralElementKind::Shell(shell) => shell.node_ids.as_slice(),
            };
            let node_indices = explicit_nodes
                .iter()
                .map(|node_id| {
                    structural
                        .nodes
                        .iter()
                        .position(|node| node.node_id == *node_id)
                        .and_then(|index| node_space.explicit_node_indices.get(index).copied())
                        .ok_or_else(|| {
                            ObservationError::Invalid(format!(
                                "structural element `{}` references absent node {}",
                                element.element_id, node_id
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let centroid_m = centroid(&node_indices, &topology.nodes)?;
            let projected = StructuralObservationElement {
                identity: explicit_element_identity(
                    explicit_digest,
                    &element.element_id,
                    &element.region_id,
                    &element.kind,
                )
                .map_err(|error| ObservationError::Invalid(error.to_string()))?,
                region_id: element.region_id.clone(),
                centroid_m,
            };
            match element.kind {
                StructuralElementKind::Beam(_) => topology.beam_elements.push(projected),
                StructuralElementKind::Shell(_) => topology.shell_elements.push(projected),
            }
        }
        Ok(topology)
    }

    pub fn digest(&self) -> Digest {
        self.topology_digest
    }

    pub fn node_index(&self, node_id: u64) -> Result<Option<usize>, ObservationError> {
        match self.node_selectors.get(&node_id).map(Vec::as_slice) {
            None => Ok(None),
            Some([index]) => Ok(Some(*index)),
            Some(_) => Err(ObservationError::Invalid(format!(
                "node selector `node:{node_id}` is ambiguous across structural topology"
            ))),
        }
    }

    pub fn elements(
        &self,
        domain: RecoveredFieldEntityDomain,
    ) -> Result<&[StructuralObservationElement], ObservationError> {
        match domain {
            RecoveredFieldEntityDomain::SolidElements => Ok(&self.solid_elements),
            RecoveredFieldEntityDomain::BeamElements => Ok(&self.beam_elements),
            RecoveredFieldEntityDomain::ShellElements => Ok(&self.shell_elements),
            RecoveredFieldEntityDomain::StructuralNodes => Err(ObservationError::Invalid(
                "nodal field domain cannot resolve element topology".into(),
            )),
        }
    }
}

fn centroid(
    indices: &[usize],
    nodes: &[StructuralObservationNode],
) -> Result<[f64; 3], ObservationError> {
    let mut sum = [0.0; 3];
    for index in indices {
        let node = nodes.get(*index).ok_or(ObservationError::Identity(
            "structural element exceeds its canonical node space",
        ))?;
        for (total, coordinate) in sum.iter_mut().zip(node.coordinates_m) {
            *total += coordinate;
        }
    }
    let count = indices.len() as f64;
    if count == 0.0 {
        return Err(ObservationError::Invalid(
            "structural observation element has no nodes".into(),
        ));
    }
    Ok(sum.map(|value| value / count))
}
