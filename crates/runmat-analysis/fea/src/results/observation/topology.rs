use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, StructuralElementKind};
use runmat_execution::Digest;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact, StableDigest};

use crate::{
    physics::structural::topology::{
        explicit_element_identity, node_space::CanonicalStructuralNodeSpace,
        CanonicalStructuralTopologyIdentity,
    },
    results::field::{RecoveredFieldEntityCounts, RecoveredFieldEntityDomain},
    topology::{quadrature::TETRAHEDRON_FOUR_POINT_RULE, AnalysisProjection},
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
    pub solid_integration_points: Vec<StructuralObservationElement>,
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
        let solid_integration_points = integration_point_elements(projection);
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
            solid_integration_points,
            beam_elements: Vec::new(),
            shell_elements: Vec::new(),
            solid_projection: Some(projection.clone()),
            node_selectors,
        }
    }

    pub(crate) fn entity_counts(&self) -> RecoveredFieldEntityCounts {
        RecoveredFieldEntityCounts {
            structural_nodes: self.nodes.len(),
            solid_elements: self.solid_elements.len(),
            solid_integration_points: self.solid_integration_points.len(),
            beam_elements: self.beam_elements.len(),
            shell_elements: self.shell_elements.len(),
            connectors: 0,
            constraint_groups: 0,
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
            solid_integration_points: projection
                .as_ref()
                .map(integration_point_elements)
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
            RecoveredFieldEntityDomain::SolidIntegrationPoints => {
                Ok(&self.solid_integration_points)
            }
            RecoveredFieldEntityDomain::BeamElements => Ok(&self.beam_elements),
            RecoveredFieldEntityDomain::ShellElements => Ok(&self.shell_elements),
            RecoveredFieldEntityDomain::Connectors => Err(ObservationError::Unsupported(
                "connectors are resolved by the canonical connector plan",
            )),
            RecoveredFieldEntityDomain::ConstraintGroups => Err(ObservationError::Unsupported(
                "constraint groups are resolved by the canonical constraint plan",
            )),
            RecoveredFieldEntityDomain::StructuralNodes => Err(ObservationError::Invalid(
                "nodal field domain cannot resolve element topology".into(),
            )),
        }
    }
}

fn integration_point_elements(
    projection: &AnalysisProjection,
) -> Vec<StructuralObservationElement> {
    projection
        .elements
        .iter()
        .flat_map(|element| {
            let coordinates = element
                .node_indices
                .iter()
                .map(|index| projection.nodes[*index].coordinates_m)
                .collect::<Vec<_>>();
            TETRAHEDRON_FOUR_POINT_RULE.into_iter().enumerate().map(
                move |(ordinal, barycentric)| {
                    let shape = recovery_shape_values(element.order, barycentric);
                    let position_m = std::array::from_fn(|axis| {
                        shape
                            .iter()
                            .zip(&coordinates)
                            .map(|(shape, node)| shape * node[axis])
                            .sum()
                    });
                    StructuralObservationElement {
                        identity: integration_point_identity(element.stable_identity, ordinal),
                        region_id: element.region_id.source_topology_id.clone(),
                        centroid_m: position_m,
                    }
                },
            )
        })
        .collect()
}

fn recovery_shape_values(order: ElementOrder, barycentric: [f64; 4]) -> Vec<f64> {
    match order {
        ElementOrder::Tet4 => barycentric.to_vec(),
        ElementOrder::Tet10 => {
            let mut values = vec![0.0; 10];
            for node in 0..4 {
                values[node] = barycentric[node] * (2.0 * barycentric[node] - 1.0);
            }
            for (edge, [left, right]) in [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]
                .into_iter()
                .enumerate()
            {
                values[4 + edge] = 4.0 * barycentric[left] * barycentric[right];
            }
            values
        }
    }
}

fn integration_point_identity(element: StableDigest, ordinal: usize) -> StableDigest {
    let mut bytes = b"runmat.fea.structural.integration-point\0".to_vec();
    bytes.extend_from_slice(element.bytes());
    bytes.extend_from_slice(&(ordinal as u64).to_le_bytes());
    StableDigest::from_bytes(*Digest::sha256(bytes).bytes())
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
