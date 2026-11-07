use crate::topology::storage::SortedValues;
use std::borrow::Cow;

use runmat_analysis_core::{AnalysisModel, StructuralElementKind};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use crate::{
    physics::structural::topology::{
        explicit_element_identity, node_space::CanonicalStructuralNodeSpace,
        CanonicalStructuralTopologyIdentity,
    },
    results::field::{
        RecoveredFieldEntityCounts, RecoveredFieldEntityDomain, SampledVolumeFieldTopology,
    },
    topology::{quadrature::TETRAHEDRON_FOUR_POINT_RULE, AnalysisProjection},
};

use super::ObservationError;

#[derive(Clone, Debug)]
pub(crate) struct ObservationNode {
    pub identity: StableDigest,
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
}

#[derive(Clone, Debug)]
pub(crate) struct ObservationElement {
    pub identity: StableDigest,
    pub region_id: String,
    pub centroid_m: [f64; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct ObservationTopology<'a> {
    pub identity: Option<CanonicalStructuralTopologyIdentity>,
    topology_digest: Digest,
    pub nodes: Vec<ObservationNode>,
    pub volume_elements: Vec<ObservationElement>,
    volume_integration_points: IntegrationPoints,
    pub oriented_edges: Vec<ObservationElement>,
    pub oriented_faces: Vec<ObservationElement>,
    pub finite_volume: Option<crate::results::field::FiniteVolumeFieldTopology>,
    pub face_boundary_sources: Vec<Option<usize>>,
    /// Owner and neighbor projection element of every oriented face.
    pub face_elements: Vec<(usize, Option<usize>)>,
    pub beam_elements: Vec<ObservationElement>,
    pub shell_elements: Vec<ObservationElement>,
    pub solid_projection: Option<Cow<'a, AnalysisProjection>>,
    pub sampling: Option<SampledVolumeFieldTopology>,
    integration_points_by_parent: ParentIntegrationPoints,
    node_selectors: SortedValues<u64, Vec<usize>>,
}

/// Integration-point entities. A sampled layout names its samples by (element, rule point), so
/// their records (derived identity, parent region, rule position) are built only when an
/// observation actually addresses integration points.
#[derive(Clone, Debug)]
enum IntegrationPoints {
    Materialized(Vec<ObservationElement>),
    Sampled {
        count: usize,
        points: std::sync::OnceLock<Vec<ObservationElement>>,
    },
}

/// Parent element -> integration-point indices: explicit lists, or a fixed number of
/// consecutive samples per parent for a sampled layout.
#[derive(Clone, Debug)]
enum ParentIntegrationPoints {
    Explicit(Vec<Vec<usize>>),
    Uniform { parents: usize, per_parent: usize },
}
impl ParentIntegrationPoints {
    fn count(&self, parent: usize) -> Option<usize> {
        match self {
            Self::Explicit(lists) => lists.get(parent).map(Vec::len),
            Self::Uniform {
                parents,
                per_parent,
            } => (parent < *parents).then_some(*per_parent),
        }
    }
    fn extend(&self, parent: usize, indices: &mut Vec<usize>) {
        match self {
            Self::Explicit(lists) => indices.extend_from_slice(&lists[parent]),
            Self::Uniform { per_parent, .. } => {
                indices.extend(parent * per_parent..(parent + 1) * per_parent)
            }
        }
    }
}

pub(super) mod resources;
mod sampled;
#[cfg(test)]
mod tests;

impl<'a> ObservationTopology<'a> {
    pub fn from_projection(projection: &'a AnalysisProjection) -> Result<Self, ObservationError> {
        Self::from_solid_projection(Cow::Borrowed(projection))
    }

    /// Fields with a cell or face location live on the mesh's owner-oriented faces too.
    pub(crate) fn needs_faces<'f>(
        definitions: impl IntoIterator<Item = &'f crate::results::field::RecoveredFieldDefinition>,
    ) -> bool {
        definitions.into_iter().any(|definition| {
            definition.entity_domain()
                == crate::results::field::RecoveredFieldEntityDomain::OrientedFaces
                || definition.location() == runmat_analysis_core::FieldLocation::Cell
        })
    }

    /// The canonical solver mesh topology, with its owner-oriented faces when `faces`.
    pub fn from_projection_for(
        projection: &'a AnalysisProjection,
        faces: bool,
    ) -> Result<Self, ObservationError> {
        let topology = Self::from_projection(projection)?;
        if faces {
            topology.with_projection_faces()
        } else {
            Ok(topology)
        }
    }

    /// Add the owner-oriented faces of the projection: every interior face once and every
    /// boundary face. Faces are derived from the projection alone, so the topology digest stays
    /// the projection digest.
    pub(crate) fn with_projection_faces(mut self) -> Result<Self, ObservationError> {
        let projection = self
            .solid_projection
            .as_deref()
            .ok_or(ObservationError::Identity(
                "mesh faces need the topology's projection",
            ))?;
        let geometry = crate::topology::FiniteVolumeTopology::derive_selected(
            projection,
            crate::topology::FiniteVolumeProjectionPolicy::covering(projection),
            crate::topology::FiniteVolumeCellSelection::AllElements,
        )
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
        // `geometry` was derived from `projection` just above.
        let layout =
            crate::results::field::FiniteVolumeFieldTopology::from_derived(projection, &geometry)
                .map_err(|error| ObservationError::Invalid(error.to_string()))?;
        self.oriented_faces = layout
            .oriented_faces()
            .iter()
            .zip(&geometry.faces)
            .map(|(face, geo)| ObservationElement {
                identity: face.identity,
                region_id: String::new(),
                centroid_m: geo.centroid_m,
            })
            .collect();
        self.face_boundary_sources = geometry
            .faces
            .iter()
            .map(|f| f.boundary_source.map(|b| b.projection_boundary_index))
            .collect();
        self.face_elements = geometry
            .faces
            .iter()
            .map(|f| {
                (
                    geometry.cells[f.owner].source_element_index,
                    f.neighbor.map(|n| geometry.cells[n].source_element_index),
                )
            })
            .collect();
        self.finite_volume = Some(layout);
        Ok(self)
    }

    /// The topology a field support resolves against, when it has a mesh projection.
    pub(crate) fn support_topology(&self) -> Option<crate::results::field::SupportTopology<'_>> {
        self.solid_projection
            .as_deref()
            .map(|projection| crate::results::field::SupportTopology {
                projection,
                face_elements: (!self.oriented_faces.is_empty())
                    .then_some(self.face_elements.as_slice()),
            })
    }

    fn from_solid_projection(
        projection: Cow<'a, AnalysisProjection>,
    ) -> Result<Self, ObservationError> {
        let node_selectors = SortedValues::derive_grouped(
            projection
                .nodes
                .iter()
                .enumerate()
                .map(|(index, node)| (node.node_id, index)),
            projection.nodes.len(),
        )
        .map_err(storage_error)?;
        let volume_integration_points = integration_point_elements(&projection)?;
        let integration_points_by_parent = structural_integration_indices(&projection)?;
        Ok(Self {
            identity: Some(CanonicalStructuralTopologyIdentity::Solid {
                mesh_digest: projection.mesh_digest,
            }),
            topology_digest: projection.projection_digest,
            nodes: projection
                .nodes
                .iter()
                .map(|node| ObservationNode {
                    identity: node.stable_identity,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(node.coordinates_m.canonical_value()),
                })
                .collect(),
            volume_elements: projection
                .elements
                .iter()
                .map(|element| ObservationElement {
                    identity: element.stable_identity,
                    region_id: String::new(),
                    centroid_m: element.centroid_m,
                })
                .collect(),
            volume_integration_points: IntegrationPoints::Materialized(volume_integration_points),
            oriented_edges: Vec::new(),
            oriented_faces: Vec::new(),
            finite_volume: None,
            face_boundary_sources: Vec::new(),
            face_elements: Vec::new(),
            beam_elements: Vec::new(),
            shell_elements: Vec::new(),
            solid_projection: Some(projection),
            sampling: None,
            integration_points_by_parent: ParentIntegrationPoints::Explicit(
                integration_points_by_parent,
            ),
            node_selectors,
        })
    }

    pub(crate) fn entity_counts(&self) -> RecoveredFieldEntityCounts {
        RecoveredFieldEntityCounts {
            oriented_faces: self.oriented_faces.len(),
            nodes: self.nodes.len(),
            volume_elements: self.volume_elements.len(),
            volume_integration_points: self.volume_integration_point_count(),
            oriented_edges: self.oriented_edges.len(),
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
            let mut topology = projection
                .map(|projection| Self::from_solid_projection(Cow::Owned(projection)))
                .transpose()?
                .ok_or(ObservationError::Unsupported(
                    "structural observations require solid or explicit topology",
                ))?;
            let identity = CanonicalStructuralTopologyIdentity::Solid {
                mesh_digest: topology
                    .solid_projection
                    .as_ref()
                    .ok_or(ObservationError::Unsupported(
                        "structural observations require solid topology",
                    ))?
                    .mesh_digest,
            };
            topology.topology_digest = identity.discretization_digest();
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
        let solid_count = projection.as_ref().map_or(0, |p| p.nodes.len());
        let count = solid_count
            .checked_add(structural.nodes.len())
            .ok_or(ObservationError::Limit("observation node aliases"))?;
        let solid = projection.iter().flat_map(|p| {
            p.nodes
                .iter()
                .enumerate()
                .map(|(i, node)| (node.node_id, i))
        });
        let explicit = structural
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (u64::from(node.node_id), node_space.explicit_node_indices[i]));
        let node_selectors =
            SortedValues::derive_grouped(solid.chain(explicit), count).map_err(storage_error)?;
        let mut topology = Self {
            identity: Some(node_space.identity),
            topology_digest: node_space.identity.discretization_digest(),
            nodes: node_space
                .nodes
                .iter()
                .map(|node| ObservationNode {
                    identity: node.identity,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(node.coordinates_m.canonical_value()),
                })
                .collect(),
            volume_elements: projection
                .as_ref()
                .map(|projection| {
                    projection
                        .elements
                        .iter()
                        .map(|element| ObservationElement {
                            identity: element.stable_identity,
                            region_id: String::new(),
                            centroid_m: element.centroid_m,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            volume_integration_points: IntegrationPoints::Materialized(
                projection
                    .as_ref()
                    .map(integration_point_elements)
                    .transpose()?
                    .unwrap_or_default(),
            ),
            beam_elements: Vec::new(),
            shell_elements: Vec::new(),
            oriented_edges: Vec::new(),
            oriented_faces: Vec::new(),
            finite_volume: None,
            face_boundary_sources: Vec::new(),
            face_elements: Vec::new(),
            sampling: None,
            integration_points_by_parent: ParentIntegrationPoints::Explicit(
                projection
                    .as_ref()
                    .map(structural_integration_indices)
                    .transpose()?
                    .unwrap_or_default(),
            ),
            solid_projection: projection.map(Cow::Owned),
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
            let projected = ObservationElement {
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

    pub fn volume_integration_point_count(&self) -> usize {
        match &self.volume_integration_points {
            IntegrationPoints::Materialized(points) => points.len(),
            IntegrationPoints::Sampled { count, .. } => *count,
        }
    }

    /// Integration-point entities; a sampled layout builds them on first use.
    pub fn volume_integration_points(&self) -> Result<&[ObservationElement], ObservationError> {
        match &self.volume_integration_points {
            IntegrationPoints::Materialized(points) => Ok(points),
            IntegrationPoints::Sampled { points, .. } => {
                if points.get().is_none() {
                    let sampling = self.sampling.as_ref().ok_or(ObservationError::Identity(
                        "sampled integration points require their sampled layout",
                    ))?;
                    let projection =
                        self.solid_projection
                            .as_deref()
                            .ok_or(ObservationError::Identity(
                                "sampled integration points require their projection",
                            ))?;
                    let built = sampled::integration_points(sampling, projection)?;
                    let _ = points.set(built);
                }
                Ok(points.get().map(Vec::as_slice).unwrap_or_default())
            }
        }
    }

    /// Expand actual parent membership, not a solver-specific fixed sampling count.
    /// Reject the complete selection before allocating its output indices.
    pub fn integration_indices(
        &self,
        parents: impl Iterator<Item = usize> + Clone,
        maximum: usize,
    ) -> Result<Vec<usize>, ObservationError> {
        let count = parents.clone().try_fold(0usize, |count, parent| {
            let points = self.integration_points_by_parent.count(parent).ok_or(
                ObservationError::Identity("integration-point parent exceeds observation topology"),
            )?;
            let count = count
                .checked_add(points)
                .ok_or(ObservationError::Limit("integration-point target count"))?;
            if count > maximum {
                return Err(ObservationError::Limit("integration-point target count"));
            }
            Ok(count)
        })?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(count)
            .map_err(|_| ObservationError::Limit("integration-point target allocation"))?;
        for parent in parents {
            self.integration_points_by_parent
                .extend(parent, &mut indices);
        }
        Ok(indices)
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
    ) -> Result<&[ObservationElement], ObservationError> {
        match domain {
            RecoveredFieldEntityDomain::VolumeElements => Ok(&self.volume_elements),
            RecoveredFieldEntityDomain::VolumeIntegrationPoints => self.volume_integration_points(),
            RecoveredFieldEntityDomain::BeamElements => Ok(&self.beam_elements),
            RecoveredFieldEntityDomain::ShellElements => Ok(&self.shell_elements),
            RecoveredFieldEntityDomain::Connectors => Err(ObservationError::Unsupported(
                "connectors are resolved by the canonical connector plan",
            )),
            RecoveredFieldEntityDomain::ConstraintGroups => Err(ObservationError::Unsupported(
                "constraint groups are resolved by the canonical constraint plan",
            )),
            RecoveredFieldEntityDomain::Nodes => Err(ObservationError::Invalid(
                "nodal field domain cannot resolve element topology".into(),
            )),
            RecoveredFieldEntityDomain::OrientedEdges if self.sampling.is_some() => {
                Ok(&self.oriented_edges)
            }
            RecoveredFieldEntityDomain::OrientedEdges => Err(ObservationError::Unsupported(
                "oriented edges require sampled volume topology",
            )),
            RecoveredFieldEntityDomain::OrientedFaces if self.finite_volume.is_some() => {
                Ok(&self.oriented_faces)
            }
            RecoveredFieldEntityDomain::OrientedFaces => Err(ObservationError::Unsupported(
                "owner-oriented faces require canonical finite-volume topology",
            )),
        }
    }
}

fn structural_integration_indices(
    projection: &AnalysisProjection,
) -> Result<Vec<Vec<usize>>, ObservationError> {
    let points = TETRAHEDRON_FOUR_POINT_RULE.len();
    let mut parents =
        crate::topology::storage::reserve(projection.elements.len()).map_err(storage_error)?;
    for parent in 0..projection.elements.len() {
        let start = parent
            .checked_mul(points)
            .ok_or(ObservationError::Limit("integration-point count"))?;
        let mut indices = crate::topology::storage::reserve(points).map_err(storage_error)?;
        indices.extend(start..start + points);
        parents.push(indices);
    }
    Ok(parents)
}

fn integration_point_elements(
    projection: &AnalysisProjection,
) -> Result<Vec<ObservationElement>, ObservationError> {
    let count = projection
        .elements
        .len()
        .checked_mul(TETRAHEDRON_FOUR_POINT_RULE.len())
        .ok_or(ObservationError::Limit("integration-point count"))?;
    let mut points = crate::topology::storage::reserve(count).map_err(storage_error)?;
    for element in &projection.elements {
        crate::topology::storage::check_cancelled().map_err(storage_error)?;
        for (ordinal, barycentric) in TETRAHEDRON_FOUR_POINT_RULE.into_iter().enumerate() {
            let (shape, _) = crate::topology::quadrature::volume_shape(element.order, barycentric)
                .map_err(ObservationError::Accounting)?;
            let mut position_m = [0.0; 3];
            for (value, index) in shape.iter().zip(&element.node_indices) {
                let node = projection
                    .nodes
                    .get(*index)
                    .ok_or(ObservationError::Identity(
                        "observation element references an absent projection node",
                    ))?;
                for (position, coordinate) in position_m
                    .iter_mut()
                    .zip(node.coordinates_m.canonical_value())
                {
                    *position += value * coordinate;
                }
            }
            points.push(ObservationElement {
                identity: integration_point_identity(element.stable_identity, ordinal),
                region_id: element.region_id.source_topology_id.clone(),
                centroid_m: position_m,
            });
        }
    }
    Ok(points)
}

fn integration_point_identity(element: StableDigest, ordinal: usize) -> StableDigest {
    let ordinal = (ordinal as u64).to_le_bytes();
    StableDigest::from_bytes(
        *Digest::sha256_parts([
            b"runmat.fea.structural.integration-point\0".as_slice(),
            element.bytes().as_slice(),
            ordinal.as_slice(),
        ])
        .bytes(),
    )
}

fn centroid(indices: &[usize], nodes: &[ObservationNode]) -> Result<[f64; 3], ObservationError> {
    let mut sum = [0.0; 3];
    for index in indices {
        let node = nodes.get(*index).ok_or(ObservationError::Identity(
            "structural element exceeds its canonical node space",
        ))?;
        for (total, coordinate) in sum.iter_mut().zip(node.coordinates_m.canonical_value()) {
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

fn storage_error(error: crate::topology::storage::TopologyStorageError) -> ObservationError {
    ObservationError::Topology(error.into())
}
