//! Size is measured along edges under the metric field at their endpoints.
//!
//! A boundary node carries the field its faces were meshed with: each face's constant sizing
//! intersected with the face's incident entity-local sources graded to the node, exactly as the
//! surface stage evaluates it. An interior node carries its region's constant sizing intersected
//! with every entity-local source and every face's constant sizing, each graded by distance from
//! its support. An edge's metric length is the integral of the metric along it
//! (`integrated_edge_length`), so a cell next to a fine feature is measured by the sizes at its
//! own corners rather than by one tensor for the whole cell. Anisotropic sizing (for example
//! proximity across a thin section) keeps its directions all the way into the interior.
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::StableDigest;
use runmat_meshing_size::metric::{
    integrated_edge_length, LocalMetricSource, MetricFieldRequest, MetricTensor3,
    ResolvedMetricField, SurfaceMetricSource,
};

use super::{
    error, metric_error, DelaunayVolumeProvenance, DelaunayVolumeQualityError,
    DelaunayVolumeQualityErrorKind, TETRAHEDRON_EDGES,
};
use crate::cdt::DelaunayVolumeNode;

struct FaceScope {
    constant: MetricTensor3,
    local: Vec<LocalMetricSource>,
}

pub(in crate::cdt) struct NodeSizeField {
    field: ResolvedMetricField,
    uniform: Option<MetricTensor3>,
    faces: BTreeMap<PersistentEntityId, FaceScope>,
    face_incidence: BTreeMap<PersistentEntityId, BTreeSet<PersistentEntityId>>,
    face_triangles: BTreeMap<PersistentEntityId, Vec<[StableDigest; 3]>>,
    node_faces: BTreeMap<StableDigest, Vec<PersistentEntityId>>,
    all_local: Vec<LocalMetricSource>,
    surfaces: RefCell<Option<Vec<SurfaceMetricSource>>>,
    regions: RefCell<BTreeMap<PersistentEntityId, MetricTensor3>>,
    cache: RefCell<HashMap<(StableDigest, [u64; 3], Option<PersistentEntityId>), MetricTensor3>>,
}

impl NodeSizeField {
    pub(in crate::cdt) fn new(
        request: &MetricFieldRequest,
        provenance: &DelaunayVolumeProvenance,
    ) -> Result<Self, DelaunayVolumeQualityError> {
        let field = ResolvedMetricField::new(request).map_err(metric_error)?;
        let uniform = field
            .uniform_evaluation()
            .map_err(metric_error)?
            .map(|evaluation| evaluation.metric);
        let mut node = Self {
            field,
            uniform,
            faces: BTreeMap::new(),
            face_incidence: BTreeMap::new(),
            face_triangles: BTreeMap::new(),
            node_faces: BTreeMap::new(),
            all_local: Vec::new(),
            surfaces: RefCell::new(None),
            regions: RefCell::new(BTreeMap::new()),
            cache: RefCell::new(HashMap::new()),
        };
        if node.uniform.is_some() {
            return Ok(node);
        }
        // Face incidence from the protected provenance: the face (and contact scopes), its
        // regions, the exact edges of its facet edges and the exact vertices of its nodes.
        let node_entities = provenance
            .nodes
            .iter()
            .map(|binding| (binding.node_identity, binding.entity_ids.as_slice()))
            .collect::<BTreeMap<_, _>>();
        let segment_entities = provenance
            .segments
            .iter()
            .map(|binding| (binding.node_identities, binding.entity_ids.as_slice()))
            .collect::<BTreeMap<_, _>>();
        for facet in &provenance.facets {
            for face in facet
                .entity_ids
                .iter()
                .filter(|entity| entity.kind == PersistentEntityKind::Face)
            {
                let incident = node.face_incidence.entry(face.clone()).or_default();
                incident.extend(facet.entity_ids.iter().cloned());
                incident.extend(facet.region_ids.iter().cloned());
                for (left, right) in [(0, 1), (0, 2), (1, 2)] {
                    let mut edge = [facet.node_identities[left], facet.node_identities[right]];
                    edge.sort_unstable();
                    if let Some(entities) = segment_entities.get(&edge) {
                        incident.extend(entities.iter().cloned());
                    }
                }
                for identity in facet.node_identities {
                    if let Some(entities) = node_entities.get(&identity) {
                        incident.extend(entities.iter().cloned());
                    }
                    let faces = node.node_faces.entry(identity).or_default();
                    if !faces.contains(face) {
                        faces.push(face.clone());
                    }
                }
                node.face_triangles
                    .entry(face.clone())
                    .or_default()
                    .push(facet.node_identities);
            }
        }
        for (face, incident) in &node.face_incidence {
            let constant = node
                .field
                .resolve_constant(incident)
                .map_err(metric_error)?
                .metric;
            let local = node.field.local_sources(incident).map_err(metric_error)?;
            node.faces
                .insert(face.clone(), FaceScope { constant, local });
        }
        node.all_local = node.field.all_local_sources().map_err(metric_error)?;
        Ok(node)
    }

    /// Metrics at the four corners of a cell in `region`. `nodes` is the complete current node
    /// inventory, used once to place the faces' triangles.
    pub(in crate::cdt) fn corner_metrics(
        &self,
        nodes: &[DelaunayVolumeNode],
        corners: [DelaunayVolumeNode; 4],
        region: &PersistentEntityId,
    ) -> Result<[MetricTensor3; 4], DelaunayVolumeQualityError> {
        if let Some(uniform) = self.uniform {
            return Ok([uniform; 4]);
        }
        let mut metrics = [MetricTensor3::isotropic_length_m(1.0).map_err(metric_error)?; 4];
        for (slot, corner) in corners.iter().enumerate() {
            metrics[slot] = self.node_metric(nodes, corner, region)?;
        }
        Ok(metrics)
    }

    fn node_metric(
        &self,
        nodes: &[DelaunayVolumeNode],
        node: &DelaunayVolumeNode,
        region: &PersistentEntityId,
    ) -> Result<MetricTensor3, DelaunayVolumeQualityError> {
        let point = node.coordinates_m.canonical_value();
        let faces = self.node_faces.get(&node.identity);
        let key = (
            node.identity,
            point.map(f64::to_bits),
            faces.is_none().then(|| region.clone()),
        );
        if let Some(metric) = self.cache.borrow().get(&key) {
            return Ok(*metric);
        }
        let metric = match faces {
            Some(faces) => {
                let mut metric = None;
                for face in faces {
                    let scope = self.faces.get(face).ok_or_else(|| {
                        error(
                            DelaunayVolumeQualityErrorKind::InvalidMetricContext,
                            None,
                            "boundary node names a face without metric incidence",
                        )
                    })?;
                    let at_face = self
                        .field
                        .evaluate_graded(scope.constant, &scope.local, point)
                        .map_err(metric_error)?;
                    metric = Some(match metric {
                        None => at_face,
                        Some(previous) => intersect(previous, at_face)?,
                    });
                }
                metric.ok_or_else(|| {
                    error(
                        DelaunayVolumeQualityErrorKind::InvalidMetricContext,
                        None,
                        "boundary node has no incident face",
                    )
                })?
            }
            None => {
                self.prepare_surfaces(nodes)?;
                let constant = self.region_constant(region)?;
                let surfaces = self.surfaces.borrow();
                self.field
                    .evaluate_graded_with_surfaces(
                        constant,
                        &self.all_local,
                        surfaces.as_deref().unwrap_or_default(),
                        point,
                    )
                    .map_err(metric_error)?
            }
        };
        let mut cache = self.cache.borrow_mut();
        if cache.len() >= 1 << 20 {
            cache.clear();
        }
        cache.insert(key, metric);
        Ok(metric)
    }

    fn region_constant(
        &self,
        region: &PersistentEntityId,
    ) -> Result<MetricTensor3, DelaunayVolumeQualityError> {
        if let Some(metric) = self.regions.borrow().get(region) {
            return Ok(*metric);
        }
        let metric = self
            .field
            .resolve_constant(&BTreeSet::from([region.clone()]))
            .map_err(metric_error)?
            .metric;
        self.regions.borrow_mut().insert(region.clone(), metric);
        Ok(metric)
    }

    fn prepare_surfaces(
        &self,
        nodes: &[DelaunayVolumeNode],
    ) -> Result<(), DelaunayVolumeQualityError> {
        if self.surfaces.borrow().is_some() {
            return Ok(());
        }
        let positions = nodes
            .iter()
            .map(|node| (node.identity, node.coordinates_m.canonical_value()))
            .collect::<HashMap<_, _>>();
        let baseline = self
            .field
            .resolve_constant(&BTreeSet::new())
            .map_err(metric_error)?
            .metric;
        let mut surfaces = Vec::new();
        for (face, incident) in &self.face_incidence {
            let triangles = self.face_triangles[face]
                .iter()
                .filter_map(|triangle| {
                    Some([
                        *positions.get(&triangle[0])?,
                        *positions.get(&triangle[1])?,
                        *positions.get(&triangle[2])?,
                    ])
                })
                .collect::<Vec<_>>();
            // The face's own scope without its regions: region sizing already applies to every
            // interior node of that region directly.
            let scope = incident
                .iter()
                .filter(|entity| entity.kind != PersistentEntityKind::Region)
                .cloned()
                .collect();
            if let Some(source) = self
                .field
                .surface_source(&scope, baseline, triangles)
                .map_err(metric_error)?
            {
                surfaces.push(source);
            }
        }
        *self.surfaces.borrow_mut() = Some(surfaces);
        Ok(())
    }
}

/// Longest integrated metric length over the edges not on a protected facet (zero when every
/// edge is protected).
pub(in crate::cdt) fn volume_owned_maximum_integrated_edge_length(
    points: [[f64; 3]; 4],
    metrics: [MetricTensor3; 4],
    protected_edges: u8,
) -> f64 {
    TETRAHEDRON_EDGES
        .iter()
        .enumerate()
        .filter(|(bit, _)| protected_edges & (1 << bit) == 0)
        .map(|(_, (left, right))| {
            let delta = std::array::from_fn(|axis| points[*right][axis] - points[*left][axis]);
            integrated_edge_length(delta, metrics[*left], metrics[*right])
        })
        .fold(0.0, f64::max)
}

fn intersect(
    left: MetricTensor3,
    right: MetricTensor3,
) -> Result<MetricTensor3, DelaunayVolumeQualityError> {
    runmat_meshing_size::metric::intersect_metrics(left, right).map_err(metric_error)
}
