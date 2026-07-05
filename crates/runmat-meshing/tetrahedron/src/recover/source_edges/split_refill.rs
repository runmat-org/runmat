use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        MeshingStage, ProtectedBoundaryComplex, Tetrahedron4Element, TetrahedronBoundaryFace,
        TetrahedronMesh, TetrahedronMeshNode, TopologyEntityId,
    },
    quality::predicate::{
        tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_signed_volume,
    },
};

use crate::cavity::constrained::{
    constrained_cavity_from_selected_tetrahedra,
    recover_constrained_cavity_source_edge_by_split_refill, CavityTetrahedron,
    ConstrainedCavityNode, ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavitySourceEdgeSplitError, ConstrainedCavitySourceEdgeSplitRecovery,
};
use crate::protected_edges::{face_edges, sorted_edge};

use crate::recover::{
    topology::sorted_topology_ids, TetrahedronProtectedEdgeTopology, TetrahedronRecoveryKind,
    TetrahedronRecoveryQueue, TetrahedronRecoveryStatus,
};

pub(in crate::recover) struct SourceEdgeSplitRefillRecovery {
    pub(in crate::recover) attempted_source_edge_count: usize,
    pub(in crate::recover) attempted_cad_curve_source_edge_count: usize,
    pub(in crate::recover) accepted_source_edge_count: usize,
    pub(in crate::recover) accepted_cad_curve_source_edge_count: usize,
    pub(in crate::recover) applied_source_edge_count: usize,
    pub(in crate::recover) applied_cad_curve_source_edge_count: usize,
    pub(in crate::recover) rejected_source_edge_count: usize,
    pub(in crate::recover) rejected_cad_curve_source_edge_count: usize,
    pub(in crate::recover) rejection_counts: BTreeMap<String, usize>,
}

pub(in crate::recover) fn evaluate_source_edge_split_refill_recovery(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &TetrahedronMesh,
) -> SourceEdgeSplitRefillRecovery {
    let Some(adapter) = SplitRefillAdapter::from_mesh(tetrahedron_mesh) else {
        return SourceEdgeSplitRefillRecovery::rejected_all(
            initial_recovery_queue,
            plc,
            "source_edge_split_refill_id_mapping_overflow",
        );
    };
    let cad_curve_source_edge_ids = cad_curve_source_edge_ids(plc);
    let mut recovery = SourceEdgeSplitRefillRecovery::default();

    for item in initial_recovery_queue.items.iter().filter(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.status == TetrahedronRecoveryStatus::Missing
            && matches!(
                item.protected_edge_topology,
                Some(
                    TetrahedronProtectedEdgeTopology::VolumeEdge
                        | TetrahedronProtectedEdgeTopology::InteriorEdge
                )
            )
    }) {
        let is_cad_curve_source_edge = item
            .source_entity_id
            .as_ref()
            .is_some_and(|source_edge_id| cad_curve_source_edge_ids.contains(source_edge_id));
        recovery.attempted_source_edge_count += 1;
        if is_cad_curve_source_edge {
            recovery.attempted_cad_curve_source_edge_count += 1;
        }

        let Some(edge) = item
            .protected_edge_node_ids
            .as_ref()
            .and_then(|edge| adapter.edge_to_cavity_ids(edge))
        else {
            recovery.reject(
                "source_edge_split_refill_missing_edge_node_mapping",
                is_cad_curve_source_edge,
            );
            continue;
        };
        let Some(seed_index) = adapter.first_tetrahedron_index_incident_to_edge(edge) else {
            recovery.reject(
                "source_edge_split_refill_no_incident_tetrahedron",
                is_cad_curve_source_edge,
            );
            continue;
        };
        let Ok(cavity) = constrained_cavity_from_selected_tetrahedra(
            &adapter.tetrahedra,
            &[seed_index],
            Vec::new(),
        ) else {
            recovery.reject(
                "source_edge_split_refill_cavity_extraction",
                is_cad_curve_source_edge,
            );
            continue;
        };

        match recover_constrained_cavity_source_edge_by_split_refill(
            &cavity,
            &adapter.nodes,
            &adapter.nodes,
            &adapter.tetrahedra,
            edge,
            ConstrainedCavityRefillOptions::default(),
        ) {
            Ok(split_refill) if split_refill.refill_evaluation.refill.is_some() => {
                recovery.accepted_source_edge_count += 1;
                if is_cad_curve_source_edge {
                    recovery.accepted_cad_curve_source_edge_count += 1;
                }
            }
            Ok(split_refill) => {
                recovery.rejected_source_edge_count += 1;
                if is_cad_curve_source_edge {
                    recovery.rejected_cad_curve_source_edge_count += 1;
                }
                if split_refill.refill_evaluation.rejected_by_reason.is_empty() {
                    *recovery
                        .rejection_counts
                        .entry("source_edge_split_refill_no_refill_candidate".to_string())
                        .or_default() += 1;
                } else {
                    for (reason, count) in split_refill.refill_evaluation.rejected_by_reason {
                        *recovery
                            .rejection_counts
                            .entry(format!("source_edge_split_refill_{reason}"))
                            .or_default() += count;
                    }
                }
            }
            Err(error) => recovery.reject(error_rejection_key(&error), is_cad_curve_source_edge),
        }
    }

    recovery
}

pub(in crate::recover) fn apply_source_edge_split_refill_recovery(
    plc: &ProtectedBoundaryComplex,
    recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> SourceEdgeSplitRefillRecovery {
    let cad_curve_source_edge_ids = cad_curve_source_edge_ids(plc);
    let mut recovery = SourceEdgeSplitRefillRecovery::default();

    for item in recovery_queue.items.iter().filter(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.status == TetrahedronRecoveryStatus::Missing
            && matches!(
                item.protected_edge_topology,
                Some(
                    TetrahedronProtectedEdgeTopology::VolumeEdge
                        | TetrahedronProtectedEdgeTopology::InteriorEdge
                )
            )
    }) {
        let is_cad_curve_source_edge = item
            .source_entity_id
            .as_ref()
            .is_some_and(|source_edge_id| cad_curve_source_edge_ids.contains(source_edge_id));
        recovery.attempted_source_edge_count += 1;
        if is_cad_curve_source_edge {
            recovery.attempted_cad_curve_source_edge_count += 1;
        }

        let Some(source_edge_id) = item.source_entity_id.as_ref() else {
            recovery.reject(
                "source_edge_split_refill_missing_source_edge_id",
                is_cad_curve_source_edge,
            );
            continue;
        };
        let Some(protected_edge_node_ids) = item.protected_edge_node_ids.as_ref() else {
            recovery.reject(
                "source_edge_split_refill_missing_protected_edge_nodes",
                is_cad_curve_source_edge,
            );
            continue;
        };
        let Some(adapter) = SplitRefillAdapter::from_mesh(tetrahedron_mesh) else {
            recovery.reject(
                "source_edge_split_refill_id_mapping_overflow",
                is_cad_curve_source_edge,
            );
            continue;
        };
        let Some(edge) = adapter.edge_to_cavity_ids(protected_edge_node_ids) else {
            recovery.reject(
                "source_edge_split_refill_missing_edge_node_mapping",
                is_cad_curve_source_edge,
            );
            continue;
        };
        let Some(split_refill) =
            evaluate_split_refill(&adapter, edge, is_cad_curve_source_edge, &mut recovery)
        else {
            continue;
        };
        recovery.accepted_source_edge_count += 1;
        if is_cad_curve_source_edge {
            recovery.accepted_cad_curve_source_edge_count += 1;
        }
        match apply_split_refill(
            plc,
            tetrahedron_mesh,
            &adapter,
            &split_refill,
            protected_edge_node_ids,
            source_edge_id,
        ) {
            Ok(()) => {
                recovery.applied_source_edge_count += 1;
                if is_cad_curve_source_edge {
                    recovery.applied_cad_curve_source_edge_count += 1;
                }
            }
            Err(reason) => recovery.reject(reason, is_cad_curve_source_edge),
        }
    }

    recovery
}

impl Default for SourceEdgeSplitRefillRecovery {
    fn default() -> Self {
        Self {
            attempted_source_edge_count: 0,
            attempted_cad_curve_source_edge_count: 0,
            accepted_source_edge_count: 0,
            accepted_cad_curve_source_edge_count: 0,
            applied_source_edge_count: 0,
            applied_cad_curve_source_edge_count: 0,
            rejected_source_edge_count: 0,
            rejected_cad_curve_source_edge_count: 0,
            rejection_counts: BTreeMap::new(),
        }
    }
}

impl SourceEdgeSplitRefillRecovery {
    fn rejected_all(
        initial_recovery_queue: &TetrahedronRecoveryQueue,
        plc: &ProtectedBoundaryComplex,
        reason: &'static str,
    ) -> Self {
        let cad_curve_source_edge_ids = cad_curve_source_edge_ids(plc);
        let attempted_source_edge_count = initial_recovery_queue
            .items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceEdge
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && matches!(
                        item.protected_edge_topology,
                        Some(
                            TetrahedronProtectedEdgeTopology::VolumeEdge
                                | TetrahedronProtectedEdgeTopology::InteriorEdge
                        )
                    )
            })
            .count();
        let attempted_cad_curve_source_edge_count = initial_recovery_queue
            .items
            .iter()
            .filter(|item| {
                item.kind == TetrahedronRecoveryKind::SourceEdge
                    && item.status == TetrahedronRecoveryStatus::Missing
                    && matches!(
                        item.protected_edge_topology,
                        Some(
                            TetrahedronProtectedEdgeTopology::VolumeEdge
                                | TetrahedronProtectedEdgeTopology::InteriorEdge
                        )
                    )
                    && item
                        .source_entity_id
                        .as_ref()
                        .is_some_and(|source_edge_id| {
                            cad_curve_source_edge_ids.contains(source_edge_id)
                        })
            })
            .count();
        let mut rejection_counts = BTreeMap::new();
        if attempted_source_edge_count > 0 {
            rejection_counts.insert(reason.to_string(), attempted_source_edge_count);
        }
        Self {
            attempted_source_edge_count,
            attempted_cad_curve_source_edge_count,
            accepted_source_edge_count: 0,
            accepted_cad_curve_source_edge_count: 0,
            applied_source_edge_count: 0,
            applied_cad_curve_source_edge_count: 0,
            rejected_source_edge_count: attempted_source_edge_count,
            rejected_cad_curve_source_edge_count: attempted_cad_curve_source_edge_count,
            rejection_counts,
        }
    }

    fn reject(&mut self, reason: impl Into<String>, is_cad_curve_source_edge: bool) {
        self.rejected_source_edge_count += 1;
        if is_cad_curve_source_edge {
            self.rejected_cad_curve_source_edge_count += 1;
        }
        *self.rejection_counts.entry(reason.into()).or_default() += 1;
    }
}

struct SplitRefillAdapter {
    nodes: Vec<ConstrainedCavityNode>,
    tetrahedra: Vec<CavityTetrahedron>,
    node_ids: BTreeMap<TopologyEntityId, u32>,
    topology_node_ids: Vec<TopologyEntityId>,
    element_ids: Vec<TopologyEntityId>,
}

impl SplitRefillAdapter {
    fn from_mesh(tetrahedron_mesh: &TetrahedronMesh) -> Option<Self> {
        if tetrahedron_mesh.nodes.len() > u32::MAX as usize
            || tetrahedron_mesh.elements.len() > u32::MAX as usize
        {
            return None;
        }
        let node_ids = tetrahedron_mesh
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.node_id.clone(), index as u32))
            .collect::<BTreeMap<_, _>>();
        let nodes = tetrahedron_mesh
            .nodes
            .iter()
            .map(|node| ConstrainedCavityNode {
                node_id: node_ids[&node.node_id],
                coordinates_m: node.coordinates_m,
            })
            .collect::<Vec<_>>();
        let topology_node_ids = tetrahedron_mesh
            .nodes
            .iter()
            .map(|node| node.node_id.clone())
            .collect::<Vec<_>>();
        let element_ids = tetrahedron_mesh
            .elements
            .iter()
            .map(|element| element.element_id.clone())
            .collect::<Vec<_>>();
        let node_coordinates = tetrahedron_mesh
            .nodes
            .iter()
            .map(|node| (node.node_id.clone(), node.coordinates_m))
            .collect::<BTreeMap<_, _>>();
        let mut tetrahedra =
            Vec::<CavityTetrahedron>::with_capacity(tetrahedron_mesh.elements.len());
        for (index, element) in tetrahedron_mesh.elements.iter().enumerate() {
            let node_ids = [
                node_ids.get(&element.node_ids[0]).copied()?,
                node_ids.get(&element.node_ids[1]).copied()?,
                node_ids.get(&element.node_ids[2]).copied()?,
                node_ids.get(&element.node_ids[3]).copied()?,
            ];
            let points = [
                *node_coordinates.get(&element.node_ids[0])?,
                *node_coordinates.get(&element.node_ids[1])?,
                *node_coordinates.get(&element.node_ids[2])?,
                *node_coordinates.get(&element.node_ids[3])?,
            ];
            let volume_m3 = tetrahedron_signed_volume(points).abs();
            tetrahedra.push(CavityTetrahedron {
                tetrahedron_id: index as u32,
                component_id: 0,
                node_ids,
                source_surface_element_id: 0,
                region_ids: vec![element.material_region_id.clone()],
                volume_m3,
                aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(points).abs(),
            });
        }

        Some(Self {
            nodes,
            tetrahedra,
            node_ids,
            topology_node_ids,
            element_ids,
        })
    }

    fn edge_to_cavity_ids(&self, edge: &[TopologyEntityId; 2]) -> Option<[u32; 2]> {
        Some(sorted_u32_edge([
            *self.node_ids.get(&edge[0])?,
            *self.node_ids.get(&edge[1])?,
        ]))
    }

    fn first_tetrahedron_index_incident_to_edge(&self, edge: [u32; 2]) -> Option<usize> {
        self.tetrahedra.iter().position(|tetrahedron| {
            edge.iter()
                .all(|node_id| tetrahedron.node_ids.contains(node_id))
        })
    }
}

fn evaluate_split_refill(
    adapter: &SplitRefillAdapter,
    edge: [u32; 2],
    is_cad_curve_source_edge: bool,
    recovery: &mut SourceEdgeSplitRefillRecovery,
) -> Option<ConstrainedCavitySourceEdgeSplitRecovery> {
    let Some(seed_index) = adapter.first_tetrahedron_index_incident_to_edge(edge) else {
        recovery.reject(
            "source_edge_split_refill_no_incident_tetrahedron",
            is_cad_curve_source_edge,
        );
        return None;
    };
    let Ok(cavity) =
        constrained_cavity_from_selected_tetrahedra(&adapter.tetrahedra, &[seed_index], Vec::new())
    else {
        recovery.reject(
            "source_edge_split_refill_cavity_extraction",
            is_cad_curve_source_edge,
        );
        return None;
    };

    match recover_constrained_cavity_source_edge_by_split_refill(
        &cavity,
        &adapter.nodes,
        &adapter.nodes,
        &adapter.tetrahedra,
        edge,
        ConstrainedCavityRefillOptions::default(),
    ) {
        Ok(split_refill) if split_refill.refill_evaluation.refill.is_some() => Some(split_refill),
        Ok(split_refill) => {
            recovery.rejected_source_edge_count += 1;
            if is_cad_curve_source_edge {
                recovery.rejected_cad_curve_source_edge_count += 1;
            }
            if split_refill.refill_evaluation.rejected_by_reason.is_empty() {
                *recovery
                    .rejection_counts
                    .entry("source_edge_split_refill_no_refill_candidate".to_string())
                    .or_default() += 1;
            } else {
                for (reason, count) in split_refill.refill_evaluation.rejected_by_reason {
                    *recovery
                        .rejection_counts
                        .entry(format!("source_edge_split_refill_{reason}"))
                        .or_default() += count;
                }
            }
            None
        }
        Err(error) => {
            recovery.reject(error_rejection_key(&error), is_cad_curve_source_edge);
            None
        }
    }
}

fn apply_split_refill(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
    adapter: &SplitRefillAdapter,
    split_refill: &ConstrainedCavitySourceEdgeSplitRecovery,
    protected_edge_node_ids: &[TopologyEntityId; 2],
    source_edge_id: &TopologyEntityId,
) -> Result<(), &'static str> {
    let Some(refill) = split_refill.refill_evaluation.refill.as_ref() else {
        return Err("source_edge_split_refill_no_refill_candidate");
    };
    let split_node_id = split_node_topology_id(protected_edge_node_ids);
    if !tetrahedron_mesh
        .nodes
        .iter()
        .any(|node| node.node_id == split_node_id)
    {
        tetrahedron_mesh.nodes.push(TetrahedronMeshNode {
            node_id: split_node_id.clone(),
            coordinates_m: split_refill.split.split_node.coordinates_m,
        });
    }

    let removed_tetrahedron_ids = split_refill
        .split
        .cavity
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let material_region_id = refill_material_region_id(
        &split_refill.split.source_tetrahedra,
        &removed_tetrahedron_ids,
    )
    .ok_or("source_edge_split_refill_material_region")?;
    let mut next_tetrahedron_id = split_refill
        .split
        .source_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.tetrahedron_id)
        .max()
        .unwrap_or_default()
        .saturating_add(1);
    let mut final_tetrahedra = split_refill
        .split
        .source_tetrahedra
        .iter()
        .filter(|tetrahedron| !removed_tetrahedron_ids.contains(&tetrahedron.tetrahedron_id))
        .cloned()
        .collect::<Vec<_>>();
    for refill_tetrahedron in &refill.tetrahedra {
        final_tetrahedra.push(refill_tetrahedron_to_cavity_tetrahedron(
            refill_tetrahedron,
            next_tetrahedron_id,
            &material_region_id,
        ));
        next_tetrahedron_id = next_tetrahedron_id.saturating_add(1);
    }

    tetrahedron_mesh.elements = final_tetrahedra
        .iter()
        .enumerate()
        .map(|(index, tetrahedron)| {
            let node_ids = tetrahedron.node_ids.map(|node_id| {
                topology_id_for_cavity_node(adapter, &split_refill.split, &split_node_id, node_id)
            });
            let Some(node_ids) = node_ids.into_iter().collect::<Option<Vec<_>>>() else {
                return Err("source_edge_split_refill_node_mapping");
            };
            Ok(Tetrahedron4Element {
                element_id: element_topology_id(adapter, tetrahedron, source_edge_id, index),
                node_ids: [
                    node_ids[0].clone(),
                    node_ids[1].clone(),
                    node_ids[2].clone(),
                    node_ids[3].clone(),
                ],
                material_region_id: tetrahedron
                    .region_ids
                    .first()
                    .cloned()
                    .unwrap_or_else(|| material_region_id.clone()),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    replace_split_boundary_faces(
        plc,
        tetrahedron_mesh,
        protected_edge_node_ids,
        &split_node_id,
        source_edge_id,
    );
    Ok(())
}

fn refill_material_region_id(
    source_tetrahedra: &[CavityTetrahedron],
    removed_tetrahedron_ids: &BTreeSet<u32>,
) -> Option<String> {
    let material_regions = source_tetrahedra
        .iter()
        .filter(|tetrahedron| removed_tetrahedron_ids.contains(&tetrahedron.tetrahedron_id))
        .filter_map(|tetrahedron| tetrahedron.region_ids.first().cloned())
        .collect::<BTreeSet<_>>();
    (material_regions.len() == 1).then(|| material_regions.into_iter().next().unwrap())
}

fn refill_tetrahedron_to_cavity_tetrahedron(
    refill_tetrahedron: &ConstrainedCavityRefillTetrahedron,
    tetrahedron_id: u32,
    material_region_id: &str,
) -> CavityTetrahedron {
    CavityTetrahedron {
        tetrahedron_id,
        component_id: 0,
        node_ids: refill_tetrahedron.node_ids,
        source_surface_element_id: 0,
        region_ids: vec![material_region_id.to_string()],
        volume_m3: refill_tetrahedron.volume_m3,
        aspect_ratio: refill_tetrahedron.aspect_ratio,
        exact_scaled_jacobian: refill_tetrahedron.exact_scaled_jacobian,
    }
}

fn topology_id_for_cavity_node(
    adapter: &SplitRefillAdapter,
    split: &crate::cavity::constrained::ConstrainedCavitySourceEdgeSplit,
    split_node_id: &TopologyEntityId,
    node_id: u32,
) -> Option<TopologyEntityId> {
    adapter
        .topology_node_ids
        .get(node_id as usize)
        .cloned()
        .or_else(|| (node_id == split.split_node.node_id).then(|| split_node_id.clone()))
}

fn element_topology_id(
    adapter: &SplitRefillAdapter,
    tetrahedron: &CavityTetrahedron,
    source_edge_id: &TopologyEntityId,
    index: usize,
) -> TopologyEntityId {
    adapter
        .element_ids
        .get(tetrahedron.tetrahedron_id as usize)
        .cloned()
        .unwrap_or_else(|| TopologyEntityId {
            stage: MeshingStage::TetrahedronMesh,
            id: format!("source_edge_split_refill_{}_{}", source_edge_id.id, index),
        })
}

fn split_node_topology_id(protected_edge_node_ids: &[TopologyEntityId; 2]) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::TetrahedronMesh,
        id: format!(
            "source_edge_split_{}_{}",
            protected_edge_node_ids[0].id, protected_edge_node_ids[1].id
        ),
    }
}

fn replace_split_boundary_faces(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
    protected_edge_node_ids: &[TopologyEntityId; 2],
    split_node_id: &TopologyEntityId,
    source_edge_id: &TopologyEntityId,
) {
    let protected_edge_key = sorted_edge(protected_edge_node_ids.clone());
    tetrahedron_mesh.boundary_faces.retain(|face| {
        sorted_topology_face_edges(face.node_ids.clone())
            .into_iter()
            .all(|edge| edge != protected_edge_key)
    });
    let element_face_counts = element_face_counts(&tetrahedron_mesh.elements);
    let mut boundary_face_keys = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| sorted_topology_ids(face.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    for facet in plc
        .facets
        .iter()
        .filter(|facet| facet_contains_edge(facet.node_ids.clone(), protected_edge_key.clone()))
    {
        let Some(opposite_node_id) = facet
            .node_ids
            .iter()
            .find(|node_id| !protected_edge_key.contains(node_id))
        else {
            continue;
        };
        for (index, node_ids) in [
            [
                protected_edge_key[0].clone(),
                split_node_id.clone(),
                opposite_node_id.clone(),
            ],
            [
                split_node_id.clone(),
                protected_edge_key[1].clone(),
                opposite_node_id.clone(),
            ],
        ]
        .into_iter()
        .enumerate()
        {
            let face_key = sorted_topology_ids(node_ids.clone());
            if element_face_counts.get(&face_key).copied() != Some(1)
                || !boundary_face_keys.insert(face_key)
            {
                continue;
            }
            tetrahedron_mesh
                .boundary_faces
                .push(TetrahedronBoundaryFace {
                    face_id: TopologyEntityId {
                        stage: MeshingStage::TetrahedronMesh,
                        id: format!(
                            "source_edge_split_refill_face_{}_{}",
                            facet.facet_id.id, index
                        ),
                    },
                    source_edge_ids: split_source_edge_ids_for_face(
                        plc,
                        protected_edge_node_ids,
                        split_node_id,
                        source_edge_id,
                        node_ids.clone(),
                    ),
                    node_ids,
                    source_face_id: facet.source_face_id.clone(),
                });
        }
    }
}

fn split_source_edge_ids_for_face(
    plc: &ProtectedBoundaryComplex,
    protected_edge_node_ids: &[TopologyEntityId; 2],
    split_node_id: &TopologyEntityId,
    source_edge_id: &TopologyEntityId,
    node_ids: [TopologyEntityId; 3],
) -> [Option<TopologyEntityId>; 3] {
    let mut left_child = [protected_edge_node_ids[0].clone(), split_node_id.clone()];
    let mut right_child = [protected_edge_node_ids[1].clone(), split_node_id.clone()];
    left_child.sort();
    right_child.sort();
    face_edges(node_ids).map(|edge| {
        if edge == left_child || edge == right_child {
            Some(source_edge_id.clone())
        } else {
            plc.protected_edges
                .iter()
                .find(|protected_edge| sorted_edge(protected_edge.node_ids.clone()) == edge)
                .map(|protected_edge| protected_edge.source_edge_id.clone())
        }
    })
}

fn facet_contains_edge(facet_node_ids: [TopologyEntityId; 3], edge: [TopologyEntityId; 2]) -> bool {
    edge.iter().all(|node_id| facet_node_ids.contains(node_id))
}

fn sorted_topology_face_edges(node_ids: [TopologyEntityId; 3]) -> [[TopologyEntityId; 2]; 3] {
    face_edges(node_ids)
}

fn element_face_counts(elements: &[Tetrahedron4Element]) -> BTreeMap<[TopologyEntityId; 3], usize> {
    let mut counts = BTreeMap::<[TopologyEntityId; 3], usize>::new();
    for element in elements {
        for face in tetrahedron_faces(element.node_ids.clone()) {
            *counts.entry(face).or_default() += 1;
        }
    }
    counts
}

fn tetrahedron_faces(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 3]; 4] {
    [
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
    ]
}

fn sorted_u32_edge(mut edge: [u32; 2]) -> [u32; 2] {
    edge.sort_unstable();
    edge
}

fn cad_curve_source_edge_ids(plc: &ProtectedBoundaryComplex) -> BTreeSet<TopologyEntityId> {
    plc.protected_edges
        .iter()
        .filter(|edge| edge.cad_curve_boundary.is_some())
        .map(|edge| edge.source_edge_id.clone())
        .collect()
}

fn error_rejection_key(error: &ConstrainedCavitySourceEdgeSplitError) -> &'static str {
    match error {
        ConstrainedCavitySourceEdgeSplitError::MissingBoundaryEdge { .. } => {
            "source_edge_split_refill_missing_boundary_edge"
        }
        ConstrainedCavitySourceEdgeSplitError::MissingBoundaryNode { .. } => {
            "source_edge_split_refill_missing_boundary_node"
        }
        ConstrainedCavitySourceEdgeSplitError::MissingSourceNode { .. } => {
            "source_edge_split_refill_missing_source_node"
        }
        ConstrainedCavitySourceEdgeSplitError::MissingRemovedSourceTetrahedron { .. } => {
            "source_edge_split_refill_missing_removed_source_tetrahedron"
        }
        ConstrainedCavitySourceEdgeSplitError::NoIncidentSourceTetrahedron { .. } => {
            "source_edge_split_refill_no_incident_tetrahedron"
        }
        ConstrainedCavitySourceEdgeSplitError::DegenerateSplitTetrahedron { .. } => {
            "source_edge_split_refill_degenerate_split_tetrahedron"
        }
        ConstrainedCavitySourceEdgeSplitError::Validation(_) => {
            "source_edge_split_refill_validation"
        }
        ConstrainedCavitySourceEdgeSplitError::Refill(_) => "source_edge_split_refill_refill",
    }
}
