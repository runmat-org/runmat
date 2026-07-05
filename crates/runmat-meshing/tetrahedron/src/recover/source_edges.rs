use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{ProtectedBoundaryComplex, TetrahedronMesh, TopologyEntityId},
    quality::predicate::{
        tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_signed_volume,
    },
};

use crate::cavity::constrained::{
    constrained_cavity_from_selected_tetrahedra,
    recover_constrained_cavity_source_edge_by_split_refill, CavityTetrahedron,
    ConstrainedCavityNode, ConstrainedCavityRefillOptions, ConstrainedCavitySourceEdgeSplitError,
};

use super::{
    TetrahedronProtectedEdgeTopology, TetrahedronRecoveryKind, TetrahedronRecoveryQueue,
    TetrahedronRecoveryStatus,
};

pub(super) struct SourceEdgeSplitRefillRecovery {
    pub attempted_source_edge_count: usize,
    pub attempted_cad_curve_source_edge_count: usize,
    pub accepted_source_edge_count: usize,
    pub accepted_cad_curve_source_edge_count: usize,
    pub rejected_source_edge_count: usize,
    pub rejected_cad_curve_source_edge_count: usize,
    pub rejection_counts: BTreeMap<String, usize>,
}

pub(super) fn evaluate_source_edge_split_refill_recovery(
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

impl Default for SourceEdgeSplitRefillRecovery {
    fn default() -> Self {
        Self {
            attempted_source_edge_count: 0,
            attempted_cad_curve_source_edge_count: 0,
            accepted_source_edge_count: 0,
            accepted_cad_curve_source_edge_count: 0,
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
