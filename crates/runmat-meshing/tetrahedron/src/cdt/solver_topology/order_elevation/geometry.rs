use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    sort_solver_node_exact_parameters, MeshingCancellationSignal, SolverMeshNode,
    SolverMeshTopology, SolverNodeExactParameter, StableDigest,
};

use super::{sorted_edge, MidpointMap, TETRAHEDRON_EDGES};
use crate::cdt::solver_topology::{
    checkpoint, error, DelaunayExactEvaluation, DelaunaySolverTopologyError,
    DelaunaySolverTopologyErrorKind, DelaunaySolverTopologyInput, DelaunaySolverTopologyOptions,
};

mod evaluation;
use evaluation::{
    curve_parameter, evaluate_surface, midpoint_geometry, pcurve_uv, require_matching_points,
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct SurfaceUse {
    source_face_id: PersistentEntityId,
    chart_id: StableDigest,
}

pub(super) fn append_midpoint_nodes(
    input: &DelaunaySolverTopologyInput<'_>,
    topology: &mut SolverMeshTopology,
    evaluation: DelaunayExactEvaluation<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<MidpointMap, DelaunaySolverTopologyError> {
    let node_index = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    let edges = topology
        .volume_elements
        .iter()
        .flat_map(|element| {
            TETRAHEDRON_EDGES
                .map(|edge| sorted_edge([element.node_ids[edge[0]], element.node_ids[edge[1]]]))
        })
        .collect::<BTreeSet<_>>();
    let final_node_count = topology
        .nodes
        .len()
        .checked_add(edges.len())
        .ok_or_else(|| resource("Tet10 node inventory overflows the platform index space"))?;
    if final_node_count as u64 > input.request.resources.maximum_nodes {
        return Err(resource(format!(
            "Tet10 elevation requires {final_node_count} nodes but the hard limit is {}",
            input.request.resources.maximum_nodes
        )));
    }
    let surface_uses = boundary_surface_uses(input, topology)?;
    let exact_edges = boundary_exact_edges(topology)?;
    complete_boundary_surface_parameters(
        input,
        topology,
        &node_index,
        &surface_uses,
        &exact_edges,
        evaluation,
        options,
    )?;
    let mut midpoint_by_edge = BTreeMap::new();
    for (work, edge) in edges.into_iter().enumerate() {
        checkpoint(work as u64, options, cancellation)?;
        let left = topology
            .nodes
            .get(
                *node_index
                    .get(&edge[0])
                    .ok_or_else(|| invalid("edge node is absent"))?,
            )
            .ok_or_else(|| invalid("edge node index is invalid"))?;
        let right = topology
            .nodes
            .get(
                *node_index
                    .get(&edge[1])
                    .ok_or_else(|| invalid("edge node is absent"))?,
            )
            .ok_or_else(|| invalid("edge node index is invalid"))?;
        let uses = surface_uses.get(&edge).cloned().unwrap_or_default();
        let exact_edge = exact_edges.get(&edge);
        let (coordinates_m, exact_parameters) =
            midpoint_geometry(input, left, right, exact_edge, &uses, evaluation, options)?;
        let provenance = midpoint_provenance(topology, edge, &uses, exact_edge)?;
        let node_id = topology.nodes.len() as u64 + 1;
        topology.nodes.push(SolverMeshNode {
            node_id,
            coordinates_m,
            provenance,
            exact_parameters,
        });
        midpoint_by_edge.insert(edge, node_id);
    }
    Ok(midpoint_by_edge)
}

fn complete_boundary_surface_parameters(
    input: &DelaunaySolverTopologyInput<'_>,
    topology: &mut SolverMeshTopology,
    node_index: &BTreeMap<u64, usize>,
    surface_uses: &BTreeMap<[u64; 2], Vec<SurfaceUse>>,
    exact_edges: &BTreeMap<[u64; 2], PersistentEntityId>,
    evaluation: DelaunayExactEvaluation<'_>,
    options: DelaunaySolverTopologyOptions,
) -> Result<(), DelaunaySolverTopologyError> {
    for (edge, edge_id) in exact_edges {
        let Some(uses) = surface_uses.get(edge) else {
            continue;
        };
        for node_id in edge {
            let node = topology
                .nodes
                .get_mut(
                    *node_index
                        .get(node_id)
                        .ok_or_else(|| invalid("edge node is absent"))?,
                )
                .ok_or_else(|| invalid("edge node index is invalid"))?;
            let parameter = curve_parameter(node, edge_id)?;
            for surface_use in uses {
                let evaluator_uv = pcurve_uv(input, edge_id, parameter, surface_use, evaluation)?;
                let point =
                    evaluate_surface(input, surface_use, evaluator_uv, evaluation, options)?;
                require_matching_points(input, node.coordinates_m, point)?;
                insert_surface_parameter(node, surface_use, evaluator_uv)?;
            }
        }
    }
    Ok(())
}

fn insert_surface_parameter(
    node: &mut SolverMeshNode,
    surface_use: &SurfaceUse,
    evaluator_uv: [f64; 2],
) -> Result<(), DelaunaySolverTopologyError> {
    if let Some(existing) =
        node.exact_parameters
            .iter()
            .find_map(|parameter| match parameter {
                SolverNodeExactParameter::Surface {
                    source_face_id,
                    chart_id,
                    evaluator_uv,
                } if source_face_id == &surface_use.source_face_id
                    && chart_id == &surface_use.chart_id =>
                {
                    Some(*evaluator_uv)
                }
                SolverNodeExactParameter::Curve { .. }
                | SolverNodeExactParameter::Surface { .. } => None,
            })
    {
        if existing.map(f64::to_bits) != evaluator_uv.map(f64::to_bits) {
            return Err(invalid(
                "exact coedge images disagree on a boundary node surface parameter",
            ));
        }
        return Ok(());
    }
    node.exact_parameters
        .push(SolverNodeExactParameter::Surface {
            source_face_id: surface_use.source_face_id.clone(),
            chart_id: surface_use.chart_id,
            evaluator_uv,
        });
    sort_solver_node_exact_parameters(&mut node.exact_parameters);
    Ok(())
}

fn boundary_surface_uses(
    input: &DelaunaySolverTopologyInput<'_>,
    topology: &SolverMeshTopology,
) -> Result<BTreeMap<[u64; 2], Vec<SurfaceUse>>, DelaunaySolverTopologyError> {
    let mut uses = BTreeMap::<[u64; 2], BTreeSet<SurfaceUse>>::new();
    for face in &topology.boundary_faces {
        let facet = input
            .volume_mesh
            .provenance
            .facets
            .get(face.face_id as usize - 1)
            .ok_or_else(|| invalid("boundary face has no recovered facet lineage"))?;
        let source_face_id = facet
            .entity_ids
            .iter()
            .find(|entity| entity.kind == PersistentEntityKind::Face)
            .ok_or_else(|| invalid("recovered facet has no exact face"))?
            .clone();
        let use_record = SurfaceUse {
            source_face_id,
            chart_id: facet.chart_id,
        };
        for pair in [[0, 1], [1, 2], [2, 0]] {
            uses.entry(sorted_edge([
                face.node_ids[pair[0]],
                face.node_ids[pair[1]],
            ]))
            .or_default()
            .insert(use_record.clone());
        }
    }
    Ok(uses
        .into_iter()
        .map(|(edge, uses)| (edge, uses.into_iter().collect()))
        .collect())
}

fn boundary_exact_edges(
    topology: &SolverMeshTopology,
) -> Result<BTreeMap<[u64; 2], PersistentEntityId>, DelaunaySolverTopologyError> {
    let mut result = BTreeMap::new();
    for edge in &topology.boundary_edges {
        let exact = edge
            .provenance
            .iter()
            .filter(|entity| entity.kind == PersistentEntityKind::Edge)
            .collect::<Vec<_>>();
        if exact.len() > 1 {
            return Err(invalid(
                "one boundary mesh edge has multiple exact edge owners",
            ));
        }
        if let Some(exact) = exact.first() {
            result.insert(
                sorted_edge([edge.node_ids[0], edge.node_ids[1]]),
                (*exact).clone(),
            );
        }
    }
    Ok(result)
}

fn midpoint_provenance(
    topology: &SolverMeshTopology,
    edge: [u64; 2],
    surface_uses: &[SurfaceUse],
    exact_edge: Option<&PersistentEntityId>,
) -> Result<Vec<PersistentEntityId>, DelaunaySolverTopologyError> {
    let mut provenance = topology
        .volume_elements
        .iter()
        .filter(|element| edge.iter().all(|node| element.node_ids.contains(node)))
        .flat_map(|element| {
            element
                .provenance
                .iter()
                .cloned()
                .chain(std::iter::once(element.region_id.clone()))
        })
        .collect::<BTreeSet<_>>();
    provenance.extend(
        surface_uses
            .iter()
            .map(|use_record| use_record.source_face_id.clone()),
    );
    if let Some(edge) = exact_edge {
        provenance.insert(edge.clone());
    }
    if provenance.is_empty()
        || provenance.len() > super::super::construction::MAX_PROVENANCE_PER_ENTITY
    {
        return Err(invalid(
            "elevated node provenance is empty or exceeds its hard bound",
        ));
    }
    Ok(provenance.into_iter().collect())
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidGeometry, reason)
}

fn resource(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::ResourceLimit, reason)
}
