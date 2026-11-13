//! Reuse admitted reference buffers for per-node incidence checks. No per-point
//! trees, owned identity copies or quadratic duplicate scans are required.
use super::*;
use crate::{ExactFaceMeshEdgeParameter, ExactFaceMeshNodeUse};
use runmat_geometry_core::PersistentEntityKind;
use std::cmp::Ordering;

fn compare_uses(left: &ExactFaceMeshNodeUse, right: &ExactFaceMeshNodeUse) -> Ordering {
    left.source_face_id
        .cmp(&right.source_face_id)
        .then_with(|| left.chart_id.cmp(&right.chart_id))
        .then_with(|| left.uv.map(f64::to_bits).cmp(&right.uv.map(f64::to_bits)))
        .then_with(|| {
            left.evaluator_uv
                .map(f64::to_bits)
                .cmp(&right.evaluator_uv.map(f64::to_bits))
        })
}
fn compare_parameters(
    left: &ExactFaceMeshEdgeParameter,
    right: &ExactFaceMeshEdgeParameter,
) -> Ordering {
    left.source_coedge_id
        .cmp(&right.source_coedge_id)
        .then_with(|| left.source_edge_id.cmp(&right.source_edge_id))
        .then_with(|| left.parameter.to_bits().cmp(&right.parameter.to_bits()))
}

pub(super) fn validate(
    mesh: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    control: Option<&dyn MeshingCancellationSignal>,
    allocations: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<(), ExactSurfaceMeshError> {
    let mut use_count = 0;
    let mut parameter_count = 0;
    for node in &mesh.nodes {
        checkpoint(control)?;
        use_count = use_count.max(node.uses.len());
        for usage in &node.uses {
            checkpoint(control)?;
            parameter_count = parameter_count.max(usage.exact_edge_parameters.len());
        }
    }
    let _uses_storage =
        super::super::join::nodes::reserve::<&ExactFaceMeshNodeUse>(use_count, allocations)?;
    let _parameters_storage = super::super::join::nodes::reserve::<&ExactFaceMeshEdgeParameter>(
        parameter_count,
        allocations,
    )?;
    let mut uses = Vec::with_capacity(use_count);
    let mut parameters = Vec::with_capacity(parameter_count);
    for node in &mesh.nodes {
        checkpoint(control)?;
        uses.clear();
        for usage in &node.uses {
            checkpoint(control)?;
            if mesh.face_ids.binary_search(&usage.source_face_id).is_err()
                || usage.chart_id == StableDigest::ZERO
                || usage
                    .uv
                    .iter()
                    .chain(&usage.evaluator_uv)
                    .any(|value| !value.is_finite())
            {
                return Err(invalid("published surface node use is invalid"));
            }
            uses.push(usage);
            parameters.clear();
            for parameter in &usage.exact_edge_parameters {
                checkpoint(control)?;
                let coedge = topology
                    .coedges
                    .binary_search_by(|coedge| coedge.id.cmp(&parameter.source_coedge_id))
                    .ok()
                    .map(|index| &topology.coedges[index]);
                if parameter.source_coedge_id.kind != PersistentEntityKind::Coedge
                    || parameter.source_edge_id.kind != PersistentEntityKind::Edge
                    || coedge.is_none_or(|coedge| {
                        coedge.face_id != usage.source_face_id
                            || coedge.edge_id != parameter.source_edge_id
                    })
                    || !parameter.parameter.is_finite()
                {
                    return Err(invalid("published surface exact-edge parameter is invalid"));
                }
                parameters.push(parameter);
            }
            runmat_cpu_execution::sort_in_place_with_checkpoint(
                &mut parameters,
                |left, right| compare_parameters(left, right),
                || checkpoint(control),
            )?;
            if parameters
                .windows(2)
                .any(|pair| compare_parameters(pair[0], pair[1]).is_eq())
            {
                return Err(invalid("published surface exact-edge parameter is invalid"));
            }
        }
        runmat_cpu_execution::sort_in_place_with_checkpoint(
            &mut uses,
            |left, right| compare_uses(left, right),
            || checkpoint(control),
        )?;
        if uses
            .windows(2)
            .any(|pair| compare_uses(pair[0], pair[1]).is_eq())
        {
            return Err(invalid("published surface node use is invalid"));
        }
    }
    Ok(())
}
