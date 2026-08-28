use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactWire, PersistentEntityId, TopologicalOrientation,
};
use runmat_meshing_curve::{SharedCurve, SharedCurveMesh};

use super::{
    crossings::validate_face_segment_intersections, ExactFaceBoundaryLoop, ExactSurfaceBoundary,
    ExactSurfaceBoundaryError, ExactSurfaceBoundaryErrorKind,
    EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION,
};

const MAX_BOUNDARY_SEGMENTS: usize = 100_000_000;

pub fn validate_exact_surface_boundary(
    boundary: &ExactSurfaceBoundary,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<(), ExactSurfaceBoundaryError> {
    curves.validate_against(topology).map_err(|error| {
        ExactSurfaceBoundaryError::new(
            ExactSurfaceBoundaryErrorKind::InvalidCurveInput,
            error.edge_id.clone(),
            error.to_string(),
        )
    })?;
    if boundary.schema_version != EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION
        || boundary.faces.len() != topology.faces.len()
    {
        return Err(invalid(
            None,
            "schema or face inventory differs from exact topology",
        ));
    }
    let wires = topology
        .wires
        .iter()
        .map(|wire| (&wire.id, wire))
        .collect::<BTreeMap<_, _>>();
    let coedges = topology
        .coedges
        .iter()
        .map(|coedge| (&coedge.id, coedge))
        .collect::<BTreeMap<_, _>>();
    let curves = curves
        .edges
        .iter()
        .map(|curve| (&curve.source_edge_id, curve))
        .collect::<BTreeMap<_, _>>();
    let mut segment_count = 0usize;
    for (actual, face) in boundary.faces.iter().zip(&topology.faces) {
        if actual.source_face_id != face.id
            || actual.outer_loop.source_wire_id != face.outer_wire_id
            || actual.inner_loops.len() != face.inner_wire_ids.len()
            || actual
                .inner_loops
                .iter()
                .zip(&face.inner_wire_ids)
                .any(|(actual, expected)| &actual.source_wire_id != expected)
        {
            return Err(invalid(
                Some(&face.id),
                "face and wire ownership is not canonical",
            ));
        }
        for loop_boundary in std::iter::once(&actual.outer_loop).chain(&actual.inner_loops) {
            validate_loop(loop_boundary, &wires, &coedges, &curves)?;
            segment_count = segment_count.saturating_add(loop_boundary.segments.len());
        }
        validate_face_segment_intersections(actual)?;
    }
    if segment_count > MAX_BOUNDARY_SEGMENTS {
        return Err(ExactSurfaceBoundaryError::new(
            ExactSurfaceBoundaryErrorKind::ResourceLimit,
            None,
            "surface boundary exceeds its hard segment bound",
        ));
    }
    Ok(())
}

fn validate_loop(
    actual: &ExactFaceBoundaryLoop,
    wires: &BTreeMap<&PersistentEntityId, &ExactWire>,
    coedges: &BTreeMap<&PersistentEntityId, &runmat_geometry_core::ExactCoedge>,
    curves: &BTreeMap<&PersistentEntityId, &SharedCurve>,
) -> Result<(), ExactSurfaceBoundaryError> {
    let wire = wires
        .get(&actual.source_wire_id)
        .ok_or_else(|| invalid(Some(&actual.source_wire_id), "source wire is absent"))?;
    if actual.orientation != wire.orientation || actual.segments.is_empty() {
        return Err(invalid(
            Some(&wire.id),
            "wire orientation or segment inventory is invalid",
        ));
    }
    let mut offset = 0usize;
    for coedge_id in &wire.coedge_ids {
        let coedge = coedges
            .get(coedge_id)
            .ok_or_else(|| invalid(Some(coedge_id), "source coedge is absent"))?;
        let curve = curves
            .get(&coedge.edge_id)
            .ok_or_else(|| invalid(Some(&coedge.edge_id), "source curve is absent"))?;
        let face_use = curve
            .face_uses
            .iter()
            .find(|face_use| face_use.coedge_id == coedge.id)
            .ok_or_else(|| invalid(Some(&coedge.id), "curve face use is absent"))?;
        let count = curve.nodes.len() - 1;
        let Some(actual_segments) = actual.segments.get(offset..offset + count) else {
            return Err(invalid(
                Some(&wire.id),
                "wire segment coverage is incomplete",
            ));
        };
        for (local, segment) in actual_segments.iter().enumerate() {
            let index = match coedge.orientation {
                TopologicalOrientation::Forward => local,
                TopologicalOrientation::Reversed => count - 1 - local,
            };
            let pair = match coedge.orientation {
                TopologicalOrientation::Forward => [index, index + 1],
                TopologicalOrientation::Reversed => [index + 1, index],
            };
            if segment.source_coedge_id != coedge.id
                || segment.source_edge_id != coedge.edge_id
                || segment.seam_image != face_use.seam_image
                || segment.node_ids != pair.map(|index| curve.nodes[index].node_id)
                || segment.edge_parameters != pair.map(|index| curve.nodes[index].parameter)
                || segment.node_uv != pair.map(|index| face_use.node_uv[index])
                || segment
                    .edge_parameters
                    .iter()
                    .any(|value| !value.is_finite())
                || segment
                    .node_uv
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
            {
                return Err(invalid(
                    Some(&coedge.id),
                    "boundary segment differs from exact incidence",
                ));
            }
        }
        offset += count;
    }
    if offset != actual.segments.len() {
        return Err(invalid(
            Some(&wire.id),
            "wire contains unowned boundary segments",
        ));
    }
    if actual
        .segments
        .iter()
        .zip(actual.segments.iter().cycle().skip(1))
        .take(actual.segments.len())
        .any(|(left, right)| left.node_ids[1] != right.node_ids[0])
    {
        return Err(invalid(
            Some(&wire.id),
            "wire segments do not form one identity-connected cycle",
        ));
    }
    Ok(())
}

fn invalid(
    entity_id: Option<&PersistentEntityId>,
    reason: impl Into<String>,
) -> ExactSurfaceBoundaryError {
    ExactSurfaceBoundaryError::new(
        ExactSurfaceBoundaryErrorKind::InvalidContract,
        entity_id.cloned(),
        reason,
    )
}
