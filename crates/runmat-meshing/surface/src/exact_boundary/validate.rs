use super::control::{checkpoint, validate_curves};
use runmat_meshing_core::MeshingCancellationSignal;
use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactWire, PersistentEntityId, TopologicalOrientation,
};
use runmat_meshing_curve::{SharedCurve, SharedCurveMesh};

use super::{
    crossings::validate_face_segment_intersections_controlled, ExactFaceBoundaryLoop,
    ExactSurfaceBoundary, ExactSurfaceBoundaryError, ExactSurfaceBoundaryErrorKind,
    EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION,
};

const MAX_BOUNDARY_SEGMENTS: usize = 100_000_000;

pub(super) fn validate_exact_surface_boundary_controlled(
    boundary: &ExactSurfaceBoundary,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceBoundaryError> {
    validate_curves(curves, topology, control)?;
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
        .map(|wire| {
            checkpoint(control)?;
            Ok((&wire.id, wire))
        })
        .collect::<Result<BTreeMap<_, _>, ExactSurfaceBoundaryError>>()?;
    let coedges = topology
        .coedges
        .iter()
        .map(|coedge| {
            checkpoint(control)?;
            Ok((&coedge.id, coedge))
        })
        .collect::<Result<BTreeMap<_, _>, ExactSurfaceBoundaryError>>()?;
    let curves = curves
        .edges
        .iter()
        .map(|curve| {
            checkpoint(control)?;
            Ok((&curve.source_edge_id, curve))
        })
        .collect::<Result<BTreeMap<_, _>, ExactSurfaceBoundaryError>>()?;
    let mut segment_count = 0usize;
    for (actual, face) in boundary.faces.iter().zip(&topology.faces) {
        checkpoint(control)?;
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
            validate_loop(loop_boundary, &wires, &coedges, &curves, control)?;
            segment_count = segment_count.saturating_add(loop_boundary.segments.len());
        }
        validate_face_segment_intersections_controlled(
            actual,
            [face.periodic_u, face.periodic_v],
            control,
        )?;
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
    control: Option<&dyn MeshingCancellationSignal>,
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
        checkpoint(control)?;
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
            checkpoint(control)?;
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
    for (left, right) in actual
        .segments
        .iter()
        .zip(actual.segments.iter().cycle().skip(1))
        .take(actual.segments.len())
    {
        checkpoint(control)?;
        if left.node_ids[1] != right.node_ids[0] {
            return Err(invalid(
                Some(&wire.id),
                "wire segments do not form one identity-connected cycle",
            ));
        }
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

/// Build or independently admit exact boundary incidence without changing its coordinates.
pub fn validate_exact_surface_boundary(
    boundary: &ExactSurfaceBoundary,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<(), ExactSurfaceBoundaryError> {
    validate_exact_surface_boundary_controlled(boundary, topology, curves, None)
}

/// Cooperative counterpart with identical boundary and incidence rules.
pub fn validate_exact_surface_boundary_with_cancellation(
    boundary: &ExactSurfaceBoundary,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    control: &dyn MeshingCancellationSignal,
) -> Result<(), ExactSurfaceBoundaryError> {
    validate_exact_surface_boundary_controlled(boundary, topology, curves, Some(control))
}
