use super::control::{checkpoint, validate_curves};
use runmat_meshing_core::MeshingCancellationSignal;
use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactWire, PersistentEntityId, TopologicalOrientation,
};
use runmat_meshing_curve::{SharedCurve, SharedCurveFaceUse, SharedCurveMesh};

use super::{
    validate::validate_exact_surface_boundary_controlled, ExactFaceBoundary, ExactFaceBoundaryLoop,
    ExactFaceBoundarySegment, ExactSurfaceBoundary, ExactSurfaceBoundaryError,
    ExactSurfaceBoundaryErrorKind, EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION,
};

pub(super) fn build_exact_surface_boundary_controlled(
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<ExactSurfaceBoundary, ExactSurfaceBoundaryError> {
    validate_curves(curves, topology, control)?;
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
    let curve_by_edge = curves
        .edges
        .iter()
        .map(|curve| {
            checkpoint(control)?;
            Ok((&curve.source_edge_id, curve))
        })
        .collect::<Result<BTreeMap<_, _>, ExactSurfaceBoundaryError>>()?;
    let mut faces = Vec::with_capacity(topology.faces.len());
    for face in &topology.faces {
        checkpoint(control)?;
        let outer_loop = project_loop(
            &face.outer_wire_id,
            &wires,
            &coedges,
            &curve_by_edge,
            control,
        )?;
        let inner_loops = face
            .inner_wire_ids
            .iter()
            .map(|wire_id| project_loop(wire_id, &wires, &coedges, &curve_by_edge, control))
            .collect::<Result<Vec<_>, _>>()?;
        faces.push(ExactFaceBoundary {
            source_face_id: face.id.clone(),
            outer_loop,
            inner_loops,
        });
    }
    let boundary = ExactSurfaceBoundary {
        schema_version: EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION,
        faces,
    };
    validate_exact_surface_boundary_controlled(&boundary, topology, curves, control)?;
    Ok(boundary)
}

fn project_loop(
    wire_id: &PersistentEntityId,
    wires: &BTreeMap<&PersistentEntityId, &ExactWire>,
    coedges: &BTreeMap<&PersistentEntityId, &runmat_geometry_core::ExactCoedge>,
    curves: &BTreeMap<&PersistentEntityId, &SharedCurve>,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<ExactFaceBoundaryLoop, ExactSurfaceBoundaryError> {
    let wire = wires.get(wire_id).ok_or_else(|| missing(wire_id, "wire"))?;
    let mut segments = Vec::new();
    for coedge_id in &wire.coedge_ids {
        checkpoint(control)?;
        let coedge = coedges
            .get(coedge_id)
            .ok_or_else(|| missing(coedge_id, "coedge"))?;
        let curve = curves
            .get(&coedge.edge_id)
            .ok_or_else(|| missing(&coedge.edge_id, "shared curve"))?;
        let face_use = curve
            .face_uses
            .iter()
            .find(|face_use| face_use.coedge_id == coedge.id)
            .ok_or_else(|| missing(&coedge.id, "shared curve face use"))?;
        push_segments(&mut segments, coedge, curve, face_use, control)?;
    }
    Ok(ExactFaceBoundaryLoop {
        source_wire_id: wire.id.clone(),
        orientation: wire.orientation,
        segments,
    })
}

fn push_segments(
    output: &mut Vec<ExactFaceBoundarySegment>,
    coedge: &runmat_geometry_core::ExactCoedge,
    curve: &SharedCurve,
    face_use: &SharedCurveFaceUse,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceBoundaryError> {
    let indices: Box<dyn Iterator<Item = usize>> = match coedge.orientation {
        TopologicalOrientation::Forward => Box::new(0..curve.nodes.len() - 1),
        TopologicalOrientation::Reversed => Box::new((0..curve.nodes.len() - 1).rev()),
    };
    for index in indices {
        checkpoint(control)?;
        let pair = match coedge.orientation {
            TopologicalOrientation::Forward => [index, index + 1],
            TopologicalOrientation::Reversed => [index + 1, index],
        };
        output.push(ExactFaceBoundarySegment {
            source_coedge_id: coedge.id.clone(),
            source_edge_id: curve.source_edge_id.clone(),
            seam_image: face_use.seam_image,
            node_ids: pair.map(|index| curve.nodes[index].node_id),
            edge_parameters: pair.map(|index| curve.nodes[index].parameter),
            node_uv: pair.map(|index| face_use.node_uv[index]),
        });
    }
    Ok(())
}

fn missing(entity_id: &PersistentEntityId, dependency: &str) -> ExactSurfaceBoundaryError {
    ExactSurfaceBoundaryError::new(
        ExactSurfaceBoundaryErrorKind::MissingTopology,
        Some(entity_id.clone()),
        format!("missing {dependency}"),
    )
}

/// Build or independently admit exact boundary incidence without changing its coordinates.
pub fn build_exact_surface_boundary(
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<ExactSurfaceBoundary, ExactSurfaceBoundaryError> {
    build_exact_surface_boundary_controlled(topology, curves, None)
}

/// Cooperative counterpart with identical boundary and incidence rules.
pub fn build_exact_surface_boundary_with_cancellation(
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    control: &dyn MeshingCancellationSignal,
) -> Result<ExactSurfaceBoundary, ExactSurfaceBoundaryError> {
    build_exact_surface_boundary_controlled(topology, curves, Some(control))
}
