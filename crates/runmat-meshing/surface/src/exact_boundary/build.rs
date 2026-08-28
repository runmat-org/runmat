use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactWire, PersistentEntityId, TopologicalOrientation,
};
use runmat_meshing_curve::{SharedCurve, SharedCurveFaceUse, SharedCurveMesh};

use super::{
    validate_exact_surface_boundary, ExactFaceBoundary, ExactFaceBoundaryLoop,
    ExactFaceBoundarySegment, ExactSurfaceBoundary, ExactSurfaceBoundaryError,
    ExactSurfaceBoundaryErrorKind, EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION,
};

pub fn build_exact_surface_boundary(
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<ExactSurfaceBoundary, ExactSurfaceBoundaryError> {
    curves.validate_against(topology).map_err(|error| {
        ExactSurfaceBoundaryError::new(
            ExactSurfaceBoundaryErrorKind::InvalidCurveInput,
            error.edge_id.clone(),
            error.to_string(),
        )
    })?;
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
    let curve_by_edge = curves
        .edges
        .iter()
        .map(|curve| (&curve.source_edge_id, curve))
        .collect::<BTreeMap<_, _>>();
    let mut faces = Vec::with_capacity(topology.faces.len());
    for face in &topology.faces {
        let outer_loop = project_loop(&face.outer_wire_id, &wires, &coedges, &curve_by_edge)?;
        let inner_loops = face
            .inner_wire_ids
            .iter()
            .map(|wire_id| project_loop(wire_id, &wires, &coedges, &curve_by_edge))
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
    validate_exact_surface_boundary(&boundary, topology, curves)?;
    Ok(boundary)
}

fn project_loop(
    wire_id: &PersistentEntityId,
    wires: &BTreeMap<&PersistentEntityId, &ExactWire>,
    coedges: &BTreeMap<&PersistentEntityId, &runmat_geometry_core::ExactCoedge>,
    curves: &BTreeMap<&PersistentEntityId, &SharedCurve>,
) -> Result<ExactFaceBoundaryLoop, ExactSurfaceBoundaryError> {
    let wire = wires.get(wire_id).ok_or_else(|| missing(wire_id, "wire"))?;
    let mut segments = Vec::new();
    for coedge_id in &wire.coedge_ids {
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
        push_segments(&mut segments, coedge, curve, face_use);
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
) {
    let indices: Box<dyn Iterator<Item = usize>> = match coedge.orientation {
        TopologicalOrientation::Forward => Box::new(0..curve.nodes.len() - 1),
        TopologicalOrientation::Reversed => Box::new((0..curve.nodes.len() - 1).rev()),
    };
    for index in indices {
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
}

fn missing(entity_id: &PersistentEntityId, dependency: &str) -> ExactSurfaceBoundaryError {
    ExactSurfaceBoundaryError::new(
        ExactSurfaceBoundaryErrorKind::MissingTopology,
        Some(entity_id.clone()),
        format!("missing {dependency}"),
    )
}
