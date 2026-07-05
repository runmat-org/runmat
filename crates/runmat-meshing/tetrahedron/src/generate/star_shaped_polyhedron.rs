use runmat_meshing_core::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId},
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
    quality::tolerance::MeshingTolerance,
};

use super::convex_polyhedron::bounds::{bounds_span, plc_coordinates_and_bounds, plc_node_average};
use super::evidence::{record_input_plc_evidence, record_tetrahedron_material_evidence};
use super::material::plc_material_region_id;
use super::validation::validate_tetrahedron_generation_plc;
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};

pub fn generate_star_shaped_polyhedron_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    validate_tetrahedron_generation_plc(plc)?;
    if plc.nodes.len() < 4 || plc.facets.len() < 4 {
        return Err(TetrahedronGenerationError::UnsupportedStarShapedPolyhedronPlc);
    }

    let (coordinates_by_id, bounds) = plc_coordinates_and_bounds(plc)?;
    let tolerance = MeshingTolerance::from_bounds(bounds[0], bounds[1]);
    let interior_selection = select_interior_point(plc, &coordinates_by_id, bounds, tolerance)?;
    let interior = interior_selection.point;
    let span = bounds_span(bounds).max(1.0);
    let volume_epsilon = tolerance.volume_epsilon(span);
    let material_region_id = plc_material_region_id(plc);
    let interior_id = TopologyEntityId {
        stage: MeshingStage::TetrahedronMesh,
        id: "star_shaped_polyhedron_interior_0".to_string(),
    };
    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    nodes.push(TetrahedronMeshNode {
        node_id: interior_id.clone(),
        coordinates_m: interior,
    });

    let mut elements = Vec::<Tetrahedron4Element>::with_capacity(plc.facets.len());
    let mut min_scaled_jacobian = f64::INFINITY;
    for (element_index, facet) in plc.facets.iter().enumerate() {
        let mut node_ids = [
            facet.node_ids[0].clone(),
            facet.node_ids[1].clone(),
            facet.node_ids[2].clone(),
            interior_id.clone(),
        ];
        let signed_volume = tetrahedron_signed_volume(node_ids.clone().map(|node_id| {
            if node_id == interior_id {
                interior
            } else {
                coordinates_by_id[&node_id]
            }
        }));
        if signed_volume.abs() <= volume_epsilon {
            return Err(TetrahedronGenerationError::DegenerateStarShapedPolyhedronPlc);
        }
        if signed_volume < 0.0 {
            node_ids.swap(1, 2);
        }
        let points = node_ids.clone().map(|node_id| {
            if node_id == interior_id {
                interior
            } else {
                coordinates_by_id[&node_id]
            }
        });
        min_scaled_jacobian = min_scaled_jacobian.min(tetrahedron_scaled_jacobian(points));
        elements.push(Tetrahedron4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("star_shaped_polyhedron_tetrahedron_{element_index}"),
            },
            node_ids,
            material_region_id: material_region_id.clone(),
        });
    }

    let boundary_faces = plc
        .facets
        .iter()
        .map(|facet| TetrahedronBoundaryFace {
            face_id: facet.facet_id.clone(),
            node_ids: facet.node_ids.clone(),
            source_face_id: facet.source_face_id.clone(),
            source_edge_ids: super::source_edge_ids_for_face_edges(
                &plc.protected_edges,
                facet.node_ids.clone(),
            ),
        })
        .collect::<Vec<_>>();

    let mut evidence = StageEvidence::complete(MeshingStage::TetrahedronMesh);
    evidence
        .entity_counts
        .insert("nodes".to_string(), nodes.len());
    evidence
        .entity_counts
        .insert("interior_nodes".to_string(), 1);
    evidence
        .entity_counts
        .insert("tetrahedron4_elements".to_string(), elements.len());
    evidence
        .entity_counts
        .insert("boundary_faces".to_string(), boundary_faces.len());
    evidence
        .entity_counts
        .insert("plc_boundary_nodes".to_string(), plc.nodes.len());
    evidence.entity_counts.insert(
        "star_shaped_polyhedron_facets".to_string(),
        plc.facets.len(),
    );
    evidence.entity_counts.insert(
        "interior_smoothing_candidate_points".to_string(),
        interior_selection.candidate_count,
    );
    evidence.entity_counts.insert(
        "interior_smoothing_accepted_points".to_string(),
        usize::from(interior_selection.improved),
    );
    record_input_plc_evidence(plc, &mut evidence);
    record_tetrahedron_material_evidence(&elements, &mut evidence);
    evidence.min_scaled_jacobian = Some(min_scaled_jacobian);

    Ok(TetrahedronMesh {
        mesh_id: "star_shaped_polyhedron_tetrahedron_mesh".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct InteriorPointSelection {
    point: [f64; 3],
    min_scaled_jacobian: f64,
    candidate_count: usize,
    improved: bool,
}

fn select_interior_point(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &std::collections::BTreeMap<TopologyEntityId, [f64; 3]>,
    bounds: [[f64; 3]; 2],
    tolerance: MeshingTolerance,
) -> Result<InteriorPointSelection, TetrahedronGenerationError> {
    let span = bounds_span(bounds).max(1.0);
    let volume_epsilon = tolerance.volume_epsilon(span);
    let mut candidates = interior_candidates(plc, bounds, tolerance)?;
    let candidate_count = candidates.len();
    let mut baseline_quality = None::<f64>;
    let mut best = None::<InteriorPointSelection>;

    for candidate in candidates.drain(..) {
        let Ok(min_scaled_jacobian) =
            min_scaled_jacobian_for_interior(plc, coordinates_by_id, candidate, volume_epsilon)
        else {
            continue;
        };
        let baseline = *baseline_quality.get_or_insert(min_scaled_jacobian);
        if best
            .as_ref()
            .is_none_or(|current| min_scaled_jacobian > current.min_scaled_jacobian)
        {
            best = Some(InteriorPointSelection {
                point: candidate,
                min_scaled_jacobian,
                candidate_count,
                improved: min_scaled_jacobian > baseline + 1.0e-12,
            });
        }
    }

    best.ok_or(TetrahedronGenerationError::UnsupportedStarShapedPolyhedronPlc)
}

fn interior_candidates(
    plc: &ProtectedBoundaryComplex,
    bounds: [[f64; 3]; 2],
    tolerance: MeshingTolerance,
) -> Result<Vec<[f64; 3]>, TetrahedronGenerationError> {
    let mut candidates = Vec::<[f64; 3]>::new();
    candidates.push(plc_node_average(plc)?);
    candidates.push(bounds_center(bounds));
    for fraction in [0.25, 0.35, 0.5, 0.65, 0.75] {
        candidates.push([
            lerp(bounds[0][0], bounds[1][0], fraction),
            lerp(bounds[0][1], bounds[1][1], fraction),
            lerp(bounds[0][2], bounds[1][2], fraction),
        ]);
    }
    for x_fraction in [0.35, 0.5, 0.65] {
        for y_fraction in [0.35, 0.5, 0.65] {
            for z_fraction in [0.35, 0.5, 0.65] {
                candidates.push([
                    lerp(bounds[0][0], bounds[1][0], x_fraction),
                    lerp(bounds[0][1], bounds[1][1], y_fraction),
                    lerp(bounds[0][2], bounds[1][2], z_fraction),
                ]);
            }
        }
    }
    candidates.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
            .then_with(|| left[2].total_cmp(&right[2]))
    });
    candidates.dedup_by(|left, right| {
        (0..3).all(|axis| (left[axis] - right[axis]).abs() <= tolerance.absolute_m)
    });
    Ok(candidates)
}

fn min_scaled_jacobian_for_interior(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &std::collections::BTreeMap<TopologyEntityId, [f64; 3]>,
    interior: [f64; 3],
    volume_epsilon: f64,
) -> Result<f64, TetrahedronGenerationError> {
    if interior.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(TetrahedronGenerationError::NonFiniteInteriorPoint);
    }
    let mut orientation_sign = 0_i8;
    let mut min_scaled_jacobian = f64::INFINITY;
    for facet in &plc.facets {
        let mut points = [
            coordinates_by_id[&facet.node_ids[0]],
            coordinates_by_id[&facet.node_ids[1]],
            coordinates_by_id[&facet.node_ids[2]],
            interior,
        ];
        let signed_volume = tetrahedron_signed_volume(points);
        if signed_volume.abs() <= volume_epsilon {
            return Err(TetrahedronGenerationError::DegenerateStarShapedPolyhedronPlc);
        }
        let sign = if signed_volume > 0.0 { 1 } else { -1 };
        if orientation_sign == 0 {
            orientation_sign = sign;
        } else if orientation_sign != sign {
            return Err(TetrahedronGenerationError::UnsupportedStarShapedPolyhedronPlc);
        }
        if signed_volume < 0.0 {
            points.swap(1, 2);
        }
        min_scaled_jacobian = min_scaled_jacobian.min(tetrahedron_scaled_jacobian(points));
    }
    Ok(min_scaled_jacobian)
}

fn bounds_center(bounds: [[f64; 3]; 2]) -> [f64; 3] {
    [
        (bounds[0][0] + bounds[1][0]) * 0.5,
        (bounds[0][1] + bounds[1][1]) * 0.5,
        (bounds[0][2] + bounds[1][2]) * 0.5,
    ]
}

fn lerp(min: f64, max: f64, fraction: f64) -> f64 {
    min + (max - min) * fraction
}
