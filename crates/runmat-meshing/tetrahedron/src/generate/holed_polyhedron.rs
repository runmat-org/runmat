use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId,
        UNCLASSIFIED_MATERIAL_REGION_ID,
    },
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
    quality::tolerance::MeshingTolerance,
};
use runmat_meshing_plc::validate::{
    classify_boundary_components, validate_protected_boundary_complex,
};

use super::convex_polyhedron::bounds::plc_coordinates_and_bounds;
use super::evidence::{record_input_plc_evidence, record_tetrahedron_material_evidence};
use super::material::plc_material_region_id;
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};
use crate::protected_edges::source_edge_ids_for_boundary_face_edges;

pub fn generate_holed_polyhedron_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    validate_protected_boundary_complex(plc)
        .map_err(|error| TetrahedronGenerationError::InvalidProtectedBoundaryComplex { error })?;
    let surface_hole_loop_count = plc
        .evidence
        .entity_counts
        .get("surface_hole_loops")
        .copied()
        .unwrap_or_default();
    let component_report = classify_boundary_components(plc);
    if component_report.component_count != 1 {
        return Err(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc);
    }

    let (coordinates_by_id, bounds) = plc_coordinates_and_bounds(plc)?;
    let tolerance = MeshingTolerance::from_bounds(bounds[0], bounds[1]);
    let material_region_id = plc_material_region_id(plc);
    if material_region_id == UNCLASSIFIED_MATERIAL_REGION_ID {
        return generate_segment_star_holed_polyhedron_mesh(
            plc,
            surface_hole_loop_count,
            &coordinates_by_id,
            bounds,
            tolerance.absolute_m,
            &material_region_id,
        );
    }
    let grid = axis_aligned_rectangular_through_hole_grid(
        plc,
        &coordinates_by_id,
        bounds,
        tolerance.absolute_m,
    )?;
    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    let mut coordinates_by_mesh_node_id = coordinates_by_id.clone();
    let grid_node_ids = holed_polyhedron_grid_nodes(
        &grid,
        &mut nodes,
        &mut coordinates_by_mesh_node_id,
        tolerance.absolute_m,
    )?;

    let mut elements = Vec::<Tetrahedron4Element>::new();
    let mut min_scaled_jacobian = f64::INFINITY;
    append_ring_cell_tetrahedra(
        &grid,
        &grid_node_ids,
        &coordinates_by_mesh_node_id,
        &material_region_id,
        &mut elements,
        &mut min_scaled_jacobian,
    )?;
    let boundary_faces = holed_polyhedron_boundary_faces(
        plc,
        &grid,
        &elements,
        &coordinates_by_mesh_node_id,
        tolerance.absolute_m,
    )?;

    let mut evidence = StageEvidence::complete(MeshingStage::TetrahedronMesh);
    evidence
        .entity_counts
        .insert("nodes".to_string(), nodes.len());
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
        "holed_polyhedron_surface_hole_loops".to_string(),
        surface_hole_loop_count,
    );
    evidence.entity_counts.insert(
        "holed_polyhedron_segments".to_string(),
        grid.ring_cell_count(),
    );
    evidence
        .entity_counts
        .insert("holed_polyhedron_support_nodes".to_string(), 0);
    evidence.entity_counts.insert(
        "holed_polyhedron_ring_cells".to_string(),
        grid.ring_cell_count(),
    );
    record_input_plc_evidence(plc, &mut evidence);
    record_tetrahedron_material_evidence(&elements, &mut evidence);
    evidence.min_scaled_jacobian = Some(min_scaled_jacobian);

    Ok(TetrahedronMesh {
        mesh_id: "holed_polyhedron_tetrahedron_mesh".to_string(),
        tetrahedron_generation_family: "holed_polyhedron".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

#[derive(Debug, Clone)]
struct RectangularThroughHoleGrid {
    source_x_values: [f64; 4],
    source_y_values: [f64; 4],
    source_z_values: [f64; 2],
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    z_values: Vec<f64>,
    x_hole_min_index: usize,
    x_hole_max_index: usize,
    y_hole_min_index: usize,
    y_hole_max_index: usize,
}

impl RectangularThroughHoleGrid {
    fn ring_cell_count(&self) -> usize {
        let x_cell_count = self.x_values.len().saturating_sub(1);
        let y_cell_count = self.y_values.len().saturating_sub(1);
        let z_cell_count = self.z_values.len().saturating_sub(1);
        let hole_cell_count = (self.x_hole_max_index - self.x_hole_min_index)
            * (self.y_hole_max_index - self.y_hole_min_index)
            * z_cell_count;
        x_cell_count * y_cell_count * z_cell_count - hole_cell_count
    }

    fn contains_hole_cell(&self, x_index: usize, y_index: usize) -> bool {
        (self.x_hole_min_index..self.x_hole_max_index).contains(&x_index)
            && (self.y_hole_min_index..self.y_hole_max_index).contains(&y_index)
    }
}

fn axis_aligned_rectangular_through_hole_grid(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    bounds: [[f64; 3]; 2],
    tolerance_m: f64,
) -> Result<RectangularThroughHoleGrid, TetrahedronGenerationError> {
    if plc.nodes.len() != 16 || plc.facets.len() != 32 {
        return Err(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc);
    }
    let x_values = unique_axis_values(plc, 0, tolerance_m);
    let y_values = unique_axis_values(plc, 1, tolerance_m);
    let z_values = unique_axis_values(plc, 2, tolerance_m);
    if x_values.len() != 4 || y_values.len() != 4 || z_values.len() != 2 {
        return Err(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc);
    }
    if !nearly_equal(x_values[0], bounds[0][0], tolerance_m)
        || !nearly_equal(x_values[3], bounds[1][0], tolerance_m)
        || !nearly_equal(y_values[0], bounds[0][1], tolerance_m)
        || !nearly_equal(y_values[3], bounds[1][1], tolerance_m)
        || !nearly_equal(z_values[0], bounds[0][2], tolerance_m)
        || !nearly_equal(z_values[1], bounds[1][2], tolerance_m)
    {
        return Err(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc);
    }
    let node_at = |x, y, z| node_at(coordinates_by_id, [x, y, z], tolerance_m);
    for coordinates in [
        [x_values[0], y_values[0], z_values[0]],
        [x_values[3], y_values[0], z_values[0]],
        [x_values[3], y_values[3], z_values[0]],
        [x_values[0], y_values[3], z_values[0]],
        [x_values[1], y_values[1], z_values[0]],
        [x_values[2], y_values[1], z_values[0]],
        [x_values[2], y_values[2], z_values[0]],
        [x_values[1], y_values[2], z_values[0]],
        [x_values[0], y_values[0], z_values[1]],
        [x_values[3], y_values[0], z_values[1]],
        [x_values[3], y_values[3], z_values[1]],
        [x_values[0], y_values[3], z_values[1]],
        [x_values[1], y_values[1], z_values[1]],
        [x_values[2], y_values[1], z_values[1]],
        [x_values[2], y_values[2], z_values[1]],
        [x_values[1], y_values[2], z_values[1]],
    ] {
        node_at(coordinates[0], coordinates[1], coordinates[2])?;
    }
    let source_x_values = [x_values[0], x_values[1], x_values[2], x_values[3]];
    let source_y_values = [y_values[0], y_values[1], y_values[2], y_values[3]];
    let source_z_values = [z_values[0], z_values[1]];
    let refinement_length =
        smallest_axis_interval(&[&x_values, &y_values, &z_values], tolerance_m)?;
    let x_values = refined_axis_values(&x_values, refinement_length, tolerance_m);
    let y_values = refined_axis_values(&y_values, refinement_length, tolerance_m);
    let z_values = refined_axis_values(&z_values, refinement_length, tolerance_m);
    let x_hole_min_index = refined_axis_index(&x_values, source_x_values[1], tolerance_m)?;
    let x_hole_max_index = refined_axis_index(&x_values, source_x_values[2], tolerance_m)?;
    let y_hole_min_index = refined_axis_index(&y_values, source_y_values[1], tolerance_m)?;
    let y_hole_max_index = refined_axis_index(&y_values, source_y_values[2], tolerance_m)?;
    Ok(RectangularThroughHoleGrid {
        source_x_values,
        source_y_values,
        source_z_values,
        x_values,
        y_values,
        z_values,
        x_hole_min_index,
        x_hole_max_index,
        y_hole_min_index,
        y_hole_max_index,
    })
}

fn holed_polyhedron_grid_nodes(
    grid: &RectangularThroughHoleGrid,
    nodes: &mut Vec<TetrahedronMeshNode>,
    coordinates_by_id: &mut BTreeMap<TopologyEntityId, [f64; 3]>,
    tolerance_m: f64,
) -> Result<Vec<TopologyEntityId>, TetrahedronGenerationError> {
    let mut grid_node_ids = Vec::<TopologyEntityId>::with_capacity(32);
    for z in grid.z_values.iter().copied() {
        for y in grid.y_values.iter().copied() {
            for x in grid.x_values.iter().copied() {
                let coordinates = [x, y, z];
                if let Ok(node_id) = node_at(coordinates_by_id, coordinates, tolerance_m) {
                    grid_node_ids.push(node_id);
                    continue;
                }
                let node_id = TopologyEntityId {
                    stage: MeshingStage::TetrahedronMesh,
                    id: format!("holed_polyhedron_grid_node_{}", nodes.len()),
                };
                nodes.push(TetrahedronMeshNode {
                    node_id: node_id.clone(),
                    coordinates_m: coordinates,
                });
                coordinates_by_id.insert(node_id.clone(), coordinates);
                grid_node_ids.push(node_id);
            }
        }
    }
    Ok(grid_node_ids)
}

fn append_ring_cell_tetrahedra(
    grid: &RectangularThroughHoleGrid,
    grid_node_ids: &[TopologyEntityId],
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    material_region_id: &str,
    elements: &mut Vec<Tetrahedron4Element>,
    min_scaled_jacobian: &mut f64,
) -> Result<(), TetrahedronGenerationError> {
    let tetrahedron_corners = [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ];
    for z_index in 0..grid.z_values.len() - 1 {
        for y_index in 0..grid.y_values.len() - 1 {
            for x_index in 0..grid.x_values.len() - 1 {
                if grid.contains_hole_cell(x_index, y_index) {
                    continue;
                }
                let cell_corner_ids = holed_polyhedron_cell_corner_ids(
                    grid,
                    grid_node_ids,
                    [x_index, y_index, z_index],
                );
                for corners in tetrahedron_corners {
                    let mut node_ids = corners.map(|corner| cell_corner_ids[corner].clone());
                    let points = node_ids.clone().map(|node_id| coordinates_by_id[&node_id]);
                    if tetrahedron_signed_volume(points) < 0.0 {
                        node_ids.swap(1, 2);
                    }
                    let points = node_ids.clone().map(|node_id| coordinates_by_id[&node_id]);
                    let scaled_jacobian = tetrahedron_scaled_jacobian(points);
                    *min_scaled_jacobian = min_scaled_jacobian.min(scaled_jacobian);
                    elements.push(Tetrahedron4Element {
                        element_id: TopologyEntityId {
                            stage: MeshingStage::TetrahedronMesh,
                            id: format!("holed_polyhedron_tetrahedron_{}", elements.len()),
                        },
                        node_ids,
                        material_region_id: material_region_id.to_string(),
                    });
                }
            }
        }
    }
    if elements.len() == grid.ring_cell_count() * 6 && min_scaled_jacobian.is_finite() {
        Ok(())
    } else {
        Err(TetrahedronGenerationError::DegenerateHoledPolyhedronPlc)
    }
}

fn holed_polyhedron_cell_corner_ids(
    grid: &RectangularThroughHoleGrid,
    grid_node_ids: &[TopologyEntityId],
    cell_index: [usize; 3],
) -> [TopologyEntityId; 8] {
    let [x_index, y_index, z_index] = cell_index;
    [
        holed_polyhedron_grid_node_id(grid, grid_node_ids, x_index, y_index, z_index),
        holed_polyhedron_grid_node_id(grid, grid_node_ids, x_index + 1, y_index, z_index),
        holed_polyhedron_grid_node_id(grid, grid_node_ids, x_index, y_index + 1, z_index),
        holed_polyhedron_grid_node_id(grid, grid_node_ids, x_index + 1, y_index + 1, z_index),
        holed_polyhedron_grid_node_id(grid, grid_node_ids, x_index, y_index, z_index + 1),
        holed_polyhedron_grid_node_id(grid, grid_node_ids, x_index + 1, y_index, z_index + 1),
        holed_polyhedron_grid_node_id(grid, grid_node_ids, x_index, y_index + 1, z_index + 1),
        holed_polyhedron_grid_node_id(grid, grid_node_ids, x_index + 1, y_index + 1, z_index + 1),
    ]
}

fn holed_polyhedron_grid_node_id(
    grid: &RectangularThroughHoleGrid,
    grid_node_ids: &[TopologyEntityId],
    x_index: usize,
    y_index: usize,
    z_index: usize,
) -> TopologyEntityId {
    grid_node_ids[z_index * grid.y_values.len() * grid.x_values.len()
        + y_index * grid.x_values.len()
        + x_index]
        .clone()
}

fn holed_polyhedron_boundary_faces(
    plc: &ProtectedBoundaryComplex,
    grid: &RectangularThroughHoleGrid,
    elements: &[Tetrahedron4Element],
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    tolerance_m: f64,
) -> Result<Vec<TetrahedronBoundaryFace>, TetrahedronGenerationError> {
    let source_faces_by_surface =
        source_faces_by_holed_surface(plc, grid, coordinates_by_id, tolerance_m)?;
    let mut face_counts = BTreeMap::<[TopologyEntityId; 3], ([TopologyEntityId; 3], usize)>::new();
    for element in elements {
        for face in tetrahedron_faces(element.node_ids.clone()) {
            face_counts
                .entry(sorted_face(face.clone()))
                .and_modify(|(_, count)| *count += 1)
                .or_insert((face, 1));
        }
    }

    let mut boundary_faces = Vec::<TetrahedronBoundaryFace>::new();
    for (boundary_face_index, (_, (node_ids, count))) in face_counts.into_iter().enumerate() {
        if count != 1 {
            continue;
        }
        let points = node_ids.clone().map(|node_id| coordinates_by_id[&node_id]);
        let surface_key = holed_surface_key(&points, grid, tolerance_m)
            .ok_or(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)?;
        let source_face_id = source_faces_by_surface
            .get(&surface_key)
            .cloned()
            .ok_or(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)?;
        boundary_faces.push(TetrahedronBoundaryFace {
            face_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("holed_polyhedron_boundary_face_{boundary_face_index}"),
            },
            source_edge_ids: source_edge_ids_for_boundary_face_edges(
                &plc.protected_edges,
                coordinates_by_id,
                node_ids.clone(),
                tolerance_m,
            ),
            node_ids,
            source_face_id,
        });
    }
    Ok(boundary_faces)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum HoledSurfaceKey {
    Top,
    Bottom,
    OuterYMin,
    OuterXMax,
    OuterYMax,
    OuterXMin,
    HoleYMin,
    HoleXMax,
    HoleYMax,
    HoleXMin,
}

fn source_faces_by_holed_surface(
    plc: &ProtectedBoundaryComplex,
    grid: &RectangularThroughHoleGrid,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    tolerance_m: f64,
) -> Result<BTreeMap<HoledSurfaceKey, TopologyEntityId>, TetrahedronGenerationError> {
    let mut source_faces_by_surface = BTreeMap::<HoledSurfaceKey, TopologyEntityId>::new();
    for facet in &plc.facets {
        let points = facet.node_ids.clone().map(|node_id| {
            coordinates_by_id.get(&node_id).copied().ok_or_else(|| {
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: node_id.id.clone(),
                }
            })
        });
        let points = points.into_iter().collect::<Result<Vec<_>, _>>()?;
        let points = [points[0], points[1], points[2]];
        let surface_key = holed_surface_key(&points, grid, tolerance_m)
            .ok_or(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)?;
        source_faces_by_surface
            .entry(surface_key)
            .or_insert_with(|| facet.source_face_id.clone());
    }
    Ok(source_faces_by_surface)
}

fn holed_surface_key(
    points: &[[f64; 3]; 3],
    grid: &RectangularThroughHoleGrid,
    tolerance_m: f64,
) -> Option<HoledSurfaceKey> {
    let [x_min, x_inner_min, x_inner_max, x_max] = grid.source_x_values;
    let [y_min, y_inner_min, y_inner_max, y_max] = grid.source_y_values;
    let [z_min, z_max] = grid.source_z_values;
    let all_axis = |axis: usize, value: f64| {
        points
            .iter()
            .all(|point| nearly_equal(point[axis], value, tolerance_m))
    };
    if all_axis(2, z_max) {
        Some(HoledSurfaceKey::Top)
    } else if all_axis(2, z_min) {
        Some(HoledSurfaceKey::Bottom)
    } else if all_axis(1, y_min) {
        Some(HoledSurfaceKey::OuterYMin)
    } else if all_axis(0, x_max) {
        Some(HoledSurfaceKey::OuterXMax)
    } else if all_axis(1, y_max) {
        Some(HoledSurfaceKey::OuterYMax)
    } else if all_axis(0, x_min) {
        Some(HoledSurfaceKey::OuterXMin)
    } else if all_axis(1, y_inner_min) {
        Some(HoledSurfaceKey::HoleYMin)
    } else if all_axis(0, x_inner_max) {
        Some(HoledSurfaceKey::HoleXMax)
    } else if all_axis(1, y_inner_max) {
        Some(HoledSurfaceKey::HoleYMax)
    } else if all_axis(0, x_inner_min) {
        Some(HoledSurfaceKey::HoleXMin)
    } else {
        None
    }
}

fn tetrahedron_faces(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 3]; 4] {
    [
        [
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
        ],
        [
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[3].clone(),
        ],
        [
            node_ids[0].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ],
        [
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ],
    ]
}

fn sorted_face(mut node_ids: [TopologyEntityId; 3]) -> [TopologyEntityId; 3] {
    node_ids.sort();
    node_ids
}

fn generate_segment_star_holed_polyhedron_mesh(
    plc: &ProtectedBoundaryComplex,
    surface_hole_loop_count: usize,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    bounds: [[f64; 3]; 2],
    tolerance_m: f64,
    material_region_id: &str,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    let segments = axis_aligned_rectangular_through_hole_segments(
        plc,
        coordinates_by_id,
        bounds,
        tolerance_m,
    )?;
    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    let mut coordinates_by_mesh_node_id = coordinates_by_id.clone();
    let internal_faces_by_segment = internal_segment_faces(&segments);

    let mut elements = Vec::<Tetrahedron4Element>::new();
    let mut min_scaled_jacobian = f64::INFINITY;
    for (segment_index, segment) in segments.iter().enumerate() {
        let segment_faces = segment_faces(plc, segment, &internal_faces_by_segment[segment_index]);
        let support_node =
            segment_support_node(segment, coordinates_by_id, &segment_faces, segment_index)?;
        coordinates_by_mesh_node_id
            .insert(support_node.node_id.clone(), support_node.coordinates_m);
        nodes.push(support_node.clone());
        append_segment_star_tetrahedra(
            &segment_faces,
            &support_node,
            &coordinates_by_mesh_node_id,
            material_region_id,
            &mut elements,
            &mut min_scaled_jacobian,
        )?;
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
        .insert("tetrahedron4_elements".to_string(), elements.len());
    evidence
        .entity_counts
        .insert("boundary_faces".to_string(), boundary_faces.len());
    evidence
        .entity_counts
        .insert("plc_boundary_nodes".to_string(), plc.nodes.len());
    evidence.entity_counts.insert(
        "holed_polyhedron_surface_hole_loops".to_string(),
        surface_hole_loop_count,
    );
    evidence
        .entity_counts
        .insert("holed_polyhedron_segments".to_string(), segments.len());
    evidence
        .entity_counts
        .insert("holed_polyhedron_support_nodes".to_string(), segments.len());
    record_input_plc_evidence(plc, &mut evidence);
    record_tetrahedron_material_evidence(&elements, &mut evidence);
    evidence.min_scaled_jacobian = Some(min_scaled_jacobian);

    Ok(TetrahedronMesh {
        mesh_id: "holed_polyhedron_tetrahedron_mesh".to_string(),
        tetrahedron_generation_family: "holed_polyhedron".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

#[derive(Debug, Clone)]
struct ThroughHoleSegment {
    local_node_ids: [TopologyEntityId; 8],
}

fn axis_aligned_rectangular_through_hole_segments(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    bounds: [[f64; 3]; 2],
    tolerance_m: f64,
) -> Result<[ThroughHoleSegment; 4], TetrahedronGenerationError> {
    let grid =
        axis_aligned_rectangular_through_hole_grid(plc, coordinates_by_id, bounds, tolerance_m)?;
    let [x_min, x_inner_min, x_inner_max, x_max] = grid.source_x_values;
    let [y_min, y_inner_min, y_inner_max, y_max] = grid.source_y_values;
    let [z_min, z_max] = grid.source_z_values;
    let node_at = |x, y, z| node_at(coordinates_by_id, [x, y, z], tolerance_m);
    Ok([
        ThroughHoleSegment {
            local_node_ids: [
                node_at(x_min, y_min, z_min)?,
                node_at(x_max, y_min, z_min)?,
                node_at(x_inner_max, y_inner_min, z_min)?,
                node_at(x_inner_min, y_inner_min, z_min)?,
                node_at(x_min, y_min, z_max)?,
                node_at(x_max, y_min, z_max)?,
                node_at(x_inner_max, y_inner_min, z_max)?,
                node_at(x_inner_min, y_inner_min, z_max)?,
            ],
        },
        ThroughHoleSegment {
            local_node_ids: [
                node_at(x_max, y_min, z_min)?,
                node_at(x_max, y_max, z_min)?,
                node_at(x_inner_max, y_inner_max, z_min)?,
                node_at(x_inner_max, y_inner_min, z_min)?,
                node_at(x_max, y_min, z_max)?,
                node_at(x_max, y_max, z_max)?,
                node_at(x_inner_max, y_inner_max, z_max)?,
                node_at(x_inner_max, y_inner_min, z_max)?,
            ],
        },
        ThroughHoleSegment {
            local_node_ids: [
                node_at(x_max, y_max, z_min)?,
                node_at(x_min, y_max, z_min)?,
                node_at(x_inner_min, y_inner_max, z_min)?,
                node_at(x_inner_max, y_inner_max, z_min)?,
                node_at(x_max, y_max, z_max)?,
                node_at(x_min, y_max, z_max)?,
                node_at(x_inner_min, y_inner_max, z_max)?,
                node_at(x_inner_max, y_inner_max, z_max)?,
            ],
        },
        ThroughHoleSegment {
            local_node_ids: [
                node_at(x_min, y_max, z_min)?,
                node_at(x_min, y_min, z_min)?,
                node_at(x_inner_min, y_inner_min, z_min)?,
                node_at(x_inner_min, y_inner_max, z_min)?,
                node_at(x_min, y_max, z_max)?,
                node_at(x_min, y_min, z_max)?,
                node_at(x_inner_min, y_inner_min, z_max)?,
                node_at(x_inner_min, y_inner_max, z_max)?,
            ],
        },
    ])
}

fn append_segment_star_tetrahedra(
    faces: &[[TopologyEntityId; 3]],
    support_node: &TetrahedronMeshNode,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    material_region_id: &str,
    elements: &mut Vec<Tetrahedron4Element>,
    min_scaled_jacobian: &mut f64,
) -> Result<(), TetrahedronGenerationError> {
    for face in faces {
        let scaled_jacobian = append_positive_tetrahedron(
            face,
            support_node,
            coordinates_by_id,
            material_region_id,
            elements,
        )?;
        *min_scaled_jacobian = min_scaled_jacobian.min(scaled_jacobian);
    }
    Ok(())
}

fn segment_faces(
    plc: &ProtectedBoundaryComplex,
    segment: &ThroughHoleSegment,
    internal_faces: &[[TopologyEntityId; 3]],
) -> Vec<[TopologyEntityId; 3]> {
    let segment_nodes = segment
        .local_node_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut boundary_faces = plc
        .facets
        .iter()
        .filter_map(|facet| {
            facet
                .node_ids
                .iter()
                .all(|node_id| segment_nodes.contains(node_id))
                .then_some(facet.node_ids.clone())
        })
        .collect::<Vec<_>>();
    boundary_faces.extend(internal_faces.iter().cloned());
    boundary_faces
}

fn append_positive_tetrahedron(
    face: &[TopologyEntityId; 3],
    support_node: &TetrahedronMeshNode,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    material_region_id: &str,
    elements: &mut Vec<Tetrahedron4Element>,
) -> Result<f64, TetrahedronGenerationError> {
    let mut node_ids = [
        face[0].clone(),
        face[1].clone(),
        face[2].clone(),
        support_node.node_id.clone(),
    ];
    let mut points = node_ids.clone().map(|node_id| coordinates_by_id[&node_id]);
    if tetrahedron_signed_volume(points).abs() <= f64::EPSILON {
        return Err(TetrahedronGenerationError::DegenerateHoledPolyhedronPlc);
    }
    if tetrahedron_signed_volume(points) < 0.0 {
        node_ids.swap(1, 2);
        points.swap(1, 2);
    }
    let scaled_jacobian = tetrahedron_scaled_jacobian(points);
    elements.push(Tetrahedron4Element {
        element_id: TopologyEntityId {
            stage: MeshingStage::TetrahedronMesh,
            id: format!("holed_polyhedron_tetrahedron_{}", elements.len()),
        },
        node_ids,
        material_region_id: material_region_id.to_string(),
    });
    Ok(scaled_jacobian)
}

fn segment_support_node(
    segment: &ThroughHoleSegment,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    candidate_faces: &[[TopologyEntityId; 3]],
    segment_index: usize,
) -> Result<TetrahedronMeshNode, TetrahedronGenerationError> {
    let mut best_candidate = None::<([f64; 3], f64)>;
    for candidate in segment_support_candidates(segment, coordinates_by_id) {
        let candidate_score =
            segment_support_candidate_score(&candidate, candidate_faces, coordinates_by_id);
        if best_candidate
            .as_ref()
            .is_none_or(|(_, best_score)| candidate_score > *best_score)
        {
            best_candidate = Some((candidate, candidate_score));
        }
    }
    let coordinates_m = best_candidate
        .filter(|(_, score)| *score > 0.0)
        .map(|(candidate, _)| candidate)
        .ok_or(TetrahedronGenerationError::DegenerateHoledPolyhedronPlc)?;
    Ok(TetrahedronMeshNode {
        node_id: TopologyEntityId {
            stage: MeshingStage::TetrahedronMesh,
            id: format!("holed_polyhedron_segment_support_{segment_index}"),
        },
        coordinates_m,
    })
}

fn segment_support_candidates(
    segment: &ThroughHoleSegment,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Vec<[f64; 3]> {
    let bottom_quad =
        [0_usize, 1, 2, 3].map(|index| coordinates_by_id[&segment.local_node_ids[index]]);
    let z_mid = 0.5
        * (coordinates_by_id[&segment.local_node_ids[0]][2]
            + coordinates_by_id[&segment.local_node_ids[4]][2]);
    let mut candidates = vec![segment_centroid(segment, coordinates_by_id)];
    for u_index in 2..=18 {
        for v_index in 2..=18 {
            let u = u_index as f64 / 20.0;
            let v = v_index as f64 / 20.0;
            let mut candidate = bilinear_quad_point(bottom_quad, u, v);
            candidate[2] = z_mid;
            candidates.push(candidate);
        }
    }
    candidates
}

fn segment_centroid(
    segment: &ThroughHoleSegment,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> [f64; 3] {
    let mut centroid = [0.0_f64; 3];
    for node_id in &segment.local_node_ids {
        let point = coordinates_by_id[node_id];
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    for coordinate in &mut centroid {
        *coordinate /= segment.local_node_ids.len() as f64;
    }
    centroid
}

fn bilinear_quad_point(quad: [[f64; 3]; 4], u: f64, v: f64) -> [f64; 3] {
    let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
    let mut point = [0.0_f64; 3];
    for (index, weight) in weights.into_iter().enumerate() {
        point[0] += weight * quad[index][0];
        point[1] += weight * quad[index][1];
        point[2] += weight * quad[index][2];
    }
    point
}

fn segment_support_candidate_score(
    candidate: &[f64; 3],
    faces: &[[TopologyEntityId; 3]],
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> f64 {
    faces
        .iter()
        .map(|face| {
            let points = [
                coordinates_by_id[&face[0]],
                coordinates_by_id[&face[1]],
                coordinates_by_id[&face[2]],
                *candidate,
            ];
            if tetrahedron_signed_volume(points).abs() <= f64::EPSILON {
                0.0
            } else {
                tetrahedron_scaled_jacobian(points)
            }
        })
        .fold(f64::INFINITY, f64::min)
}

fn internal_segment_faces(segments: &[ThroughHoleSegment; 4]) -> [Vec<[TopologyEntityId; 3]>; 4] {
    let mut faces = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (left_segment, right_segment, quad) in [
        (0_usize, 1_usize, shared_quad(&segments[0], [5, 1, 2, 6])),
        (1, 2, shared_quad(&segments[1], [5, 1, 2, 6])),
        (2, 3, shared_quad(&segments[2], [5, 1, 2, 6])),
        (3, 0, shared_quad(&segments[3], [5, 1, 2, 6])),
    ] {
        let triangles = [
            [quad[0].clone(), quad[1].clone(), quad[2].clone()],
            [quad[0].clone(), quad[2].clone(), quad[3].clone()],
        ];
        faces[left_segment].extend(triangles.clone());
        faces[right_segment].extend(triangles);
    }
    faces
}

fn shared_quad(segment: &ThroughHoleSegment, indices: [usize; 4]) -> [TopologyEntityId; 4] {
    indices.map(|index| segment.local_node_ids[index].clone())
}

fn unique_axis_values(plc: &ProtectedBoundaryComplex, axis: usize, tolerance_m: f64) -> Vec<f64> {
    let mut values = Vec::<f64>::new();
    for node in &plc.nodes {
        let value = node.coordinates_m[axis];
        if !values
            .iter()
            .any(|existing| nearly_equal(*existing, value, tolerance_m))
        {
            values.push(value);
        }
    }
    values.sort_by(|left, right| left.total_cmp(right));
    values
}

fn smallest_axis_interval(
    axis_values: &[&Vec<f64>],
    tolerance_m: f64,
) -> Result<f64, TetrahedronGenerationError> {
    let mut smallest_interval = f64::INFINITY;
    for values in axis_values {
        for window in values.windows(2) {
            let interval = (window[1] - window[0]).abs();
            if interval > tolerance_m.max(1.0e-12) {
                smallest_interval = smallest_interval.min(interval);
            }
        }
    }
    if smallest_interval.is_finite() {
        Ok(smallest_interval)
    } else {
        Err(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)
    }
}

fn refined_axis_values(source_values: &[f64], target_interval: f64, tolerance_m: f64) -> Vec<f64> {
    let mut values = Vec::<f64>::new();
    values.push(source_values[0]);
    for window in source_values.windows(2) {
        let start = window[0];
        let end = window[1];
        let interval = (end - start).abs();
        let partition_count = (interval / target_interval).ceil().max(1.0) as usize;
        for partition_index in 1..=partition_count {
            let fraction = partition_index as f64 / partition_count as f64;
            let value = start + fraction * (end - start);
            if values
                .last()
                .is_none_or(|existing| !nearly_equal(*existing, value, tolerance_m))
            {
                values.push(value);
            }
        }
    }
    values
}

fn refined_axis_index(
    values: &[f64],
    source_value: f64,
    tolerance_m: f64,
) -> Result<usize, TetrahedronGenerationError> {
    values
        .iter()
        .position(|value| nearly_equal(*value, source_value, tolerance_m))
        .ok_or(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)
}

fn node_at(
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    coordinates_m: [f64; 3],
    tolerance_m: f64,
) -> Result<TopologyEntityId, TetrahedronGenerationError> {
    coordinates_by_id
        .iter()
        .find_map(|(node_id, point)| {
            (nearly_equal(point[0], coordinates_m[0], tolerance_m)
                && nearly_equal(point[1], coordinates_m[1], tolerance_m)
                && nearly_equal(point[2], coordinates_m[2], tolerance_m))
            .then(|| node_id.clone())
        })
        .ok_or(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)
}

fn nearly_equal(left: f64, right: f64, tolerance_m: f64) -> bool {
    (left - right).abs() <= tolerance_m.max(1.0e-12)
}
