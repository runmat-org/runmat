use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId,
        DEFAULT_MATERIAL_REGION_ID,
    },
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
    quality::tolerance::MeshingTolerance,
};
use runmat_meshing_plc::validate::{
    classify_boundary_components, validate_protected_boundary_complex,
};

use super::convex_polyhedron::bounds::plc_coordinates_and_bounds;
use super::evidence::{record_input_plc_evidence, record_tetrahedron_material_evidence};
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};
use crate::protected_edges::source_edge_ids_for_boundary_face_edges;

const HOLED_POLYHEDRON_MAX_REFINED_CELL_ASPECT: f64 = 2.0;

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
    let grid = axis_aligned_rectangular_through_hole_grid(
        plc,
        &coordinates_by_id,
        bounds,
        tolerance.absolute_m,
    )?;
    let material_classifier =
        holed_polyhedron_material_classifier(plc, &grid, &coordinates_by_id, tolerance.absolute_m)?;
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
        &material_classifier,
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

    fn cell_center(&self, cell_index: [usize; 3]) -> [f64; 3] {
        let [x_index, y_index, z_index] = cell_index;
        [
            0.5 * (self.x_values[x_index] + self.x_values[x_index + 1]),
            0.5 * (self.y_values[y_index] + self.y_values[y_index + 1]),
            0.5 * (self.z_values[z_index] + self.z_values[z_index + 1]),
        ]
    }
}

fn axis_aligned_rectangular_through_hole_grid(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    bounds: [[f64; 3]; 2],
    tolerance_m: f64,
) -> Result<RectangularThroughHoleGrid, TetrahedronGenerationError> {
    if plc.nodes.len() != 16 || plc.facets.is_empty() {
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
        holed_polyhedron_refinement_length(&[&x_values, &y_values, &z_values], tolerance_m)?;
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
    material_classifier: &HoledPolyhedronMaterialClassifier,
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
                let material_region_id = material_classifier.material_region_for_cell(
                    grid.cell_center([x_index, y_index, z_index]),
                    grid,
                )?;
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
                        material_region_id: material_region_id.clone(),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum HoledRingRegionKey {
    Bottom,
    Right,
    Top,
    Left,
}

#[derive(Debug, Clone)]
struct HoledPolyhedronMaterialClassifier {
    default_material_region_id: Option<String>,
    facets: Vec<HoledMaterialFacet>,
    tolerance_m: f64,
}

impl HoledPolyhedronMaterialClassifier {
    fn material_region_for_cell(
        &self,
        cell_center: [f64; 3],
        grid: &RectangularThroughHoleGrid,
    ) -> Result<String, TetrahedronGenerationError> {
        if let Some(material_region_id) = &self.default_material_region_id {
            return Ok(material_region_id.clone());
        }
        let ring_region = holed_ring_region_key(cell_center, grid, 0.0)
            .ok_or(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)?;
        let surface_key = holed_ring_region_outer_surface_key(ring_region);
        let boundary_point = holed_ring_region_outer_boundary_point(cell_center, grid, ring_region);
        self.facets
            .iter()
            .find(|facet| {
                facet.surface_key == surface_key
                    && point_lies_in_triangle(boundary_point, facet.points, self.tolerance_m)
            })
            .map(|facet| facet.material_region_id.clone())
            .ok_or(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)
    }
}

#[derive(Debug, Clone)]
struct HoledMaterialFacet {
    surface_key: HoledSurfaceKey,
    points: [[f64; 3]; 3],
    material_region_id: String,
}

fn holed_polyhedron_material_classifier(
    plc: &ProtectedBoundaryComplex,
    grid: &RectangularThroughHoleGrid,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    tolerance_m: f64,
) -> Result<HoledPolyhedronMaterialClassifier, TetrahedronGenerationError> {
    let plc_material_region_ids = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    if plc_material_region_ids.is_empty() {
        return Ok(HoledPolyhedronMaterialClassifier {
            default_material_region_id: Some(DEFAULT_MATERIAL_REGION_ID.to_string()),
            facets: Vec::new(),
            tolerance_m,
        });
    }

    let mut facets = Vec::<HoledMaterialFacet>::with_capacity(plc.facets.len());
    for facet in &plc.facets {
        if facet.material_interface_ids.is_empty() {
            continue;
        }
        if facet.material_interface_ids.len() != 1 {
            return Err(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc);
        }
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
        facets.push(HoledMaterialFacet {
            surface_key,
            points,
            material_region_id: facet.material_interface_ids[0].clone(),
        });
    }
    Ok(HoledPolyhedronMaterialClassifier {
        default_material_region_id: None,
        facets,
        tolerance_m,
    })
}

fn holed_ring_region_key(
    point: [f64; 3],
    grid: &RectangularThroughHoleGrid,
    tolerance_m: f64,
) -> Option<HoledRingRegionKey> {
    let [_, x_inner_min, x_inner_max, _] = grid.source_x_values;
    let [_, y_inner_min, y_inner_max, _] = grid.source_y_values;
    let candidates = [
        (HoledRingRegionKey::Bottom, y_inner_min - point[1]),
        (HoledRingRegionKey::Right, point[0] - x_inner_max),
        (HoledRingRegionKey::Top, point[1] - y_inner_max),
        (HoledRingRegionKey::Left, x_inner_min - point[0]),
    ];
    candidates
        .into_iter()
        .filter(|(_, distance)| *distance > tolerance_m)
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(ring_region, _)| ring_region)
}

fn holed_ring_region_outer_surface_key(ring_region: HoledRingRegionKey) -> HoledSurfaceKey {
    match ring_region {
        HoledRingRegionKey::Bottom => HoledSurfaceKey::OuterYMin,
        HoledRingRegionKey::Right => HoledSurfaceKey::OuterXMax,
        HoledRingRegionKey::Top => HoledSurfaceKey::OuterYMax,
        HoledRingRegionKey::Left => HoledSurfaceKey::OuterXMin,
    }
}

fn holed_ring_region_outer_boundary_point(
    cell_center: [f64; 3],
    grid: &RectangularThroughHoleGrid,
    ring_region: HoledRingRegionKey,
) -> [f64; 3] {
    let [x_min, _, _, x_max] = grid.source_x_values;
    let [y_min, _, _, y_max] = grid.source_y_values;
    match ring_region {
        HoledRingRegionKey::Bottom => [cell_center[0], y_min, cell_center[2]],
        HoledRingRegionKey::Right => [x_max, cell_center[1], cell_center[2]],
        HoledRingRegionKey::Top => [cell_center[0], y_max, cell_center[2]],
        HoledRingRegionKey::Left => [x_min, cell_center[1], cell_center[2]],
    }
}

fn triangle_centroid(points: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (points[0][0] + points[1][0] + points[2][0]) / 3.0,
        (points[0][1] + points[1][1] + points[2][1]) / 3.0,
        (points[0][2] + points[1][2] + points[2][2]) / 3.0,
    ]
}

fn point_lies_in_triangle(point: [f64; 3], triangle: [[f64; 3]; 3], tolerance_m: f64) -> bool {
    let v0 = vector_subtract(triangle[2], triangle[0]);
    let v1 = vector_subtract(triangle[1], triangle[0]);
    let v2 = vector_subtract(point, triangle[0]);
    let dot00 = dot(v0, v0);
    let dot01 = dot(v0, v1);
    let dot02 = dot(v0, v2);
    let dot11 = dot(v1, v1);
    let dot12 = dot(v1, v2);
    let denominator = dot00 * dot11 - dot01 * dot01;
    if denominator.abs() <= f64::EPSILON {
        return false;
    }
    let inverse_denominator = 1.0 / denominator;
    let u = (dot11 * dot02 - dot01 * dot12) * inverse_denominator;
    let v = (dot00 * dot12 - dot01 * dot02) * inverse_denominator;
    let tolerance = tolerance_m.max(1.0e-12);
    u >= -tolerance && v >= -tolerance && u + v <= 1.0 + tolerance
}

fn vector_subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
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
        let source_face_id = source_face_for_holed_boundary_face(
            plc,
            grid,
            &points,
            coordinates_by_id,
            tolerance_m,
        )?;
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

fn source_face_for_holed_boundary_face(
    plc: &ProtectedBoundaryComplex,
    grid: &RectangularThroughHoleGrid,
    points: &[[f64; 3]; 3],
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    tolerance_m: f64,
) -> Result<TopologyEntityId, TetrahedronGenerationError> {
    let surface_key = holed_surface_key(points, grid, tolerance_m)
        .ok_or(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)?;
    let centroid = triangle_centroid(*points);
    for facet in &plc.facets {
        let facet_points = facet.node_ids.clone().map(|node_id| {
            coordinates_by_id.get(&node_id).copied().ok_or_else(|| {
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: node_id.id.clone(),
                }
            })
        });
        let facet_points = facet_points.into_iter().collect::<Result<Vec<_>, _>>()?;
        let facet_points = [facet_points[0], facet_points[1], facet_points[2]];
        if holed_surface_key(&facet_points, grid, tolerance_m) == Some(surface_key)
            && point_lies_in_triangle(centroid, facet_points, tolerance_m)
        {
            return Ok(facet.source_face_id.clone());
        }
    }
    Err(TetrahedronGenerationError::UnsupportedHoledPolyhedronPlc)
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

fn holed_polyhedron_refinement_length(
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
        Ok(smallest_interval * HOLED_POLYHEDRON_MAX_REFINED_CELL_ASPECT)
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
