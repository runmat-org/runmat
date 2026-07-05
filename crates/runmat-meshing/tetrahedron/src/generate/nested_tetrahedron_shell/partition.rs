use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{MeshingStage, PlcFacet, ProtectedBoundaryComplex, TopologyEntityId},
    quality::predicate::{
        orient_tetrahedron_node_ids, solve_3x3, tetrahedron_edge_aspect_ratio,
        tetrahedron_scaled_jacobian, Point3,
    },
    quality::tolerance::MeshingTolerance,
};

use crate::{
    cavity::constrained::{
        ConstrainedCavityBoundaryFace, ConstrainedCavityNode, ConstrainedCavityRefill,
        ConstrainedCavityRefillTetrahedron,
    },
    generate::TetrahedronGenerationError,
};

use super::{
    refill::{NestedTetrahedronShellRefill, NestedTetrahedronShellRefillStrategy},
    shell::NestedTetrahedronShell,
};

const BARYCENTRIC_KEY_SCALE: f64 = 1.0e12;
const MIN_PARTITION_SCALED_JACOBIAN: f64 = 0.15;
const MAX_BARYCENTRIC_PARTITION_DIVISIONS: usize = 12;

pub(super) fn barycentric_partition_refill(
    plc: &ProtectedBoundaryComplex,
    shell: &NestedTetrahedronShell,
    cavity_id_to_node_id: &BTreeMap<u32, TopologyEntityId>,
    node_id_to_cavity_id: &BTreeMap<TopologyEntityId, u32>,
    target_volume_m3: f64,
) -> Result<Option<NestedTetrahedronShellRefill>, TetrahedronGenerationError> {
    if shell.outer_node_ids.len() != 4 || shell.inner_node_ids.len() != 4 || plc.nodes.len() != 8 {
        return Ok(None);
    }
    let coordinates_by_node_id = plc
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let outer_points = shell
        .outer_node_ids
        .iter()
        .map(|node_id| {
            coordinates_by_node_id.get(node_id).copied().ok_or_else(|| {
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: node_id.id.clone(),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let partition =
        central_inner_tetrahedron_partition(shell, &coordinates_by_node_id, &outer_points)?;
    let Some(partition) = partition else {
        return Ok(None);
    };

    let mut builder = PartitionBuilder::new(cavity_id_to_node_id.clone(), outer_points.clone());
    for (index, node_id) in shell.outer_node_ids.iter().enumerate() {
        let cavity_id = cavity_node_id(node_id_to_cavity_id, node_id)?;
        let mut barycentric = [0.0; 4];
        barycentric[index] = 1.0;
        builder.insert_existing_node(barycentric, cavity_id);
    }
    for (index, node_id) in partition.inner_node_ids.iter().enumerate() {
        let cavity_id = cavity_node_id(node_id_to_cavity_id, node_id)?;
        let mut barycentric = partition.inner_lower_bounds;
        barycentric[index] += partition.inner_scale;
        builder.insert_existing_node(barycentric, cavity_id);
    }

    let cells = partition_cells(
        partition.divisions,
        partition.inner_lower_bounds,
        &mut builder,
    );
    if cells.is_empty() {
        return Ok(None);
    }

    let outer_source_faces = shell_source_faces(plc, &shell.outer_node_ids)?;
    let inner_source_faces = shell_source_faces(plc, &partition.inner_node_ids)?;
    let mut face_triangle_cache = BTreeMap::<Vec<u32>, Vec<[u32; 3]>>::new();
    let mut tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut total_volume_m3 = 0.0_f64;
    let mut min_scaled_jacobian = f64::INFINITY;

    for cell in &cells {
        let centroid_id = builder.insert_generated_node(cell_centroid(cell, &builder));
        let mut seen_cell_faces = BTreeSet::<Vec<u32>>::new();
        for face in cell_faces(cell, &builder) {
            let key = sorted_face_vec(&face);
            if !seen_cell_faces.insert(key.clone()) {
                continue;
            }
            let triangles = match face_triangle_cache.get(&key) {
                Some(triangles) => triangles.clone(),
                None => {
                    let triangles = triangulated_polygon_faces(&key, &builder);
                    face_triangle_cache.insert(key, triangles.clone());
                    triangles
                }
            };
            for triangle in triangles {
                let node_ids = [centroid_id, triangle[0], triangle[1], triangle[2]];
                let points = node_ids.map(|node_id| builder.coordinates(node_id));
                let (node_ids, volume_m3) = orient_tetrahedron_node_ids(node_ids, points);
                if volume_m3 <= 1.0e-18 {
                    continue;
                }
                let points = node_ids.map(|node_id| builder.coordinates(node_id));
                let scaled_jacobian = tetrahedron_scaled_jacobian(points);
                if !scaled_jacobian.is_finite() {
                    return Ok(None);
                }
                min_scaled_jacobian = min_scaled_jacobian.min(scaled_jacobian);
                total_volume_m3 += volume_m3;
                tetrahedra.push(ConstrainedCavityRefillTetrahedron {
                    node_ids,
                    volume_m3,
                    aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                    exact_scaled_jacobian: scaled_jacobian,
                });
            }
        }
    }
    if tetrahedra.is_empty()
        || min_scaled_jacobian < MIN_PARTITION_SCALED_JACOBIAN
        || ((total_volume_m3 - target_volume_m3).abs() > target_volume_m3.abs().max(1.0) * 1.0e-9)
    {
        return Ok(None);
    }

    let boundary_faces = partition_boundary_faces(
        &tetrahedra,
        &builder,
        partition.inner_lower_bounds,
        &outer_source_faces,
        &inner_source_faces,
    )?;
    Ok(Some(NestedTetrahedronShellRefill {
        cavity_id_to_node_id: builder.cavity_id_to_node_id,
        generated_nodes: builder.generated_nodes,
        strategy: NestedTetrahedronShellRefillStrategy::BarycentricPartition,
        boundary_centroid_refinement_attempted: false,
        boundary_centroid_refinement_rejected: false,
        refill: ConstrainedCavityRefill {
            tetrahedra,
            boundary_faces,
            inserted_nodes: Vec::new(),
            total_volume_m3,
        },
    }))
}

#[derive(Debug, Clone)]
struct CentralInnerTetrahedronPartition {
    inner_lower_bounds: [f64; 4],
    inner_scale: f64,
    divisions: usize,
    inner_node_ids: [TopologyEntityId; 4],
}

fn central_inner_tetrahedron_partition(
    shell: &NestedTetrahedronShell,
    coordinates_by_node_id: &BTreeMap<TopologyEntityId, Point3>,
    outer_points: &[Point3],
) -> Result<Option<CentralInnerTetrahedronPartition>, TetrahedronGenerationError> {
    let tolerance = MeshingTolerance::default();
    let mut candidates = Vec::<(TopologyEntityId, [f64; 4])>::new();
    let mut inner_lower_bounds = [f64::INFINITY; 4];
    for node_id in &shell.inner_node_ids {
        let point = coordinates_by_node_id
            .get(node_id)
            .copied()
            .ok_or_else(|| TetrahedronGenerationError::MissingPlcNode {
                node_id: node_id.id.clone(),
            })?;
        let Some(barycentric) = barycentric_coordinates(point, outer_points, tolerance) else {
            return Ok(None);
        };
        if barycentric.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        candidates.push((node_id.clone(), barycentric));
        for index in 0..4 {
            inner_lower_bounds[index] = inner_lower_bounds[index].min(barycentric[index]);
        }
    }
    if inner_lower_bounds
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Ok(None);
    }
    let inner_scale = 1.0 - inner_lower_bounds.iter().sum::<f64>();
    if !inner_scale.is_finite() || inner_scale <= 0.0 {
        return Ok(None);
    }
    let divisions = (1.0 / inner_scale).round() as usize;
    if !(2..=MAX_BARYCENTRIC_PARTITION_DIVISIONS).contains(&divisions) {
        return Ok(None);
    }
    let aligned_scale = 1.0 / divisions as f64;
    if (inner_scale - aligned_scale).abs() > 1.0e-8 {
        return Ok(None);
    }
    for lower_bound in &inner_lower_bounds {
        let aligned = (lower_bound * divisions as f64).round() / divisions as f64;
        if (*lower_bound - aligned).abs() > 1.0e-8 {
            return Ok(None);
        }
    }
    let mut inner_node_ids = [(); 4].map(|_| TopologyEntityId {
        stage: MeshingStage::ProtectedBoundaryComplex,
        id: String::new(),
    });
    let mut seen = BTreeSet::<usize>::new();
    for (node_id, barycentric) in candidates {
        let elevated = (0..4)
            .filter(|index| {
                (barycentric[*index] - (inner_lower_bounds[*index] + inner_scale)).abs() <= 1.0e-8
            })
            .collect::<Vec<_>>();
        if elevated.len() != 1 {
            return Ok(None);
        }
        let elevated_index = elevated[0];
        for (index, value) in barycentric.iter().enumerate() {
            let expected = if index == elevated_index {
                inner_lower_bounds[index] + inner_scale
            } else {
                inner_lower_bounds[index]
            };
            if (*value - expected).abs() > 1.0e-8 {
                return Ok(None);
            }
        }
        if !seen.insert(elevated_index) {
            return Ok(None);
        }
        inner_node_ids[elevated_index] = node_id;
    }
    if seen.len() != 4 {
        return Ok(None);
    }
    Ok(Some(CentralInnerTetrahedronPartition {
        inner_lower_bounds,
        inner_scale,
        divisions,
        inner_node_ids,
    }))
}

fn barycentric_coordinates(
    point: Point3,
    outer_points: &[Point3],
    tolerance: MeshingTolerance,
) -> Option<[f64; 4]> {
    let origin = outer_points[0];
    let matrix = [
        [
            outer_points[1][0] - origin[0],
            outer_points[2][0] - origin[0],
            outer_points[3][0] - origin[0],
        ],
        [
            outer_points[1][1] - origin[1],
            outer_points[2][1] - origin[1],
            outer_points[3][1] - origin[1],
        ],
        [
            outer_points[1][2] - origin[2],
            outer_points[2][2] - origin[2],
            outer_points[3][2] - origin[2],
        ],
    ];
    let rhs = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    let solved = solve_3x3(matrix, rhs, tolerance)?;
    Some([
        1.0 - solved[0] - solved[1] - solved[2],
        solved[0],
        solved[1],
        solved[2],
    ])
}

#[derive(Debug, Clone)]
struct PartitionCell {
    node_ids: Vec<u32>,
    lower_bounds: [f64; 4],
    upper_bounds: [f64; 4],
}

fn partition_cells(
    divisions: usize,
    inner_lower_bounds: [f64; 4],
    builder: &mut PartitionBuilder,
) -> Vec<PartitionCell> {
    let mut cells = Vec::<PartitionCell>::new();
    for first in 0..divisions {
        for second in 0..divisions {
            for third in 0..divisions {
                for fourth in 0..divisions {
                    let lower_bounds = [
                        first as f64 / divisions as f64,
                        second as f64 / divisions as f64,
                        third as f64 / divisions as f64,
                        fourth as f64 / divisions as f64,
                    ];
                    let upper_bounds = [
                        (first + 1) as f64 / divisions as f64,
                        (second + 1) as f64 / divisions as f64,
                        (third + 1) as f64 / divisions as f64,
                        (fourth + 1) as f64 / divisions as f64,
                    ];
                    if lower_bounds.iter().sum::<f64>() > 1.0 + 1.0e-12
                        || upper_bounds.iter().sum::<f64>() < 1.0 - 1.0e-12
                        || lower_bounds
                            .iter()
                            .enumerate()
                            .all(|(index, value)| *value >= inner_lower_bounds[index] - 1.0e-12)
                    {
                        continue;
                    }
                    let mut node_ids = BTreeSet::<u32>::new();
                    for active in three_active_coordinates() {
                        for mask in 0..8 {
                            let mut barycentric = [f64::NAN; 4];
                            for (bit_index, coordinate_index) in active.iter().enumerate() {
                                barycentric[*coordinate_index] = if (mask & (1 << bit_index)) == 0 {
                                    lower_bounds[*coordinate_index]
                                } else {
                                    upper_bounds[*coordinate_index]
                                };
                            }
                            let free_index = (0..4)
                                .find(|index| barycentric[*index].is_nan())
                                .expect("one free barycentric coordinate");
                            barycentric[free_index] = 1.0
                                - barycentric
                                    .iter()
                                    .filter(|value| !value.is_nan())
                                    .sum::<f64>();
                            if barycentric.iter().enumerate().all(|(index, value)| {
                                *value >= lower_bounds[index] - 1.0e-12
                                    && *value <= upper_bounds[index] + 1.0e-12
                            }) {
                                node_ids.insert(builder.insert_node(barycentric));
                            }
                        }
                    }
                    if node_ids.len() >= 4 {
                        cells.push(PartitionCell {
                            node_ids: node_ids.into_iter().collect(),
                            lower_bounds,
                            upper_bounds,
                        });
                    }
                }
            }
        }
    }
    cells
}

fn three_active_coordinates() -> [[usize; 3]; 4] {
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
}

#[derive(Debug, Clone)]
struct PartitionBuilder {
    cavity_id_to_node_id: BTreeMap<u32, TopologyEntityId>,
    barycentric_by_id: BTreeMap<u32, [f64; 4]>,
    cavity_id_by_barycentric: BTreeMap<[i64; 4], u32>,
    generated_nodes: Vec<ConstrainedCavityNode>,
    outer_points: Vec<Point3>,
    next_cavity_id: u32,
}

impl PartitionBuilder {
    fn new(
        cavity_id_to_node_id: BTreeMap<u32, TopologyEntityId>,
        outer_points: Vec<Point3>,
    ) -> Self {
        let next_cavity_id = cavity_id_to_node_id
            .keys()
            .next_back()
            .map(|id| id + 1)
            .unwrap_or(0);
        Self {
            cavity_id_to_node_id,
            barycentric_by_id: BTreeMap::new(),
            cavity_id_by_barycentric: BTreeMap::new(),
            generated_nodes: Vec::new(),
            outer_points,
            next_cavity_id,
        }
    }

    fn insert_existing_node(&mut self, barycentric: [f64; 4], cavity_id: u32) {
        self.cavity_id_by_barycentric
            .insert(barycentric_key(barycentric), cavity_id);
        self.barycentric_by_id.insert(cavity_id, barycentric);
    }

    fn insert_node(&mut self, barycentric: [f64; 4]) -> u32 {
        let key = barycentric_key(barycentric);
        if let Some(node_id) = self.cavity_id_by_barycentric.get(&key) {
            return *node_id;
        }
        self.insert_generated_node(barycentric)
    }

    fn insert_generated_node(&mut self, barycentric: [f64; 4]) -> u32 {
        let key = barycentric_key(barycentric);
        if let Some(node_id) = self.cavity_id_by_barycentric.get(&key) {
            return *node_id;
        }
        let node_id = self.next_cavity_id;
        self.next_cavity_id += 1;
        let coordinates_m = self.point(barycentric);
        self.cavity_id_by_barycentric.insert(key, node_id);
        self.barycentric_by_id.insert(node_id, barycentric);
        self.cavity_id_to_node_id.insert(
            node_id,
            TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("nested_tetrahedron_shell_partition_node_{node_id}"),
            },
        );
        self.generated_nodes.push(ConstrainedCavityNode {
            node_id,
            coordinates_m,
        });
        node_id
    }

    fn point(&self, barycentric: [f64; 4]) -> Point3 {
        let mut point = [0.0; 3];
        for (index, weight) in barycentric.iter().enumerate() {
            for (axis, coordinate) in point.iter_mut().enumerate() {
                *coordinate += weight * self.outer_points[index][axis];
            }
        }
        point
    }

    fn coordinates(&self, node_id: u32) -> Point3 {
        self.point(self.barycentric_by_id[&node_id])
    }
}

fn barycentric_key(barycentric: [f64; 4]) -> [i64; 4] {
    barycentric.map(|value| (value * BARYCENTRIC_KEY_SCALE).round() as i64)
}

fn cell_centroid(cell: &PartitionCell, builder: &PartitionBuilder) -> [f64; 4] {
    let mut barycentric = [0.0; 4];
    for node_id in &cell.node_ids {
        let node_barycentric = builder.barycentric_by_id[node_id];
        for index in 0..4 {
            barycentric[index] += node_barycentric[index];
        }
    }
    for value in &mut barycentric {
        *value /= cell.node_ids.len() as f64;
    }
    barycentric
}

fn cell_faces(cell: &PartitionCell, builder: &PartitionBuilder) -> Vec<Vec<u32>> {
    let mut faces = Vec::<Vec<u32>>::new();
    for coordinate_index in 0..4 {
        for value in [
            cell.lower_bounds[coordinate_index],
            cell.upper_bounds[coordinate_index],
        ] {
            let face = cell
                .node_ids
                .iter()
                .copied()
                .filter(|node_id| {
                    let barycentric = builder.barycentric_by_id[node_id];
                    (barycentric[coordinate_index] - value).abs() <= 1.0e-12
                })
                .collect::<Vec<_>>();
            if face.len() >= 3 {
                faces.push(face);
            }
        }
    }
    faces
}

fn sorted_face_vec(node_ids: &[u32]) -> Vec<u32> {
    let mut sorted = node_ids.to_vec();
    sorted.sort();
    sorted
}

fn triangulated_polygon_faces(face_key: &[u32], builder: &PartitionBuilder) -> Vec<[u32; 3]> {
    let ordered = polygon_order(face_key, builder);
    let Some((anchor_index, anchor)) = ordered
        .iter()
        .copied()
        .enumerate()
        .min_by_key(|(_, node_id)| *node_id)
    else {
        return Vec::new();
    };
    let mut rotated = ordered[anchor_index..].to_vec();
    rotated.extend_from_slice(&ordered[..anchor_index]);
    (1..rotated.len().saturating_sub(1))
        .map(|index| [anchor, rotated[index], rotated[index + 1]])
        .collect()
}

fn polygon_order(face_key: &[u32], builder: &PartitionBuilder) -> Vec<u32> {
    let mut center = [0.0; 3];
    for node_id in face_key {
        let point = builder.coordinates(*node_id);
        for axis in 0..3 {
            center[axis] += point[axis];
        }
    }
    for coordinate in &mut center {
        *coordinate /= face_key.len() as f64;
    }
    let points = face_key
        .iter()
        .map(|node_id| (*node_id, builder.coordinates(*node_id)))
        .collect::<Vec<_>>();
    let normal = polygon_normal(&points).unwrap_or([0.0, 0.0, 1.0]);
    let mut first_axis = sub(points[0].1, center);
    let first_axis_norm = norm(first_axis);
    if first_axis_norm <= f64::EPSILON {
        return face_key.to_vec();
    }
    first_axis = scale(first_axis, 1.0 / first_axis_norm);
    let second_axis = cross(normal, first_axis);
    let mut ordered = points
        .into_iter()
        .map(|(node_id, point)| {
            let relative = sub(point, center);
            let angle = dot(relative, second_axis).atan2(dot(relative, first_axis));
            (node_id, angle)
        })
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.1.total_cmp(&right.1));
    ordered.into_iter().map(|(node_id, _)| node_id).collect()
}

fn polygon_normal(points: &[(u32, Point3)]) -> Option<Point3> {
    for first in 0..points.len() {
        for second in (first + 1)..points.len() {
            for third in (second + 1)..points.len() {
                let normal = cross(
                    sub(points[second].1, points[first].1),
                    sub(points[third].1, points[first].1),
                );
                let length = norm(normal);
                if length > 1.0e-12 {
                    return Some(scale(normal, 1.0 / length));
                }
            }
        }
    }
    None
}

fn partition_boundary_faces(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    builder: &PartitionBuilder,
    inner_lower_bounds: [f64; 4],
    outer_source_faces: &BTreeMap<usize, usize>,
    inner_source_faces: &BTreeMap<usize, usize>,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, TetrahedronGenerationError> {
    let mut face_counts = BTreeMap::<[u32; 3], ([u32; 3], usize)>::new();
    for tetrahedron in tetrahedra {
        for face in tetrahedron_faces(tetrahedron.node_ids) {
            let key = sorted_face(face);
            face_counts
                .entry(key)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((face, 1));
        }
    }
    let mut boundary_faces = Vec::<ConstrainedCavityBoundaryFace>::new();
    for (_, (node_ids, count)) in face_counts {
        if count != 1 {
            continue;
        }
        let source_facet_index = boundary_source_facet_index(
            node_ids,
            builder,
            inner_lower_bounds,
            outer_source_faces,
            inner_source_faces,
        )?;
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: Some(
                u32::try_from(source_facet_index).map_err(|_| {
                    TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc
                })?,
            ),
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        });
    }
    Ok(boundary_faces)
}

fn boundary_source_facet_index(
    node_ids: [u32; 3],
    builder: &PartitionBuilder,
    inner_lower_bounds: [f64; 4],
    outer_source_faces: &BTreeMap<usize, usize>,
    inner_source_faces: &BTreeMap<usize, usize>,
) -> Result<usize, TetrahedronGenerationError> {
    for coordinate_index in 0..4 {
        if node_ids
            .iter()
            .all(|node_id| builder.barycentric_by_id[node_id][coordinate_index].abs() <= 1.0e-12)
        {
            return outer_source_faces
                .get(&coordinate_index)
                .copied()
                .ok_or(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
        }
        if node_ids.iter().all(|node_id| {
            (builder.barycentric_by_id[node_id][coordinate_index]
                - inner_lower_bounds[coordinate_index])
                .abs()
                <= 1.0e-12
        }) {
            return inner_source_faces
                .get(&coordinate_index)
                .copied()
                .ok_or(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
        }
    }
    Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)
}

fn shell_source_faces(
    plc: &ProtectedBoundaryComplex,
    shell_node_ids: &[TopologyEntityId],
) -> Result<BTreeMap<usize, usize>, TetrahedronGenerationError> {
    if shell_node_ids.len() != 4 {
        return Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc);
    }
    let mut source_faces = BTreeMap::<usize, usize>::new();
    for omitted_index in 0..4 {
        let mut face_nodes = shell_node_ids
            .iter()
            .enumerate()
            .filter_map(|(index, node_id)| (index != omitted_index).then_some(node_id.clone()))
            .collect::<Vec<_>>();
        face_nodes.sort();
        let facet_index = plc
            .facets
            .iter()
            .position(|facet| sorted_plc_facet_nodes(facet) == face_nodes)
            .ok_or(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?;
        source_faces.insert(omitted_index, facet_index);
    }
    Ok(source_faces)
}

fn sorted_plc_facet_nodes(facet: &PlcFacet) -> Vec<TopologyEntityId> {
    let mut node_ids = facet.node_ids.to_vec();
    node_ids.sort();
    node_ids
}

fn tetrahedron_faces(node_ids: [u32; 4]) -> [[u32; 3]; 4] {
    [
        [node_ids[0], node_ids[1], node_ids[2]],
        [node_ids[0], node_ids[1], node_ids[3]],
        [node_ids[0], node_ids[2], node_ids[3]],
        [node_ids[1], node_ids[2], node_ids[3]],
    ]
}

fn sorted_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort();
    node_ids
}

fn cavity_node_id(
    node_id_to_cavity_id: &BTreeMap<TopologyEntityId, u32>,
    node_id: &TopologyEntityId,
) -> Result<u32, TetrahedronGenerationError> {
    node_id_to_cavity_id.get(node_id).copied().ok_or_else(|| {
        TetrahedronGenerationError::MissingPlcNode {
            node_id: node_id.id.clone(),
        }
    })
}

fn sub(left: Point3, right: Point3) -> Point3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: Point3, right: Point3) -> Point3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: Point3, right: Point3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm(vector: Point3) -> f64 {
    dot(vector, vector).sqrt()
}

fn scale(vector: Point3, factor: f64) -> Point3 {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}
