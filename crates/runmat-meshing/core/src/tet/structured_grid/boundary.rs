use super::*;

pub(super) fn grid_boundary_faces(
    input: &BoundaryMeshInput,
    grid: &StructuredGrid,
    occupied_cells: &[bool],
    cell_tet_ids: &[Option<Vec<String>>],
    node_id_at: &impl Fn(usize, usize, usize) -> u32,
) -> Vec<AnalysisBoundaryFace> {
    let mut faces = Vec::new();
    for k in 0..grid.nz() {
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                let cell_index = grid.cell_index(i, j, k);
                if !occupied_cells[cell_index] {
                    continue;
                }
                let adjacent_volume_element_ids =
                    cell_tet_ids[cell_index].clone().unwrap_or_default();
                for side in BoundarySide::ALL {
                    if neighbor_is_occupied(grid, occupied_cells, i, j, k, side) {
                        continue;
                    }
                    let quad = boundary_side_quad(side, i, j, k, node_id_at);
                    let centroid = quad_centroid(quad, grid, node_id_at);
                    let region_ids = nearest_boundary_triangle_regions(input, centroid)
                        .unwrap_or_else(|| input.region_ids.clone());
                    for tri in [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]] {
                        faces.push(AnalysisBoundaryFace {
                            face_id: format!("bf_{}", faces.len() + 1),
                            kind: BoundaryElementKind::Tri3,
                            node_ids: tri.to_vec(),
                            adjacent_volume_element_ids: adjacent_volume_element_ids.clone(),
                            region_ids: region_ids.clone(),
                            provenance: vec![MeshEntityProvenance {
                                source_geometry_id: input.source_geometry_id.clone(),
                                source_geometry_revision: input.source_geometry_revision,
                                source_entity_kind: SourceEntityKind::Region,
                                source_entity_id: region_ids.join(","),
                                region_ids: region_ids.clone(),
                            }],
                        });
                    }
                }
            }
        }
    }
    faces
}

fn neighbor_is_occupied(
    grid: &StructuredGrid,
    occupied_cells: &[bool],
    i: usize,
    j: usize,
    k: usize,
    side: BoundarySide,
) -> bool {
    let neighbor = match side {
        BoundarySide::XMin => i.checked_sub(1).map(|i| (i, j, k)),
        BoundarySide::XMax => (i + 1 < grid.nx()).then_some((i + 1, j, k)),
        BoundarySide::YMin => j.checked_sub(1).map(|j| (i, j, k)),
        BoundarySide::YMax => (j + 1 < grid.ny()).then_some((i, j + 1, k)),
        BoundarySide::ZMin => k.checked_sub(1).map(|k| (i, j, k)),
        BoundarySide::ZMax => (k + 1 < grid.nz()).then_some((i, j, k + 1)),
    };
    neighbor
        .map(|(i, j, k)| occupied_cells[grid.cell_index(i, j, k)])
        .unwrap_or(false)
}

fn boundary_side_quad(
    side: BoundarySide,
    i: usize,
    j: usize,
    k: usize,
    node_id_at: &impl Fn(usize, usize, usize) -> u32,
) -> [u32; 4] {
    match side {
        BoundarySide::XMin => [
            node_id_at(i, j, k),
            node_id_at(i, j + 1, k),
            node_id_at(i, j + 1, k + 1),
            node_id_at(i, j, k + 1),
        ],
        BoundarySide::XMax => [
            node_id_at(i + 1, j, k),
            node_id_at(i + 1, j, k + 1),
            node_id_at(i + 1, j + 1, k + 1),
            node_id_at(i + 1, j + 1, k),
        ],
        BoundarySide::YMin => [
            node_id_at(i, j, k),
            node_id_at(i, j, k + 1),
            node_id_at(i + 1, j, k + 1),
            node_id_at(i + 1, j, k),
        ],
        BoundarySide::YMax => [
            node_id_at(i, j + 1, k),
            node_id_at(i + 1, j + 1, k),
            node_id_at(i + 1, j + 1, k + 1),
            node_id_at(i, j + 1, k + 1),
        ],
        BoundarySide::ZMin => [
            node_id_at(i, j, k),
            node_id_at(i + 1, j, k),
            node_id_at(i + 1, j + 1, k),
            node_id_at(i, j + 1, k),
        ],
        BoundarySide::ZMax => [
            node_id_at(i, j, k + 1),
            node_id_at(i, j + 1, k + 1),
            node_id_at(i + 1, j + 1, k + 1),
            node_id_at(i + 1, j, k + 1),
        ],
    }
}

fn quad_centroid(
    quad: [u32; 4],
    grid: &StructuredGrid,
    node_id_at: &impl Fn(usize, usize, usize) -> u32,
) -> [f64; 3] {
    let mut centroid = [0.0; 3];
    for node_id in quad {
        let index = node_id - 1;
        let x_index = index as usize % grid.x.len();
        let yz = index as usize / grid.x.len();
        let y_index = yz % grid.y.len();
        let z_index = yz / grid.y.len();
        debug_assert_eq!(
            node_id,
            node_id_at(x_index, y_index, z_index),
            "structured node id mapping changed"
        );
        centroid[0] += grid.x[x_index] * 0.25;
        centroid[1] += grid.y[y_index] * 0.25;
        centroid[2] += grid.z[z_index] * 0.25;
    }
    centroid
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BoundarySide {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

impl BoundarySide {
    const ALL: [Self; 6] = [
        Self::XMin,
        Self::XMax,
        Self::YMin,
        Self::YMax,
        Self::ZMin,
        Self::ZMax,
    ];
}

pub(super) fn occupied_cells(input: &BoundaryMeshInput, grid: &StructuredGrid) -> Vec<bool> {
    let mut occupied = vec![false; grid.cell_count()];
    let boundary_cells = boundary_triangle_centroid_cells(input, grid);
    for k in 0..grid.nz() {
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                let cell_index = grid.cell_index(i, j, k);
                if boundary_cells[cell_index] {
                    occupied[cell_index] = true;
                    continue;
                }
                let center = [
                    (grid.x[i] + grid.x[i + 1]) * 0.5,
                    (grid.y[j] + grid.y[j + 1]) * 0.5,
                    (grid.z[k] + grid.z[k + 1]) * 0.5,
                ];
                occupied[cell_index] = point_inside_closed_surface(input, center)
                    || cell_corners(i, j, k, grid)
                        .into_iter()
                        .any(|corner| point_inside_closed_surface(input, corner));
            }
        }
    }
    if occupied.iter().any(|cell| *cell) {
        largest_connected_occupied_component(grid, occupied)
    } else {
        vec![true; grid.cell_count()]
    }
}

pub(super) fn boundary_triangle_centroid_cells(
    input: &BoundaryMeshInput,
    grid: &StructuredGrid,
) -> Vec<bool> {
    let mut cells = vec![false; grid.cell_count()];
    if grid.cell_count() == 0 {
        return cells;
    }
    for triangle in &input.triangles {
        let Some(vertices) = triangle_vertices(input, triangle.node_ids) else {
            continue;
        };
        let centroid = triangle_centroid(vertices);
        let Some(i) = axis_cell_index(&grid.x, centroid[0]) else {
            continue;
        };
        let Some(j) = axis_cell_index(&grid.y, centroid[1]) else {
            continue;
        };
        let Some(k) = axis_cell_index(&grid.z, centroid[2]) else {
            continue;
        };
        cells[grid.cell_index(i, j, k)] = true;
    }
    cells
}

fn axis_cell_index(axis: &[f64], value: f64) -> Option<usize> {
    if axis.len() < 2 || !value.is_finite() {
        return None;
    }
    let first = *axis.first()?;
    let last = *axis.last()?;
    if value < first || value > last {
        return None;
    }
    if value == last {
        return Some(axis.len() - 2);
    }
    let upper = axis.partition_point(|breakpoint| *breakpoint <= value);
    upper.checked_sub(1).filter(|index| *index + 1 < axis.len())
}

fn cell_corners(i: usize, j: usize, k: usize, grid: &StructuredGrid) -> [[f64; 3]; 8] {
    [
        [grid.x[i], grid.y[j], grid.z[k]],
        [grid.x[i + 1], grid.y[j], grid.z[k]],
        [grid.x[i], grid.y[j + 1], grid.z[k]],
        [grid.x[i + 1], grid.y[j + 1], grid.z[k]],
        [grid.x[i], grid.y[j], grid.z[k + 1]],
        [grid.x[i + 1], grid.y[j], grid.z[k + 1]],
        [grid.x[i], grid.y[j + 1], grid.z[k + 1]],
        [grid.x[i + 1], grid.y[j + 1], grid.z[k + 1]],
    ]
}

pub(super) fn largest_connected_occupied_component(
    grid: &StructuredGrid,
    occupied_cells: Vec<bool>,
) -> Vec<bool> {
    let mut visited = vec![false; occupied_cells.len()];
    let mut largest_component = Vec::<usize>::new();
    for cell_index in 0..occupied_cells.len() {
        if !occupied_cells[cell_index] || visited[cell_index] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([cell_index]);
        visited[cell_index] = true;
        while let Some(current) = queue.pop_front() {
            component.push(current);
            let (i, j, k) = grid.cell_coordinates(current);
            for neighbor in grid.cell_neighbors(i, j, k) {
                if occupied_cells[neighbor] && !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        if component.len() > largest_component.len() {
            largest_component = component;
        }
    }
    if largest_component.is_empty() {
        return occupied_cells;
    }

    let mut retained = vec![false; occupied_cells.len()];
    for cell_index in largest_component {
        retained[cell_index] = true;
    }
    retained
}

pub(super) fn point_inside_closed_surface(input: &BoundaryMeshInput, point: [f64; 3]) -> bool {
    let epsilon = boundary_max_span(input).max(1.0) * 1.0e-10;
    let probes = [
        ([1.0, 0.0, 0.0], [-0.37, 0.19, 0.11]),
        ([0.0, 1.0, 0.0], [0.13, -0.41, 0.23]),
        ([0.0, 0.0, 1.0], [0.17, 0.29, -0.43]),
    ];
    probes
        .into_iter()
        .filter(|(direction, jitter)| {
            ray_odd_intersection_count(input, point, *direction, *jitter, epsilon)
        })
        .count()
        >= 2
}

fn ray_odd_intersection_count(
    input: &BoundaryMeshInput,
    point: [f64; 3],
    direction: [f64; 3],
    jitter: [f64; 3],
    epsilon: f64,
) -> bool {
    let origin = [
        point[0] + epsilon * jitter[0],
        point[1] + epsilon * jitter[1],
        point[2] + epsilon * jitter[2],
    ];
    let mut intersections = Vec::<f64>::new();
    for triangle in &input.triangles {
        let Some(vertices) = triangle_vertices(input, triangle.node_ids) else {
            continue;
        };
        let Some(distance) = ray_triangle_intersection(origin, direction, vertices, epsilon) else {
            continue;
        };
        if distance > epsilon {
            intersections.push(distance);
        }
    }
    intersections.sort_by(f64::total_cmp);
    intersections.dedup_by(|left, right| (*left - *right).abs() <= epsilon);
    intersections.len() % 2 == 1
}

fn ray_triangle_intersection(
    origin: [f64; 3],
    direction: [f64; 3],
    vertices: [[f64; 3]; 3],
    epsilon: f64,
) -> Option<f64> {
    let edge1 = sub(vertices[1], vertices[0]);
    let edge2 = sub(vertices[2], vertices[0]);
    let h = cross(direction, edge2);
    let determinant = dot(edge1, h);
    if determinant.abs() <= epsilon {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    let s = sub(origin, vertices[0]);
    let u = inverse_determinant * dot(s, h);
    if u < -epsilon || u > 1.0 + epsilon {
        return None;
    }
    let q = cross(s, edge1);
    let v = inverse_determinant * dot(direction, q);
    if v < -epsilon || u + v > 1.0 + epsilon {
        return None;
    }
    let distance = inverse_determinant * dot(edge2, q);
    distance.is_finite().then_some(distance)
}

fn nearest_boundary_triangle_regions(
    input: &BoundaryMeshInput,
    point: [f64; 3],
) -> Option<Vec<String>> {
    input
        .triangles
        .iter()
        .filter_map(|triangle| {
            let vertices = triangle_vertices(input, triangle.node_ids)?;
            let centroid = triangle_centroid(vertices);
            Some((distance(point, centroid), triangle.region_ids.clone()))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, mut region_ids)| {
            region_ids.sort();
            region_ids.dedup();
            region_ids
        })
        .filter(|region_ids| !region_ids.is_empty())
}
