use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    surface::{SurfaceDiscretization, SurfaceElement},
    volume_candidate::{VolumeCandidateComponent, VolumeCandidateSet},
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetCandidateOptions {
    pub min_volume_m3: f64,
    pub max_aspect_ratio: f64,
    pub interior_target_size_m: Option<f64>,
    pub max_interior_seed_points: usize,
}

impl Default for TetCandidateOptions {
    fn default() -> Self {
        Self {
            min_volume_m3: 1.0e-18,
            max_aspect_ratio: 1.0e6,
            interior_target_size_m: None,
            max_interior_seed_points: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetCandidateNode {
    pub node_id: u32,
    pub coordinates_m: [f64; 3],
    pub source: TetCandidateNodeSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetCandidateNodeSource {
    Surface,
    InteriorSeed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetCandidate {
    pub tet_id: u32,
    pub component_id: u32,
    pub node_ids: [u32; 4],
    pub source_surface_element_id: u32,
    pub region_ids: Vec<String>,
    pub volume_m3: f64,
    pub aspect_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetCandidateSet {
    pub nodes: Vec<TetCandidateNode>,
    pub tets: Vec<TetCandidate>,
    pub interior_seed_points: Vec<[f64; 3]>,
    pub total_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetCandidateError {
    MissingSurfaceNode { node_id: u32 },
    MissingSurfaceElement { element_id: u32 },
    InvalidOptions,
    EmptyCandidateSet,
}

impl std::fmt::Display for TetCandidateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSurfaceNode { node_id } => {
                write!(formatter, "surface node {node_id} is missing")
            }
            Self::MissingSurfaceElement { element_id } => {
                write!(formatter, "surface element {element_id} is missing")
            }
            Self::InvalidOptions => write!(
                formatter,
                "Tet candidate options must use finite positive volume and aspect ratio limits"
            ),
            Self::EmptyCandidateSet => write!(formatter, "no valid Tet candidates were generated"),
        }
    }
}

impl std::error::Error for TetCandidateError {}

pub fn form_tet_candidates(
    surface: &SurfaceDiscretization,
    volume_candidates: &VolumeCandidateSet,
    options: TetCandidateOptions,
) -> Result<TetCandidateSet, TetCandidateError> {
    validate_options(options)?;
    let mut nodes = surface
        .nodes
        .iter()
        .map(|node| TetCandidateNode {
            node_id: node.node_id,
            coordinates_m: node.coordinates_m,
            source: TetCandidateNodeSource::Surface,
        })
        .collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.node_id);

    let surface_nodes = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let surface_elements = surface
        .elements
        .iter()
        .map(|element| (element.element_id, element))
        .collect::<BTreeMap<_, _>>();

    let mut tets = Vec::<TetCandidate>::new();
    let mut next_node_id = nodes
        .iter()
        .map(|node| node.node_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

    let mut interior_seed_points = Vec::<[f64; 3]>::new();
    for component in &volume_candidates.components {
        let component_seed_points =
            sample_component_interior_points(component, surface, &surface_elements, options)?;

        let mut component_seed_node_ids = Vec::<u32>::with_capacity(component_seed_points.len());
        for point in &component_seed_points {
            let node_id = next_node_id;
            next_node_id = next_node_id.saturating_add(1);
            component_seed_node_ids.push(node_id);
            nodes.push(TetCandidateNode {
                node_id,
                coordinates_m: *point,
                source: TetCandidateNodeSource::InteriorSeed,
            });
        }
        interior_seed_points.extend(component_seed_points.iter().copied());

        let insertion_start = tets.len();
        append_component_insertion_tets(
            component,
            &component_seed_node_ids,
            &component_seed_points,
            &surface_nodes,
            &surface_elements,
            surface,
            options,
            &mut tets,
        )?;
        if tets.len() == insertion_start {
            let fan_seed_point = select_component_fan_seed_point(
                component,
                &component_seed_points,
                &surface_nodes,
                &surface_elements,
                options,
            )?;
            let fan_seed_node_id = component_seed_node_ids[component_seed_points
                .iter()
                .position(|point| distance(*point, fan_seed_point) <= 1.0e-12)
                .unwrap_or(0)];
            append_component_tets(
                component,
                fan_seed_node_id,
                fan_seed_point,
                &surface_nodes,
                &surface_elements,
                options,
                &mut tets,
            )?;
        }
    }

    if tets.is_empty() {
        return Err(TetCandidateError::EmptyCandidateSet);
    }
    let total_volume_m3 = tets.iter().map(|tet| tet.volume_m3).sum();
    Ok(TetCandidateSet {
        nodes,
        tets,
        interior_seed_points,
        total_volume_m3,
    })
}

fn validate_options(options: TetCandidateOptions) -> Result<(), TetCandidateError> {
    if !options.min_volume_m3.is_finite()
        || options.min_volume_m3 <= 0.0
        || !options.max_aspect_ratio.is_finite()
        || options.max_aspect_ratio <= 0.0
        || options.max_interior_seed_points == 0
        || options
            .interior_target_size_m
            .is_some_and(|size| !size.is_finite() || size <= 0.0)
    {
        return Err(TetCandidateError::InvalidOptions);
    }
    Ok(())
}

fn append_component_tets(
    component: &VolumeCandidateComponent,
    interior_node_id: u32,
    interior: [f64; 3],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
    tets: &mut Vec<TetCandidate>,
) -> Result<(), TetCandidateError> {
    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        let mut node_ids = [
            element.node_ids[0],
            element.node_ids[1],
            element.node_ids[2],
            interior_node_id,
        ];
        let points = tet_points(node_ids, interior, surface_nodes)?;
        let mut signed_volume_m3 = tet_signed_volume(points);
        if signed_volume_m3 < 0.0 {
            node_ids.swap(1, 2);
            signed_volume_m3 = -signed_volume_m3;
        }
        let volume_m3 = signed_volume_m3.abs();
        if volume_m3 < options.min_volume_m3 {
            continue;
        }
        let aspect_ratio = tet_aspect_ratio(points);
        if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
            continue;
        }
        tets.push(TetCandidate {
            tet_id: tets.len() as u32,
            component_id: component.component_id,
            node_ids,
            source_surface_element_id: element.element_id,
            region_ids: element.region_ids.clone(),
            volume_m3,
            aspect_ratio,
        });
    }
    Ok(())
}

fn append_component_insertion_tets(
    component: &VolumeCandidateComponent,
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    surface: &SurfaceDiscretization,
    options: TetCandidateOptions,
    tets: &mut Vec<TetCandidate>,
) -> Result<(), TetCandidateError> {
    if seed_node_ids.is_empty() || seed_node_ids.len() != seed_points.len() {
        return Ok(());
    }

    let mut points = Vec::<ConnectivityPoint>::new();
    for node_id in &component.node_ids {
        let coordinates_m = *surface_nodes
            .get(node_id)
            .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?;
        points.push(ConnectivityPoint {
            node_id: *node_id,
            coordinates_m,
            is_super: false,
        });
    }
    for (node_id, point) in seed_node_ids.iter().zip(seed_points.iter()) {
        points.push(ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: *point,
            is_super: false,
        });
    }

    let candidate_tets = tetrahedralize_points(&points);
    let mut accepted_tets = Vec::<TetCandidate>::new();
    for candidate in candidate_tets {
        let tet_points = candidate.vertices.map(|index| points[index].coordinates_m);
        let centroid = tet_centroid(tet_points);
        if !point_is_inside_component(centroid, component, surface, surface_elements)? {
            continue;
        }
        let mut node_ids = candidate.vertices.map(|index| points[index].node_id);
        let mut signed_volume_m3 = tet_signed_volume(tet_points);
        if signed_volume_m3 < 0.0 {
            node_ids.swap(1, 2);
            signed_volume_m3 = -signed_volume_m3;
        }
        let volume_m3 = signed_volume_m3.abs();
        if volume_m3 < options.min_volume_m3 {
            continue;
        }
        let aspect_ratio = tet_aspect_ratio(tet_points);
        if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
            continue;
        }
        let source_surface_element_id =
            nearest_surface_element_id(centroid, component, surface, surface_elements)?;
        let region_ids = surface_elements
            .get(&source_surface_element_id)
            .map(|element| element.region_ids.clone())
            .unwrap_or_default();
        accepted_tets.push(TetCandidate {
            tet_id: 0,
            component_id: component.component_id,
            node_ids,
            source_surface_element_id,
            region_ids,
            volume_m3,
            aspect_ratio,
        });
    }
    if insertion_tets_are_acceptable(component, &accepted_tets) {
        for mut tet in accepted_tets {
            tet.tet_id = tets.len() as u32;
            tets.push(tet);
        }
    }
    Ok(())
}

fn insertion_tets_are_acceptable(
    component: &VolumeCandidateComponent,
    tets: &[TetCandidate],
) -> bool {
    if tets.is_empty() || component.volume_m3 <= f64::EPSILON {
        return false;
    }
    let total_volume_m3 = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
    let volume_ratio = total_volume_m3 / component.volume_m3;
    let max_aspect_ratio = tets
        .iter()
        .map(|tet| tet.aspect_ratio)
        .fold(0.0_f64, f64::max);
    (0.90..=1.10).contains(&volume_ratio) && max_aspect_ratio <= 1.0 / 0.15
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ConnectivityPoint {
    node_id: u32,
    coordinates_m: [f64; 3],
    is_super: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConnectivityTet {
    vertices: [usize; 4],
}

fn tetrahedralize_points(input_points: &[ConnectivityPoint]) -> Vec<ConnectivityTet> {
    if input_points.len() < 4 {
        return Vec::new();
    }
    let mut points = input_points.to_vec();
    let super_start = points.len();
    points.extend(super_tetrahedron_points(input_points));
    let mut tets = vec![ConnectivityTet {
        vertices: [
            super_start,
            super_start + 1,
            super_start + 2,
            super_start + 3,
        ],
    }];

    for point_index in 0..input_points.len() {
        let point = points[point_index].coordinates_m;
        let mut bad_indices = Vec::<usize>::new();
        for (tet_index, tet) in tets.iter().enumerate() {
            if tet_circumsphere_contains_point(
                tet.vertices.map(|index| points[index].coordinates_m),
                point,
            ) {
                bad_indices.push(tet_index);
            }
        }
        if bad_indices.is_empty() {
            continue;
        }

        let bad_set = bad_indices.iter().copied().collect::<BTreeSet<_>>();
        let mut face_counts = BTreeMap::<[usize; 3], usize>::new();
        for tet_index in &bad_indices {
            for face in tet_faces(tets[*tet_index].vertices) {
                *face_counts.entry(sorted_face(face)).or_default() += 1;
            }
        }
        let cavity_faces = face_counts
            .into_iter()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect::<Vec<_>>();

        tets = tets
            .into_iter()
            .enumerate()
            .filter_map(|(tet_index, tet)| (!bad_set.contains(&tet_index)).then_some(tet))
            .collect();
        for face in cavity_faces {
            let vertices = [face[0], face[1], face[2], point_index];
            let points_for_tet = vertices.map(|index| points[index].coordinates_m);
            if tet_signed_volume(points_for_tet).abs() > 1.0e-24 {
                tets.push(ConnectivityTet { vertices });
            }
        }
    }

    tets.into_iter()
        .filter(|tet| !tet.vertices.iter().any(|index| points[*index].is_super))
        .collect()
}

fn super_tetrahedron_points(points: &[ConnectivityPoint]) -> [ConnectivityPoint; 4] {
    let mut min = points[0].coordinates_m;
    let mut max = points[0].coordinates_m;
    for point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point.coordinates_m[axis]);
            max[axis] = max[axis].max(point.coordinates_m[axis]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let span = (0..3)
        .map(|axis| max[axis] - min[axis])
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let radius = span * 16.0;
    [
        ConnectivityPoint {
            node_id: u32::MAX - 3,
            coordinates_m: [center[0] + radius, center[1], center[2] - radius],
            is_super: true,
        },
        ConnectivityPoint {
            node_id: u32::MAX - 2,
            coordinates_m: [center[0] - radius, center[1] + radius, center[2] - radius],
            is_super: true,
        },
        ConnectivityPoint {
            node_id: u32::MAX - 1,
            coordinates_m: [center[0] - radius, center[1] - radius, center[2] - radius],
            is_super: true,
        },
        ConnectivityPoint {
            node_id: u32::MAX,
            coordinates_m: [center[0], center[1], center[2] + radius],
            is_super: true,
        },
    ]
}

fn tet_faces(vertices: [usize; 4]) -> [[usize; 3]; 4] {
    [
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[1], vertices[3]],
        [vertices[0], vertices[2], vertices[3]],
        [vertices[1], vertices[2], vertices[3]],
    ]
}

fn sorted_face(mut face: [usize; 3]) -> [usize; 3] {
    face.sort();
    face
}

fn tet_circumsphere_contains_point(tet_points: [[f64; 3]; 4], point: [f64; 3]) -> bool {
    let Some((center, radius_squared)) = tet_circumsphere(tet_points) else {
        return false;
    };
    distance_squared(center, point) <= radius_squared * (1.0 + 1.0e-10)
}

fn tet_circumsphere(points: [[f64; 3]; 4]) -> Option<([f64; 3], f64)> {
    let a = [
        [
            2.0 * (points[1][0] - points[0][0]),
            2.0 * (points[1][1] - points[0][1]),
            2.0 * (points[1][2] - points[0][2]),
        ],
        [
            2.0 * (points[2][0] - points[0][0]),
            2.0 * (points[2][1] - points[0][1]),
            2.0 * (points[2][2] - points[0][2]),
        ],
        [
            2.0 * (points[3][0] - points[0][0]),
            2.0 * (points[3][1] - points[0][1]),
            2.0 * (points[3][2] - points[0][2]),
        ],
    ];
    let b = [
        dot(points[1], points[1]) - dot(points[0], points[0]),
        dot(points[2], points[2]) - dot(points[0], points[0]),
        dot(points[3], points[3]) - dot(points[0], points[0]),
    ];
    let center = solve_3x3(a, b)?;
    Some((center, distance_squared(center, points[0])))
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for pivot in 0..3 {
        let mut pivot_row = pivot;
        for row in (pivot + 1)..3 {
            if a[row][pivot].abs() > a[pivot_row][pivot].abs() {
                pivot_row = row;
            }
        }
        if a[pivot_row][pivot].abs() <= 1.0e-14 {
            return None;
        }
        if pivot_row != pivot {
            a.swap(pivot, pivot_row);
            b.swap(pivot, pivot_row);
        }
        let pivot_value = a[pivot][pivot];
        for column in pivot..3 {
            a[pivot][column] /= pivot_value;
        }
        b[pivot] /= pivot_value;
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            for column in pivot..3 {
                a[row][column] -= factor * a[pivot][column];
            }
            b[row] -= factor * b[pivot];
        }
    }
    Some(b)
}

fn nearest_surface_element_id(
    point: [f64; 3],
    component: &VolumeCandidateComponent,
    surface: &SurfaceDiscretization,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
) -> Result<u32, TetCandidateError> {
    let mut best = None::<(u32, f64)>;
    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        let centroid = triangle_centroid(surface_element_points(surface, element)?);
        let distance_squared = distance_squared(point, centroid);
        if best.is_none_or(|(_, best_distance)| distance_squared < best_distance) {
            best = Some((element.element_id, distance_squared));
        }
    }
    best.map(|(element_id, _)| element_id)
        .ok_or(TetCandidateError::MissingSurfaceElement { element_id: 0 })
}

fn triangle_centroid(points: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (points[0][0] + points[1][0] + points[2][0]) / 3.0,
        (points[0][1] + points[1][1] + points[2][1]) / 3.0,
        (points[0][2] + points[1][2] + points[2][2]) / 3.0,
    ]
}

fn tet_centroid(points: [[f64; 3]; 4]) -> [f64; 3] {
    [
        (points[0][0] + points[1][0] + points[2][0] + points[3][0]) * 0.25,
        (points[0][1] + points[1][1] + points[2][1] + points[3][1]) * 0.25,
        (points[0][2] + points[1][2] + points[2][2] + points[3][2]) * 0.25,
    ]
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FanSeedScore {
    point: [f64; 3],
    valid_tet_count: usize,
    volume_error_ratio: f64,
    max_aspect_ratio: f64,
    mean_aspect_ratio: f64,
}

fn select_component_fan_seed_point(
    component: &VolumeCandidateComponent,
    seed_points: &[[f64; 3]],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
) -> Result<[f64; 3], TetCandidateError> {
    let mut best_score = None::<FanSeedScore>;
    for point in seed_points {
        let score =
            score_fan_seed_point(component, *point, surface_nodes, surface_elements, options)?;
        if best_score.is_none_or(|best| fan_seed_score_is_better(score, best)) {
            best_score = Some(score);
        }
    }
    Ok(best_score
        .map(|score| score.point)
        .unwrap_or_else(|| component_interior_point(component)))
}

fn score_fan_seed_point(
    component: &VolumeCandidateComponent,
    point: [f64; 3],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
) -> Result<FanSeedScore, TetCandidateError> {
    let mut valid_tet_count = 0_usize;
    let mut total_volume_m3 = 0.0_f64;
    let mut aspect_ratio_sum = 0.0_f64;
    let mut max_aspect_ratio = 0.0_f64;

    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        let points = tet_points(
            [
                element.node_ids[0],
                element.node_ids[1],
                element.node_ids[2],
                u32::MAX,
            ],
            point,
            surface_nodes,
        )?;
        let volume_m3 = tet_signed_volume(points).abs();
        let aspect_ratio = tet_aspect_ratio(points);
        if volume_m3 < options.min_volume_m3
            || !aspect_ratio.is_finite()
            || aspect_ratio > options.max_aspect_ratio
        {
            continue;
        }
        valid_tet_count += 1;
        total_volume_m3 += volume_m3;
        aspect_ratio_sum += aspect_ratio;
        max_aspect_ratio = max_aspect_ratio.max(aspect_ratio);
    }

    let mean_aspect_ratio = if valid_tet_count == 0 {
        f64::INFINITY
    } else {
        aspect_ratio_sum / valid_tet_count as f64
    };
    let volume_error_ratio = if component.volume_m3 > 0.0 {
        ((total_volume_m3 - component.volume_m3).abs() / component.volume_m3).abs()
    } else {
        f64::INFINITY
    };
    Ok(FanSeedScore {
        point,
        valid_tet_count,
        volume_error_ratio,
        max_aspect_ratio,
        mean_aspect_ratio,
    })
}

fn fan_seed_score_is_better(candidate: FanSeedScore, best: FanSeedScore) -> bool {
    candidate
        .valid_tet_count
        .cmp(&best.valid_tet_count)
        .then_with(|| {
            best.volume_error_ratio
                .partial_cmp(&candidate.volume_error_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            best.max_aspect_ratio
                .partial_cmp(&candidate.max_aspect_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            best.mean_aspect_ratio
                .partial_cmp(&candidate.mean_aspect_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .is_gt()
}

fn component_interior_point(component: &VolumeCandidateComponent) -> [f64; 3] {
    [
        (component.bounds_min_m[0] + component.bounds_max_m[0]) * 0.5,
        (component.bounds_min_m[1] + component.bounds_max_m[1]) * 0.5,
        (component.bounds_min_m[2] + component.bounds_max_m[2]) * 0.5,
    ]
}

fn sample_component_interior_points(
    component: &VolumeCandidateComponent,
    surface: &SurfaceDiscretization,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
) -> Result<Vec<[f64; 3]>, TetCandidateError> {
    let mut points = Vec::<[f64; 3]>::new();
    let center = component_interior_point(component);
    if point_is_inside_component(center, component, surface, surface_elements)? {
        points.push(center);
    }

    if let Some(target_size_m) = options.interior_target_size_m {
        let spans = [
            component.bounds_max_m[0] - component.bounds_min_m[0],
            component.bounds_max_m[1] - component.bounds_min_m[1],
            component.bounds_max_m[2] - component.bounds_min_m[2],
        ];
        let divisions = spans.map(|span| ((span / target_size_m).ceil() as usize).max(1));
        'x_axis: for x_index in 0..divisions[0] {
            for y_index in 0..divisions[1] {
                for z_index in 0..divisions[2] {
                    if points.len() >= options.max_interior_seed_points {
                        break 'x_axis;
                    }
                    let point = [
                        grid_center(component.bounds_min_m[0], spans[0], divisions[0], x_index),
                        grid_center(component.bounds_min_m[1], spans[1], divisions[1], y_index),
                        grid_center(component.bounds_min_m[2], spans[2], divisions[2], z_index),
                    ];
                    if contains_point(&points, point) {
                        continue;
                    }
                    if point_is_inside_component(point, component, surface, surface_elements)? {
                        points.push(point);
                    }
                }
            }
        }
    }

    if points.is_empty() {
        points.push(center);
    }
    Ok(points)
}

fn grid_center(minimum: f64, span: f64, divisions: usize, index: usize) -> f64 {
    minimum + span * (index as f64 + 0.5) / divisions as f64
}

fn contains_point(points: &[[f64; 3]], candidate: [f64; 3]) -> bool {
    points
        .iter()
        .any(|point| distance(*point, candidate) <= 1.0e-12)
}

fn point_is_inside_component(
    point: [f64; 3],
    component: &VolumeCandidateComponent,
    surface: &SurfaceDiscretization,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
) -> Result<bool, TetCandidateError> {
    let mut hits = Vec::<f64>::new();
    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        let triangle = surface_element_points(surface, element)?;
        if let Some(hit) = ray_x_triangle_hit(point, triangle) {
            if !hits
                .iter()
                .any(|existing| (existing - hit).abs() <= 1.0e-10)
            {
                hits.push(hit);
            }
        }
    }
    Ok(hits.len() % 2 == 1)
}

fn surface_element_points(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
) -> Result<[[f64; 3]; 3], TetCandidateError> {
    Ok([
        surface_node(surface, element.node_ids[0])?,
        surface_node(surface, element.node_ids[1])?,
        surface_node(surface, element.node_ids[2])?,
    ])
}

fn surface_node(
    surface: &SurfaceDiscretization,
    node_id: u32,
) -> Result<[f64; 3], TetCandidateError> {
    surface
        .nodes
        .get(node_id as usize)
        .filter(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
        .ok_or(TetCandidateError::MissingSurfaceNode { node_id })
}

fn ray_x_triangle_hit(origin: [f64; 3], triangle: [[f64; 3]; 3]) -> Option<f64> {
    let direction = [1.0, 0.0, 0.0];
    let edge_1 = sub(triangle[1], triangle[0]);
    let edge_2 = sub(triangle[2], triangle[0]);
    let h = cross(direction, edge_2);
    let determinant = dot(edge_1, h);
    if determinant.abs() <= 1.0e-12 {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    let s = sub(origin, triangle[0]);
    let u = inverse_determinant * dot(s, h);
    if !(-1.0e-12..=1.0 + 1.0e-12).contains(&u) {
        return None;
    }
    let q = cross(s, edge_1);
    let v = inverse_determinant * dot(direction, q);
    if v < -1.0e-12 || u + v > 1.0 + 1.0e-12 {
        return None;
    }
    let t = inverse_determinant * dot(edge_2, q);
    if t > 1.0e-12 {
        Some(t)
    } else {
        None
    }
}

fn tet_points(
    node_ids: [u32; 4],
    interior: [f64; 3],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
) -> Result<[[f64; 3]; 4], TetCandidateError> {
    Ok([
        *surface_nodes
            .get(&node_ids[0])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[0],
            })?,
        *surface_nodes
            .get(&node_ids[1])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[1],
            })?,
        *surface_nodes
            .get(&node_ids[2])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[2],
            })?,
        interior,
    ])
}

fn tet_signed_volume(points: [[f64; 3]; 4]) -> f64 {
    dot(
        sub(points[1], points[0]),
        cross(sub(points[2], points[0]), sub(points[3], points[0])),
    ) / 6.0
}

fn tet_aspect_ratio(points: [[f64; 3]; 4]) -> f64 {
    let mut min_edge = f64::INFINITY;
    let mut max_edge = 0.0_f64;
    for left_index in 0..4 {
        for right_index in (left_index + 1)..4 {
            let length = distance(points[left_index], points[right_index]);
            min_edge = min_edge.min(length);
            max_edge = max_edge.max(length);
        }
    }
    max_edge / min_edge.max(f64::EPSILON)
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    distance_squared(left, right).sqrt()
}

fn distance_squared(left: [f64; 3], right: [f64; 3]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        discretize_topology_surfaces, extract_source_topology, prepare_volume_candidates,
        SurfaceDiscretizationOptions, VolumeCandidateOptions,
    };
    use runmat_geometry_core::{
        EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
        RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
        UnitSystem,
    };

    #[test]
    fn forms_positive_tet_candidates_from_closed_cube_surface() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();

        let candidates =
            form_tet_candidates(&surface, &volume_candidates, TetCandidateOptions::default())
                .expect("Tet candidates should form");

        assert_eq!(candidates.nodes.len(), 9);
        assert_eq!(candidates.tets.len(), 12);
        assert_eq!(candidates.interior_seed_points.len(), 1);
        assert!(candidates
            .tets
            .iter()
            .all(|tet| tet.volume_m3 > 0.0 && tet.aspect_ratio.is_finite()));
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
        assert!(candidates
            .nodes
            .iter()
            .any(|node| matches!(node.source, TetCandidateNodeSource::InteriorSeed)));
    }

    #[test]
    fn samples_bounded_interior_seed_points_from_closed_component() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();

        let candidates = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                interior_target_size_m: Some(0.4),
                max_interior_seed_points: 8,
                ..TetCandidateOptions::default()
            },
        )
        .expect("Tet candidates should form");

        assert_eq!(candidates.interior_seed_points.len(), 8);
        assert_eq!(candidates.interior_seed_points[0], [0.5, 0.5, 0.5]);
        assert!(candidates.interior_seed_points.iter().all(|point| {
            point
                .iter()
                .all(|coordinate| *coordinate > 0.0 && *coordinate < 1.0)
        }));
        assert_eq!(candidates.nodes.len(), 16);
        assert!(candidates.tets.len() > 12);
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
        assert!(candidates
            .tets
            .iter()
            .all(|tet| tet.volume_m3 > 0.0 && tet.aspect_ratio <= 1.0 / 0.15));
    }

    #[test]
    fn rejects_invalid_candidate_options() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let err = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                min_volume_m3: 0.0,
                ..TetCandidateOptions::default()
            },
        )
        .expect_err("invalid options should fail");

        assert_eq!(err, TetCandidateError::InvalidOptions);
    }

    fn cube_surface_and_volume_candidates(
    ) -> (crate::SurfaceDiscretization, crate::VolumeCandidateSet) {
        let topology = extract_source_topology(&cube_geometry()).expect("topology should extract");
        let surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");
        let volume_candidates =
            prepare_volume_candidates(&surface, VolumeCandidateOptions::default())
                .expect("volume candidates should prepare");
        (surface, volume_candidates)
    }

    fn cube_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_tet_candidate_cube".to_string(),
            source: GeometrySource {
                path: "/fixtures/generic_cube.step".to_string(),
                sha256: "generic-cube".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 1,
            meshes: vec![MeshDescriptor {
                mesh_id: "cube_surface".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 8,
                element_count: 12,
            }],
            surface_meshes: vec![SurfaceMesh::new(
                "cube_surface",
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                vec![
                    [0, 2, 1],
                    [0, 3, 2],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ],
            )],
            regions: vec![
                Region {
                    region_id: "root".to_string(),
                    name: "root".to_string(),
                    tag: None,
                    cad_ownership: None,
                },
                Region {
                    region_id: "tip".to_string(),
                    name: "tip".to_string(),
                    tag: None,
                    cad_ownership: None,
                },
            ],
            region_entity_mappings: vec![
                RegionEntityMapping::new(
                    "root",
                    "cube_surface",
                    EntityKind::Face,
                    vec![EntityIdRange::new(0, 6)],
                ),
                RegionEntityMapping::new(
                    "tip",
                    "cube_surface",
                    EntityKind::Face,
                    vec![EntityIdRange::new(6, 6)],
                ),
            ],
            diagnostics: Vec::new(),
        }
    }
}
