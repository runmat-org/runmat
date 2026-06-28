use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    predicate::{
        add, distance, distance_squared, point_triangle_distance, ray_triangle_intersection, scale,
        tet_centroid, tet_circumsphere, tet_circumsphere_contains_point, tet_edge_aspect_ratio,
        tet_signed_volume, triangle_centroid, PointInClosedSurface, Triangle3,
    },
    spatial_index::{Aabb3, LinearSpatialIndex, SpatialEntry, UniformGridSpatialIndex},
    surface::{SurfaceDiscretization, SurfaceElement},
    tolerance::MeshingTolerance,
    volume_candidate::{VolumeCandidateComponent, VolumeCandidateSet},
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetCandidateOptions {
    pub min_volume_m3: f64,
    pub max_aspect_ratio: f64,
    pub interior_target_size_m: Option<f64>,
    pub max_interior_seed_points: usize,
    pub max_global_insertion_points: usize,
    pub allow_fan_fallback: bool,
    pub dense_recovery_layer_count: usize,
    pub max_dense_recovery_nodes: usize,
    pub max_refinement_passes: usize,
    pub max_radius_edge_ratio: f64,
    pub sizing_compliance_tolerance: f64,
    pub max_quality_recovery_seed_candidates: usize,
    pub max_optimization_passes: usize,
    pub smoothing_relaxation: f64,
    pub sliver_aspect_ratio: f64,
}

impl Default for TetCandidateOptions {
    fn default() -> Self {
        Self {
            min_volume_m3: 1.0e-18,
            max_aspect_ratio: 1.0e6,
            interior_target_size_m: None,
            max_interior_seed_points: 1,
            max_global_insertion_points: 512,
            allow_fan_fallback: true,
            dense_recovery_layer_count: 4,
            max_dense_recovery_nodes: 20_000,
            max_refinement_passes: 0,
            max_radius_edge_ratio: 3.0,
            sizing_compliance_tolerance: 0.25,
            max_quality_recovery_seed_candidates: 16,
            max_optimization_passes: 0,
            smoothing_relaxation: 0.35,
            sliver_aspect_ratio: 20.0,
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
    pub recovery: TetRecoveryReport,
    pub total_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetRecoveryReport {
    pub component_count: usize,
    pub insertion_component_count: usize,
    pub fan_fallback_component_count: usize,
    pub recovered_component_ratio: f64,
    pub total_candidate_volume_ratio: f64,
    pub max_aspect_ratio: f64,
    pub refinement_pass_count: usize,
    pub refinement_point_count: usize,
    pub max_radius_edge_ratio: f64,
    pub sizing_violation_count: usize,
    pub optimization_pass_count: usize,
    pub smoothed_point_count: usize,
    pub sliver_candidate_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetCandidateError {
    MissingSurfaceNode { node_id: u32 },
    MissingSurfaceElement { element_id: u32 },
    InvalidOptions,
    RecoveryFailed { component_id: u32 },
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
            Self::RecoveryFailed { component_id } => write!(
                formatter,
                "Tet candidate recovery failed for component {component_id}"
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
    let mut insertion_component_count = 0_usize;
    let mut fan_fallback_component_count = 0_usize;
    let mut refinement_pass_count = 0_usize;
    let mut refinement_point_count = 0_usize;
    let mut sizing_violation_count = 0_usize;
    let mut optimization_pass_count = 0_usize;
    let mut smoothed_point_count = 0_usize;
    let mut sliver_candidate_count = 0_usize;
    for component in &volume_candidates.components {
        let tolerance =
            MeshingTolerance::from_bounds(component.bounds_min_m, component.bounds_max_m);
        let mut component_seed_points = sample_component_interior_points(
            component,
            surface,
            &surface_elements,
            options,
            tolerance,
        )?;
        let refinement = refine_component_seed_points(
            component,
            &mut component_seed_points,
            &surface_nodes,
            &surface_elements,
            surface,
            options,
            tolerance,
            next_node_id,
        )?;
        refinement_pass_count += refinement.pass_count;
        refinement_point_count += refinement.inserted_point_count;
        sizing_violation_count += refinement.sizing_violation_count;
        let optimization = smooth_component_seed_points(
            component,
            &mut component_seed_points,
            &surface_nodes,
            &surface_elements,
            surface,
            options,
            tolerance,
            next_node_id,
        )?;
        optimization_pass_count += optimization.pass_count;
        smoothed_point_count += optimization.smoothed_point_count;
        sliver_candidate_count += optimization.sliver_candidate_count;
        if dense_component_for_global_insertion(component, component_seed_points.len(), options) {
            add_dense_recovery_layer_points(
                component,
                &mut component_seed_points,
                &surface_nodes,
                &surface_elements,
                options,
                tolerance,
            )?;
        }

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

        let insertion_status = append_component_insertion_tets(
            component,
            &component_seed_node_ids,
            &component_seed_points,
            &surface_nodes,
            &surface_elements,
            surface,
            options,
            tolerance,
            &mut tets,
        )?;
        if insertion_status.accepted {
            insertion_component_count += 1;
        } else {
            if !options.allow_fan_fallback {
                return Err(TetCandidateError::RecoveryFailed {
                    component_id: component.component_id,
                });
            }
            fan_fallback_component_count += 1;
            let fan_seed_point = select_component_fan_seed_point(
                component,
                &component_seed_points,
                &surface_nodes,
                &surface_elements,
                options,
            )?;
            let fan_seed_node_id = component_seed_node_ids[component_seed_points
                .iter()
                .position(|point| tolerance.point_nearly_equal(*point, fan_seed_point, 1.0))
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
    let expected_volume_m3 = volume_candidates.total_volume_m3;
    let total_candidate_volume_ratio = if expected_volume_m3 > f64::EPSILON {
        total_volume_m3 / expected_volume_m3
    } else {
        1.0
    };
    let max_aspect_ratio = tets
        .iter()
        .map(|tet| tet.aspect_ratio)
        .fold(0.0_f64, f64::max);
    let max_radius_edge_ratio = max_tet_radius_edge_ratio(&nodes, &tets);
    let component_count = volume_candidates.components.len();
    Ok(TetCandidateSet {
        nodes,
        tets,
        interior_seed_points,
        recovery: TetRecoveryReport {
            component_count,
            insertion_component_count,
            fan_fallback_component_count,
            recovered_component_ratio: if component_count == 0 {
                1.0
            } else {
                insertion_component_count as f64 / component_count as f64
            },
            total_candidate_volume_ratio,
            max_aspect_ratio,
            refinement_pass_count,
            refinement_point_count,
            max_radius_edge_ratio,
            sizing_violation_count,
            optimization_pass_count,
            smoothed_point_count,
            sliver_candidate_count,
        },
        total_volume_m3,
    })
}

fn validate_options(options: TetCandidateOptions) -> Result<(), TetCandidateError> {
    if !options.min_volume_m3.is_finite()
        || options.min_volume_m3 <= 0.0
        || !options.max_aspect_ratio.is_finite()
        || options.max_aspect_ratio <= 0.0
        || options.max_interior_seed_points == 0
        || options.max_global_insertion_points < 4
        || options.dense_recovery_layer_count < 2
        || options.max_dense_recovery_nodes == 0
        || options.max_quality_recovery_seed_candidates == 0
        || !options.max_radius_edge_ratio.is_finite()
        || options.max_radius_edge_ratio <= 0.0
        || !options.sizing_compliance_tolerance.is_finite()
        || options.sizing_compliance_tolerance < 0.0
        || !options.smoothing_relaxation.is_finite()
        || !(0.0..=1.0).contains(&options.smoothing_relaxation)
        || !options.sliver_aspect_ratio.is_finite()
        || options.sliver_aspect_ratio <= 0.0
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
        let aspect_ratio = tet_edge_aspect_ratio(points);
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

fn append_candidate_tet(
    component: &VolumeCandidateComponent,
    element: &SurfaceElement,
    mut node_ids: [u32; 4],
    points: [[f64; 3]; 4],
    options: TetCandidateOptions,
    tets: &mut Vec<TetCandidate>,
) {
    let mut signed_volume_m3 = tet_signed_volume(points);
    if signed_volume_m3 < 0.0 {
        node_ids.swap(1, 2);
        signed_volume_m3 = -signed_volume_m3;
    }
    let volume_m3 = signed_volume_m3.abs();
    if volume_m3 < options.min_volume_m3 {
        return;
    }
    let aspect_ratio = tet_edge_aspect_ratio(points);
    if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
        return;
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

fn add_dense_recovery_layer_points(
    component: &VolumeCandidateComponent,
    seed_points: &mut Vec<[f64; 3]>,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
    tolerance: MeshingTolerance,
) -> Result<(), TetCandidateError> {
    if options.dense_recovery_layer_count < 2 || seed_points.is_empty() {
        return Ok(());
    }
    let fan_seed_point = select_component_fan_seed_point(
        component,
        seed_points,
        surface_nodes,
        surface_elements,
        options,
    )?;
    if let Some(fan_index) = seed_points
        .iter()
        .position(|point| tolerance.point_nearly_equal(*point, fan_seed_point, 1.0))
    {
        seed_points.swap(0, fan_index);
    } else {
        seed_points.push(fan_seed_point);
        let fan_index = seed_points.len() - 1;
        seed_points.swap(0, fan_index);
    }
    let max_extra_points = options
        .max_dense_recovery_nodes
        .saturating_sub(component.node_ids.len())
        .saturating_sub(seed_points.len());
    if max_extra_points == 0 {
        return Ok(());
    }

    let mut inserted = 0_usize;
    for node_id in component_surface_node_ids(component, surface_elements)? {
        if inserted >= max_extra_points {
            break;
        }
        let boundary_point = *surface_nodes
            .get(&node_id)
            .ok_or(TetCandidateError::MissingSurfaceNode { node_id })?;
        for layer in 1..options.dense_recovery_layer_count {
            if inserted >= max_extra_points {
                break;
            }
            let point = dense_recovery_layer_point(boundary_point, fan_seed_point, layer, options);
            if contains_point(seed_points, point, tolerance) {
                continue;
            }
            seed_points.push(point);
            inserted += 1;
        }
    }
    Ok(())
}

fn dense_recovery_layer_point(
    boundary_point: [f64; 3],
    fan_seed_point: [f64; 3],
    layer: usize,
    options: TetCandidateOptions,
) -> [f64; 3] {
    let t = layer as f64 / options.dense_recovery_layer_count as f64;
    [
        boundary_point[0] * (1.0 - t) + fan_seed_point[0] * t,
        boundary_point[1] * (1.0 - t) + fan_seed_point[1] * t,
        boundary_point[2] * (1.0 - t) + fan_seed_point[2] * t,
    ]
}

fn node_id_for_seed_point(
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
    point: [f64; 3],
    tolerance: MeshingTolerance,
) -> Option<u32> {
    seed_node_ids
        .iter()
        .zip(seed_points.iter())
        .find_map(|(node_id, seed_point)| {
            tolerance
                .point_nearly_equal(*seed_point, point, 1.0)
                .then_some(*node_id)
        })
}

fn seed_nodes_by_point(
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
) -> BTreeMap<[u64; 3], (u32, [f64; 3])> {
    seed_node_ids
        .iter()
        .zip(seed_points.iter())
        .map(|(node_id, point)| (point_key(*point), (*node_id, *point)))
        .collect()
}

fn point_key(point: [f64; 3]) -> [u64; 3] {
    [point[0].to_bits(), point[1].to_bits(), point[2].to_bits()]
}

fn append_component_insertion_tets(
    component: &VolumeCandidateComponent,
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    surface: &SurfaceDiscretization,
    options: TetCandidateOptions,
    tolerance: MeshingTolerance,
    tets: &mut Vec<TetCandidate>,
) -> Result<InsertionStatus, TetCandidateError> {
    if dense_component_for_global_insertion(component, seed_node_ids.len(), options) {
        if let Some(layered_tets) = quality_recovery_layered_star_tets(
            component,
            seed_node_ids,
            seed_points,
            surface_nodes,
            surface_elements,
            options,
            tolerance,
        )? {
            for mut tet in layered_tets {
                tet.tet_id = tets.len() as u32;
                tets.push(tet);
            }
            return Ok(InsertionStatus {
                accepted: true,
                volume_ratio: 1.0,
                max_aspect_ratio: tets
                    .iter()
                    .filter(|tet| tet.component_id == component.component_id)
                    .map(|tet| tet.aspect_ratio)
                    .fold(0.0_f64, f64::max),
            });
        }
        if let Some(star_tets) = quality_recovery_star_tets(
            component,
            seed_node_ids,
            seed_points,
            surface_nodes,
            surface_elements,
            options,
        )? {
            for mut tet in star_tets {
                tet.tet_id = tets.len() as u32;
                tets.push(tet);
            }
            return Ok(InsertionStatus {
                accepted: true,
                volume_ratio: 1.0,
                max_aspect_ratio: tets
                    .iter()
                    .filter(|tet| tet.component_id == component.component_id)
                    .map(|tet| tet.aspect_ratio)
                    .fold(0.0_f64, f64::max),
            });
        }
        return Ok(InsertionStatus::rejected(0.0, f64::INFINITY));
    }

    let (_draft_status, accepted_tets) = component_insertion_tet_drafts(
        component,
        seed_node_ids,
        seed_points,
        surface_nodes,
        surface_elements,
        surface,
        options,
        tolerance,
    )?;
    let status = insertion_tet_status(component, &accepted_tets, options);
    if status.accepted {
        for mut tet in accepted_tets {
            tet.tet_id = tets.len() as u32;
            tets.push(tet);
        }
        return Ok(status);
    }
    if let Some(star_tets) = quality_recovery_star_tets(
        component,
        seed_node_ids,
        seed_points,
        surface_nodes,
        surface_elements,
        options,
    )? {
        for mut tet in star_tets {
            tet.tet_id = tets.len() as u32;
            tets.push(tet);
        }
        return Ok(InsertionStatus {
            accepted: true,
            volume_ratio: 1.0,
            max_aspect_ratio: tets
                .iter()
                .filter(|tet| tet.component_id == component.component_id)
                .map(|tet| tet.aspect_ratio)
                .fold(0.0_f64, f64::max),
        });
    }
    Ok(status)
}

fn quality_recovery_star_tets(
    component: &VolumeCandidateComponent,
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    if seed_node_ids.is_empty() || seed_node_ids.len() != seed_points.len() {
        return Ok(None);
    }
    let fan_seed_point = select_component_fan_seed_point(
        component,
        seed_points,
        surface_nodes,
        surface_elements,
        options,
    )?;
    let fan_seed_node_id = seed_node_ids[seed_points
        .iter()
        .position(|point| {
            distance_squared(*point, fan_seed_point)
                <= MeshingTolerance::from_bounds(component.bounds_min_m, component.bounds_max_m)
                    .absolute_m
                    .powi(2)
        })
        .unwrap_or(0)];
    let mut tets = Vec::<TetCandidate>::new();
    append_component_tets(
        component,
        fan_seed_node_id,
        fan_seed_point,
        surface_nodes,
        surface_elements,
        options,
        &mut tets,
    )?;
    let status = insertion_tet_status(component, &tets, options);
    Ok(status.accepted.then_some(tets))
}

fn quality_recovery_layered_star_tets(
    component: &VolumeCandidateComponent,
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
    tolerance: MeshingTolerance,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    if options.dense_recovery_layer_count < 2
        || seed_node_ids.is_empty()
        || seed_node_ids.len() != seed_points.len()
    {
        return Ok(None);
    }
    let Some(fan_seed_point) = select_layered_fan_seed_point(
        component,
        seed_points,
        surface_nodes,
        surface_elements,
        options,
    )?
    else {
        return Ok(None);
    };
    let Some(fan_seed_node_id) =
        node_id_for_seed_point(seed_node_ids, seed_points, fan_seed_point, tolerance)
    else {
        return Ok(None);
    };
    let seed_nodes_by_point = seed_nodes_by_point(seed_node_ids, seed_points);
    let mut tets = Vec::<TetCandidate>::new();
    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        append_layered_surface_tets(
            component,
            element,
            fan_seed_node_id,
            fan_seed_point,
            &seed_nodes_by_point,
            surface_nodes,
            options,
            &mut tets,
        )?;
    }
    let status = insertion_tet_status(component, &tets, options);
    Ok(status.accepted.then_some(tets))
}

fn select_layered_fan_seed_point(
    component: &VolumeCandidateComponent,
    seed_points: &[[f64; 3]],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
) -> Result<Option<[f64; 3]>, TetCandidateError> {
    let seed_keys = seed_points
        .iter()
        .map(|point| point_key(*point))
        .collect::<BTreeSet<_>>();
    let mut best_score = None::<FanSeedScore>;
    for point in quality_recovery_seed_candidates(seed_points, options) {
        if !dense_recovery_layer_nodes_exist(
            component,
            point,
            surface_nodes,
            surface_elements,
            options,
            &seed_keys,
        )? {
            continue;
        }
        let score =
            score_fan_seed_point(component, point, surface_nodes, surface_elements, options)?;
        if best_score.is_none_or(|best| fan_seed_score_is_better(score, best)) {
            best_score = Some(score);
        }
    }
    Ok(best_score.map(|score| score.point))
}

fn dense_recovery_layer_nodes_exist(
    component: &VolumeCandidateComponent,
    fan_seed_point: [f64; 3],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
    seed_keys: &BTreeSet<[u64; 3]>,
) -> Result<bool, TetCandidateError> {
    for node_id in component_surface_node_ids(component, surface_elements)? {
        let boundary_point = *surface_nodes
            .get(&node_id)
            .ok_or(TetCandidateError::MissingSurfaceNode { node_id })?;
        for layer in 1..options.dense_recovery_layer_count {
            let point = dense_recovery_layer_point(boundary_point, fan_seed_point, layer, options);
            if !seed_keys.contains(&point_key(point)) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn component_surface_node_ids(
    component: &VolumeCandidateComponent,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
) -> Result<BTreeSet<u32>, TetCandidateError> {
    let mut node_ids = BTreeSet::<u32>::new();
    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        node_ids.extend(element.node_ids);
    }
    Ok(node_ids)
}

#[allow(clippy::too_many_arguments)]
fn append_layered_surface_tets(
    component: &VolumeCandidateComponent,
    element: &SurfaceElement,
    fan_seed_node_id: u32,
    fan_seed_point: [f64; 3],
    seed_nodes_by_point: &BTreeMap<[u64; 3], (u32, [f64; 3])>,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
    tets: &mut Vec<TetCandidate>,
) -> Result<(), TetCandidateError> {
    let mut layer_node_ids = Vec::<[u32; 3]>::new();
    let mut layer_points = Vec::<[[f64; 3]; 3]>::new();
    layer_node_ids.push(element.node_ids);
    layer_points.push([
        *surface_nodes
            .get(&element.node_ids[0])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: element.node_ids[0],
            })?,
        *surface_nodes
            .get(&element.node_ids[1])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: element.node_ids[1],
            })?,
        *surface_nodes
            .get(&element.node_ids[2])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: element.node_ids[2],
            })?,
    ]);
    for layer in 1..options.dense_recovery_layer_count {
        let mut ids = [0_u32; 3];
        let mut points = [[0.0; 3]; 3];
        for corner in 0..3 {
            let point =
                dense_recovery_layer_point(layer_points[0][corner], fan_seed_point, layer, options);
            let Some((node_id, coordinates_m)) =
                seed_nodes_by_point.get(&point_key(point)).copied()
            else {
                return Ok(());
            };
            ids[corner] = node_id;
            points[corner] = coordinates_m;
        }
        layer_node_ids.push(ids);
        layer_points.push(points);
    }

    for layer in 0..layer_node_ids.len().saturating_sub(1) {
        let outer_ids = layer_node_ids[layer];
        let inner_ids = layer_node_ids[layer + 1];
        let outer_points = layer_points[layer];
        let inner_points = layer_points[layer + 1];
        append_candidate_tet(
            component,
            element,
            [outer_ids[0], outer_ids[1], outer_ids[2], inner_ids[0]],
            [
                outer_points[0],
                outer_points[1],
                outer_points[2],
                inner_points[0],
            ],
            options,
            tets,
        );
        append_candidate_tet(
            component,
            element,
            [outer_ids[1], inner_ids[1], outer_ids[2], inner_ids[0]],
            [
                outer_points[1],
                inner_points[1],
                outer_points[2],
                inner_points[0],
            ],
            options,
            tets,
        );
        append_candidate_tet(
            component,
            element,
            [inner_ids[1], inner_ids[2], outer_ids[2], inner_ids[0]],
            [
                inner_points[1],
                inner_points[2],
                outer_points[2],
                inner_points[0],
            ],
            options,
            tets,
        );
    }
    let last_index = layer_node_ids.len() - 1;
    append_candidate_tet(
        component,
        element,
        [
            layer_node_ids[last_index][0],
            layer_node_ids[last_index][1],
            layer_node_ids[last_index][2],
            fan_seed_node_id,
        ],
        [
            layer_points[last_index][0],
            layer_points[last_index][1],
            layer_points[last_index][2],
            fan_seed_point,
        ],
        options,
        tets,
    );
    Ok(())
}

fn component_insertion_tet_drafts(
    component: &VolumeCandidateComponent,
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    surface: &SurfaceDiscretization,
    options: TetCandidateOptions,
    tolerance: MeshingTolerance,
) -> Result<(InsertionStatus, Vec<TetCandidate>), TetCandidateError> {
    if seed_node_ids.is_empty() || seed_node_ids.len() != seed_points.len() {
        return Ok((InsertionStatus::rejected(0.0, 0.0), Vec::new()));
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

    let classifier =
        ComponentSurfaceClassifier::new(component, surface, surface_elements, tolerance)?;
    let candidate_tets = tetrahedralize_points(&points);
    let mut accepted_tets = Vec::<TetCandidate>::new();
    for candidate in candidate_tets {
        let tet_points = candidate.vertices.map(|index| points[index].coordinates_m);
        let centroid = tet_centroid(tet_points);
        if !classifier.contains_point(centroid) {
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
        let aspect_ratio = tet_edge_aspect_ratio(tet_points);
        if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
            continue;
        }
        let source_surface_element_id =
            nearest_surface_element_id(centroid, surface, surface_elements, classifier.index())?;
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
    Ok((
        insertion_tet_status(component, &accepted_tets, options),
        accepted_tets,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SeedRefinementSummary {
    pass_count: usize,
    inserted_point_count: usize,
    sizing_violation_count: usize,
}

fn refine_component_seed_points(
    component: &VolumeCandidateComponent,
    seed_points: &mut Vec<[f64; 3]>,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    surface: &SurfaceDiscretization,
    options: TetCandidateOptions,
    tolerance: MeshingTolerance,
    first_seed_node_id: u32,
) -> Result<SeedRefinementSummary, TetCandidateError> {
    if options.max_refinement_passes == 0
        || seed_points.len() >= options.max_interior_seed_points
        || options.interior_target_size_m.is_none()
        || dense_component_for_global_insertion(component, seed_points.len(), options)
    {
        return Ok(SeedRefinementSummary {
            pass_count: 0,
            inserted_point_count: 0,
            sizing_violation_count: 0,
        });
    }

    let classifier =
        ComponentSurfaceClassifier::new(component, surface, surface_elements, tolerance)?;
    let mut pass_count = 0_usize;
    let mut inserted_point_count = 0_usize;
    let mut sizing_violation_count = 0_usize;
    for _ in 0..options.max_refinement_passes {
        if seed_points.len() >= options.max_interior_seed_points {
            break;
        }
        let seed_node_ids = seed_node_ids(first_seed_node_id, seed_points.len());
        let (status, candidate_tets) = component_insertion_tet_drafts(
            component,
            &seed_node_ids,
            seed_points,
            surface_nodes,
            surface_elements,
            surface,
            options,
            tolerance,
        )?;
        if candidate_tets.is_empty() {
            break;
        }

        let point_budget = options.max_interior_seed_points - seed_points.len();
        let refinement_points = refinement_points_for_tets(
            &candidate_tets,
            surface_nodes,
            &seed_node_ids,
            seed_points,
            tolerance,
            &classifier,
            options,
            point_budget,
        )?;
        sizing_violation_count += refinement_points.sizing_violation_count;
        if refinement_points.points.is_empty() {
            break;
        }
        pass_count += 1;
        for point in refinement_points.points {
            if seed_points.len() >= options.max_interior_seed_points {
                break;
            }
            if !contains_point(seed_points, point, tolerance) {
                seed_points.push(point);
                inserted_point_count += 1;
            }
        }
        if status.accepted && inserted_point_count == 0 {
            break;
        }
    }

    Ok(SeedRefinementSummary {
        pass_count,
        inserted_point_count,
        sizing_violation_count,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct RefinementPointSet {
    points: Vec<[f64; 3]>,
    sizing_violation_count: usize,
}

#[allow(clippy::too_many_arguments)]
fn refinement_points_for_tets(
    tets: &[TetCandidate],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
    tolerance: MeshingTolerance,
    classifier: &ComponentSurfaceClassifier,
    options: TetCandidateOptions,
    point_budget: usize,
) -> Result<RefinementPointSet, TetCandidateError> {
    let Some(target_size_m) = options.interior_target_size_m else {
        return Ok(RefinementPointSet {
            points: Vec::new(),
            sizing_violation_count: 0,
        });
    };
    let all_nodes = candidate_node_coordinates(surface_nodes, seed_node_ids, seed_points);
    let mut ranked = Vec::<([f64; 3], f64, bool)>::new();
    let mut sizing_violation_count = 0_usize;
    for tet in tets {
        let points = candidate_tet_points(tet, &all_nodes)?;
        let radius_edge_ratio = tet_radius_edge_ratio(points, tolerance);
        let max_edge_m = tet_max_edge_length(points);
        let sizing_violation =
            max_edge_m > target_size_m * (1.0 + options.sizing_compliance_tolerance);
        if sizing_violation {
            sizing_violation_count += 1;
        }
        if radius_edge_ratio <= options.max_radius_edge_ratio && !sizing_violation {
            continue;
        }
        let point = tet_circumsphere(points, tolerance)
            .map(|(center, _)| center)
            .unwrap_or_else(|| tet_centroid(points));
        let point = if classifier.contains_point(point) {
            point
        } else {
            tet_centroid(points)
        };
        if !classifier.contains_point(point) {
            continue;
        }
        ranked.push((
            point,
            radius_edge_ratio.max(max_edge_m / target_size_m),
            sizing_violation,
        ));
    }
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| right.2.cmp(&left.2))
    });
    let mut points = Vec::<[f64; 3]>::new();
    for (point, _, _) in ranked {
        if points.len() >= point_budget {
            break;
        }
        if contains_point(seed_points, point, tolerance)
            || contains_point(&points, point, tolerance)
        {
            continue;
        }
        points.push(point);
    }
    Ok(RefinementPointSet {
        points,
        sizing_violation_count,
    })
}

fn seed_node_ids(first_seed_node_id: u32, seed_count: usize) -> Vec<u32> {
    (0..seed_count)
        .map(|offset| first_seed_node_id.saturating_add(offset as u32))
        .collect()
}

fn candidate_node_coordinates(
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
) -> BTreeMap<u32, [f64; 3]> {
    let mut nodes = surface_nodes.clone();
    for (node_id, point) in seed_node_ids.iter().zip(seed_points.iter()) {
        nodes.insert(*node_id, *point);
    }
    nodes
}

fn candidate_tet_points(
    tet: &TetCandidate,
    nodes: &BTreeMap<u32, [f64; 3]>,
) -> Result<[[f64; 3]; 4], TetCandidateError> {
    Ok([
        *nodes
            .get(&tet.node_ids[0])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: tet.node_ids[0],
            })?,
        *nodes
            .get(&tet.node_ids[1])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: tet.node_ids[1],
            })?,
        *nodes
            .get(&tet.node_ids[2])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: tet.node_ids[2],
            })?,
        *nodes
            .get(&tet.node_ids[3])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: tet.node_ids[3],
            })?,
    ])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SmoothingSummary {
    pass_count: usize,
    smoothed_point_count: usize,
    sliver_candidate_count: usize,
}

fn smooth_component_seed_points(
    component: &VolumeCandidateComponent,
    seed_points: &mut Vec<[f64; 3]>,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    surface: &SurfaceDiscretization,
    options: TetCandidateOptions,
    tolerance: MeshingTolerance,
    first_seed_node_id: u32,
) -> Result<SmoothingSummary, TetCandidateError> {
    if options.max_optimization_passes == 0
        || seed_points.is_empty()
        || dense_component_for_global_insertion(component, seed_points.len(), options)
    {
        return Ok(SmoothingSummary {
            pass_count: 0,
            smoothed_point_count: 0,
            sliver_candidate_count: 0,
        });
    }

    let classifier =
        ComponentSurfaceClassifier::new(component, surface, surface_elements, tolerance)?;
    let mut pass_count = 0_usize;
    let mut smoothed_point_count = 0_usize;
    let mut sliver_candidate_count = 0_usize;

    for _ in 0..options.max_optimization_passes {
        let seed_node_ids = seed_node_ids(first_seed_node_id, seed_points.len());
        let (current_status, current_tets) = component_insertion_tet_drafts(
            component,
            &seed_node_ids,
            seed_points,
            surface_nodes,
            surface_elements,
            surface,
            options,
            tolerance,
        )?;
        if current_tets.is_empty() {
            break;
        }
        let current_quality = CandidateQualitySnapshot::from_tets(&current_tets, options);
        sliver_candidate_count += current_quality.sliver_count;
        let proposed = smoothed_seed_points(
            seed_points,
            &seed_node_ids,
            &current_tets,
            surface_nodes,
            &classifier,
            options,
        )?;
        if proposed == *seed_points {
            break;
        }
        let (proposed_status, proposed_tets) = component_insertion_tet_drafts(
            component,
            &seed_node_ids,
            &proposed,
            surface_nodes,
            surface_elements,
            surface,
            options,
            tolerance,
        )?;
        if !proposed_status.accepted && current_status.accepted {
            break;
        }
        let proposed_quality = CandidateQualitySnapshot::from_tets(&proposed_tets, options);
        if !candidate_quality_is_no_worse(proposed_quality, current_quality) {
            break;
        }
        let moved_count = seed_points
            .iter()
            .zip(proposed.iter())
            .filter(|(left, right)| !tolerance.point_nearly_equal(**left, **right, 1.0))
            .count();
        if moved_count == 0 {
            break;
        }
        *seed_points = proposed;
        pass_count += 1;
        smoothed_point_count += moved_count;
    }

    Ok(SmoothingSummary {
        pass_count,
        smoothed_point_count,
        sliver_candidate_count,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CandidateQualitySnapshot {
    max_aspect_ratio: f64,
    max_radius_edge_ratio: f64,
    volume_ratio_error: f64,
    sliver_count: usize,
}

impl CandidateQualitySnapshot {
    fn from_tets(tets: &[TetCandidate], options: TetCandidateOptions) -> Self {
        let max_aspect_ratio = tets
            .iter()
            .map(|tet| tet.aspect_ratio)
            .fold(0.0_f64, f64::max);
        let sliver_count = tets
            .iter()
            .filter(|tet| tet.aspect_ratio > options.sliver_aspect_ratio)
            .count();
        Self {
            max_aspect_ratio,
            max_radius_edge_ratio: 0.0,
            volume_ratio_error: 0.0,
            sliver_count,
        }
    }
}

fn candidate_quality_is_no_worse(
    proposed: CandidateQualitySnapshot,
    current: CandidateQualitySnapshot,
) -> bool {
    proposed.sliver_count <= current.sliver_count
        && proposed.max_aspect_ratio <= current.max_aspect_ratio * (1.0 + 1.0e-9)
}

#[allow(clippy::too_many_arguments)]
fn smoothed_seed_points(
    seed_points: &[[f64; 3]],
    seed_node_ids: &[u32],
    tets: &[TetCandidate],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    classifier: &ComponentSurfaceClassifier,
    options: TetCandidateOptions,
) -> Result<Vec<[f64; 3]>, TetCandidateError> {
    let seed_index = seed_node_ids
        .iter()
        .enumerate()
        .map(|(index, node_id)| (*node_id, index))
        .collect::<BTreeMap<_, _>>();
    let all_nodes = candidate_node_coordinates(surface_nodes, seed_node_ids, seed_points);
    let mut sums = vec![[0.0, 0.0, 0.0]; seed_points.len()];
    let mut counts = vec![0_usize; seed_points.len()];
    for tet in tets {
        let points = candidate_tet_points(tet, &all_nodes)?;
        let centroid = tet_centroid(points);
        for node_id in tet.node_ids {
            let Some(index) = seed_index.get(&node_id).copied() else {
                continue;
            };
            for axis in 0..3 {
                sums[index][axis] += centroid[axis];
            }
            counts[index] += 1;
        }
    }

    let mut proposed = seed_points.to_vec();
    for (index, point) in seed_points.iter().enumerate() {
        if counts[index] == 0 {
            continue;
        }
        let average = [
            sums[index][0] / counts[index] as f64,
            sums[index][1] / counts[index] as f64,
            sums[index][2] / counts[index] as f64,
        ];
        let candidate = [
            point[0] * (1.0 - options.smoothing_relaxation)
                + average[0] * options.smoothing_relaxation,
            point[1] * (1.0 - options.smoothing_relaxation)
                + average[1] * options.smoothing_relaxation,
            point[2] * (1.0 - options.smoothing_relaxation)
                + average[2] * options.smoothing_relaxation,
        ];
        if classifier.contains_point(candidate) {
            proposed[index] = candidate;
        }
    }
    Ok(proposed)
}

fn max_tet_radius_edge_ratio(nodes: &[TetCandidateNode], tets: &[TetCandidate]) -> f64 {
    let nodes = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    tets.iter()
        .filter_map(|tet| candidate_tet_points(tet, &nodes).ok())
        .map(|points| tet_radius_edge_ratio(points, MeshingTolerance::default()))
        .filter(|ratio| ratio.is_finite())
        .fold(0.0_f64, f64::max)
}

fn tet_radius_edge_ratio(points: [[f64; 3]; 4], tolerance: MeshingTolerance) -> f64 {
    let Some((_, radius_squared)) = tet_circumsphere(points, tolerance) else {
        return f64::INFINITY;
    };
    let min_edge = tet_min_edge_length(points);
    if min_edge <= f64::EPSILON {
        return f64::INFINITY;
    }
    radius_squared.sqrt() / min_edge
}

fn tet_min_edge_length(points: [[f64; 3]; 4]) -> f64 {
    let mut min_edge = f64::INFINITY;
    for left_index in 0..4 {
        for right_index in (left_index + 1)..4 {
            min_edge = min_edge.min(distance(points[left_index], points[right_index]));
        }
    }
    min_edge
}

fn tet_max_edge_length(points: [[f64; 3]; 4]) -> f64 {
    let mut max_edge = 0.0_f64;
    for left_index in 0..4 {
        for right_index in (left_index + 1)..4 {
            max_edge = max_edge.max(distance(points[left_index], points[right_index]));
        }
    }
    max_edge
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct InsertionStatus {
    accepted: bool,
    volume_ratio: f64,
    max_aspect_ratio: f64,
}

impl InsertionStatus {
    fn rejected(volume_ratio: f64, max_aspect_ratio: f64) -> Self {
        Self {
            accepted: false,
            volume_ratio,
            max_aspect_ratio,
        }
    }
}

fn insertion_tet_status(
    component: &VolumeCandidateComponent,
    tets: &[TetCandidate],
    options: TetCandidateOptions,
) -> InsertionStatus {
    if tets.is_empty() || component.volume_m3 <= f64::EPSILON {
        return InsertionStatus::rejected(0.0, 0.0);
    }
    let total_volume_m3 = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
    let volume_ratio = total_volume_m3 / component.volume_m3;
    let max_aspect_ratio = tets
        .iter()
        .map(|tet| tet.aspect_ratio)
        .fold(0.0_f64, f64::max);
    InsertionStatus {
        accepted: (0.90..=1.10).contains(&volume_ratio)
            && max_aspect_ratio <= options.sliver_aspect_ratio,
        volume_ratio,
        max_aspect_ratio,
    }
}

fn dense_component_for_global_insertion(
    component: &VolumeCandidateComponent,
    seed_count: usize,
    options: TetCandidateOptions,
) -> bool {
    component.node_ids.len().saturating_add(seed_count) > options.max_global_insertion_points
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
                MeshingTolerance::default(),
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
            if tet_signed_volume(points_for_tet).abs()
                > MeshingTolerance::default().volume_epsilon(1.0)
            {
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

fn nearest_surface_element_id(
    point: [f64; 3],
    surface: &SurfaceDiscretization,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    triangle_index: &LinearSpatialIndex<u32>,
) -> Result<u32, TetCandidateError> {
    let mut best = None::<(u32, f64)>;
    for entry in triangle_index.entries() {
        let element_id = entry.payload;
        let element = surface_elements
            .get(&element_id)
            .ok_or(TetCandidateError::MissingSurfaceElement { element_id })?;
        let centroid = triangle_centroid(surface_element_points(surface, element)?);
        let distance_squared = distance_squared(point, centroid);
        if best.is_none_or(|(_, best_distance)| distance_squared < best_distance) {
            best = Some((element.element_id, distance_squared));
        }
    }
    best.map(|(element_id, _)| element_id)
        .ok_or(TetCandidateError::MissingSurfaceElement { element_id: 0 })
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
    for point in quality_recovery_seed_candidates(seed_points, options) {
        let score =
            score_fan_seed_point(component, point, surface_nodes, surface_elements, options)?;
        if best_score.is_none_or(|best| fan_seed_score_is_better(score, best)) {
            best_score = Some(score);
        }
    }
    Ok(best_score
        .map(|score| score.point)
        .unwrap_or_else(|| component_interior_point(component)))
}

fn quality_recovery_seed_candidates(
    seed_points: &[[f64; 3]],
    options: TetCandidateOptions,
) -> Vec<[f64; 3]> {
    if seed_points.len() <= options.max_quality_recovery_seed_candidates {
        return seed_points.to_vec();
    }
    let candidate_count = options.max_quality_recovery_seed_candidates;
    let mut candidates = Vec::<[f64; 3]>::with_capacity(candidate_count);
    for candidate_index in 0..candidate_count {
        let seed_index = if candidate_count == 1 {
            0
        } else {
            candidate_index * (seed_points.len() - 1) / (candidate_count - 1)
        };
        let point = seed_points[seed_index];
        if !candidates.contains(&point) {
            candidates.push(point);
        }
    }
    candidates
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
        let aspect_ratio = tet_edge_aspect_ratio(points);
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
    tolerance: MeshingTolerance,
) -> Result<Vec<[f64; 3]>, TetCandidateError> {
    let mut points = Vec::<[f64; 3]>::new();
    let center = component_interior_point(component);
    let classifier =
        ComponentSurfaceClassifier::new(component, surface, surface_elements, tolerance)?;
    if classifier.contains_point(center) {
        points.push(center);
    }

    if let Some(target_size_m) = options.interior_target_size_m {
        let spans = [
            component.bounds_max_m[0] - component.bounds_min_m[0],
            component.bounds_max_m[1] - component.bounds_min_m[1],
            component.bounds_max_m[2] - component.bounds_min_m[2],
        ];
        let divisions = seed_grid_divisions(spans, target_size_m, options.max_interior_seed_points);
        for x_index in 0..divisions[0] {
            for y_index in 0..divisions[1] {
                for z_index in 0..divisions[2] {
                    if points.len() >= options.max_interior_seed_points {
                        return Ok(points);
                    }
                    let point = [
                        grid_center(component.bounds_min_m[0], spans[0], divisions[0], x_index),
                        grid_center(component.bounds_min_m[1], spans[1], divisions[1], y_index),
                        grid_center(component.bounds_min_m[2], spans[2], divisions[2], z_index),
                    ];
                    if contains_point(&points, point, tolerance) {
                        continue;
                    }
                    if classifier.contains_point(point) {
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

fn seed_grid_divisions(spans: [f64; 3], target_size_m: f64, max_seed_points: usize) -> [usize; 3] {
    let max_grid_points = max_seed_points.max(1);
    let mut divisions = spans.map(|span| ((span / target_size_m).ceil() as usize).max(1));
    while divisions.iter().product::<usize>() > max_grid_points {
        let axis = (0..3).max_by_key(|axis| divisions[*axis]).unwrap_or(0);
        if divisions[axis] <= 1 {
            break;
        }
        divisions[axis] -= 1;
    }
    divisions
}

fn grid_center(minimum: f64, span: f64, divisions: usize, index: usize) -> f64 {
    minimum + span * (index as f64 + 0.5) / divisions as f64
}

fn contains_point(points: &[[f64; 3]], candidate: [f64; 3], tolerance: MeshingTolerance) -> bool {
    points
        .iter()
        .any(|point| tolerance.point_nearly_equal(*point, candidate, 1.0))
}

#[derive(Debug, Clone)]
struct ComponentSurfaceClassifier {
    triangles_by_element_id: BTreeMap<u32, Triangle3>,
    triangle_index: LinearSpatialIndex<u32>,
    grid_index: UniformGridSpatialIndex<u32>,
    tolerance: MeshingTolerance,
}

impl ComponentSurfaceClassifier {
    fn new(
        component: &VolumeCandidateComponent,
        surface: &SurfaceDiscretization,
        surface_elements: &BTreeMap<u32, &SurfaceElement>,
        tolerance: MeshingTolerance,
    ) -> Result<Self, TetCandidateError> {
        let mut triangles_by_element_id = BTreeMap::<u32, Triangle3>::new();
        let mut entries = Vec::<SpatialEntry<u32>>::new();
        for element_id in &component.surface_element_ids {
            let element = surface_elements.get(element_id).ok_or(
                TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                },
            )?;
            let triangle = surface_element_points(surface, element)?;
            triangles_by_element_id.insert(*element_id, triangle);
            entries.push(SpatialEntry {
                bounds: Aabb3::from_triangle(triangle).expanded(tolerance),
                payload: *element_id,
            });
        }
        let triangle_index = LinearSpatialIndex::with_entries(entries.clone());
        let grid_index = UniformGridSpatialIndex::from_entries(entries);
        Ok(Self {
            triangles_by_element_id,
            triangle_index,
            grid_index,
            tolerance,
        })
    }

    fn index(&self) -> &LinearSpatialIndex<u32> {
        &self.triangle_index
    }

    fn contains_point(&self, point: [f64; 3]) -> bool {
        if self.point_is_on_boundary(point) {
            return true;
        }
        let epsilon = self.tolerance.absolute_m;
        let probes = [
            ([1.0, 0.0, 0.0], [-0.37, 0.19, 0.11]),
            ([0.0, 1.0, 0.0], [0.13, -0.41, 0.23]),
            ([0.0, 0.0, 1.0], [0.17, 0.29, -0.43]),
        ];
        let inside_votes = probes
            .into_iter()
            .filter(|(direction, jitter)| {
                self.ray_has_odd_surface_intersections(
                    add(point, scale(*jitter, epsilon)),
                    *direction,
                )
            })
            .count();
        matches!(
            if inside_votes >= 2 {
                PointInClosedSurface::Inside
            } else {
                PointInClosedSurface::Outside
            },
            PointInClosedSurface::Inside | PointInClosedSurface::OnBoundary
        )
    }

    fn point_is_on_boundary(&self, point: [f64; 3]) -> bool {
        self.grid_index.query_point(point).into_iter().any(|entry| {
            self.triangles_by_element_id
                .get(&entry.payload)
                .is_some_and(|triangle| {
                    point_triangle_distance(point, *triangle) <= self.tolerance.absolute_m
                })
        })
    }

    fn ray_has_odd_surface_intersections(&self, origin: [f64; 3], direction: [f64; 3]) -> bool {
        let mut intersections = Vec::<f64>::new();
        for entry in self.grid_index.query_ray(origin, direction) {
            let Some(triangle) = self.triangles_by_element_id.get(&entry.payload).copied() else {
                continue;
            };
            let Some(hit) = ray_triangle_intersection(origin, direction, triangle, self.tolerance)
            else {
                continue;
            };
            if hit.distance > self.tolerance.absolute_m {
                intersections.push(hit.distance);
            }
        }
        intersections.sort_by(f64::total_cmp);
        intersections.dedup_by(|left, right| (*left - *right).abs() <= self.tolerance.absolute_m);
        intersections.len() % 2 == 1
    }
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
        assert_eq!(candidates.recovery.component_count, 1);
        assert_eq!(
            candidates.recovery.insertion_component_count
                + candidates.recovery.fan_fallback_component_count,
            1
        );
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

        assert!(candidates.interior_seed_points.len() > 1);
        assert!(candidates.interior_seed_points.len() <= 8);
        assert_eq!(candidates.interior_seed_points[0], [0.5, 0.5, 0.5]);
        assert!(candidates.interior_seed_points.iter().all(|point| {
            point
                .iter()
                .all(|coordinate| *coordinate > 0.0 && *coordinate < 1.0)
        }));
        assert_eq!(
            candidates.nodes.len(),
            8 + candidates.interior_seed_points.len()
        );
        assert!(candidates.tets.len() > 12);
        assert_eq!(candidates.recovery.insertion_component_count, 1);
        assert_eq!(candidates.recovery.fan_fallback_component_count, 0);
        assert_eq!(candidates.recovery.recovered_component_ratio, 1.0);
        assert!((candidates.recovery.total_candidate_volume_ratio - 1.0).abs() < 1.0e-12);
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
        assert!(candidates
            .tets
            .iter()
            .all(|tet| tet.volume_m3 > 0.0 && tet.aspect_ratio <= 1.0 / 0.15));
    }

    #[test]
    fn refinement_pass_adds_bounded_seed_points_for_candidate_quality() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();

        let candidates = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                interior_target_size_m: Some(0.8),
                max_interior_seed_points: 12,
                max_refinement_passes: 1,
                max_radius_edge_ratio: 1.0,
                ..TetCandidateOptions::default()
            },
        )
        .expect("Tet candidates should form with refinement");

        assert!(candidates.interior_seed_points.len() > 9);
        assert!(candidates.interior_seed_points.len() <= 12);
        assert_eq!(candidates.recovery.refinement_pass_count, 1);
        assert!(candidates.recovery.refinement_point_count > 0);
        assert!(candidates.recovery.max_radius_edge_ratio.is_finite());
        assert_eq!(candidates.recovery.fan_fallback_component_count, 0);
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn optimization_pass_smooths_interior_seed_points_without_fallback() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();

        let candidates = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                interior_target_size_m: Some(0.8),
                max_interior_seed_points: 12,
                max_optimization_passes: 2,
                smoothing_relaxation: 0.2,
                allow_fan_fallback: false,
                ..TetCandidateOptions::default()
            },
        )
        .expect("Tet candidates should form with smoothing");

        assert_eq!(candidates.recovery.fan_fallback_component_count, 0);
        assert!(candidates.recovery.optimization_pass_count <= 2);
        assert!(
            candidates.recovery.optimization_pass_count > 0
                || candidates.recovery.sliver_candidate_count == 0
        );
        assert!(
            candidates.recovery.smoothed_point_count <= candidates.interior_seed_points.len() * 2
        );
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn dense_components_use_quality_recovery_without_global_insertion() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();

        let candidates = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                max_global_insertion_points: 4,
                allow_fan_fallback: false,
                ..TetCandidateOptions::default()
            },
        )
        .expect("dense component should use bounded quality recovery");

        assert_eq!(candidates.recovery.insertion_component_count, 1);
        assert_eq!(candidates.recovery.fan_fallback_component_count, 0);
        assert_eq!(candidates.recovery.recovered_component_ratio, 1.0);
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn dense_components_do_not_fall_through_to_global_insertion_when_recovery_fails() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();

        let err = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                max_global_insertion_points: 4,
                allow_fan_fallback: false,
                sliver_aspect_ratio: 1.0,
                ..TetCandidateOptions::default()
            },
        )
        .expect_err("dense rejected recovery should not fall through to global insertion");

        assert_eq!(err, TetCandidateError::RecoveryFailed { component_id: 0 });
    }

    #[test]
    fn quality_recovery_seed_candidates_are_bounded_and_deterministic() {
        let seed_points = (0..10)
            .map(|index| [index as f64, 0.0, 0.0])
            .collect::<Vec<_>>();

        let candidates = quality_recovery_seed_candidates(
            &seed_points,
            TetCandidateOptions {
                max_quality_recovery_seed_candidates: 4,
                ..TetCandidateOptions::default()
            },
        );

        assert_eq!(
            candidates,
            vec![
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [9.0, 0.0, 0.0]
            ]
        );
    }

    #[test]
    fn rejects_unrecovered_insertion_when_fallback_is_disabled() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();

        let err = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                max_aspect_ratio: 1.01,
                allow_fan_fallback: false,
                ..TetCandidateOptions::default()
            },
        )
        .expect_err("disabled fallback should expose recovery failure");

        assert_eq!(err, TetCandidateError::RecoveryFailed { component_id: 0 });
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
                cad_evaluators: Vec::new(),
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
