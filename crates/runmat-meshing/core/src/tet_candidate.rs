use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    constrained_cavity::{
        constrained_cavity_from_selected_tets_with_anchor_trim,
        evaluate_constrained_cavity_refill_candidates, ConstrainedCavityExtractionError,
        ConstrainedCavityNode, ConstrainedCavityRefillOptions, ConstrainedCavityValidationError,
    },
    predicate::{
        add, distance, distance_squared, point_triangle_distance, ray_triangle_intersection, scale,
        tet_centroid, tet_circumsphere, tet_circumsphere_contains_point, tet_edge_aspect_ratio,
        tet_scaled_jacobian, tet_signed_volume, triangle_centroid, Triangle3,
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
    pub requested_refinement_points: [[f64; 3]; 16],
    pub requested_refinement_point_count: usize,
    pub max_requested_refinement_candidates_per_point: usize,
    pub max_interior_seed_points: usize,
    pub max_global_insertion_points: usize,
    pub allow_fan_fallback: bool,
    pub dense_recovery_layer_count: usize,
    pub max_dense_recovery_nodes: usize,
    pub max_refinement_passes: usize,
    pub max_radius_edge_ratio: f64,
    pub sizing_compliance_tolerance: f64,
    pub min_scaled_jacobian: f64,
    pub max_quality_recovery_seed_candidates: usize,
    pub max_exact_quality_repair_targets: usize,
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
            requested_refinement_points: [[0.0; 3]; 16],
            requested_refinement_point_count: 0,
            max_requested_refinement_candidates_per_point: 16,
            max_interior_seed_points: 1,
            max_global_insertion_points: 512,
            allow_fan_fallback: false,
            dense_recovery_layer_count: 4,
            max_dense_recovery_nodes: 20_000,
            max_refinement_passes: 0,
            max_radius_edge_ratio: 3.0,
            sizing_compliance_tolerance: 0.25,
            min_scaled_jacobian: 0.15,
            max_quality_recovery_seed_candidates: 16,
            max_exact_quality_repair_targets: 512,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetCandidateNodeSource {
    Surface,
    BoundaryRecovery,
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
    #[serde(default)]
    pub exact_scaled_jacobian: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetCandidateSet {
    pub nodes: Vec<TetCandidateNode>,
    pub tets: Vec<TetCandidate>,
    pub interior_seed_points: Vec<[f64; 3]>,
    #[serde(default)]
    pub accepted_requested_refinement_points: Vec<[f64; 3]>,
    #[serde(default)]
    pub accepted_requested_refinement_sample_indices: Vec<usize>,
    #[serde(default)]
    pub dropped_requested_refinement_sample_indices: Vec<usize>,
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
    #[serde(default)]
    pub requested_refinement_point_count: usize,
    #[serde(default)]
    pub accepted_requested_refinement_candidate_count: usize,
    #[serde(default)]
    pub accepted_requested_refinement_point_count: usize,
    #[serde(default)]
    pub accepted_requested_refinement_surrogate_point_count: usize,
    #[serde(default)]
    pub rejected_requested_refinement_point_count: usize,
    #[serde(default)]
    pub requested_refinement_rejected_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub dropped_requested_refinement_point_count: usize,
    #[serde(default)]
    pub requested_refinement_dropped_by_reason: BTreeMap<String, usize>,
    pub max_radius_edge_ratio: f64,
    pub sizing_violation_count: usize,
    pub min_exact_scaled_jacobian: f64,
    pub exact_scaled_jacobian_below_threshold_count: usize,
    #[serde(default)]
    pub exact_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub optimization_pass_count: usize,
    pub smoothed_point_count: usize,
    pub sliver_candidate_count: usize,
    #[serde(default)]
    pub sliver_removed_count: usize,
    #[serde(default)]
    pub optimization_target_seed_count: usize,
    #[serde(default)]
    pub optimization_skipped_target_seed_count: usize,
    #[serde(default)]
    pub optimization_rejected_edit_count: usize,
    #[serde(default)]
    pub optimization_initial_max_aspect_ratio: f64,
    #[serde(default)]
    pub optimization_final_max_aspect_ratio: f64,
    #[serde(default)]
    pub optimization_initial_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub optimization_final_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub untangling_pass_count: usize,
    #[serde(default)]
    pub untangling_initial_near_singular_count: usize,
    #[serde(default)]
    pub untangling_final_near_singular_count: usize,
    #[serde(default)]
    pub untangling_relocated_seed_count: usize,
    #[serde(default)]
    pub untangling_reconnected_edge_star_count: usize,
    #[serde(default)]
    pub untangling_reconnected_boundary_adjacent_cavity_count: usize,
    #[serde(default)]
    pub untangling_reconnected_node_adjacent_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_repair_pass_count: usize,
    #[serde(default)]
    pub exact_quality_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_reconnection_quality_gain_count: usize,
    #[serde(default)]
    pub exact_quality_face_neighbor_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_connected_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_node_adjacent_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_boundary_adjacent_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_expanded_connected_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_split_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_seed_star_collapse_count: usize,
    #[serde(default)]
    pub exact_quality_seed_star_relocation_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_total_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_general_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_boundary_adjacent_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_node_adjacent_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_interior_seed_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_edge_star_count: usize,
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

#[derive(Debug, Clone, PartialEq)]
struct RetainedRequestedRefinement {
    points: Vec<[f64; 3]>,
    sample_indices: Vec<usize>,
    dropped_sample_indices: Vec<usize>,
}

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
    let mut accepted_requested_refinement_seed_points = Vec::<(u32, [f64; 3], usize)>::new();
    let mut insertion_component_count = 0_usize;
    let mut fan_fallback_component_count = 0_usize;
    let mut refinement_pass_count = 0_usize;
    let mut refinement_point_count = 0_usize;
    let requested_refinement_point_count = options.requested_refinement_point_count;
    let mut accepted_requested_refinement_point_count = 0_usize;
    let mut accepted_requested_refinement_surrogate_point_count = 0_usize;
    let mut rejected_requested_refinement_point_count = 0_usize;
    let mut attempted_requested_refinement_ids = BTreeSet::<usize>::new();
    let mut duplicate_requested_refinement_ids = BTreeSet::<usize>::new();
    let mut sizing_violation_count = 0_usize;
    let mut optimization_pass_count = 0_usize;
    let mut smoothed_point_count = 0_usize;
    let mut sliver_candidate_count = 0_usize;
    let mut optimization_rejected_edit_count = 0_usize;
    let mut optimization_quality = OptimizationQualityAggregate::default();
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
        accepted_requested_refinement_point_count += refinement.accepted_requested_point_count;
        accepted_requested_refinement_surrogate_point_count +=
            refinement.accepted_requested_surrogate_point_count;
        rejected_requested_refinement_point_count += refinement.rejected_requested_point_count;
        attempted_requested_refinement_ids
            .extend(refinement.attempted_requested_ids.iter().copied());
        duplicate_requested_refinement_ids
            .extend(refinement.duplicate_requested_ids.iter().copied());
        sizing_violation_count += refinement.sizing_violation_count;
        let dense_component =
            dense_component_for_global_insertion(component, component_seed_points.len(), options);
        if dense_component {
            add_dense_recovery_layer_points(
                component,
                &mut component_seed_points,
                &surface_nodes,
                &surface_elements,
                options,
                tolerance,
            )?;
            add_sweep_recovery_layer_points(
                component,
                &mut component_seed_points,
                &surface_nodes,
                &surface_elements,
                options,
                tolerance,
            )?;
        }
        let optimization = if dense_component {
            SmoothingSummary::empty()
        } else {
            smooth_component_seed_points(
                component,
                &mut component_seed_points,
                &surface_nodes,
                &surface_elements,
                surface,
                options,
                tolerance,
                next_node_id,
            )?
        };
        optimization_pass_count += optimization.pass_count;
        smoothed_point_count += optimization.smoothed_point_count;
        sliver_candidate_count += optimization.sliver_candidate_count;
        optimization_rejected_edit_count += optimization.rejected_edit_count;
        optimization_quality.record(optimization);

        let node_classifier =
            ComponentSurfaceClassifier::new(component, surface, &surface_elements, tolerance)?;
        let mut component_seed_node_ids = Vec::<u32>::with_capacity(component_seed_points.len());
        let mut component_seed_node_sources =
            Vec::<TetCandidateNodeSource>::with_capacity(component_seed_points.len());
        for point in &component_seed_points {
            let node_id = next_node_id;
            next_node_id = next_node_id.saturating_add(1);
            component_seed_node_ids.push(node_id);
            let source = seed_node_source(*point, &node_classifier);
            component_seed_node_sources.push(source);
            nodes.push(TetCandidateNode {
                node_id,
                coordinates_m: *point,
                source,
            });
        }
        for accepted in &refinement.accepted_requested_points {
            let index = accepted.seed_index;
            if index < component_seed_node_ids.len() {
                accepted_requested_refinement_seed_points.push((
                    component_seed_node_ids[index],
                    accepted.requested_point,
                    accepted.requested_id,
                ));
            }
        }
        interior_seed_points.extend(
            component_seed_points
                .iter()
                .zip(component_seed_node_sources.iter())
                .filter_map(|(point, source)| {
                    matches!(source, TetCandidateNodeSource::InteriorSeed).then_some(*point)
                }),
        );

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
    let untangling =
        untangle_near_singular_tets(&mut nodes, &mut tets, &mut interior_seed_points, options)?;
    let repair = repair_exact_quality_tets(
        &mut nodes,
        &mut tets,
        &mut interior_seed_points,
        &mut next_node_id,
        options,
    )?;
    if tets.is_empty() {
        return Err(TetCandidateError::EmptyCandidateSet);
    }
    let retained_node_ids = nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    let accepted_requested_refinement_candidate_count =
        accepted_requested_refinement_seed_points.len();
    let accepted_requested_refinement_ids = accepted_requested_refinement_seed_points
        .iter()
        .map(|(_, _, requested_id)| *requested_id)
        .collect::<BTreeSet<_>>();
    let retained_requested_refinement = retained_requested_refinement_points(
        accepted_requested_refinement_seed_points,
        &retained_node_ids,
    );
    let accepted_requested_refinement_points = retained_requested_refinement.points;
    let accepted_requested_refinement_sample_indices = retained_requested_refinement.sample_indices;
    let dropped_requested_refinement_sample_indices =
        retained_requested_refinement.dropped_sample_indices;
    accepted_requested_refinement_point_count = accepted_requested_refinement_points.len();
    accepted_requested_refinement_surrogate_point_count =
        retained_requested_refinement_surrogate_count(
            &accepted_requested_refinement_points,
            &accepted_requested_refinement_sample_indices,
            options,
        );
    let dropped_requested_refinement_point_count =
        dropped_requested_refinement_sample_indices.len();
    let requested_refinement_rejected_by_reason = requested_refinement_rejection_reasons(
        options.requested_refinement_point_count,
        &attempted_requested_refinement_ids,
        &duplicate_requested_refinement_ids,
        &accepted_requested_refinement_ids,
    );
    rejected_requested_refinement_point_count = requested_refinement_rejected_by_reason
        .values()
        .copied()
        .sum();
    let requested_refinement_dropped_by_reason =
        requested_refinement_drop_reasons(&dropped_requested_refinement_sample_indices);
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
    let quality_summary = tet_candidate_quality_summary(&nodes, &tets, options)?;
    let unrepaired_quality = remaining_exact_quality_violation_counts(&nodes, &tets, options);
    let component_count = volume_candidates.components.len();
    Ok(TetCandidateSet {
        nodes,
        tets,
        interior_seed_points,
        accepted_requested_refinement_points,
        accepted_requested_refinement_sample_indices,
        dropped_requested_refinement_sample_indices,
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
            requested_refinement_point_count,
            accepted_requested_refinement_candidate_count,
            accepted_requested_refinement_point_count,
            accepted_requested_refinement_surrogate_point_count,
            rejected_requested_refinement_point_count,
            requested_refinement_rejected_by_reason,
            dropped_requested_refinement_point_count,
            requested_refinement_dropped_by_reason,
            max_radius_edge_ratio: quality_summary.max_radius_edge_ratio,
            sizing_violation_count,
            min_exact_scaled_jacobian: quality_summary.min_exact_scaled_jacobian,
            exact_scaled_jacobian_below_threshold_count: quality_summary
                .exact_scaled_jacobian_below_threshold_count,
            exact_scaled_jacobian_bins: quality_summary.exact_scaled_jacobian_bins,
            optimization_pass_count,
            smoothed_point_count,
            sliver_candidate_count,
            sliver_removed_count: optimization_quality.sliver_removed_count(),
            optimization_target_seed_count: optimization_quality.target_seed_count(),
            optimization_skipped_target_seed_count: optimization_quality
                .skipped_target_seed_count(),
            optimization_rejected_edit_count,
            optimization_initial_max_aspect_ratio: optimization_quality.initial_max_aspect_ratio(),
            optimization_final_max_aspect_ratio: optimization_quality.final_max_aspect_ratio(),
            optimization_initial_min_exact_scaled_jacobian: optimization_quality
                .initial_min_exact_scaled_jacobian(),
            optimization_final_min_exact_scaled_jacobian: optimization_quality
                .final_min_exact_scaled_jacobian(),
            untangling_pass_count: untangling.pass_count,
            untangling_initial_near_singular_count: untangling.initial_near_singular_count,
            untangling_final_near_singular_count: untangling.final_near_singular_count,
            untangling_relocated_seed_count: untangling.relocated_seed_count,
            untangling_reconnected_edge_star_count: untangling.reconnected_edge_star_count,
            untangling_reconnected_boundary_adjacent_cavity_count: untangling
                .reconnected_boundary_adjacent_cavity_count,
            untangling_reconnected_node_adjacent_cavity_count: untangling
                .reconnected_node_adjacent_cavity_count,
            exact_quality_repair_pass_count: repair.pass_count,
            exact_quality_reconnected_cavity_count: repair.reconnected_cavity_count,
            exact_quality_reconnection_quality_gain_count: repair.reconnection_quality_gain_count,
            exact_quality_face_neighbor_reconnected_cavity_count: repair
                .face_neighbor_reconnected_cavity_count,
            exact_quality_connected_reconnected_cavity_count: repair
                .connected_reconnected_cavity_count,
            exact_quality_node_adjacent_reconnected_cavity_count: repair
                .node_adjacent_reconnected_cavity_count,
            exact_quality_boundary_adjacent_reconnected_cavity_count: repair
                .boundary_adjacent_reconnected_cavity_count,
            exact_quality_expanded_connected_reconnected_cavity_count: repair
                .expanded_connected_reconnected_cavity_count,
            exact_quality_split_cavity_count: repair.split_cavity_count,
            exact_quality_seed_star_collapse_count: repair.seed_star_collapse_count,
            exact_quality_seed_star_relocation_count: repair.seed_star_relocation_count,
            exact_quality_unrepaired_total_count: unrepaired_quality.total_count,
            exact_quality_unrepaired_general_cavity_count: unrepaired_quality.general_cavity_count,
            exact_quality_unrepaired_boundary_adjacent_count: unrepaired_quality
                .boundary_adjacent_count,
            exact_quality_unrepaired_node_adjacent_count: unrepaired_quality.node_adjacent_count,
            exact_quality_unrepaired_interior_seed_count: unrepaired_quality.interior_seed_count,
            exact_quality_unrepaired_edge_star_count: unrepaired_quality.edge_star_count,
        },
        total_volume_m3,
    })
}

fn retained_requested_refinement_points(
    accepted_seed_points: Vec<(u32, [f64; 3], usize)>,
    retained_node_ids: &BTreeSet<u32>,
) -> RetainedRequestedRefinement {
    let accepted_sample_ids = accepted_seed_points
        .iter()
        .map(|(_, _, requested_id)| *requested_id)
        .collect::<BTreeSet<_>>();
    let retained = accepted_seed_points
        .into_iter()
        .filter(|(node_id, _, _)| retained_node_ids.contains(node_id))
        .collect::<Vec<_>>();
    let retained_sample_ids = retained
        .iter()
        .map(|(_, _, requested_id)| *requested_id)
        .collect::<BTreeSet<_>>();
    let dropped_sample_indices = accepted_sample_ids
        .into_iter()
        .filter(|requested_id| !retained_sample_ids.contains(requested_id))
        .collect::<Vec<_>>();
    RetainedRequestedRefinement {
        points: retained.iter().map(|(_, point, _)| *point).collect(),
        sample_indices: retained
            .iter()
            .map(|(_, _, requested_id)| *requested_id)
            .collect(),
        dropped_sample_indices,
    }
}

fn retained_requested_refinement_surrogate_count(
    accepted_points: &[[f64; 3]],
    accepted_sample_indices: &[usize],
    options: TetCandidateOptions,
) -> usize {
    accepted_points
        .iter()
        .zip(accepted_sample_indices)
        .filter(|(point, requested_id)| {
            **requested_id < options.requested_refinement_point_count
                && distance_squared(**point, options.requested_refinement_points[**requested_id])
                    > 1.0e-24
        })
        .count()
}

fn requested_refinement_rejection_reasons(
    requested_count: usize,
    attempted_ids: &BTreeSet<usize>,
    duplicate_ids: &BTreeSet<usize>,
    accepted_ids: &BTreeSet<usize>,
) -> BTreeMap<String, usize> {
    let mut by_reason = BTreeMap::<String, usize>::new();
    for requested_id in 0..requested_count {
        if accepted_ids.contains(&requested_id) {
            continue;
        }
        let Some(reason) = (if attempted_ids.contains(&requested_id) {
            Some("quality_or_recovery")
        } else if duplicate_ids.contains(&requested_id) {
            None
        } else {
            None
        }) else {
            continue;
        };
        *by_reason.entry(reason.to_string()).or_default() += 1;
    }
    by_reason
}

fn requested_refinement_drop_reasons(dropped_sample_indices: &[usize]) -> BTreeMap<String, usize> {
    let mut by_reason = BTreeMap::<String, usize>::new();
    if !dropped_sample_indices.is_empty() {
        by_reason.insert(
            "not_retained_after_repair".to_string(),
            dropped_sample_indices.len(),
        );
    }
    by_reason
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
        || options.max_exact_quality_repair_targets == 0
        || !options.max_radius_edge_ratio.is_finite()
        || options.max_radius_edge_ratio <= 0.0
        || !options.sizing_compliance_tolerance.is_finite()
        || options.sizing_compliance_tolerance < 0.0
        || !options.min_scaled_jacobian.is_finite()
        || options.min_scaled_jacobian < 0.0
        || !options.smoothing_relaxation.is_finite()
        || !(0.0..=1.0).contains(&options.smoothing_relaxation)
        || !options.sliver_aspect_ratio.is_finite()
        || options.sliver_aspect_ratio <= 0.0
        || options
            .interior_target_size_m
            .is_some_and(|size| !size.is_finite() || size <= 0.0)
        || options.requested_refinement_point_count > options.requested_refinement_points.len()
        || options
            .requested_refinement_points
            .iter()
            .take(options.requested_refinement_point_count)
            .any(|point| point.iter().any(|value| !value.is_finite()))
        || options.max_requested_refinement_candidates_per_point == 0
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
        let exact_scaled_jacobian = tet_scaled_jacobian(points);
        tets.push(TetCandidate {
            tet_id: tets.len() as u32,
            component_id: component.component_id,
            node_ids,
            source_surface_element_id: element.element_id,
            region_ids: element.region_ids.clone(),
            volume_m3,
            aspect_ratio,
            exact_scaled_jacobian,
        });
    }
    Ok(())
}

fn append_candidate_tet(
    component: &VolumeCandidateComponent,
    element: &SurfaceElement,
    node_ids: [u32; 4],
    points: [[f64; 3]; 4],
    options: TetCandidateOptions,
    tets: &mut Vec<TetCandidate>,
) {
    if let Some(tet) = candidate_tet(component, element, node_ids, points, options) {
        tets.push(tet);
    }
}

fn candidate_tet(
    component: &VolumeCandidateComponent,
    element: &SurfaceElement,
    mut node_ids: [u32; 4],
    points: [[f64; 3]; 4],
    options: TetCandidateOptions,
) -> Option<TetCandidate> {
    let mut signed_volume_m3 = tet_signed_volume(points);
    if signed_volume_m3 < 0.0 {
        node_ids.swap(1, 2);
        signed_volume_m3 = -signed_volume_m3;
    }
    let volume_m3 = signed_volume_m3.abs();
    if volume_m3 < options.min_volume_m3 {
        return None;
    }
    let aspect_ratio = tet_edge_aspect_ratio(points);
    if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
        return None;
    }
    let exact_scaled_jacobian = tet_scaled_jacobian(points);
    Some(TetCandidate {
        tet_id: 0,
        component_id: component.component_id,
        node_ids,
        source_surface_element_id: element.element_id,
        region_ids: element.region_ids.clone(),
        volume_m3,
        aspect_ratio,
        exact_scaled_jacobian,
    })
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
    if dense_recovery_component_exceeds_budget(component, options) {
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
    if let Some(status) = append_sweep_tets(
        component,
        seed_node_ids,
        seed_points,
        surface_nodes,
        surface_elements,
        options,
        tolerance,
        tets,
    )? {
        return Ok(status);
    }

    if dense_component_for_global_insertion(component, seed_node_ids.len(), options) {
        if dense_recovery_component_exceeds_budget(component, options) {
            return Ok(InsertionStatus::rejected(0.0, f64::INFINITY));
        }
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

fn append_sweep_tets(
    component: &VolumeCandidateComponent,
    seed_node_ids: &[u32],
    seed_points: &[[f64; 3]],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
    tolerance: MeshingTolerance,
    tets: &mut Vec<TetCandidate>,
) -> Result<Option<InsertionStatus>, TetCandidateError> {
    let Some(axis) = sweep_axis(component, surface_nodes, surface_elements, tolerance)? else {
        return Ok(None);
    };
    let mut lower_elements = Vec::<&SurfaceElement>::new();
    let mut upper_node_by_key = BTreeMap::<[i64; 2], (u32, [f64; 3])>::new();

    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        let node_points = element_node_points(element, surface_nodes)?;
        if points_on_axis_plane(&node_points, axis, component.bounds_min_m[axis], tolerance) {
            lower_elements.push(element);
        } else if points_on_axis_plane(&node_points, axis, component.bounds_max_m[axis], tolerance)
        {
            for (node_id, point) in element.node_ids.iter().zip(node_points) {
                upper_node_by_key
                    .entry(projected_key(point, axis, tolerance))
                    .or_insert((*node_id, point));
            }
        }
    }
    for (node_id, point) in seed_node_ids.iter().zip(seed_points.iter()) {
        if points_on_axis_plane(&[*point; 3], axis, component.bounds_max_m[axis], tolerance) {
            upper_node_by_key
                .entry(projected_key(*point, axis, tolerance))
                .or_insert((*node_id, *point));
        }
    }

    if lower_elements.is_empty() || upper_node_by_key.is_empty() {
        return Ok(None);
    }

    let seed_nodes_by_point = seed_nodes_by_point(seed_node_ids, seed_points);
    let start_len = tets.len();
    for lower in lower_elements {
        let lower_keys = element_projected_keys(lower, surface_nodes, axis, tolerance)?;
        let mut upper_ids = [0_u32; 3];
        let mut upper_points = [[0.0; 3]; 3];
        for (index, key) in lower_keys.iter().enumerate() {
            let Some((node_id, point)) = upper_node_by_key.get(key).copied() else {
                tets.truncate(start_len);
                return Ok(None);
            };
            upper_ids[index] = node_id;
            upper_points[index] = point;
        }
        let lower_points = element_node_points(lower, surface_nodes)?;
        let Some((layer_node_ids, layer_points)) = sweep_column_layers(
            lower.node_ids,
            upper_ids,
            lower_points,
            upper_points,
            &seed_nodes_by_point,
            options,
        ) else {
            tets.truncate(start_len);
            return Ok(None);
        };
        for layer in 0..layer_node_ids.len().saturating_sub(1) {
            append_consistent_sweep_frustum_tets(
                component,
                lower,
                layer_node_ids[layer],
                layer_node_ids[layer + 1],
                layer_points[layer],
                layer_points[layer + 1],
                options,
                tets,
            );
        }
    }

    let status = insertion_tet_status(component, &tets[start_len..], options);
    if status.accepted {
        for (offset, tet) in tets[start_len..].iter_mut().enumerate() {
            tet.tet_id = (start_len + offset) as u32;
        }
        Ok(Some(status))
    } else {
        tets.truncate(start_len);
        Ok(None)
    }
}

fn add_sweep_recovery_layer_points(
    component: &VolumeCandidateComponent,
    seed_points: &mut Vec<[f64; 3]>,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
    tolerance: MeshingTolerance,
) -> Result<(), TetCandidateError> {
    if options.dense_recovery_layer_count < 2 {
        return Ok(());
    }
    let Some(axis) = sweep_axis(component, surface_nodes, surface_elements, tolerance)? else {
        return Ok(());
    };
    let mut lower_node_by_key = BTreeMap::<[i64; 2], [f64; 3]>::new();
    let mut upper_node_by_key = BTreeMap::<[i64; 2], [f64; 3]>::new();
    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        let node_points = element_node_points(element, surface_nodes)?;
        if points_on_axis_plane(&node_points, axis, component.bounds_min_m[axis], tolerance) {
            for point in node_points {
                lower_node_by_key
                    .entry(projected_key(point, axis, tolerance))
                    .or_insert(point);
            }
        } else if points_on_axis_plane(&node_points, axis, component.bounds_max_m[axis], tolerance)
        {
            for point in node_points {
                upper_node_by_key
                    .entry(projected_key(point, axis, tolerance))
                    .or_insert(point);
            }
        }
    }
    let max_extra_points = options
        .max_dense_recovery_nodes
        .saturating_sub(component.node_ids.len())
        .saturating_sub(seed_points.len());
    if max_extra_points == 0 {
        return Ok(());
    }
    let mut inserted = 0_usize;
    for (key, lower_point) in &lower_node_by_key {
        if upper_node_by_key.contains_key(key) {
            continue;
        }
        if inserted >= max_extra_points {
            return Ok(());
        }
        let mut upper_point = *lower_point;
        upper_point[axis] = component.bounds_max_m[axis];
        if contains_point(seed_points, upper_point, tolerance) {
            continue;
        }
        seed_points.push(upper_point);
        upper_node_by_key.insert(*key, upper_point);
        inserted += 1;
    }
    for (key, lower_point) in lower_node_by_key {
        let Some(upper_point) = upper_node_by_key.get(&key).copied() else {
            continue;
        };
        for layer in 1..options.dense_recovery_layer_count {
            if inserted >= max_extra_points {
                return Ok(());
            }
            let point = sweep_layer_point(lower_point, upper_point, layer, options);
            if contains_point(seed_points, point, tolerance) {
                continue;
            }
            seed_points.push(point);
            inserted += 1;
        }
    }
    Ok(())
}

fn sweep_column_layers(
    lower_ids: [u32; 3],
    upper_ids: [u32; 3],
    lower_points: [[f64; 3]; 3],
    upper_points: [[f64; 3]; 3],
    seed_nodes_by_point: &BTreeMap<[u64; 3], (u32, [f64; 3])>,
    options: TetCandidateOptions,
) -> Option<(Vec<[u32; 3]>, Vec<[[f64; 3]; 3]>)> {
    let layer_count = options.dense_recovery_layer_count.max(1);
    let mut layer_node_ids = Vec::<[u32; 3]>::with_capacity(layer_count + 1);
    let mut layer_points = Vec::<[[f64; 3]; 3]>::with_capacity(layer_count + 1);
    layer_node_ids.push(lower_ids);
    layer_points.push(lower_points);
    for layer in 1..layer_count {
        let mut ids = [0_u32; 3];
        let mut points = [[0.0; 3]; 3];
        for corner in 0..3 {
            let point =
                sweep_layer_point(lower_points[corner], upper_points[corner], layer, options);
            let (node_id, coordinates_m) = seed_nodes_by_point.get(&point_key(point)).copied()?;
            ids[corner] = node_id;
            points[corner] = coordinates_m;
        }
        layer_node_ids.push(ids);
        layer_points.push(points);
    }
    layer_node_ids.push(upper_ids);
    layer_points.push(upper_points);
    Some((layer_node_ids, layer_points))
}

fn sweep_layer_point(
    lower_point: [f64; 3],
    upper_point: [f64; 3],
    layer: usize,
    options: TetCandidateOptions,
) -> [f64; 3] {
    let t = layer as f64 / options.dense_recovery_layer_count.max(1) as f64;
    [
        lower_point[0] * (1.0 - t) + upper_point[0] * t,
        lower_point[1] * (1.0 - t) + upper_point[1] * t,
        lower_point[2] * (1.0 - t) + upper_point[2] * t,
    ]
}

fn append_consistent_sweep_frustum_tets(
    component: &VolumeCandidateComponent,
    element: &SurfaceElement,
    outer_ids: [u32; 3],
    inner_ids: [u32; 3],
    outer_points: [[f64; 3]; 3],
    inner_points: [[f64; 3]; 3],
    options: TetCandidateOptions,
    tets: &mut Vec<TetCandidate>,
) {
    let order = sweep_column_order(outer_ids);
    let outer_ids = order.map(|index| outer_ids[index]);
    let inner_ids = order.map(|index| inner_ids[index]);
    let outer_points = order.map(|index| outer_points[index]);
    let inner_points = order.map(|index| inner_points[index]);
    if let Some(split) = layered_frustum_split(
        component,
        element,
        outer_ids,
        inner_ids,
        outer_points,
        inner_points,
        3,
        options,
    ) {
        tets.extend(split.tets);
    }
}

fn sweep_column_order(node_ids: [u32; 3]) -> [usize; 3] {
    let mut order = [0_usize, 1, 2];
    order.sort_by_key(|index| node_ids[*index]);
    order
}

fn sweep_axis(
    component: &VolumeCandidateComponent,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    tolerance: MeshingTolerance,
) -> Result<Option<usize>, TetCandidateError> {
    let mut best = None::<(usize, f64, usize)>;
    for axis in 0..3 {
        let cap =
            sweep_axis_cap_match(component, surface_nodes, surface_elements, axis, tolerance)?;
        if cap.is_match
            && best.is_none_or(|(_, area_m2, count)| {
                cap.lower_projected_area_m2 > area_m2 + tolerance.absolute_m
                    || ((cap.lower_projected_area_m2 - area_m2).abs() <= tolerance.absolute_m
                        && cap.lower_element_count > count)
            })
        {
            best = Some((axis, cap.lower_projected_area_m2, cap.lower_element_count));
        }
    }
    Ok(best.map(|(axis, _, _)| axis))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SweepAxisCapMatch {
    is_match: bool,
    lower_element_count: usize,
    lower_projected_area_m2: f64,
}

fn sweep_axis_cap_match(
    component: &VolumeCandidateComponent,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    axis: usize,
    tolerance: MeshingTolerance,
) -> Result<SweepAxisCapMatch, TetCandidateError> {
    let span_m = component.bounds_max_m[axis] - component.bounds_min_m[axis];
    if !span_m.is_finite() || span_m <= tolerance.absolute_m {
        return Ok(SweepAxisCapMatch {
            is_match: false,
            lower_element_count: 0,
            lower_projected_area_m2: 0.0,
        });
    }

    let mut lower_keys = BTreeSet::<[i64; 2]>::new();
    let mut upper_keys = BTreeSet::<[i64; 2]>::new();
    let mut lower_element_count = 0_usize;
    let mut upper_element_count = 0_usize;
    let mut lower_projected_area_m2 = 0.0_f64;
    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        let node_points = element_node_points(element, surface_nodes)?;
        if points_on_axis_plane(&node_points, axis, component.bounds_min_m[axis], tolerance) {
            lower_element_count += 1;
            lower_projected_area_m2 += projected_triangle_area_m2(&node_points, axis);
            lower_keys.extend(node_points.map(|point| projected_key(point, axis, tolerance)));
        } else if points_on_axis_plane(&node_points, axis, component.bounds_max_m[axis], tolerance)
        {
            upper_element_count += 1;
            upper_keys.extend(node_points.map(|point| projected_key(point, axis, tolerance)));
        }
    }

    let keys_match = lower_keys == upper_keys;
    let bounds_match = projected_key_bounds(&lower_keys) == projected_key_bounds(&upper_keys);
    Ok(SweepAxisCapMatch {
        is_match: lower_element_count > 0
            && upper_element_count > 0
            && lower_keys.len() >= 3
            && upper_keys.len() >= 3
            && (keys_match || bounds_match),
        lower_element_count,
        lower_projected_area_m2,
    })
}

fn projected_key_bounds(keys: &BTreeSet<[i64; 2]>) -> Option<([i64; 2], [i64; 2])> {
    let mut iter = keys.iter();
    let first = *iter.next()?;
    let mut min_key = first;
    let mut max_key = first;
    for key in iter {
        for axis in 0..2 {
            min_key[axis] = min_key[axis].min(key[axis]);
            max_key[axis] = max_key[axis].max(key[axis]);
        }
    }
    Some((min_key, max_key))
}

fn element_node_points(
    element: &SurfaceElement,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
) -> Result<[[f64; 3]; 3], TetCandidateError> {
    Ok([
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
    ])
}

fn points_on_axis_plane(
    points: &[[f64; 3]; 3],
    axis: usize,
    coordinate_m: f64,
    tolerance: MeshingTolerance,
) -> bool {
    points
        .iter()
        .all(|point| tolerance.nearly_equal(point[axis], coordinate_m, 1.0))
}

fn element_projected_keys(
    element: &SurfaceElement,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    axis: usize,
    tolerance: MeshingTolerance,
) -> Result<[[i64; 2]; 3], TetCandidateError> {
    let points = element_node_points(element, surface_nodes)?;
    Ok(points.map(|point| projected_key(point, axis, tolerance)))
}

fn projected_key(point: [f64; 3], axis: usize, tolerance: MeshingTolerance) -> [i64; 2] {
    let scale_m = tolerance.absolute_m.max(1.0e-12);
    let mut key = [0_i64; 2];
    let mut out = 0_usize;
    for coordinate_axis in 0..3 {
        if coordinate_axis == axis {
            continue;
        }
        key[out] = (point[coordinate_axis] / scale_m).round() as i64;
        out += 1;
    }
    key
}

fn projected_triangle_area_m2(points: &[[f64; 3]; 3], axis: usize) -> f64 {
    let mut coordinate_axes = [0_usize; 2];
    let mut out = 0_usize;
    for coordinate_axis in 0..3 {
        if coordinate_axis == axis {
            continue;
        }
        coordinate_axes[out] = coordinate_axis;
        out += 1;
    }
    let a = [points[0][coordinate_axes[0]], points[0][coordinate_axes[1]]];
    let b = [points[1][coordinate_axes[0]], points[1][coordinate_axes[1]]];
    let c = [points[2][coordinate_axes[0]], points[2][coordinate_axes[1]]];
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
}

fn dense_recovery_component_exceeds_budget(
    component: &VolumeCandidateComponent,
    options: TetCandidateOptions,
) -> bool {
    options.dense_recovery_layer_count <= 3
        && ((component.node_ids.len() > options.max_global_insertion_points
            && component.node_ids.len() > 64)
            || component.surface_element_ids.len()
                > options
                    .max_global_insertion_points
                    .saturating_mul(2)
                    .max(96))
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
        let score = score_layered_fan_seed_point(
            component,
            point,
            surface_nodes,
            surface_elements,
            options,
        )?;
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

fn score_layered_fan_seed_point(
    component: &VolumeCandidateComponent,
    fan_seed_point: [f64; 3],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
) -> Result<FanSeedScore, TetCandidateError> {
    let mut valid_tet_count = 0_usize;
    let mut below_threshold_count = 0_usize;
    let mut total_volume_m3 = 0.0_f64;
    let mut aspect_ratio_sum = 0.0_f64;
    let mut max_aspect_ratio = 0.0_f64;
    let mut min_scaled_jacobian = f64::INFINITY;

    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        let outer_points = [
            *surface_nodes.get(&element.node_ids[0]).ok_or(
                TetCandidateError::MissingSurfaceNode {
                    node_id: element.node_ids[0],
                },
            )?,
            *surface_nodes.get(&element.node_ids[1]).ok_or(
                TetCandidateError::MissingSurfaceNode {
                    node_id: element.node_ids[1],
                },
            )?,
            *surface_nodes.get(&element.node_ids[2]).ok_or(
                TetCandidateError::MissingSurfaceNode {
                    node_id: element.node_ids[2],
                },
            )?,
        ];
        let mut layer_ids = element.node_ids;
        let mut layer_points = outer_points;
        for layer in 1..options.dense_recovery_layer_count {
            let mut inner_ids = [0_u32; 3];
            let mut inner_points = [[0.0; 3]; 3];
            for corner in 0..3 {
                inner_ids[corner] = layered_score_node_id(element.element_id, layer, corner);
                inner_points[corner] = dense_recovery_layer_point(
                    outer_points[corner],
                    fan_seed_point,
                    layer,
                    options,
                );
            }
            let mut best = None::<LayeredFrustumSplit>;
            for split_index in 0..6 {
                let Some(split) = layered_frustum_split(
                    component,
                    element,
                    layer_ids,
                    inner_ids,
                    layer_points,
                    inner_points,
                    split_index,
                    options,
                ) else {
                    continue;
                };
                if best
                    .as_ref()
                    .is_none_or(|current| layered_split_is_better(&split, current))
                {
                    best = Some(split);
                }
            }
            let Some(split) = best else {
                continue;
            };
            accumulate_fan_seed_score_tets(
                &split.tets,
                split.below_threshold_count,
                split.min_scaled_jacobian,
                &mut valid_tet_count,
                &mut below_threshold_count,
                &mut total_volume_m3,
                &mut aspect_ratio_sum,
                &mut max_aspect_ratio,
                &mut min_scaled_jacobian,
            );
            layer_ids = inner_ids;
            layer_points = inner_points;
        }

        let final_points = [
            layer_points[0],
            layer_points[1],
            layer_points[2],
            fan_seed_point,
        ];
        if let Some(final_tet) = candidate_tet(
            component,
            element,
            [layer_ids[0], layer_ids[1], layer_ids[2], u32::MAX],
            final_points,
            options,
        ) {
            let scaled_jacobian = tet_scaled_jacobian(final_points);
            accumulate_fan_seed_score_tets(
                &[final_tet],
                usize::from(scaled_jacobian < options.min_scaled_jacobian),
                scaled_jacobian,
                &mut valid_tet_count,
                &mut below_threshold_count,
                &mut total_volume_m3,
                &mut aspect_ratio_sum,
                &mut max_aspect_ratio,
                &mut min_scaled_jacobian,
            );
        }
    }

    Ok(fan_seed_score_from_accumulators(
        fan_seed_point,
        valid_tet_count,
        below_threshold_count,
        total_volume_m3,
        aspect_ratio_sum,
        max_aspect_ratio,
        min_scaled_jacobian,
        component.volume_m3,
    ))
}

fn layered_score_node_id(element_id: u32, layer: usize, corner: usize) -> u32 {
    1_000_000_u32
        .saturating_add(element_id.saturating_mul(128))
        .saturating_add((layer as u32).saturating_mul(3))
        .saturating_add(corner as u32)
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
        append_best_layered_frustum_tets(
            component,
            element,
            outer_ids,
            inner_ids,
            outer_points,
            inner_points,
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

fn append_best_layered_frustum_tets(
    component: &VolumeCandidateComponent,
    element: &SurfaceElement,
    outer_ids: [u32; 3],
    inner_ids: [u32; 3],
    outer_points: [[f64; 3]; 3],
    inner_points: [[f64; 3]; 3],
    options: TetCandidateOptions,
    tets: &mut Vec<TetCandidate>,
) {
    let mut best = None::<LayeredFrustumSplit>;
    for split_index in 0..6 {
        let split = layered_frustum_split(
            component,
            element,
            outer_ids,
            inner_ids,
            outer_points,
            inner_points,
            split_index,
            options,
        );
        let Some(split) = split else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|current| layered_split_is_better(&split, current))
        {
            best = Some(split);
        }
    }
    if let Some(split) = best {
        tets.extend(split.tets);
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LayeredFrustumSplit {
    tets: Vec<TetCandidate>,
    below_threshold_count: usize,
    min_scaled_jacobian: f64,
    max_aspect_ratio: f64,
}

fn layered_split_is_better(candidate: &LayeredFrustumSplit, best: &LayeredFrustumSplit) -> bool {
    candidate
        .below_threshold_count
        .cmp(&best.below_threshold_count)
        .reverse()
        .then_with(|| {
            candidate
                .min_scaled_jacobian
                .total_cmp(&best.min_scaled_jacobian)
        })
        .then_with(|| best.max_aspect_ratio.total_cmp(&candidate.max_aspect_ratio))
        .is_gt()
}

#[allow(clippy::too_many_arguments)]
fn layered_frustum_split(
    component: &VolumeCandidateComponent,
    element: &SurfaceElement,
    outer_ids: [u32; 3],
    inner_ids: [u32; 3],
    outer_points: [[f64; 3]; 3],
    inner_points: [[f64; 3]; 3],
    split_index: usize,
    options: TetCandidateOptions,
) -> Option<LayeredFrustumSplit> {
    let diagonal_index = split_index % 3;
    let a = diagonal_index;
    let b = (diagonal_index + 1) % 3;
    let c = (diagonal_index + 2) % 3;
    let candidates = if split_index < 3 {
        [
            (
                [outer_ids[a], outer_ids[b], outer_ids[c], inner_ids[a]],
                [
                    outer_points[a],
                    outer_points[b],
                    outer_points[c],
                    inner_points[a],
                ],
            ),
            (
                [outer_ids[b], inner_ids[b], outer_ids[c], inner_ids[a]],
                [
                    outer_points[b],
                    inner_points[b],
                    outer_points[c],
                    inner_points[a],
                ],
            ),
            (
                [inner_ids[b], inner_ids[c], outer_ids[c], inner_ids[a]],
                [
                    inner_points[b],
                    inner_points[c],
                    outer_points[c],
                    inner_points[a],
                ],
            ),
        ]
    } else {
        [
            (
                [outer_ids[a], outer_ids[b], outer_ids[c], inner_ids[c]],
                [
                    outer_points[a],
                    outer_points[b],
                    outer_points[c],
                    inner_points[c],
                ],
            ),
            (
                [outer_ids[a], outer_ids[b], inner_ids[b], inner_ids[c]],
                [
                    outer_points[a],
                    outer_points[b],
                    inner_points[b],
                    inner_points[c],
                ],
            ),
            (
                [outer_ids[a], inner_ids[a], inner_ids[b], inner_ids[c]],
                [
                    outer_points[a],
                    inner_points[a],
                    inner_points[b],
                    inner_points[c],
                ],
            ),
        ]
    };
    let tets = candidates
        .iter()
        .map(|(node_ids, points)| candidate_tet(component, element, *node_ids, *points, options))
        .collect::<Option<Vec<_>>>()?;
    let min_scaled_jacobian = candidates
        .iter()
        .map(|(_, points)| tet_scaled_jacobian(*points))
        .fold(f64::INFINITY, f64::min);
    let below_threshold_count = candidates
        .iter()
        .filter(|(_, points)| tet_scaled_jacobian(*points) < options.min_scaled_jacobian)
        .count();
    let max_aspect_ratio = max_candidate_aspect_ratio(&tets);
    Some(LayeredFrustumSplit {
        tets,
        below_threshold_count,
        min_scaled_jacobian,
        max_aspect_ratio,
    })
}

fn max_candidate_aspect_ratio(tets: &[TetCandidate]) -> f64 {
    tets.iter()
        .map(|tet| tet.aspect_ratio)
        .fold(0.0_f64, f64::max)
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
        let exact_scaled_jacobian = tet_scaled_jacobian(tet_points);
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
            exact_scaled_jacobian,
        });
    }
    Ok((
        insertion_tet_status(component, &accepted_tets, options),
        accepted_tets,
    ))
}

#[derive(Debug, Clone, PartialEq)]
struct SeedRefinementSummary {
    pass_count: usize,
    inserted_point_count: usize,
    requested_point_count: usize,
    accepted_requested_point_count: usize,
    accepted_requested_surrogate_point_count: usize,
    rejected_requested_point_count: usize,
    attempted_requested_ids: BTreeSet<usize>,
    duplicate_requested_ids: BTreeSet<usize>,
    accepted_requested_points: Vec<AcceptedRequestedRefinementPoint>,
    sizing_violation_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AcceptedRequestedRefinementPoint {
    seed_index: usize,
    requested_point: [f64; 3],
    requested_id: usize,
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
        || (dense_component_for_global_insertion(component, seed_points.len(), options)
            && options.requested_refinement_point_count == 0)
    {
        return Ok(SeedRefinementSummary {
            pass_count: 0,
            inserted_point_count: 0,
            requested_point_count: 0,
            accepted_requested_point_count: 0,
            accepted_requested_surrogate_point_count: 0,
            rejected_requested_point_count: 0,
            attempted_requested_ids: BTreeSet::new(),
            duplicate_requested_ids: BTreeSet::new(),
            accepted_requested_points: Vec::new(),
            sizing_violation_count: 0,
        });
    }

    let classifier =
        ComponentSurfaceClassifier::new(component, surface, surface_elements, tolerance)?;
    let mut pass_count = 0_usize;
    let mut inserted_point_count = 0_usize;
    let mut requested_point_count = 0_usize;
    let mut accepted_requested_point_count = 0_usize;
    let mut accepted_requested_surrogate_point_count = 0_usize;
    let mut accepted_requested_points = Vec::<AcceptedRequestedRefinementPoint>::new();
    let mut accepted_requested_ids = BTreeSet::<usize>::new();
    let mut attempted_requested_ids = BTreeSet::<usize>::new();
    let mut duplicate_requested_ids = BTreeSet::<usize>::new();
    let mut sizing_violation_count = 0_usize;
    for _ in 0..options.max_refinement_passes {
        if seed_points.len() >= options.max_interior_seed_points {
            break;
        }
        let current_seed_node_ids = seed_node_ids(first_seed_node_id, seed_points.len());
        let (status, candidate_tets) = component_insertion_tet_drafts(
            component,
            &current_seed_node_ids,
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

        let include_quality_driven_refinement =
            !dense_component_for_global_insertion(component, seed_points.len(), options);
        let point_budget = options.max_interior_seed_points - seed_points.len();
        let refinement_points = refinement_points_for_tets(
            &candidate_tets,
            surface_nodes,
            &current_seed_node_ids,
            seed_points,
            tolerance,
            &classifier,
            options,
            point_budget,
            &accepted_requested_ids,
            include_quality_driven_refinement,
        )?;
        requested_point_count += refinement_points.requested_point_count;
        attempted_requested_ids.extend(refinement_points.requested_ids.iter().copied());
        duplicate_requested_ids.extend(refinement_points.duplicate_requested_ids.iter().copied());
        sizing_violation_count += refinement_points.sizing_violation_count;
        if refinement_points.points.is_empty() {
            break;
        }
        let mut accepted_this_pass = 0_usize;
        let mut current_status = status;
        let mut current_tets = candidate_tets;
        for candidate in refinement_points.points {
            let point = candidate.point;
            if seed_points.len() >= options.max_interior_seed_points {
                break;
            }
            if candidate
                .requested_id
                .is_some_and(|requested_id| accepted_requested_ids.contains(&requested_id))
            {
                continue;
            }
            if contains_point(seed_points, point, tolerance) {
                continue;
            }
            let current_quality = CandidateQualitySnapshot::from_tets(&current_tets, options);
            let mut trial_seed_points = seed_points.clone();
            trial_seed_points.push(point);
            let trial_seed_node_ids = seed_node_ids(first_seed_node_id, trial_seed_points.len());
            let (trial_status, trial_tets) = component_insertion_tet_drafts(
                component,
                &trial_seed_node_ids,
                &trial_seed_points,
                surface_nodes,
                surface_elements,
                surface,
                options,
                tolerance,
            )?;
            if trial_tets.is_empty() || (current_status.accepted && !trial_status.accepted) {
                continue;
            }
            let trial_quality = CandidateQualitySnapshot::from_tets(&trial_tets, options);
            let quality_is_acceptable = if candidate.requested_id.is_some() {
                candidate_quality_preserves_thresholds(trial_quality, current_quality)
            } else {
                candidate_quality_is_no_worse(trial_quality, current_quality)
            };
            if !quality_is_acceptable {
                continue;
            }
            let seed_index = seed_points.len();
            seed_points.push(point);
            current_status = trial_status;
            current_tets = trial_tets;
            inserted_point_count += 1;
            if let Some(requested_id) = candidate.requested_id {
                accepted_requested_ids.insert(requested_id);
                accepted_requested_point_count += 1;
                if !tolerance.point_nearly_equal(
                    point,
                    options.requested_refinement_points[requested_id],
                    1.0,
                ) {
                    accepted_requested_surrogate_point_count += 1;
                }
                accepted_requested_points.push(AcceptedRequestedRefinementPoint {
                    seed_index,
                    requested_point: point,
                    requested_id,
                });
            }
            accepted_this_pass += 1;
        }
        if accepted_this_pass == 0 {
            break;
        }
        pass_count += 1;
    }

    Ok(SeedRefinementSummary {
        pass_count,
        inserted_point_count,
        requested_point_count,
        accepted_requested_point_count,
        accepted_requested_surrogate_point_count,
        rejected_requested_point_count: rejected_requested_refinement_point_count(
            &attempted_requested_ids,
            &accepted_requested_ids,
        ),
        attempted_requested_ids,
        duplicate_requested_ids,
        accepted_requested_points,
        sizing_violation_count,
    })
}

fn rejected_requested_refinement_point_count(
    attempted_requested_ids: &BTreeSet<usize>,
    accepted_requested_ids: &BTreeSet<usize>,
) -> usize {
    attempted_requested_ids
        .difference(accepted_requested_ids)
        .count()
}

#[derive(Debug, Clone, PartialEq)]
struct RefinementPointCandidate {
    point: [f64; 3],
    requested_id: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct RefinementPointSet {
    points: Vec<RefinementPointCandidate>,
    requested_ids: Vec<usize>,
    duplicate_requested_ids: Vec<usize>,
    requested_point_count: usize,
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
    accepted_requested_ids: &BTreeSet<usize>,
    include_quality_driven_refinement: bool,
) -> Result<RefinementPointSet, TetCandidateError> {
    let Some(target_size_m) = options.interior_target_size_m else {
        return Ok(RefinementPointSet {
            points: Vec::new(),
            requested_ids: Vec::new(),
            duplicate_requested_ids: Vec::new(),
            requested_point_count: 0,
            sizing_violation_count: 0,
        });
    };
    let all_nodes = candidate_node_coordinates(surface_nodes, seed_node_ids, seed_points);
    let mut ranked = Vec::<RankedRefinementPoint>::new();
    let mut requested_ids = Vec::<usize>::new();
    let mut duplicate_requested_ids = Vec::<usize>::new();
    let mut requested_point_count = 0_usize;
    let mut sizing_violation_count = 0_usize;
    for (requested_id, point) in options
        .requested_refinement_points
        .iter()
        .take(options.requested_refinement_point_count)
        .enumerate()
    {
        if accepted_requested_ids.contains(&requested_id) {
            continue;
        }
        if classifier.contains_point(*point) && contains_point(seed_points, *point, tolerance) {
            duplicate_requested_ids.push(requested_id);
            continue;
        }
        if classifier.contains_point(*point) {
            for candidate_point in requested_refinement_candidate_points(
                *point,
                seed_points,
                classifier,
                target_size_m,
                tolerance,
            )
            .into_iter()
            {
                let requested_distance_m = distance(candidate_point, *point);
                ranked.push(RankedRefinementPoint {
                    point: candidate_point,
                    score: requested_refinement_score(requested_distance_m, target_size_m),
                    requested_id: Some(requested_id),
                    requested_distance_m,
                    quality_driven: false,
                });
            }
            requested_ids.push(requested_id);
            requested_point_count += 1;
        }
    }
    if include_quality_driven_refinement {
        for tet in tets {
            let points = candidate_tet_points(tet, &all_nodes)?;
            let radius_edge_ratio = tet_radius_edge_ratio(points, tolerance);
            let exact_scaled_jacobian = tet_scaled_jacobian(points);
            let max_edge_m = tet_max_edge_length(points);
            let sizing_violation =
                max_edge_m > target_size_m * (1.0 + options.sizing_compliance_tolerance);
            let exact_quality_violation = exact_scaled_jacobian < options.min_scaled_jacobian;
            if sizing_violation {
                sizing_violation_count += 1;
            }
            if radius_edge_ratio <= options.max_radius_edge_ratio
                && !sizing_violation
                && !exact_quality_violation
            {
                continue;
            }
            let point = if exact_quality_violation {
                tet_centroid(points)
            } else {
                tet_circumsphere(points, tolerance)
                    .map(|(center, _)| center)
                    .unwrap_or_else(|| tet_centroid(points))
            };
            let point =
                if classifier.contains_protected_interior_point(point, target_size_m, tolerance) {
                    point
                } else {
                    tet_centroid(points)
                };
            if !classifier.contains_protected_interior_point(point, target_size_m, tolerance) {
                continue;
            }
            let exact_quality_error = if exact_quality_violation {
                (options.min_scaled_jacobian - exact_scaled_jacobian) / options.min_scaled_jacobian
            } else {
                0.0
            };
            ranked.push(RankedRefinementPoint {
                point,
                score: radius_edge_ratio
                    .max(max_edge_m / target_size_m)
                    .max(exact_quality_error),
                requested_id: None,
                requested_distance_m: f64::INFINITY,
                quality_driven: sizing_violation || exact_quality_violation,
            });
        }
    }
    ranked.sort_by(compare_ranked_refinement_points);
    let mut points = Vec::<RefinementPointCandidate>::new();
    let mut unrequested_point_count = 0_usize;
    let mut requested_candidate_counts = BTreeMap::<usize, usize>::new();
    for ranked_point in ranked {
        if ranked_point.requested_id.is_none() && unrequested_point_count >= point_budget {
            break;
        }
        if let Some(requested_id) = ranked_point.requested_id {
            if requested_candidate_counts
                .get(&requested_id)
                .copied()
                .unwrap_or_default()
                >= options.max_requested_refinement_candidates_per_point
            {
                continue;
            }
        }
        if contains_point(seed_points, ranked_point.point, tolerance)
            || points.iter().any(|candidate| {
                tolerance.point_nearly_equal(candidate.point, ranked_point.point, 1.0)
            })
        {
            continue;
        }
        points.push(RefinementPointCandidate {
            point: ranked_point.point,
            requested_id: ranked_point.requested_id,
        });
        if ranked_point.requested_id.is_none() {
            unrequested_point_count += 1;
        } else if let Some(requested_id) = ranked_point.requested_id {
            *requested_candidate_counts.entry(requested_id).or_default() += 1;
        }
    }
    Ok(RefinementPointSet {
        points,
        requested_ids,
        duplicate_requested_ids,
        requested_point_count,
        sizing_violation_count,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RankedRefinementPoint {
    point: [f64; 3],
    score: f64,
    requested_id: Option<usize>,
    requested_distance_m: f64,
    quality_driven: bool,
}

fn compare_ranked_refinement_points(
    left: &RankedRefinementPoint,
    right: &RankedRefinementPoint,
) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| match (left.requested_id, right.requested_id) {
            (Some(left_id), Some(right_id)) => left_id
                .cmp(&right_id)
                .then_with(|| {
                    left.requested_distance_m
                        .total_cmp(&right.requested_distance_m)
                })
                .then_with(|| compare_points_lexicographically(left.point, right.point)),
            _ => right
                .quality_driven
                .cmp(&left.quality_driven)
                .then_with(|| compare_points_lexicographically(left.point, right.point)),
        })
}

fn compare_points_lexicographically(left: [f64; 3], right: [f64; 3]) -> std::cmp::Ordering {
    left[0]
        .total_cmp(&right[0])
        .then_with(|| left[1].total_cmp(&right[1]))
        .then_with(|| left[2].total_cmp(&right[2]))
}

fn requested_refinement_score(distance_m: f64, target_size_m: f64) -> f64 {
    let normalized_distance = if target_size_m.is_finite() && target_size_m > 0.0 {
        distance_m / target_size_m
    } else {
        distance_m
    };
    1.0e12 - normalized_distance.max(0.0)
}

fn requested_refinement_candidate_points(
    requested_point: [f64; 3],
    seed_points: &[[f64; 3]],
    classifier: &ComponentSurfaceClassifier,
    target_size_m: f64,
    tolerance: MeshingTolerance,
) -> Vec<[f64; 3]> {
    let mut candidates = Vec::<[f64; 3]>::new();
    if seed_points.is_empty() {
        push_requested_refinement_candidate(
            &mut candidates,
            requested_point,
            seed_points,
            classifier,
            target_size_m,
            tolerance,
        );
        return candidates;
    }
    let clearance = classifier.nearest_surface_distance(requested_point);
    let safe_clearance = target_size_m.min(1.0) * 0.01;
    let near_boundary = clearance <= safe_clearance.max(tolerance.absolute_m * 10.0);
    if !near_boundary {
        candidates.push(requested_point);
    }
    let anchor = seed_points
        .iter()
        .copied()
        .min_by(|left, right| {
            distance_squared(*left, requested_point)
                .total_cmp(&distance_squared(*right, requested_point))
        })
        .unwrap_or(requested_point);
    let fractions: &[f64] = if near_boundary {
        &[0.25, 0.5, 0.75, 0.9]
    } else {
        &[0.25, 0.5, 0.75]
    };
    for fraction in fractions {
        let candidate = [
            requested_point[0] * (1.0 - *fraction) + anchor[0] * *fraction,
            requested_point[1] * (1.0 - *fraction) + anchor[1] * *fraction,
            requested_point[2] * (1.0 - *fraction) + anchor[2] * *fraction,
        ];
        push_requested_refinement_candidate(
            &mut candidates,
            candidate,
            seed_points,
            classifier,
            target_size_m,
            tolerance,
        );
    }
    let anchor_distance = distance(requested_point, anchor);
    let base_radius = target_size_m
        .min(anchor_distance * 0.5)
        .min(clearance * 0.5)
        .max(tolerance.absolute_m * 10.0);
    let local_radii = [base_radius * 0.5, base_radius];
    let local_directions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for radius in local_radii {
        if !radius.is_finite() || radius <= tolerance.absolute_m {
            continue;
        }
        for direction in local_directions {
            let candidate = [
                requested_point[0] + direction[0] * radius,
                requested_point[1] + direction[1] * radius,
                requested_point[2] + direction[2] * radius,
            ];
            push_requested_refinement_candidate(
                &mut candidates,
                candidate,
                seed_points,
                classifier,
                target_size_m,
                tolerance,
            );
        }
    }
    let face_diagonal_scale = 1.0 / 2.0_f64.sqrt();
    let face_diagonal_directions = [
        [face_diagonal_scale, face_diagonal_scale, 0.0],
        [face_diagonal_scale, -face_diagonal_scale, 0.0],
        [-face_diagonal_scale, face_diagonal_scale, 0.0],
        [-face_diagonal_scale, -face_diagonal_scale, 0.0],
        [face_diagonal_scale, 0.0, face_diagonal_scale],
        [face_diagonal_scale, 0.0, -face_diagonal_scale],
        [-face_diagonal_scale, 0.0, face_diagonal_scale],
        [-face_diagonal_scale, 0.0, -face_diagonal_scale],
        [0.0, face_diagonal_scale, face_diagonal_scale],
        [0.0, face_diagonal_scale, -face_diagonal_scale],
        [0.0, -face_diagonal_scale, face_diagonal_scale],
        [0.0, -face_diagonal_scale, -face_diagonal_scale],
    ];
    for radius in local_radii {
        if !radius.is_finite() || radius <= tolerance.absolute_m {
            continue;
        }
        for direction in face_diagonal_directions {
            let candidate = [
                requested_point[0] + direction[0] * radius,
                requested_point[1] + direction[1] * radius,
                requested_point[2] + direction[2] * radius,
            ];
            push_requested_refinement_candidate(
                &mut candidates,
                candidate,
                seed_points,
                classifier,
                target_size_m,
                tolerance,
            );
        }
    }
    let diagonal_scale = 1.0 / 3.0_f64.sqrt();
    let diagonal_directions = [
        [diagonal_scale, diagonal_scale, diagonal_scale],
        [diagonal_scale, diagonal_scale, -diagonal_scale],
        [diagonal_scale, -diagonal_scale, diagonal_scale],
        [diagonal_scale, -diagonal_scale, -diagonal_scale],
        [-diagonal_scale, diagonal_scale, diagonal_scale],
        [-diagonal_scale, diagonal_scale, -diagonal_scale],
        [-diagonal_scale, -diagonal_scale, diagonal_scale],
        [-diagonal_scale, -diagonal_scale, -diagonal_scale],
    ];
    for radius in local_radii {
        if !radius.is_finite() || radius <= tolerance.absolute_m {
            continue;
        }
        for direction in diagonal_directions {
            let candidate = [
                requested_point[0] + direction[0] * radius,
                requested_point[1] + direction[1] * radius,
                requested_point[2] + direction[2] * radius,
            ];
            push_requested_refinement_candidate(
                &mut candidates,
                candidate,
                seed_points,
                classifier,
                target_size_m,
                tolerance,
            );
        }
    }
    candidates
}

fn push_requested_refinement_candidate(
    candidates: &mut Vec<[f64; 3]>,
    candidate: [f64; 3],
    seed_points: &[[f64; 3]],
    classifier: &ComponentSurfaceClassifier,
    target_size_m: f64,
    tolerance: MeshingTolerance,
) {
    if classifier.contains_protected_interior_point(candidate, target_size_m, tolerance)
        && !contains_point(candidates, candidate, tolerance)
        && !contains_point(seed_points, candidate, tolerance)
    {
        candidates.push(candidate);
    }
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct SmoothingSummary {
    pass_count: usize,
    smoothed_point_count: usize,
    sliver_candidate_count: usize,
    sliver_removed_count: usize,
    target_seed_count: usize,
    skipped_target_seed_count: usize,
    rejected_edit_count: usize,
    quality_sample_count: usize,
    initial_max_aspect_ratio: f64,
    final_max_aspect_ratio: f64,
    initial_min_exact_scaled_jacobian: f64,
    final_min_exact_scaled_jacobian: f64,
}

impl SmoothingSummary {
    fn empty() -> Self {
        Self {
            pass_count: 0,
            smoothed_point_count: 0,
            sliver_candidate_count: 0,
            sliver_removed_count: 0,
            target_seed_count: 0,
            skipped_target_seed_count: 0,
            rejected_edit_count: 0,
            quality_sample_count: 0,
            initial_max_aspect_ratio: 0.0,
            final_max_aspect_ratio: 0.0,
            initial_min_exact_scaled_jacobian: 0.0,
            final_min_exact_scaled_jacobian: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct OptimizationQualityAggregate {
    quality_sample_count: usize,
    sliver_removed_count: usize,
    target_seed_count: usize,
    skipped_target_seed_count: usize,
    rejected_edit_count: usize,
    initial_max_aspect_ratio: f64,
    final_max_aspect_ratio: f64,
    initial_min_exact_scaled_jacobian: f64,
    final_min_exact_scaled_jacobian: f64,
}

impl OptimizationQualityAggregate {
    fn record(&mut self, summary: SmoothingSummary) {
        self.sliver_removed_count += summary.sliver_removed_count;
        self.target_seed_count += summary.target_seed_count;
        self.skipped_target_seed_count += summary.skipped_target_seed_count;
        self.rejected_edit_count += summary.rejected_edit_count;
        if summary.quality_sample_count == 0 {
            return;
        }
        if self.quality_sample_count == 0 {
            self.initial_max_aspect_ratio = summary.initial_max_aspect_ratio;
            self.final_max_aspect_ratio = summary.final_max_aspect_ratio;
            self.initial_min_exact_scaled_jacobian = summary.initial_min_exact_scaled_jacobian;
            self.final_min_exact_scaled_jacobian = summary.final_min_exact_scaled_jacobian;
        } else {
            self.initial_max_aspect_ratio = self
                .initial_max_aspect_ratio
                .max(summary.initial_max_aspect_ratio);
            self.final_max_aspect_ratio = self
                .final_max_aspect_ratio
                .max(summary.final_max_aspect_ratio);
            self.initial_min_exact_scaled_jacobian = self
                .initial_min_exact_scaled_jacobian
                .min(summary.initial_min_exact_scaled_jacobian);
            self.final_min_exact_scaled_jacobian = self
                .final_min_exact_scaled_jacobian
                .min(summary.final_min_exact_scaled_jacobian);
        }
        self.quality_sample_count += summary.quality_sample_count;
    }

    fn initial_max_aspect_ratio(self) -> f64 {
        if self.quality_sample_count == 0 {
            0.0
        } else {
            self.initial_max_aspect_ratio
        }
    }

    fn final_max_aspect_ratio(self) -> f64 {
        if self.quality_sample_count == 0 {
            0.0
        } else {
            self.final_max_aspect_ratio
        }
    }

    fn initial_min_exact_scaled_jacobian(self) -> f64 {
        if self.quality_sample_count == 0 {
            0.0
        } else {
            self.initial_min_exact_scaled_jacobian
        }
    }

    fn final_min_exact_scaled_jacobian(self) -> f64 {
        if self.quality_sample_count == 0 {
            0.0
        } else {
            self.final_min_exact_scaled_jacobian
        }
    }

    fn sliver_removed_count(self) -> usize {
        self.sliver_removed_count
    }

    fn target_seed_count(self) -> usize {
        self.target_seed_count
    }

    fn skipped_target_seed_count(self) -> usize {
        self.skipped_target_seed_count
    }
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
    if options.max_optimization_passes == 0 || seed_points.is_empty() {
        return Ok(SmoothingSummary::empty());
    }

    let classifier =
        ComponentSurfaceClassifier::new(component, surface, surface_elements, tolerance)?;
    let mut pass_count = 0_usize;
    let mut smoothed_point_count = 0_usize;
    let mut sliver_candidate_count = 0_usize;
    let mut sliver_removed_count = 0_usize;
    let mut target_seed_count = 0_usize;
    let mut skipped_target_seed_count = 0_usize;
    let mut rejected_edit_count = 0_usize;
    let mut initial_quality = None::<CandidateQualitySnapshot>;
    let mut final_quality = None::<CandidateQualitySnapshot>;

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
        initial_quality.get_or_insert(current_quality);
        final_quality = Some(current_quality);
        sliver_candidate_count += current_quality.sliver_count;
        let proposed = smoothed_seed_points(
            seed_points,
            &seed_node_ids,
            &current_tets,
            surface_nodes,
            &classifier,
            options,
        )?;
        if proposed != *seed_points {
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
                rejected_edit_count += 1;
            } else {
                let proposed_quality = CandidateQualitySnapshot::from_tets(&proposed_tets, options);
                if candidate_quality_is_no_worse(proposed_quality, current_quality) {
                    let moved_count = seed_points
                        .iter()
                        .zip(proposed.iter())
                        .filter(|(left, right)| !tolerance.point_nearly_equal(**left, **right, 1.0))
                        .count();
                    if moved_count > 0 {
                        *seed_points = proposed;
                        final_quality = Some(proposed_quality);
                        pass_count += 1;
                        smoothed_point_count += moved_count;
                        sliver_removed_count += current_quality
                            .sliver_count
                            .saturating_sub(proposed_quality.sliver_count);
                        continue;
                    }
                } else {
                    rejected_edit_count += 1;
                }
            }
        }

        let Some(local_choice) = best_local_seed_smoothing(
            component,
            seed_points,
            &proposed,
            &seed_node_ids,
            &current_tets,
            &classifier,
            surface_nodes,
            surface_elements,
            surface,
            options,
            tolerance,
            current_status.accepted,
            current_quality,
        )?
        else {
            break;
        };
        target_seed_count += local_choice.target_seed_count;
        skipped_target_seed_count += local_choice.skipped_target_seed_count;
        *seed_points = local_choice.seed_points;
        final_quality = Some(local_choice.quality);
        pass_count += 1;
        smoothed_point_count += 1;
        sliver_candidate_count += local_choice.quality.sliver_count;
        sliver_removed_count += current_quality
            .sliver_count
            .saturating_sub(local_choice.quality.sliver_count);
    }

    let initial_quality = initial_quality.unwrap_or_else(CandidateQualitySnapshot::empty);
    let final_quality = final_quality.unwrap_or(initial_quality);
    Ok(SmoothingSummary {
        pass_count,
        smoothed_point_count,
        sliver_candidate_count,
        sliver_removed_count,
        target_seed_count,
        skipped_target_seed_count,
        rejected_edit_count,
        quality_sample_count: usize::from(initial_quality.has_samples()),
        initial_max_aspect_ratio: initial_quality.max_aspect_ratio,
        final_max_aspect_ratio: final_quality.max_aspect_ratio,
        initial_min_exact_scaled_jacobian: initial_quality.min_exact_scaled_jacobian,
        final_min_exact_scaled_jacobian: final_quality.min_exact_scaled_jacobian,
    })
}

#[allow(clippy::too_many_arguments)]
fn best_local_seed_smoothing(
    component: &VolumeCandidateComponent,
    seed_points: &[[f64; 3]],
    proposed_points: &[[f64; 3]],
    seed_node_ids: &[u32],
    current_tets: &[TetCandidate],
    classifier: &ComponentSurfaceClassifier,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    surface: &SurfaceDiscretization,
    options: TetCandidateOptions,
    tolerance: MeshingTolerance,
    current_status_accepted: bool,
    current_quality: CandidateQualitySnapshot,
) -> Result<Option<LocalSmoothingChoice>, TetCandidateError> {
    let target_summary = optimization_target_seed_indices(current_tets, seed_node_ids, options);
    let mut best = None::<(Vec<[f64; 3]>, CandidateQualitySnapshot)>;
    for index in &target_summary.indices {
        for proposed_point in local_seed_smoothing_candidate_points(
            seed_points[*index],
            proposed_points[*index],
            classifier,
            tolerance,
        ) {
            let mut trial = seed_points.to_vec();
            trial[*index] = proposed_point;
            let (trial_status, trial_tets) = component_insertion_tet_drafts(
                component,
                seed_node_ids,
                &trial,
                surface_nodes,
                surface_elements,
                surface,
                options,
                tolerance,
            )?;
            if !trial_status.accepted && current_status_accepted {
                continue;
            }
            let trial_quality = CandidateQualitySnapshot::from_tets(&trial_tets, options);
            if !candidate_quality_is_no_worse(trial_quality, current_quality)
                || !candidate_quality_is_better(trial_quality, current_quality)
            {
                continue;
            }
            if best.as_ref().is_none_or(|(_, best_quality)| {
                candidate_quality_is_better(trial_quality, *best_quality)
            }) {
                best = Some((trial, trial_quality));
            }
        }
    }
    Ok(best.map(|(seed_points, quality)| LocalSmoothingChoice {
        seed_points,
        quality,
        target_seed_count: target_summary.total_count,
        skipped_target_seed_count: target_summary.skipped_count,
    }))
}

#[derive(Debug, Clone, PartialEq)]
struct LocalSmoothingChoice {
    seed_points: Vec<[f64; 3]>,
    quality: CandidateQualitySnapshot,
    target_seed_count: usize,
    skipped_target_seed_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OptimizationTargetSeedSummary {
    indices: Vec<usize>,
    total_count: usize,
    skipped_count: usize,
}

fn local_seed_smoothing_candidate_points(
    current: [f64; 3],
    proposed: [f64; 3],
    classifier: &ComponentSurfaceClassifier,
    tolerance: MeshingTolerance,
) -> Vec<[f64; 3]> {
    const LOCAL_SMOOTHING_CANDIDATE_LIMIT: usize = 18;
    let mut candidates = Vec::<[f64; 3]>::new();
    push_local_seed_smoothing_candidate(&mut candidates, current, proposed, classifier, tolerance);
    push_local_seed_smoothing_candidate(
        &mut candidates,
        current,
        scale(add(current, proposed), 0.5),
        classifier,
        tolerance,
    );

    let proposed_distance = distance(current, proposed);
    let clearance = classifier.nearest_surface_distance(current);
    let radius = if proposed_distance.is_finite() && proposed_distance > tolerance.absolute_m {
        proposed_distance * 0.5
    } else if clearance.is_finite() && clearance > tolerance.absolute_m {
        clearance * 0.05
    } else {
        0.0
    };
    if radius <= tolerance.absolute_m {
        return candidates;
    }
    let axis_directions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for fraction in [0.5, 1.0] {
        for direction in axis_directions {
            if candidates.len() >= LOCAL_SMOOTHING_CANDIDATE_LIMIT {
                return candidates;
            }
            push_local_seed_smoothing_candidate(
                &mut candidates,
                current,
                add(current, scale(direction, radius * fraction)),
                classifier,
                tolerance,
            );
        }
    }
    let face_diagonal_scale = 1.0 / 2.0_f64.sqrt();
    let face_diagonal_directions = [
        [face_diagonal_scale, face_diagonal_scale, 0.0],
        [face_diagonal_scale, -face_diagonal_scale, 0.0],
        [-face_diagonal_scale, face_diagonal_scale, 0.0],
        [-face_diagonal_scale, -face_diagonal_scale, 0.0],
        [face_diagonal_scale, 0.0, face_diagonal_scale],
        [face_diagonal_scale, 0.0, -face_diagonal_scale],
        [-face_diagonal_scale, 0.0, face_diagonal_scale],
        [-face_diagonal_scale, 0.0, -face_diagonal_scale],
        [0.0, face_diagonal_scale, face_diagonal_scale],
        [0.0, face_diagonal_scale, -face_diagonal_scale],
        [0.0, -face_diagonal_scale, face_diagonal_scale],
        [0.0, -face_diagonal_scale, -face_diagonal_scale],
    ];
    for fraction in [0.5, 1.0] {
        for direction in face_diagonal_directions {
            if candidates.len() >= LOCAL_SMOOTHING_CANDIDATE_LIMIT {
                return candidates;
            }
            push_local_seed_smoothing_candidate(
                &mut candidates,
                current,
                add(current, scale(direction, radius * fraction)),
                classifier,
                tolerance,
            );
        }
    }
    let body_diagonal_scale = 1.0 / 3.0_f64.sqrt();
    let body_diagonal_directions = [
        [
            body_diagonal_scale,
            body_diagonal_scale,
            body_diagonal_scale,
        ],
        [
            body_diagonal_scale,
            body_diagonal_scale,
            -body_diagonal_scale,
        ],
        [
            body_diagonal_scale,
            -body_diagonal_scale,
            body_diagonal_scale,
        ],
        [
            body_diagonal_scale,
            -body_diagonal_scale,
            -body_diagonal_scale,
        ],
        [
            -body_diagonal_scale,
            body_diagonal_scale,
            body_diagonal_scale,
        ],
        [
            -body_diagonal_scale,
            body_diagonal_scale,
            -body_diagonal_scale,
        ],
        [
            -body_diagonal_scale,
            -body_diagonal_scale,
            body_diagonal_scale,
        ],
        [
            -body_diagonal_scale,
            -body_diagonal_scale,
            -body_diagonal_scale,
        ],
    ];
    for fraction in [0.5, 1.0] {
        for direction in body_diagonal_directions {
            if candidates.len() >= LOCAL_SMOOTHING_CANDIDATE_LIMIT {
                return candidates;
            }
            push_local_seed_smoothing_candidate(
                &mut candidates,
                current,
                add(current, scale(direction, radius * fraction)),
                classifier,
                tolerance,
            );
        }
    }
    candidates
}

fn push_local_seed_smoothing_candidate(
    candidates: &mut Vec<[f64; 3]>,
    current: [f64; 3],
    candidate: [f64; 3],
    classifier: &ComponentSurfaceClassifier,
    tolerance: MeshingTolerance,
) {
    if tolerance.point_nearly_equal(candidate, current, 1.0)
        || !classifier.contains_interior_point(candidate)
        || candidates
            .iter()
            .any(|existing| tolerance.point_nearly_equal(*existing, candidate, 1.0))
    {
        return;
    }
    candidates.push(candidate);
}

fn optimization_target_seed_indices(
    tets: &[TetCandidate],
    seed_node_ids: &[u32],
    options: TetCandidateOptions,
) -> OptimizationTargetSeedSummary {
    let seed_index = seed_node_ids
        .iter()
        .enumerate()
        .map(|(index, node_id)| (*node_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut scores = BTreeMap::<usize, (usize, usize)>::new();
    for tet in tets {
        let exact_quality_target = tet.exact_scaled_jacobian < options.min_scaled_jacobian;
        let sliver_target = tet.aspect_ratio > options.sliver_aspect_ratio;
        if !exact_quality_target && !sliver_target {
            continue;
        }
        for node_id in tet.node_ids {
            if let Some(index) = seed_index.get(&node_id) {
                let score = scores.entry(*index).or_default();
                score.0 += usize::from(exact_quality_target);
                score.1 += usize::from(sliver_target);
            }
        }
    }
    let mut indices = scores.into_iter().collect::<Vec<(usize, (usize, usize))>>();
    indices.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score
            .0
            .cmp(&left_score.0)
            .then_with(|| right_score.1.cmp(&left_score.1))
            .then_with(|| left_index.cmp(right_index))
    });
    let total_count = indices.len();
    indices.truncate(options.max_quality_recovery_seed_candidates);
    OptimizationTargetSeedSummary {
        indices: indices.into_iter().map(|(index, _)| index).collect(),
        total_count,
        skipped_count: total_count.saturating_sub(options.max_quality_recovery_seed_candidates),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CandidateQualitySnapshot {
    max_aspect_ratio: f64,
    max_radius_edge_ratio: f64,
    volume_ratio_error: f64,
    sliver_count: usize,
    exact_quality_violation_count: usize,
    min_exact_scaled_jacobian: f64,
}

impl CandidateQualitySnapshot {
    fn empty() -> Self {
        Self {
            max_aspect_ratio: 0.0,
            max_radius_edge_ratio: 0.0,
            volume_ratio_error: 0.0,
            sliver_count: 0,
            exact_quality_violation_count: 0,
            min_exact_scaled_jacobian: 0.0,
        }
    }

    fn has_samples(self) -> bool {
        self.max_aspect_ratio.is_finite()
            && self.max_aspect_ratio > 0.0
            && self.min_exact_scaled_jacobian.is_finite()
    }

    fn from_tets(tets: &[TetCandidate], options: TetCandidateOptions) -> Self {
        if tets.is_empty() {
            return Self::empty();
        }
        let max_aspect_ratio = tets
            .iter()
            .map(|tet| tet.aspect_ratio)
            .fold(0.0_f64, f64::max);
        let sliver_count = tets
            .iter()
            .filter(|tet| tet.aspect_ratio > options.sliver_aspect_ratio)
            .count();
        let exact_quality_violation_count = tets
            .iter()
            .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
            .count();
        let min_exact_scaled_jacobian = tets
            .iter()
            .map(|tet| tet.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        Self {
            max_aspect_ratio,
            max_radius_edge_ratio: 0.0,
            volume_ratio_error: 0.0,
            sliver_count,
            exact_quality_violation_count,
            min_exact_scaled_jacobian,
        }
    }
}

fn candidate_quality_is_no_worse(
    proposed: CandidateQualitySnapshot,
    current: CandidateQualitySnapshot,
) -> bool {
    proposed.sliver_count <= current.sliver_count
        && proposed.exact_quality_violation_count <= current.exact_quality_violation_count
        && proposed.min_exact_scaled_jacobian + 1.0e-12 >= current.min_exact_scaled_jacobian
        && proposed.max_aspect_ratio <= current.max_aspect_ratio + 1.0e-12
}

fn candidate_quality_preserves_thresholds(
    proposed: CandidateQualitySnapshot,
    current: CandidateQualitySnapshot,
) -> bool {
    proposed.sliver_count <= current.sliver_count
        && proposed.exact_quality_violation_count <= current.exact_quality_violation_count
}

fn candidate_quality_is_better(
    proposed: CandidateQualitySnapshot,
    current: CandidateQualitySnapshot,
) -> bool {
    proposed.exact_quality_violation_count < current.exact_quality_violation_count
        || (proposed.exact_quality_violation_count == current.exact_quality_violation_count
            && proposed.min_exact_scaled_jacobian > current.min_exact_scaled_jacobian + 1.0e-12)
        || (proposed.exact_quality_violation_count == current.exact_quality_violation_count
            && (proposed.min_exact_scaled_jacobian - current.min_exact_scaled_jacobian).abs()
                <= 1.0e-12
            && proposed.sliver_count < current.sliver_count)
        || (proposed.exact_quality_violation_count == current.exact_quality_violation_count
            && proposed.sliver_count == current.sliver_count
            && (proposed.min_exact_scaled_jacobian - current.min_exact_scaled_jacobian).abs()
                <= 1.0e-12
            && proposed.max_aspect_ratio + 1.0e-12 < current.max_aspect_ratio)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TetQualityRepairSummary {
    pass_count: usize,
    reconnected_cavity_count: usize,
    reconnection_quality_gain_count: usize,
    face_neighbor_reconnected_cavity_count: usize,
    connected_reconnected_cavity_count: usize,
    node_adjacent_reconnected_cavity_count: usize,
    boundary_adjacent_reconnected_cavity_count: usize,
    expanded_connected_reconnected_cavity_count: usize,
    split_cavity_count: usize,
    seed_star_collapse_count: usize,
    seed_star_relocation_count: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TetQualityRepairPassSummary {
    changed: bool,
    reconnected_cavity_count: usize,
    reconnection_quality_gain_count: usize,
    face_neighbor_reconnected_cavity_count: usize,
    connected_reconnected_cavity_count: usize,
    node_adjacent_reconnected_cavity_count: usize,
    boundary_adjacent_reconnected_cavity_count: usize,
    expanded_connected_reconnected_cavity_count: usize,
    split_cavity_count: usize,
    seed_star_collapse_count: usize,
    seed_star_relocation_count: usize,
}

fn repair_exact_quality_tets(
    nodes: &mut Vec<TetCandidateNode>,
    tets: &mut Vec<TetCandidate>,
    interior_seed_points: &mut Vec<[f64; 3]>,
    next_node_id: &mut u32,
    options: TetCandidateOptions,
) -> Result<TetQualityRepairSummary, TetCandidateError> {
    let mut summary = TetQualityRepairSummary::default();
    if count_exact_quality_violations(tets.iter(), options.min_scaled_jacobian)
        > options.max_exact_quality_repair_targets
    {
        return Ok(summary);
    }
    let pass_limit = exact_quality_repair_pass_limit(options);
    for _ in 0..pass_limit {
        if !tets
            .iter()
            .any(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
        {
            break;
        }
        let pass = repair_exact_quality_tets_once(
            nodes,
            tets,
            interior_seed_points,
            next_node_id,
            options,
        )?;
        if !pass.changed {
            break;
        }
        summary.pass_count += 1;
        summary.reconnected_cavity_count += pass.reconnected_cavity_count;
        summary.reconnection_quality_gain_count += pass.reconnection_quality_gain_count;
        summary.face_neighbor_reconnected_cavity_count +=
            pass.face_neighbor_reconnected_cavity_count;
        summary.connected_reconnected_cavity_count += pass.connected_reconnected_cavity_count;
        summary.node_adjacent_reconnected_cavity_count +=
            pass.node_adjacent_reconnected_cavity_count;
        summary.boundary_adjacent_reconnected_cavity_count +=
            pass.boundary_adjacent_reconnected_cavity_count;
        summary.expanded_connected_reconnected_cavity_count +=
            pass.expanded_connected_reconnected_cavity_count;
        summary.split_cavity_count += pass.split_cavity_count;
        summary.seed_star_collapse_count += pass.seed_star_collapse_count;
        summary.seed_star_relocation_count += pass.seed_star_relocation_count;
    }
    Ok(summary)
}

fn exact_quality_repair_pass_limit(options: TetCandidateOptions) -> usize {
    options
        .max_refinement_passes
        .max(1)
        .saturating_mul(8)
        .min(32)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RemainingExactQualityViolationCounts {
    total_count: usize,
    general_cavity_count: usize,
    boundary_adjacent_count: usize,
    node_adjacent_count: usize,
    interior_seed_count: usize,
    edge_star_count: usize,
}

fn remaining_exact_quality_violation_counts(
    nodes: &[TetCandidateNode],
    tets: &[TetCandidate],
    options: TetCandidateOptions,
) -> RemainingExactQualityViolationCounts {
    let face_adjacency = tet_face_adjacency(tets);
    let edge_adjacency = tet_edge_adjacency(tets);
    let node_adjacency = tet_node_adjacency(tets);
    let interior_node_ids = nodes
        .iter()
        .filter_map(|node| {
            matches!(node.source, TetCandidateNodeSource::InteriorSeed).then_some(node.node_id)
        })
        .collect::<BTreeSet<_>>();
    let mut counts = RemainingExactQualityViolationCounts::default();
    for (tet_index, tet) in tets
        .iter()
        .enumerate()
        .filter(|(_, tet)| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
    {
        counts.total_count += 1;
        let mut classified = false;
        let face_closure =
            connected_bad_tet_cavity_with_face_closure(tet_index, tets, &face_adjacency, options);
        let node_closure = connected_bad_tet_cavity_with_node_closure(
            tet_index,
            tets,
            &face_adjacency,
            &node_adjacency,
            options,
        );
        if tet_node_faces(tet.node_ids)
            .map(sorted_node_face)
            .into_iter()
            .any(|face| face_adjacency.get(&face).map_or(0, Vec::len) == 1)
        {
            counts.boundary_adjacent_count += 1;
            classified = true;
        }
        if node_closure.len() > face_closure.len() {
            counts.node_adjacent_count += 1;
            classified = true;
        }
        if tet
            .node_ids
            .into_iter()
            .any(|node_id| interior_node_ids.contains(&node_id))
        {
            counts.interior_seed_count += 1;
            classified = true;
        }
        if tet_node_edges(tet.node_ids)
            .into_iter()
            .any(|edge| edge_adjacency.get(&edge).map_or(0, Vec::len) >= 3)
        {
            counts.edge_star_count += 1;
            classified = true;
        }
        if !classified {
            counts.general_cavity_count += 1;
        }
    }
    counts
}

fn repair_exact_quality_tets_once(
    nodes: &mut Vec<TetCandidateNode>,
    tets: &mut Vec<TetCandidate>,
    interior_seed_points: &mut Vec<[f64; 3]>,
    next_node_id: &mut u32,
    options: TetCandidateOptions,
) -> Result<TetQualityRepairPassSummary, TetCandidateError> {
    let mut node_points = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut repaired = Vec::<TetCandidate>::with_capacity(tets.len());
    let face_adjacency = tet_face_adjacency(tets);
    let edge_adjacency = tet_edge_adjacency(tets);
    let node_adjacency = tet_node_adjacency(tets);
    let interior_node_ids = nodes
        .iter()
        .filter_map(|node| {
            matches!(node.source, TetCandidateNodeSource::InteriorSeed).then_some(node.node_id)
        })
        .collect::<BTreeSet<_>>();
    let removable_seed_node_ids = nodes
        .iter()
        .filter_map(|node| {
            matches!(
                node.source,
                TetCandidateNodeSource::InteriorSeed | TetCandidateNodeSource::BoundaryRecovery
            )
            .then_some(node.node_id)
        })
        .collect::<BTreeSet<_>>();
    let mut consumed = vec![false; tets.len()];
    let mut summary = TetQualityRepairPassSummary::default();
    for (tet_index, tet) in tets.iter().enumerate() {
        if consumed[tet_index] {
            continue;
        }
        if tet.exact_scaled_jacobian >= options.min_scaled_jacobian {
            repaired.push(tet.clone());
            continue;
        }
        if let Some((neighbor_indices, candidates, inserted_nodes)) =
            best_constrained_interior_seed_star_refill(
                tet_index,
                tets,
                &node_adjacency,
                &removable_seed_node_ids,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.node_adjacent_reconnected_cavity_count += 1;
            for node in inserted_nodes {
                *next_node_id = (*next_node_id).max(node.node_id.saturating_add(1));
                node_points.insert(node.node_id, node.coordinates_m);
                interior_seed_points.push(node.coordinates_m);
                nodes.push(TetCandidateNode {
                    node_id: node.node_id,
                    coordinates_m: node.coordinates_m,
                    source: TetCandidateNodeSource::InteriorSeed,
                });
            }
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates)) = best_interior_seed_node_collapse(
            tet_index,
            tets,
            &node_adjacency,
            &removable_seed_node_ids,
            &node_points,
            InteriorSeedCollapseScope::FourTetOnly,
            options,
        )? {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.seed_star_collapse_count += 1;
            repaired.extend(candidates);
            continue;
        }
        if let Some((interior_node_id, relocated_point, neighbor_indices, candidates)) =
            best_interior_seed_node_relocation(
                tet_index,
                tets,
                &node_adjacency,
                &interior_node_ids,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            let old_point = node_points.get(&interior_node_id).copied().ok_or(
                TetCandidateError::MissingSurfaceNode {
                    node_id: interior_node_id,
                },
            )?;
            for node in nodes.iter_mut() {
                if node.node_id == interior_node_id {
                    node.coordinates_m = relocated_point;
                }
            }
            node_points.insert(interior_node_id, relocated_point);
            replace_interior_seed_point(interior_seed_points, old_point, relocated_point);
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.seed_star_relocation_count += 1;
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates)) = best_interior_seed_node_collapse(
            tet_index,
            tets,
            &node_adjacency,
            &removable_seed_node_ids,
            &node_points,
            InteriorSeedCollapseScope::LargerStarsOnly,
            options,
        )? {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.seed_star_collapse_count += 1;
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates, quality_gain_only)) =
            best_multi_tet_edge_reconnection(
                tet_index,
                tets,
                &edge_adjacency,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.reconnection_quality_gain_count += usize::from(quality_gain_only);
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates, quality_gain_only)) =
            best_componentized_edge_reconnection(
                tet_index,
                tets,
                &edge_adjacency,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.reconnection_quality_gain_count += usize::from(quality_gain_only);
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates, quality_gain_only)) =
            best_three_tet_edge_reconnection(
                tet_index,
                tets,
                &edge_adjacency,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.reconnection_quality_gain_count += usize::from(quality_gain_only);
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_index, candidates, quality_gain_only)) =
            best_two_tet_reconnection(tet_index, tets, &face_adjacency, &node_points, options)?
        {
            if consumed[neighbor_index] {
                repaired.push(tet.clone());
                continue;
            }
            consumed[tet_index] = true;
            consumed[neighbor_index] = true;
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.reconnection_quality_gain_count += usize::from(quality_gain_only);
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates, quality_gain_only)) =
            best_boundary_adjacent_cavity_reconnection(
                tet_index,
                tets,
                &face_adjacency,
                &node_adjacency,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.boundary_adjacent_reconnected_cavity_count += 1;
            summary.reconnection_quality_gain_count += usize::from(quality_gain_only);
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates, quality_gain_only)) =
            best_node_adjacent_cavity_reconnection(
                tet_index,
                tets,
                &face_adjacency,
                &node_adjacency,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.node_adjacent_reconnected_cavity_count += 1;
            summary.reconnection_quality_gain_count += usize::from(quality_gain_only);
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates, quality_gain_only)) =
            best_expanded_connected_bad_cavity_reconnection(
                tet_index,
                tets,
                &face_adjacency,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.expanded_connected_reconnected_cavity_count += 1;
            summary.reconnection_quality_gain_count += usize::from(quality_gain_only);
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates, quality_gain_only)) =
            best_connected_bad_cavity_reconnection(
                tet_index,
                tets,
                &face_adjacency,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.connected_reconnected_cavity_count += 1;
            summary.reconnection_quality_gain_count += usize::from(quality_gain_only);
            repaired.extend(candidates);
            continue;
        }
        if let Some((neighbor_indices, candidates, quality_gain_only)) =
            best_face_neighbor_cavity_reconnection(
                tet_index,
                tets,
                &face_adjacency,
                &node_points,
                options,
            )?
        {
            if neighbor_indices.iter().any(|index| consumed[*index]) {
                repaired.push(tet.clone());
                continue;
            }
            for index in neighbor_indices {
                consumed[index] = true;
            }
            summary.changed = true;
            summary.reconnected_cavity_count += 1;
            summary.face_neighbor_reconnected_cavity_count += 1;
            summary.reconnection_quality_gain_count += usize::from(quality_gain_only);
            repaired.extend(candidates);
            continue;
        }
        let points = candidate_tet_points(tet, &node_points)?;
        let Some((split_point, candidates)) =
            best_centroid_split_tets(tet, *next_node_id, points, options)
        else {
            repaired.push(tet.clone());
            continue;
        };
        let candidate_below_threshold = candidates
            .iter()
            .filter(|candidate| candidate.exact_scaled_jacobian < options.min_scaled_jacobian)
            .count();
        let candidate_min_exact = candidates
            .iter()
            .map(|candidate| candidate.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        let split_improves_exact_quality = candidate_below_threshold == 0
            || (candidate_below_threshold <= 1
                && candidate_min_exact > tet.exact_scaled_jacobian + 1.0e-12);
        if candidates.len() == 4 && split_improves_exact_quality {
            let split_node_id = *next_node_id;
            *next_node_id = next_node_id.saturating_add(1);
            node_points.insert(split_node_id, split_point);
            interior_seed_points.push(split_point);
            nodes.push(TetCandidateNode {
                node_id: split_node_id,
                coordinates_m: split_point,
                source: TetCandidateNodeSource::InteriorSeed,
            });
            repaired.extend(candidates);
            summary.changed = true;
            summary.split_cavity_count += 1;
        } else {
            repaired.push(tet.clone());
        }
    }
    for (index, tet) in repaired.iter_mut().enumerate() {
        tet.tet_id = index as u32;
    }
    let referenced_node_ids = repaired
        .iter()
        .flat_map(|tet| tet.node_ids)
        .collect::<BTreeSet<_>>();
    nodes.retain(|node| {
        referenced_node_ids.contains(&node.node_id)
            || matches!(node.source, TetCandidateNodeSource::Surface)
    });
    let retained_interior_points = nodes
        .iter()
        .filter_map(|node| {
            matches!(node.source, TetCandidateNodeSource::InteriorSeed)
                .then_some(node.coordinates_m)
        })
        .collect::<Vec<_>>();
    let tolerance = MeshingTolerance::default();
    interior_seed_points.retain(|point| {
        retained_interior_points
            .iter()
            .any(|retained| tolerance.point_nearly_equal(*point, *retained, 1.0))
    });
    *tets = repaired;
    Ok(summary)
}

fn replace_interior_seed_point(
    interior_seed_points: &mut Vec<[f64; 3]>,
    old_point: [f64; 3],
    new_point: [f64; 3],
) {
    let tolerance = MeshingTolerance::default();
    if let Some(point) = interior_seed_points
        .iter_mut()
        .find(|point| tolerance.point_nearly_equal(**point, old_point, 1.0))
    {
        *point = new_point;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct UntanglingSummary {
    pass_count: usize,
    initial_near_singular_count: usize,
    final_near_singular_count: usize,
    relocated_seed_count: usize,
    reconnected_edge_star_count: usize,
    reconnected_boundary_adjacent_cavity_count: usize,
    reconnected_node_adjacent_cavity_count: usize,
}

fn untangle_near_singular_tets(
    nodes: &mut [TetCandidateNode],
    tets: &mut Vec<TetCandidate>,
    interior_seed_points: &mut Vec<[f64; 3]>,
    options: TetCandidateOptions,
) -> Result<UntanglingSummary, TetCandidateError> {
    let mut summary = UntanglingSummary::default();
    let threshold = untangling_exact_quality_threshold(options);
    if threshold <= 0.0 || !threshold.is_finite() {
        return Ok(summary);
    }
    summary.initial_near_singular_count = count_tets_below_exact_quality(tets.iter(), threshold);
    let pass_limit = options.max_refinement_passes.max(1);
    for _ in 0..pass_limit {
        if !tets.iter().any(|tet| tet.exact_scaled_jacobian < threshold) {
            break;
        }
        let node_points = nodes
            .iter()
            .map(|node| (node.node_id, node.coordinates_m))
            .collect::<BTreeMap<_, _>>();
        let node_adjacency = tet_node_adjacency(tets);
        let edge_adjacency = tet_edge_adjacency(tets);
        let face_adjacency = tet_face_adjacency(tets);
        let interior_node_ids = nodes
            .iter()
            .filter_map(|node| {
                matches!(node.source, TetCandidateNodeSource::InteriorSeed).then_some(node.node_id)
            })
            .collect::<BTreeSet<_>>();
        let mut applied = false;
        for (tet_index, tet) in tets.iter().enumerate() {
            if tet.exact_scaled_jacobian >= threshold {
                continue;
            }
            if let Some((interior_node_id, relocated_point, indices, candidates)) =
                best_interior_seed_node_untangling(
                    tet_index,
                    tets,
                    &node_adjacency,
                    &interior_node_ids,
                    &node_points,
                    threshold,
                    options,
                )?
            {
                let old_point = node_points.get(&interior_node_id).copied().ok_or(
                    TetCandidateError::MissingSurfaceNode {
                        node_id: interior_node_id,
                    },
                )?;
                for node in nodes.iter_mut() {
                    if node.node_id == interior_node_id {
                        node.coordinates_m = relocated_point;
                    }
                }
                replace_interior_seed_point(interior_seed_points, old_point, relocated_point);
                replace_tet_indices(tets, &indices, candidates);
                applied = true;
                summary.pass_count += 1;
                summary.relocated_seed_count += 1;
                break;
            }
            if let Some((indices, candidates)) = best_edge_star_untangling(
                tet_index,
                tets,
                &edge_adjacency,
                &node_points,
                threshold,
                options,
            )? {
                replace_tet_indices(tets, &indices, candidates);
                applied = true;
                summary.pass_count += 1;
                summary.reconnected_edge_star_count += 1;
                break;
            }
            if let Some((indices, candidates)) = best_boundary_adjacent_cavity_untangling(
                tet_index,
                tets,
                &face_adjacency,
                &node_adjacency,
                &node_points,
                threshold,
                options,
            )? {
                replace_tet_indices(tets, &indices, candidates);
                applied = true;
                summary.pass_count += 1;
                summary.reconnected_boundary_adjacent_cavity_count += 1;
                break;
            }
            if let Some((indices, candidates)) = best_node_adjacent_cavity_untangling(
                tet_index,
                tets,
                &face_adjacency,
                &node_adjacency,
                &node_points,
                threshold,
                options,
            )? {
                replace_tet_indices(tets, &indices, candidates);
                applied = true;
                summary.pass_count += 1;
                summary.reconnected_node_adjacent_cavity_count += 1;
                break;
            }
        }
        if !applied {
            break;
        }
    }
    summary.final_near_singular_count = count_tets_below_exact_quality(tets.iter(), threshold);
    Ok(summary)
}

fn untangling_exact_quality_threshold(options: TetCandidateOptions) -> f64 {
    (options.min_scaled_jacobian * 0.25)
        .max(1.0e-6)
        .min(options.min_scaled_jacobian)
}

fn replace_tet_indices(
    tets: &mut Vec<TetCandidate>,
    indices: &[usize],
    candidates: Vec<TetCandidate>,
) {
    let removed = indices.iter().copied().collect::<BTreeSet<_>>();
    let mut replaced = Vec::<TetCandidate>::with_capacity(tets.len() + candidates.len());
    replaced.extend(
        tets.iter()
            .enumerate()
            .filter_map(|(index, tet)| (!removed.contains(&index)).then_some(tet.clone())),
    );
    replaced.extend(candidates);
    for (index, tet) in replaced.iter_mut().enumerate() {
        tet.tet_id = index as u32;
    }
    *tets = replaced;
}

fn tet_face_adjacency(tets: &[TetCandidate]) -> BTreeMap<[u32; 3], Vec<usize>> {
    let mut adjacency = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (tet_index, tet) in tets.iter().enumerate() {
        for face in tet_node_faces(tet.node_ids) {
            adjacency
                .entry(sorted_node_face(face))
                .or_default()
                .push(tet_index);
        }
    }
    adjacency
}

fn tet_node_faces(node_ids: [u32; 4]) -> [[u32; 3]; 4] {
    [
        [node_ids[0], node_ids[1], node_ids[2]],
        [node_ids[0], node_ids[1], node_ids[3]],
        [node_ids[0], node_ids[2], node_ids[3]],
        [node_ids[1], node_ids[2], node_ids[3]],
    ]
}

fn sorted_node_face(mut face: [u32; 3]) -> [u32; 3] {
    face.sort();
    face
}

fn tet_edge_adjacency(tets: &[TetCandidate]) -> BTreeMap<[u32; 2], Vec<usize>> {
    let mut adjacency = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (tet_index, tet) in tets.iter().enumerate() {
        for edge in tet_node_edges(tet.node_ids) {
            adjacency.entry(edge).or_default().push(tet_index);
        }
    }
    adjacency
}

fn tet_node_edges(node_ids: [u32; 4]) -> [[u32; 2]; 6] {
    [
        sorted_node_edge([node_ids[0], node_ids[1]]),
        sorted_node_edge([node_ids[0], node_ids[2]]),
        sorted_node_edge([node_ids[0], node_ids[3]]),
        sorted_node_edge([node_ids[1], node_ids[2]]),
        sorted_node_edge([node_ids[1], node_ids[3]]),
        sorted_node_edge([node_ids[2], node_ids[3]]),
    ]
}

fn sorted_node_edge(mut edge: [u32; 2]) -> [u32; 2] {
    edge.sort();
    edge
}

fn tet_node_adjacency(tets: &[TetCandidate]) -> BTreeMap<u32, Vec<usize>> {
    let mut adjacency = BTreeMap::<u32, Vec<usize>>::new();
    for (tet_index, tet) in tets.iter().enumerate() {
        for node_id in tet.node_ids {
            adjacency.entry(node_id).or_default().push(tet_index);
        }
    }
    adjacency
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteriorSeedCollapseScope {
    FourTetOnly,
    LargerStarsOnly,
}

const MAX_INTERIOR_SEED_COLLAPSE_STAR_SIZE: usize = 24;
const MAX_INTERIOR_SEED_RELOCATION_STAR_SIZE: usize = 40;
pub(crate) const MAX_EDGE_STAR_RECONNECTION_SIZE: usize = 18;
const MAX_NODE_CAVITY_EXTRA_GROUP_CANDIDATES: usize = 8;

fn interior_seed_collapse_scope_matches(
    scope: InteriorSeedCollapseScope,
    star_size: usize,
) -> bool {
    match scope {
        InteriorSeedCollapseScope::FourTetOnly => star_size == 4,
        InteriorSeedCollapseScope::LargerStarsOnly => {
            (5..=MAX_INTERIOR_SEED_COLLAPSE_STAR_SIZE).contains(&star_size)
        }
    }
}

fn interior_seed_relocation_scope_matches(star_size: usize) -> bool {
    (5..=MAX_INTERIOR_SEED_RELOCATION_STAR_SIZE).contains(&star_size)
}

fn best_constrained_interior_seed_star_refill(
    tet_index: usize,
    tets: &[TetCandidate],
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    interior_node_ids: &BTreeSet<u32>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, Vec<ConstrainedCavityNode>)>, TetCandidateError>
{
    let tet = &tets[tet_index];
    let mut best = None::<(
        Vec<usize>,
        Vec<TetCandidate>,
        Vec<ConstrainedCavityNode>,
        usize,
        f64,
    )>;
    for interior_node_id in tet
        .node_ids
        .into_iter()
        .filter(|node_id| interior_node_ids.contains(node_id))
    {
        let Some(adjacent) = node_adjacency.get(&interior_node_id) else {
            continue;
        };
        if !adjacent.contains(&tet_index) || adjacent.len() < 5 {
            continue;
        }
        let mut candidate_groups = vec![adjacent.clone()];
        candidate_groups.extend(
            interior_seed_star_face_components(interior_node_id, adjacent, tets)
                .into_iter()
                .filter(|component| component.len() != adjacent.len()),
        );
        let mut seen_groups = BTreeSet::<Vec<usize>>::new();
        for group in candidate_groups {
            if !seen_groups.insert(group.clone()) || !group.contains(&tet_index) || group.len() < 5
            {
                continue;
            }
            let Some((neighbor_indices, candidates, inserted_nodes)) =
                constrained_interior_seed_star_refill_candidates(
                    tet_index,
                    tets,
                    &group,
                    node_points,
                    options,
                )?
            else {
                continue;
            };
            let original_below_count = count_exact_quality_violations(
                neighbor_indices.iter().map(|index| &tets[*index]),
                options.min_scaled_jacobian,
            );
            let candidate_below_count =
                count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
            let candidate_min_exact = min_exact_scaled_jacobian(candidates.iter());
            if candidate_below_count >= original_below_count {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(_, _, _, best_below_count, best_min_exact)| {
                    candidate_below_count < *best_below_count
                        || (candidate_below_count == *best_below_count
                            && candidate_min_exact > *best_min_exact)
                })
            {
                best = Some((
                    neighbor_indices,
                    candidates,
                    inserted_nodes,
                    candidate_below_count,
                    candidate_min_exact,
                ));
            }
        }
    }
    Ok(best
        .map(|(indices, candidates, inserted_nodes, _, _)| (indices, candidates, inserted_nodes)))
}

#[cfg(test)]
pub(crate) fn diagnostic_constrained_seed_star_refill_rejection_reason(
    tet_index: usize,
    tets: &[TetCandidate],
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    interior_node_ids: &BTreeSet<u32>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<&'static str, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut saw_seed = false;
    let mut saw_group = false;
    let mut saw_candidate = false;
    let mut saw_non_improving_candidate = false;
    for interior_node_id in tet
        .node_ids
        .into_iter()
        .filter(|node_id| interior_node_ids.contains(node_id))
    {
        saw_seed = true;
        let Some(adjacent) = node_adjacency.get(&interior_node_id) else {
            continue;
        };
        if !adjacent.contains(&tet_index) || adjacent.len() < 5 {
            continue;
        }
        let mut candidate_groups = vec![adjacent.clone()];
        candidate_groups.extend(
            interior_seed_star_face_components(interior_node_id, adjacent, tets)
                .into_iter()
                .filter(|component| component.len() != adjacent.len()),
        );
        let mut seen_groups = BTreeSet::<Vec<usize>>::new();
        for group in candidate_groups {
            if !seen_groups.insert(group.clone()) || !group.contains(&tet_index) || group.len() < 5
            {
                continue;
            }
            saw_group = true;
            let Some((neighbor_indices, candidates, _)) =
                constrained_interior_seed_star_refill_candidates(
                    tet_index,
                    tets,
                    &group,
                    node_points,
                    options,
                )?
            else {
                return diagnostic_constrained_seed_star_refill_no_candidate_reason(
                    tet_index,
                    tets,
                    &group,
                    node_points,
                    options,
                );
            };
            saw_candidate = true;
            let original_below_count = count_exact_quality_violations(
                neighbor_indices.iter().map(|index| &tets[*index]),
                options.min_scaled_jacobian,
            );
            let candidate_below_count =
                count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
            if candidate_below_count < original_below_count {
                return Ok("constrained_seed_star_refill_reconnectable");
            }
            saw_non_improving_candidate = true;
        }
    }
    if saw_non_improving_candidate {
        Ok("constrained_seed_star_refill_no_improvement")
    } else if saw_candidate {
        Ok("constrained_seed_star_refill_candidate_unclassified")
    } else if saw_group {
        Ok("constrained_seed_star_refill_no_candidate")
    } else if saw_seed {
        Ok("constrained_seed_star_refill_no_valid_group")
    } else {
        Ok("constrained_seed_star_refill_no_interior_seed")
    }
}

#[cfg(test)]
fn diagnostic_constrained_seed_star_refill_no_candidate_reason(
    tet_index: usize,
    tets: &[TetCandidate],
    adjacent: &[usize],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<&'static str, TetCandidateError> {
    let cavity = match constrained_cavity_from_selected_tets_with_anchor_trim(
        tets,
        adjacent,
        tet_index,
        vec![],
    ) {
        Ok(Some(cavity)) => cavity,
        Ok(None) => return Ok("constrained_seed_star_refill_empty_trimmed_cavity"),
        Err(err) => {
            return Ok(diagnostic_constrained_seed_star_cavity_extraction_bucket(
                &err,
            ))
        }
    };
    let boundary_node_ids = cavity
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids)
        .collect::<BTreeSet<_>>();
    let boundary_nodes = boundary_node_ids
        .iter()
        .map(|node_id| {
            Ok(ConstrainedCavityNode {
                node_id: *node_id,
                coordinates_m: *node_points
                    .get(node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?,
            })
        })
        .collect::<Result<Vec<_>, TetCandidateError>>()?;
    let interior_candidates = constrained_seed_star_refill_interior_candidates(
        tets,
        tet_index,
        adjacent,
        &boundary_node_ids,
        node_points,
    )?;
    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &boundary_nodes,
        &interior_candidates,
        ConstrainedCavityRefillOptions {
            min_volume_m3: options.min_volume_m3,
            max_aspect_ratio: options.max_aspect_ratio,
            min_scaled_jacobian: options.min_scaled_jacobian,
            volume_relative_tolerance: 1.0e-9,
            min_protected_node_distance_m: 0.0,
        },
    )
    .map_err(|_| TetCandidateError::InvalidOptions)?;
    if evaluation.refill.is_some() {
        return Ok("constrained_seed_star_refill_candidate_unclassified");
    }
    Ok(diagnostic_constrained_seed_star_refill_reason_bucket(
        &evaluation.rejected_by_reason,
    ))
}

#[cfg(test)]
fn diagnostic_constrained_seed_star_cavity_extraction_bucket(
    err: &ConstrainedCavityExtractionError,
) -> &'static str {
    match err {
        ConstrainedCavityExtractionError::EmptySelection => {
            "constrained_seed_star_refill_empty_selection"
        }
        ConstrainedCavityExtractionError::SelectedTetIndexOutOfBounds { .. } => {
            "constrained_seed_star_refill_selected_tet_out_of_bounds"
        }
        ConstrainedCavityExtractionError::DuplicateSelectedTetIndex { .. } => {
            "constrained_seed_star_refill_duplicate_selected_tet"
        }
        ConstrainedCavityExtractionError::Validation(err) => {
            diagnostic_constrained_seed_star_cavity_validation_bucket(err)
        }
    }
}

#[cfg(test)]
fn diagnostic_constrained_seed_star_cavity_validation_bucket(
    err: &ConstrainedCavityValidationError,
) -> &'static str {
    match err {
        ConstrainedCavityValidationError::EmptyRemovedTetSet => {
            "constrained_seed_star_refill_empty_removed_tet_set"
        }
        ConstrainedCavityValidationError::InvalidTargetVolume { .. } => {
            "constrained_seed_star_refill_invalid_target_volume"
        }
        ConstrainedCavityValidationError::TooFewBoundaryFaces { .. } => {
            "constrained_seed_star_refill_too_few_boundary_faces"
        }
        ConstrainedCavityValidationError::DegenerateBoundaryFace { .. } => {
            "constrained_seed_star_refill_degenerate_boundary_face"
        }
        ConstrainedCavityValidationError::DuplicateBoundaryFace { .. } => {
            "constrained_seed_star_refill_duplicate_boundary_face"
        }
        ConstrainedCavityValidationError::NonManifoldBoundaryEdge { .. } => {
            "constrained_seed_star_refill_non_manifold_boundary_edge"
        }
        ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary { .. } => {
            "constrained_seed_star_refill_protected_node_outside_boundary"
        }
        ConstrainedCavityValidationError::InvalidRefillVolume { .. } => {
            "constrained_seed_star_refill_invalid_refill_volume"
        }
        ConstrainedCavityValidationError::BoundaryFaceCountMismatch { .. } => {
            "constrained_seed_star_refill_boundary_face_count_mismatch"
        }
        ConstrainedCavityValidationError::MissingBoundaryFace { .. } => {
            "constrained_seed_star_refill_missing_boundary_face"
        }
        ConstrainedCavityValidationError::UnexpectedBoundaryFace { .. } => {
            "constrained_seed_star_refill_unexpected_boundary_face"
        }
        ConstrainedCavityValidationError::BoundarySourceFaceMismatch { .. } => {
            "constrained_seed_star_refill_boundary_source_face_mismatch"
        }
        ConstrainedCavityValidationError::BoundarySourceEdgeMismatch { .. } => {
            "constrained_seed_star_refill_boundary_source_edge_mismatch"
        }
        ConstrainedCavityValidationError::BoundaryRegionMismatch { .. } => {
            "constrained_seed_star_refill_boundary_region_mismatch"
        }
    }
}

#[cfg(test)]
fn diagnostic_constrained_seed_star_refill_reason_bucket(
    rejected_by_reason: &BTreeMap<String, usize>,
) -> &'static str {
    for reason in [
        "star_tet_scaled_jacobian",
        "missing_boundary_face",
        "unexpected_boundary_face",
        "boundary_face_count_mismatch",
        "volume_mismatch",
        "interior_point_outside_cavity",
        "protected_boundary_distance",
        "boundary_node_delaunay_empty",
        "star_tet_aspect_ratio",
        "star_tet_min_volume",
    ] {
        if rejected_by_reason.contains_key(reason) {
            return match reason {
                "star_tet_scaled_jacobian" => {
                    "constrained_seed_star_refill_star_tet_scaled_jacobian"
                }
                "missing_boundary_face" => "constrained_seed_star_refill_missing_boundary_face",
                "unexpected_boundary_face" => {
                    "constrained_seed_star_refill_unexpected_boundary_face"
                }
                "boundary_face_count_mismatch" => {
                    "constrained_seed_star_refill_boundary_face_count_mismatch"
                }
                "volume_mismatch" => "constrained_seed_star_refill_volume_mismatch",
                "interior_point_outside_cavity" => {
                    "constrained_seed_star_refill_interior_point_outside_cavity"
                }
                "protected_boundary_distance" => {
                    "constrained_seed_star_refill_protected_boundary_distance"
                }
                "boundary_node_delaunay_empty" => {
                    "constrained_seed_star_refill_boundary_node_delaunay_empty"
                }
                "star_tet_aspect_ratio" => "constrained_seed_star_refill_star_tet_aspect_ratio",
                "star_tet_min_volume" => "constrained_seed_star_refill_star_tet_min_volume",
                _ => "constrained_seed_star_refill_no_candidate",
            };
        }
    }
    if rejected_by_reason.is_empty() {
        "constrained_seed_star_refill_no_rejection_reason"
    } else {
        "constrained_seed_star_refill_other_rejection"
    }
}

fn interior_seed_star_face_components(
    seed_node_id: u32,
    adjacent: &[usize],
    tets: &[TetCandidate],
) -> Vec<Vec<usize>> {
    let adjacent_set = adjacent.iter().copied().collect::<BTreeSet<_>>();
    let mut face_owners = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for tet_index in adjacent {
        for face in tet_node_faces(tets[*tet_index].node_ids) {
            if face.contains(&seed_node_id) {
                face_owners
                    .entry(sorted_node_face(face))
                    .or_default()
                    .push(*tet_index);
            }
        }
    }
    let mut graph = BTreeMap::<usize, BTreeSet<usize>>::new();
    for tet_index in adjacent {
        graph.entry(*tet_index).or_default();
    }
    for owners in face_owners.values() {
        for left in owners {
            for right in owners {
                if left != right && adjacent_set.contains(left) && adjacent_set.contains(right) {
                    graph.entry(*left).or_default().insert(*right);
                }
            }
        }
    }

    let mut visited = BTreeSet::<usize>::new();
    let mut components = Vec::<Vec<usize>>::new();
    for start in adjacent {
        if !visited.insert(*start) {
            continue;
        }
        let mut component = Vec::<usize>::new();
        let mut pending = vec![*start];
        while let Some(index) = pending.pop() {
            component.push(index);
            if let Some(neighbors) = graph.get(&index) {
                for neighbor in neighbors {
                    if visited.insert(*neighbor) {
                        pending.push(*neighbor);
                    }
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

fn constrained_interior_seed_star_refill_candidates(
    tet_index: usize,
    tets: &[TetCandidate],
    adjacent: &[usize],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, Vec<ConstrainedCavityNode>)>, TetCandidateError>
{
    let Some(cavity) =
        constrained_cavity_from_selected_tets_with_anchor_trim(tets, adjacent, tet_index, vec![])
            .map_err(|_| TetCandidateError::InvalidOptions)?
    else {
        return Ok(None);
    };
    let index_by_tet_id = tets
        .iter()
        .enumerate()
        .map(|(index, tet)| (tet.tet_id, index))
        .collect::<BTreeMap<_, _>>();
    let selected_indices = cavity
        .removed_tet_ids
        .iter()
        .map(|tet_id| {
            index_by_tet_id
                .get(tet_id)
                .copied()
                .ok_or(TetCandidateError::InvalidOptions)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if selected_indices.is_empty() || !selected_indices.contains(&tet_index) {
        return Ok(None);
    }
    let boundary_node_ids = cavity
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids)
        .collect::<BTreeSet<_>>();
    let boundary_nodes = boundary_node_ids
        .iter()
        .map(|node_id| {
            Ok(ConstrainedCavityNode {
                node_id: *node_id,
                coordinates_m: *node_points
                    .get(node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?,
            })
        })
        .collect::<Result<Vec<_>, TetCandidateError>>()?;
    let interior_candidates = constrained_seed_star_refill_interior_candidates(
        tets,
        tet_index,
        adjacent,
        &boundary_node_ids,
        node_points,
    )?;
    let refill = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &boundary_nodes,
        &interior_candidates,
        ConstrainedCavityRefillOptions {
            min_volume_m3: options.min_volume_m3,
            max_aspect_ratio: options.max_aspect_ratio,
            min_scaled_jacobian: options.min_scaled_jacobian,
            volume_relative_tolerance: 1.0e-9,
            min_protected_node_distance_m: 0.0,
        },
    )
    .map_err(|_| TetCandidateError::InvalidOptions)?
    .refill;
    let Some(refill) = refill else {
        return Ok(None);
    };
    let reference = &tets[selected_indices[0]];
    let mut inserted_nodes = refill.inserted_nodes.clone();
    let mut inserted_node_ids = inserted_nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    for node_id in refill
        .tets
        .iter()
        .flat_map(|tet| tet.node_ids)
        .filter(|node_id| !node_points.contains_key(node_id))
        .collect::<BTreeSet<_>>()
    {
        if inserted_node_ids.contains(&node_id) {
            continue;
        }
        let node = interior_candidates
            .iter()
            .find(|candidate| candidate.node_id == node_id)
            .cloned()
            .ok_or(TetCandidateError::MissingSurfaceNode { node_id })?;
        inserted_node_ids.insert(node.node_id);
        inserted_nodes.push(node);
    }
    let candidates = refill
        .tets
        .into_iter()
        .map(|tet| TetCandidate {
            tet_id: 0,
            component_id: reference.component_id,
            node_ids: tet.node_ids,
            source_surface_element_id: reference.source_surface_element_id,
            region_ids: reference.region_ids.clone(),
            volume_m3: tet.volume_m3,
            aspect_ratio: tet.aspect_ratio,
            exact_scaled_jacobian: tet.exact_scaled_jacobian,
        })
        .collect::<Vec<_>>();
    Ok(Some((selected_indices, candidates, inserted_nodes)))
}

fn constrained_seed_star_refill_interior_candidates(
    tets: &[TetCandidate],
    tet_index: usize,
    adjacent: &[usize],
    boundary_node_ids: &BTreeSet<u32>,
    node_points: &BTreeMap<u32, [f64; 3]>,
) -> Result<Vec<ConstrainedCavityNode>, TetCandidateError> {
    let mut candidates = Vec::<ConstrainedCavityNode>::new();
    let mut used_node_ids = BTreeSet::<u32>::new();
    for node_id in tets[tet_index].node_ids {
        if !boundary_node_ids.contains(&node_id) {
            candidates.push(ConstrainedCavityNode {
                node_id,
                coordinates_m: *node_points
                    .get(&node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id })?,
            });
            used_node_ids.insert(node_id);
        }
    }
    if let Some(centroid) = boundary_node_centroid(boundary_node_ids, node_points)? {
        push_synthetic_refill_candidate(tets, &mut used_node_ids, &mut candidates, centroid);
    }
    let tet_points = candidate_tet_points(&tets[tet_index], node_points)?;
    let target_tet_centroid = tet_centroid(tet_points);
    push_synthetic_refill_candidate(
        tets,
        &mut used_node_ids,
        &mut candidates,
        target_tet_centroid,
    );
    if adjacent.len() > 1 {
        let mut weighted = [0.0_f64; 3];
        let mut total_volume = 0.0_f64;
        for index in adjacent {
            let tet = &tets[*index];
            let centroid = tet_centroid(candidate_tet_points(tet, node_points)?);
            let weight = tet.volume_m3.max(0.0);
            weighted[0] += centroid[0] * weight;
            weighted[1] += centroid[1] * weight;
            weighted[2] += centroid[2] * weight;
            total_volume += weight;
        }
        if total_volume > 0.0 {
            let weighted_centroid = [
                weighted[0] / total_volume,
                weighted[1] / total_volume,
                weighted[2] / total_volume,
            ];
            push_synthetic_refill_candidate(
                tets,
                &mut used_node_ids,
                &mut candidates,
                weighted_centroid,
            );
        }
    }
    dedup_refill_candidate_points(candidates)
}

fn push_synthetic_refill_candidate(
    tets: &[TetCandidate],
    used_node_ids: &mut BTreeSet<u32>,
    candidates: &mut Vec<ConstrainedCavityNode>,
    coordinates_m: [f64; 3],
) {
    let node_id = synthetic_refill_candidate_node_id(tets, used_node_ids);
    candidates.push(ConstrainedCavityNode {
        node_id,
        coordinates_m,
    });
}

fn boundary_node_centroid(
    boundary_node_ids: &BTreeSet<u32>,
    node_points: &BTreeMap<u32, [f64; 3]>,
) -> Result<Option<[f64; 3]>, TetCandidateError> {
    if boundary_node_ids.is_empty() {
        return Ok(None);
    }
    let mut centroid = [0.0_f64; 3];
    for node_id in boundary_node_ids {
        let point = node_points
            .get(node_id)
            .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?;
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    let scale = 1.0 / boundary_node_ids.len() as f64;
    Ok(Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ]))
}

fn synthetic_refill_candidate_node_id(
    tets: &[TetCandidate],
    used_node_ids: &mut BTreeSet<u32>,
) -> u32 {
    let mut node_id = tets
        .iter()
        .flat_map(|tet| tet.node_ids)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    while !used_node_ids.insert(node_id) {
        node_id = node_id.saturating_add(1);
    }
    node_id
}

fn dedup_refill_candidate_points(
    candidates: Vec<ConstrainedCavityNode>,
) -> Result<Vec<ConstrainedCavityNode>, TetCandidateError> {
    let tolerance = MeshingTolerance::default();
    let mut deduped = Vec::<ConstrainedCavityNode>::new();
    for candidate in candidates {
        if candidate
            .coordinates_m
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(TetCandidateError::InvalidOptions);
        }
        if deduped.iter().any(|existing| {
            distance_squared(existing.coordinates_m, candidate.coordinates_m)
                <= tolerance.absolute_m * tolerance.absolute_m
        }) {
            continue;
        }
        deduped.push(candidate);
    }
    Ok(deduped)
}

fn best_interior_seed_node_collapse(
    tet_index: usize,
    tets: &[TetCandidate],
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    interior_node_ids: &BTreeSet<u32>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    scope: InteriorSeedCollapseScope,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>)>, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut best = None::<(Vec<usize>, Vec<TetCandidate>, usize, f64)>;
    for interior_node_id in tet
        .node_ids
        .into_iter()
        .filter(|node_id| interior_node_ids.contains(node_id))
    {
        let Some(adjacent) = node_adjacency.get(&interior_node_id) else {
            continue;
        };
        if !interior_seed_collapse_scope_matches(scope, adjacent.len())
            || !adjacent.contains(&tet_index)
        {
            continue;
        }
        let original_below_count = count_exact_quality_violations(
            adjacent.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let Some(candidates) = interior_seed_node_collapse_candidates(
            adjacent,
            interior_node_id,
            tets,
            node_points,
            options,
        )?
        else {
            continue;
        };
        let candidate_below_count = candidates
            .iter()
            .filter(|candidate| candidate.exact_scaled_jacobian < options.min_scaled_jacobian)
            .count();
        let candidate_min_exact = candidates
            .iter()
            .map(|candidate| candidate.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        let original_min_exact = adjacent
            .iter()
            .map(|index| tets[*index].exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        let improves = candidate_below_count < original_below_count
            || (candidate_below_count == original_below_count
                && candidate_min_exact > original_min_exact + 1.0e-12);
        if !improves {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(_, _, best_below_count, best_min_exact)| {
                candidate_below_count < *best_below_count
                    || (candidate_below_count == *best_below_count
                        && candidate_min_exact > *best_min_exact)
            })
        {
            best = Some((
                adjacent.clone(),
                candidates,
                candidate_below_count,
                candidate_min_exact,
            ));
        }
    }
    Ok(best.map(|(indices, candidates, _, _)| (indices, candidates)))
}

fn interior_seed_node_collapse_candidates(
    adjacent: &[usize],
    interior_node_id: u32,
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    let reference = &tets[adjacent[0]];
    let mut original_volume = 0.0_f64;
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for index in adjacent {
        let tet = &tets[*index];
        original_volume += tet.volume_m3;
        if tet.component_id != reference.component_id || !tet.node_ids.contains(&interior_node_id) {
            return Ok(None);
        }
        for node_id in tet.node_ids {
            if node_id != interior_node_id {
                boundary_nodes.insert(node_id);
            }
        }
    }
    if boundary_nodes.len() != 4 {
        return generalized_interior_seed_node_collapse_candidates(
            reference,
            original_volume,
            &boundary_nodes,
            node_points,
            options,
        );
    }
    let node_ids = boundary_nodes.into_iter().collect::<Vec<_>>();
    let node_ids = [node_ids[0], node_ids[1], node_ids[2], node_ids[3]];
    let points = [
        *node_points
            .get(&node_ids[0])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[0],
            })?,
        *node_points
            .get(&node_ids[1])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[1],
            })?,
        *node_points
            .get(&node_ids[2])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[2],
            })?,
        *node_points
            .get(&node_ids[3])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[3],
            })?,
    ];
    let Some(candidate) = raw_candidate_tet(
        reference.component_id,
        reference.source_surface_element_id,
        &reference.region_ids,
        node_ids,
        points,
        options,
    ) else {
        return Ok(None);
    };
    if (candidate.volume_m3 - original_volume).abs() > original_volume.max(1.0e-18) * 1.0e-9 {
        return Ok(None);
    }
    Ok(Some(vec![candidate]))
}

fn generalized_interior_seed_node_collapse_candidates(
    reference: &TetCandidate,
    original_volume: f64,
    boundary_nodes: &BTreeSet<u32>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    if boundary_nodes.len() < 5 || boundary_nodes.len() > 24 {
        return Ok(None);
    }
    let points = boundary_nodes
        .iter()
        .map(|node_id| {
            Ok(ConnectivityPoint {
                node_id: *node_id,
                coordinates_m: *node_points
                    .get(node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?,
                is_super: false,
            })
        })
        .collect::<Result<Vec<_>, TetCandidateError>>()?;
    let mut candidates = Vec::<TetCandidate>::new();
    for tet in tetrahedralize_points(&points) {
        let node_ids = tet.vertices.map(|index| points[index].node_id);
        let tet_points = tet.vertices.map(|index| points[index].coordinates_m);
        let Some(candidate) = raw_candidate_tet(
            reference.component_id,
            reference.source_surface_element_id,
            &reference.region_ids,
            node_ids,
            tet_points,
            options,
        ) else {
            return Ok(None);
        };
        candidates.push(candidate);
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    let candidate_volume = candidates
        .iter()
        .map(|candidate| candidate.volume_m3)
        .sum::<f64>();
    if (candidate_volume - original_volume).abs() > original_volume.max(1.0e-18) * 1.0e-9 {
        return Ok(None);
    }
    Ok(Some(candidates))
}

fn best_interior_seed_node_relocation(
    tet_index: usize,
    tets: &[TetCandidate],
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    interior_node_ids: &BTreeSet<u32>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(u32, [f64; 3], Vec<usize>, Vec<TetCandidate>)>, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut best = None::<(u32, [f64; 3], Vec<usize>, Vec<TetCandidate>, usize, f64)>;
    for interior_node_id in tet
        .node_ids
        .into_iter()
        .filter(|node_id| interior_node_ids.contains(node_id))
    {
        let Some(adjacent) = node_adjacency.get(&interior_node_id) else {
            continue;
        };
        if !interior_seed_relocation_scope_matches(adjacent.len()) || !adjacent.contains(&tet_index)
        {
            continue;
        }
        let original_below_count = count_exact_quality_violations(
            adjacent.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        if original_below_count == 0 {
            continue;
        }
        let original_min_exact = adjacent
            .iter()
            .map(|index| tets[*index].exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        for relocated_point in
            interior_seed_node_relocation_points(adjacent, interior_node_id, tets, node_points)?
        {
            let Some(candidates) = interior_seed_node_relocation_candidates(
                adjacent,
                interior_node_id,
                relocated_point,
                tets,
                node_points,
                options,
            )?
            else {
                continue;
            };
            let candidate_below_count = candidates
                .iter()
                .filter(|candidate| candidate.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count();
            let candidate_min_exact = candidates
                .iter()
                .map(|candidate| candidate.exact_scaled_jacobian)
                .fold(f64::INFINITY, f64::min);
            let improves = candidate_below_count < original_below_count
                || (candidate_below_count == original_below_count
                    && candidate_min_exact > original_min_exact + 1.0e-12);
            if !improves {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(_, _, _, _, best_below_count, best_min_exact)| {
                    candidate_below_count < *best_below_count
                        || (candidate_below_count == *best_below_count
                            && candidate_min_exact > *best_min_exact)
                })
            {
                best = Some((
                    interior_node_id,
                    relocated_point,
                    adjacent.clone(),
                    candidates,
                    candidate_below_count,
                    candidate_min_exact,
                ));
            }
        }
    }
    Ok(best
        .map(|(node_id, point, indices, candidates, _, _)| (node_id, point, indices, candidates)))
}

fn best_interior_seed_node_untangling(
    tet_index: usize,
    tets: &[TetCandidate],
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    interior_node_ids: &BTreeSet<u32>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    threshold: f64,
    options: TetCandidateOptions,
) -> Result<Option<(u32, [f64; 3], Vec<usize>, Vec<TetCandidate>)>, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut best = None::<(u32, [f64; 3], Vec<usize>, Vec<TetCandidate>, usize, f64)>;
    for interior_node_id in tet
        .node_ids
        .into_iter()
        .filter(|node_id| interior_node_ids.contains(node_id))
    {
        let Some(adjacent) = node_adjacency.get(&interior_node_id) else {
            continue;
        };
        if !interior_seed_relocation_scope_matches(adjacent.len()) || !adjacent.contains(&tet_index)
        {
            continue;
        }
        let original_near_singular_count =
            count_tets_below_exact_quality(adjacent.iter().map(|index| &tets[*index]), threshold);
        if original_near_singular_count == 0 {
            continue;
        }
        let original_full_bad_count = count_exact_quality_violations(
            adjacent.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact =
            min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
        for relocated_point in
            interior_seed_node_relocation_points(adjacent, interior_node_id, tets, node_points)?
        {
            let Some(candidates) = interior_seed_node_relocation_candidates(
                adjacent,
                interior_node_id,
                relocated_point,
                tets,
                node_points,
                options,
            )?
            else {
                continue;
            };
            let candidate_full_bad_count =
                count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
            if candidate_full_bad_count > original_full_bad_count {
                continue;
            }
            let candidate_near_singular_count =
                count_tets_below_exact_quality(candidates.iter(), threshold);
            let candidate_min_exact = min_exact_scaled_jacobian(candidates.iter());
            let improves = candidate_near_singular_count < original_near_singular_count
                || (candidate_near_singular_count == original_near_singular_count
                    && candidate_min_exact > original_min_exact + 1.0e-12);
            if !improves {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(_, _, _, _, best_count, best_min_exact)| {
                    candidate_near_singular_count < *best_count
                        || (candidate_near_singular_count == *best_count
                            && candidate_min_exact > *best_min_exact)
                })
            {
                best = Some((
                    interior_node_id,
                    relocated_point,
                    adjacent.clone(),
                    candidates,
                    candidate_near_singular_count,
                    candidate_min_exact,
                ));
            }
        }
    }
    Ok(best
        .map(|(node_id, point, indices, candidates, _, _)| (node_id, point, indices, candidates)))
}

fn best_edge_star_untangling(
    tet_index: usize,
    tets: &[TetCandidate],
    edge_adjacency: &BTreeMap<[u32; 2], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    threshold: f64,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>)>, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut best = None::<(Vec<usize>, Vec<TetCandidate>, usize, f64)>;
    for edge in tet_node_edges(tet.node_ids) {
        let Some(adjacent) = edge_adjacency.get(&edge) else {
            continue;
        };
        if !(3..=MAX_EDGE_STAR_RECONNECTION_SIZE).contains(&adjacent.len())
            || !adjacent.contains(&tet_index)
        {
            continue;
        }
        let original_near_singular_count =
            count_tets_below_exact_quality(adjacent.iter().map(|index| &tets[*index]), threshold);
        if original_near_singular_count == 0 {
            continue;
        }
        let original_full_bad_count = count_exact_quality_violations(
            adjacent.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact =
            min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
        let candidates = if adjacent.len() == 3 {
            three_tet_edge_reconnection_candidates(adjacent, edge, tets, node_points, options)?
        } else {
            multi_tet_edge_reconnection_candidates(adjacent, edge, tets, node_points, options)?
        };
        let Some(candidates) = candidates else {
            continue;
        };
        let candidate_full_bad_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        if candidate_full_bad_count > original_full_bad_count {
            continue;
        }
        let candidate_near_singular_count =
            count_tets_below_exact_quality(candidates.iter(), threshold);
        let candidate_min_exact = min_exact_scaled_jacobian(candidates.iter());
        let improves = candidate_near_singular_count < original_near_singular_count
            || (candidate_near_singular_count == original_near_singular_count
                && candidate_min_exact > original_min_exact + 1.0e-12);
        if !improves {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(_, _, best_count, best_min_exact)| {
                candidate_near_singular_count < *best_count
                    || (candidate_near_singular_count == *best_count
                        && candidate_min_exact > *best_min_exact)
            })
        {
            best = Some((
                adjacent.clone(),
                candidates,
                candidate_near_singular_count,
                candidate_min_exact,
            ));
        }
    }
    Ok(best.map(|(indices, candidates, _, _)| (indices, candidates)))
}

fn best_boundary_adjacent_cavity_untangling(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    threshold: f64,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>)>, TetCandidateError> {
    let adjacent = boundary_adjacent_bad_tet_cavity_with_node_closure(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
    );
    let expanded = boundary_adjacent_bad_tet_cavity_with_node_closure_layers(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
        2,
    );
    let mut candidate_groups = vec![adjacent];
    if candidate_groups
        .first()
        .is_some_and(|group| group.as_slice() != expanded.as_slice())
    {
        candidate_groups.push(expanded);
    }

    let mut best = None::<(Vec<usize>, Vec<TetCandidate>, usize, f64)>;
    for group in candidate_groups {
        if group.len() < 4 || group.len() > 24 {
            continue;
        }
        let original_near_singular_count =
            count_tets_below_exact_quality(group.iter().map(|index| &tets[*index]), threshold);
        if original_near_singular_count == 0 {
            continue;
        }
        let original_full_bad_count = count_exact_quality_violations(
            group.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact = min_exact_scaled_jacobian(group.iter().map(|index| &tets[*index]));
        let Some(candidates) =
            face_neighbor_cavity_reconnection_candidates(&group, tets, node_points, options)?
        else {
            continue;
        };
        let candidate_full_bad_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        if candidate_full_bad_count > original_full_bad_count {
            continue;
        }
        let candidate_near_singular_count =
            count_tets_below_exact_quality(candidates.iter(), threshold);
        let candidate_min_exact = min_exact_scaled_jacobian(candidates.iter());
        let improves = candidate_near_singular_count < original_near_singular_count
            || (candidate_near_singular_count == original_near_singular_count
                && candidate_min_exact > original_min_exact + 1.0e-12);
        if !improves {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(_, _, best_count, best_min_exact)| {
                candidate_near_singular_count < *best_count
                    || (candidate_near_singular_count == *best_count
                        && candidate_min_exact > *best_min_exact)
            })
        {
            best = Some((
                group,
                candidates,
                candidate_near_singular_count,
                candidate_min_exact,
            ));
        }
    }

    Ok(best.map(|(indices, candidates, _, _)| (indices, candidates)))
}

fn best_node_adjacent_cavity_untangling(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    threshold: f64,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>)>, TetCandidateError> {
    let adjacent = connected_bad_tet_cavity_with_node_closure(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
    );
    if adjacent.len() < 3 || adjacent.len() > 24 {
        return Ok(None);
    }
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    if adjacent.len() <= face_closure.len() {
        return Ok(None);
    }

    let base = face_closure.into_iter().collect::<BTreeSet<_>>();
    let extra = bounded_node_cavity_extra_indices(
        adjacent
            .iter()
            .copied()
            .filter(|index| !base.contains(index))
            .collect::<Vec<_>>(),
        tets,
    );
    let mut candidate_groups = vec![adjacent.clone()];
    for extra_index in &extra {
        let mut group = base.clone();
        group.insert(*extra_index);
        candidate_groups.push(group.into_iter().collect());
    }
    for left in 0..extra.len() {
        for right in (left + 1)..extra.len() {
            let mut group = base.clone();
            group.insert(extra[left]);
            group.insert(extra[right]);
            candidate_groups.push(group.into_iter().collect());
        }
    }

    let mut best = None::<(Vec<usize>, Vec<TetCandidate>, usize, f64)>;
    for group in candidate_groups {
        if group.len() < 3 || group.len() > 24 {
            continue;
        }
        let original_near_singular_count =
            count_tets_below_exact_quality(group.iter().map(|index| &tets[*index]), threshold);
        if original_near_singular_count == 0 {
            continue;
        }
        let original_full_bad_count = count_exact_quality_violations(
            group.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact = min_exact_scaled_jacobian(group.iter().map(|index| &tets[*index]));
        let Some(candidates) =
            face_neighbor_cavity_reconnection_candidates(&group, tets, node_points, options)?
        else {
            continue;
        };
        let candidate_full_bad_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        if candidate_full_bad_count > original_full_bad_count {
            continue;
        }
        let candidate_near_singular_count =
            count_tets_below_exact_quality(candidates.iter(), threshold);
        let candidate_min_exact = min_exact_scaled_jacobian(candidates.iter());
        let improves = candidate_near_singular_count < original_near_singular_count
            || (candidate_near_singular_count == original_near_singular_count
                && candidate_min_exact > original_min_exact + 1.0e-12);
        if !improves {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(_, _, best_count, best_min_exact)| {
                candidate_near_singular_count < *best_count
                    || (candidate_near_singular_count == *best_count
                        && candidate_min_exact > *best_min_exact)
            })
        {
            best = Some((
                group,
                candidates,
                candidate_near_singular_count,
                candidate_min_exact,
            ));
        }
    }

    Ok(best.map(|(indices, candidates, _, _)| (indices, candidates)))
}

fn interior_seed_node_relocation_points(
    adjacent: &[usize],
    interior_node_id: u32,
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
) -> Result<Vec<[f64; 3]>, TetCandidateError> {
    let current =
        *node_points
            .get(&interior_node_id)
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: interior_node_id,
            })?;
    let mut boundary_nodes = BTreeSet::<u32>::new();
    let mut weighted_sum = [0.0_f64; 3];
    let mut weight_total = 0.0_f64;
    for index in adjacent {
        let tet = &tets[*index];
        if !tet.node_ids.contains(&interior_node_id) {
            return Ok(Vec::new());
        }
        let points = candidate_tet_points(tet, node_points)?;
        let centroid = tet_centroid(points);
        let weight = tet.volume_m3.max(1.0e-18);
        for axis in 0..3 {
            weighted_sum[axis] += centroid[axis] * weight;
        }
        weight_total += weight;
        for node_id in tet.node_ids {
            if node_id != interior_node_id {
                boundary_nodes.insert(node_id);
            }
        }
    }
    if boundary_nodes.len() < 4 || weight_total <= f64::EPSILON {
        return Ok(Vec::new());
    }
    let mut boundary_centroid = [0.0_f64; 3];
    let mut min_boundary_distance = f64::INFINITY;
    for node_id in &boundary_nodes {
        let point = *node_points
            .get(node_id)
            .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?;
        min_boundary_distance = min_boundary_distance.min(distance(current, point));
        for axis in 0..3 {
            boundary_centroid[axis] += point[axis];
        }
    }
    for value in &mut boundary_centroid {
        *value /= boundary_nodes.len() as f64;
    }
    let weighted_centroid = [
        weighted_sum[0] / weight_total,
        weighted_sum[1] / weight_total,
        weighted_sum[2] / weight_total,
    ];
    let mut points = Vec::<[f64; 3]>::new();
    for target in [boundary_centroid, weighted_centroid] {
        points.push(target);
        for fraction in [0.75, 0.5, 0.25] {
            points.push([
                current[0] * (1.0 - fraction) + target[0] * fraction,
                current[1] * (1.0 - fraction) + target[1] * fraction,
                current[2] * (1.0 - fraction) + target[2] * fraction,
            ]);
        }
    }
    let local_radius = min_boundary_distance * 0.10;
    if local_radius.is_finite() && local_radius > MeshingTolerance::default().absolute_m {
        let directions = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for fraction in [0.5, 1.0] {
            let radius = local_radius * fraction;
            for direction in directions {
                points.push([
                    current[0] + direction[0] * radius,
                    current[1] + direction[1] * radius,
                    current[2] + direction[2] * radius,
                ]);
            }
        }
    }
    let tolerance = MeshingTolerance::default();
    let mut unique = Vec::<[f64; 3]>::new();
    for point in points {
        if !tolerance.point_nearly_equal(point, current, 1.0)
            && !unique
                .iter()
                .any(|existing| tolerance.point_nearly_equal(*existing, point, 1.0))
        {
            unique.push(point);
        }
    }
    Ok(unique)
}

fn interior_seed_node_relocation_candidates(
    adjacent: &[usize],
    interior_node_id: u32,
    relocated_point: [f64; 3],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    let reference = &tets[adjacent[0]];
    if !relocated_seed_point_has_star_boundary_clearance(
        adjacent,
        interior_node_id,
        relocated_point,
        tets,
        node_points,
        MeshingTolerance::default(),
    )? {
        return Ok(None);
    }
    let original_volume = adjacent
        .iter()
        .map(|index| tets[*index].volume_m3)
        .sum::<f64>();
    let mut candidates = Vec::<TetCandidate>::with_capacity(adjacent.len());
    for index in adjacent {
        let tet = &tets[*index];
        if tet.component_id != reference.component_id || !tet.node_ids.contains(&interior_node_id) {
            return Ok(None);
        }
        let mut points = [[0.0_f64; 3]; 4];
        for (node_index, node_id) in tet.node_ids.iter().copied().enumerate() {
            points[node_index] = if node_id == interior_node_id {
                relocated_point
            } else {
                *node_points
                    .get(&node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id })?
            };
        }
        let Some(candidate) = raw_candidate_tet(
            tet.component_id,
            tet.source_surface_element_id,
            &tet.region_ids,
            tet.node_ids,
            points,
            options,
        ) else {
            return Ok(None);
        };
        candidates.push(candidate);
    }
    let candidate_volume = candidates
        .iter()
        .map(|candidate| candidate.volume_m3)
        .sum::<f64>();
    if (candidate_volume - original_volume).abs() > original_volume.max(1.0e-18) * 1.0e-9 {
        return Ok(None);
    }
    Ok(Some(candidates))
}

fn relocated_seed_point_has_star_boundary_clearance(
    adjacent: &[usize],
    interior_node_id: u32,
    relocated_point: [f64; 3],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    tolerance: MeshingTolerance,
) -> Result<bool, TetCandidateError> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for index in adjacent {
        let tet = &tets[*index];
        if !tet.node_ids.contains(&interior_node_id) {
            return Ok(false);
        }
        for face in tet_node_faces(tet.node_ids) {
            *face_counts.entry(sorted_node_face(face)).or_default() += 1;
        }
    }

    let mut shell_face_count = 0_usize;
    for (face, count) in face_counts {
        if count != 1 {
            continue;
        }
        if face.contains(&interior_node_id) {
            return Ok(false);
        }
        shell_face_count += 1;
        let triangle = [
            *node_points
                .get(&face[0])
                .ok_or(TetCandidateError::MissingSurfaceNode { node_id: face[0] })?,
            *node_points
                .get(&face[1])
                .ok_or(TetCandidateError::MissingSurfaceNode { node_id: face[1] })?,
            *node_points
                .get(&face[2])
                .ok_or(TetCandidateError::MissingSurfaceNode { node_id: face[2] })?,
        ];
        if point_triangle_distance(relocated_point, triangle) <= tolerance.absolute_m {
            return Ok(false);
        }
    }
    Ok(shell_face_count >= 4)
}

fn best_three_tet_edge_reconnection(
    tet_index: usize,
    tets: &[TetCandidate],
    edge_adjacency: &BTreeMap<[u32; 2], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, bool)>, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut best = None::<(Vec<usize>, Vec<TetCandidate>, usize, f64, bool)>;
    for edge in tet_node_edges(tet.node_ids) {
        let Some(adjacent) = edge_adjacency.get(&edge) else {
            continue;
        };
        if adjacent.len() != 3 || !adjacent.contains(&tet_index) {
            continue;
        }
        let original_below_count = count_exact_quality_violations(
            adjacent.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact =
            min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
        let Some(candidates) =
            three_tet_edge_reconnection_candidates(adjacent, edge, tets, node_points, options)?
        else {
            continue;
        };
        let candidate_below_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        let min_exact = min_exact_scaled_jacobian(candidates.iter());
        if !cavity_reconnection_improves_quality(
            candidate_below_count,
            min_exact,
            original_below_count,
            original_min_exact,
        ) {
            continue;
        }
        let quality_gain_only = candidate_below_count == original_below_count;
        if best
            .as_ref()
            .is_none_or(|(_, _, best_below_count, best_min_exact, _)| {
                candidate_below_count < *best_below_count
                    || (candidate_below_count == *best_below_count && min_exact > *best_min_exact)
            })
        {
            best = Some((
                adjacent.clone(),
                candidates,
                candidate_below_count,
                min_exact,
                quality_gain_only,
            ));
        }
    }
    Ok(best.map(|(indices, candidates, _, _, quality_gain_only)| {
        (indices, candidates, quality_gain_only)
    }))
}

fn best_multi_tet_edge_reconnection(
    tet_index: usize,
    tets: &[TetCandidate],
    edge_adjacency: &BTreeMap<[u32; 2], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, bool)>, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut best = None::<(Vec<usize>, Vec<TetCandidate>, usize, f64, bool)>;
    for edge in tet_node_edges(tet.node_ids) {
        let Some(adjacent) = edge_adjacency.get(&edge) else {
            continue;
        };
        if adjacent.len() < 4
            || adjacent.len() > MAX_EDGE_STAR_RECONNECTION_SIZE
            || !adjacent.contains(&tet_index)
        {
            continue;
        }
        let original_below_count = count_exact_quality_violations(
            adjacent.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact =
            min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
        let Some(candidates) =
            multi_tet_edge_reconnection_candidates(adjacent, edge, tets, node_points, options)?
        else {
            continue;
        };
        let candidate_below_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        let min_exact = min_exact_scaled_jacobian(candidates.iter());
        if !cavity_reconnection_improves_quality(
            candidate_below_count,
            min_exact,
            original_below_count,
            original_min_exact,
        ) {
            continue;
        }
        let quality_gain_only = candidate_below_count == original_below_count;
        if best
            .as_ref()
            .is_none_or(|(_, _, best_below_count, best_min_exact, _)| {
                candidate_below_count < *best_below_count
                    || (candidate_below_count == *best_below_count && min_exact > *best_min_exact)
            })
        {
            best = Some((
                adjacent.clone(),
                candidates,
                candidate_below_count,
                min_exact,
                quality_gain_only,
            ));
        }
    }
    Ok(best.map(|(indices, candidates, _, _, quality_gain_only)| {
        (indices, candidates, quality_gain_only)
    }))
}

fn best_componentized_edge_reconnection(
    tet_index: usize,
    tets: &[TetCandidate],
    edge_adjacency: &BTreeMap<[u32; 2], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, bool)>, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut best = None::<(Vec<usize>, Vec<TetCandidate>, usize, f64, bool)>;
    for edge in tet_node_edges(tet.node_ids) {
        let Some(adjacent) = edge_adjacency.get(&edge) else {
            continue;
        };
        if adjacent.len() <= 3 || !adjacent.contains(&tet_index) {
            continue;
        }
        let mut candidate_groups = edge_star_ring_components(adjacent, edge, tets)?;
        candidate_groups.extend(edge_star_simple_cycle_components(
            adjacent, edge, tet_index, tets,
        )?);
        let mut seen_groups = BTreeSet::<Vec<usize>>::new();
        for component in candidate_groups {
            if !seen_groups.insert(component.clone()) {
                continue;
            }
            if !component.contains(&tet_index)
                || component.len() < 3
                || component.len() > MAX_EDGE_STAR_RECONNECTION_SIZE
            {
                continue;
            }
            if component.len() == adjacent.len() {
                continue;
            }
            let original_below_count = count_exact_quality_violations(
                component.iter().map(|index| &tets[*index]),
                options.min_scaled_jacobian,
            );
            let original_min_exact =
                min_exact_scaled_jacobian(component.iter().map(|index| &tets[*index]));
            let candidates = if component.len() == 3 {
                three_tet_edge_reconnection_candidates(
                    &component,
                    edge,
                    tets,
                    node_points,
                    options,
                )?
            } else {
                multi_tet_edge_reconnection_candidates(
                    &component,
                    edge,
                    tets,
                    node_points,
                    options,
                )?
            };
            let Some(candidates) = candidates else {
                continue;
            };
            let candidate_below_count =
                count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
            let min_exact = min_exact_scaled_jacobian(candidates.iter());
            if !cavity_reconnection_improves_quality(
                candidate_below_count,
                min_exact,
                original_below_count,
                original_min_exact,
            ) {
                continue;
            }
            let quality_gain_only = candidate_below_count == original_below_count;
            if best
                .as_ref()
                .is_none_or(|(_, _, best_below_count, best_min_exact, _)| {
                    candidate_below_count < *best_below_count
                        || (candidate_below_count == *best_below_count
                            && min_exact > *best_min_exact)
                })
            {
                best = Some((
                    component,
                    candidates,
                    candidate_below_count,
                    min_exact,
                    quality_gain_only,
                ));
            }
        }
    }
    Ok(best.map(|(indices, candidates, _, _, quality_gain_only)| {
        (indices, candidates, quality_gain_only)
    }))
}

#[cfg(test)]
pub(crate) fn diagnostic_edge_reconnection_rejection_reason(
    tet_index: usize,
    edge: [u32; 2],
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<&'static str, TetCandidateError> {
    if !adjacent.contains(&tet_index) {
        return Ok("target_not_in_edge_star");
    }
    if adjacent.len() < 3 {
        return Ok("too_few_edge_tets");
    }
    if (4..=MAX_EDGE_STAR_RECONNECTION_SIZE).contains(&adjacent.len()) {
        let original_below_count = count_exact_quality_violations(
            adjacent.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact =
            min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
        if let Some(candidates) =
            multi_tet_edge_reconnection_candidates(adjacent, edge, tets, node_points, options)?
        {
            let candidate_below_count =
                count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
            let min_exact = min_exact_scaled_jacobian(candidates.iter());
            if cavity_reconnection_improves_quality(
                candidate_below_count,
                min_exact,
                original_below_count,
                original_min_exact,
            ) {
                return Ok("whole_edge_star_reconnectable");
            }
            return Ok("whole_edge_star_no_improving_reconnection");
        }
    }
    if adjacent.len() == 3 {
        let original_below_count = count_exact_quality_violations(
            adjacent.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact =
            min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
        if let Some(candidates) =
            three_tet_edge_reconnection_candidates(adjacent, edge, tets, node_points, options)?
        {
            let candidate_below_count =
                count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
            let min_exact = min_exact_scaled_jacobian(candidates.iter());
            if cavity_reconnection_improves_quality(
                candidate_below_count,
                min_exact,
                original_below_count,
                original_min_exact,
            ) {
                return Ok("three_tet_edge_star_reconnectable");
            }
            return Ok("three_tet_edge_star_no_improving_reconnection");
        }
    }

    let mut candidate_groups = edge_star_ring_components(adjacent, edge, tets)?;
    candidate_groups.extend(edge_star_simple_cycle_components(
        adjacent, edge, tet_index, tets,
    )?);
    let mut seen_groups = BTreeSet::<Vec<usize>>::new();
    let mut saw_target_group = false;
    let mut saw_too_small_group = false;
    let mut saw_too_large_group = false;
    let mut saw_whole_star_group = false;
    let mut saw_invalid_candidate = false;
    let mut saw_non_improving_candidate = false;
    for component in candidate_groups {
        if !seen_groups.insert(component.clone()) || !component.contains(&tet_index) {
            continue;
        }
        saw_target_group = true;
        if component.len() < 3 {
            saw_too_small_group = true;
            continue;
        }
        if component.len() > MAX_EDGE_STAR_RECONNECTION_SIZE {
            saw_too_large_group = true;
            continue;
        }
        if component.len() == adjacent.len() {
            saw_whole_star_group = true;
            continue;
        }
        let original_below_count = count_exact_quality_violations(
            component.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact =
            min_exact_scaled_jacobian(component.iter().map(|index| &tets[*index]));
        let candidates = if component.len() == 3 {
            three_tet_edge_reconnection_candidates(&component, edge, tets, node_points, options)?
        } else {
            multi_tet_edge_reconnection_candidates(&component, edge, tets, node_points, options)?
        };
        let Some(candidates) = candidates else {
            saw_invalid_candidate = true;
            continue;
        };
        let candidate_below_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        let min_exact = min_exact_scaled_jacobian(candidates.iter());
        if cavity_reconnection_improves_quality(
            candidate_below_count,
            min_exact,
            original_below_count,
            original_min_exact,
        ) {
            return Ok("component_edge_star_reconnectable");
        }
        saw_non_improving_candidate = true;
    }
    if saw_non_improving_candidate {
        Ok("component_edge_star_no_improving_reconnection")
    } else if saw_invalid_candidate {
        Ok("component_edge_star_candidate_invalid")
    } else if saw_too_large_group {
        Ok("component_edge_star_over_reconnection_limit")
    } else if saw_too_small_group {
        Ok("component_edge_star_too_small")
    } else if saw_whole_star_group {
        Ok("component_edge_star_whole_star_only")
    } else if saw_target_group {
        Ok("component_edge_star_no_candidate")
    } else {
        Ok("no_component_edge_star_for_target")
    }
}

#[cfg(test)]
pub(crate) fn diagnostic_node_cavity_reconnection_rejection_reason(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<&'static str, TetCandidateError> {
    let adjacent = connected_bad_tet_cavity_with_node_closure(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
    );
    if adjacent.len() < 3 {
        return Ok("node_cavity_too_small");
    }
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    if adjacent.len() <= face_closure.len() {
        return Ok("node_cavity_no_node_expansion");
    }

    let base = face_closure.into_iter().collect::<BTreeSet<_>>();
    let extra = bounded_node_cavity_extra_indices(
        adjacent
            .iter()
            .copied()
            .filter(|index| !base.contains(index))
            .collect::<Vec<_>>(),
        tets,
    );
    let mut candidate_groups = vec![adjacent.clone()];
    for extra_index in &extra {
        let mut group = base.clone();
        group.insert(*extra_index);
        candidate_groups.push(group.into_iter().collect());
    }
    for left in 0..extra.len() {
        for right in (left + 1)..extra.len() {
            let mut group = base.clone();
            group.insert(extra[left]);
            group.insert(extra[right]);
            candidate_groups.push(group.into_iter().collect());
        }
    }
    diagnostic_cavity_group_reconnection_rejection_reason(
        candidate_groups,
        tets,
        node_points,
        options,
        "node_cavity",
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_bad_cavity_sizes(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    options: TetCandidateOptions,
) -> (usize, usize, usize) {
    (
        one_ring_tet_cavity(tet_index, tets, face_adjacency).len(),
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options).len(),
        connected_bad_tet_cavity_with_node_closure(
            tet_index,
            tets,
            face_adjacency,
            node_adjacency,
            options,
        )
        .len(),
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_small_cavity_reconnection_rejection_reasons(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<(&'static str, &'static str), TetCandidateError> {
    let one_ring = one_ring_tet_cavity(tet_index, tets, face_adjacency);
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    Ok((
        diagnostic_face_cavity_reconnection_rejection_reason(
            &one_ring,
            tets,
            node_points,
            options,
            "one_ring",
        )?,
        diagnostic_face_cavity_reconnection_rejection_reason(
            &face_closure,
            tets,
            node_points,
            options,
            "face_closure",
        )?,
    ))
}

#[cfg(test)]
pub(crate) fn diagnostic_small_cavity_boundary_mismatch_shapes(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<((usize, usize, usize, usize), (usize, usize, usize, usize)), TetCandidateError> {
    let one_ring = one_ring_tet_cavity(tet_index, tets, face_adjacency);
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    Ok((
        diagnostic_face_cavity_boundary_mismatch_shape(&one_ring, tets, node_points, options)?,
        diagnostic_face_cavity_boundary_mismatch_shape(&face_closure, tets, node_points, options)?,
    ))
}

#[cfg(test)]
pub(crate) fn diagnostic_small_cavity_exact_cover_rejection_reasons(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<(&'static str, &'static str), TetCandidateError> {
    let one_ring = one_ring_tet_cavity(tet_index, tets, face_adjacency);
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    Ok((
        diagnostic_face_cavity_exact_cover_rejection_reason(
            &one_ring,
            tets,
            node_points,
            options,
            "one_ring",
        )?,
        diagnostic_face_cavity_exact_cover_rejection_reason(
            &face_closure,
            tets,
            node_points,
            options,
            "face_closure",
        )?,
    ))
}

#[cfg(test)]
pub(crate) fn diagnostic_small_cavity_star_insertion_rejection_reasons(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    split_node_id: u32,
    options: TetCandidateOptions,
) -> Result<(&'static str, &'static str), TetCandidateError> {
    let one_ring = one_ring_tet_cavity(tet_index, tets, face_adjacency);
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    Ok((
        diagnostic_face_cavity_star_insertion_rejection_reason(
            tet_index,
            &one_ring,
            tets,
            node_points,
            split_node_id,
            options,
            "one_ring",
        )?,
        diagnostic_face_cavity_star_insertion_rejection_reason(
            tet_index,
            &face_closure,
            tets,
            node_points,
            split_node_id,
            options,
            "face_closure",
        )?,
    ))
}

#[cfg(test)]
struct BoundarySplitDiagnosticInput<'a> {
    tet_index: usize,
    adjacent: &'a [usize],
    tets: &'a [TetCandidate],
    face_adjacency: &'a BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &'a BTreeMap<u32, [f64; 3]>,
    split_node_id: u32,
    options: TetCandidateOptions,
    prefix: &'static str,
}

#[cfg(test)]
pub(crate) fn diagnostic_small_cavity_boundary_split_rejection_reasons(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    split_node_id: u32,
    options: TetCandidateOptions,
) -> Result<(&'static str, &'static str), TetCandidateError> {
    let one_ring = one_ring_tet_cavity(tet_index, tets, face_adjacency);
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    Ok((
        diagnostic_face_cavity_boundary_split_rejection_reason(BoundarySplitDiagnosticInput {
            tet_index,
            adjacent: &one_ring,
            tets,
            face_adjacency,
            node_points,
            split_node_id,
            options,
            prefix: "one_ring",
        })?,
        diagnostic_face_cavity_boundary_split_rejection_reason(BoundarySplitDiagnosticInput {
            tet_index,
            adjacent: &face_closure,
            tets,
            face_adjacency,
            node_points,
            split_node_id,
            options,
            prefix: "face_closure",
        })?,
    ))
}

#[cfg(test)]
pub(crate) fn diagnostic_small_cavity_missing_face_classes(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<((usize, usize, usize), (usize, usize, usize)), TetCandidateError> {
    let one_ring = one_ring_tet_cavity(tet_index, tets, face_adjacency);
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    Ok((
        diagnostic_face_cavity_missing_face_class(
            &one_ring,
            tets,
            face_adjacency,
            node_points,
            options,
        )?,
        diagnostic_face_cavity_missing_face_class(
            &face_closure,
            tets,
            face_adjacency,
            node_points,
            options,
        )?,
    ))
}

#[cfg(test)]
type MissingFaceTopology = (usize, usize, usize, usize, usize, usize, usize);

#[cfg(test)]
pub(crate) fn diagnostic_small_cavity_missing_face_topology(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<(MissingFaceTopology, MissingFaceTopology), TetCandidateError> {
    let one_ring = one_ring_tet_cavity(tet_index, tets, face_adjacency);
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    Ok((
        diagnostic_face_cavity_missing_face_topology(
            &one_ring,
            tets,
            face_adjacency,
            node_points,
            options,
        )?,
        diagnostic_face_cavity_missing_face_topology(
            &face_closure,
            tets,
            face_adjacency,
            node_points,
            options,
        )?,
    ))
}

#[cfg(test)]
fn diagnostic_face_cavity_missing_face_class(
    adjacent: &[usize],
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<(usize, usize, usize), TetCandidateError> {
    let Some(reference) = adjacent.first().map(|index| &tets[*index]) else {
        return Ok((0, 0, 0));
    };
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for index in adjacent {
        let tet = &tets[*index];
        if tet.component_id != reference.component_id {
            return Ok((0, 0, 0));
        }
        for face in tet_node_faces(tet.node_ids).map(sorted_node_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let boundary_faces = face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect::<BTreeSet<_>>();
    for face in &boundary_faces {
        boundary_nodes.extend(face.iter().copied());
    }
    if boundary_nodes.len() < 4 || boundary_nodes.len() > 16 {
        return Ok((0, 0, 0));
    }
    let points = boundary_nodes
        .iter()
        .map(|node_id| {
            Ok(ConnectivityPoint {
                node_id: *node_id,
                coordinates_m: *node_points
                    .get(node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?,
                is_super: false,
            })
        })
        .collect::<Result<Vec<_>, TetCandidateError>>()?;
    let mut candidates = Vec::<TetCandidate>::new();
    for tet in tetrahedralize_points(&points) {
        let node_ids = tet.vertices.map(|index| points[index].node_id);
        let tet_points = tet.vertices.map(|index| points[index].coordinates_m);
        let Some(candidate) = raw_candidate_tet(
            reference.component_id,
            reference.source_surface_element_id,
            &reference.region_ids,
            node_ids,
            tet_points,
            options,
        ) else {
            return Ok((0, 0, 0));
        };
        candidates.push(candidate);
    }
    let candidate_boundary_faces = boundary_faces_from_tets(&candidates);
    let mut missing_global = 0_usize;
    let mut missing_internal = 0_usize;
    for face in &boundary_faces {
        if candidate_boundary_faces.contains(face) {
            continue;
        }
        if face_adjacency.get(face).map_or(0, Vec::len) == 1 {
            missing_global += 1;
        } else {
            missing_internal += 1;
        }
    }
    let unexpected = candidate_boundary_faces
        .iter()
        .filter(|face| !boundary_faces.contains(*face))
        .count();
    Ok((missing_global, missing_internal, unexpected))
}

#[cfg(test)]
fn diagnostic_face_cavity_missing_face_topology(
    adjacent: &[usize],
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<MissingFaceTopology, TetCandidateError> {
    let Some(reference) = adjacent.first().map(|index| &tets[*index]) else {
        return Ok((0, 0, 0, 0, 0, 0, 0));
    };
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for index in adjacent {
        let tet = &tets[*index];
        if tet.component_id != reference.component_id {
            return Ok((0, 0, 0, 0, 0, 0, 0));
        }
        for face in tet_node_faces(tet.node_ids).map(sorted_node_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let boundary_faces = face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect::<BTreeSet<_>>();
    for face in &boundary_faces {
        boundary_nodes.extend(face.iter().copied());
    }
    if boundary_nodes.len() < 4 || boundary_nodes.len() > 16 {
        return Ok((0, 0, 0, 0, 0, 0, 0));
    }
    let points = boundary_nodes
        .iter()
        .map(|node_id| {
            Ok(ConnectivityPoint {
                node_id: *node_id,
                coordinates_m: *node_points
                    .get(node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?,
                is_super: false,
            })
        })
        .collect::<Result<Vec<_>, TetCandidateError>>()?;
    let mut candidates = Vec::<TetCandidate>::new();
    for tet in tetrahedralize_points(&points) {
        let node_ids = tet.vertices.map(|index| points[index].node_id);
        let tet_points = tet.vertices.map(|index| points[index].coordinates_m);
        let Some(candidate) = raw_candidate_tet(
            reference.component_id,
            reference.source_surface_element_id,
            &reference.region_ids,
            node_ids,
            tet_points,
            options,
        ) else {
            return Ok((0, 0, 0, 0, 0, 0, 0));
        };
        candidates.push(candidate);
    }
    let candidate_boundary_faces = boundary_faces_from_tets(&candidates);
    let missing_faces = boundary_faces
        .iter()
        .filter(|face| !candidate_boundary_faces.contains(*face))
        .copied()
        .collect::<Vec<_>>();
    let unexpected = candidate_boundary_faces
        .iter()
        .filter(|face| !boundary_faces.contains(*face))
        .count();
    Ok(diagnostic_missing_face_topology(
        &missing_faces,
        face_adjacency,
        unexpected,
    ))
}

#[cfg(test)]
fn diagnostic_missing_face_topology(
    missing_faces: &[[u32; 3]],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    unexpected: usize,
) -> MissingFaceTopology {
    let mut missing_global = 0_usize;
    let mut missing_internal = 0_usize;
    let mut node_degrees = BTreeMap::<u32, usize>::new();
    let mut edge_degrees = BTreeMap::<[u32; 2], usize>::new();
    for face in missing_faces {
        if face_adjacency.get(face).map_or(0, Vec::len) == 1 {
            missing_global += 1;
        } else {
            missing_internal += 1;
        }
        for node_id in face {
            *node_degrees.entry(*node_id).or_default() += 1;
        }
        for edge in face_node_edges(*face) {
            *edge_degrees.entry(edge).or_default() += 1;
        }
    }
    let max_node_degree = node_degrees.values().copied().max().unwrap_or(0);
    let max_edge_degree = edge_degrees.values().copied().max().unwrap_or(0);
    let shared_edge_count = edge_degrees.values().filter(|count| **count > 1).count();
    let edge_component_count = missing_face_edge_component_count(missing_faces);
    (
        missing_global,
        missing_internal,
        edge_component_count,
        max_node_degree,
        max_edge_degree,
        shared_edge_count,
        unexpected,
    )
}

#[cfg(test)]
fn missing_face_edge_component_count(missing_faces: &[[u32; 3]]) -> usize {
    let mut remaining = (0..missing_faces.len()).collect::<BTreeSet<_>>();
    let mut component_count = 0_usize;
    while let Some(seed) = remaining.first().copied() {
        remaining.remove(&seed);
        component_count += 1;
        let mut stack = vec![seed];
        while let Some(current) = stack.pop() {
            let current_edges = face_node_edges(missing_faces[current])
                .into_iter()
                .collect::<BTreeSet<_>>();
            let neighbors = remaining
                .iter()
                .copied()
                .filter(|candidate| {
                    face_node_edges(missing_faces[*candidate])
                        .into_iter()
                        .any(|edge| current_edges.contains(&edge))
                })
                .collect::<Vec<_>>();
            for neighbor in neighbors {
                remaining.remove(&neighbor);
                stack.push(neighbor);
            }
        }
    }
    component_count
}

#[cfg(test)]
fn face_node_edges(node_ids: [u32; 3]) -> [[u32; 2]; 3] {
    [
        sorted_node_edge([node_ids[0], node_ids[1]]),
        sorted_node_edge([node_ids[0], node_ids[2]]),
        sorted_node_edge([node_ids[1], node_ids[2]]),
    ]
}

#[cfg(test)]
fn diagnostic_face_cavity_star_insertion_rejection_reason(
    tet_index: usize,
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    split_node_id: u32,
    options: TetCandidateOptions,
    prefix: &'static str,
) -> Result<&'static str, TetCandidateError> {
    if adjacent.len() < 3 {
        return Ok(match prefix {
            "one_ring" => "one_ring_star_insert_too_small",
            _ => "face_closure_star_insert_too_small",
        });
    }
    if adjacent.len() > 8 {
        return Ok(match prefix {
            "one_ring" => "one_ring_star_insert_over_limit",
            _ => "face_closure_star_insert_over_limit",
        });
    }
    let Some((boundary_faces, _, original_volume, reference)) =
        diagnostic_face_cavity_boundary_data(adjacent, tets)
    else {
        return Ok(match prefix {
            "one_ring" => "one_ring_star_insert_component_mismatch",
            _ => "face_closure_star_insert_component_mismatch",
        });
    };
    if boundary_faces.len() < 4 || boundary_faces.len() > 24 {
        return Ok(match prefix {
            "one_ring" => "one_ring_star_insert_boundary_limit",
            _ => "face_closure_star_insert_boundary_limit",
        });
    }
    let original_below_count = count_exact_quality_violations(
        adjacent.iter().map(|index| &tets[*index]),
        options.min_scaled_jacobian,
    );
    let original_min_exact = min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
    let mut saw_invalid = false;
    let mut saw_boundary_mismatch = false;
    let mut saw_volume_mismatch = false;
    let mut saw_no_improvement = false;
    for split_point in diagnostic_star_insertion_points(tet_index, adjacent, tets, node_points)? {
        let Some(candidates) = diagnostic_star_boundary_face_candidates(
            reference,
            &boundary_faces,
            node_points,
            split_node_id,
            split_point,
            options,
        )?
        else {
            saw_invalid = true;
            continue;
        };
        if boundary_faces_from_tets(&candidates) != boundary_faces {
            saw_boundary_mismatch = true;
            continue;
        }
        let candidate_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        if (candidate_volume - original_volume).abs() > original_volume.max(1.0e-18) * 1.0e-9 {
            saw_volume_mismatch = true;
            continue;
        }
        let candidate_below_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        let candidate_min_exact = min_exact_scaled_jacobian(candidates.iter());
        if cavity_reconnection_improves_quality(
            candidate_below_count,
            candidate_min_exact,
            original_below_count,
            original_min_exact,
        ) {
            return Ok(match prefix {
                "one_ring" => "one_ring_star_insert_reconnectable",
                _ => "face_closure_star_insert_reconnectable",
            });
        }
        saw_no_improvement = true;
    }
    Ok(
        match (
            prefix,
            saw_no_improvement,
            saw_volume_mismatch,
            saw_boundary_mismatch,
            saw_invalid,
        ) {
            ("one_ring", true, _, _, _) => "one_ring_star_insert_no_improvement",
            ("one_ring", _, true, _, _) => "one_ring_star_insert_volume_mismatch",
            ("one_ring", _, _, true, _) => "one_ring_star_insert_boundary_mismatch",
            ("one_ring", _, _, _, true) => "one_ring_star_insert_invalid",
            ("one_ring", _, _, _, _) => "one_ring_star_insert_no_candidate",
            (_, true, _, _, _) => "face_closure_star_insert_no_improvement",
            (_, _, true, _, _) => "face_closure_star_insert_volume_mismatch",
            (_, _, _, true, _) => "face_closure_star_insert_boundary_mismatch",
            (_, _, _, _, true) => "face_closure_star_insert_invalid",
            _ => "face_closure_star_insert_no_candidate",
        },
    )
}

#[cfg(test)]
fn diagnostic_star_insertion_points(
    tet_index: usize,
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
) -> Result<Vec<[f64; 3]>, TetCandidateError> {
    let target_centroid = tet_centroid(candidate_tet_points(&tets[tet_index], node_points)?);
    let mut weighted = [0.0; 3];
    let mut total_volume = 0.0;
    for index in adjacent {
        let tet = &tets[*index];
        let centroid = tet_centroid(candidate_tet_points(tet, node_points)?);
        let weight = tet.volume_m3.max(0.0);
        weighted[0] += centroid[0] * weight;
        weighted[1] += centroid[1] * weight;
        weighted[2] += centroid[2] * weight;
        total_volume += weight;
    }
    let cavity_centroid = if total_volume > 0.0 {
        [
            weighted[0] / total_volume,
            weighted[1] / total_volume,
            weighted[2] / total_volume,
        ]
    } else {
        target_centroid
    };
    let mut points = vec![
        target_centroid,
        cavity_centroid,
        [
            0.5 * (target_centroid[0] + cavity_centroid[0]),
            0.5 * (target_centroid[1] + cavity_centroid[1]),
            0.5 * (target_centroid[2] + cavity_centroid[2]),
        ],
    ];
    points.dedup_by(|left, right| distance_squared(*left, *right) <= 1.0e-24);
    Ok(points)
}

#[cfg(test)]
fn diagnostic_star_boundary_face_candidates(
    reference: &TetCandidate,
    boundary_faces: &BTreeSet<[u32; 3]>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    split_node_id: u32,
    split_point: [f64; 3],
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    let mut candidates = Vec::<TetCandidate>::with_capacity(boundary_faces.len());
    for face in boundary_faces {
        let node_ids = [split_node_id, face[0], face[1], face[2]];
        let points = [
            split_point,
            *node_points
                .get(&face[0])
                .ok_or(TetCandidateError::MissingSurfaceNode { node_id: face[0] })?,
            *node_points
                .get(&face[1])
                .ok_or(TetCandidateError::MissingSurfaceNode { node_id: face[1] })?,
            *node_points
                .get(&face[2])
                .ok_or(TetCandidateError::MissingSurfaceNode { node_id: face[2] })?,
        ];
        let Some(candidate) = raw_candidate_tet(
            reference.component_id,
            reference.source_surface_element_id,
            &reference.region_ids,
            node_ids,
            points,
            options,
        ) else {
            return Ok(None);
        };
        candidates.push(candidate);
    }
    Ok(Some(candidates))
}

#[cfg(test)]
fn diagnostic_face_cavity_boundary_split_rejection_reason(
    input: BoundarySplitDiagnosticInput<'_>,
) -> Result<&'static str, TetCandidateError> {
    let BoundarySplitDiagnosticInput {
        tet_index,
        adjacent,
        tets,
        face_adjacency,
        node_points,
        split_node_id,
        options,
        prefix,
    } = input;
    if adjacent.len() < 3 {
        return Ok(match prefix {
            "one_ring" => "one_ring_boundary_split_too_small",
            _ => "face_closure_boundary_split_too_small",
        });
    }
    if adjacent.len() > 12 {
        return Ok(match prefix {
            "one_ring" => "one_ring_boundary_split_over_limit",
            _ => "face_closure_boundary_split_over_limit",
        });
    }
    let Some(cavity) =
        constrained_cavity_from_selected_tets_with_anchor_trim(tets, adjacent, tet_index, vec![])
            .map_err(|_| TetCandidateError::InvalidOptions)?
    else {
        return Ok(match prefix {
            "one_ring" => "one_ring_boundary_split_invalid_cavity",
            _ => "face_closure_boundary_split_invalid_cavity",
        });
    };
    if cavity.boundary_faces.len() < 4 || cavity.boundary_faces.len() > 24 {
        return Ok(match prefix {
            "one_ring" => "one_ring_boundary_split_boundary_limit",
            _ => "face_closure_boundary_split_boundary_limit",
        });
    }
    let index_by_tet_id = tets
        .iter()
        .enumerate()
        .map(|(index, tet)| (tet.tet_id, index))
        .collect::<BTreeMap<_, _>>();
    let selected_indices = cavity
        .removed_tet_ids
        .iter()
        .map(|tet_id| {
            index_by_tet_id
                .get(tet_id)
                .copied()
                .ok_or(TetCandidateError::InvalidOptions)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let Some(reference) = selected_indices.first().map(|index| &tets[*index]) else {
        return Ok(match prefix {
            "one_ring" => "one_ring_boundary_split_invalid_cavity",
            _ => "face_closure_boundary_split_invalid_cavity",
        });
    };
    let original_below_count = count_exact_quality_violations(
        selected_indices.iter().map(|index| &tets[*index]),
        options.min_scaled_jacobian,
    );
    let original_min_exact =
        min_exact_scaled_jacobian(selected_indices.iter().map(|index| &tets[*index]));

    let mut saw_global_face = false;
    let mut saw_invalid_split = false;
    let mut saw_no_refill = false;
    let mut saw_no_improvement = false;
    let mut saw_boundary_refill_error = false;
    for face in &cavity.boundary_faces {
        let face_key = sorted_node_face(face.node_ids);
        if face_adjacency.get(&face_key).map_or(0, Vec::len) != 1 {
            continue;
        }
        saw_global_face = true;
        let [Some(first), Some(second), Some(third)] = face
            .node_ids
            .map(|node_id| node_points.get(&node_id).copied())
        else {
            saw_invalid_split = true;
            continue;
        };
        let split_point = triangle_centroid([first, second, third]);
        let mut split_cavity = cavity.clone();
        split_cavity.boundary_faces =
            match crate::constrained_cavity::split_constrained_cavity_boundary_faces(
                &cavity.boundary_faces,
                face.node_ids,
                split_node_id,
            ) {
                Ok(faces) => faces,
                Err(_) => {
                    saw_invalid_split = true;
                    continue;
                }
            };
        let boundary_node_ids = split_cavity
            .boundary_faces
            .iter()
            .flat_map(|boundary_face| boundary_face.node_ids)
            .collect::<BTreeSet<_>>();
        let mut boundary_nodes =
            Vec::<ConstrainedCavityNode>::with_capacity(boundary_node_ids.len());
        for node_id in boundary_node_ids {
            let coordinates_m = if node_id == split_node_id {
                split_point
            } else {
                *node_points
                    .get(&node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id })?
            };
            boundary_nodes.push(ConstrainedCavityNode {
                node_id,
                coordinates_m,
            });
        }
        let evaluation = evaluate_constrained_cavity_refill_candidates(
            &split_cavity,
            &boundary_nodes,
            &[],
            ConstrainedCavityRefillOptions {
                min_volume_m3: options.min_volume_m3,
                max_aspect_ratio: options.max_aspect_ratio,
                min_scaled_jacobian: options.min_scaled_jacobian,
                volume_relative_tolerance: 1.0e-9,
                min_protected_node_distance_m: 0.0,
            },
        );
        let Some(refill) = (match evaluation {
            Ok(evaluation) => evaluation.refill,
            Err(_) => {
                saw_boundary_refill_error = true;
                continue;
            }
        }) else {
            saw_no_refill = true;
            continue;
        };
        let candidates = refill
            .tets
            .into_iter()
            .map(|tet| TetCandidate {
                tet_id: 0,
                component_id: reference.component_id,
                node_ids: tet.node_ids,
                source_surface_element_id: reference.source_surface_element_id,
                region_ids: reference.region_ids.clone(),
                volume_m3: tet.volume_m3,
                aspect_ratio: tet.aspect_ratio,
                exact_scaled_jacobian: tet.exact_scaled_jacobian,
            })
            .collect::<Vec<_>>();
        let candidate_below_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        let candidate_min_exact = min_exact_scaled_jacobian(candidates.iter());
        if cavity_reconnection_improves_quality(
            candidate_below_count,
            candidate_min_exact,
            original_below_count,
            original_min_exact,
        ) {
            return Ok(match prefix {
                "one_ring" => "one_ring_boundary_split_reconnectable",
                _ => "face_closure_boundary_split_reconnectable",
            });
        }
        saw_no_improvement = true;
    }
    Ok(
        match (
            prefix,
            saw_no_improvement,
            saw_no_refill,
            saw_boundary_refill_error,
            saw_invalid_split,
            saw_global_face,
        ) {
            ("one_ring", true, _, _, _, _) => "one_ring_boundary_split_no_improvement",
            ("one_ring", _, true, _, _, _) => "one_ring_boundary_split_no_refill",
            ("one_ring", _, _, true, _, _) => "one_ring_boundary_split_refill_error",
            ("one_ring", _, _, _, true, _) => "one_ring_boundary_split_invalid",
            ("one_ring", _, _, _, _, false) => "one_ring_boundary_split_no_global_face",
            ("one_ring", _, _, _, _, _) => "one_ring_boundary_split_no_candidate",
            (_, true, _, _, _, _) => "face_closure_boundary_split_no_improvement",
            (_, _, true, _, _, _) => "face_closure_boundary_split_no_refill",
            (_, _, _, true, _, _) => "face_closure_boundary_split_refill_error",
            (_, _, _, _, true, _) => "face_closure_boundary_split_invalid",
            (_, _, _, _, _, false) => "face_closure_boundary_split_no_global_face",
            _ => "face_closure_boundary_split_no_candidate",
        },
    )
}

#[cfg(test)]
fn diagnostic_face_cavity_exact_cover_rejection_reason(
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
    prefix: &'static str,
) -> Result<&'static str, TetCandidateError> {
    if adjacent.len() < 3 {
        return Ok(match prefix {
            "one_ring" => "one_ring_exact_cover_too_small",
            _ => "face_closure_exact_cover_too_small",
        });
    }
    let Some((boundary_faces, boundary_nodes, original_volume, reference)) =
        diagnostic_face_cavity_boundary_data(adjacent, tets)
    else {
        return Ok(match prefix {
            "one_ring" => "one_ring_exact_cover_component_mismatch",
            _ => "face_closure_exact_cover_component_mismatch",
        });
    };
    if boundary_nodes.len() < 4 || boundary_nodes.len() > 8 || boundary_faces.len() > 16 {
        return Ok(match prefix {
            "one_ring" => "one_ring_exact_cover_over_limit",
            _ => "face_closure_exact_cover_over_limit",
        });
    }
    let Some(candidates) = exhaustive_boundary_cover_candidates(
        reference,
        &boundary_faces,
        &boundary_nodes,
        node_points,
        options,
    )?
    else {
        return Ok(match prefix {
            "one_ring" => "one_ring_exact_cover_no_raw_candidates",
            _ => "face_closure_exact_cover_no_raw_candidates",
        });
    };
    let Some(cover) = select_boundary_exact_cover(&candidates, &boundary_faces, original_volume)
    else {
        return Ok(match prefix {
            "one_ring" => "one_ring_exact_cover_not_found",
            _ => "face_closure_exact_cover_not_found",
        });
    };
    let original_below_count = count_exact_quality_violations(
        adjacent.iter().map(|index| &tets[*index]),
        options.min_scaled_jacobian,
    );
    let original_min_exact = min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
    let candidate_below_count =
        count_exact_quality_violations(cover.iter(), options.min_scaled_jacobian);
    let candidate_min_exact = min_exact_scaled_jacobian(cover.iter());
    if cavity_reconnection_improves_quality(
        candidate_below_count,
        candidate_min_exact,
        original_below_count,
        original_min_exact,
    ) {
        Ok(match prefix {
            "one_ring" => "one_ring_exact_cover_reconnectable",
            _ => "face_closure_exact_cover_reconnectable",
        })
    } else {
        Ok(match prefix {
            "one_ring" => "one_ring_exact_cover_no_improvement",
            _ => "face_closure_exact_cover_no_improvement",
        })
    }
}

#[cfg(test)]
fn diagnostic_face_cavity_boundary_data<'a>(
    adjacent: &[usize],
    tets: &'a [TetCandidate],
) -> Option<(BTreeSet<[u32; 3]>, BTreeSet<u32>, f64, &'a TetCandidate)> {
    let reference = adjacent.first().map(|index| &tets[*index])?;
    let mut original_volume = 0.0_f64;
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for index in adjacent {
        let tet = &tets[*index];
        if tet.component_id != reference.component_id {
            return None;
        }
        original_volume += tet.volume_m3;
        for face in tet_node_faces(tet.node_ids).map(sorted_node_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let boundary_faces = face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect::<BTreeSet<_>>();
    for face in &boundary_faces {
        boundary_nodes.extend(face.iter().copied());
    }
    Some((boundary_faces, boundary_nodes, original_volume, reference))
}

#[cfg(test)]
fn exhaustive_boundary_cover_candidates(
    reference: &TetCandidate,
    boundary_faces: &BTreeSet<[u32; 3]>,
    boundary_nodes: &BTreeSet<u32>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    let node_ids = boundary_nodes.iter().copied().collect::<Vec<_>>();
    let mut candidates = Vec::<TetCandidate>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tet_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tet_node_faces(tet_node_ids)
                        .map(sorted_node_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = [
                        *node_points.get(&tet_node_ids[0]).ok_or(
                            TetCandidateError::MissingSurfaceNode {
                                node_id: tet_node_ids[0],
                            },
                        )?,
                        *node_points.get(&tet_node_ids[1]).ok_or(
                            TetCandidateError::MissingSurfaceNode {
                                node_id: tet_node_ids[1],
                            },
                        )?,
                        *node_points.get(&tet_node_ids[2]).ok_or(
                            TetCandidateError::MissingSurfaceNode {
                                node_id: tet_node_ids[2],
                            },
                        )?,
                        *node_points.get(&tet_node_ids[3]).ok_or(
                            TetCandidateError::MissingSurfaceNode {
                                node_id: tet_node_ids[3],
                            },
                        )?,
                    ];
                    let Some(candidate) = raw_candidate_tet(
                        reference.component_id,
                        reference.source_surface_element_id,
                        &reference.region_ids,
                        tet_node_ids,
                        points,
                        options,
                    ) else {
                        continue;
                    };
                    candidates.push(candidate);
                }
            }
        }
    }
    Ok((!candidates.is_empty() && candidates.len() <= 80).then_some(candidates))
}

#[cfg(test)]
fn select_boundary_exact_cover(
    candidates: &[TetCandidate],
    boundary_faces: &BTreeSet<[u32; 3]>,
    original_volume: f64,
) -> Option<Vec<TetCandidate>> {
    let candidate_faces = candidates
        .iter()
        .map(|candidate| tet_node_faces(candidate.node_ids).map(sorted_node_face))
        .collect::<Vec<_>>();
    let mut selected = Vec::<usize>::new();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let tolerance = original_volume.max(1.0e-18) * 1.0e-9;
    let mut attempts = 0_usize;
    let selected_indices = search_boundary_exact_cover(
        candidates,
        &candidate_faces,
        boundary_faces,
        original_volume,
        tolerance,
        0.0,
        &mut face_counts,
        &mut selected,
        &mut attempts,
    )?;
    Some(
        selected_indices
            .into_iter()
            .map(|index| candidates[index].clone())
            .collect(),
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn search_boundary_exact_cover(
    candidates: &[TetCandidate],
    candidate_faces: &[[[u32; 3]; 4]],
    boundary_faces: &BTreeSet<[u32; 3]>,
    target_volume: f64,
    tolerance: f64,
    current_volume: f64,
    face_counts: &mut BTreeMap<[u32; 3], usize>,
    selected: &mut Vec<usize>,
    attempts: &mut usize,
) -> Option<Vec<usize>> {
    *attempts += 1;
    if *attempts > 5_000 || current_volume > target_volume + tolerance {
        return None;
    }
    let Some(target_face) = boundary_faces
        .iter()
        .find(|face| face_counts.get(*face).copied().unwrap_or(0) == 0)
        .copied()
    else {
        let boundary_ok = boundary_faces
            .iter()
            .all(|face| face_counts.get(face).copied().unwrap_or(0) == 1);
        let interior_ok = face_counts
            .iter()
            .all(|(face, count)| boundary_faces.contains(face) || *count == 2);
        if boundary_ok && interior_ok && (current_volume - target_volume).abs() <= tolerance {
            return Some(selected.clone());
        }
        return None;
    };
    for candidate_index in 0..candidates.len() {
        if selected.contains(&candidate_index)
            || !candidate_faces[candidate_index].contains(&target_face)
            || !candidate_faces[candidate_index].iter().all(|face| {
                let count = face_counts.get(face).copied().unwrap_or(0);
                if boundary_faces.contains(face) {
                    count == 0
                } else {
                    count < 2
                }
            })
        {
            continue;
        }
        for face in candidate_faces[candidate_index] {
            *face_counts.entry(face).or_default() += 1;
        }
        selected.push(candidate_index);
        if let Some(result) = search_boundary_exact_cover(
            candidates,
            candidate_faces,
            boundary_faces,
            target_volume,
            tolerance,
            current_volume + candidates[candidate_index].volume_m3,
            face_counts,
            selected,
            attempts,
        ) {
            return Some(result);
        }
        selected.pop();
        for face in candidate_faces[candidate_index] {
            if let Some(count) = face_counts.get_mut(&face) {
                *count -= 1;
                if *count == 0 {
                    face_counts.remove(&face);
                }
            }
        }
    }
    None
}

#[cfg(test)]
fn diagnostic_face_cavity_boundary_mismatch_shape(
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<(usize, usize, usize, usize), TetCandidateError> {
    let Some(reference) = adjacent.first().map(|index| &tets[*index]) else {
        return Ok((0, 0, 0, 0));
    };
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for index in adjacent {
        let tet = &tets[*index];
        if tet.component_id != reference.component_id {
            return Ok((0, 0, 0, 0));
        }
        for face in tet_node_faces(tet.node_ids).map(sorted_node_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let boundary_faces = face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect::<BTreeSet<_>>();
    for face in &boundary_faces {
        boundary_nodes.extend(face.iter().copied());
    }
    if boundary_nodes.len() < 4 || boundary_nodes.len() > 16 {
        return Ok((boundary_nodes.len(), boundary_faces.len(), 0, 0));
    }
    let points = boundary_nodes
        .iter()
        .map(|node_id| {
            Ok(ConnectivityPoint {
                node_id: *node_id,
                coordinates_m: *node_points
                    .get(node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?,
                is_super: false,
            })
        })
        .collect::<Result<Vec<_>, TetCandidateError>>()?;
    let mut candidates = Vec::<TetCandidate>::new();
    for tet in tetrahedralize_points(&points) {
        let node_ids = tet.vertices.map(|index| points[index].node_id);
        let tet_points = tet.vertices.map(|index| points[index].coordinates_m);
        let Some(candidate) = raw_candidate_tet(
            reference.component_id,
            reference.source_surface_element_id,
            &reference.region_ids,
            node_ids,
            tet_points,
            options,
        ) else {
            return Ok((boundary_nodes.len(), boundary_faces.len(), 0, 0));
        };
        candidates.push(candidate);
    }
    let candidate_boundary_faces = boundary_faces_from_tets(&candidates);
    let missing_count = boundary_faces
        .iter()
        .filter(|face| !candidate_boundary_faces.contains(*face))
        .count();
    let unexpected_count = candidate_boundary_faces
        .iter()
        .filter(|face| !boundary_faces.contains(*face))
        .count();
    Ok((
        boundary_nodes.len(),
        boundary_faces.len(),
        missing_count,
        unexpected_count,
    ))
}

#[cfg(test)]
fn diagnostic_face_cavity_reconnection_rejection_reason(
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
    prefix: &'static str,
) -> Result<&'static str, TetCandidateError> {
    if adjacent.len() < 3 {
        return Ok(match prefix {
            "one_ring" => "one_ring_too_small",
            _ => "face_closure_too_small",
        });
    }
    if adjacent.len() > 16 {
        return Ok(match prefix {
            "one_ring" => "one_ring_over_reconnection_limit",
            _ => "face_closure_over_reconnection_limit",
        });
    }
    let original_below_count = count_exact_quality_violations(
        adjacent.iter().map(|index| &tets[*index]),
        options.min_scaled_jacobian,
    );
    let original_min_exact = min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
    let candidate_result = diagnostic_face_neighbor_cavity_reconnection_candidates(
        adjacent,
        tets,
        node_points,
        options,
    )?;
    let Some((candidates, candidate_reason)) = candidate_result else {
        return Ok(match prefix {
            "one_ring" => "one_ring_empty_candidate",
            _ => "face_closure_empty_candidate",
        });
    };
    if let Some(reason) = candidate_reason {
        return Ok(match (prefix, reason) {
            ("one_ring", "component_mismatch") => "one_ring_component_mismatch",
            ("one_ring", "too_few_boundary_faces") => "one_ring_too_few_boundary_faces",
            ("one_ring", "boundary_node_count") => "one_ring_boundary_node_count",
            ("one_ring", "raw_candidate_rejected") => "one_ring_raw_candidate_rejected",
            ("one_ring", "empty_tetrahedralization") => "one_ring_empty_tetrahedralization",
            ("one_ring", "boundary_face_mismatch") => "one_ring_boundary_face_mismatch",
            ("one_ring", "boundary_face_mismatch_constrained_available") => {
                "one_ring_boundary_face_mismatch_constrained_available"
            }
            ("one_ring", "boundary_face_mismatch_constrained_no_refill") => {
                "one_ring_boundary_face_mismatch_constrained_no_refill"
            }
            ("one_ring", "boundary_face_mismatch_constrained_invalid") => {
                "one_ring_boundary_face_mismatch_constrained_invalid"
            }
            ("one_ring", "boundary_face_mismatch_constrained_non_manifold_boundary_edge") => {
                "one_ring_boundary_face_mismatch_constrained_non_manifold_boundary_edge"
            }
            ("one_ring", "boundary_face_mismatch_constrained_non_manifold_trim_available") => {
                "one_ring_boundary_face_mismatch_constrained_non_manifold_trim_available"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_reconnectable") => {
                "one_ring_boundary_face_mismatch_constrained_trim_reconnectable"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_no_candidate") => {
                "one_ring_boundary_face_mismatch_constrained_trim_no_candidate"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_component_mismatch") => {
                "one_ring_boundary_face_mismatch_constrained_trim_component_mismatch"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_too_few_boundary_faces") => {
                "one_ring_boundary_face_mismatch_constrained_trim_too_few_boundary_faces"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_boundary_node_count") => {
                "one_ring_boundary_face_mismatch_constrained_trim_boundary_node_count"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_boundary_face_mismatch") => {
                "one_ring_boundary_face_mismatch_constrained_trim_boundary_face_mismatch"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_raw_candidate_rejected") => {
                "one_ring_boundary_face_mismatch_constrained_trim_raw_candidate_rejected"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_empty_tetrahedralization") => {
                "one_ring_boundary_face_mismatch_constrained_trim_empty_tetrahedralization"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_volume_mismatch") => {
                "one_ring_boundary_face_mismatch_constrained_trim_volume_mismatch"
            }
            ("one_ring", "boundary_face_mismatch_constrained_trim_no_improvement") => {
                "one_ring_boundary_face_mismatch_constrained_trim_no_improvement"
            }
            ("one_ring", "boundary_face_mismatch_constrained_non_manifold_trim_not_found") => {
                "one_ring_boundary_face_mismatch_constrained_non_manifold_trim_not_found"
            }
            ("one_ring", "boundary_face_mismatch_constrained_duplicate_boundary_face") => {
                "one_ring_boundary_face_mismatch_constrained_duplicate_boundary_face"
            }
            ("one_ring", "boundary_face_mismatch_constrained_degenerate_boundary_face") => {
                "one_ring_boundary_face_mismatch_constrained_degenerate_boundary_face"
            }
            ("one_ring", "boundary_face_mismatch_constrained_too_few_boundary_faces") => {
                "one_ring_boundary_face_mismatch_constrained_too_few_boundary_faces"
            }
            ("one_ring", "volume_mismatch") => "one_ring_volume_mismatch",
            (_, "component_mismatch") => "face_closure_component_mismatch",
            (_, "too_few_boundary_faces") => "face_closure_too_few_boundary_faces",
            (_, "boundary_node_count") => "face_closure_boundary_node_count",
            (_, "raw_candidate_rejected") => "face_closure_raw_candidate_rejected",
            (_, "empty_tetrahedralization") => "face_closure_empty_tetrahedralization",
            (_, "boundary_face_mismatch") => "face_closure_boundary_face_mismatch",
            (_, "boundary_face_mismatch_constrained_available") => {
                "face_closure_boundary_face_mismatch_constrained_available"
            }
            (_, "boundary_face_mismatch_constrained_no_refill") => {
                "face_closure_boundary_face_mismatch_constrained_no_refill"
            }
            (_, "boundary_face_mismatch_constrained_invalid") => {
                "face_closure_boundary_face_mismatch_constrained_invalid"
            }
            (_, "boundary_face_mismatch_constrained_non_manifold_boundary_edge") => {
                "face_closure_boundary_face_mismatch_constrained_non_manifold_boundary_edge"
            }
            (_, "boundary_face_mismatch_constrained_non_manifold_trim_available") => {
                "face_closure_boundary_face_mismatch_constrained_non_manifold_trim_available"
            }
            (_, "boundary_face_mismatch_constrained_trim_reconnectable") => {
                "face_closure_boundary_face_mismatch_constrained_trim_reconnectable"
            }
            (_, "boundary_face_mismatch_constrained_trim_no_candidate") => {
                "face_closure_boundary_face_mismatch_constrained_trim_no_candidate"
            }
            (_, "boundary_face_mismatch_constrained_trim_component_mismatch") => {
                "face_closure_boundary_face_mismatch_constrained_trim_component_mismatch"
            }
            (_, "boundary_face_mismatch_constrained_trim_too_few_boundary_faces") => {
                "face_closure_boundary_face_mismatch_constrained_trim_too_few_boundary_faces"
            }
            (_, "boundary_face_mismatch_constrained_trim_boundary_node_count") => {
                "face_closure_boundary_face_mismatch_constrained_trim_boundary_node_count"
            }
            (_, "boundary_face_mismatch_constrained_trim_boundary_face_mismatch") => {
                "face_closure_boundary_face_mismatch_constrained_trim_boundary_face_mismatch"
            }
            (_, "boundary_face_mismatch_constrained_trim_raw_candidate_rejected") => {
                "face_closure_boundary_face_mismatch_constrained_trim_raw_candidate_rejected"
            }
            (_, "boundary_face_mismatch_constrained_trim_empty_tetrahedralization") => {
                "face_closure_boundary_face_mismatch_constrained_trim_empty_tetrahedralization"
            }
            (_, "boundary_face_mismatch_constrained_trim_volume_mismatch") => {
                "face_closure_boundary_face_mismatch_constrained_trim_volume_mismatch"
            }
            (_, "boundary_face_mismatch_constrained_trim_no_improvement") => {
                "face_closure_boundary_face_mismatch_constrained_trim_no_improvement"
            }
            (_, "boundary_face_mismatch_constrained_non_manifold_trim_not_found") => {
                "face_closure_boundary_face_mismatch_constrained_non_manifold_trim_not_found"
            }
            (_, "boundary_face_mismatch_constrained_duplicate_boundary_face") => {
                "face_closure_boundary_face_mismatch_constrained_duplicate_boundary_face"
            }
            (_, "boundary_face_mismatch_constrained_degenerate_boundary_face") => {
                "face_closure_boundary_face_mismatch_constrained_degenerate_boundary_face"
            }
            (_, "boundary_face_mismatch_constrained_too_few_boundary_faces") => {
                "face_closure_boundary_face_mismatch_constrained_too_few_boundary_faces"
            }
            (_, "volume_mismatch") => "face_closure_volume_mismatch",
            _ => "face_closure_candidate_rejected",
        });
    }
    let candidate_below_count =
        count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
    let min_exact = min_exact_scaled_jacobian(candidates.iter());
    if cavity_reconnection_improves_quality(
        candidate_below_count,
        min_exact,
        original_below_count,
        original_min_exact,
    ) {
        return Ok(match prefix {
            "one_ring" => "one_ring_reconnectable",
            _ => "face_closure_reconnectable",
        });
    }
    Ok(match prefix {
        "one_ring" => "one_ring_no_improving_reconnection",
        _ => "face_closure_no_improving_reconnection",
    })
}

#[cfg(test)]
fn diagnostic_face_neighbor_cavity_reconnection_candidates(
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<TetCandidate>, Option<&'static str>)>, TetCandidateError> {
    let Some(reference) = adjacent.first().map(|index| &tets[*index]) else {
        return Ok(None);
    };
    let mut original_volume = 0.0_f64;
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for index in adjacent {
        let tet = &tets[*index];
        if tet.component_id != reference.component_id {
            return Ok(Some((Vec::new(), Some("component_mismatch"))));
        }
        original_volume += tet.volume_m3;
        for face in tet_node_faces(tet.node_ids).map(sorted_node_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let boundary_faces = face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect::<BTreeSet<_>>();
    if boundary_faces.len() < 4 {
        return Ok(Some((Vec::new(), Some("too_few_boundary_faces"))));
    }
    for face in &boundary_faces {
        boundary_nodes.extend(face.iter().copied());
    }
    if boundary_nodes.len() < 4 || boundary_nodes.len() > 16 {
        return Ok(Some((Vec::new(), Some("boundary_node_count"))));
    }
    let points = boundary_nodes
        .iter()
        .map(|node_id| {
            Ok(ConnectivityPoint {
                node_id: *node_id,
                coordinates_m: *node_points
                    .get(node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?,
                is_super: false,
            })
        })
        .collect::<Result<Vec<_>, TetCandidateError>>()?;
    let mut candidates = Vec::<TetCandidate>::new();
    for tet in tetrahedralize_points(&points) {
        let node_ids = tet.vertices.map(|index| points[index].node_id);
        let tet_points = tet.vertices.map(|index| points[index].coordinates_m);
        let Some(candidate) = raw_candidate_tet(
            reference.component_id,
            reference.source_surface_element_id,
            &reference.region_ids,
            node_ids,
            tet_points,
            options,
        ) else {
            return Ok(Some((Vec::new(), Some("raw_candidate_rejected"))));
        };
        candidates.push(candidate);
    }
    if candidates.is_empty() {
        return Ok(Some((Vec::new(), Some("empty_tetrahedralization"))));
    }
    let candidate_boundary_faces = boundary_faces_from_tets(&candidates);
    if candidate_boundary_faces != boundary_faces {
        let reason = diagnostic_constrained_boundary_mismatch_reason(
            adjacent,
            tets,
            &boundary_faces,
            &boundary_nodes,
            original_volume,
            node_points,
            options,
        )?;
        return Ok(Some((candidates, Some(reason))));
    }
    let candidate_volume = candidates
        .iter()
        .map(|candidate| candidate.volume_m3)
        .sum::<f64>();
    if (candidate_volume - original_volume).abs() > original_volume.max(1.0e-18) * 1.0e-9 {
        return Ok(Some((candidates, Some("volume_mismatch"))));
    }
    Ok(Some((candidates, None)))
}

#[cfg(test)]
fn diagnostic_constrained_boundary_mismatch_reason(
    adjacent: &[usize],
    tets: &[TetCandidate],
    boundary_faces: &BTreeSet<[u32; 3]>,
    boundary_nodes: &BTreeSet<u32>,
    original_volume: f64,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<&'static str, TetCandidateError> {
    let cavity = crate::constrained_cavity::ConstrainedCavity {
        removed_tet_ids: adjacent.iter().map(|index| *index as u32).collect(),
        boundary_faces: boundary_faces
            .iter()
            .map(
                |face| crate::constrained_cavity::ConstrainedCavityBoundaryFace {
                    node_ids: *face,
                    source_face_id: None,
                    source_edge_ids: [None, None, None],
                    region_ids: Vec::new(),
                },
            )
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: original_volume,
    };
    let boundary_nodes = boundary_nodes
        .iter()
        .map(|node_id| {
            Ok(ConstrainedCavityNode {
                node_id: *node_id,
                coordinates_m: *node_points
                    .get(node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?,
            })
        })
        .collect::<Result<Vec<_>, TetCandidateError>>()?;
    match evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &boundary_nodes,
        &[],
        ConstrainedCavityRefillOptions {
            min_volume_m3: options.min_volume_m3,
            max_aspect_ratio: options.max_aspect_ratio,
            min_scaled_jacobian: options.min_scaled_jacobian,
            volume_relative_tolerance: 1.0e-9,
            min_protected_node_distance_m: 0.0,
        },
    ) {
        Ok(evaluation) if evaluation.refill.is_some() => {
            Ok("boundary_face_mismatch_constrained_available")
        }
        Ok(_) => Ok("boundary_face_mismatch_constrained_no_refill"),
        Err(err) => Ok(diagnostic_constrained_boundary_mismatch_error_reason(
            &err,
            adjacent,
            tets,
            node_points,
            options,
        )),
    }
}

#[cfg(test)]
fn diagnostic_constrained_boundary_mismatch_error_reason(
    err: &crate::constrained_cavity::ConstrainedCavityRefillError,
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> &'static str {
    match err {
        crate::constrained_cavity::ConstrainedCavityRefillError::InvalidOptions => {
            "boundary_face_mismatch_constrained_invalid_options"
        }
        crate::constrained_cavity::ConstrainedCavityRefillError::Validation(err) => match err {
            ConstrainedCavityValidationError::NonManifoldBoundaryEdge { .. } => {
                diagnostic_constrained_boundary_mismatch_trim_reason(
                    adjacent,
                    tets,
                    node_points,
                    options,
                )
            }
            _ => diagnostic_constrained_boundary_mismatch_validation_reason(err),
        },
        crate::constrained_cavity::ConstrainedCavityRefillError::MissingBoundaryNode { .. } => {
            "boundary_face_mismatch_constrained_missing_boundary_node"
        }
        crate::constrained_cavity::ConstrainedCavityRefillError::DuplicateInteriorNode { .. } => {
            "boundary_face_mismatch_constrained_duplicate_interior_node"
        }
        crate::constrained_cavity::ConstrainedCavityRefillError::InteriorNodeReusesBoundaryNode {
            ..
        } => "boundary_face_mismatch_constrained_interior_node_reuses_boundary_node",
        crate::constrained_cavity::ConstrainedCavityRefillError::InteriorPointOutsideCavity {
            ..
        } => "boundary_face_mismatch_constrained_interior_point_outside_cavity",
        crate::constrained_cavity::ConstrainedCavityRefillError::NoValidCandidate { .. } => {
            "boundary_face_mismatch_constrained_no_valid_candidate"
        }
    }
}

#[cfg(test)]
fn diagnostic_constrained_boundary_mismatch_trim_reason(
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> &'static str {
    let Some(anchor_tet_index) = adjacent.first().copied() else {
        return "boundary_face_mismatch_constrained_non_manifold_boundary_edge";
    };
    let cavity = match constrained_cavity_from_selected_tets_with_anchor_trim(
        tets,
        adjacent,
        anchor_tet_index,
        Vec::new(),
    ) {
        Ok(Some(cavity)) => cavity,
        Ok(None) => return "boundary_face_mismatch_constrained_non_manifold_trim_not_found",
        Err(_) => return "boundary_face_mismatch_constrained_non_manifold_boundary_edge",
    };
    let index_by_tet_id = tets
        .iter()
        .enumerate()
        .map(|(index, tet)| (tet.tet_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut trimmed = Vec::<usize>::new();
    for tet_id in cavity.removed_tet_ids {
        let Some(index) = index_by_tet_id.get(&tet_id).copied() else {
            return "boundary_face_mismatch_constrained_non_manifold_trim_available";
        };
        trimmed.push(index);
    }
    if trimmed.len() < 3 || trimmed.len() > 24 {
        return "boundary_face_mismatch_constrained_non_manifold_trim_available";
    }
    let original_below_count = count_exact_quality_violations(
        trimmed.iter().map(|index| &tets[*index]),
        options.min_scaled_jacobian,
    );
    let original_min_exact = min_exact_scaled_jacobian(trimmed.iter().map(|index| &tets[*index]));
    let (candidates, rejection) = match diagnostic_face_neighbor_cavity_reconnection_candidates(
        &trimmed,
        tets,
        node_points,
        options,
    ) {
        Ok(Some(result)) => result,
        Ok(None) => return "boundary_face_mismatch_constrained_trim_no_candidate",
        Err(_) => return "boundary_face_mismatch_constrained_trim_no_candidate",
    };
    if let Some(reason) = rejection {
        return diagnostic_constrained_trim_candidate_reason(reason);
    }
    let candidate_below_count =
        count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
    let candidate_min_exact = min_exact_scaled_jacobian(candidates.iter());
    if cavity_reconnection_improves_quality(
        candidate_below_count,
        candidate_min_exact,
        original_below_count,
        original_min_exact,
    ) {
        "boundary_face_mismatch_constrained_trim_reconnectable"
    } else {
        "boundary_face_mismatch_constrained_trim_no_improvement"
    }
}

#[cfg(test)]
fn diagnostic_constrained_trim_candidate_reason(reason: &'static str) -> &'static str {
    match reason {
        "component_mismatch" => "boundary_face_mismatch_constrained_trim_component_mismatch",
        "too_few_boundary_faces" => {
            "boundary_face_mismatch_constrained_trim_too_few_boundary_faces"
        }
        "boundary_node_count" => "boundary_face_mismatch_constrained_trim_boundary_node_count",
        "raw_candidate_rejected" => {
            "boundary_face_mismatch_constrained_trim_raw_candidate_rejected"
        }
        "empty_tetrahedralization" => {
            "boundary_face_mismatch_constrained_trim_empty_tetrahedralization"
        }
        "boundary_face_mismatch"
        | "boundary_face_mismatch_constrained_available"
        | "boundary_face_mismatch_constrained_no_refill"
        | "boundary_face_mismatch_constrained_invalid"
        | "boundary_face_mismatch_constrained_non_manifold_boundary_edge"
        | "boundary_face_mismatch_constrained_non_manifold_trim_available"
        | "boundary_face_mismatch_constrained_non_manifold_trim_not_found"
        | "boundary_face_mismatch_constrained_trim_no_candidate"
        | "boundary_face_mismatch_constrained_trim_no_improvement" => {
            "boundary_face_mismatch_constrained_trim_boundary_face_mismatch"
        }
        "volume_mismatch" => "boundary_face_mismatch_constrained_trim_volume_mismatch",
        _ => "boundary_face_mismatch_constrained_trim_no_candidate",
    }
}

#[cfg(test)]
fn diagnostic_constrained_boundary_mismatch_validation_reason(
    err: &ConstrainedCavityValidationError,
) -> &'static str {
    match err {
        ConstrainedCavityValidationError::EmptyRemovedTetSet => {
            "boundary_face_mismatch_constrained_empty_removed_tet_set"
        }
        ConstrainedCavityValidationError::InvalidTargetVolume { .. } => {
            "boundary_face_mismatch_constrained_invalid_target_volume"
        }
        ConstrainedCavityValidationError::TooFewBoundaryFaces { .. } => {
            "boundary_face_mismatch_constrained_too_few_boundary_faces"
        }
        ConstrainedCavityValidationError::DegenerateBoundaryFace { .. } => {
            "boundary_face_mismatch_constrained_degenerate_boundary_face"
        }
        ConstrainedCavityValidationError::DuplicateBoundaryFace { .. } => {
            "boundary_face_mismatch_constrained_duplicate_boundary_face"
        }
        ConstrainedCavityValidationError::NonManifoldBoundaryEdge { .. } => {
            "boundary_face_mismatch_constrained_non_manifold_boundary_edge"
        }
        ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary { .. } => {
            "boundary_face_mismatch_constrained_protected_node_outside_boundary"
        }
        ConstrainedCavityValidationError::InvalidRefillVolume { .. } => {
            "boundary_face_mismatch_constrained_invalid_refill_volume"
        }
        ConstrainedCavityValidationError::BoundaryFaceCountMismatch { .. } => {
            "boundary_face_mismatch_constrained_boundary_face_count_mismatch"
        }
        ConstrainedCavityValidationError::MissingBoundaryFace { .. } => {
            "boundary_face_mismatch_constrained_missing_boundary_face"
        }
        ConstrainedCavityValidationError::UnexpectedBoundaryFace { .. } => {
            "boundary_face_mismatch_constrained_unexpected_boundary_face"
        }
        ConstrainedCavityValidationError::BoundarySourceFaceMismatch { .. } => {
            "boundary_face_mismatch_constrained_boundary_source_face_mismatch"
        }
        ConstrainedCavityValidationError::BoundarySourceEdgeMismatch { .. } => {
            "boundary_face_mismatch_constrained_boundary_source_edge_mismatch"
        }
        ConstrainedCavityValidationError::BoundaryRegionMismatch { .. } => {
            "boundary_face_mismatch_constrained_boundary_region_mismatch"
        }
    }
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_cavity_reconnection_rejection_reason(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<&'static str, TetCandidateError> {
    let adjacent = boundary_adjacent_bad_tet_cavity_with_node_closure(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
    );
    let expanded = boundary_adjacent_bad_tet_cavity_with_node_closure_layers(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
        2,
    );
    let mut candidate_groups = vec![adjacent];
    if candidate_groups
        .first()
        .is_some_and(|group| group.as_slice() != expanded.as_slice())
    {
        candidate_groups.push(expanded);
    }
    diagnostic_cavity_group_reconnection_rejection_reason(
        candidate_groups,
        tets,
        node_points,
        options,
        "boundary_cavity",
    )
}

#[cfg(test)]
fn diagnostic_cavity_group_reconnection_rejection_reason(
    candidate_groups: Vec<Vec<usize>>,
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
    prefix: &'static str,
) -> Result<&'static str, TetCandidateError> {
    let mut saw_group = false;
    let mut saw_oversized_group = false;
    let mut saw_star_sized_group = false;
    let mut saw_invalid_candidate = false;
    let mut invalid_candidate_reason = None::<&'static str>;
    let mut saw_non_improving_candidate = false;
    for group in candidate_groups {
        if group.len() < 3 {
            continue;
        }
        saw_group = true;
        if group.len() > 40 {
            saw_oversized_group = true;
            continue;
        }
        if group.len() > 24 {
            saw_star_sized_group = true;
        }
        let original_below_count = count_exact_quality_violations(
            group.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact = min_exact_scaled_jacobian(group.iter().map(|index| &tets[*index]));
        let Some((candidates, candidate_reason)) =
            diagnostic_face_neighbor_cavity_reconnection_candidates(
                &group,
                tets,
                node_points,
                options,
            )?
        else {
            saw_invalid_candidate = true;
            invalid_candidate_reason.get_or_insert("empty_candidate");
            continue;
        };
        if let Some(reason) = candidate_reason {
            saw_invalid_candidate = true;
            invalid_candidate_reason.get_or_insert(reason);
            continue;
        }
        let candidate_below_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        let min_exact = min_exact_scaled_jacobian(candidates.iter());
        if cavity_reconnection_improves_quality(
            candidate_below_count,
            min_exact,
            original_below_count,
            original_min_exact,
        ) {
            return Ok(match prefix {
                "boundary_cavity" => "boundary_cavity_reconnectable",
                _ => "node_cavity_reconnectable",
            });
        }
        saw_non_improving_candidate = true;
    }
    Ok(
        match (
            prefix,
            saw_non_improving_candidate,
            saw_invalid_candidate,
            saw_oversized_group,
            saw_star_sized_group,
            saw_group,
        ) {
            ("boundary_cavity", true, _, _, _, _) => "boundary_cavity_no_improving_reconnection",
            ("boundary_cavity", _, true, _, _, _) => diagnostic_cavity_candidate_invalid_bucket(
                "boundary_cavity",
                invalid_candidate_reason,
            ),
            ("boundary_cavity", _, _, true, _, _) => "boundary_cavity_over_star_insertion_limit",
            ("boundary_cavity", _, _, _, true, _) => "boundary_cavity_over_reconnection_limit",
            ("boundary_cavity", _, _, _, _, true) => "boundary_cavity_no_candidate",
            ("boundary_cavity", _, _, _, _, _) => "boundary_cavity_too_small",
            (_, true, _, _, _, _) => "node_cavity_no_improving_reconnection",
            (_, _, true, _, _, _) => {
                diagnostic_cavity_candidate_invalid_bucket("node_cavity", invalid_candidate_reason)
            }
            (_, _, _, true, _, _) => "node_cavity_over_star_insertion_limit",
            (_, _, _, _, true, _) => "node_cavity_over_reconnection_limit",
            (_, _, _, _, _, true) => "node_cavity_no_candidate",
            _ => "node_cavity_too_small",
        },
    )
}

#[cfg(test)]
fn diagnostic_cavity_candidate_invalid_bucket(
    prefix: &'static str,
    reason: Option<&'static str>,
) -> &'static str {
    match (prefix, reason) {
        ("boundary_cavity", Some("component_mismatch")) => "boundary_cavity_component_mismatch",
        ("boundary_cavity", Some("too_few_boundary_faces")) => {
            "boundary_cavity_too_few_boundary_faces"
        }
        ("boundary_cavity", Some("boundary_node_count")) => "boundary_cavity_boundary_node_count",
        ("boundary_cavity", Some("raw_candidate_rejected")) => {
            "boundary_cavity_raw_candidate_rejected"
        }
        ("boundary_cavity", Some("empty_tetrahedralization")) => {
            "boundary_cavity_empty_tetrahedralization"
        }
        ("boundary_cavity", Some("boundary_face_mismatch")) => {
            "boundary_cavity_boundary_face_mismatch"
        }
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_available")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_available"
        }
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_no_refill")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_no_refill"
        }
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_invalid")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_invalid"
        }
        (
            "boundary_cavity",
            Some("boundary_face_mismatch_constrained_non_manifold_boundary_edge"),
        ) => "boundary_cavity_boundary_face_mismatch_constrained_non_manifold_boundary_edge",
        (
            "boundary_cavity",
            Some("boundary_face_mismatch_constrained_non_manifold_trim_available"),
        ) => "boundary_cavity_boundary_face_mismatch_constrained_non_manifold_trim_available",
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_trim_reconnectable")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_trim_reconnectable"
        }
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_trim_no_candidate")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_trim_no_candidate"
        }
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_trim_component_mismatch")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_trim_component_mismatch"
        }
        (
            "boundary_cavity",
            Some("boundary_face_mismatch_constrained_trim_too_few_boundary_faces"),
        ) => "boundary_cavity_boundary_face_mismatch_constrained_trim_too_few_boundary_faces",
        (
            "boundary_cavity",
            Some("boundary_face_mismatch_constrained_trim_boundary_node_count"),
        ) => "boundary_cavity_boundary_face_mismatch_constrained_trim_boundary_node_count",
        (
            "boundary_cavity",
            Some("boundary_face_mismatch_constrained_trim_boundary_face_mismatch"),
        ) => "boundary_cavity_boundary_face_mismatch_constrained_trim_boundary_face_mismatch",
        (
            "boundary_cavity",
            Some("boundary_face_mismatch_constrained_trim_raw_candidate_rejected"),
        ) => "boundary_cavity_boundary_face_mismatch_constrained_trim_raw_candidate_rejected",
        (
            "boundary_cavity",
            Some("boundary_face_mismatch_constrained_trim_empty_tetrahedralization"),
        ) => "boundary_cavity_boundary_face_mismatch_constrained_trim_empty_tetrahedralization",
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_trim_volume_mismatch")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_trim_volume_mismatch"
        }
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_trim_no_improvement")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_trim_no_improvement"
        }
        (
            "boundary_cavity",
            Some("boundary_face_mismatch_constrained_non_manifold_trim_not_found"),
        ) => "boundary_cavity_boundary_face_mismatch_constrained_non_manifold_trim_not_found",
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_duplicate_boundary_face")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_duplicate_boundary_face"
        }
        (
            "boundary_cavity",
            Some("boundary_face_mismatch_constrained_degenerate_boundary_face"),
        ) => "boundary_cavity_boundary_face_mismatch_constrained_degenerate_boundary_face",
        ("boundary_cavity", Some("boundary_face_mismatch_constrained_too_few_boundary_faces")) => {
            "boundary_cavity_boundary_face_mismatch_constrained_too_few_boundary_faces"
        }
        ("boundary_cavity", Some("volume_mismatch")) => "boundary_cavity_volume_mismatch",
        ("boundary_cavity", Some("empty_candidate")) => "boundary_cavity_empty_candidate",
        ("boundary_cavity", _) => "boundary_cavity_candidate_invalid",
        (_, Some("component_mismatch")) => "node_cavity_component_mismatch",
        (_, Some("too_few_boundary_faces")) => "node_cavity_too_few_boundary_faces",
        (_, Some("boundary_node_count")) => "node_cavity_boundary_node_count",
        (_, Some("raw_candidate_rejected")) => "node_cavity_raw_candidate_rejected",
        (_, Some("empty_tetrahedralization")) => "node_cavity_empty_tetrahedralization",
        (_, Some("boundary_face_mismatch")) => "node_cavity_boundary_face_mismatch",
        (_, Some("boundary_face_mismatch_constrained_available")) => {
            "node_cavity_boundary_face_mismatch_constrained_available"
        }
        (_, Some("boundary_face_mismatch_constrained_no_refill")) => {
            "node_cavity_boundary_face_mismatch_constrained_no_refill"
        }
        (_, Some("boundary_face_mismatch_constrained_invalid")) => {
            "node_cavity_boundary_face_mismatch_constrained_invalid"
        }
        (_, Some("boundary_face_mismatch_constrained_non_manifold_boundary_edge")) => {
            "node_cavity_boundary_face_mismatch_constrained_non_manifold_boundary_edge"
        }
        (_, Some("boundary_face_mismatch_constrained_non_manifold_trim_available")) => {
            "node_cavity_boundary_face_mismatch_constrained_non_manifold_trim_available"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_reconnectable")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_reconnectable"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_no_candidate")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_no_candidate"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_component_mismatch")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_component_mismatch"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_too_few_boundary_faces")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_too_few_boundary_faces"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_boundary_node_count")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_boundary_node_count"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_boundary_face_mismatch")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_boundary_face_mismatch"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_raw_candidate_rejected")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_raw_candidate_rejected"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_empty_tetrahedralization")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_empty_tetrahedralization"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_volume_mismatch")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_volume_mismatch"
        }
        (_, Some("boundary_face_mismatch_constrained_trim_no_improvement")) => {
            "node_cavity_boundary_face_mismatch_constrained_trim_no_improvement"
        }
        (_, Some("boundary_face_mismatch_constrained_non_manifold_trim_not_found")) => {
            "node_cavity_boundary_face_mismatch_constrained_non_manifold_trim_not_found"
        }
        (_, Some("boundary_face_mismatch_constrained_duplicate_boundary_face")) => {
            "node_cavity_boundary_face_mismatch_constrained_duplicate_boundary_face"
        }
        (_, Some("boundary_face_mismatch_constrained_degenerate_boundary_face")) => {
            "node_cavity_boundary_face_mismatch_constrained_degenerate_boundary_face"
        }
        (_, Some("boundary_face_mismatch_constrained_too_few_boundary_faces")) => {
            "node_cavity_boundary_face_mismatch_constrained_too_few_boundary_faces"
        }
        (_, Some("volume_mismatch")) => "node_cavity_volume_mismatch",
        (_, Some("empty_candidate")) => "node_cavity_empty_candidate",
        _ => "node_cavity_candidate_invalid",
    }
}

fn bounded_node_cavity_extra_indices(mut extra: Vec<usize>, tets: &[TetCandidate]) -> Vec<usize> {
    extra.sort_by(|left, right| {
        tets[*left]
            .exact_scaled_jacobian
            .total_cmp(&tets[*right].exact_scaled_jacobian)
            .then_with(|| left.cmp(right))
    });
    extra.truncate(MAX_NODE_CAVITY_EXTRA_GROUP_CANDIDATES);
    extra
}

fn edge_star_ring_components(
    adjacent: &[usize],
    edge: [u32; 2],
    tets: &[TetCandidate],
) -> Result<Vec<Vec<usize>>, TetCandidateError> {
    let mut node_to_tets = BTreeMap::<u32, Vec<usize>>::new();
    for index in adjacent {
        let tet = tets.get(*index).ok_or(TetCandidateError::InvalidOptions)?;
        if !tet.node_ids.contains(&edge[0]) || !tet.node_ids.contains(&edge[1]) {
            return Ok(Vec::new());
        }
        for node_id in tet
            .node_ids
            .into_iter()
            .filter(|node_id| !edge.contains(node_id))
        {
            node_to_tets.entry(node_id).or_default().push(*index);
        }
    }
    let adjacent_set = adjacent.iter().copied().collect::<BTreeSet<_>>();
    let mut graph = BTreeMap::<usize, BTreeSet<usize>>::new();
    for index in adjacent {
        graph.entry(*index).or_default();
    }
    for owners in node_to_tets.values() {
        for left in owners {
            for right in owners {
                if left != right && adjacent_set.contains(left) && adjacent_set.contains(right) {
                    graph.entry(*left).or_default().insert(*right);
                }
            }
        }
    }
    let mut components = Vec::<Vec<usize>>::new();
    let mut visited = BTreeSet::<usize>::new();
    for start in adjacent {
        if !visited.insert(*start) {
            continue;
        }
        let mut pending = vec![*start];
        let mut component = Vec::<usize>::new();
        while let Some(index) = pending.pop() {
            component.push(index);
            if let Some(neighbors) = graph.get(&index) {
                for neighbor in neighbors {
                    if visited.insert(*neighbor) {
                        pending.push(*neighbor);
                    }
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort();
    Ok(components)
}

fn edge_star_simple_cycle_components(
    adjacent: &[usize],
    edge: [u32; 2],
    tet_index: usize,
    tets: &[TetCandidate],
) -> Result<Vec<Vec<usize>>, TetCandidateError> {
    let target_opposite_edge = edge_star_opposite_edge(tets, tet_index, edge)?;
    let mut edge_to_tets = BTreeMap::<[u32; 2], Vec<usize>>::new();
    let mut graph = BTreeMap::<u32, BTreeSet<u32>>::new();
    for index in adjacent {
        let opposite_edge = edge_star_opposite_edge(tets, *index, edge)?;
        edge_to_tets.entry(opposite_edge).or_default().push(*index);
        graph
            .entry(opposite_edge[0])
            .or_default()
            .insert(opposite_edge[1]);
        graph
            .entry(opposite_edge[1])
            .or_default()
            .insert(opposite_edge[0]);
    }
    if !edge_to_tets
        .get(&target_opposite_edge)
        .is_some_and(|indices| indices.contains(&tet_index))
    {
        return Ok(Vec::new());
    }
    remove_graph_edge(&mut graph, target_opposite_edge);

    let mut paths = Vec::<Vec<u32>>::new();
    let mut path = vec![target_opposite_edge[1]];
    collect_simple_paths_between_ring_nodes(
        target_opposite_edge[1],
        target_opposite_edge[0],
        &graph,
        &mut path,
        &mut paths,
        MAX_EDGE_STAR_RECONNECTION_SIZE.saturating_sub(1),
    );

    let mut components = Vec::<Vec<usize>>::new();
    for path in paths {
        if path.len() < 3 {
            continue;
        }
        let mut component = vec![tet_index];
        let mut valid = true;
        for pair in path.windows(2) {
            let ring_edge = sorted_node_edge([pair[0], pair[1]]);
            let Some(indices) = edge_to_tets.get(&ring_edge) else {
                valid = false;
                break;
            };
            if indices.len() != 1 {
                valid = false;
                break;
            }
            component.push(indices[0]);
        }
        if !valid {
            continue;
        }
        component.sort_unstable();
        component.dedup();
        if component.len() == path.len() && component.contains(&tet_index) {
            components.push(component);
        }
    }
    components.sort();
    components.dedup();
    Ok(components)
}

fn edge_star_opposite_edge(
    tets: &[TetCandidate],
    tet_index: usize,
    edge: [u32; 2],
) -> Result<[u32; 2], TetCandidateError> {
    let tet = tets
        .get(tet_index)
        .ok_or(TetCandidateError::InvalidOptions)?;
    if !tet.node_ids.contains(&edge[0]) || !tet.node_ids.contains(&edge[1]) {
        return Err(TetCandidateError::InvalidOptions);
    }
    let opposite = tet
        .node_ids
        .into_iter()
        .filter(|node_id| !edge.contains(node_id))
        .collect::<Vec<_>>();
    if opposite.len() != 2 {
        return Err(TetCandidateError::InvalidOptions);
    }
    Ok(sorted_node_edge([opposite[0], opposite[1]]))
}

fn remove_graph_edge(graph: &mut BTreeMap<u32, BTreeSet<u32>>, edge: [u32; 2]) {
    if let Some(neighbors) = graph.get_mut(&edge[0]) {
        neighbors.remove(&edge[1]);
    }
    if let Some(neighbors) = graph.get_mut(&edge[1]) {
        neighbors.remove(&edge[0]);
    }
}

fn collect_simple_paths_between_ring_nodes(
    current: u32,
    target: u32,
    graph: &BTreeMap<u32, BTreeSet<u32>>,
    path: &mut Vec<u32>,
    paths: &mut Vec<Vec<u32>>,
    max_edges: usize,
) {
    if path.len().saturating_sub(1) > max_edges {
        return;
    }
    if current == target {
        paths.push(path.clone());
        return;
    }
    let Some(neighbors) = graph.get(&current) else {
        return;
    };
    for neighbor in neighbors {
        if path.contains(neighbor) {
            continue;
        }
        path.push(*neighbor);
        collect_simple_paths_between_ring_nodes(*neighbor, target, graph, path, paths, max_edges);
        path.pop();
    }
}

fn multi_tet_edge_reconnection_candidates(
    adjacent: &[usize],
    edge: [u32; 2],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    let reference = &tets[adjacent[0]];
    let mut original_volume = 0.0_f64;
    let mut ring_edges = BTreeSet::<[u32; 2]>::new();
    let mut ring_nodes = BTreeSet::<u32>::new();
    for index in adjacent {
        let tet = &tets[*index];
        original_volume += tet.volume_m3;
        if tet.component_id != reference.component_id
            || !tet.node_ids.contains(&edge[0])
            || !tet.node_ids.contains(&edge[1])
        {
            return Ok(None);
        }
        let opposite = tet
            .node_ids
            .into_iter()
            .filter(|node_id| !edge.contains(node_id))
            .collect::<Vec<_>>();
        if opposite.len() != 2 {
            return Ok(None);
        }
        ring_nodes.insert(opposite[0]);
        ring_nodes.insert(opposite[1]);
        ring_edges.insert(sorted_node_edge([opposite[0], opposite[1]]));
    }
    if ring_nodes.len() != adjacent.len() || ring_edges.len() != adjacent.len() {
        return Ok(None);
    }
    let ring = order_ring_cycle(&ring_nodes, &ring_edges)?;
    if ring.len() != adjacent.len() {
        return Ok(None);
    }

    let mut best = None::<(Vec<TetCandidate>, usize, f64)>;
    for root_index in 0..ring.len() {
        let mut candidates = Vec::<TetCandidate>::with_capacity((ring.len() - 2) * 2);
        for offset in 1..(ring.len() - 1) {
            let tri = [
                ring[root_index],
                ring[(root_index + offset) % ring.len()],
                ring[(root_index + offset + 1) % ring.len()],
            ];
            for node_ids in [
                [edge[0], tri[0], tri[1], tri[2]],
                [edge[1], tri[0], tri[2], tri[1]],
            ] {
                let points = [
                    *node_points.get(&node_ids[0]).ok_or(
                        TetCandidateError::MissingSurfaceNode {
                            node_id: node_ids[0],
                        },
                    )?,
                    *node_points.get(&node_ids[1]).ok_or(
                        TetCandidateError::MissingSurfaceNode {
                            node_id: node_ids[1],
                        },
                    )?,
                    *node_points.get(&node_ids[2]).ok_or(
                        TetCandidateError::MissingSurfaceNode {
                            node_id: node_ids[2],
                        },
                    )?,
                    *node_points.get(&node_ids[3]).ok_or(
                        TetCandidateError::MissingSurfaceNode {
                            node_id: node_ids[3],
                        },
                    )?,
                ];
                let Some(candidate) = raw_candidate_tet(
                    reference.component_id,
                    reference.source_surface_element_id,
                    &reference.region_ids,
                    node_ids,
                    points,
                    options,
                ) else {
                    candidates.clear();
                    break;
                };
                candidates.push(candidate);
            }
            if candidates.is_empty() {
                break;
            }
        }
        if candidates.is_empty() {
            continue;
        }
        let candidate_volume = candidates
            .iter()
            .map(|candidate| candidate.volume_m3)
            .sum::<f64>();
        if (candidate_volume - original_volume).abs() > original_volume.max(1.0e-18) * 1.0e-9 {
            continue;
        }
        let below_count = candidates
            .iter()
            .filter(|candidate| candidate.exact_scaled_jacobian < options.min_scaled_jacobian)
            .count();
        let min_exact = candidates
            .iter()
            .map(|candidate| candidate.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best.as_ref().is_none_or(|(_, best_below, best_min)| {
            below_count < *best_below || (below_count == *best_below && min_exact > *best_min)
        }) {
            best = Some((candidates, below_count, min_exact));
        }
    }
    Ok(best.map(|(candidates, _, _)| candidates))
}

fn order_ring_cycle(
    ring_nodes: &BTreeSet<u32>,
    ring_edges: &BTreeSet<[u32; 2]>,
) -> Result<Vec<u32>, TetCandidateError> {
    let mut adjacency = BTreeMap::<u32, Vec<u32>>::new();
    for edge in ring_edges {
        adjacency.entry(edge[0]).or_default().push(edge[1]);
        adjacency.entry(edge[1]).or_default().push(edge[0]);
    }
    if ring_nodes
        .iter()
        .any(|node_id| adjacency.get(node_id).map_or(0, Vec::len) != 2)
    {
        return Ok(Vec::new());
    }
    let start = *ring_nodes.iter().next().unwrap_or(&0);
    let mut ordered = vec![start];
    let mut previous = None::<u32>;
    let mut current = start;
    while ordered.len() < ring_nodes.len() {
        let neighbors = adjacency
            .get(&current)
            .ok_or(TetCandidateError::MissingSurfaceNode { node_id: current })?;
        let Some(next) = neighbors
            .iter()
            .copied()
            .find(|neighbor| Some(*neighbor) != previous && !ordered.contains(neighbor))
        else {
            return Ok(Vec::new());
        };
        previous = Some(current);
        current = next;
        ordered.push(current);
    }
    let closes_cycle = adjacency
        .get(&current)
        .is_some_and(|neighbors| neighbors.contains(&start));
    if closes_cycle {
        Ok(ordered)
    } else {
        Ok(Vec::new())
    }
}

fn three_tet_edge_reconnection_candidates(
    adjacent: &[usize],
    edge: [u32; 2],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    let mut ring = BTreeSet::<u32>::new();
    let mut original_volume = 0.0_f64;
    let reference = &tets[adjacent[0]];
    for index in adjacent {
        let tet = &tets[*index];
        original_volume += tet.volume_m3;
        if !tet.node_ids.contains(&edge[0]) || !tet.node_ids.contains(&edge[1]) {
            return Ok(None);
        }
        for node_id in tet.node_ids {
            if !edge.contains(&node_id) {
                ring.insert(node_id);
            }
        }
    }
    if ring.len() != 3 {
        return Ok(None);
    }
    let ring = ring.into_iter().collect::<Vec<_>>();
    let candidate_node_ids = [
        [edge[0], ring[0], ring[1], ring[2]],
        [edge[1], ring[0], ring[2], ring[1]],
    ];
    let mut candidates = Vec::<TetCandidate>::with_capacity(2);
    for node_ids in candidate_node_ids {
        let points = [
            *node_points
                .get(&node_ids[0])
                .ok_or(TetCandidateError::MissingSurfaceNode {
                    node_id: node_ids[0],
                })?,
            *node_points
                .get(&node_ids[1])
                .ok_or(TetCandidateError::MissingSurfaceNode {
                    node_id: node_ids[1],
                })?,
            *node_points
                .get(&node_ids[2])
                .ok_or(TetCandidateError::MissingSurfaceNode {
                    node_id: node_ids[2],
                })?,
            *node_points
                .get(&node_ids[3])
                .ok_or(TetCandidateError::MissingSurfaceNode {
                    node_id: node_ids[3],
                })?,
        ];
        let Some(candidate) = raw_candidate_tet(
            reference.component_id,
            reference.source_surface_element_id,
            &reference.region_ids,
            node_ids,
            points,
            options,
        ) else {
            return Ok(None);
        };
        candidates.push(candidate);
    }
    let candidate_volume = candidates
        .iter()
        .map(|candidate| candidate.volume_m3)
        .sum::<f64>();
    if (candidate_volume - original_volume).abs() > original_volume.max(1.0e-18) * 1.0e-9 {
        return Ok(None);
    }
    Ok(Some(candidates))
}

fn best_two_tet_reconnection(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(usize, Vec<TetCandidate>, bool)>, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut best = None::<(usize, Vec<TetCandidate>, usize, f64, bool)>;
    for shared_face in tet_node_faces(tet.node_ids).map(sorted_node_face) {
        let Some(adjacent) = face_adjacency.get(&shared_face) else {
            continue;
        };
        if adjacent.len() != 2 {
            continue;
        }
        let neighbor_index = if adjacent[0] == tet_index {
            adjacent[1]
        } else if adjacent[1] == tet_index {
            adjacent[0]
        } else {
            continue;
        };
        let neighbor = &tets[neighbor_index];
        let original_below_count =
            usize::from(tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                + usize::from(neighbor.exact_scaled_jacobian < options.min_scaled_jacobian);
        let original_min_exact = tet
            .exact_scaled_jacobian
            .min(neighbor.exact_scaled_jacobian);
        let Some(candidates) =
            two_tet_reconnection_candidates(tet, neighbor, shared_face, node_points, options)?
        else {
            continue;
        };
        let candidate_below_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        let min_exact = min_exact_scaled_jacobian(candidates.iter());
        if !cavity_reconnection_improves_quality(
            candidate_below_count,
            min_exact,
            original_below_count,
            original_min_exact,
        ) {
            continue;
        }
        let quality_gain_only = candidate_below_count == original_below_count;
        if best
            .as_ref()
            .is_none_or(|(_, _, best_below_count, best_min_exact, _)| {
                candidate_below_count < *best_below_count
                    || (candidate_below_count == *best_below_count && min_exact > *best_min_exact)
            })
        {
            best = Some((
                neighbor_index,
                candidates,
                candidate_below_count,
                min_exact,
                quality_gain_only,
            ));
        }
    }
    Ok(
        best.map(|(neighbor_index, candidates, _, _, quality_gain_only)| {
            (neighbor_index, candidates, quality_gain_only)
        }),
    )
}

fn best_face_neighbor_cavity_reconnection(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, bool)>, TetCandidateError> {
    let tet = &tets[tet_index];
    let mut neighbor_indices = BTreeSet::<usize>::from([tet_index]);
    for face in tet_node_faces(tet.node_ids).map(sorted_node_face) {
        if let Some(adjacent) = face_adjacency.get(&face) {
            neighbor_indices.extend(adjacent.iter().copied());
        }
    }
    if neighbor_indices.len() < 3 || neighbor_indices.len() > 10 {
        return Ok(None);
    }
    let adjacent = neighbor_indices.into_iter().collect::<Vec<_>>();
    let original_below_count = count_exact_quality_violations(
        adjacent.iter().map(|index| &tets[*index]),
        options.min_scaled_jacobian,
    );
    let original_min_exact = min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
    let Some(candidates) =
        face_neighbor_cavity_reconnection_candidates(&adjacent, tets, node_points, options)?
    else {
        return Ok(None);
    };
    let candidate_below_count =
        count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
    let min_exact = min_exact_scaled_jacobian(candidates.iter());
    if !cavity_reconnection_improves_quality(
        candidate_below_count,
        min_exact,
        original_below_count,
        original_min_exact,
    ) {
        return Ok(None);
    }
    let quality_gain_only = candidate_below_count == original_below_count;
    Ok(Some((adjacent, candidates, quality_gain_only)))
}

fn best_connected_bad_cavity_reconnection(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, bool)>, TetCandidateError> {
    let adjacent =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    if adjacent.len() < 4 || adjacent.len() > 16 {
        return Ok(None);
    }
    let one_ring_count = one_ring_tet_cavity(tet_index, tets, face_adjacency).len();
    if adjacent.len() <= one_ring_count {
        return Ok(None);
    }
    let original_below_count = count_exact_quality_violations(
        adjacent.iter().map(|index| &tets[*index]),
        options.min_scaled_jacobian,
    );
    let original_min_exact = min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
    let Some(candidates) =
        face_neighbor_cavity_reconnection_candidates(&adjacent, tets, node_points, options)?
    else {
        return Ok(None);
    };
    let candidate_below_count =
        count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
    let min_exact = min_exact_scaled_jacobian(candidates.iter());
    if !cavity_reconnection_improves_quality(
        candidate_below_count,
        min_exact,
        original_below_count,
        original_min_exact,
    ) {
        return Ok(None);
    }
    let quality_gain_only = candidate_below_count == original_below_count;
    Ok(Some((adjacent, candidates, quality_gain_only)))
}

fn best_expanded_connected_bad_cavity_reconnection(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, bool)>, TetCandidateError> {
    let adjacent = connected_bad_tet_cavity_with_face_closure_layers(
        tet_index,
        tets,
        face_adjacency,
        options,
        2,
    );
    if adjacent.len() < 5 || adjacent.len() > 16 {
        return Ok(None);
    }
    let one_layer_count =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options).len();
    if adjacent.len() <= one_layer_count {
        return Ok(None);
    }
    let original_below_count = count_exact_quality_violations(
        adjacent.iter().map(|index| &tets[*index]),
        options.min_scaled_jacobian,
    );
    let original_min_exact = min_exact_scaled_jacobian(adjacent.iter().map(|index| &tets[*index]));
    let Some(candidates) =
        face_neighbor_cavity_reconnection_candidates(&adjacent, tets, node_points, options)?
    else {
        return Ok(None);
    };
    let candidate_below_count =
        count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
    let min_exact = min_exact_scaled_jacobian(candidates.iter());
    if !cavity_reconnection_improves_quality(
        candidate_below_count,
        min_exact,
        original_below_count,
        original_min_exact,
    ) {
        return Ok(None);
    }
    let quality_gain_only = candidate_below_count == original_below_count;
    Ok(Some((adjacent, candidates, quality_gain_only)))
}

fn best_boundary_adjacent_cavity_reconnection(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, bool)>, TetCandidateError> {
    let adjacent = boundary_adjacent_bad_tet_cavity_with_node_closure(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
    );
    let expanded = boundary_adjacent_bad_tet_cavity_with_node_closure_layers(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
        2,
    );
    let mut candidate_groups = vec![adjacent];
    if candidate_groups
        .first()
        .is_some_and(|group| group.as_slice() != expanded.as_slice())
    {
        candidate_groups.push(expanded);
    }

    let mut best = None::<(Vec<usize>, Vec<TetCandidate>, usize, f64, bool)>;
    for group in candidate_groups {
        if group.len() < 4 || group.len() > 24 {
            continue;
        }
        let original_below_count = count_exact_quality_violations(
            group.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact = min_exact_scaled_jacobian(group.iter().map(|index| &tets[*index]));
        let Some(candidates) =
            face_neighbor_cavity_reconnection_candidates(&group, tets, node_points, options)?
        else {
            continue;
        };
        let candidate_below_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        let min_exact = min_exact_scaled_jacobian(candidates.iter());
        if !cavity_reconnection_improves_quality(
            candidate_below_count,
            min_exact,
            original_below_count,
            original_min_exact,
        ) {
            continue;
        }
        let quality_gain_only = candidate_below_count == original_below_count;
        if best
            .as_ref()
            .is_none_or(|(_, _, best_below_count, best_min_exact, _)| {
                candidate_below_count < *best_below_count
                    || (candidate_below_count == *best_below_count && min_exact > *best_min_exact)
            })
        {
            best = Some((
                group,
                candidates,
                candidate_below_count,
                min_exact,
                quality_gain_only,
            ));
        }
    }

    Ok(best.map(|(indices, candidates, _, _, quality_gain_only)| {
        (indices, candidates, quality_gain_only)
    }))
}

fn best_node_adjacent_cavity_reconnection(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<(Vec<usize>, Vec<TetCandidate>, bool)>, TetCandidateError> {
    let adjacent = connected_bad_tet_cavity_with_node_closure(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
    );
    if adjacent.len() < 3 {
        return Ok(None);
    }
    let face_closure =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options);
    if adjacent.len() <= face_closure.len() {
        return Ok(None);
    }

    let base = face_closure.into_iter().collect::<BTreeSet<_>>();
    let extra = bounded_node_cavity_extra_indices(
        adjacent
            .iter()
            .copied()
            .filter(|index| !base.contains(index))
            .collect::<Vec<_>>(),
        tets,
    );
    let mut candidate_groups = Vec::<Vec<usize>>::new();
    if adjacent.len() <= 24 {
        candidate_groups.push(adjacent.clone());
    }
    for extra_index in &extra {
        let mut group = base.clone();
        group.insert(*extra_index);
        candidate_groups.push(group.into_iter().collect());
    }
    for left in 0..extra.len() {
        for right in (left + 1)..extra.len() {
            let mut group = base.clone();
            group.insert(extra[left]);
            group.insert(extra[right]);
            candidate_groups.push(group.into_iter().collect());
        }
    }

    let mut best = None::<(Vec<usize>, Vec<TetCandidate>, usize, f64, bool)>;
    for group in candidate_groups {
        if group.len() < 3 || group.len() > 24 {
            continue;
        }
        let original_below_count = count_exact_quality_violations(
            group.iter().map(|index| &tets[*index]),
            options.min_scaled_jacobian,
        );
        let original_min_exact = min_exact_scaled_jacobian(group.iter().map(|index| &tets[*index]));
        let Some(candidates) =
            face_neighbor_cavity_reconnection_candidates(&group, tets, node_points, options)?
        else {
            continue;
        };
        let candidate_below_count =
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian);
        let min_exact = min_exact_scaled_jacobian(candidates.iter());
        if !cavity_reconnection_improves_quality(
            candidate_below_count,
            min_exact,
            original_below_count,
            original_min_exact,
        ) {
            continue;
        }
        let quality_gain_only = candidate_below_count == original_below_count;
        if best
            .as_ref()
            .is_none_or(|(_, _, best_below_count, best_min_exact, _)| {
                candidate_below_count < *best_below_count
                    || (candidate_below_count == *best_below_count && min_exact > *best_min_exact)
            })
        {
            best = Some((
                group,
                candidates,
                candidate_below_count,
                min_exact,
                quality_gain_only,
            ));
        }
    }
    Ok(best.map(|(indices, candidates, _, _, quality_gain_only)| {
        (indices, candidates, quality_gain_only)
    }))
}

fn connected_bad_tet_cavity(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    options: TetCandidateOptions,
) -> Vec<usize> {
    if tets[tet_index].exact_scaled_jacobian >= options.min_scaled_jacobian {
        return Vec::new();
    }
    let mut visited = BTreeSet::<usize>::new();
    let mut pending = vec![tet_index];
    while let Some(index) = pending.pop() {
        if !visited.insert(index) {
            continue;
        }
        for face in tet_node_faces(tets[index].node_ids).map(sorted_node_face) {
            if let Some(adjacent) = face_adjacency.get(&face) {
                for neighbor in adjacent {
                    if !visited.contains(neighbor)
                        && tets[*neighbor].component_id == tets[tet_index].component_id
                        && tets[*neighbor].exact_scaled_jacobian < options.min_scaled_jacobian
                    {
                        pending.push(*neighbor);
                    }
                }
            }
        }
    }
    visited.into_iter().collect()
}

fn connected_bad_tet_cavity_with_face_closure(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    options: TetCandidateOptions,
) -> Vec<usize> {
    connected_bad_tet_cavity_with_face_closure_layers(tet_index, tets, face_adjacency, options, 1)
}

fn connected_bad_tet_cavity_with_face_closure_layers(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    options: TetCandidateOptions,
    layer_count: usize,
) -> Vec<usize> {
    let mut adjacent = connected_bad_tet_cavity(tet_index, tets, face_adjacency, options)
        .into_iter()
        .collect::<BTreeSet<_>>();
    for _ in 0..layer_count {
        let mut changed = false;
        for index in adjacent.clone() {
            for face in tet_node_faces(tets[index].node_ids).map(sorted_node_face) {
                if let Some(face_neighbors) = face_adjacency.get(&face) {
                    for neighbor in face_neighbors {
                        if tets[*neighbor].component_id == tets[tet_index].component_id
                            && adjacent.insert(*neighbor)
                        {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    adjacent.into_iter().collect()
}

fn connected_bad_tet_cavity_with_node_closure(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    options: TetCandidateOptions,
) -> Vec<usize> {
    let mut adjacent =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options)
            .into_iter()
            .collect::<BTreeSet<_>>();
    if adjacent.is_empty() {
        return Vec::new();
    }
    let component_id = tets[tet_index].component_id;
    let cavity_nodes = adjacent
        .iter()
        .flat_map(|index| tets[*index].node_ids)
        .collect::<BTreeSet<_>>();
    for node_id in cavity_nodes {
        if let Some(node_neighbors) = node_adjacency.get(&node_id) {
            for neighbor in node_neighbors {
                if tets[*neighbor].component_id == component_id {
                    adjacent.insert(*neighbor);
                }
            }
        }
    }
    adjacent.into_iter().collect()
}

fn boundary_adjacent_bad_tet_cavity_with_node_closure(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    options: TetCandidateOptions,
) -> Vec<usize> {
    boundary_adjacent_bad_tet_cavity_with_node_closure_layers(
        tet_index,
        tets,
        face_adjacency,
        node_adjacency,
        options,
        1,
    )
}

fn boundary_adjacent_bad_tet_cavity_with_node_closure_layers(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
    node_adjacency: &BTreeMap<u32, Vec<usize>>,
    options: TetCandidateOptions,
    layer_count: usize,
) -> Vec<usize> {
    let mut adjacent =
        connected_bad_tet_cavity_with_face_closure(tet_index, tets, face_adjacency, options)
            .into_iter()
            .collect::<BTreeSet<_>>();
    if adjacent.is_empty() {
        return Vec::new();
    }
    let component_id = tets[tet_index].component_id;
    let has_boundary_face = adjacent.iter().any(|index| {
        tet_node_faces(tets[*index].node_ids)
            .map(sorted_node_face)
            .into_iter()
            .any(|face| face_adjacency.get(&face).map_or(0, Vec::len) == 1)
    });
    if !has_boundary_face {
        return Vec::new();
    }
    let cavity_nodes = adjacent
        .iter()
        .flat_map(|index| tets[*index].node_ids)
        .collect::<BTreeSet<_>>();
    let mut frontier = cavity_nodes;
    for _ in 0..layer_count {
        let mut next_frontier = BTreeSet::<u32>::new();
        for node_id in frontier {
            if let Some(node_neighbors) = node_adjacency.get(&node_id) {
                for neighbor in node_neighbors {
                    if tets[*neighbor].component_id == component_id && adjacent.insert(*neighbor) {
                        next_frontier.extend(tets[*neighbor].node_ids);
                    }
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
    adjacent.into_iter().collect()
}

fn one_ring_tet_cavity(
    tet_index: usize,
    tets: &[TetCandidate],
    face_adjacency: &BTreeMap<[u32; 3], Vec<usize>>,
) -> Vec<usize> {
    let mut neighbor_indices = BTreeSet::<usize>::from([tet_index]);
    for face in tet_node_faces(tets[tet_index].node_ids).map(sorted_node_face) {
        if let Some(adjacent) = face_adjacency.get(&face) {
            neighbor_indices.extend(adjacent.iter().copied());
        }
    }
    neighbor_indices.into_iter().collect()
}

fn face_neighbor_cavity_reconnection_candidates(
    adjacent: &[usize],
    tets: &[TetCandidate],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    let reference = &tets[adjacent[0]];
    let mut original_volume = 0.0_f64;
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for index in adjacent {
        let tet = &tets[*index];
        if tet.component_id != reference.component_id {
            return Ok(None);
        }
        original_volume += tet.volume_m3;
        for face in tet_node_faces(tet.node_ids).map(sorted_node_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let boundary_faces = face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect::<BTreeSet<_>>();
    if boundary_faces.len() < 4 {
        return Ok(None);
    }
    for face in &boundary_faces {
        boundary_nodes.extend(face.iter().copied());
    }
    if boundary_nodes.len() < 4 || boundary_nodes.len() > 16 {
        return Ok(None);
    }
    let points = boundary_nodes
        .iter()
        .map(|node_id| {
            Ok(ConnectivityPoint {
                node_id: *node_id,
                coordinates_m: *node_points
                    .get(node_id)
                    .ok_or(TetCandidateError::MissingSurfaceNode { node_id: *node_id })?,
                is_super: false,
            })
        })
        .collect::<Result<Vec<_>, TetCandidateError>>()?;
    let mut candidates = Vec::<TetCandidate>::new();
    for tet in tetrahedralize_points(&points) {
        let node_ids = tet.vertices.map(|index| points[index].node_id);
        let tet_points = tet.vertices.map(|index| points[index].coordinates_m);
        let Some(candidate) = raw_candidate_tet(
            reference.component_id,
            reference.source_surface_element_id,
            &reference.region_ids,
            node_ids,
            tet_points,
            options,
        ) else {
            return Ok(None);
        };
        candidates.push(candidate);
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    let candidate_boundary_faces = boundary_faces_from_tets(&candidates);
    if candidate_boundary_faces != boundary_faces {
        return Ok(None);
    }
    let candidate_volume = candidates
        .iter()
        .map(|candidate| candidate.volume_m3)
        .sum::<f64>();
    if (candidate_volume - original_volume).abs() > original_volume.max(1.0e-18) * 1.0e-9 {
        return Ok(None);
    }
    Ok(Some(candidates))
}

fn boundary_faces_from_tets(tets: &[TetCandidate]) -> BTreeSet<[u32; 3]> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tet in tets {
        for face in tet_node_faces(tet.node_ids).map(sorted_node_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect()
}

fn count_exact_quality_violations<'a>(
    tets: impl Iterator<Item = &'a TetCandidate>,
    min_scaled_jacobian: f64,
) -> usize {
    tets.filter(|tet| tet.exact_scaled_jacobian < min_scaled_jacobian)
        .count()
}

fn count_tets_below_exact_quality<'a>(
    tets: impl Iterator<Item = &'a TetCandidate>,
    threshold: f64,
) -> usize {
    tets.filter(|tet| tet.exact_scaled_jacobian < threshold)
        .count()
}

fn min_exact_scaled_jacobian<'a>(tets: impl Iterator<Item = &'a TetCandidate>) -> f64 {
    tets.map(|tet| tet.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min)
}

fn cavity_reconnection_improves_quality(
    candidate_below_count: usize,
    candidate_min_exact: f64,
    original_below_count: usize,
    original_min_exact: f64,
) -> bool {
    candidate_below_count < original_below_count
        || (candidate_below_count == original_below_count
            && candidate_min_exact > original_min_exact + 1.0e-12)
}

fn two_tet_reconnection_candidates(
    tet: &TetCandidate,
    neighbor: &TetCandidate,
    shared_face: [u32; 3],
    node_points: &BTreeMap<u32, [f64; 3]>,
    options: TetCandidateOptions,
) -> Result<Option<Vec<TetCandidate>>, TetCandidateError> {
    let Some(tet_apex) = opposite_tet_node(tet.node_ids, shared_face) else {
        return Ok(None);
    };
    let Some(neighbor_apex) = opposite_tet_node(neighbor.node_ids, shared_face) else {
        return Ok(None);
    };
    let original_volume = tet.volume_m3 + neighbor.volume_m3;
    let candidates = [
        [tet_apex, neighbor_apex, shared_face[0], shared_face[1]],
        [tet_apex, neighbor_apex, shared_face[1], shared_face[2]],
        [tet_apex, neighbor_apex, shared_face[2], shared_face[0]],
    ]
    .into_iter()
    .map(|node_ids| {
        let points = [
            *node_points
                .get(&node_ids[0])
                .ok_or(TetCandidateError::MissingSurfaceNode {
                    node_id: node_ids[0],
                })?,
            *node_points
                .get(&node_ids[1])
                .ok_or(TetCandidateError::MissingSurfaceNode {
                    node_id: node_ids[1],
                })?,
            *node_points
                .get(&node_ids[2])
                .ok_or(TetCandidateError::MissingSurfaceNode {
                    node_id: node_ids[2],
                })?,
            *node_points
                .get(&node_ids[3])
                .ok_or(TetCandidateError::MissingSurfaceNode {
                    node_id: node_ids[3],
                })?,
        ];
        Ok(raw_candidate_tet(
            tet.component_id,
            tet.source_surface_element_id,
            &tet.region_ids,
            node_ids,
            points,
            options,
        ))
    })
    .collect::<Result<Option<Vec<_>>, TetCandidateError>>()?;
    let Some(candidates) = candidates else {
        return Ok(None);
    };
    let candidate_volume = candidates
        .iter()
        .map(|candidate| candidate.volume_m3)
        .sum::<f64>();
    if (candidate_volume - original_volume).abs() > original_volume.max(1.0e-18) * 1.0e-9 {
        return Ok(None);
    }
    Ok(Some(candidates))
}

fn opposite_tet_node(node_ids: [u32; 4], face: [u32; 3]) -> Option<u32> {
    node_ids.into_iter().find(|node_id| !face.contains(node_id))
}

fn best_centroid_split_tets(
    tet: &TetCandidate,
    split_node_id: u32,
    points: [[f64; 3]; 4],
    options: TetCandidateOptions,
) -> Option<([f64; 3], Vec<TetCandidate>)> {
    let centroid = tet_centroid(points);
    let mut best = None::<([f64; 3], Vec<TetCandidate>, usize, f64)>;
    for split_point in centroid_repair_points(centroid, points) {
        let candidates = centroid_split_tets(tet, split_node_id, split_point, points, options);
        if candidates.len() != 4 {
            continue;
        }
        let below_threshold_count = candidates
            .iter()
            .filter(|candidate| candidate.exact_scaled_jacobian < options.min_scaled_jacobian)
            .count();
        let min_exact = candidates
            .iter()
            .map(|candidate| candidate.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best.as_ref().is_none_or(|(_, _, best_count, best_min)| {
            below_threshold_count < *best_count
                || (below_threshold_count == *best_count && min_exact > *best_min)
        }) {
            best = Some((split_point, candidates, below_threshold_count, min_exact));
        }
    }
    best.map(|(point, candidates, _, _)| (point, candidates))
}

fn centroid_repair_points(centroid: [f64; 3], points: [[f64; 3]; 4]) -> Vec<[f64; 3]> {
    let mut candidates = Vec::with_capacity(93);
    candidates.push(centroid);
    for point in points {
        candidates.push([
            centroid[0] * 0.75 + point[0] * 0.25,
            centroid[1] * 0.75 + point[1] * 0.25,
            centroid[2] * 0.75 + point[2] * 0.25,
        ]);
        candidates.push([
            centroid[0] * 0.50 + point[0] * 0.50,
            centroid[1] * 0.50 + point[1] * 0.50,
            centroid[2] * 0.50 + point[2] * 0.50,
        ]);
    }
    let denominator = 10_usize;
    for a in 1..denominator {
        for b in 1..(denominator - a) {
            for c in 1..(denominator - a - b) {
                let d = denominator - a - b - c;
                if d == 0 {
                    continue;
                }
                let weights = [
                    a as f64 / denominator as f64,
                    b as f64 / denominator as f64,
                    c as f64 / denominator as f64,
                    d as f64 / denominator as f64,
                ];
                candidates.push([
                    points[0][0] * weights[0]
                        + points[1][0] * weights[1]
                        + points[2][0] * weights[2]
                        + points[3][0] * weights[3],
                    points[0][1] * weights[0]
                        + points[1][1] * weights[1]
                        + points[2][1] * weights[2]
                        + points[3][1] * weights[3],
                    points[0][2] * weights[0]
                        + points[1][2] * weights[1]
                        + points[2][2] * weights[2]
                        + points[3][2] * weights[3],
                ]);
            }
        }
    }
    candidates
}

fn centroid_split_tets(
    tet: &TetCandidate,
    split_node_id: u32,
    split_point: [f64; 3],
    points: [[f64; 3]; 4],
    options: TetCandidateOptions,
) -> Vec<TetCandidate> {
    let faces = [
        (
            [tet.node_ids[0], tet.node_ids[1], tet.node_ids[2]],
            [points[0], points[1], points[2]],
        ),
        (
            [tet.node_ids[0], tet.node_ids[1], tet.node_ids[3]],
            [points[0], points[1], points[3]],
        ),
        (
            [tet.node_ids[0], tet.node_ids[2], tet.node_ids[3]],
            [points[0], points[2], points[3]],
        ),
        (
            [tet.node_ids[1], tet.node_ids[2], tet.node_ids[3]],
            [points[1], points[2], points[3]],
        ),
    ];
    let mut split_tets = Vec::<TetCandidate>::with_capacity(4);
    for (face_node_ids, face_points) in faces {
        let node_ids = [
            face_node_ids[0],
            face_node_ids[1],
            face_node_ids[2],
            split_node_id,
        ];
        let split_points = [face_points[0], face_points[1], face_points[2], split_point];
        let Some(candidate) = raw_candidate_tet(
            tet.component_id,
            tet.source_surface_element_id,
            &tet.region_ids,
            node_ids,
            split_points,
            options,
        ) else {
            return Vec::new();
        };
        split_tets.push(candidate);
    }
    split_tets
}

fn raw_candidate_tet(
    component_id: u32,
    source_surface_element_id: u32,
    region_ids: &[String],
    mut node_ids: [u32; 4],
    points: [[f64; 3]; 4],
    options: TetCandidateOptions,
) -> Option<TetCandidate> {
    let mut signed_volume_m3 = tet_signed_volume(points);
    if signed_volume_m3 < 0.0 {
        node_ids.swap(1, 2);
        signed_volume_m3 = -signed_volume_m3;
    }
    let volume_m3 = signed_volume_m3.abs();
    if volume_m3 < options.min_volume_m3 {
        return None;
    }
    let aspect_ratio = tet_edge_aspect_ratio(points);
    if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
        return None;
    }
    Some(TetCandidate {
        tet_id: 0,
        component_id,
        node_ids,
        source_surface_element_id,
        region_ids: region_ids.to_vec(),
        volume_m3,
        aspect_ratio,
        exact_scaled_jacobian: tet_scaled_jacobian(points),
    })
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

#[derive(Debug, Clone, PartialEq)]
struct TetCandidateQualitySummary {
    max_radius_edge_ratio: f64,
    min_exact_scaled_jacobian: f64,
    exact_scaled_jacobian_below_threshold_count: usize,
    exact_scaled_jacobian_bins: BTreeMap<String, usize>,
}

fn tet_candidate_quality_summary(
    nodes: &[TetCandidateNode],
    tets: &[TetCandidate],
    options: TetCandidateOptions,
) -> Result<TetCandidateQualitySummary, TetCandidateError> {
    let nodes = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut max_radius_edge_ratio = 0.0_f64;
    let mut min_exact_scaled_jacobian = f64::INFINITY;
    let mut exact_scaled_jacobian_below_threshold_count = 0_usize;
    let mut exact_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    for tet in tets {
        let points = candidate_tet_points(tet, &nodes)?;
        let radius_edge_ratio = tet_radius_edge_ratio(points, MeshingTolerance::default());
        if radius_edge_ratio.is_finite() {
            max_radius_edge_ratio = max_radius_edge_ratio.max(radius_edge_ratio);
        }
        let exact_scaled_jacobian = tet.exact_scaled_jacobian;
        min_exact_scaled_jacobian = min_exact_scaled_jacobian.min(exact_scaled_jacobian);
        *exact_scaled_jacobian_bins
            .entry(exact_scaled_jacobian_bin(exact_scaled_jacobian))
            .or_default() += 1;
        if exact_scaled_jacobian < options.min_scaled_jacobian {
            exact_scaled_jacobian_below_threshold_count += 1;
        }
    }
    Ok(TetCandidateQualitySummary {
        max_radius_edge_ratio,
        min_exact_scaled_jacobian,
        exact_scaled_jacobian_below_threshold_count,
        exact_scaled_jacobian_bins,
    })
}

fn exact_scaled_jacobian_bin(value: f64) -> String {
    if value < 0.0 {
        "lt_0".to_string()
    } else if value < 0.15 {
        "0_to_0_15".to_string()
    } else if value < 0.35 {
        "0_15_to_0_35".to_string()
    } else if value < 0.65 {
        "0_35_to_0_65".to_string()
    } else {
        "gte_0_65".to_string()
    }
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
pub(crate) struct ConnectivityPoint {
    pub(crate) node_id: u32,
    pub(crate) coordinates_m: [f64; 3],
    pub(crate) is_super: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConnectivityTet {
    pub(crate) vertices: [usize; 4],
}

pub(crate) fn tetrahedralize_points(input_points: &[ConnectivityPoint]) -> Vec<ConnectivityTet> {
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
    below_threshold_count: usize,
    min_scaled_jacobian: f64,
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
    let mut min_scaled_jacobian = f64::INFINITY;
    let mut below_threshold_count = 0_usize;

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
        let scaled_jacobian = tet_scaled_jacobian(points);
        min_scaled_jacobian = min_scaled_jacobian.min(scaled_jacobian);
        if scaled_jacobian < options.min_scaled_jacobian {
            below_threshold_count += 1;
        }
    }

    Ok(fan_seed_score_from_accumulators(
        point,
        valid_tet_count,
        below_threshold_count,
        total_volume_m3,
        aspect_ratio_sum,
        max_aspect_ratio,
        min_scaled_jacobian,
        component.volume_m3,
    ))
}

#[allow(clippy::too_many_arguments)]
fn accumulate_fan_seed_score_tets(
    tets: &[TetCandidate],
    below_threshold_count: usize,
    min_scaled_jacobian: f64,
    valid_tet_count: &mut usize,
    total_below_threshold_count: &mut usize,
    total_volume_m3: &mut f64,
    aspect_ratio_sum: &mut f64,
    max_aspect_ratio: &mut f64,
    total_min_scaled_jacobian: &mut f64,
) {
    *valid_tet_count += tets.len();
    *total_below_threshold_count += below_threshold_count;
    *total_volume_m3 += tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
    *aspect_ratio_sum += tets.iter().map(|tet| tet.aspect_ratio).sum::<f64>();
    *max_aspect_ratio = max_aspect_ratio.max(max_candidate_aspect_ratio(tets));
    *total_min_scaled_jacobian = total_min_scaled_jacobian.min(min_scaled_jacobian);
}

#[allow(clippy::too_many_arguments)]
fn fan_seed_score_from_accumulators(
    point: [f64; 3],
    valid_tet_count: usize,
    below_threshold_count: usize,
    total_volume_m3: f64,
    aspect_ratio_sum: f64,
    max_aspect_ratio: f64,
    min_scaled_jacobian: f64,
    expected_volume_m3: f64,
) -> FanSeedScore {
    let mean_aspect_ratio = if valid_tet_count == 0 {
        f64::INFINITY
    } else {
        aspect_ratio_sum / valid_tet_count as f64
    };
    let volume_error_ratio = if expected_volume_m3 > 0.0 {
        ((total_volume_m3 - expected_volume_m3).abs() / expected_volume_m3).abs()
    } else {
        f64::INFINITY
    };
    FanSeedScore {
        point,
        valid_tet_count,
        below_threshold_count,
        min_scaled_jacobian,
        volume_error_ratio,
        max_aspect_ratio,
        mean_aspect_ratio,
    }
}

fn fan_seed_score_is_better(candidate: FanSeedScore, best: FanSeedScore) -> bool {
    candidate
        .valid_tet_count
        .cmp(&best.valid_tet_count)
        .then_with(|| {
            best.below_threshold_count
                .cmp(&candidate.below_threshold_count)
        })
        .then_with(|| {
            candidate
                .min_scaled_jacobian
                .partial_cmp(&best.min_scaled_jacobian)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
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
    let center_clearance_size = options.interior_target_size_m.unwrap_or_else(|| {
        (0..3)
            .map(|axis| component.bounds_max_m[axis] - component.bounds_min_m[axis])
            .filter(|span| span.is_finite())
            .fold(0.0_f64, f64::max)
            .max(1.0)
    });
    if classifier.contains_protected_interior_point(center, center_clearance_size, tolerance) {
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
                    if classifier.contains_protected_interior_point(point, target_size_m, tolerance)
                    {
                        points.push(point);
                    }
                }
            }
        }
    }

    if points.is_empty() && classifier.contains_interior_point(center) {
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

fn seed_node_source(
    point: [f64; 3],
    classifier: &ComponentSurfaceClassifier,
) -> TetCandidateNodeSource {
    if classifier.point_is_on_boundary(point) {
        TetCandidateNodeSource::BoundaryRecovery
    } else {
        TetCandidateNodeSource::InteriorSeed
    }
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
        self.point_is_inside_by_votes(point)
    }

    fn contains_interior_point(&self, point: [f64; 3]) -> bool {
        !self.point_is_on_boundary(point) && self.point_is_inside_by_votes(point)
    }

    fn contains_protected_interior_point(
        &self,
        point: [f64; 3],
        target_size_m: f64,
        tolerance: MeshingTolerance,
    ) -> bool {
        self.contains_interior_point(point)
            && self.nearest_surface_distance(point)
                > protected_interior_seed_clearance(target_size_m, tolerance)
    }

    fn point_is_inside_by_votes(&self, point: [f64; 3]) -> bool {
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
        inside_votes >= 2
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

    fn nearest_surface_distance(&self, point: [f64; 3]) -> f64 {
        self.triangles_by_element_id
            .values()
            .map(|triangle| point_triangle_distance(point, *triangle))
            .fold(f64::INFINITY, f64::min)
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

fn protected_interior_seed_clearance(target_size_m: f64, tolerance: MeshingTolerance) -> f64 {
    let target_size_m = if target_size_m.is_finite() && target_size_m > 0.0 {
        target_size_m
    } else {
        1.0
    };
    (target_size_m.min(1.0) * 0.01).max(tolerance.absolute_m * 10.0)
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
        assert_eq!(
            candidates
                .recovery
                .exact_scaled_jacobian_bins
                .values()
                .sum::<usize>(),
            candidates.tets.len()
        );
        assert!(!candidates.recovery.exact_scaled_jacobian_bins.is_empty());
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
    fn classifier_distinguishes_boundary_containment_from_interior_seed_eligibility() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let component = &volume_candidates.components[0];
        let surface_elements = surface
            .elements
            .iter()
            .map(|element| (element.element_id, element))
            .collect::<BTreeMap<_, _>>();
        let classifier = ComponentSurfaceClassifier::new(
            component,
            &surface,
            &surface_elements,
            MeshingTolerance::default(),
        )
        .expect("classifier should build");

        assert!(classifier.contains_point([0.0, 0.5, 0.5]));
        assert!(!classifier.contains_interior_point([0.0, 0.5, 0.5]));
        assert!(classifier.contains_interior_point([0.5, 0.5, 0.5]));
        assert!(classifier.contains_interior_point([0.001, 0.5, 0.5]));
        assert!(!classifier.contains_protected_interior_point(
            [0.001, 0.5, 0.5],
            0.4,
            MeshingTolerance::default(),
        ));
        assert!(!classifier.contains_point([1.5, 0.5, 0.5]));
        assert!(!classifier.contains_interior_point([1.5, 0.5, 0.5]));
    }

    #[test]
    fn boundary_seed_points_are_classified_as_boundary_recovery_nodes() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let component = &volume_candidates.components[0];
        let surface_elements = surface
            .elements
            .iter()
            .map(|element| (element.element_id, element))
            .collect::<BTreeMap<_, _>>();
        let tolerance =
            MeshingTolerance::from_bounds(component.bounds_min_m, component.bounds_max_m);
        let classifier =
            ComponentSurfaceClassifier::new(component, &surface, &surface_elements, tolerance)
                .expect("cube classifier should build");

        assert_eq!(
            seed_node_source([0.0, 0.5, 0.5], &classifier),
            TetCandidateNodeSource::BoundaryRecovery
        );
        assert_eq!(
            seed_node_source([0.5, 0.5, 0.5], &classifier),
            TetCandidateNodeSource::InteriorSeed
        );
    }

    #[test]
    fn refinement_pass_rolls_back_quality_regressing_seed_points() {
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

        assert_eq!(candidates.interior_seed_points.len(), 9);
        assert!(candidates.interior_seed_points.len() <= 12);
        assert_eq!(candidates.recovery.refinement_pass_count, 0);
        assert_eq!(candidates.recovery.refinement_point_count, 0);
        assert!(candidates.recovery.sizing_violation_count > 0);
        assert!(candidates.recovery.max_radius_edge_ratio.is_finite());
        assert_eq!(candidates.recovery.fan_fallback_component_count, 0);
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
        assert_eq!(
            candidates
                .tets
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < 0.15)
                .count(),
            0
        );
    }

    #[test]
    fn quality_refinement_rejects_boundary_centroid_seed_points() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let component = &volume_candidates.components[0];
        let surface_nodes = surface
            .nodes
            .iter()
            .map(|node| (node.node_id, node.coordinates_m))
            .collect::<BTreeMap<_, _>>();
        let surface_elements = surface
            .elements
            .iter()
            .map(|element| (element.element_id, element))
            .collect::<BTreeMap<_, _>>();
        let tolerance =
            MeshingTolerance::from_bounds(component.bounds_min_m, component.bounds_max_m);
        let classifier =
            ComponentSurfaceClassifier::new(component, &surface, &surface_elements, tolerance)
                .expect("cube classifier should build");
        let bad_boundary_tet = TetCandidate {
            tet_id: 0,
            component_id: component.component_id,
            node_ids: [0, 1, 2, 3],
            source_surface_element_id: 0,
            region_ids: vec!["body".to_string()],
            volume_m3: 0.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: -1.0,
        };

        let points = refinement_points_for_tets(
            &[bad_boundary_tet],
            &surface_nodes,
            &[],
            &[],
            tolerance,
            &classifier,
            TetCandidateOptions {
                interior_target_size_m: Some(0.5),
                max_refinement_passes: 1,
                ..TetCandidateOptions::default()
            },
            4,
            &BTreeSet::new(),
            true,
        )
        .expect("refinement point selection should complete");

        assert!(
            points.points.is_empty(),
            "quality-driven refinement must not add boundary centroid seed points"
        );
    }

    #[test]
    fn requested_refinement_points_are_accepted_when_quality_safe() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let mut requested_refinement_points = [[0.0; 3]; 16];
        requested_refinement_points[0] = [0.25, 0.25, 0.25];

        let candidates = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                interior_target_size_m: Some(2.0),
                requested_refinement_points,
                requested_refinement_point_count: 1,
                max_interior_seed_points: 2,
                max_refinement_passes: 1,
                max_radius_edge_ratio: 10.0,
                ..TetCandidateOptions::default()
            },
        )
        .expect("Tet candidates should form with requested refinement");

        assert!(candidates
            .interior_seed_points
            .iter()
            .any(|point| distance_squared(*point, requested_refinement_points[0]) <= 1.0e-24));
        assert_eq!(
            candidates.accepted_requested_refinement_points,
            vec![requested_refinement_points[0]]
        );
        assert_eq!(
            candidates.accepted_requested_refinement_sample_indices,
            vec![0]
        );
        assert!(candidates
            .dropped_requested_refinement_sample_indices
            .is_empty());
        assert_eq!(candidates.recovery.requested_refinement_point_count, 1);
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_candidate_count,
            1
        );
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_point_count,
            1
        );
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_surrogate_point_count,
            0
        );
        assert_eq!(
            candidates
                .recovery
                .rejected_requested_refinement_point_count,
            0
        );
        assert!(candidates
            .recovery
            .requested_refinement_rejected_by_reason
            .is_empty());
        assert_eq!(candidates.recovery.refinement_pass_count, 1);
        assert_eq!(candidates.recovery.refinement_point_count, 1);
        assert_eq!(candidates.recovery.fan_fallback_component_count, 0);
        assert_eq!(
            candidates
                .tets
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < 0.15)
                .count(),
            0
        );
    }

    #[test]
    fn boundary_adjacent_requested_refinement_uses_quality_safe_surrogate() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let mut requested_refinement_points = [[0.0; 3]; 16];
        requested_refinement_points[0] = [1.0e-6, 1.0e-6, 1.0e-6];

        let candidates = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                interior_target_size_m: Some(2.0),
                requested_refinement_points,
                requested_refinement_point_count: 1,
                max_interior_seed_points: 2,
                max_refinement_passes: 1,
                max_radius_edge_ratio: 10.0,
                ..TetCandidateOptions::default()
            },
        )
        .expect("Tet candidates should form with requested refinement rollback");

        assert!(!candidates
            .interior_seed_points
            .iter()
            .any(|point| distance_squared(*point, requested_refinement_points[0]) <= 1.0e-24));
        assert_eq!(candidates.accepted_requested_refinement_points.len(), 1);
        assert_eq!(
            candidates.accepted_requested_refinement_sample_indices,
            vec![0]
        );
        assert!(candidates
            .dropped_requested_refinement_sample_indices
            .is_empty());
        assert!(candidates
            .recovery
            .requested_refinement_dropped_by_reason
            .is_empty());
        assert!(
            distance_squared(
                candidates.accepted_requested_refinement_points[0],
                requested_refinement_points[0]
            ) > 1.0e-12,
            "accepted point should move inward from the boundary-adjacent request"
        );
        assert!(candidates
            .interior_seed_points
            .iter()
            .any(|point| distance_squared(
                *point,
                candidates.accepted_requested_refinement_points[0]
            ) <= 1.0e-24));
        assert_eq!(candidates.recovery.requested_refinement_point_count, 1);
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_candidate_count,
            1
        );
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_point_count,
            1
        );
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_surrogate_point_count,
            1
        );
        assert_eq!(
            candidates
                .recovery
                .rejected_requested_refinement_point_count,
            0
        );
        assert_eq!(candidates.recovery.refinement_pass_count, 1);
        assert_eq!(candidates.recovery.refinement_point_count, 1);
        assert_eq!(
            candidates
                .tets
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < 0.15)
                .count(),
            0
        );
    }

    #[test]
    fn requested_refinement_uses_nearby_quality_safe_point_when_exact_point_regresses() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let mut requested_refinement_points = [[0.0; 3]; 16];
        requested_refinement_points[0] = [0.05, 0.05, 0.05];

        let candidates = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                interior_target_size_m: Some(2.0),
                requested_refinement_points,
                requested_refinement_point_count: 1,
                max_interior_seed_points: 2,
                max_refinement_passes: 1,
                max_radius_edge_ratio: 10.0,
                ..TetCandidateOptions::default()
            },
        )
        .expect("Tet candidates should form with requested refinement fallback");

        assert_eq!(candidates.recovery.requested_refinement_point_count, 1);
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_candidate_count,
            1
        );
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_point_count,
            1
        );
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_surrogate_point_count,
            1
        );
        assert_eq!(
            candidates
                .recovery
                .rejected_requested_refinement_point_count,
            0
        );
        assert_eq!(candidates.recovery.refinement_pass_count, 1);
        assert_eq!(candidates.recovery.refinement_point_count, 1);
        assert_eq!(candidates.accepted_requested_refinement_points.len(), 1);
        assert_eq!(
            candidates.accepted_requested_refinement_sample_indices,
            vec![0]
        );
        assert!(candidates
            .dropped_requested_refinement_sample_indices
            .is_empty());
        assert!(
            distance_squared(
                candidates.accepted_requested_refinement_points[0],
                requested_refinement_points[0]
            ) > 1.0e-12,
            "accepted point should be a quality-safe surrogate rather than the exact requested point"
        );
        assert!(candidates
            .interior_seed_points
            .iter()
            .any(|point| distance_squared(
                *point,
                candidates.accepted_requested_refinement_points[0]
            ) <= 1.0e-24));
        assert_eq!(
            candidates
                .tets
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < 0.15)
                .count(),
            0
        );
    }

    #[test]
    fn retained_requested_refinement_tracks_markers_removed_by_repair() {
        let retained = retained_requested_refinement_points(
            vec![
                (10, [0.1, 0.1, 0.1], 0),
                (11, [0.2, 0.2, 0.2], 1),
                (12, [0.3, 0.3, 0.3], 2),
            ],
            &BTreeSet::from([10, 12]),
        );

        assert_eq!(retained.points, vec![[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]);
        assert_eq!(retained.sample_indices, vec![0, 2]);
        assert_eq!(retained.dropped_sample_indices, vec![1]);
        assert_eq!(
            requested_refinement_drop_reasons(&retained.dropped_sample_indices),
            BTreeMap::from([("not_retained_after_repair".to_string(), 1)])
        );
    }

    #[test]
    fn rejected_requested_refinement_counts_attempted_unaccepted_markers() {
        let attempted = BTreeSet::from([0_usize, 1, 2, 4]);
        let accepted = BTreeSet::from([1_usize, 4]);

        assert_eq!(
            rejected_requested_refinement_point_count(&attempted, &accepted),
            2
        );
    }

    #[test]
    fn requested_refinement_rejection_reasons_are_actionable() {
        let attempted = BTreeSet::from([0_usize, 1]);
        let duplicate = BTreeSet::from([2_usize]);
        let accepted = BTreeSet::from([1_usize]);

        assert_eq!(
            requested_refinement_rejection_reasons(4, &attempted, &duplicate, &accepted),
            BTreeMap::from([("quality_or_recovery".to_string(), 1)])
        );
    }

    #[test]
    fn requested_refinement_surrogates_include_bounded_local_stencil() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let component = &volume_candidates.components[0];
        let surface_elements = surface
            .elements
            .iter()
            .map(|element| (element.element_id, element))
            .collect::<BTreeMap<_, _>>();
        let tolerance = MeshingTolerance::default();
        let classifier =
            ComponentSurfaceClassifier::new(component, &surface, &surface_elements, tolerance)
                .expect("classifier should build");
        let requested_point = [0.3, 0.3, 0.3];
        let seed_points = [[0.5, 0.5, 0.5]];

        let candidates = requested_refinement_candidate_points(
            requested_point,
            &seed_points,
            &classifier,
            0.4,
            tolerance,
        );

        assert!(candidates.len() > 32);
        assert!(candidates.len() <= 56);
        assert_eq!(candidates[0], requested_point);
        assert!(candidates
            .iter()
            .all(|candidate| classifier.contains_point(*candidate)));
        assert!(candidates.iter().any(|candidate| {
            candidate[0] > requested_point[0]
                && (candidate[1] - requested_point[1]).abs() <= f64::EPSILON
                && (candidate[2] - requested_point[2]).abs() <= f64::EPSILON
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate[0] > requested_point[0]
                && candidate[1] > requested_point[1]
                && candidate[2] > requested_point[2]
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate[0] > requested_point[0]
                && candidate[1] > requested_point[1]
                && (candidate[2] - requested_point[2]).abs() <= f64::EPSILON
        }));
        for (left_index, left) in candidates.iter().enumerate() {
            for right in candidates.iter().skip(left_index + 1) {
                assert!(!tolerance.point_nearly_equal(*left, *right, 1.0));
            }
        }
    }

    #[test]
    fn requested_refinement_ranking_is_deterministic_and_distance_aware() {
        let mut ranked = vec![
            RankedRefinementPoint {
                point: [0.4, 0.0, 0.0],
                score: requested_refinement_score(0.2, 1.0),
                requested_id: Some(1),
                requested_distance_m: 0.2,
                quality_driven: false,
            },
            RankedRefinementPoint {
                point: [0.3, 0.0, 0.0],
                score: requested_refinement_score(0.3, 1.0),
                requested_id: Some(0),
                requested_distance_m: 0.3,
                quality_driven: false,
            },
            RankedRefinementPoint {
                point: [0.1, 0.0, 0.0],
                score: requested_refinement_score(0.1, 1.0),
                requested_id: Some(1),
                requested_distance_m: 0.1,
                quality_driven: false,
            },
            RankedRefinementPoint {
                point: [0.2, 0.0, 0.0],
                score: requested_refinement_score(0.2, 1.0),
                requested_id: Some(1),
                requested_distance_m: 0.2,
                quality_driven: false,
            },
            RankedRefinementPoint {
                point: [0.0, 0.0, 0.0],
                score: 10.0,
                requested_id: None,
                requested_distance_m: f64::INFINITY,
                quality_driven: true,
            },
        ];

        ranked.sort_by(compare_ranked_refinement_points);

        assert_eq!(ranked[0].requested_id, Some(1));
        assert_eq!(ranked[0].point, [0.1, 0.0, 0.0]);
        assert_eq!(ranked[1].requested_id, Some(1));
        assert_eq!(ranked[1].point, [0.2, 0.0, 0.0]);
        assert_eq!(ranked[2].requested_id, Some(1));
        assert_eq!(ranked[2].point, [0.4, 0.0, 0.0]);
        assert_eq!(ranked[3].requested_id, Some(0));
        assert_eq!(ranked[4].requested_id, None);
    }

    #[test]
    fn requested_refinement_candidates_are_capped_per_marker_after_ranking() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let component = &volume_candidates.components[0];
        let surface_elements = surface
            .elements
            .iter()
            .map(|element| (element.element_id, element))
            .collect::<BTreeMap<_, _>>();
        let tolerance = MeshingTolerance::default();
        let classifier =
            ComponentSurfaceClassifier::new(component, &surface, &surface_elements, tolerance)
                .expect("classifier should build");
        let mut requested_refinement_points = [[0.0; 3]; 16];
        requested_refinement_points[0] = [0.3, 0.3, 0.3];

        let candidates = refinement_points_for_tets(
            &[],
            &BTreeMap::new(),
            &[],
            &[[0.5, 0.5, 0.5]],
            tolerance,
            &classifier,
            TetCandidateOptions {
                interior_target_size_m: Some(0.4),
                requested_refinement_points,
                requested_refinement_point_count: 1,
                max_requested_refinement_candidates_per_point: 4,
                ..TetCandidateOptions::default()
            },
            16,
            &BTreeSet::new(),
            false,
        )
        .expect("requested refinement candidates should build");

        assert_eq!(candidates.requested_point_count, 1);
        assert_eq!(candidates.points.len(), 4);
        assert!(candidates
            .points
            .iter()
            .all(|candidate| candidate.requested_id == Some(0)));
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
        assert!(candidates
            .recovery
            .optimization_initial_max_aspect_ratio
            .is_finite());
        assert!(candidates
            .recovery
            .optimization_final_max_aspect_ratio
            .is_finite());
        assert!(
            candidates.recovery.optimization_final_max_aspect_ratio
                <= candidates.recovery.optimization_initial_max_aspect_ratio + 1.0e-12
        );
        assert!(candidates
            .recovery
            .optimization_initial_min_exact_scaled_jacobian
            .is_finite());
        assert!(
            candidates
                .recovery
                .optimization_final_min_exact_scaled_jacobian
                + 1.0e-12
                >= candidates
                    .recovery
                    .optimization_initial_min_exact_scaled_jacobian
        );
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn optimization_quality_ordering_tracks_aspect_ratio_without_relaxing_exact_quality() {
        let current = CandidateQualitySnapshot {
            max_aspect_ratio: 12.0,
            max_radius_edge_ratio: 0.0,
            volume_ratio_error: 0.0,
            sliver_count: 0,
            exact_quality_violation_count: 0,
            min_exact_scaled_jacobian: 0.45,
        };
        let aspect_improved = CandidateQualitySnapshot {
            max_aspect_ratio: 8.0,
            ..current
        };
        let aspect_regressed = CandidateQualitySnapshot {
            max_aspect_ratio: 16.0,
            ..current
        };
        let exact_regressed = CandidateQualitySnapshot {
            min_exact_scaled_jacobian: 0.44,
            max_aspect_ratio: 8.0,
            ..current
        };

        assert!(candidate_quality_is_no_worse(aspect_improved, current));
        assert!(candidate_quality_is_better(aspect_improved, current));
        assert!(!candidate_quality_is_no_worse(aspect_regressed, current));
        assert!(!candidate_quality_is_better(aspect_regressed, current));
        assert!(!candidate_quality_is_no_worse(exact_regressed, current));
        assert!(!candidate_quality_is_better(exact_regressed, current));
    }

    #[test]
    fn optimization_quality_aggregate_tracks_rejected_edits() {
        let mut aggregate = OptimizationQualityAggregate::default();
        aggregate.record(SmoothingSummary {
            pass_count: 1,
            smoothed_point_count: 2,
            sliver_candidate_count: 3,
            sliver_removed_count: 2,
            target_seed_count: 5,
            skipped_target_seed_count: 2,
            rejected_edit_count: 4,
            quality_sample_count: 1,
            initial_max_aspect_ratio: 9.0,
            final_max_aspect_ratio: 7.0,
            initial_min_exact_scaled_jacobian: 0.35,
            final_min_exact_scaled_jacobian: 0.42,
        });
        aggregate.record(SmoothingSummary {
            pass_count: 0,
            smoothed_point_count: 0,
            sliver_candidate_count: 1,
            sliver_removed_count: 1,
            target_seed_count: 3,
            skipped_target_seed_count: 1,
            rejected_edit_count: 2,
            quality_sample_count: 1,
            initial_max_aspect_ratio: 8.0,
            final_max_aspect_ratio: 6.0,
            initial_min_exact_scaled_jacobian: 0.30,
            final_min_exact_scaled_jacobian: 0.40,
        });

        assert_eq!(aggregate.rejected_edit_count, 6);
        assert_eq!(aggregate.sliver_removed_count(), 3);
        assert_eq!(aggregate.target_seed_count(), 8);
        assert_eq!(aggregate.skipped_target_seed_count(), 3);
        assert_eq!(aggregate.initial_max_aspect_ratio(), 9.0);
        assert_eq!(aggregate.final_max_aspect_ratio(), 7.0);
        assert_eq!(aggregate.initial_min_exact_scaled_jacobian(), 0.30);
        assert_eq!(aggregate.final_min_exact_scaled_jacobian(), 0.40);
    }

    #[test]
    fn optimization_targets_include_sliver_only_seed_tets() {
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.0]),
            (1, [0.0, 0.0, 2.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [-0.5, 3.0_f64.sqrt() * 0.5, 0.0]),
        ]);
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.01,
            sliver_aspect_ratio: 2.0,
            ..TetCandidateOptions::default()
        };
        let tet = raw_candidate_tet(
            0,
            0,
            &[],
            [0, 1, 2, 3],
            [0, 1, 2, 3].map(|node_id| node_points[&node_id]),
            options,
        )
        .expect("sliver-only fixture tet should be valid");
        assert!(tet.exact_scaled_jacobian >= options.min_scaled_jacobian);
        assert!(tet.aspect_ratio > options.sliver_aspect_ratio);

        let seed_summary = optimization_target_seed_indices(&[tet], &[1, 3], options);

        assert_eq!(seed_summary.indices, vec![0, 1]);
        assert_eq!(seed_summary.total_count, 2);
        assert_eq!(seed_summary.skipped_count, 0);
    }

    #[test]
    fn optimization_targets_are_ranked_and_bounded() {
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.25,
            sliver_aspect_ratio: 8.0,
            max_quality_recovery_seed_candidates: 3,
            ..TetCandidateOptions::default()
        };
        let tets = vec![
            TetCandidate {
                tet_id: 0,
                component_id: 0,
                node_ids: [10, 11, 12, 20],
                source_surface_element_id: 0,
                region_ids: Vec::new(),
                volume_m3: 1.0,
                aspect_ratio: 2.0,
                exact_scaled_jacobian: 0.1,
            },
            TetCandidate {
                tet_id: 1,
                component_id: 0,
                node_ids: [10, 11, 13, 21],
                source_surface_element_id: 0,
                region_ids: Vec::new(),
                volume_m3: 1.0,
                aspect_ratio: 10.0,
                exact_scaled_jacobian: 0.4,
            },
            TetCandidate {
                tet_id: 2,
                component_id: 0,
                node_ids: [12, 13, 14, 22],
                source_surface_element_id: 0,
                region_ids: Vec::new(),
                volume_m3: 1.0,
                aspect_ratio: 12.0,
                exact_scaled_jacobian: 0.5,
            },
        ];

        let seed_summary = optimization_target_seed_indices(&tets, &[10, 11, 12, 13, 14], options);

        assert_eq!(seed_summary.indices, vec![0, 1, 2]);
        assert_eq!(seed_summary.total_count, 5);
        assert_eq!(seed_summary.skipped_count, 2);
    }

    #[test]
    fn local_smoothing_candidates_include_bounded_stencil_when_proposed_is_unchanged() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let component = &volume_candidates.components[0];
        let surface_elements = surface
            .elements
            .iter()
            .map(|element| (element.element_id, element))
            .collect::<BTreeMap<_, _>>();
        let tolerance =
            MeshingTolerance::from_bounds(component.bounds_min_m, component.bounds_max_m);
        let classifier =
            ComponentSurfaceClassifier::new(component, &surface, &surface_elements, tolerance)
                .expect("classifier should build");
        let current = [0.5, 0.5, 0.5];

        let candidates =
            local_seed_smoothing_candidate_points(current, current, &classifier, tolerance);

        assert!(!candidates.is_empty());
        assert!(candidates
            .iter()
            .all(|candidate| classifier.contains_point(*candidate)));
        assert!(candidates
            .iter()
            .all(|candidate| !tolerance.point_nearly_equal(*candidate, current, 1.0)));
        assert!(candidates.iter().any(|candidate| {
            let changed_axes = candidate
                .iter()
                .zip(current.iter())
                .filter(|(left, right)| (*left - *right).abs() > tolerance.absolute_m)
                .count();
            changed_axes >= 2
        }));
    }

    #[test]
    fn cavity_reconnection_acceptance_tracks_strict_quality_improvement() {
        assert!(cavity_reconnection_improves_quality(1, 0.05, 2, 0.10));
        assert!(cavity_reconnection_improves_quality(2, 0.12, 2, 0.10));
        assert!(!cavity_reconnection_improves_quality(2, 0.10, 2, 0.10));
        assert!(!cavity_reconnection_improves_quality(2, 0.09, 2, 0.10));
        assert!(!cavity_reconnection_improves_quality(3, 0.20, 2, 0.10));
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
        assert_eq!(candidates.recovery.optimization_pass_count, 0);
        assert_eq!(candidates.recovery.smoothed_point_count, 0);
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn dense_components_accept_requested_refinement_without_global_insertion() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let mut requested_refinement_points = [[0.0; 3]; 16];
        requested_refinement_points[0] = [0.25, 0.25, 0.25];

        let candidates = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                interior_target_size_m: Some(2.0),
                requested_refinement_points,
                requested_refinement_point_count: 1,
                max_interior_seed_points: 2,
                max_refinement_passes: 1,
                max_global_insertion_points: 4,
                allow_fan_fallback: false,
                ..TetCandidateOptions::default()
            },
        )
        .expect("dense component should retain quality-safe requested refinement");

        assert_eq!(candidates.recovery.insertion_component_count, 1);
        assert_eq!(candidates.recovery.fan_fallback_component_count, 0);
        assert_eq!(candidates.recovery.requested_refinement_point_count, 1);
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_candidate_count,
            1
        );
        assert_eq!(
            candidates
                .recovery
                .accepted_requested_refinement_point_count,
            1
        );
        assert_eq!(
            candidates.accepted_requested_refinement_sample_indices,
            vec![0]
        );
        assert!(candidates
            .dropped_requested_refinement_sample_indices
            .is_empty());
        assert!(candidates
            .interior_seed_points
            .iter()
            .any(|point| distance_squared(*point, requested_refinement_points[0]) <= 1.0e-24));
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
    fn thin_sweep_pairs_cap_triangles_into_prisms() {
        let topology =
            extract_source_topology(&thin_cube_geometry()).expect("topology should extract");
        let surface = discretize_topology_surfaces(
            &topology,
            SurfaceDiscretizationOptions {
                max_curve_segments_per_edge: 8,
                ..SurfaceDiscretizationOptions::default()
            },
        )
        .expect("surface should discretize");
        let volume_candidates =
            prepare_volume_candidates(&surface, VolumeCandidateOptions::default())
                .expect("volume candidates should prepare");
        let component = &volume_candidates.components[0];
        let surface_nodes = surface
            .nodes
            .iter()
            .map(|node| (node.node_id, node.coordinates_m))
            .collect::<BTreeMap<_, _>>();
        let surface_elements = surface
            .elements
            .iter()
            .map(|element| (element.element_id, element))
            .collect::<BTreeMap<_, _>>();
        let tolerance =
            MeshingTolerance::from_bounds(component.bounds_min_m, component.bounds_max_m);
        let options = TetCandidateOptions {
            dense_recovery_layer_count: 1,
            ..TetCandidateOptions::default()
        };
        let mut tets = Vec::new();

        let status = append_sweep_tets(
            component,
            &[],
            &[],
            &surface_nodes,
            &surface_elements,
            options,
            tolerance,
            &mut tets,
        )
        .expect("sweep should evaluate");

        assert!(
            status.is_some(),
            "thin sweep should match paired cap triangles"
        );
        assert!(!tets.is_empty());
        assert!(tets.iter().all(|tet| tet.exact_scaled_jacobian > 0.0));
        let total_volume = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((total_volume - component.volume_m3).abs() <= component.volume_m3 * 1.0e-9);
    }

    #[test]
    fn sweep_axis_uses_projected_cap_area_when_cap_interiors_differ() {
        let surface_nodes = BTreeMap::from([
            (0, [0.0, 0.0, 0.0]),
            (1, [1.0, 0.0, 0.0]),
            (2, [1.0, 1.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
            (4, [0.5, 0.5, 0.0]),
            (5, [0.0, 0.0, 0.2]),
            (6, [1.0, 0.0, 0.2]),
            (7, [1.0, 1.0, 0.2]),
            (8, [0.0, 1.0, 0.2]),
            (9, [0.45, 0.55, 0.2]),
        ]);
        let elements = vec![
            sweep_test_surface_element(0, [0, 1, 4]),
            sweep_test_surface_element(1, [1, 2, 4]),
            sweep_test_surface_element(2, [2, 3, 4]),
            sweep_test_surface_element(3, [3, 0, 4]),
            sweep_test_surface_element(4, [5, 9, 6]),
            sweep_test_surface_element(5, [6, 9, 7]),
            sweep_test_surface_element(6, [7, 9, 8]),
            sweep_test_surface_element(7, [8, 9, 5]),
            sweep_test_surface_element(8, [0, 5, 6]),
            sweep_test_surface_element(9, [0, 6, 1]),
            sweep_test_surface_element(10, [1, 6, 7]),
            sweep_test_surface_element(11, [1, 7, 2]),
            sweep_test_surface_element(12, [2, 7, 8]),
            sweep_test_surface_element(13, [2, 8, 3]),
            sweep_test_surface_element(14, [3, 8, 5]),
            sweep_test_surface_element(15, [3, 5, 0]),
        ];
        let surface_elements = elements
            .iter()
            .map(|element| (element.element_id, element))
            .collect::<BTreeMap<_, _>>();
        let component = VolumeCandidateComponent {
            component_id: 0,
            surface_element_ids: elements.iter().map(|element| element.element_id).collect(),
            source_face_ids: Vec::new(),
            node_ids: surface_nodes.keys().copied().collect(),
            region_ids: Vec::new(),
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [1.0, 1.0, 0.2],
            surface_area_m2: 2.8,
            signed_volume_m3: 0.2,
            volume_m3: 0.2,
        };

        let axis = sweep_axis(
            &component,
            &surface_nodes,
            &surface_elements,
            MeshingTolerance::from_bounds(component.bounds_min_m, component.bounds_max_m),
        )
        .expect("sweep axis should evaluate");

        assert_eq!(axis, Some(2));
    }

    fn sweep_test_surface_element(element_id: u32, node_ids: [u32; 3]) -> SurfaceElement {
        SurfaceElement {
            element_id,
            source_face_id: element_id,
            cad_face_id: None,
            source_edge_ids: [crate::INTERNAL_SOURCE_EDGE_ID; 3],
            node_ids,
            parametric_node_uv: [[0.0, 0.0]; 3],
            max_projection_error_m: 0.0,
            region_ids: Vec::new(),
            area_m2: 1.0,
            unit_normal: [0.0, 0.0, 1.0],
        }
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
    fn three_tet_edge_reconnection_repairs_long_edge_slivers() {
        let root_three = 3.0_f64.sqrt();
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.0]),
            (1, [0.0, 0.0, 2.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [-0.5, root_three * 0.5, 0.0]),
            (4, [-0.5, -root_three * 0.5, 0.0]),
        ]);
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.5,
            ..TetCandidateOptions::default()
        };
        let tets = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]]
            .into_iter()
            .map(|node_ids| {
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            tets.iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            3
        );

        let edge_adjacency = tet_edge_adjacency(&tets);
        let (reconnected_indices, candidates, quality_gain_only) =
            best_three_tet_edge_reconnection(0, &tets, &edge_adjacency, &node_points, options)
                .expect("reconnection should evaluate")
                .expect("three-tet edge reconnection should be available");

        assert_eq!(reconnected_indices, vec![0, 1, 2]);
        assert!(!quality_gain_only);
        assert_eq!(candidates.len(), 2);
        assert_eq!(
            candidates
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            0
        );
        let original_volume = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        let candidate_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((candidate_volume - original_volume).abs() < 1.0e-12);
    }

    #[test]
    fn edge_star_untangling_reconnects_near_singular_edge_star() {
        let root_three = 3.0_f64.sqrt();
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.0]),
            (1, [0.0, 0.0, 2.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [-0.5, root_three * 0.5, 0.0]),
            (4, [-0.5, -root_three * 0.5, 0.0]),
        ]);
        let options = TetCandidateOptions {
            min_scaled_jacobian: 2.0,
            ..TetCandidateOptions::default()
        };
        let tets = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]]
            .into_iter()
            .map(|node_ids| {
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        let threshold = untangling_exact_quality_threshold(options);
        assert!(count_tets_below_exact_quality(tets.iter(), threshold) > 0);
        let edge_adjacency = tet_edge_adjacency(&tets);

        let (indices, candidates) =
            best_edge_star_untangling(0, &tets, &edge_adjacency, &node_points, threshold, options)
                .expect("edge-star untangling should evaluate")
                .expect("edge-star untangling should be available");

        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(
            count_tets_below_exact_quality(candidates.iter(), threshold),
            0
        );
        assert!(
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian)
                <= count_exact_quality_violations(tets.iter(), options.min_scaled_jacobian)
        );
    }

    #[test]
    fn untangling_reconnects_edge_star_when_no_seed_relocation_is_available() {
        let root_three = 3.0_f64.sqrt();
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.0]),
            (1, [0.0, 0.0, 2.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [-0.5, root_three * 0.5, 0.0]),
            (4, [-0.5, -root_three * 0.5, 0.0]),
        ]);
        let options = TetCandidateOptions {
            min_scaled_jacobian: 2.0,
            max_refinement_passes: 1,
            ..TetCandidateOptions::default()
        };
        let mut tets = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]]
            .into_iter()
            .map(|node_ids| {
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        let mut nodes = node_points
            .iter()
            .map(|(node_id, coordinates_m)| TetCandidateNode {
                node_id: *node_id,
                coordinates_m: *coordinates_m,
                source: TetCandidateNodeSource::Surface,
            })
            .collect::<Vec<_>>();
        let threshold = untangling_exact_quality_threshold(options);
        let initial_near_singular_count = count_tets_below_exact_quality(tets.iter(), threshold);
        assert!(initial_near_singular_count > 0);
        let mut interior_seed_points = Vec::new();

        let summary =
            untangle_near_singular_tets(&mut nodes, &mut tets, &mut interior_seed_points, options)
                .expect("untangling should evaluate");

        assert_eq!(summary.pass_count, 1);
        assert_eq!(
            summary.initial_near_singular_count,
            initial_near_singular_count
        );
        assert_eq!(summary.final_near_singular_count, 0);
        assert_eq!(summary.relocated_seed_count, 0);
        assert_eq!(summary.reconnected_edge_star_count, 1);
        assert_eq!(count_tets_below_exact_quality(tets.iter(), threshold), 0);
        assert!(interior_seed_points.is_empty());
    }

    #[test]
    fn multi_tet_edge_reconnection_repairs_larger_edge_stars() {
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.0]),
            (1, [0.0, 0.0, 2.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
            (4, [-1.0, 0.0, 0.0]),
            (5, [0.0, -1.0, 0.0]),
        ]);
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.5,
            ..TetCandidateOptions::default()
        };
        let tets = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 2]]
            .into_iter()
            .map(|node_ids| {
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            tets.iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            4
        );

        let edge_adjacency = tet_edge_adjacency(&tets);
        let (reconnected_indices, candidates, quality_gain_only) =
            best_multi_tet_edge_reconnection(0, &tets, &edge_adjacency, &node_points, options)
                .expect("reconnection should evaluate")
                .expect("multi-tet edge reconnection should be available");

        assert_eq!(reconnected_indices, vec![0, 1, 2, 3]);
        assert!(!quality_gain_only);
        assert_eq!(candidates.len(), 4);
        assert_eq!(
            candidates
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            0
        );
        let original_volume = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        let candidate_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((candidate_volume - original_volume).abs() < 1.0e-12);
    }

    #[test]
    fn multi_tet_edge_reconnection_repairs_nine_tet_edge_star() {
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.641306075584024]),
            (1, [0.0, 0.0, 2.641306075584024]),
            (2, [1.672221732238, 0.0, -0.08728527497905242]),
            (
                3,
                [1.6100567452868202, 1.3509980211446626, -0.05592238391241849],
            ),
            (
                4,
                [
                    0.12295201980691002,
                    0.6972955546162051,
                    -0.01706108461239957,
                ],
            ),
            (
                5,
                [-0.819520110261051, 1.4194504687965892, -0.07777184361504533],
            ),
            (
                6,
                [-1.209384853099018, 0.4401800883004466, 0.08230867573851615],
            ),
            (
                7,
                [
                    -1.823022278068695,
                    -0.6635258456211687,
                    -0.026744827538847306,
                ],
            ),
            (
                8,
                [
                    -0.7593990888507061,
                    -1.3153178051109335,
                    -0.018458105000630897,
                ],
            ),
            (
                9,
                [
                    0.22175055126931031,
                    -1.2576098699038472,
                    -0.006673065389847666,
                ],
            ),
            (
                10,
                [
                    0.8362045922769009,
                    -0.7016589649682959,
                    -0.05557397025847204,
                ],
            ),
        ]);
        let ring_count = 9_usize;
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.15,
            ..TetCandidateOptions::default()
        };
        let tets = (0..ring_count)
            .map(|index| {
                let node_ids = [
                    0,
                    1,
                    (index + 2) as u32,
                    ((index + 1) % ring_count + 2) as u32,
                ];
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            tets.iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            3
        );

        let edge_adjacency = tet_edge_adjacency(&tets);
        let (reconnected_indices, candidates, quality_gain_only) =
            best_multi_tet_edge_reconnection(0, &tets, &edge_adjacency, &node_points, options)
                .expect("reconnection should evaluate")
                .expect("nine-tet edge-star reconnection should be available");

        assert_eq!(reconnected_indices, (0..ring_count).collect::<Vec<_>>());
        assert!(!quality_gain_only);
        assert_eq!(candidates.len(), (ring_count - 2) * 2);
        assert_eq!(
            candidates
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            0
        );
        let original_volume = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        let candidate_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((candidate_volume - original_volume).abs() < 1.0e-12);
    }

    #[test]
    fn componentized_edge_reconnection_repairs_simple_ring_inside_disconnected_star() {
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.0]),
            (1, [0.0, 0.0, 2.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
            (4, [-1.0, 0.0, 0.0]),
            (5, [0.0, -1.0, 0.0]),
            (6, [3.0, 0.0, 0.0]),
            (7, [3.0, 1.0, 0.0]),
            (8, [2.2, 0.5, 0.0]),
        ]);
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.5,
            ..TetCandidateOptions::default()
        };
        let tets = [
            [0, 1, 2, 3],
            [0, 1, 3, 4],
            [0, 1, 4, 5],
            [0, 1, 5, 2],
            [0, 1, 6, 7],
            [0, 1, 7, 8],
            [0, 1, 8, 6],
        ]
        .into_iter()
        .map(|node_ids| {
            let points = node_ids.map(|node_id| node_points[&node_id]);
            raw_candidate_tet(0, 0, &[], node_ids, points, options)
                .expect("fixture tet should be valid")
        })
        .collect::<Vec<_>>();
        let edge_adjacency = tet_edge_adjacency(&tets);
        assert_eq!(edge_adjacency[&[0, 1]].len(), 7);
        assert!(multi_tet_edge_reconnection_candidates(
            &edge_adjacency[&[0, 1]],
            [0, 1],
            &tets,
            &node_points,
            options,
        )
        .expect("whole disconnected ring should evaluate")
        .is_none());

        let (reconnected_indices, candidates, quality_gain_only) =
            best_componentized_edge_reconnection(0, &tets, &edge_adjacency, &node_points, options)
                .expect("componentized reconnection should evaluate")
                .expect("simple component ring should reconnect");

        assert_eq!(reconnected_indices, vec![0, 1, 2, 3]);
        assert!(!quality_gain_only);
        assert_eq!(
            candidates
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            0
        );
        let original_volume = reconnected_indices
            .iter()
            .map(|index| tets[*index].volume_m3)
            .sum::<f64>();
        let candidate_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((candidate_volume - original_volume).abs() < 1.0e-12);
    }

    #[test]
    fn componentized_edge_reconnection_repairs_simple_cycle_inside_branched_star() {
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.0]),
            (1, [0.0, 0.0, 2.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
            (4, [-1.0, 0.0, 0.0]),
            (5, [0.0, -1.0, 0.0]),
            (6, [0.8, 1.8, 0.0]),
            (7, [-0.8, 1.8, 0.0]),
        ]);
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.5,
            ..TetCandidateOptions::default()
        };
        let tets = [
            [0, 1, 2, 3],
            [0, 1, 3, 4],
            [0, 1, 4, 5],
            [0, 1, 5, 2],
            [0, 1, 3, 6],
            [0, 1, 6, 7],
            [0, 1, 7, 3],
        ]
        .into_iter()
        .map(|node_ids| {
            let points = node_ids.map(|node_id| node_points[&node_id]);
            raw_candidate_tet(0, 0, &[], node_ids, points, options)
                .expect("fixture tet should be valid")
        })
        .collect::<Vec<_>>();
        let edge_adjacency = tet_edge_adjacency(&tets);
        assert_eq!(edge_adjacency[&[0, 1]].len(), 7);
        assert!(multi_tet_edge_reconnection_candidates(
            &edge_adjacency[&[0, 1]],
            [0, 1],
            &tets,
            &node_points,
            options,
        )
        .expect("whole branched ring should evaluate")
        .is_none());

        let (reconnected_indices, candidates, quality_gain_only) =
            best_componentized_edge_reconnection(0, &tets, &edge_adjacency, &node_points, options)
                .expect("componentized reconnection should evaluate")
                .expect("simple cycle should reconnect");

        assert_eq!(reconnected_indices, vec![0, 1, 2, 3]);
        assert!(!quality_gain_only);
        assert_eq!(
            candidates
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            0
        );
        let original_volume = reconnected_indices
            .iter()
            .map(|index| tets[*index].volume_m3)
            .sum::<f64>();
        let candidate_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((candidate_volume - original_volume).abs() < 1.0e-12);
    }

    #[test]
    fn componentized_edge_reconnection_repairs_cycle_with_duplicated_target_edge() {
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.0]),
            (1, [0.0, 0.0, 2.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
            (4, [-1.0, 0.0, 0.0]),
            (5, [0.0, -1.0, 0.0]),
        ]);
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.5,
            ..TetCandidateOptions::default()
        };
        let tets = [
            [0, 1, 2, 3],
            [0, 1, 3, 4],
            [0, 1, 4, 5],
            [0, 1, 5, 2],
            [0, 1, 2, 3],
        ]
        .into_iter()
        .map(|node_ids| {
            let points = node_ids.map(|node_id| node_points[&node_id]);
            raw_candidate_tet(0, 0, &[], node_ids, points, options)
                .expect("fixture tet should be valid")
        })
        .collect::<Vec<_>>();
        let edge_adjacency = tet_edge_adjacency(&tets);
        assert_eq!(edge_adjacency[&[0, 1]].len(), 5);
        assert!(multi_tet_edge_reconnection_candidates(
            &edge_adjacency[&[0, 1]],
            [0, 1],
            &tets,
            &node_points,
            options,
        )
        .expect("whole non-manifold ring should evaluate")
        .is_none());

        let (reconnected_indices, candidates, quality_gain_only) =
            best_componentized_edge_reconnection(0, &tets, &edge_adjacency, &node_points, options)
                .expect("componentized reconnection should evaluate")
                .expect("simple cycle with duplicated target edge should reconnect");

        assert_eq!(reconnected_indices, vec![0, 1, 2, 3]);
        assert!(!quality_gain_only);
        assert_eq!(
            candidates
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            0
        );
        let original_volume = reconnected_indices
            .iter()
            .map(|index| tets[*index].volume_m3)
            .sum::<f64>();
        let candidate_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((candidate_volume - original_volume).abs() < 1.0e-12);
    }

    #[test]
    fn remaining_exact_quality_violations_are_classified_by_local_topology() {
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, -2.0]),
            (1, [0.0, 0.0, 2.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
            (4, [-1.0, 0.0, 0.0]),
            (5, [0.0, -1.0, 0.0]),
        ]);
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.5,
            ..TetCandidateOptions::default()
        };
        let tets = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 2]]
            .into_iter()
            .map(|node_ids| {
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        let nodes = node_points
            .iter()
            .map(|(node_id, coordinates_m)| TetCandidateNode {
                node_id: *node_id,
                coordinates_m: *coordinates_m,
                source: if *node_id == 0 {
                    TetCandidateNodeSource::InteriorSeed
                } else {
                    TetCandidateNodeSource::Surface
                },
            })
            .collect::<Vec<_>>();

        let counts = remaining_exact_quality_violation_counts(&nodes, &tets, options);

        assert_eq!(counts.total_count, 4);
        assert_eq!(counts.general_cavity_count, 0);
        assert_eq!(counts.boundary_adjacent_count, 4);
        assert_eq!(counts.interior_seed_count, 4);
        assert_eq!(counts.edge_star_count, 4);
    }

    #[test]
    fn missing_face_topology_groups_edge_connected_clusters() {
        let missing_faces = [[0, 1, 2], [0, 1, 3], [3, 4, 5]];
        let face_adjacency = BTreeMap::from([
            ([0, 1, 2], vec![0]),
            ([0, 1, 3], vec![0, 1]),
            ([3, 4, 5], vec![2, 3]),
        ]);

        let topology = diagnostic_missing_face_topology(&missing_faces, &face_adjacency, 4);

        assert_eq!(topology, (1, 2, 2, 2, 2, 1, 4));
    }

    #[test]
    fn boundary_split_diagnostic_classifies_unfillable_split_cavity() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 100.0,
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let outer_tet = raw_candidate_tet(0, 0, &[], [0, 1, 2, 3], points, options)
            .expect("outer tet should be valid");
        let split_point = [0.04, 0.04, 0.04];
        let split_tets = centroid_split_tets(&outer_tet, 4, split_point, points, options);
        let node_points = BTreeMap::from([
            (0, points[0]),
            (1, points[1]),
            (2, points[2]),
            (3, points[3]),
            (4, split_point),
        ]);
        let face_adjacency = tet_face_adjacency(&split_tets);

        let reasons = diagnostic_small_cavity_boundary_split_rejection_reasons(
            0,
            &split_tets,
            &face_adjacency,
            &node_points,
            5,
            options,
        )
        .expect("boundary split diagnostic should evaluate");

        assert_eq!(
            reasons,
            (
                "one_ring_boundary_split_no_improvement",
                "face_closure_boundary_split_no_improvement"
            )
        );
    }

    #[test]
    fn exact_quality_repair_pass_limit_is_bounded_above_refinement_passes() {
        assert_eq!(
            exact_quality_repair_pass_limit(TetCandidateOptions::default()),
            8
        );
        assert_eq!(
            exact_quality_repair_pass_limit(TetCandidateOptions {
                max_refinement_passes: 3,
                ..TetCandidateOptions::default()
            }),
            24
        );
        assert_eq!(
            exact_quality_repair_pass_limit(TetCandidateOptions {
                max_refinement_passes: 100,
                ..TetCandidateOptions::default()
            }),
            32
        );
    }

    #[test]
    fn interior_seed_repair_star_size_scopes_are_bounded() {
        assert!(interior_seed_collapse_scope_matches(
            InteriorSeedCollapseScope::FourTetOnly,
            4
        ));
        assert!(!interior_seed_collapse_scope_matches(
            InteriorSeedCollapseScope::FourTetOnly,
            5
        ));
        assert!(interior_seed_collapse_scope_matches(
            InteriorSeedCollapseScope::LargerStarsOnly,
            MAX_INTERIOR_SEED_COLLAPSE_STAR_SIZE
        ));
        assert!(!interior_seed_collapse_scope_matches(
            InteriorSeedCollapseScope::LargerStarsOnly,
            MAX_INTERIOR_SEED_COLLAPSE_STAR_SIZE + 1
        ));
        assert!(interior_seed_relocation_scope_matches(
            MAX_INTERIOR_SEED_COLLAPSE_STAR_SIZE + 1
        ));
        assert!(interior_seed_relocation_scope_matches(
            MAX_INTERIOR_SEED_RELOCATION_STAR_SIZE
        ));
        assert!(!interior_seed_relocation_scope_matches(
            MAX_INTERIOR_SEED_RELOCATION_STAR_SIZE + 1
        ));
    }

    #[test]
    fn repair_collapses_bad_interior_seed_star_when_quality_improves() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 100.0,
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let outer_tet = raw_candidate_tet(0, 0, &[], [0, 1, 2, 3], points, options)
            .expect("outer tet should be valid");
        let split_point = [0.04, 0.04, 0.04];
        let mut split_tets = centroid_split_tets(&outer_tet, 4, split_point, points, options);
        assert_eq!(split_tets.len(), 4);
        assert!(split_tets
            .iter()
            .any(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian));
        let split_volume = split_tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((split_volume - outer_tet.volume_m3).abs() < 1.0e-12);

        let mut nodes = vec![
            TetCandidateNode {
                node_id: 0,
                coordinates_m: points[0],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 1,
                coordinates_m: points[1],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 2,
                coordinates_m: points[2],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 3,
                coordinates_m: points[3],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 4,
                coordinates_m: split_point,
                source: TetCandidateNodeSource::InteriorSeed,
            },
        ];
        let mut interior_seed_points = vec![split_point];
        let mut next_node_id = 5;

        let repair = repair_exact_quality_tets_once(
            &mut nodes,
            &mut split_tets,
            &mut interior_seed_points,
            &mut next_node_id,
            options,
        )
        .expect("repair should evaluate");

        assert!(repair.changed);
        assert_eq!(repair.seed_star_collapse_count, 1);
        assert_eq!(repair.reconnected_cavity_count, 0);
        assert_eq!(repair.split_cavity_count, 0);
        assert_eq!(split_tets.len(), 1);
        assert!(
            split_tets[0].exact_scaled_jacobian >= options.min_scaled_jacobian,
            "collapsed tet should clear the exact-quality gate"
        );
        assert!((split_tets[0].volume_m3 - outer_tet.volume_m3).abs() < 1.0e-12);
        assert!(!nodes.iter().any(|node| node.node_id == 4));
        assert!(interior_seed_points.is_empty());
    }

    #[test]
    fn repair_collapses_bad_boundary_recovery_seed_star_when_quality_improves() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 100.0,
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let outer_tet = raw_candidate_tet(0, 0, &[], [0, 1, 2, 3], points, options)
            .expect("outer tet should be valid");
        let split_point = [0.04, 0.04, 0.04];
        let mut split_tets = centroid_split_tets(&outer_tet, 4, split_point, points, options);
        let mut nodes = vec![
            TetCandidateNode {
                node_id: 0,
                coordinates_m: points[0],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 1,
                coordinates_m: points[1],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 2,
                coordinates_m: points[2],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 3,
                coordinates_m: points[3],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 4,
                coordinates_m: split_point,
                source: TetCandidateNodeSource::BoundaryRecovery,
            },
        ];
        let mut interior_seed_points = Vec::new();
        let mut next_node_id = 5;

        let repair = repair_exact_quality_tets_once(
            &mut nodes,
            &mut split_tets,
            &mut interior_seed_points,
            &mut next_node_id,
            options,
        )
        .expect("repair should evaluate");

        assert!(repair.changed);
        assert_eq!(repair.seed_star_collapse_count, 1);
        assert_eq!(split_tets.len(), 1);
        assert!(split_tets[0].exact_scaled_jacobian >= options.min_scaled_jacobian);
        assert!(!nodes.iter().any(|node| node.node_id == 4));
        assert!(interior_seed_points.is_empty());
    }

    #[test]
    fn larger_interior_seed_star_collapse_reconstructs_boundary_only_cavity() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 100.0,
            min_scaled_jacobian: 0.35,
            ..TetCandidateOptions::default()
        };
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, 1.2]),
            (1, [0.0, 0.0, -1.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [0.0, 1.1, 0.0]),
            (4, [-1.0, 0.0, 0.0]),
            (5, [0.0, -1.0, 0.0]),
            (6, [0.78, 0.0, 0.0]),
        ]);
        let tet_node_ids = [
            [6, 0, 2, 3],
            [6, 0, 3, 4],
            [6, 0, 4, 5],
            [6, 0, 5, 2],
            [6, 1, 3, 2],
            [6, 1, 4, 3],
            [6, 1, 5, 4],
            [6, 1, 2, 5],
        ];
        let tets = tet_node_ids
            .into_iter()
            .map(|node_ids| {
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        let original_below_count = tets
            .iter()
            .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
            .count();
        assert!(original_below_count > 0);
        let original_volume = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        let node_adjacency = tet_node_adjacency(&tets);
        let interior_node_ids = BTreeSet::from([6]);
        let (indices, candidates) = best_interior_seed_node_collapse(
            0,
            &tets,
            &node_adjacency,
            &interior_node_ids,
            &node_points,
            InteriorSeedCollapseScope::LargerStarsOnly,
            options,
        )
        .expect("collapse should evaluate")
        .expect("larger seed-star collapse should be available");

        assert_eq!(indices.len(), 8);
        assert!(
            candidates.len() < tets.len(),
            "collapse should replace the interior star with fewer boundary-only Tets"
        );
        assert!(candidates.iter().all(|tet| !tet.node_ids.contains(&6)));
        assert_eq!(
            candidates
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            0
        );
        let collapsed_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((collapsed_volume - original_volume).abs() < 1.0e-12);
    }

    #[test]
    fn constrained_refill_replaces_low_quality_interior_seed_star() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 0.0,
            ..TetCandidateOptions::default()
        };
        let node_points = BTreeMap::from([
            (0, [1.0, 0.0, 0.0]),
            (1, [0.0, 1.0, 0.0]),
            (2, [-1.0, 0.0, 0.0]),
            (3, [0.0, -1.0, 0.0]),
            (4, [0.0, 0.0, 1.0]),
            (5, [0.0, 0.0, -1.0]),
            (6, [0.0, 0.0, 0.0]),
        ]);
        let boundary_faces = [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ];
        let tets = boundary_faces
            .into_iter()
            .enumerate()
            .map(|(tet_id, face)| TetCandidate {
                tet_id: tet_id as u32,
                component_id: 0,
                node_ids: [face[0], face[1], face[2], 6],
                source_surface_element_id: 0,
                region_ids: vec!["body".to_string()],
                volume_m3: 1.0 / 6.0,
                aspect_ratio: 1.0,
                exact_scaled_jacobian: -0.1,
            })
            .collect::<Vec<_>>();
        let node_adjacency = tet_node_adjacency(&tets);
        let interior_node_ids = BTreeSet::from([6]);
        let boundary_node_ids = (0..=5).collect::<BTreeSet<_>>();
        let interior_candidates = constrained_seed_star_refill_interior_candidates(
            &tets,
            0,
            &(0..8).collect::<Vec<_>>(),
            &boundary_node_ids,
            &node_points,
        )
        .expect("interior candidate generation should evaluate");
        assert!(interior_candidates.len() >= 2);
        assert_eq!(interior_candidates[0].node_id, 6);
        assert!(
            interior_candidates
                .iter()
                .any(|candidate| distance(candidate.coordinates_m, [0.25, 0.25, 0.25]) < 1.0e-12),
            "target tet centroid should be included as a bounded interior candidate"
        );
        let reason = diagnostic_constrained_seed_star_refill_rejection_reason(
            0,
            &tets,
            &node_adjacency,
            &interior_node_ids,
            &node_points,
            options,
        )
        .expect("diagnostic should evaluate");
        assert_eq!(reason, "constrained_seed_star_refill_reconnectable");

        let (indices, candidates, inserted_nodes) = best_constrained_interior_seed_star_refill(
            0,
            &tets,
            &node_adjacency,
            &interior_node_ids,
            &node_points,
            options,
        )
        .expect("constrained refill should evaluate")
        .expect("closed seed star should be refilled from boundary nodes");

        assert_eq!(indices, (0..8).collect::<Vec<_>>());
        assert!(inserted_nodes
            .iter()
            .all(|node| !node_points.contains_key(&node.node_id)));
        assert!(!candidates.is_empty());
        assert!(candidates
            .iter()
            .all(|candidate| candidate.exact_scaled_jacobian >= options.min_scaled_jacobian));
        assert_eq!(
            boundary_faces_from_tets(&candidates),
            boundary_faces_from_tets(&tets)
        );
    }

    #[test]
    fn constrained_refill_repairs_component_of_disconnected_interior_seed_star() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 0.0,
            ..TetCandidateOptions::default()
        };
        let node_points = BTreeMap::from([
            (0, [1.0, 0.0, 0.0]),
            (1, [0.0, 1.0, 0.0]),
            (2, [-1.0, 0.0, 0.0]),
            (3, [0.0, -1.0, 0.0]),
            (4, [0.0, 0.0, 1.0]),
            (5, [0.0, 0.0, -1.0]),
            (6, [0.0, 0.0, 0.0]),
            (7, [4.0, 0.0, 0.0]),
            (8, [5.0, 0.0, 0.0]),
            (9, [4.0, 1.0, 0.0]),
            (10, [4.0, 0.0, 1.0]),
        ]);
        let first_component_faces = [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ];
        let second_component_faces = [[7, 8, 9], [8, 7, 10], [7, 9, 10], [9, 8, 10]];
        let mut tets = first_component_faces
            .into_iter()
            .enumerate()
            .map(|(tet_id, face)| TetCandidate {
                tet_id: tet_id as u32,
                component_id: 0,
                node_ids: [face[0], face[1], face[2], 6],
                source_surface_element_id: 0,
                region_ids: vec!["body".to_string()],
                volume_m3: 1.0 / 6.0,
                aspect_ratio: 1.0,
                exact_scaled_jacobian: -0.1,
            })
            .collect::<Vec<_>>();
        tets.extend(
            second_component_faces
                .into_iter()
                .enumerate()
                .map(|(offset, face)| TetCandidate {
                    tet_id: (first_component_faces.len() + offset) as u32,
                    component_id: 0,
                    node_ids: [face[0], face[1], face[2], 6],
                    source_surface_element_id: 0,
                    region_ids: vec!["body".to_string()],
                    volume_m3: 1.0,
                    aspect_ratio: 1.0,
                    exact_scaled_jacobian: 0.8,
                }),
        );
        let node_adjacency = tet_node_adjacency(&tets);
        let interior_node_ids = BTreeSet::from([6]);

        let (indices, candidates, inserted_nodes) = best_constrained_interior_seed_star_refill(
            0,
            &tets,
            &node_adjacency,
            &interior_node_ids,
            &node_points,
            options,
        )
        .expect("componentized constrained refill should evaluate")
        .expect("valid seed-star component should refill");

        assert_eq!(indices, (0..8).collect::<Vec<_>>());
        assert!(inserted_nodes
            .iter()
            .all(|node| !node_points.contains_key(&node.node_id)));
        assert!(candidates
            .iter()
            .all(|candidate| candidate.exact_scaled_jacobian >= options.min_scaled_jacobian));
        assert_eq!(
            boundary_faces_from_tets(&candidates),
            boundary_faces_from_tets(&tets[..8])
        );
    }

    #[test]
    fn constrained_refill_reports_synthetic_inserted_seed_node() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 0.15,
            ..TetCandidateOptions::default()
        };
        let node_points = BTreeMap::from([
            (0, [1.0, 0.0, 0.0]),
            (1, [0.0, 1.0, 0.0]),
            (2, [-1.0, 0.0, 0.0]),
            (3, [0.0, -1.0, 0.0]),
            (4, [0.0, 0.0, 1.0]),
            (5, [0.0, 0.0, -1.0]),
            (6, [0.92, 0.0, 0.0]),
        ]);
        let boundary_faces = [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ];
        let tets = boundary_faces
            .into_iter()
            .enumerate()
            .map(|(tet_id, face)| TetCandidate {
                tet_id: tet_id as u32,
                component_id: 0,
                node_ids: [face[0], face[1], face[2], 6],
                source_surface_element_id: 0,
                region_ids: vec!["body".to_string()],
                volume_m3: 1.0 / 6.0,
                aspect_ratio: 1.0,
                exact_scaled_jacobian: -0.1,
            })
            .collect::<Vec<_>>();
        let node_adjacency = tet_node_adjacency(&tets);
        let interior_node_ids = BTreeSet::from([6]);

        let (_, candidates, inserted_nodes) = best_constrained_interior_seed_star_refill(
            0,
            &tets,
            &node_adjacency,
            &interior_node_ids,
            &node_points,
            options,
        )
        .expect("constrained refill should evaluate")
        .expect("synthetic inserted seed should repair the star");

        assert_eq!(inserted_nodes.len(), 1);
        let inserted_node_id = inserted_nodes[0].node_id;
        assert!(!node_points.contains_key(&inserted_node_id));
        assert!(candidates
            .iter()
            .all(|candidate| candidate.node_ids.contains(&inserted_node_id)));
        assert!(candidates
            .iter()
            .all(|candidate| candidate.exact_scaled_jacobian >= options.min_scaled_jacobian));
    }

    #[test]
    fn repair_relocates_bad_interior_seed_star_when_quality_improves() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 100.0,
            min_scaled_jacobian: 0.35,
            ..TetCandidateOptions::default()
        };
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, 1.0]),
            (1, [0.0, 0.0, -1.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
            (4, [-1.0, 0.0, 0.0]),
            (5, [0.0, -1.0, 0.0]),
            (6, [0.78, 0.0, 0.0]),
        ]);
        let tet_node_ids = [
            [6, 0, 2, 3],
            [6, 0, 3, 4],
            [6, 0, 4, 5],
            [6, 0, 5, 2],
            [6, 1, 3, 2],
            [6, 1, 4, 3],
            [6, 1, 5, 4],
            [6, 1, 2, 5],
        ];
        let mut tets = tet_node_ids
            .into_iter()
            .map(|node_ids| {
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        let original_below_count = tets
            .iter()
            .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
            .count();
        assert!(
            original_below_count > 0,
            "off-center interior seed should create exact-quality violations"
        );
        let original_volume = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        let node_adjacency = tet_node_adjacency(&tets);
        let interior_node_ids = BTreeSet::from([6]);
        let (node_id, relocated_point, indices, candidates) = best_interior_seed_node_relocation(
            0,
            &tets,
            &node_adjacency,
            &interior_node_ids,
            &node_points,
            options,
        )
        .expect("relocation should evaluate")
        .expect("relocation should be available");

        assert_eq!(node_id, 6);
        assert_eq!(indices.len(), 8);
        assert!(
            distance(relocated_point, [0.0, 0.0, 0.0]) < 1.0e-12,
            "best relocation should move the interior seed to the closed-star centroid"
        );
        assert_eq!(
            candidates
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            0
        );
        let relocated_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((relocated_volume - original_volume).abs() < 1.0e-12);

        let mut nodes = node_points
            .iter()
            .map(|(node_id, coordinates_m)| TetCandidateNode {
                node_id: *node_id,
                coordinates_m: *coordinates_m,
                source: if *node_id == 6 {
                    TetCandidateNodeSource::InteriorSeed
                } else {
                    TetCandidateNodeSource::Surface
                },
            })
            .collect::<Vec<_>>();
        let mut interior_seed_points = vec![node_points[&6]];
        let mut next_node_id = 7;
        let repair = repair_exact_quality_tets_once(
            &mut nodes,
            &mut tets,
            &mut interior_seed_points,
            &mut next_node_id,
            options,
        )
        .expect("repair should evaluate");

        assert!(repair.changed);
        assert_eq!(repair.seed_star_relocation_count, 1);
        assert_eq!(repair.seed_star_collapse_count, 0);
        assert_eq!(repair.reconnected_cavity_count, 0);
        assert_eq!(repair.split_cavity_count, 0);
        assert_eq!(next_node_id, 7);
        assert_eq!(tets.len(), 8);
        assert_eq!(
            tets.iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count(),
            0
        );
        assert!(
            distance(
                nodes
                    .iter()
                    .find(|node| node.node_id == 6)
                    .expect("interior node retained")
                    .coordinates_m,
                [0.0, 0.0, 0.0],
            ) < 1.0e-12
        );
        assert_eq!(interior_seed_points, vec![[0.0, 0.0, 0.0]]);
        let repaired_volume = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((repaired_volume - original_volume).abs() < 1.0e-12);
    }

    #[test]
    fn untangling_relocates_near_singular_interior_seed_star() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 0.35,
            ..TetCandidateOptions::default()
        };
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, 1.0]),
            (1, [0.0, 0.0, -1.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
            (4, [-1.0, 0.0, 0.0]),
            (5, [0.0, -1.0, 0.0]),
            (6, [0.98, 0.0, 0.0]),
        ]);
        let tet_node_ids = [
            [6, 0, 2, 3],
            [6, 0, 3, 4],
            [6, 0, 4, 5],
            [6, 0, 5, 2],
            [6, 1, 3, 2],
            [6, 1, 4, 3],
            [6, 1, 5, 4],
            [6, 1, 2, 5],
        ];
        let mut tets = tet_node_ids
            .into_iter()
            .map(|node_ids| {
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        let threshold = untangling_exact_quality_threshold(options);
        let initial_near_singular_count = count_tets_below_exact_quality(tets.iter(), threshold);
        assert!(
            initial_near_singular_count > 0,
            "fixture should contain a near-singular local star"
        );
        let original_volume = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        let mut nodes = node_points
            .iter()
            .map(|(node_id, coordinates_m)| TetCandidateNode {
                node_id: *node_id,
                coordinates_m: *coordinates_m,
                source: if *node_id == 6 {
                    TetCandidateNodeSource::InteriorSeed
                } else {
                    TetCandidateNodeSource::Surface
                },
            })
            .collect::<Vec<_>>();
        let mut interior_seed_points = vec![node_points[&6]];

        let summary =
            untangle_near_singular_tets(&mut nodes, &mut tets, &mut interior_seed_points, options)
                .expect("untangling should evaluate");

        assert_eq!(summary.pass_count, 1);
        assert_eq!(
            summary.initial_near_singular_count,
            initial_near_singular_count
        );
        assert_eq!(summary.final_near_singular_count, 0);
        assert_eq!(summary.relocated_seed_count, 1);
        assert_eq!(count_tets_below_exact_quality(tets.iter(), threshold), 0);
        assert!(
            distance(
                nodes
                    .iter()
                    .find(|node| node.node_id == 6)
                    .expect("interior node retained")
                    .coordinates_m,
                [0.0, 0.0, 0.0],
            ) < 1.0e-12
        );
        assert_eq!(interior_seed_points, vec![[0.0, 0.0, 0.0]]);
        let untangled_volume = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((untangled_volume - original_volume).abs() < 1.0e-12);
    }

    #[test]
    fn interior_seed_relocation_points_include_bounded_local_stencil() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 100.0,
            min_scaled_jacobian: 0.35,
            ..TetCandidateOptions::default()
        };
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, 1.0]),
            (1, [0.0, 0.0, -1.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [0.0, 1.0, 0.0]),
            (4, [-1.0, 0.0, 0.0]),
            (5, [0.0, -1.0, 0.0]),
            (6, [0.78, 0.0, 0.0]),
        ]);
        let tet_node_ids = [
            [6, 0, 2, 3],
            [6, 0, 3, 4],
            [6, 0, 4, 5],
            [6, 0, 5, 2],
            [6, 1, 3, 2],
            [6, 1, 4, 3],
            [6, 1, 5, 4],
            [6, 1, 2, 5],
        ];
        let tets = tet_node_ids
            .into_iter()
            .map(|node_ids| {
                let points = node_ids.map(|node_id| node_points[&node_id]);
                raw_candidate_tet(0, 0, &[], node_ids, points, options)
                    .expect("fixture tet should be valid")
            })
            .collect::<Vec<_>>();
        let adjacent = (0..tets.len()).collect::<Vec<_>>();

        let points = interior_seed_node_relocation_points(&adjacent, 6, &tets, &node_points)
            .expect("relocation candidates should build");

        assert!(points.len() > 8);
        assert!(points.len() <= 20);
        assert!(points.iter().any(|point| {
            point[0] > node_points[&6][0]
                && (point[1] - node_points[&6][1]).abs() <= f64::EPSILON
                && (point[2] - node_points[&6][2]).abs() <= f64::EPSILON
        }));
        let tolerance = MeshingTolerance::default();
        for (left_index, left) in points.iter().enumerate() {
            for right in points.iter().skip(left_index + 1) {
                assert!(!tolerance.point_nearly_equal(*left, *right, 1.0));
            }
        }
    }

    #[test]
    fn relocated_interior_seed_points_must_clear_star_boundary_shell() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 100.0,
            min_scaled_jacobian: 0.1,
            ..TetCandidateOptions::default()
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let outer_tet = raw_candidate_tet(0, 0, &[], [0, 1, 2, 3], points, options)
            .expect("outer tet should be valid");
        let split_point = [0.25, 0.25, 0.25];
        let split_tets = centroid_split_tets(&outer_tet, 4, split_point, points, options);
        let node_points = BTreeMap::from([
            (0, points[0]),
            (1, points[1]),
            (2, points[2]),
            (3, points[3]),
            (4, split_point),
        ]);
        let adjacent = (0..split_tets.len()).collect::<Vec<_>>();

        assert!(relocated_seed_point_has_star_boundary_clearance(
            &adjacent,
            4,
            split_point,
            &split_tets,
            &node_points,
            MeshingTolerance::default(),
        )
        .expect("clearance should evaluate"));
        assert!(!relocated_seed_point_has_star_boundary_clearance(
            &adjacent,
            4,
            [0.25, 0.25, 0.0],
            &split_tets,
            &node_points,
            MeshingTolerance::default(),
        )
        .expect("clearance should evaluate"));
    }

    #[test]
    fn layered_frustum_split_uses_lowest_aspect_decomposition() {
        let component = VolumeCandidateComponent {
            component_id: 7,
            surface_element_ids: vec![3],
            source_face_ids: vec![2],
            node_ids: vec![0, 1, 2],
            region_ids: vec!["region".to_string()],
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [1.0, 1.0, 1.0],
            surface_area_m2: 1.0,
            signed_volume_m3: 1.0,
            volume_m3: 1.0,
        };
        let element = SurfaceElement {
            element_id: 3,
            source_face_id: 2,
            cad_face_id: None,
            source_edge_ids: [0, 1, 2],
            node_ids: [0, 1, 2],
            parametric_node_uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            max_projection_error_m: 0.0,
            region_ids: vec!["region".to_string()],
            area_m2: 0.5,
            unit_normal: [0.0, 0.0, 1.0],
        };
        let outer_ids = [0, 1, 2];
        let inner_ids = [3, 4, 5];
        let outer_points = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.2, 1.1, 0.0]];
        let inner_points = [[0.1, 0.2, 0.3], [1.8, 0.1, 0.4], [0.4, 1.0, 0.7]];
        let options = TetCandidateOptions {
            max_aspect_ratio: 1.0e6,
            ..TetCandidateOptions::default()
        };
        let expected = (0..6)
            .filter_map(|split_index| {
                layered_frustum_split(
                    &component,
                    &element,
                    outer_ids,
                    inner_ids,
                    outer_points,
                    inner_points,
                    split_index,
                    options,
                )
            })
            .reduce(|best, candidate| {
                if layered_split_is_better(&candidate, &best) {
                    candidate
                } else {
                    best
                }
            })
            .expect("at least one split should be valid");

        let mut tets = Vec::<TetCandidate>::new();
        append_best_layered_frustum_tets(
            &component,
            &element,
            outer_ids,
            inner_ids,
            outer_points,
            inner_points,
            options,
            &mut tets,
        );

        assert_eq!(tets.len(), 3);
        assert!((max_candidate_aspect_ratio(&tets) - expected.max_aspect_ratio).abs() < 1.0e-12);
    }

    #[test]
    fn face_neighbor_cavity_reconnection_repairs_general_split_cavity() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 100.0,
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let outer_tet = raw_candidate_tet(0, 0, &[], [0, 1, 2, 3], points, options)
            .expect("outer tet should be valid");
        let split_point = [0.04, 0.04, 0.04];
        let split_tets = centroid_split_tets(&outer_tet, 4, split_point, points, options);
        assert_eq!(split_tets.len(), 4);
        assert!(split_tets
            .iter()
            .any(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian));
        let node_points = BTreeMap::from([
            (0, points[0]),
            (1, points[1]),
            (2, points[2]),
            (3, points[3]),
            (4, split_point),
        ]);
        let face_adjacency = tet_face_adjacency(&split_tets);

        let (indices, candidates, quality_gain_only) = best_face_neighbor_cavity_reconnection(
            0,
            &split_tets,
            &face_adjacency,
            &node_points,
            options,
        )
        .expect("general cavity reconnection should evaluate")
        .expect("general cavity reconnection should be available");

        assert_eq!(indices.len(), split_tets.len());
        assert!(!quality_gain_only);
        assert_eq!(
            boundary_faces_from_tets(&candidates),
            boundary_faces_from_tets(&split_tets)
        );
        assert!(
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian)
                < count_exact_quality_violations(split_tets.iter(), options.min_scaled_jacobian)
        );
        let original_volume = split_tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        let candidate_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!((candidate_volume - original_volume).abs() < 1.0e-12);
        assert_eq!(candidates.len(), 1);
        assert!((candidates[0].volume_m3 - outer_tet.volume_m3).abs() < 1.0e-12);

        let mut nodes = vec![
            TetCandidateNode {
                node_id: 0,
                coordinates_m: points[0],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 1,
                coordinates_m: points[1],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 2,
                coordinates_m: points[2],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 3,
                coordinates_m: points[3],
                source: TetCandidateNodeSource::Surface,
            },
            TetCandidateNode {
                node_id: 4,
                coordinates_m: split_point,
                source: TetCandidateNodeSource::Surface,
            },
        ];
        let mut repair_tets = split_tets.clone();
        let mut interior_seed_points = Vec::new();
        let mut next_node_id = 5;
        let repair = repair_exact_quality_tets_once(
            &mut nodes,
            &mut repair_tets,
            &mut interior_seed_points,
            &mut next_node_id,
            options,
        )
        .expect("repair should evaluate");

        assert!(repair.changed);
        assert_eq!(repair.reconnected_cavity_count, 1);
        assert_eq!(repair.boundary_adjacent_reconnected_cavity_count, 1);
        assert_eq!(repair.face_neighbor_reconnected_cavity_count, 0);
        assert_eq!(repair.reconnection_quality_gain_count, 0);
        assert_eq!(repair.split_cavity_count, 0);
        assert_eq!(repair_tets.len(), 1);
    }

    #[test]
    fn connected_bad_cavity_reconnection_repairs_nested_split_cavity() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let outer_tet = raw_candidate_tet(0, 0, &[], [0, 1, 2, 3], points, options)
            .expect("outer tet should be valid");
        let first_split_point = [0.08, 0.08, 0.08];
        let first_split_tets =
            centroid_split_tets(&outer_tet, 4, first_split_point, points, options);
        assert_eq!(first_split_tets.len(), 4);
        let base_node_points = BTreeMap::from([
            (0, points[0]),
            (1, points[1]),
            (2, points[2]),
            (3, points[3]),
            (4, first_split_point),
        ]);
        let nested_points = first_split_tets[0]
            .node_ids
            .map(|node_id| base_node_points[&node_id]);
        let nested_centroid = tet_centroid(nested_points);
        let nested_split_point = [
            nested_centroid[0] * 0.75 + nested_points[0][0] * 0.25,
            nested_centroid[1] * 0.75 + nested_points[0][1] * 0.25,
            nested_centroid[2] * 0.75 + nested_points[0][2] * 0.25,
        ];
        let nested_split_tets = centroid_split_tets(
            &first_split_tets[0],
            5,
            nested_split_point,
            nested_points,
            options,
        );
        assert_eq!(nested_split_tets.len(), 4);
        let mut split_tets = nested_split_tets;
        split_tets.extend(first_split_tets[1..].iter().cloned());
        assert_eq!(split_tets.len(), 7);
        assert!(
            split_tets
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count()
                >= 4
        );
        let mut node_points = base_node_points;
        node_points.insert(5, nested_split_point);
        let face_adjacency = tet_face_adjacency(&split_tets);
        let connected =
            connected_bad_tet_cavity_with_face_closure(0, &split_tets, &face_adjacency, options);
        let one_ring = one_ring_tet_cavity(0, &split_tets, &face_adjacency);
        let direct_candidates = face_neighbor_cavity_reconnection_candidates(
            &connected,
            &split_tets,
            &node_points,
            options,
        )
        .expect("candidate check should evaluate");
        assert!(
            direct_candidates.is_some(),
            "connected={connected:?} one_ring={one_ring:?} below={} min={}",
            count_exact_quality_violations(
                connected.iter().map(|index| &split_tets[*index]),
                options.min_scaled_jacobian
            ),
            min_exact_scaled_jacobian(connected.iter().map(|index| &split_tets[*index]))
        );

        let (indices, candidates, quality_gain_only) = best_connected_bad_cavity_reconnection(
            0,
            &split_tets,
            &face_adjacency,
            &node_points,
            options,
        )
        .expect("connected cavity reconnection should evaluate")
        .expect("connected cavity reconnection should be available");

        assert!(indices.len() > one_ring.len());
        assert!(!quality_gain_only);
        let original_cluster = indices
            .iter()
            .map(|index| split_tets[*index].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            boundary_faces_from_tets(&candidates),
            boundary_faces_from_tets(&original_cluster)
        );
        assert!(
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian)
                < count_exact_quality_violations(
                    indices.iter().map(|index| &split_tets[*index]),
                    options.min_scaled_jacobian
                )
        );

        let mut nodes = node_points
            .iter()
            .map(|(node_id, coordinates_m)| TetCandidateNode {
                node_id: *node_id,
                coordinates_m: *coordinates_m,
                source: TetCandidateNodeSource::Surface,
            })
            .collect::<Vec<_>>();
        let mut repair_tets = split_tets.clone();
        let mut interior_seed_points = Vec::new();
        let mut next_node_id = 6;
        let repair = repair_exact_quality_tets_once(
            &mut nodes,
            &mut repair_tets,
            &mut interior_seed_points,
            &mut next_node_id,
            options,
        )
        .expect("repair should evaluate");

        assert!(repair.changed);
        assert_eq!(repair.reconnected_cavity_count, 1);
        assert_eq!(repair.boundary_adjacent_reconnected_cavity_count, 1);
        assert_eq!(repair.node_adjacent_reconnected_cavity_count, 0);
        assert_eq!(repair.connected_reconnected_cavity_count, 0);
        assert_eq!(repair.face_neighbor_reconnected_cavity_count, 0);
        assert_eq!(repair.split_cavity_count, 0);
    }

    #[test]
    fn connected_bad_cavity_face_closure_layers_expand_bounded() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let tets = [
            ([0, 1, 2, 3], 0.1),
            ([0, 1, 2, 4], 0.8),
            ([0, 1, 4, 5], 0.8),
        ]
        .into_iter()
        .map(|(node_ids, exact_scaled_jacobian)| TetCandidate {
            tet_id: 0,
            component_id: 0,
            node_ids,
            source_surface_element_id: 0,
            region_ids: Vec::new(),
            volume_m3: 1.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian,
        })
        .collect::<Vec<_>>();
        let face_adjacency = tet_face_adjacency(&tets);
        let one_layer =
            connected_bad_tet_cavity_with_face_closure(0, &tets, &face_adjacency, options);
        let two_layer = connected_bad_tet_cavity_with_face_closure_layers(
            0,
            &tets,
            &face_adjacency,
            options,
            2,
        );
        assert_eq!(one_layer, vec![0, 1]);
        assert_eq!(two_layer, vec![0, 1, 2]);
    }

    #[test]
    fn boundary_adjacent_node_closure_repairs_split_cavity() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let outer_tet = raw_candidate_tet(0, 0, &[], [0, 1, 2, 3], points, options)
            .expect("outer tet should be valid");
        let first_split_point = [0.08, 0.08, 0.08];
        let first_split_tets =
            centroid_split_tets(&outer_tet, 4, first_split_point, points, options);
        let mut node_points = BTreeMap::from([
            (0, points[0]),
            (1, points[1]),
            (2, points[2]),
            (3, points[3]),
            (4, first_split_point),
        ]);
        let mut split_tets = Vec::<TetCandidate>::new();
        for (tet_index, tet) in first_split_tets.iter().enumerate() {
            if tet_index == 0 || tet_index == 2 {
                let nested_points = tet.node_ids.map(|node_id| node_points[&node_id]);
                let nested_centroid = tet_centroid(nested_points);
                let nested_node_id = if tet_index == 0 { 5 } else { 6 };
                let nested_split_point = [
                    nested_centroid[0] * 0.75 + nested_points[0][0] * 0.25,
                    nested_centroid[1] * 0.75 + nested_points[0][1] * 0.25,
                    nested_centroid[2] * 0.75 + nested_points[0][2] * 0.25,
                ];
                node_points.insert(nested_node_id, nested_split_point);
                split_tets.extend(centroid_split_tets(
                    tet,
                    nested_node_id,
                    nested_split_point,
                    nested_points,
                    options,
                ));
            } else {
                split_tets.push(tet.clone());
            }
        }
        assert!(
            split_tets
                .iter()
                .filter(|tet| tet.exact_scaled_jacobian < options.min_scaled_jacobian)
                .count()
                >= 4
        );
        let original_boundary_faces = boundary_faces_from_tets(&split_tets);
        let original_volume = split_tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        let original_bad_count =
            count_exact_quality_violations(split_tets.iter(), options.min_scaled_jacobian);
        let face_adjacency = tet_face_adjacency(&split_tets);
        let node_adjacency = tet_node_adjacency(&split_tets);
        let face_closure =
            connected_bad_tet_cavity_with_face_closure(0, &split_tets, &face_adjacency, options);
        let node_closure = boundary_adjacent_bad_tet_cavity_with_node_closure(
            0,
            &split_tets,
            &face_adjacency,
            &node_adjacency,
            options,
        );
        assert_eq!(node_closure, face_closure);

        let (indices, candidates, quality_gain_only) = best_boundary_adjacent_cavity_reconnection(
            0,
            &split_tets,
            &face_adjacency,
            &node_adjacency,
            &node_points,
            options,
        )
        .expect("boundary-adjacent cavity reconnection should evaluate")
        .expect("boundary-adjacent cavity reconnection should be available");

        assert_eq!(indices, node_closure);
        assert!(!quality_gain_only);
        let original_cluster = indices
            .iter()
            .map(|index| split_tets[*index].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            boundary_faces_from_tets(&candidates),
            boundary_faces_from_tets(&original_cluster)
        );
        assert!(
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian)
                < count_exact_quality_violations(
                    indices.iter().map(|index| &split_tets[*index]),
                    options.min_scaled_jacobian
                )
        );

        let mut nodes = node_points
            .iter()
            .map(|(node_id, coordinates_m)| TetCandidateNode {
                node_id: *node_id,
                coordinates_m: *coordinates_m,
                source: TetCandidateNodeSource::Surface,
            })
            .collect::<Vec<_>>();
        let mut repair_tets = split_tets.clone();
        let mut interior_seed_points = Vec::new();
        let mut next_node_id = 7;
        let repair = repair_exact_quality_tets_once(
            &mut nodes,
            &mut repair_tets,
            &mut interior_seed_points,
            &mut next_node_id,
            options,
        )
        .expect("repair should evaluate");

        assert!(repair.changed);
        assert_eq!(repair.reconnected_cavity_count, 1);
        assert_eq!(repair.boundary_adjacent_reconnected_cavity_count, 1);
        assert_eq!(repair.node_adjacent_reconnected_cavity_count, 0);
        assert_eq!(repair.connected_reconnected_cavity_count, 0);
        assert_eq!(repair.face_neighbor_reconnected_cavity_count, 0);
        assert_eq!(
            boundary_faces_from_tets(&repair_tets),
            original_boundary_faces,
            "boundary-adjacent repair must preserve the cavity boundary"
        );
        let repaired_volume = repair_tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!(
            (repaired_volume - original_volume).abs() <= original_volume.max(1.0e-18) * 1.0e-9,
            "boundary-adjacent repair must preserve cavity volume"
        );
        assert!(
            count_exact_quality_violations(repair_tets.iter(), options.min_scaled_jacobian)
                < original_bad_count,
            "boundary-adjacent repair must reduce exact-quality violations"
        );
    }

    #[test]
    fn boundary_adjacent_cavity_untangling_reconnects_near_singular_split_cavity() {
        let options = TetCandidateOptions {
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 2.0,
            max_refinement_passes: 1,
            ..TetCandidateOptions::default()
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let outer_tet = raw_candidate_tet(0, 0, &[], [0, 1, 2, 3], points, options)
            .expect("outer tet should be valid");
        let split_point = [0.04, 0.04, 0.04];
        let split_tets = centroid_split_tets(&outer_tet, 4, split_point, points, options);
        assert_eq!(split_tets.len(), 4);
        let node_points = BTreeMap::from([
            (0, points[0]),
            (1, points[1]),
            (2, points[2]),
            (3, points[3]),
            (4, split_point),
        ]);
        let threshold = untangling_exact_quality_threshold(options);
        let initial_near_singular_count =
            count_tets_below_exact_quality(split_tets.iter(), threshold);
        assert!(
            initial_near_singular_count > 0,
            "split cavity should contain near-singular boundary-adjacent Tets"
        );
        let original_full_bad_count =
            count_exact_quality_violations(split_tets.iter(), options.min_scaled_jacobian);
        let original_boundary_faces = boundary_faces_from_tets(&split_tets);
        let original_volume = split_tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
        let face_adjacency = tet_face_adjacency(&split_tets);
        let node_adjacency = tet_node_adjacency(&split_tets);

        let (indices, candidates) = best_boundary_adjacent_cavity_untangling(
            0,
            &split_tets,
            &face_adjacency,
            &node_adjacency,
            &node_points,
            threshold,
            options,
        )
        .expect("boundary-cavity untangling should evaluate")
        .expect("boundary-cavity untangling should be available");

        assert_eq!(indices.len(), split_tets.len());
        assert_eq!(
            boundary_faces_from_tets(&candidates),
            original_boundary_faces,
            "boundary-cavity untangling must preserve the cavity boundary"
        );
        let candidate_volume = candidates.iter().map(|tet| tet.volume_m3).sum::<f64>();
        assert!(
            (candidate_volume - original_volume).abs() <= original_volume.max(1.0e-18) * 1.0e-9,
            "boundary-cavity untangling must preserve cavity volume"
        );
        assert!(
            count_tets_below_exact_quality(candidates.iter(), threshold)
                < initial_near_singular_count,
            "boundary-cavity untangling should reduce near-singular Tets"
        );
        assert!(
            count_exact_quality_violations(candidates.iter(), options.min_scaled_jacobian)
                <= original_full_bad_count,
            "untangling must not increase full exact-quality violations"
        );
    }

    #[test]
    fn boundary_adjacent_node_closure_layers_expand_bounded() {
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let tets = [
            ([0, 1, 2, 3], 0.1),
            ([0, 1, 2, 4], 0.8),
            ([4, 5, 6, 7], 0.8),
            ([7, 8, 9, 10], 0.8),
            ([11, 12, 13, 14], 0.8),
        ]
        .into_iter()
        .enumerate()
        .map(|(tet_id, (node_ids, exact_scaled_jacobian))| TetCandidate {
            tet_id: tet_id as u32,
            component_id: 0,
            node_ids,
            source_surface_element_id: 0,
            region_ids: Vec::new(),
            volume_m3: 1.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian,
        })
        .collect::<Vec<_>>();
        let face_adjacency = tet_face_adjacency(&tets);
        let node_adjacency = tet_node_adjacency(&tets);

        let face_closure =
            connected_bad_tet_cavity_with_face_closure(0, &tets, &face_adjacency, options);
        let one_layer = boundary_adjacent_bad_tet_cavity_with_node_closure_layers(
            0,
            &tets,
            &face_adjacency,
            &node_adjacency,
            options,
            1,
        );
        let two_layer = boundary_adjacent_bad_tet_cavity_with_node_closure_layers(
            0,
            &tets,
            &face_adjacency,
            &node_adjacency,
            options,
            2,
        );

        assert_eq!(face_closure, vec![0, 1]);
        assert_eq!(one_layer, vec![0, 1, 2]);
        assert_eq!(two_layer, vec![0, 1, 2, 3]);
        assert!(!two_layer.contains(&4));
    }

    #[test]
    fn node_adjacent_cavity_closure_expands_non_face_connected_cavity_bounded() {
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let tets = [
            ([0, 1, 2, 3], 0.1),
            ([0, 1, 2, 4], 0.8),
            ([0, 5, 6, 7], 0.2),
            ([8, 9, 10, 11], 0.2),
        ]
        .into_iter()
        .enumerate()
        .map(|(tet_id, (node_ids, exact_scaled_jacobian))| TetCandidate {
            tet_id: tet_id as u32,
            component_id: 0,
            node_ids,
            source_surface_element_id: 0,
            region_ids: Vec::new(),
            volume_m3: 1.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian,
        })
        .collect::<Vec<_>>();
        let face_adjacency = tet_face_adjacency(&tets);
        let node_adjacency = tet_node_adjacency(&tets);

        let face_closure =
            connected_bad_tet_cavity_with_face_closure(0, &tets, &face_adjacency, options);
        let node_closure = connected_bad_tet_cavity_with_node_closure(
            0,
            &tets,
            &face_adjacency,
            &node_adjacency,
            options,
        );

        assert_eq!(face_closure, vec![0, 1]);
        assert_eq!(node_closure, vec![0, 1, 2]);
        assert!(!node_closure.contains(&3));
    }

    #[test]
    fn node_adjacent_cavity_untangling_rejects_unclosed_node_touching_groups() {
        let options = TetCandidateOptions {
            min_scaled_jacobian: 0.4,
            ..TetCandidateOptions::default()
        };
        let tets = [
            ([0, 1, 2, 3], 0.05),
            ([0, 1, 2, 4], 0.8),
            ([0, 5, 6, 7], 0.05),
        ]
        .into_iter()
        .enumerate()
        .map(|(tet_id, (node_ids, exact_scaled_jacobian))| TetCandidate {
            tet_id: tet_id as u32,
            component_id: 0,
            node_ids,
            source_surface_element_id: 0,
            region_ids: Vec::new(),
            volume_m3: 1.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian,
        })
        .collect::<Vec<_>>();
        let node_points = BTreeMap::from([
            (0, [0.0, 0.0, 0.0]),
            (1, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
            (3, [0.0, 0.0, 1.0]),
            (4, [0.0, 0.0, -1.0]),
            (5, [2.0, 0.0, 0.0]),
            (6, [0.0, 2.0, 0.0]),
            (7, [0.0, 0.0, 2.0]),
        ]);
        let face_adjacency = tet_face_adjacency(&tets);
        let node_adjacency = tet_node_adjacency(&tets);
        let threshold = untangling_exact_quality_threshold(options);

        let repair_result = best_node_adjacent_cavity_reconnection(
            0,
            &tets,
            &face_adjacency,
            &node_adjacency,
            &node_points,
            options,
        )
        .expect("node-adjacent repair should evaluate");
        let result = best_node_adjacent_cavity_untangling(
            0,
            &tets,
            &face_adjacency,
            &node_adjacency,
            &node_points,
            threshold,
            options,
        )
        .expect("node-adjacent untangling should evaluate");

        assert!(
            repair_result.is_none(),
            "small node-touching groups that do not form a closed remeshable cavity must stay fail-closed"
        );
        assert!(
            result.is_none(),
            "node-touching groups that do not form a closed remeshable cavity must stay fail-closed"
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
                ..TetCandidateOptions::default()
            },
        )
        .expect_err("strict recovery should expose recovery failure");

        assert_eq!(err, TetCandidateError::RecoveryFailed { component_id: 0 });
    }

    #[test]
    fn fan_fallback_requires_explicit_opt_in() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();

        let strict_err = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                max_global_insertion_points: 4,
                sliver_aspect_ratio: 1.0,
                ..TetCandidateOptions::default()
            },
        )
        .expect_err("strict default should reject unrecovered insertion");
        assert_eq!(
            strict_err,
            TetCandidateError::RecoveryFailed { component_id: 0 }
        );

        let candidates = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                max_global_insertion_points: 4,
                sliver_aspect_ratio: 1.0,
                allow_fan_fallback: true,
                ..TetCandidateOptions::default()
            },
        )
        .expect("explicit compatibility fallback should recover a candidate");

        assert_eq!(candidates.recovery.insertion_component_count, 0);
        assert_eq!(candidates.recovery.fan_fallback_component_count, 1);
        assert_eq!(candidates.recovery.recovered_component_ratio, 0.0);
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
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

    fn thin_cube_geometry() -> GeometryAsset {
        let mut geometry = cube_geometry();
        geometry.geometry_id = "geo_tet_candidate_thin_cube".to_string();
        geometry.source.sha256 = "generic-thin-cube".to_string();
        if let Some(surface) = geometry.surface_meshes.first_mut() {
            for vertex in &mut surface.vertices {
                if vertex[2] > 0.0 {
                    vertex[2] = 0.1;
                }
            }
        }
        geometry
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
