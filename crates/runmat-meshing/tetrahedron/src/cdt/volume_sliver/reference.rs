//! Complete canonical search oracle; production uses transactional search.
use super::*;

pub(super) fn run(
    mut topology: DelaunayVolumeTopology,
    mut quality: DelaunayVolumeQuality,
    context: crate::cdt::volume_refinement::context::RefinementContext<'_>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
    mode: QualityConvergenceMode,
) -> Result<DelaunayVolumeSliverTreatment, DelaunayVolumeSliverError> {
    let span = tracing::info_span!(
        "meshing.tetrahedron.volume_optimization",
        tetrahedron_count = topology.tetrahedra.len(),
    );
    let _guard = span.enter();
    let mut mutations = Vec::new();
    for pass in 0..options.maximum_passes {
        checkpoint(pass, options, cancellation)?;
        if matches!(mode, QualityConvergenceMode::Coupled)
            && !mutations.is_empty()
            && quality.worst_refinement_tetrahedron.is_some()
        {
            tracing::info!(
                pass,
                mutation_count = mutations.len(),
                "volume optimization yielded to coupled refinement"
            );
            return Ok(DelaunayVolumeSliverTreatment {
                topology,
                quality,
                mutations,
            });
        }
        let objective = mode.objective(
            worst_quality_source(&quality, &topology, QualityObjective::Shape),
            worst_quality_source(&quality, &topology, QualityObjective::Refinement),
        );
        let Some(source) = worst_quality_source(&quality, &topology, objective) else {
            tracing::info!(
                pass,
                mutation_count = mutations.len(),
                "volume optimization completed"
            );
            return Ok(DelaunayVolumeSliverTreatment {
                topology,
                quality,
                mutations,
            });
        };
        tracing::info!(
            pass,
            optimization_violation_ratio = source.optimization_violation_ratio,
            boundary_node_count = source.boundary_node_count,
            protected_facet_count = source.protected_facet_count,
            metric_radius_edge_ratio = source.metric_radius_edge_ratio,
            metric_scaled_jacobian = source.metric_scaled_jacobian,
            "volume optimization pass started"
        );
        let current_spectrum =
            admission::candidate_spectrum(&quality, mode.refinement_policy(), objective);
        let current = context.with_state(&topology, &quality);
        let mut work = OptimizationWork::new(
            options.maximum_candidate_evaluations_per_pass,
            options.cancellation_check_interval,
            cancellation,
        );
        let reconnection = reconnection_candidates(
            current,
            source,
            &current_spectrum,
            cancellation,
            &mut work,
            SliverSearchPolicy {
                options,
                refinement: mode.refinement_policy(),
                objective,
            },
            reconnection::ReconnectionMode {
                allow_lookahead: true,
                maximum_candidates: None,
            },
        )?;
        if let Some(candidate) = reconnection.candidate {
            topology = candidate.topology;
            quality = candidate.quality;
            mutations.extend(candidate.mutations);
            tracing::info!(pass, "volume optimization admitted a local reconnection");
            continue;
        }
        let search = relocation_candidates(
            current,
            source,
            &current_spectrum,
            cancellation,
            &mut work,
            SliverSearchPolicy {
                options,
                refinement: mode.refinement_policy(),
                objective,
            },
            RelocationSearchScope::Complete,
        )?;
        if let Some(candidate) = search.candidate {
            topology = candidate.topology;
            quality = candidate.quality;
            mutations.push(candidate.mutation);
            continue;
        }
        let insertion = insertion_candidates(
            current,
            source,
            &current_spectrum,
            cancellation,
            &mut work,
            SliverSearchPolicy {
                options,
                refinement: mode.refinement_policy(),
                objective,
            },
            MAXIMUM_INSERTION_LOOKAHEAD_DEPTH,
        )?;
        let Some(candidate) = insertion.candidate else {
            let source_nodes = source.node_identities.map(|identity| {
                topology
                    .nodes
                    .binary_search_by_key(&identity, |node| node.identity)
                    .ok()
                    .map(|index| topology.nodes[index].coordinates_m.canonical_value())
            });
            return Err(error(
                DelaunayVolumeSliverErrorKind::NoAdmissibleOptimization,
                format!(
                    "no legal reconnection, interior relocation, or protected-boundary insertion improves volume defect {} at {:?} with {} boundary nodes and {} protected facets, optimization ratio {}, scaled Jacobian {}, and target {}; reconnection rejections {:?}; relocation rejections {:?}; insertion rejections {:?}; first insertion rejection details {:?}",
                    stable_simplex(source.node_identities),
                    source_nodes,
                    source.boundary_node_count,
                    source.protected_facet_count,
                    source.optimization_violation_ratio,
                    source.metric_scaled_jacobian,
                    context.quality_options.minimum_metric_scaled_jacobian,
                    reconnection.rejection_counts,
                    search.rejection_counts,
                    insertion.rejection_counts,
                    insertion.first_rejection_details,
                ),
            ));
        };
        topology = candidate.topology;
        quality = candidate.quality;
        mutations.extend(candidate.mutations);
        tracing::info!(
            pass,
            "volume optimization admitted a local Steiner insertion"
        );
    }
    let remaining = worst_sliver(
        &quality,
        &topology,
        context.quality_options.minimum_metric_scaled_jacobian,
    )
    .map(|tetrahedron| tetrahedron.metric_scaled_jacobian)
    .unwrap_or(context.quality_options.minimum_metric_scaled_jacobian);
    Err(error(
        DelaunayVolumeSliverErrorKind::ResourceLimit,
        format!(
            "sliver treatment exhausted {} passes with minimum metric scaled Jacobian {remaining}",
            options.maximum_passes
        ),
    ))
}

fn worst_sliver<'a>(
    quality: &'a DelaunayVolumeQuality,
    topology: &DelaunayVolumeTopology,
    _minimum: f64,
) -> Option<&'a crate::cdt::DelaunayTetrahedronQuality> {
    quality
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.requires_optimization())
        .max_by(|left, right| {
            crate::cdt::volume_quality::compare_violation_priority(
                left.optimization_violation_ratio,
                crate::cdt::volume_quality::tetrahedron_geometry_key(
                    topology,
                    left.node_identities,
                ),
                left.node_identities,
                right.optimization_violation_ratio,
                crate::cdt::volume_quality::tetrahedron_geometry_key(
                    topology,
                    right.node_identities,
                ),
                right.node_identities,
            )
        })
}
