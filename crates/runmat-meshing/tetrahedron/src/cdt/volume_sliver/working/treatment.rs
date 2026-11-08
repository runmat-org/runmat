//! Metric-aware execution owns local search state across accepted passes.
//! Complete deletion/retriangulation retains its explicit canonical admission
//! boundary until that owner's full dependency closure has local authority.
use super::*;
use crate::cdt::volume_refinement::context::RefinementContext;
use super::rejected::{CompleteSearchKey, Ranking, RejectedCompleteRelocations, SearchRecording};
use crate::cdt::volume_sliver::{insertion, reconnection, relocation, work::OptimizationWork};
use std::collections::BTreeMap;

pub(in crate::cdt::volume_sliver) fn run<'ctx>(
    topology: DelaunayVolumeTopology,
    quality: DelaunayVolumeQuality,
    context: RefinementContext<'ctx>,
    prepared: &'ctx PreparedLocalQuality<'ctx>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
    mode: QualityConvergenceMode,
) -> Result<DelaunayVolumeSliverTreatment, DelaunayVolumeSliverError> {
    let span = tracing::info_span!(
        "meshing.tetrahedron.volume_optimization",
        tetrahedron_count = topology.tetrahedra.len(),
        local_search = true
    );
    let _guard = span.enter();
    // An admitted no-op preserves its original allocations and requires no
    // index preparation or canonical reconstruction.
    checkpoint(0, options, cancellation)?;
    if options.maximum_passes > 0
        && admission::worst_quality_source(
            &quality,
            &topology,
            mode.objective(
                admission::worst_quality_source(&quality, &topology, QualityObjective::Shape),
                admission::worst_quality_source(&quality, &topology, QualityObjective::Refinement),
            ),
        )
        .is_none()
    {
        return Ok(DelaunayVolumeSliverTreatment {
            topology,
            quality,
            mutations: Vec::new(),
        });
    }
    let state = WorkingState::new(
        topology,
        quality,
        context.quality_options,
        prepared,
        context.fixed_node_identities,
        cancellation,
    )?;
    let treatment = run_in_state(
        state,
        context,
        prepared,
        options,
        cancellation,
        mode,
        &mut RejectedCompleteRelocations::default(),
    )
    .map_err(|failure| failure.error)?;
    finish(treatment.state, treatment.mutations, options, cancellation)
}

pub(in crate::cdt) struct WorkingTreatment<'ctx> {
    pub(in crate::cdt) state: WorkingState<'ctx>,
    pub(in crate::cdt) mutations: Vec<DelaunayVolumeSliverMutation>,
}

/// A terminal local minimum retains its owned state, together with the
/// mutations this call applied to reach it, so callers that continue from
/// that state keep a complete lineage. Other failures carry no partially
/// applied state. No observer or I/O lives here.
pub(in crate::cdt) struct WorkingFailure<'ctx> {
    pub(in crate::cdt) error: DelaunayVolumeSliverError,
    pub(in crate::cdt) state: Option<Box<WorkingState<'ctx>>>,
    pub(in crate::cdt) applied: Vec<DelaunayVolumeSliverMutation>,
}
impl From<DelaunayVolumeSliverError> for WorkingFailure<'_> {
    fn from(error: DelaunayVolumeSliverError) -> Self {
        Self {
            error,
            state: None,
            applied: Vec::new(),
        }
    }
}
impl From<crate::cdt::volume_state::VolumeStateError> for WorkingFailure<'_> {
    fn from(error: crate::cdt::volume_state::VolumeStateError) -> Self {
        Self::from(DelaunayVolumeSliverError::from(error))
    }
}

pub(in crate::cdt) fn run_in_state<'ctx>(
    mut state: WorkingState<'ctx>,
    context: RefinementContext<'ctx>,
    prepared: &'ctx PreparedLocalQuality<'ctx>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
    mode: QualityConvergenceMode,
    rejected: &mut RejectedCompleteRelocations,
) -> Result<WorkingTreatment<'ctx>, WorkingFailure<'ctx>> {
    let mut mutations = Vec::new();
    for pass in 0..options.maximum_passes {
        checkpoint(pass, options, cancellation)?;
        if matches!(mode, QualityConvergenceMode::Coupled)
            && !mutations.is_empty()
            && state
                .quality()
                .worst(QualityObjective::Refinement)
                .is_some()
        {
            tracing::info!(
                pass,
                mutation_count = mutations.len(),
                "volume optimization yielded to coupled refinement"
            );
            return Ok(WorkingTreatment { state, mutations });
        }
        let objective = mode.objective(
            state.quality().worst(QualityObjective::Shape),
            state.quality().worst(QualityObjective::Refinement),
        );
        let Some(source) = state.quality().worst(objective).cloned() else {
            tracing::info!(
                pass,
                mutation_count = mutations.len(),
                "volume optimization completed"
            );
            return Ok(WorkingTreatment { state, mutations });
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
        let policy = SliverSearchPolicy {
            options,
            refinement: mode.refinement_policy(),
            objective,
        };
        let mut work = OptimizationWork::new(
            options.maximum_candidate_evaluations_per_pass,
            options.cancellation_check_interval,
            cancellation,
        );
        let mut best_local = None;
        let mut reconnection = if mode == QualityConvergenceMode::ShapeOnly {
            reconnection::working::candidates(
                &mut state,
                &source,
                cancellation,
                &mut work,
                policy,
                reconnection::ReconnectionMode {
                    allow_lookahead: mode == QualityConvergenceMode::ShapeOnly,
                    maximum_candidates: None,
                },
                &[],
            )?
        } else {
            reconnection::working::deferred_candidates(
                &mut state,
                &source,
                cancellation,
                &mut work,
                policy,
                &[],
            )?
        };
        if let Some(candidate) = reconnection.candidate.take() {
            if mode == QualityConvergenceMode::ShapeOnly {
                accept(&mut state, candidate, &mut mutations, options, cancellation)?;
                tracing::info!(pass, "volume optimization admitted a local reconnection");
                continue;
            }
            WorkingSeed::retain_better(&mut best_local, candidate, policy);
        }
        let mut relocation_rejections = BTreeMap::new();
        if mode != QualityConvergenceMode::ShapeOnly {
            let local = relocation::working::candidates(
                &mut state,
                &source,
                cancellation,
                &mut work,
                policy,
                &[],
            )?;
            relocation_rejections = local.rejection_counts;
            if let Some(candidate) = local.candidate {
                WorkingSeed::retain_better(&mut best_local, candidate, policy);
            }
        }
        let mut local_insertion = None;
        if mode != QualityConvergenceMode::ShapeOnly {
            let mut local = insertion::working::candidates(
                &mut state,
                &source,
                cancellation,
                &mut work,
                policy,
                MAXIMUM_INSERTION_LOOKAHEAD_DEPTH,
                &[],
            )?;
            if let Some(candidate) = local.candidate.take() {
                WorkingSeed::retain_better(&mut best_local, candidate, policy);
            }
            local_insertion = Some(local);
        }
        if let Some(candidate) = best_local {
            accept(&mut state, candidate, &mut mutations, options, cancellation)?;
            tracing::info!(
                pass,
                "volume optimization admitted its best checked strategy"
            );
            continue;
        }
        // Reuse prepared rejected branches only after cheap checked families
        // have no improving step. Deep search must not tax every accepted pass.
        if let Some(candidate) = reconnection.complete(&mut state, cancellation, &mut work)? {
            accept(&mut state, candidate, &mut mutations, options, cancellation)?;
            tracing::info!(pass, "volume optimization admitted a compound reconnection");
            continue;
        }
        // The legacy complete strategy has no candidates without a free interior
        // source corner. Query that condition locally before a canonical handoff.
        let has_complete_source = source.node_identities.iter().any(|identity| {
            !state.protected_node(*identity)
                && !state.topology().vertex_is_boundary(
                    state
                        .topology()
                        .node_index(*identity)
                        .expect("quality corner"),
                )
        });
        let complete_key = has_complete_source
            .then(|| CompleteSearchKey::new(source.node_identities, objective));
        // The current coupled comparison floor: the worst violation of the whole state.
        let ranking = {
            let worst = |objective, ratio: fn(&crate::cdt::DelaunayTetrahedronQuality) -> f64| {
                state
                    .quality()
                    .worst(objective)
                    .map(ratio)
                    .filter(|ratio| *ratio > 1.0)
                    .map(crate::cdt::volume_quality::violation_priority_key)
            };
            Ranking {
                coupled: matches!(
                    policy.refinement,
                    admission::CandidateRefinementPolicy::AllowForCoupledConvergence
                ),
                objective,
                floor: worst(QualityObjective::Shape, |cell| {
                    cell.optimization_violation_ratio
                })
                .max(worst(QualityObjective::Refinement, |cell| {
                    cell.refinement_violation_ratio
                }))
                .unwrap_or(0),
            }
        };
        if let Some(key) = complete_key
            .as_ref()
            .filter(|key| rejected.still_rejected(state.topology(), key, ranking))
        {
            tracing::info!(
                pass,
                source = ?key,
                "volume optimization skipped a complete relocation search whose neighborhood is unchanged since it was rejected"
            );
            *relocation_rejections
                .entry("complete_search_unchanged_since_rejected")
                .or_default() += 1;
        } else if let Some(key) = complete_key {
            let started = runmat_time::Instant::now();
            tracing::info!(
                pass,
                "volume optimization entering complete relocation handoff"
            );
            let (topology, quality) = state.finish(options, cancellation)?;
            let spectrum = admission::candidate_spectrum(&quality, policy.refinement, objective);
            let mut recording = SearchRecording::default();
            let search = relocation::relocation_candidates_recording_changes(
                context.with_state(&topology, &quality),
                &source,
                &spectrum,
                cancellation,
                &mut work,
                policy,
                if mode == QualityConvergenceMode::ShapeOnly {
                    relocation::RelocationSearchScope::Complete
                } else {
                    relocation::RelocationSearchScope::TopologyChanging
                },
                &mut recording,
            )?;
            tracing::info!(
                pass,
                elapsed_seconds = started.elapsed().as_secs_f64(),
                tetrahedron_count = topology.tetrahedra.len(),
                accepted = search.candidate.is_some(),
                "volume optimization complete relocation finished"
            );
            drop(spectrum);
            for (reason, count) in search.rejection_counts {
                *relocation_rejections.entry(reason).or_default() += count;
            }
            if let Some(candidate) = search.candidate {
                // Accepted replacement consumes the previous canonical state.
                // Do not retain it while allocating the successor's indexes.
                drop((topology, quality));
                mutations.push(candidate.mutation);
                state = WorkingState::new(
                    candidate.topology,
                    candidate.quality,
                    context.quality_options,
                    prepared,
                    context.fixed_node_identities,
                    cancellation,
                )?;
                continue;
            }
            state = WorkingState::new(
                topology,
                quality,
                context.quality_options,
                prepared,
                context.fixed_node_identities,
                cancellation,
            )?;
            rejected.record(state.topology(), key, recording);
        }
        let insertion = match local_insertion {
            Some(search) => search,
            None => insertion::working::candidates(
                &mut state,
                &source,
                cancellation,
                &mut work,
                policy,
                MAXIMUM_INSERTION_LOOKAHEAD_DEPTH,
                &[],
            )?,
        };
        if let Some(candidate) = insertion.candidate {
            accept(&mut state, candidate, &mut mutations, options, cancellation)?;
            tracing::info!(
                pass,
                "volume optimization admitted a local Steiner insertion"
            );
            continue;
        }
        let source_nodes = source.node_identities.map(|identity| {
            state
                .topology()
                .node_index(identity)
                .map(|index| state.topology().node(index).coordinates_m.canonical_value())
        });
        let error = error(DelaunayVolumeSliverErrorKind::NoAdmissibleOptimization, format!(
            "no legal reconnection, interior relocation, or protected-boundary insertion improves volume defect {} at {:?} with {} boundary nodes and {} protected facets, refinement ratio {} (objective {:?}), optimization ratio {}, scaled Jacobian {}, and target {}; reconnection rejections {:?}; relocation rejections {:?}; insertion rejections {:?}; first insertion rejection details {:?}",
            stable_simplex(source.node_identities), source_nodes, source.boundary_node_count, source.protected_facet_count,
            source.refinement_violation_ratio, objective, source.optimization_violation_ratio, source.metric_scaled_jacobian, context.quality_options.minimum_metric_scaled_jacobian,
            reconnection.rejection_counts, relocation_rejections, insertion.rejection_counts, insertion.first_rejection_details));
        return Err(WorkingFailure {
            error,
            state: Some(Box::new(state)),
            applied: mutations,
        });
    }
    let remaining = state
        .quality()
        .worst(QualityObjective::Shape)
        .map(|source| source.metric_scaled_jacobian)
        .unwrap_or(context.quality_options.minimum_metric_scaled_jacobian);
    Err(error(
        DelaunayVolumeSliverErrorKind::ResourceLimit,
        format!(
            "sliver treatment exhausted {} passes with minimum metric scaled Jacobian {remaining}",
            options.maximum_passes
        ),
    )
    .into())
}

fn accept(
    state: &mut WorkingState<'_>,
    candidate: WorkingSeed,
    mutations: &mut Vec<DelaunayVolumeSliverMutation>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeSliverError> {
    let change = {
        let mut trial = state.trial();
        candidate.replay(&mut trial, options, cancellation)?;
        trial.commit_prepared(cancellation)?
    };
    state.rebase_committed(change);
    mutations.extend(candidate.mutations);
    Ok(())
}

fn finish(
    state: WorkingState<'_>,
    mutations: Vec<DelaunayVolumeSliverMutation>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeSliverTreatment, DelaunayVolumeSliverError> {
    let (topology, quality) = state.finish(options, cancellation)?;
    Ok(DelaunayVolumeSliverTreatment {
        topology,
        quality,
        mutations,
    })
}
