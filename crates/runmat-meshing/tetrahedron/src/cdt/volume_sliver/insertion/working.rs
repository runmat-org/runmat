//! Bounded insertion search with one mutable topology and exact quality owner.
//! Retained candidates contain local edits, never copied background meshes.
use super::*;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::volume_sliver::working::CaptureSeed;
use crate::cdt::volume_sliver::working::{SearchState, WorkingSeed, WorkingTrial};
mod geometry;
mod lookahead;

pub(in crate::cdt::volume_sliver) struct WorkingInsertionSearch {
    pub(in crate::cdt::volume_sliver) candidate: Option<WorkingSeed>,
    pub(in crate::cdt::volume_sliver) rejection_counts: BTreeMap<&'static str, u64>,
    pub(in crate::cdt::volume_sliver) first_rejection_details: BTreeMap<&'static str, String>,
}

pub(in crate::cdt::volume_sliver) fn candidates<'ctx>(
    state: &mut impl SearchState<'ctx>,
    source: &DelaunayTetrahedronQuality,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    remaining_lookahead_depth: u8,
    prefix: &[DelaunayVolumeSliverMutation],
) -> Result<WorkingInsertionSearch, DelaunayVolumeSliverError> {
    let prepared = state.preparation();
    let source_vertices = source.node_identities.map(|identity| {
        state
            .topology()
            .node_index(identity)
            .expect("quality source corner")
    });
    let metrics = |face: [u32; 3]| {
        prepared
            .insertion_metric(
                face.map(|vertex| state.topology().node(vertex).identity),
                &source.region_id,
            )
            .map_err(crate::cdt::volume_sliver::quality_error)
    };
    let mut choices = choices::edge_star_choices_prepared(
        state.topology(),
        source_vertices,
        &source.region_id,
        choices::InsertionProtection {
            faces: prepared.protected_facets(),
            edges: prepared.protected_edges(),
            segments: prepared.protected_segments(),
        },
        &metrics,
        cancellation,
    )?;
    if matches!(
        policy.refinement,
        crate::cdt::volume_sliver::admission::CandidateRefinementPolicy::AllowForCoupledConvergence
    ) {
        let mut front = offcenter::choices(
            state.topology(),
            source,
            prepared.protected_facets(),
            state.quality_options().minimum_metric_scaled_jacobian,
            cancellation,
        )?;
        front.extend(choices);
        choices = front;
    }
    let refinement_baseline = state.quality().capture_spectrum();
    let mut rejection_counts = BTreeMap::new();
    let mut first_rejection_details = BTreeMap::new();
    let mut best: Option<WorkingSeed> = None;
    let mut best_unranked: Option<WorkingSeed> = None;
    let mut frontier = crate::cdt::volume_sliver::frontier::RankedFrontier::new(2);
    for mut choice in choices {
        work.evaluate()?;
        if let InsertionStrategy::Cavity { neighbor_index } = choice.strategy {
            if let Some(point) = smooth_visibility_cavity_point(
                state.topology(),
                choice.coordinates_m.canonical_value(),
                VisibilityContext {
                    source_vertices,
                    neighbor_index,
                    protected_faces: prepared.protected_facets(),
                    metrics: &|face: [u32; 3]| {
                        prepared
                            .insertion_metric(
                                face.map(|vertex| state.topology().node(vertex).identity),
                                &source.region_id,
                            )
                            .map_err(crate::cdt::volume_sliver::quality_error)
                    },
                    target_scaled_jacobian: state.quality_options().minimum_metric_scaled_jacobian,
                    options: policy.options,
                    cancellation,
                },
            )? {
                choice.coordinates_m = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(point);
                choice.construction = Point3Construction::Explicit;
            }
        }
        let node = DelaunayVolumeNode {
            identity: insertion_identity(
                source.node_identities,
                choice.edge_identities,
                choice.index,
            ),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(choice.coordinates_m.canonical_value()),
            construction: choice.construction,
        };
        if state.topology().node_index(node.identity).is_some()
            || state
                .topology()
                .contains_position(node.coordinates_m.canonical_value())
        {
            reject(
                &mut rejection_counts,
                "candidate_collides_with_existing_node",
            );
            continue;
        }
        let mut trial = state.trial();
        let split = {
            let mut attempt = trial.trial();
            let outcome = geometry::split(
                &mut attempt,
                node,
                choice,
                source,
                policy,
                work,
                cancellation,
            )?;
            if outcome.is_ok() {
                attempt.commit();
            }
            outcome
        };
        if let Err(detail) = split {
            if matches!(
                choice.strategy,
                InsertionStrategy::Cavity { .. } | InsertionStrategy::BoundaryOffcenter { .. }
            ) {
                reject(&mut rejection_counts, "cavity_insertion_rejected");
                first_rejection_details
                    .entry("cavity_insertion_rejected")
                    .or_insert(detail);
                continue;
            }
            if let Err(failure) = trial
                .insert_constrained(node, policy.options, cancellation)
                .map_err(DelaunayVolumeSliverError::from)
            {
                if failure.kind != DelaunayVolumeSliverErrorKind::InvalidTopology {
                    return Err(failure);
                }
                reject(&mut rejection_counts, "edge_star_insertion_rejected");
                first_rejection_details
                    .entry("edge_star_insertion_rejected")
                    .or_insert_with(|| {
                        format!("edge-star split: {detail}; constrained insertion: {failure}")
                    });
                continue;
            }
        }
        let mut mutations = prefix.to_vec();
        mutations.push(DelaunayVolumeSliverMutation::Inserted(
            DelaunayVolumeSliverInsertion {
                inserted_node: node,
                source_tetrahedron_node_identities: source.node_identities,
                source_edge_node_identities: choice.edge_identities,
                construction_point_index: choice.index,
            },
        ));
        retain_better(&mut best_unranked, trial.capture(mutations.clone()), policy);
        if remaining_lookahead_depth > 0 && best.is_none() {
            frontier.retain(
                trial.capture(mutations.clone()),
                |candidate: &WorkingSeed, entry: &WorkingSeed| {
                    candidate
                        .compare_policy(entry, policy)
                        .then_with(|| candidate.last_node_identity.cmp(&entry.last_node_identity))
                },
            );
        }
        if crate::cdt::volume_sliver::admission::working_quality_is_admissible(
            trial.quality(),
            &refinement_baseline,
            policy.refinement,
            policy.objective,
            &mut rejection_counts,
        ) {
            retain_better(&mut best, trial.capture(mutations.clone()), policy);
        }
        let Some(next_source) = trial
            .quality()
            .worst(policy.current_objective(
                trial.quality().worst(QualityObjective::Shape),
                trial.quality().worst(QualityObjective::Refinement),
            ))
            .cloned()
        else {
            continue;
        };
        let next = crate::cdt::volume_sliver::reconnection::working::candidates(
            &mut trial,
            &next_source,
            cancellation,
            work,
            policy,
            crate::cdt::volume_sliver::reconnection::ReconnectionMode {
                allow_lookahead: false,
                maximum_candidates: Some(
                    policy.options.maximum_compound_reconnections_per_insertion,
                ),
            },
            &mutations,
        )?;
        if let Some(candidate) = next.candidate {
            retain_better(&mut best, candidate, policy);
        }
    }
    if best.is_none() && remaining_lookahead_depth > 0 {
        let next = lookahead::search(
            state,
            frontier.into_entries(),
            cancellation,
            work,
            policy,
            remaining_lookahead_depth,
        )?;
        for (reason, count) in next.rejection_counts {
            *rejection_counts.entry(reason).or_default() += count;
        }
        for (reason, detail) in next.first_rejection_details {
            first_rejection_details.entry(reason).or_insert(detail);
        }
        if next.candidate.is_some() {
            return Ok(WorkingInsertionSearch {
                candidate: next.candidate,
                rejection_counts,
                first_rejection_details,
            });
        }
    }
    if let Some(seed) = best_unranked {
        let mut trial = state.trial();
        seed.replay(&mut trial, policy.options, cancellation)?;
        first_rejection_details.insert(
            "best_unranked_optimization_spectrum",
            format!(
                "{:?}",
                trial
                    .quality()
                    .diagnostic_spectrum_prefix(policy.objective, 8)
            ),
        );
    }
    Ok(WorkingInsertionSearch {
        candidate: best,
        rejection_counts,
        first_rejection_details,
    })
}

fn retain_better(
    best: &mut Option<WorkingSeed>,
    candidate: WorkingSeed,
    policy: SliverSearchPolicy,
) {
    if best
        .as_ref()
        .is_none_or(|current| candidate.compare_policy(current, policy) == std::cmp::Ordering::Less)
    {
        *best = Some(candidate);
    }
}
