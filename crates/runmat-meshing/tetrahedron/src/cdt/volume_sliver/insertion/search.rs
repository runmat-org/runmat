//! Bounded insertion and compound-search coordination; geometry stays in the parent owner.
use super::retained::RetainedInsertion;
use super::*;

pub(in crate::cdt::volume_sliver) struct InsertionCandidate {
    pub(in crate::cdt::volume_sliver) topology: DelaunayVolumeTopology,
    pub(in crate::cdt::volume_sliver) quality: DelaunayVolumeQuality,
    pub(in crate::cdt::volume_sliver) mutations: Vec<DelaunayVolumeSliverMutation>,
}

pub(in crate::cdt::volume_sliver) struct InsertionSearch {
    pub(in crate::cdt::volume_sliver) candidate: Option<InsertionCandidate>,
    pub(in crate::cdt::volume_sliver) rejection_counts: BTreeMap<&'static str, u64>,
    pub(in crate::cdt::volume_sliver) first_rejection_details: BTreeMap<&'static str, String>,
}

pub(in crate::cdt::volume_sliver) fn insertion_candidates(
    input: DelaunayVolumeRefinementInput<'_>,
    source: &DelaunayTetrahedronQuality,
    current_spectrum: &[f64],
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    remaining_lookahead_depth: u8,
) -> Result<InsertionSearch, DelaunayVolumeSliverError> {
    let SliverSearchPolicy {
        options,
        refinement: refinement_policy,
        objective,
    } = policy;
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    let protected_segments = input
        .provenance
        .segments
        .iter()
        .map(|segment| segment.node_identities)
        .collect::<BTreeSet<_>>();
    let source_vertices = source.node_identities.map(|identity| {
        input
            .topology
            .nodes
            .binary_search_by_key(&identity, |node| node.identity)
            .map(|index| index as u32)
            .map_err(|_| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "optimization source node is absent from canonical topology",
                )
            })
    });
    let source_vertices = source_vertices
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .expect("source tetrahedron has four nodes");
    let mut rejection_counts = BTreeMap::new();
    let mut first_rejection_details = BTreeMap::new();
    let mut best_unranked_spectrum: Option<Vec<f64>> = None;
    let mut best_candidate: Option<RetainedInsertion> = None;
    const MAXIMUM_LOOKAHEAD_WIDTH: usize = 2;
    let mut lookahead_seeds = super::super::frontier::RankedFrontier::new(MAXIMUM_LOOKAHEAD_WIDTH);
    let visibility_barriers = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    let metric_preparation = crate::cdt::volume_quality::working::PreparedLocalQuality::prepare(
        input.metric_request,
        input.provenance,
        input.quality_options,
    )
    .map_err(crate::cdt::volume_sliver::quality_error)?;
    let metrics = |face: [u32; 3]| {
        metric_preparation
            .insertion_metric(
                face.map(|vertex| input.topology.nodes[vertex as usize].identity),
                &source.region_id,
            )
            .map_err(crate::cdt::volume_sliver::quality_error)
    };
    let mut choices = choices::edge_star_choices_prepared(
        input.topology,
        source_vertices,
        &source.region_id,
        choices::InsertionProtection {
            faces: &visibility_barriers,
            edges: &protected_faces
                .iter()
                .flat_map(|face| {
                    let [a, b, c] = *face;
                    [[a, b], [a, c], [b, c]].map(|mut edge| {
                        edge.sort_unstable();
                        edge
                    })
                })
                .collect(),
            segments: &protected_segments,
        },
        &metrics,
        cancellation,
    )?;
    if matches!(
        refinement_policy,
        crate::cdt::volume_sliver::admission::CandidateRefinementPolicy::AllowForCoupledConvergence
    ) {
        let mut front = offcenter::choices(
            input.topology,
            source,
            &visibility_barriers,
            input.quality_options.minimum_metric_scaled_jacobian,
            cancellation,
        )?;
        front.extend(choices);
        choices = front;
    }
    for mut choice in choices {
        work.evaluate()?;
        if let InsertionStrategy::Cavity { neighbor_index } = choice.strategy {
            if let Some(coordinates_m) = smooth_visibility_cavity_point(
                input.topology,
                choice.coordinates_m.canonical_value(),
                VisibilityContext {
                    source_vertices,
                    neighbor_index,
                    protected_faces: &visibility_barriers,
                    metrics: &metrics,
                    target_scaled_jacobian: input.quality_options.minimum_metric_scaled_jacobian,
                    options,
                    cancellation,
                },
            )? {
                choice.coordinates_m = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m);
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
        if input.topology.nodes.iter().any(|existing| {
            existing.identity == node.identity
                || existing.coordinates_m.canonical_value() == node.coordinates_m.canonical_value()
        }) {
            reject(
                &mut rejection_counts,
                "candidate_collides_with_existing_node",
            );
            continue;
        }
        let topology = match choice.strategy {
            InsertionStrategy::Cavity { neighbor_index }
            | InsertionStrategy::BoundaryOffcenter { neighbor_index } => {
                match split_two_tetrahedron_cavity(
                    input.topology,
                    node,
                    VisibilityContext {
                        source_vertices,
                        neighbor_index,
                        protected_faces: &visibility_barriers,
                        metrics: &metrics,
                        target_scaled_jacobian: input
                            .quality_options
                            .minimum_metric_scaled_jacobian,
                        options,
                        cancellation,
                    },
                    work,
                )? {
                    EdgeStarSplit::Topology(topology) => topology,
                    EdgeStarSplit::Rejected(detail) => {
                        reject(&mut rejection_counts, "cavity_insertion_rejected");
                        first_rejection_details
                            .entry("cavity_insertion_rejected")
                            .or_insert(detail);
                        continue;
                    }
                }
            }
            InsertionStrategy::EdgeStar => match split_complete_edge_star(
                input.topology,
                node,
                choice.edge_identities,
                options,
                cancellation,
            )? {
                EdgeStarSplit::Topology(topology) => topology,
                EdgeStarSplit::Rejected(edge_star_detail) => {
                    match insert_delaunay_volume_node_locally_with_barriers(
                        input.topology.clone(),
                        node,
                        &protected_faces,
                        options.insertion,
                        cancellation,
                    ) {
                        Ok(topology) => CanonicalVolumeTopology::validate(
                            topology,
                            options.insertion.topology,
                            cancellation,
                        )
                        .map_err(|failure| {
                            let kind = match failure.kind {
                                DelaunayTopologyErrorKind::Cancelled => {
                                    DelaunayVolumeSliverErrorKind::Cancelled
                                }
                                DelaunayTopologyErrorKind::ResourceLimit => {
                                    DelaunayVolumeSliverErrorKind::ResourceLimit
                                }
                                _ => DelaunayVolumeSliverErrorKind::InvalidTopology,
                            };
                            error(kind, failure.to_string())
                        })?,
                        Err(failure)
                            if matches!(
                                failure.kind,
                                DelaunayInsertionErrorKind::InvalidNode
                                    | DelaunayInsertionErrorKind::InvalidTopology
                                    | DelaunayInsertionErrorKind::PointOutsideTopology
                            ) =>
                        {
                            reject(&mut rejection_counts, "edge_star_insertion_rejected");
                            first_rejection_details
                            .entry("edge_star_insertion_rejected")
                            .or_insert_with(|| {
                                format!(
                                    "edge-star split: {edge_star_detail}; constrained insertion: {failure}"
                                )
                            });
                            continue;
                        }
                        Err(failure) => return Err(insertion_error(failure)),
                    }
                }
            },
        };
        let Some(quality) = evaluate_canonical_candidate_unranked(
            &topology,
            input,
            &protected_faces,
            options,
            cancellation,
            &mut rejection_counts,
        )?
        else {
            continue;
        };
        let candidate_spectrum = quality_spectrum(&quality, objective);
        if best_unranked_spectrum.as_ref().is_none_or(|best| {
            compare_spectra(&candidate_spectrum, best) == std::cmp::Ordering::Less
        }) {
            best_unranked_spectrum = Some(candidate_spectrum);
        }
        let insertion = DelaunayVolumeSliverMutation::Inserted(DelaunayVolumeSliverInsertion {
            inserted_node: node,
            source_tetrahedron_node_identities: source.node_identities,
            source_edge_node_identities: choice.edge_identities,
            construction_point_index: choice.index,
        });
        if remaining_lookahead_depth > 0 && best_candidate.is_none() {
            lookahead_seeds.retain_with(
                &(&topology, &quality),
                |(topology, quality), seed: &RetainedInsertion| {
                    seed.compare_candidate(quality, input.quality, policy)
                        .then_with(|| {
                            topology
                                .nodes
                                .last()
                                .map(|node| node.identity)
                                .cmp(&seed.last_node_identity)
                        })
                },
                || RetainedInsertion::capture(input, &topology, &quality, vec![insertion.clone()]),
            )?;
        }
        if candidate_quality_is_admissible(
            &quality,
            input,
            current_spectrum,
            refinement_policy,
            objective,
            &mut rejection_counts,
        ) {
            retain_better_candidate(
                &mut best_candidate,
                input,
                &topology,
                &quality,
                vec![insertion.clone()],
                policy,
            )?;
        }
        let Some(reconnection_source) =
            super::super::admission::source_for_policy(&quality, &topology, policy)
        else {
            continue;
        };
        let candidate_input = DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: input.metric_request,
            provenance: input.provenance,
            quality: &quality,
            quality_options: input.quality_options,
            fixed_node_identities: input.fixed_node_identities,
        };
        let reconnection = reconnection_candidates(
            candidate_input,
            reconnection_source,
            current_spectrum,
            cancellation,
            work,
            SliverSearchPolicy {
                options,
                refinement: refinement_policy,
                objective,
            },
            super::super::reconnection::ReconnectionMode {
                allow_lookahead: false,
                maximum_candidates: Some(options.maximum_compound_reconnections_per_insertion),
            },
        )?;
        if let Some(reconnection) = reconnection.candidate {
            let mut mutations = vec![insertion];
            mutations.extend(reconnection.mutations);
            retain_better_candidate(
                &mut best_candidate,
                input,
                &reconnection.topology,
                &reconnection.quality,
                mutations,
                policy,
            )?;
        }
    }
    if best_candidate.is_none() && remaining_lookahead_depth > 0 {
        for seed in lookahead_seeds.into_entries() {
            let seed = seed.materialize(input, options, cancellation)?;
            let Some(next_source) =
                super::super::admission::source_for_policy(&seed.quality, &seed.topology, policy)
            else {
                continue;
            };
            let seed_input = DelaunayVolumeRefinementInput {
                topology: &seed.topology,
                metric_request: input.metric_request,
                provenance: input.provenance,
                quality: &seed.quality,
                quality_options: input.quality_options,
                fixed_node_identities: input.fixed_node_identities,
            };
            let relocation = super::super::relocation::relocation_candidates(
                seed_input,
                next_source,
                current_spectrum,
                cancellation,
                work,
                SliverSearchPolicy {
                    options,
                    refinement: refinement_policy,
                    objective,
                },
                super::super::relocation::RelocationSearchScope::FixedStarOnly,
            )?;
            if let Some(relocation) = relocation.candidate {
                return Ok(InsertionSearch {
                    candidate: Some(InsertionCandidate {
                        topology: relocation.topology,
                        quality: relocation.quality,
                        mutations: {
                            let mut mutations = seed.mutations.clone();
                            mutations.push(relocation.mutation);
                            mutations
                        },
                    }),
                    rejection_counts,
                    first_rejection_details,
                });
            }
            let next = insertion_candidates(
                seed_input,
                next_source,
                current_spectrum,
                cancellation,
                work,
                SliverSearchPolicy {
                    options,
                    refinement: refinement_policy,
                    objective,
                },
                remaining_lookahead_depth - 1,
            )?;
            if let Some(next) = next.candidate {
                let mut mutations = seed.mutations;
                mutations.extend(next.mutations);
                return Ok(InsertionSearch {
                    candidate: Some(InsertionCandidate {
                        topology: next.topology,
                        quality: next.quality,
                        mutations,
                    }),
                    rejection_counts,
                    first_rejection_details,
                });
            }
            for (reason, count) in next.rejection_counts {
                *rejection_counts.entry(reason).or_default() += count;
            }
            for (reason, detail) in next.first_rejection_details {
                first_rejection_details.entry(reason).or_insert(detail);
            }
        }
    }
    if let Some(spectrum) = best_unranked_spectrum {
        first_rejection_details.insert(
            "best_unranked_optimization_spectrum",
            format!("{:?}", spectrum.into_iter().take(8).collect::<Vec<_>>()),
        );
    }
    Ok(InsertionSearch {
        candidate: best_candidate
            .map(|candidate| candidate.materialize(input, options, cancellation))
            .transpose()?,
        rejection_counts,
        first_rejection_details,
    })
}

fn retain_better_candidate(
    best: &mut Option<RetainedInsertion>,
    base: DelaunayVolumeRefinementInput<'_>,
    topology: &DelaunayVolumeTopology,
    quality: &DelaunayVolumeQuality,
    mutations: Vec<DelaunayVolumeSliverMutation>,
    policy: SliverSearchPolicy,
) -> Result<(), DelaunayVolumeSliverError> {
    let replace = best.as_ref().is_none_or(|current| {
        current.compare_candidate(quality, base.quality, policy) == std::cmp::Ordering::Less
    });
    if replace {
        *best = Some(RetainedInsertion::capture(
            base, topology, quality, mutations,
        )?);
    }
    Ok(())
}
