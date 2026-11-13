//! One-node candidate evaluation and ownership transfer; scheduling stays with the parent.
use super::repair_state::{RepairState, StarSignature};
use super::*;
pub(super) struct NodeOptimizer<'source, 'view, 'work, 'state> {
    pub(super) input: &'view ProjectionInput<'source>,
    pub(super) evaluation: PreparedEvaluation<'source, 'view>,
    pub(super) options: DelaunaySolverTopologyOptions,
    pub(super) cancellation: &'work dyn MeshingCancellationSignal,
    pub(super) jacobian_work: &'work mut u64,
    pub(super) candidate_work: &'work mut u64,
    pub(super) improve_valid_nodes: bool,
    pub(super) invalid_elements: &'work [usize],
    pub(super) counters: &'work super::super::counters::ElevationCounters,
    pub(super) repair: &'work mut RepairState<'state>,
}

impl NodeOptimizer<'_, '_, '_, '_> {
    pub(super) fn optimize(
        &mut self,
        topology: &mut SolverMeshTopology,
        movable: &MovableNode,
    ) -> Result<bool, DelaunaySolverTopologyError> {
        let slot = geometry::node_slot(&topology.nodes, movable.node_id)?;
        let original = &topology.nodes[slot];
        let initial_is_valid = movable
            .incident_elements
            .iter()
            .all(|element| self.invalid_elements.binary_search(element).is_err());
        if initial_is_valid {
            if !self.improve_valid_nodes {
                return Ok(false);
            }
            let left = &topology.nodes[geometry::node_slot(&topology.nodes, movable.edge[0])?];
            let right = &topology.nodes[geometry::node_slot(&topology.nodes, movable.edge[1])?];
            if squared_distance(
                original.coordinates_m.canonical_value(),
                midpoint(
                    left.coordinates_m.canonical_value(),
                    right.coordinates_m.canonical_value(),
                ),
            ) == 0.0
            {
                return Ok(false);
            }
        }
        let moved = transaction::edit_nodes(
            topology,
            &[slot],
            self.evaluation.control,
            |topology, control| {
                let mut operation = NodeOptimizer {
                    input: self.input,
                    evaluation: PreparedEvaluation {
                        control,
                        ..self.evaluation
                    },
                    options: self.options,
                    cancellation: self.cancellation,
                    jacobian_work: self.jacobian_work,
                    candidate_work: self.candidate_work,
                    improve_valid_nodes: self.improve_valid_nodes,
                    invalid_elements: self.invalid_elements,
                    counters: self.counters,
                    repair: &mut *self.repair,
                };
                operation.optimize_node(topology, movable)
            },
        )?;
        Ok(moved != 0)
    }
    fn optimize_node(
        &mut self,
        topology: &mut SolverMeshTopology,
        movable: &MovableNode,
    ) -> Result<bool, DelaunaySolverTopologyError> {
        let node_index = geometry::node_slot(&topology.nodes, movable.node_id)?;
        let storage = crate::cdt::solver_topology::storage::Storage(Some(self.evaluation.control));
        let left = storage
            .node(&topology.nodes[geometry::node_slot(&topology.nodes, movable.edge[0])?])?;
        let right = storage
            .node(&topology.nodes[geometry::node_slot(&topology.nodes, movable.edge[1])?])?;
        let target = midpoint(
            left.coordinates_m.canonical_value(),
            right.coordinates_m.canonical_value(),
        );
        let initial = storage.node(&topology.nodes[node_index])?;
        let initial_objective = squared_distance(initial.coordinates_m.canonical_value(), target);
        let initial_is_valid = movable
            .incident_elements
            .iter()
            .all(|element| self.invalid_elements.binary_search(element).is_err());
        if initial_is_valid && !self.improve_valid_nodes {
            return Ok(false);
        }
        if !initial_is_valid {
            return self.repair_invalid(
                topology,
                RepairRequest {
                    movable,
                    node_index,
                    endpoints: [left, right],
                    target,
                    initial,
                },
            );
        }
        // Squared distance is nonnegative. No exact-geometry candidate can improve
        // a valid node already at its arithmetic midpoint, including in mixed meshes.
        if initial_objective == 0.0 {
            return Ok(false);
        }
        let storage = crate::cdt::solver_topology::storage::Storage(Some(self.evaluation.control));
        let mut candidates = storage.vector(0)?;

        for fraction in SHAPE_CANDIDATE_FRACTIONS {
            if let Some(candidate) = self.candidate(movable, &left, &right, target, fraction)? {
                if candidate.objective < initial_objective {
                    storage.push(&mut candidates, candidate)?;
                }
            }
        }
        candidates.sort_by(|left, right| {
            left.objective
                .total_cmp(&right.objective)
                .then_with(|| left.fraction.total_cmp(&right.fraction))
        });

        for candidate in candidates {
            topology.nodes[node_index].coordinates_m =
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                    candidate.coordinates_m.canonical_value(),
                );
            topology.nodes[node_index].exact_parameters =
                storage.parameters(&candidate.exact_parameters)?;
            match jacobian::validate_elements(
                topology,
                movable.incident_elements.iter().copied(),
                jacobian::ELEMENT_SEARCH_WORK,
                self.input.request.limits.maximum_recursion_depth,
                self.options.cancellation_check_interval,
                self.cancellation,
                jacobian::ValidationWork {
                    counter: self.jacobian_work,
                    storage: crate::cdt::solver_topology::storage::Storage(Some(
                        self.evaluation.control,
                    )),
                },
            ) {
                Ok(()) => return Ok(topology.nodes[node_index] != initial),
                Err(failure) if failure.kind == DelaunaySolverTopologyErrorKind::InvalidMesh => {
                    topology.nodes[node_index] = storage.node(&initial)?;
                }
                Err(failure) => {
                    topology.nodes[node_index] = initial;
                    return Err(failure);
                }
            }
        }
        topology.nodes[node_index] = initial;
        Ok(false)
    }

    fn repair_invalid(
        &mut self,
        topology: &mut SolverMeshTopology,
        request: RepairRequest<'_>,
    ) -> Result<bool, DelaunaySolverTopologyError> {
        use super::super::counters::ElevationCounters as C;
        let signature = StarSignature::of(
            topology,
            request.movable,
            [
                &request.initial,
                &request.endpoints[0],
                &request.endpoints[1],
            ],
        )?;
        if self
            .repair
            .declined_unchanged(request.movable.node_id, &signature)
        {
            C::add(&self.counters.repairs_unchanged, 1);
            return Ok(false);
        }
        C::add(&self.counters.repairs, 1);
        let storage = crate::cdt::solver_topology::storage::Storage(Some(self.evaluation.control));
        let mut candidates = storage.vector(0)?;
        for fraction in REPAIR_CANDIDATE_FRACTIONS {
            let Some(candidate) = self.candidate(
                request.movable,
                &request.endpoints[0],
                &request.endpoints[1],
                request.target,
                fraction,
            )?
            else {
                continue;
            };
            storage.push(&mut candidates, candidate)?;
        }
        if request.movable.exact_edge_id.is_none() && request.movable.surface_uses.is_empty() {
            for (index, coordinates_m) in candidates::interior_coordinates(
                request.initial.coordinates_m.canonical_value(),
                request
                    .endpoints
                    .each_ref()
                    .map(|node| node.coordinates_m.canonical_value()),
            )
            .enumerate()
            {
                consume_candidate_work(self.candidate_work, self.options, self.cancellation)?;
                storage.push(
                    &mut candidates,
                    Candidate {
                        fraction: 2.0 + index as f64,
                        objective: squared_distance(coordinates_m, request.target),
                        coordinates_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(coordinates_m),
                        exact_parameters: Vec::new(),
                    },
                )?;
            }
        } else if request.movable.exact_edge_id.is_none() && request.movable.surface_uses.len() == 1
        {
            let surface_use = &request.movable.surface_uses[0];
            for (index, evaluator_uv) in candidates::surface_parameters(
                &request.initial,
                [&request.endpoints[0], &request.endpoints[1]],
                surface_use,
            )
            .enumerate()
            {
                consume_candidate_work(self.candidate_work, self.options, self.cancellation)?;
                let Some(point) = geometry::surface_geometry(
                    self.evaluation,
                    self.options,
                    surface_use,
                    evaluator_uv,
                )?
                else {
                    continue;
                };
                storage.push(
                    &mut candidates,
                    Candidate {
                        fraction: 2.0 + index as f64,
                        objective: squared_distance(
                            point.coordinates_m.canonical_value(),
                            request.target,
                        ),
                        coordinates_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(
                            point.coordinates_m.canonical_value()
                        ),
                        exact_parameters: point.exact_parameters,
                    },
                )?;
            }
        }
        C::add(&self.counters.repair_candidates, candidates.len() as u64);
        if candidates.is_empty() {
            // No candidate can improve on the initial profile, so it is not taken.
            C::add(&self.counters.repairs_without_candidates, 1);
            topology.nodes[request.node_index] = request.initial;
            self.repair.decline(request.movable.node_id, signature)?;
            return Ok(false);
        }
        let initial_profile = self.repair.profile(
            topology,
            request.movable.incident_elements,
            self.cancellation,
            self.counters,
        )?;
        let profiles = candidate_profiles(
            topology,
            request.movable,
            &candidates,
            self.cancellation,
            self.counters,
        )?;
        let mut best = None::<usize>;
        for (index, profile) in profiles.iter().enumerate() {
            if profile_improves(
                &initial_profile,
                best.map(|best| profiles[best].as_slice()),
                profile,
            ) {
                best = Some(index);
            }
        }
        if let Some(best) = best {
            let candidate = candidates.swap_remove(best);
            topology.nodes[request.node_index].coordinates_m =
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                    candidate.coordinates_m.canonical_value(),
                );
            topology.nodes[request.node_index].exact_parameters = candidate.exact_parameters;
            let moved = topology.nodes[request.node_index] != request.initial;
            if moved {
                C::add(&self.counters.repair_moves, 1);
                self.repair.moved(request.movable.node_id);
                return Ok(true);
            }
            topology.nodes[request.node_index] = request.initial;
            self.repair.decline(request.movable.node_id, signature)?;
            return Ok(false);
        }
        topology.nodes[request.node_index] = request.initial;
        self.repair.decline(request.movable.node_id, signature)?;
        Ok(false)
    }

    fn candidate(
        &mut self,
        movable: &MovableNode,
        left: &runmat_meshing_core::SolverMeshNode,
        right: &runmat_meshing_core::SolverMeshNode,
        target: [f64; 3],
        fraction: f64,
    ) -> Result<Option<Candidate>, DelaunaySolverTopologyError> {
        consume_candidate_work(self.candidate_work, self.options, self.cancellation)?;
        Ok(geometry::edge_geometry(
            self.input,
            self.evaluation,
            self.options,
            geometry::EdgeGeometryRequest {
                left,
                right,
                exact_edge_id: movable.exact_edge_id,
                surface_uses: movable.surface_uses,
                fraction,
            },
        )?
        .and_then(|point| {
            let objective = squared_distance(point.coordinates_m.canonical_value(), target);
            objective.is_finite().then_some(Candidate {
                fraction,
                objective,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(point.coordinates_m.canonical_value()),
                exact_parameters: point.exact_parameters,
            })
        }))
    }
}

/// The sorted margin profile of the node's incident elements with the node at each candidate, in
/// candidate order. It equals placing the node at each candidate in turn and profiling its
/// incident elements; the elements are independent, so they are profiled on all admitted cores.
pub(super) fn candidate_profiles(
    topology: &SolverMeshTopology,
    movable: &MovableNode,
    candidates: &[Candidate],
    cancellation: &dyn MeshingCancellationSignal,
    counters: &super::super::counters::ElevationCounters,
) -> Result<Vec<Vec<u64>>, DelaunaySolverTopologyError> {
    use super::super::counters::ElevationCounters as C;
    let elements = movable.incident_elements;
    if elements.is_empty() {
        return Err(invalid("Tet10 optimization margin profile is empty"));
    }
    let mut bases = Vec::with_capacity(elements.len());
    for &element in elements {
        let coordinates = jacobian::element_coordinates(topology, element)?;
        let slots = topology.volume_elements[element]
            .node_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == movable.node_id)
            .map(|(slot, _)| slot)
            .collect::<Vec<_>>();
        bases.push((coordinates, slots));
    }
    let tasks = (0..candidates.len())
        .flat_map(|candidate| (0..elements.len()).map(move |element| (candidate, element)))
        .collect::<Vec<_>>();
    let margins = runmat_cpu_execution::ordered_map(&tasks, |&(candidate, element)| {
        let (mut coordinates, slots) = bases[element].clone();
        let point = candidates[candidate].coordinates_m.canonical_value();
        for slot in slots {
            coordinates[slot] = point;
        }
        jacobian::margin_key(coordinates, cancellation)
    });
    C::add(&counters.profiles, candidates.len() as u64);
    C::add(&counters.profile_margins, tasks.len() as u64);
    let mut profiles = Vec::with_capacity(candidates.len());
    let mut margins = margins.into_iter();
    for _ in candidates {
        let mut profile = Vec::with_capacity(elements.len());
        for margin in margins.by_ref().take(elements.len()) {
            profile.push(margin?);
        }
        profile.sort_unstable();
        profiles.push(profile);
    }
    Ok(profiles)
}
