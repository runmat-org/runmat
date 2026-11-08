//! Interior relocation changes coordinates only. Connectivity and exact histories
//! remain borrowed; every attempted coordinate is restored before an error escapes.
use super::{consume_candidate_work, geometry, jacobian, OptimizationTopology, OptimizationWork};
use crate::cdt::solver_topology::{
    error, storage::Storage, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
};
use runmat_geometry_core::{GeometryAllocationScope, GeometryEvaluationControl};
use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    SolverMeshTopology,
};
const RELOCATION_FRACTIONS: [f64; 6] = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75];

pub(in crate::cdt::solver_topology::order_elevation) fn relocate_interior_corner(
    topology: &mut SolverMeshTopology,
    element_index: usize,
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    indexes: &OptimizationTopology<'_>,
    control: &dyn GeometryEvaluationControl,
    work: OptimizationWork<'_>,
) -> Result<u64, DelaunaySolverTopologyError> {
    let OptimizationWork {
        options,
        cancellation,
        jacobian_work,
        candidate_work,
        counters: _,
    } = work;
    let element = topology
        .volume_elements
        .get(element_index)
        .ok_or_else(|| invalid("interior-corner repair element index is out of bounds"))?;
    let corners: [u64; 4] = element.node_ids[..4]
        .try_into()
        .map_err(|_| invalid("interior-corner repair requires four corners"))?;
    for node_id in corners {
        let index = geometry::node_slot(&topology.nodes, node_id)?;
        if indexes.is_boundary(index) {
            continue;
        }
        let scope = GeometryAllocationScope::new(control);
        let storage = Storage(Some(&scope));
        let incident = indexes.incidence.elements(node_id)?;
        let original_signs = corner_signs(topology, incident, storage)?;
        let count = incident
            .len()
            .checked_mul(3)
            .ok_or_else(|| super::resource("interior-corner neighbor storage overflow"))?;
        let mut neighbors = storage.vector(count)?;
        for &element in incident {
            scope.checkpoint().map_err(error::geometry)?;
            neighbors.extend(
                topology.volume_elements[element].node_ids[..4]
                    .iter()
                    .copied()
                    .filter(|id| *id != node_id),
            );
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        if neighbors.is_empty() {
            continue;
        }
        let target = neighbors
            .iter()
            .try_fold([0.0; 3], |mut sum, id| {
                let point = topology.nodes[geometry::node_slot(&topology.nodes, *id)?]
                    .coordinates_m
                    .canonical_value();
                for axis in 0..3 {
                    sum[axis] += point[axis];
                }
                Ok::<_, DelaunaySolverTopologyError>(sum)
            })?
            .map(|sum| sum / neighbors.len() as f64);
        let initial = topology.nodes[index].coordinates_m.canonical_value();
        let initial_profile = jacobian::optimization_margin_profile_elements(
            topology,
            incident.iter().copied(),
            cancellation,
            storage,
        )?;
        let mut best = None;
        let mut best_profile = storage.vector(initial_profile.len())?;
        for fraction in RELOCATION_FRACTIONS {
            consume_candidate_work(candidate_work, options, cancellation)?;
            topology.nodes[index].coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(std::array::from_fn(|axis| {
                initial[axis] * (1.0 - fraction) + target[axis] * fraction
            }));
            // Restoration is unconditional even when an orientation/profile/admission check fails.
            let result = (|| {
                let attempt = GeometryAllocationScope::new(control);
                let storage = Storage(Some(&attempt));
                if corner_signs(topology, incident, storage)? != original_signs {
                    return Ok(false);
                }
                match jacobian::validate_elements(
                    topology,
                    incident.iter().copied(),
                    maximum_search_work,
                    maximum_recursion_depth,
                    options.cancellation_check_interval,
                    cancellation,
                    jacobian::ValidationWork {
                        counter: jacobian_work,
                        storage,
                    },
                ) {
                    Ok(()) => Ok(true),
                    Err(failure)
                        if failure.kind == DelaunaySolverTopologyErrorKind::InvalidMesh =>
                    {
                        let profile = jacobian::optimization_margin_profile_elements(
                            topology,
                            incident.iter().copied(),
                            cancellation,
                            storage,
                        )?;
                        if profile > initial_profile
                            && (best.is_none() || profile.as_slice() > best_profile.as_slice())
                        {
                            best_profile.clear();
                            best_profile.extend_from_slice(&profile);
                            best = Some(topology.nodes[index].coordinates_m.canonical_value());
                        }
                        Ok(false)
                    }
                    Err(failure) => Err(failure),
                }
            })();
            let coordinates = topology.nodes[index].coordinates_m.canonical_value();
            topology.nodes[index].coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(initial);
            if result? {
                topology.nodes[index].coordinates_m = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates);
                return Ok(1);
            }
        }
        if let Some(coordinates) = best {
            topology.nodes[index].coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates);
            return Ok(1);
        }
    }
    Ok(0)
}
fn corner_signs(
    topology: &SolverMeshTopology,
    elements: &[usize],
    storage: Storage<'_>,
) -> Result<Vec<PredicateSign>, DelaunaySolverTopologyError> {
    let mut signs = storage.vector(elements.len())?;
    for &index in elements {
        if let Some(control) = storage.0 {
            control.checkpoint().map_err(error::geometry)?;
        }
        let element = topology
            .volume_elements
            .get(index)
            .ok_or_else(|| invalid("interior-corner incident element is absent"))?;
        let mut points = [[0.0; 3]; 4];
        for (point, id) in points.iter_mut().zip(&element.node_ids[..4]) {
            *point = topology.nodes[geometry::node_slot(&topology.nodes, *id)?]
                .coordinates_m
                .canonical_value();
        }
        signs.push(orient3d(points).map_err(|failure| {
            invalid(format!("interior-corner orientation failed: {failure:?}"))
        })?);
    }
    Ok(signs)
}
fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
