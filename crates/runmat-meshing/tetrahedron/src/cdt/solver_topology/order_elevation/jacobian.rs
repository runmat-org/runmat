use crate::cdt::solver_topology::storage::{ReservedBuffer, Storage};

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    MeshingCancellationSignal, SolverMeshTopology,
};

use crate::cdt::solver_topology::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
};

mod exact;
mod interval;

use exact::{build_exact_reference_jacobians, exact_bernstein_coefficients_are_positive};
use interval::{
    determinant_columns, determinant_enclosure, has_certified_nonpositive_vertex, jacobian_at,
    mixed_determinant, Interval,
};

const BARYCENTRIC_DERIVATIVES: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];
const REFERENCE_TETRAHEDRON: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub(super) struct ValidationWork<'storage, 'work> {
    pub counter: &'work mut u64,
    pub storage: Storage<'storage>,
}

pub(super) fn validate_elements(
    topology: &SolverMeshTopology,
    element_indices: impl IntoIterator<Item = usize>,
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation_check_interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: ValidationWork<'_, '_>,
) -> Result<(), DelaunaySolverTopologyError> {
    for element_index in element_indices {
        let element = topology
            .volume_elements
            .get(element_index)
            .ok_or_else(|| invalid("Tet10 Jacobian element index is out of bounds"))?;
        let coordinates = element_coordinates_from_topology(element, topology)?;
        certify_element(
            coordinates,
            maximum_search_work,
            maximum_recursion_depth,
            cancellation_check_interval,
            cancellation,
            ValidationWork {
                counter: work.counter,
                storage: work.storage,
            },
        )
        .map_err(|failure| DelaunaySolverTopologyError {
            kind: failure.kind,
            reason: format!(
                "Tet10 element {} (stable identity {:?}) failed certification: {}",
                element.element_id, element.stable_identity, failure.reason
            ),
        })?;
    }
    Ok(())
}

pub(super) fn invalid_element_indices<'storage>(
    topology: &SolverMeshTopology,
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation_check_interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: ValidationWork<'storage, '_>,
) -> Result<ReservedBuffer<'storage, usize>, DelaunaySolverTopologyError> {
    let mut invalid_indices = ReservedBuffer::new(topology.volume_elements.len(), work.storage)?;
    for (element_index, element) in topology.volume_elements.iter().enumerate() {
        let coordinates = element_coordinates_from_topology(element, topology)?;
        match certify_element(
            coordinates,
            maximum_search_work,
            maximum_recursion_depth,
            cancellation_check_interval,
            cancellation,
            ValidationWork {
                counter: work.counter,
                storage: work.storage,
            },
        ) {
            Ok(()) => {}
            Err(failure) if failure.kind == DelaunaySolverTopologyErrorKind::InvalidMesh => {
                invalid_indices.push(element_index)?;
            }
            Err(failure) => return Err(failure),
        }
    }
    Ok(invalid_indices)
}

pub(super) fn optimization_margin_profile_elements<'storage>(
    topology: &SolverMeshTopology,
    element_indices: impl Iterator<Item = usize> + Clone,
    cancellation: &dyn MeshingCancellationSignal,
    storage: Storage<'storage>,
) -> Result<ReservedBuffer<'storage, u64>, DelaunaySolverTopologyError> {
    let mut profile = ReservedBuffer::new(element_indices.clone().count(), storage)?;
    for element_index in element_indices {
        margin_checkpoint(cancellation)?;
        let element = topology
            .volume_elements
            .get(element_index)
            .ok_or_else(|| invalid("Tet10 optimization element index is out of bounds"))?;
        profile.push(margin_priority_key(bernstein_margin(
            element_coordinates_from_topology(element, topology)?,
            cancellation,
        )?))?;
    }
    profile.sort_unstable();
    if profile.is_empty() {
        return Err(invalid("Tet10 optimization margin profile is empty"));
    }
    Ok(profile)
}

pub(super) fn worst_invalid_element(
    topology: &SolverMeshTopology,
    invalid_elements: &[usize],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<usize>, DelaunaySolverTopologyError> {
    let mut worst = None::<(u64, usize)>;
    for &element_index in invalid_elements {
        margin_checkpoint(cancellation)?;
        let element = topology
            .volume_elements
            .get(element_index)
            .ok_or_else(|| invalid("Tet10 optimization element index is out of bounds"))?;
        let margin = bernstein_margin(
            element_coordinates_from_topology(element, topology)?,
            cancellation,
        )?;
        let key = margin_priority_key(margin);
        if worst.as_ref().is_none_or(|(worst_key, worst_index)| {
            key < *worst_key || (key == *worst_key && element_index < *worst_index)
        }) {
            worst = Some((key, element_index));
        }
    }
    Ok(worst.map(|(_, index)| index))
}

pub(super) fn margin_priority_key(margin: f64) -> u64 {
    const DISCARDED_MANTISSA_BITS: u32 = 12;
    crate::cdt::volume_quality::ordered_float_key(margin) >> DISCARDED_MANTISSA_BITS
}

fn element_coordinates_from_topology(
    element: &runmat_meshing_core::SolverVolumeElement,
    topology: &SolverMeshTopology,
) -> Result<[[f64; 3]; 10], DelaunaySolverTopologyError> {
    if element.node_ids.len() != 10 {
        return Err(invalid("Tet10 Jacobian requires exactly ten nodes"));
    }
    let mut coordinates = [[0.0; 3]; 10];
    for (coordinate, node_id) in coordinates.iter_mut().zip(&element.node_ids) {
        let index = topology
            .nodes
            .binary_search_by_key(node_id, |node| node.node_id)
            .map_err(|_| invalid("Tet10 Jacobian references an absent node"))?;
        *coordinate = topology.nodes[index].coordinates_m.canonical_value();
    }
    Ok(coordinates)
}

fn bernstein_margin(
    coordinates: [[f64; 3]; 10],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<f64, DelaunaySolverTopologyError> {
    margin_checkpoint(cancellation)?;
    let orientation = match orient3d([
        coordinates[0],
        coordinates[1],
        coordinates[2],
        coordinates[3],
    ])
    .map_err(|failure| invalid(format!("Tet10 corner orientation failed: {failure:?}")))?
    {
        PredicateSign::Positive => -1.0,
        PredicateSign::Negative => 1.0,
        PredicateSign::Zero => return Err(invalid("Tet10 corners are exactly coplanar")),
    };
    let corner_scale = determinant_columns([
        std::array::from_fn(|axis| Interval::exact(coordinates[1][axis] - coordinates[0][axis])),
        std::array::from_fn(|axis| Interval::exact(coordinates[2][axis] - coordinates[0][axis])),
        std::array::from_fn(|axis| Interval::exact(coordinates[3][axis] - coordinates[0][axis])),
    ]);
    let scale = corner_scale.lower.abs().max(corner_scale.upper.abs());
    if !scale.is_finite() || scale == 0.0 {
        return Err(invalid("Tet10 optimization has no finite corner scale"));
    }
    let jacobians = REFERENCE_TETRAHEDRON.map(|barycentric| jacobian_at(coordinates, barycentric));
    let mut margin = f64::INFINITY;
    for first in 0..4 {
        for second in first..4 {
            for third in second..4 {
                margin_checkpoint(cancellation)?;
                let coefficient =
                    mixed_determinant([&jacobians[first], &jacobians[second], &jacobians[third]])
                        .scaled(orientation);
                let midpoint = coefficient.lower * 0.5 + coefficient.upper * 0.5;
                margin = margin.min(midpoint / scale);
            }
        }
    }
    if margin.is_finite() {
        Ok(margin)
    } else {
        Err(invalid("Tet10 optimization margin is not finite"))
    }
}

fn margin_checkpoint(
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySolverTopologyError> {
    if cancellation.is_cancelled() {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::Cancelled,
            "cancelled",
        ));
    }
    Ok(())
}

/// Subdivision work one element may spend to certify its Jacobian. Certification is charged per
/// element, so total work scales with the mesh and one hard element cannot exhaust the others'.
/// It is a property of the algorithm, not of the machine: an element that cannot be certified
/// within it is not admitted (and goes to repair) on every host alike.
pub(super) const ELEMENT_SEARCH_WORK: u64 = 1 << 16;

fn certify_element(
    coordinates: [[f64; 3]; 10],
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation_check_interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: ValidationWork<'_, '_>,
) -> Result<(), DelaunaySolverTopologyError> {
    let storage_scope = work
        .storage
        .0
        .map(runmat_geometry_core::GeometryAllocationScope::new);
    let storage = Storage(
        storage_scope
            .as_ref()
            .map(|scope| scope as &dyn runmat_geometry_core::GeometryEvaluationControl),
    );
    let total_work = work.counter;
    let mut element_work = 0_u64;
    let result = certify_element_within(
        coordinates,
        maximum_search_work,
        maximum_recursion_depth,
        cancellation,
        &storage,
        &mut element_work,
        cancellation_check_interval,
    );
    *total_work = total_work.saturating_add(element_work);
    match result {
        Err(failure)
            if failure.kind == DelaunaySolverTopologyErrorKind::ResourceLimit
                && element_work > maximum_search_work =>
        {
            Err(invalid(format!(
                "Tet10 isoparametric Jacobian could not be certified positive within the element's search work of {maximum_search_work}"
            )))
        }
        // A positive Jacobian is necessary but not sufficient: the element must also pass the
        // shared Tet10 rule (Jacobian margin and stiffness-quadrature convergence) that analysis
        // applies before integrating it, so the mesher never publishes an element it rejects.
        Ok(()) => runmat_meshing_core::tetrahedron10_stiffness_moments(&coordinates)
            .map(|_| ())
            .map_err(|invalidity| match invalidity {
                runmat_meshing_core::Tet10ElementInvalidity::Accounting(error) =>
                    error::failure(DelaunaySolverTopologyErrorKind::Accounting(error), error.to_string()),
                other => invalid(format!(
                    "Tet10 element fails the shared element-validity rule: {other:?}"
                )),
            }),
        result => result,
    }
}

fn certify_element_within(
    coordinates: [[f64; 3]; 10],
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation: &dyn MeshingCancellationSignal,
    storage: &Storage<'_>,
    work: &mut u64,
    cancellation_check_interval: u64,
) -> Result<(), DelaunaySolverTopologyError> {
    if cancellation_check_interval == 0 {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::InvalidOptions,
            "Tet10 Jacobian cancellation-check interval must be greater than zero",
        ));
    }
    let orientation = match orient3d([
        coordinates[0],
        coordinates[1],
        coordinates[2],
        coordinates[3],
    ])
    .map_err(|failure| invalid(format!("Tet10 corner orientation failed: {failure:?}")))?
    {
        PredicateSign::Positive => -1.0,
        PredicateSign::Negative => 1.0,
        PredicateSign::Zero => return Err(invalid("Tet10 corners are exactly coplanar")),
    };
    let mut exact_reference_jacobians = None;
    let mut pending = storage.collect(std::iter::once((REFERENCE_TETRAHEDRON, 0_u32)))?;
    while let Some((cell, depth)) = pending.pop() {
        *work = work
            .checked_add(4)
            .ok_or_else(|| resource("Tet10 Jacobian work counter overflowed"))?;
        if *work > maximum_search_work {
            return Err(resource(format!(
                "Tet10 Jacobian certification exceeds its hard search-work limit of {maximum_search_work}"
            )));
        }
        if cancellation.is_cancelled() {
            return Err(error::failure(
                DelaunaySolverTopologyErrorKind::Cancelled,
                "cancelled",
            ));
        }
        let determinant = determinant_enclosure(coordinates, cell)?.scaled(orientation);
        if determinant.lower > 0.0 {
            continue;
        }
        if determinant.upper <= 0.0 {
            return Err(invalid(
                "Tet10 isoparametric mapping has a certified nonpositive Jacobian region",
            ));
        }
        if has_certified_nonpositive_vertex(coordinates, cell, orientation) {
            return Err(invalid(
                "Tet10 isoparametric mapping has a certified nonpositive Jacobian point",
            ));
        }
        if depth == 0 || depth >= maximum_recursion_depth {
            if exact_reference_jacobians.is_none() {
                exact_reference_jacobians = Some(build_exact_reference_jacobians(
                    coordinates,
                    maximum_search_work,
                    cancellation,
                    work,
                )?);
            }
            let exact_reference_jacobians = exact_reference_jacobians
                .as_ref()
                .expect("exact reference Jacobians were initialized");
            if exact_bernstein_coefficients_are_positive(
                exact_reference_jacobians,
                cell,
                orientation,
                maximum_search_work,
                cancellation,
                work,
            )? {
                continue;
            }
        }
        if depth >= maximum_recursion_depth {
            return Err(invalid(format!(
                "Tet10 isoparametric Jacobian enclosure [{:.17e}, {:.17e}] could not be certified positive by recursion depth {maximum_recursion_depth} at barycentric cell {:?}",
                determinant.lower, determinant.upper, cell
            )));
        }
        let (left, right) = bisect(cell);
        storage.push(&mut pending, (right, depth + 1))?;
        storage.push(&mut pending, (left, depth + 1))?;
    }
    Ok(())
}

fn bisect(mut cell: [[f64; 4]; 4]) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let mut selected = (0, 1);
    let mut maximum_length = -1.0;
    for left in 0..4 {
        for right in left + 1..4 {
            let length = (1..4)
                .map(|axis| {
                    let delta = cell[left][axis] - cell[right][axis];
                    delta * delta
                })
                .sum::<f64>();
            if length > maximum_length {
                maximum_length = length;
                selected = (left, right);
            }
        }
    }
    let midpoint =
        std::array::from_fn(|axis| cell[selected.0][axis] * 0.5 + cell[selected.1][axis] * 0.5);
    let mut right = cell;
    cell[selected.1] = midpoint;
    right[selected.0] = midpoint;
    (cell, right)
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}

fn resource(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::ResourceLimit, reason)
}

#[cfg(test)]
mod tests;
