use std::collections::{BTreeMap, BTreeSet};

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

pub(super) fn validate(
    topology: &SolverMeshTopology,
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation_check_interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<(), DelaunaySolverTopologyError> {
    if cancellation_check_interval == 0 {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::InvalidOptions,
            "Tet10 Jacobian cancellation-check interval must be greater than zero",
        ));
    }
    let element_indices = 0..topology.volume_elements.len();
    validate_elements(
        topology,
        element_indices,
        maximum_search_work,
        maximum_recursion_depth,
        cancellation_check_interval,
        cancellation,
        work,
    )
}

pub(super) fn validate_elements(
    topology: &SolverMeshTopology,
    element_indices: impl IntoIterator<Item = usize>,
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation_check_interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<(), DelaunaySolverTopologyError> {
    let node_by_id = topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for element_index in element_indices {
        let element = topology
            .volume_elements
            .get(element_index)
            .ok_or_else(|| invalid("Tet10 Jacobian element index is out of bounds"))?;
        let coordinates = element_coordinates(element, &node_by_id)?;
        certify_element(
            coordinates,
            maximum_search_work,
            maximum_recursion_depth,
            cancellation_check_interval,
            cancellation,
            work,
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

pub(super) fn invalid_element_indices(
    topology: &SolverMeshTopology,
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation_check_interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<BTreeSet<usize>, DelaunaySolverTopologyError> {
    let node_by_id = topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut invalid_indices = BTreeSet::new();
    for (element_index, element) in topology.volume_elements.iter().enumerate() {
        let coordinates = element_coordinates(element, &node_by_id)?;
        match certify_element(
            coordinates,
            maximum_search_work,
            maximum_recursion_depth,
            cancellation_check_interval,
            cancellation,
            work,
        ) {
            Ok(()) => {}
            Err(failure) if failure.kind == DelaunaySolverTopologyErrorKind::InvalidMesh => {
                invalid_indices.insert(element_index);
            }
            Err(failure) => return Err(failure),
        }
    }
    Ok(invalid_indices)
}

fn element_coordinates(
    element: &runmat_meshing_core::SolverVolumeElement,
    node_by_id: &BTreeMap<u64, [f64; 3]>,
) -> Result<[[f64; 3]; 10], DelaunaySolverTopologyError> {
    element
        .node_ids
        .iter()
        .map(|node_id| {
            node_by_id
                .get(node_id)
                .copied()
                .ok_or_else(|| invalid("Tet10 Jacobian references an absent node"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| invalid("Tet10 Jacobian requires exactly ten nodes"))
}

pub(super) fn optimization_margin_profile_elements(
    topology: &SolverMeshTopology,
    element_indices: impl IntoIterator<Item = usize>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<u64>, DelaunaySolverTopologyError> {
    let mut profile = Vec::new();
    for element_index in element_indices {
        margin_checkpoint(cancellation)?;
        let element = topology
            .volume_elements
            .get(element_index)
            .ok_or_else(|| invalid("Tet10 optimization element index is out of bounds"))?;
        profile.push(margin_priority_key(bernstein_margin(
            element_coordinates_from_topology(element, topology)?,
            cancellation,
        )?));
    }
    profile.sort_unstable();
    if profile.is_empty() {
        return Err(invalid("Tet10 optimization margin profile is empty"));
    }
    Ok(profile)
}

pub(super) fn worst_invalid_element(
    topology: &SolverMeshTopology,
    invalid_elements: &BTreeSet<usize>,
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
    element
        .node_ids
        .iter()
        .map(|node_id| {
            topology
                .nodes
                .binary_search_by_key(node_id, |node| node.node_id)
                .ok()
                .map(|index| topology.nodes[index].coordinates_m)
                .ok_or_else(|| invalid("Tet10 Jacobian references an absent node"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| invalid("Tet10 Jacobian requires exactly ten nodes"))
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

fn certify_element(
    coordinates: [[f64; 3]; 10],
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation_check_interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
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
    let mut pending = vec![(REFERENCE_TETRAHEDRON, 0_u32)];
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
        pending.push((right, depth + 1));
        pending.push((left, depth + 1));
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
