//! Scalar unit-phase equivalence compiled into the shared real affine authority.

use crate::excitation::ProjectionScalar;
use thiserror::Error;

use super::{AffineConstraintReduction, ConstraintAlgebraError};

/// `secondary = exp(i * phase_radians) * primary`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitPhaseRelation {
    pub primary: usize,
    pub secondary: usize,
    pub phase_radians: runmat_types::physical::Angle,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum UnitPhaseConstraintError {
    #[error("unit-phase constraint dimensions are invalid")]
    InvalidDimensions,
    #[error("unit-phase relation {relation} contains an unavailable node or nonfinite phase")]
    InvalidRelation { relation: usize },
    #[error("prescribed complex scalar at node {node} is nonfinite")]
    InvalidPrescription { node: usize },
    #[error("unit-phase cycle through node {node} is inconsistent")]
    InconsistentCycle { node: usize },
    #[error("prescribed complex scalars in component {representative} conflict")]
    ConflictingPrescription { representative: usize },
    #[error(transparent)]
    Algebra(#[from] ConstraintAlgebraError),
}

#[derive(Clone, Copy)]
struct Edge {
    node: usize,
    phase: f64,
}

/// Compiles complex scalar constraints in stacked `[real nodes, imaginary nodes]` layout.
///
/// Components use their smallest node as representative. A prescribed value fixes the
/// entire component; otherwise its representative contributes two real free DOFs. Each
/// recovery row has at most two coefficients. Real transpose restriction is therefore
/// exactly the complex conjugate-transpose restriction, with no separate complex algebra.
/// Consistent redundant cycles are ignored; inconsistent cycles are rejected, rather
/// than silently treating them as constraints prescribing zero.
pub fn build_phase_reduction(
    node_count: usize,
    relations: &[UnitPhaseRelation],
    prescribed: &[Option<ProjectionScalar>],
) -> Result<AffineConstraintReduction, UnitPhaseConstraintError> {
    let full_dof_count = node_count
        .checked_mul(2)
        .filter(|count| count.checked_add(1).is_some())
        .ok_or(UnitPhaseConstraintError::InvalidDimensions)?;
    let maximum_entries = full_dof_count
        .checked_mul(2)
        .ok_or(UnitPhaseConstraintError::InvalidDimensions)?;
    if prescribed.len() != node_count {
        return Err(UnitPhaseConstraintError::InvalidDimensions);
    }
    let edge_count = relations
        .len()
        .checked_mul(2)
        .ok_or(UnitPhaseConstraintError::InvalidDimensions)?;
    let mut graph_offsets = vec![0usize; node_count + 1];
    for (index, relation) in relations.iter().enumerate() {
        if relation.primary >= node_count
            || relation.secondary >= node_count
            || !relation.phase_radians.canonical_value().is_finite()
        {
            return Err(UnitPhaseConstraintError::InvalidRelation { relation: index });
        }
        graph_offsets[relation.primary + 1] += 1;
        graph_offsets[relation.secondary + 1] += 1;
    }
    for node in 0..node_count {
        graph_offsets[node + 1] += graph_offsets[node];
    }
    let mut cursors = graph_offsets[..node_count].to_vec();
    let mut graph = vec![
        Edge {
            node: 0,
            phase: 0.0
        };
        edge_count
    ];
    for relation in relations {
        let phase = wrap_phase(relation.phase_radians.canonical_value());
        for (source, node, phase) in [
            (relation.primary, relation.secondary, phase),
            (relation.secondary, relation.primary, -phase),
        ] {
            graph[cursors[source]] = Edge { node, phase };
            cursors[source] += 1;
        }
    }
    drop(cursors);
    for node in 0..node_count {
        graph[graph_offsets[node]..graph_offsets[node + 1]].sort_unstable_by(|left, right| {
            left.node
                .cmp(&right.node)
                .then_with(|| left.phase.total_cmp(&right.phase))
        });
    }
    for (node, value) in prescribed.iter().enumerate() {
        if value.is_some_and(|value| !value.real.is_finite() || !value.imaginary.is_finite()) {
            return Err(UnitPhaseConstraintError::InvalidPrescription { node });
        }
    }
    let mut representatives = vec![usize::MAX; node_count];
    let mut phases = vec![0.0; node_count];
    let mut depths = vec![0usize; node_count];
    let mut queue = Vec::with_capacity(node_count);
    for representative in 0..node_count {
        if representatives[representative] != usize::MAX {
            continue;
        }
        queue.clear();
        queue.push(representative);
        representatives[representative] = representative;
        let mut cursor = 0;
        while cursor < queue.len() {
            let primary = queue[cursor];
            cursor += 1;
            for edge in &graph[graph_offsets[primary]..graph_offsets[primary + 1]] {
                let phase = wrap_phase(phases[primary] + edge.phase);
                if representatives[edge.node] == usize::MAX {
                    representatives[edge.node] = representative;
                    phases[edge.node] = phase;
                    depths[edge.node] = depths[primary] + 1;
                    queue.push(edge.node);
                } else {
                    let tolerance = 128.0
                        * f64::EPSILON
                        * (1.0 + depths[primary] as f64 + depths[edge.node] as f64);
                    if wrap_phase(phase - phases[edge.node]).abs() > tolerance {
                        return Err(UnitPhaseConstraintError::InconsistentCycle {
                            node: edge.node,
                        });
                    }
                }
            }
        }
    }
    drop(graph);
    drop(graph_offsets);
    drop(queue);
    let mut component_values = vec![None; node_count];
    let mut prescription_depths = vec![0usize; node_count];
    for (node, value) in prescribed.iter().enumerate() {
        let Some(value) = value else { continue };
        let representative = representatives[node];
        let canonical = rotate(*value, -phases[node]);
        if !canonical.real.is_finite() || !canonical.imaginary.is_finite() {
            return Err(UnitPhaseConstraintError::InvalidPrescription { node });
        }
        if let Some(existing) = component_values[representative] {
            let tolerance = 128.0
                * f64::EPSILON
                * (1.0 + depths[node] as f64 + prescription_depths[representative] as f64);
            if !same_scalar(existing, canonical, tolerance) {
                return Err(UnitPhaseConstraintError::ConflictingPrescription { representative });
            }
        } else {
            component_values[representative] = Some(canonical);
            prescription_depths[representative] = depths[node];
        }
    }
    drop(depths);
    drop(prescription_depths);
    let mut free_nodes = Vec::with_capacity(node_count);
    free_nodes.extend(
        (0..node_count)
            .filter(|node| representatives[*node] == *node && component_values[*node].is_none()),
    );
    let free_count = free_nodes.len();
    let mut columns = vec![usize::MAX; node_count];
    for (column, node) in free_nodes.iter().enumerate() {
        columns[*node] = column;
    }
    let mut free_dof_indices = Vec::with_capacity(2 * free_count);
    free_dof_indices.extend(free_nodes.iter().copied());
    free_dof_indices.extend(free_nodes.iter().map(|node| node + node_count));
    drop(free_nodes);
    let mut reduction = AffineConstraintReduction {
        full_dof_count,
        reduced_dof_count: 2 * free_count,
        free_dof_indices,
        pivot_dof_indices: Vec::with_capacity(full_dof_count - 2 * free_count),
        recovery_row_offsets: Vec::with_capacity(full_dof_count + 1),
        recovery_column_indices: Vec::with_capacity(maximum_entries),
        recovery_coefficients: Vec::with_capacity(maximum_entries),
        offset: vec![0.0; full_dof_count],
    };
    reduction.recovery_row_offsets.push(0);
    for row in 0..full_dof_count {
        let node = row % node_count;
        let imaginary = row >= node_count;
        let representative = representatives[node];
        if let Some(value) = component_values[representative] {
            let value = rotate(value, phases[node]);
            let scalar = if imaginary {
                value.imaginary
            } else {
                value.real
            };
            if !scalar.is_finite() {
                return Err(UnitPhaseConstraintError::InvalidPrescription { node });
            }
            reduction.offset[row] = scalar;
        } else {
            let (sine, cosine) = phases[node].sin_cos();
            let column = columns[representative];
            let coefficients = if imaginary {
                [sine, cosine]
            } else {
                [cosine, -sine]
            };
            for (column, coefficient) in [column, column + free_count].into_iter().zip(coefficients)
            {
                if coefficient != 0.0 {
                    reduction.recovery_column_indices.push(column);
                    reduction.recovery_coefficients.push(coefficient);
                }
            }
        }
        if representative != node || component_values[representative].is_some() {
            reduction.pivot_dof_indices.push(row);
        }
        reduction
            .recovery_row_offsets
            .push(reduction.recovery_coefficients.len());
    }
    drop(representatives);
    drop(phases);
    drop(component_values);
    drop(columns);
    reduction.validate()?;
    Ok(reduction)
}

fn wrap_phase(phase: f64) -> f64 {
    let wrapped = phase.rem_euclid(std::f64::consts::TAU);
    if wrapped > std::f64::consts::PI {
        wrapped - std::f64::consts::TAU
    } else {
        wrapped
    }
}

fn rotate(value: ProjectionScalar, phase: f64) -> ProjectionScalar {
    let (sine, cosine) = phase.sin_cos();
    ProjectionScalar {
        real: cosine * value.real - sine * value.imaginary,
        imaginary: sine * value.real + cosine * value.imaginary,
    }
}

fn same_scalar(left: ProjectionScalar, right: ProjectionScalar, tolerance: f64) -> bool {
    let scale = left
        .real
        .abs()
        .max(left.imaginary.abs())
        .max(right.real.abs())
        .max(right.imaginary.abs());
    scale == 0.0
        || ((left.real / scale - right.real / scale).abs() <= tolerance
            && (left.imaginary / scale - right.imaginary / scale).abs() <= tolerance)
}

#[cfg(test)]
mod tests;
