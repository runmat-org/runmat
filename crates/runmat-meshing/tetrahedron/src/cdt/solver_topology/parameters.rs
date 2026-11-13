//! Prepare exact images as borrowed flat rows, then clone each final image once.
use super::{
    checkpoint, error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyOptions, ProjectionInput,
};
use runmat_geometry_core::{GeometryAllocationReservation, PersistentEntityId};
use runmat_meshing_core::{MeshingCancellationSignal, SolverNodeExactParameter, StableDigest};

#[derive(Clone, Copy)]
enum Parameter<'a> {
    Curve(&'a PersistentEntityId, f64),
    Surface(&'a PersistentEntityId, StableDigest, [f64; 2]),
}
#[derive(Clone, Copy)]
struct Row<'a> {
    node: usize,
    parameter: Parameter<'a>,
}
/// Per-node rows follow the canonical exact-parameter order validated by the solver mesh
/// contract (`total_cmp` on values). Raw bit order would place negative parameters after
/// positive ones and reverse them among themselves.
fn compare(a: &Row<'_>, b: &Row<'_>) -> std::cmp::Ordering {
    a.node
        .cmp(&b.node)
        .then_with(|| match (a.parameter, b.parameter) {
            (Parameter::Curve(a, x), Parameter::Curve(b, y)) => {
                a.cmp(b).then_with(|| x.total_cmp(&y))
            }
            (Parameter::Curve(..), Parameter::Surface(..)) => std::cmp::Ordering::Less,
            (Parameter::Surface(..), Parameter::Curve(..)) => std::cmp::Ordering::Greater,
            (Parameter::Surface(a, ac, x), Parameter::Surface(b, bc, y)) => a
                .cmp(b)
                .then_with(|| ac.cmp(&bc))
                .then_with(|| x[0].total_cmp(&y[0]))
                .then_with(|| x[1].total_cmp(&y[1])),
        })
}
fn observations<'a>(
    input: &'a ProjectionInput<'_>,
    indices: &super::NodeIndex,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
    mut sink: impl FnMut(Row<'a>) -> Result<(), DelaunaySolverTopologyError>,
) -> Result<(), DelaunaySolverTopologyError> {
    let mut observed = 0u64;
    let mut consume = |row| {
        checkpoint(observed, options, cancellation)?;
        observed = observed.checked_add(1).ok_or_else(overflow)?;
        sink(row)
    };
    {
        let mut curve = |node, edge, value: f64| {
            if !value.is_finite() {
                return Err(invalid("exact edge parameter is not finite"));
            }
            consume(Row {
                node,
                parameter: Parameter::Curve(edge, value),
            })
        };
        for (work, segment) in input.volume_mesh.provenance.segments.iter().enumerate() {
            checkpoint(work as u64, options, cancellation)?;
            let edge = segment
                .entity_ids
                .first()
                .ok_or_else(|| invalid("exact segment has no source edge"))?;
            for (identity, value) in segment.node_identities.iter().zip(segment.edge_parameters) {
                let node = indices
                    .get(identity)
                    .copied()
                    .ok_or_else(|| invalid("exact parameter references a missing volume node"))?;
                curve(node as usize, edge, value)?;
            }
        }
    }
    for (work, node) in input.exact_surface.nodes.iter().enumerate() {
        checkpoint(work as u64, options, cancellation)?;
        let Some(index) = indices.get(&node.node_id).copied() else {
            continue;
        };
        for usage in &node.uses {
            if usage.evaluator_uv.iter().any(|value| !value.is_finite()) {
                return Err(invalid("exact surface parameter is not finite"));
            }
            consume(Row {
                node: index as usize,
                parameter: Parameter::Surface(
                    &usage.source_face_id,
                    usage.chart_id,
                    usage.evaluator_uv,
                ),
            })?;
            for parameter in &usage.exact_edge_parameters {
                if !parameter.parameter.is_finite() {
                    return Err(invalid("exact edge parameter is not finite"));
                }
                consume(Row {
                    node: index as usize,
                    parameter: Parameter::Curve(&parameter.source_edge_id, parameter.parameter),
                })?;
            }
        }
    }
    Ok(())
}

pub(super) fn build_node_exact_parameters(
    input: &ProjectionInput<'_>,
    indices: &super::NodeIndex,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<Vec<SolverNodeExactParameter>>, DelaunaySolverTopologyError> {
    let mut count = 0usize;
    observations(input, indices, options, cancellation, |_| {
        count = count.checked_add(1).ok_or_else(overflow)?;
        Ok(())
    })?;
    let bytes = (count as u64)
        .checked_mul(std::mem::size_of::<Row<'_>>() as u64)
        .ok_or_else(overflow)?;
    let control = input.exact_evaluation.map(|evaluation| evaluation.control);
    let _rows_storage = control
        .map(|control| GeometryAllocationReservation::new(control, bytes))
        .transpose()
        .map_err(error::geometry)?;
    let mut rows = Vec::with_capacity(count);
    observations(input, indices, options, cancellation, |row| {
        rows.push(row);
        Ok(())
    })?;
    let rows = normalize_rows(rows, control, options, cancellation)?;
    let storage = super::storage::Storage::for_input(input);
    let mut result = storage.vector(indices.len())?;
    let mut cursor = 0;
    for node in 0..indices.len() {
        checkpoint(node as u64, options, cancellation)?;
        let start = cursor;
        while cursor < rows.len() && rows[cursor].node == node {
            cursor += 1;
        }
        let mut parameters = storage.vector(cursor - start)?;
        for row in &rows[start..cursor] {
            parameters.push(match row.parameter {
                Parameter::Curve(id, parameter) => SolverNodeExactParameter::Curve {
                    source_edge_id: storage.entity(id)?,
                    parameter,
                },
                Parameter::Surface(id, chart_id, evaluator_uv) => {
                    SolverNodeExactParameter::Surface {
                        source_face_id: storage.entity(id)?,
                        chart_id,
                        evaluator_uv,
                    }
                }
            });
        }
        result.push(parameters);
    }
    Ok(result)
}
fn normalize_rows<'a>(
    mut rows: Vec<Row<'a>>,
    control: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<Row<'a>>, DelaunaySolverTopologyError> {
    let bytes = (rows.len() as u64)
        .checked_mul(std::mem::size_of::<Row<'_>>() as u64)
        .ok_or_else(overflow)?;
    let scratch_storage = control
        .map(|control| GeometryAllocationReservation::new(control, bytes))
        .transpose()
        .map_err(error::geometry)?;
    let mut scratch = rows.clone();
    runmat_cpu_execution::sort_copy_with_checkpoint(&mut rows, &mut scratch, compare, || {
        checkpoint(0, options, cancellation)
    })?;
    drop(scratch);
    drop(scratch_storage);
    let mut retained = 0;
    for index in 0..rows.len() {
        checkpoint(index as u64, options, cancellation)?;
        let row = rows[index];
        if retained == 0 || !compare(&rows[retained - 1], &row).is_eq() {
            rows[retained] = row;
            retained += 1;
        }
    }
    rows.truncate(retained);
    Ok(rows)
}

fn invalid(reason: &'static str) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
fn overflow() -> DelaunaySolverTopologyError {
    error::failure(
        DelaunaySolverTopologyErrorKind::ResourceLimit,
        "exact parameter inventory overflow",
    )
}

#[cfg(test)]
mod tests;
