//! Indexed affine edits keep last-write selection semantics and original slot
//! validity. Only edited destination rows are indexed; no global variable map.
use super::super::terms::Term;
use super::*;
use std::collections::HashMap;
pub(super) async fn assign(
    source: Array,
    replacement: Array,
    indices: &[usize],
    selection_shape: &[usize],
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    let count = source.count(policy)?;
    let rhs = crate::indexing::write_slice::AssignmentRhsLayout::multidimensional(
        &replacement.shape,
        replacement.count(policy)?,
        selection_shape,
    )?;
    let mut selected = HashMap::new();
    selected
        .try_reserve(indices.len())
        .map_err(|e| policy.reserve(e))?;
    for (position, &target) in indices.iter().enumerate() {
        if target >= count {
            return Err(policy.invalid("compiled assignment exceeds its original expression shape"));
        }
        selected.insert(target, rhs.source_index(position));
        cooperation.completed(1).await?;
    }
    if selected.is_empty() {
        return Ok(Ok(source));
    }
    let degree = Degree::collect(
        count,
        |row| {
            Ok(match selected.get(&row) {
                Some(&input) => replacement.degree.at(input),
                None => source.degree.at(row),
            })
        },
        policy,
        cooperation,
    )
    .await?;
    let offset = match (&source.offset, &replacement.offset) {
        (Offset::Uniform(a), Offset::Uniform(b)) if a == b => Offset::Uniform(*a),
        (_, Offset::Uniform(value)) if selected.len() == count => Offset::Uniform(*value),
        _ => {
            let mut values =
                runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
            for row in 0..count {
                let value = match selected.get(&row) {
                    Some(&input) => replacement.offset.at(input, policy)?,
                    None => source.offset.at(row, policy)?,
                };
                if !value.is_finite() {
                    return Ok(Err(Stop::NonFinite));
                }
                values.push(value);
                cooperation.completed(1).await?;
            }
            Offset::Dense(
                Tensor::from_numeric_storage(
                    runmat_value::NumericStorage::F64(values),
                    source.shape.clone(),
                )
                .map_err(|e| policy.invalid(e))?,
            )
        }
    };
    let assigned = if matches!(source.assigned, Assigned::All)
        && matches!(replacement.assigned, Assigned::All)
    {
        Assigned::All
    } else {
        let mut values = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        let mut missing = false;
        for row in 0..count {
            let valid = match selected.get(&row) {
                Some(&input) => replacement.assigned.at(input, policy)?,
                None => source.assigned.at(row, policy)?,
            };
            values.push(valid);
            missing |= !valid;
            cooperation.completed(1).await?;
        }
        if missing {
            Assigned::Partial(std::rc::Rc::new(values))
        } else {
            Assigned::All
        }
    };
    let coefficients = if source.coefficients.is_zero() && replacement.coefficients.is_zero() {
        Coefficients::Zero
    } else {
        let mut entries = Vec::new();
        for term in source.coefficients.entries() {
            if !selected.contains_key(&term.row) {
                entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                entries.push(term);
            }
            cooperation.completed(1).await?;
        }
        // Authenticate repeated destinations before forming the inverse source
        // projection: only the final original write contributes coefficients.
        let mut writes =
            runmat_value::exact_capacity_vec(selected.len()).map_err(|e| policy.storage(e))?;
        for (&row, &column) in &selected {
            writes.push(Term {
                column,
                row,
                value: 1.,
            });
            cooperation.completed(1).await?;
        }
        let map = match Coefficients::from_entries(
            writes,
            count,
            replacement.count(policy)?,
            policy,
            cooperation,
        )
        .await?
        {
            Ok(map) => map,
            Err(stop) => return Ok(Err(stop)),
        };
        for term in replacement.coefficients.entries() {
            for (row, _) in map.column(term.row) {
                entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                entries.push(Term { row, ..term });
                cooperation.completed(1).await?;
            }
        }
        match Coefficients::from_entries(entries, count, columns, policy, cooperation).await? {
            Ok(values) => values,
            Err(stop) => return Ok(Err(stop)),
        }
    };
    Ok(Ok(Array {
        shape: source.shape,
        offset,
        assigned,
        coefficients,
        degree,
    }))
}
