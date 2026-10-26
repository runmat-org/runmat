//! Exact slot transformations follow the same admitted original index and
//! assignment layouts. No textual indexing or evaluation probes are involved.
use super::*;
pub(super) async fn gather(
    input: Data,
    indices: &[usize],
    shape: SharedShape,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    if let Some(value) = input.uniform_value() {
        for &row in indices {
            if row >= input.count(policy)? {
                return Err(policy.invalid("exact selection exceeds original shape"));
            }
            cooperation.completed(1).await?;
        }
        return Ok(Data::uniform(shape, value.clone()));
    }
    let mut values = data::buffer(indices.len(), policy)?;
    for &row in indices {
        values.push(input.at(row, policy)?);
        cooperation.completed(1).await?;
    }
    Data::explicit(shape, values, policy)
}
pub(super) async fn assign(
    input: Data,
    replacement: Data,
    indices: &[usize],
    selection_shape: &[usize],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let layout = crate::indexing::write_slice::AssignmentRhsLayout::multidimensional(
        &replacement.shape,
        replacement.count(policy)?,
        selection_shape,
    )?;
    let count = input.count(policy)?;
    let mut values = data::buffer(count, policy)?;
    for row in 0..count {
        values.push(input.at(row, policy)?);
        cooperation.completed(1).await?;
    }
    for (position, &row) in indices.iter().enumerate() {
        let slot = values
            .get_mut(row)
            .ok_or_else(|| policy.invalid("exact assignment exceeds original shape"))?;
        *slot = replacement.at(layout.source_index(position), policy)?;
        cooperation.completed(1).await?;
    }
    Data::explicit(input.shape, values, policy)
}
pub(super) async fn concatenate(
    inputs: Vec<Data>,
    dimension: usize,
    shape: SharedShape,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let count = evaluation::element_count(&shape, policy)?;
    if count == 0 {
        return Ok(Data::uniform(shape, Some(Rational::zero())));
    }
    let inner = shape
        .iter()
        .take(dimension)
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| policy.invalid("exact concatenation inner block overflow"))?;
    let width = shape.get(dimension).copied().unwrap_or(1);
    let chunk = width
        .checked_mul(inner)
        .ok_or_else(|| policy.invalid("exact concatenation output block overflow"))?;
    if chunk == 0 {
        return Err(policy.invalid("exact concatenation has an empty nonempty block"));
    }
    let mut values = data::buffer(count, policy)?;
    for _ in 0..count {
        values.push(None);
        cooperation.completed(1).await?;
    }
    let mut base = 0usize;
    for input in inputs {
        let count = input.count(policy)?;
        if count == 0 {
            continue;
        }
        let block = input
            .shape
            .get(dimension)
            .copied()
            .unwrap_or(1)
            .checked_mul(inner)
            .ok_or_else(|| policy.invalid("exact concatenation source block overflow"))?;
        if block == 0 {
            return Err(policy.invalid("exact concatenation source shape is inconsistent"));
        }
        for row in 0..count {
            let target = row / block * chunk + base + row % block;
            *values
                .get_mut(target)
                .ok_or_else(|| policy.invalid("exact concatenation exceeds original geometry"))? =
                input.at(row, policy)?;
            cooperation.completed(1).await?;
        }
        base = base
            .checked_add(block)
            .ok_or_else(|| policy.invalid("exact concatenation offset overflow"))?;
    }
    if base != chunk {
        return Err(policy.invalid("exact concatenation does not cover original geometry"));
    }
    Data::explicit(shape, values, policy)
}
