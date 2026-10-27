//! Reverse source selection/assignment and column-major concatenation.
use super::*;
pub(super) async fn assign(
    state: &mut State<'_>,
    source: GcHandle,
    replacement: GcHandle,
    indices: &[usize],
    selection_shape: &[usize],
    weights: Weights,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let policy = state.policy;
    let shape = state.shape(replacement)?;
    let layout = crate::indexing::write_slice::AssignmentRhsLayout::multidimensional(
        shape,
        evaluation::element_count(shape, policy)?,
        selection_shape,
    )?;
    let mut selected = HashMap::new();
    selected
        .try_reserve(indices.len())
        .map_err(|e| policy.reserve(e))?;
    for (position, &row) in indices.iter().enumerate() {
        selected.insert(row, layout.source_index(position));
        cooperation.completed(1).await?;
    }
    for (row, value) in weights {
        if let Some(&input) = selected.get(&row) {
            state.add(replacement, input, value, cooperation).await?;
        } else {
            state.add(source, row, value, cooperation).await?;
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
pub(super) async fn concatenate(
    state: &mut State<'_>,
    dimension: usize,
    operands: &[GcHandle],
    shape: &[usize],
    weights: Weights,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let policy = state.policy;
    let inner = shape
        .iter()
        .take(dimension)
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| policy.invalid("reverse concatenation inner block overflow"))?;
    let chunk = shape
        .get(dimension)
        .copied()
        .unwrap_or(1)
        .checked_mul(inner)
        .ok_or_else(|| policy.invalid("reverse concatenation output block overflow"))?;
    if chunk == 0 && !weights.is_empty() {
        return Err(policy.invalid("reverse concatenation has an empty active output"));
    }
    let mut blocks =
        runmat_value::exact_capacity_vec(operands.len()).map_err(|e| policy.storage(e))?;
    let mut end = 0usize;
    for &operand in operands {
        let shape = state.shape(operand)?;
        if evaluation::element_count(shape, policy)? == 0 {
            continue;
        }
        let width = shape
            .get(dimension)
            .copied()
            .unwrap_or(1)
            .checked_mul(inner)
            .ok_or_else(|| policy.invalid("reverse concatenation source block overflow"))?;
        let base = end;
        end = end
            .checked_add(width)
            .ok_or_else(|| policy.invalid("reverse concatenation offset overflow"))?;
        blocks.push((operand, base, end));
        cooperation.completed(1).await?;
    }
    if end != chunk {
        return Err(policy.invalid("reverse concatenation differs from original output geometry"));
    }
    for (row, value) in weights {
        let within = row % chunk;
        let index = blocks.partition_point(|(_, _, end)| *end <= within);
        let &(source, base, end) = blocks
            .get(index)
            .ok_or_else(|| policy.invalid("reverse concatenation block is unavailable"))?;
        state
            .add(
                source,
                row / chunk * (end - base) + within - base,
                value,
                cooperation,
            )
            .await?;
        cooperation.completed(1).await?;
    }
    Ok(())
}
