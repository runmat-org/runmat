//! Source-admitted shape/edit routing. Actual selectors are already validated;
//! this maps selected coefficient rows, never replans user indices.
use super::*;
pub(super) async fn transpose(
    input: Array,
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Array> {
    if input.shape.iter().skip(2).any(|&value| value != 1) {
        return Err(policy.invalid("quadratic transpose requires a two-dimensional expression"));
    }
    let m = input.shape.first().copied().unwrap_or(1);
    let n = input.shape.get(1).copied().unwrap_or(1);
    let count = input.count(policy)?;
    if count != evaluation::element_count(&shape, policy)?
        || shape.first().copied().unwrap_or(1) != n
        || shape.get(1).copied().unwrap_or(1) != m
    {
        return Err(policy.invalid("quadratic transpose differs from original geometry"));
    }
    let mut rows = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    for target in 0..count {
        rows.push(input.at(target / n + target % n * m, policy)?);
        c.completed(1).await?;
    }
    Array::rows(shape, rows, policy)
}
pub(super) async fn assign(
    input: Array,
    replacement: Array,
    indices: &[usize],
    selection_shape: &[usize],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Array> {
    let count = input.count(policy)?;
    let rhs = crate::indexing::write_slice::AssignmentRhsLayout::multidimensional(
        &replacement.shape,
        replacement.count(policy)?,
        selection_shape,
    )?;
    let mut selected = std::collections::HashMap::new();
    selected
        .try_reserve(indices.len())
        .map_err(|e| policy.reserve(e))?;
    for (position, &target) in indices.iter().enumerate() {
        if target >= count {
            return Err(policy.invalid("quadratic assignment exceeds original shape"));
        }
        selected.insert(target, rhs.source_index(position));
        c.completed(1).await?;
    }
    if selected.is_empty() {
        return Ok(input);
    }
    let mut rows = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    for row in 0..count {
        rows.push(match selected.get(&row) {
            Some(&source) => replacement.at(source, policy)?,
            None => input.at(row, policy)?,
        });
        c.completed(1).await?;
    }
    Array::rows(input.shape, rows, policy)
}
struct Block {
    input: Array,
    chunk: usize,
    base: usize,
}
pub(super) async fn concatenate(
    inputs: Vec<Array>,
    dimension: usize,
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Array> {
    let count = evaluation::element_count(&shape, policy)?;
    if count == 0 {
        return Ok(Array::zero(shape));
    }
    let rank = shape.len().max(
        dimension
            .checked_add(1)
            .ok_or_else(|| policy.invalid("quadratic concatenation dimension overflows usize"))?,
    );
    let inner = shape
        .iter()
        .take(dimension)
        .try_fold(1usize, |n, &extent| n.checked_mul(extent))
        .ok_or_else(|| policy.invalid("quadratic concatenation block overflows usize"))?;
    let width = shape.get(dimension).copied().unwrap_or(1);
    let output_chunk = inner
        .checked_mul(width)
        .ok_or_else(|| policy.invalid("quadratic concatenation block overflows usize"))?;
    let mut blocks =
        runmat_value::exact_capacity_vec(inputs.len()).map_err(|e| policy.storage(e))?;
    let mut total = 0usize;
    for input in inputs {
        if input.count(policy)? == 0 {
            continue;
        }
        for axis in 0..rank {
            if axis != dimension
                && input.shape.get(axis).copied().unwrap_or(1)
                    != shape.get(axis).copied().unwrap_or(1)
            {
                return Err(
                    policy.invalid("quadratic concatenation differs from original geometry")
                );
            }
            c.completed(1).await?;
        }
        let extent = input.shape.get(dimension).copied().unwrap_or(1);
        let chunk = inner
            .checked_mul(extent)
            .ok_or_else(|| policy.invalid("quadratic concatenation block overflows usize"))?;
        let base = inner
            .checked_mul(total)
            .ok_or_else(|| policy.invalid("quadratic concatenation offset overflows usize"))?;
        total = total
            .checked_add(extent)
            .ok_or_else(|| policy.invalid("quadratic concatenation extent overflows usize"))?;
        blocks.push(Block { input, chunk, base });
        c.completed(1).await?;
    }
    if total != width || blocks.is_empty() || output_chunk == 0 {
        return Err(policy.invalid("quadratic concatenation differs from original output shape"));
    }
    let mut rows = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    for row in 0..count {
        let inner = row % output_chunk;
        let block = blocks
            .partition_point(|block| block.base <= inner)
            .checked_sub(1)
            .and_then(|index| blocks.get(index))
            .ok_or_else(|| policy.invalid("quadratic concatenation lost original block"))?;
        rows.push(block.input.at(
            row / output_chunk * block.chunk + inner - block.base,
            policy,
        )?);
        c.completed(1).await?;
    }
    Array::rows(shape, rows, policy)
}
