//! Concatenation of already-admitted model arrays. The original concrete node
//! shape owns geometry; compact coefficients map by column-major block offsets.
use super::super::terms::Term;
use super::*;
struct Block {
    input: Array,
    chunk: usize,
    base: usize,
}
pub(super) async fn concatenate(
    inputs: Vec<Array>,
    dimension: usize,
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    let count = evaluation::element_count(&shape, policy)?;
    if count == 0 {
        return Ok(Ok(Array::zero(shape)));
    }
    let axis = dimension
        .checked_add(1)
        .ok_or_else(|| policy.invalid("concatenation dimension overflow"))?;
    let rank = shape.len().max(axis);
    let inner = shape
        .iter()
        .take(dimension)
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| policy.invalid("concatenation inner block overflow"))?;
    let width = shape.get(dimension).copied().unwrap_or(1);
    let output_chunk = inner
        .checked_mul(width)
        .ok_or_else(|| policy.invalid("concatenation output block overflow"))?;
    let mut blocks =
        runmat_value::exact_capacity_vec(inputs.len()).map_err(|e| policy.storage(e))?;
    let mut total = 0usize;
    let mut uniform = None;
    let mut same_uniform = true;
    let mut all_assigned = true;
    for input in inputs {
        if input.count(policy)? == 0 {
            continue;
        }
        for axis in 0..rank {
            if axis != dimension
                && input.shape.get(axis).copied().unwrap_or(1)
                    != shape.get(axis).copied().unwrap_or(1)
            {
                return Err(policy.invalid("concatenation differs from original nonempty geometry"));
            }
        }
        let extent = input.shape.get(dimension).copied().unwrap_or(1);
        let chunk = extent
            .checked_mul(inner)
            .ok_or_else(|| policy.invalid("concatenation source block overflow"))?;
        match input.offset {
            Offset::Uniform(value) => {
                if let Some(previous) = uniform {
                    same_uniform &= value == previous;
                } else {
                    uniform = Some(value);
                }
            }
            _ => same_uniform = false,
        }
        all_assigned &= matches!(input.assigned, Assigned::All);
        let base = total
            .checked_mul(inner)
            .ok_or_else(|| policy.invalid("concatenation block offset overflow"))?;
        blocks.push(Block { input, chunk, base });
        total = total
            .checked_add(extent)
            .ok_or_else(|| policy.invalid("concatenation extent overflow"))?;
        cooperation.completed(rank as u64).await?;
    }
    if total != width || blocks.is_empty() || output_chunk == 0 {
        return Err(policy.invalid("concatenation differs from original output geometry"));
    }
    let degree = Degree::collect(
        count,
        |row| {
            let inner = row % output_chunk;
            let block = blocks
                .partition_point(|block| block.base <= inner)
                .checked_sub(1)
                .and_then(|index| blocks.get(index))
                .ok_or_else(|| policy.invalid("concatenated dependence exceeds original blocks"))?;
            Ok(block
                .input
                .degree
                .at(row / output_chunk * block.chunk + inner - block.base))
        },
        policy,
        cooperation,
    )
    .await?;
    let offset = if same_uniform {
        Offset::Uniform(uniform.expect("nonempty uniform concatenation"))
    } else {
        let mut values = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        for _ in 0..count {
            values.push(0.);
            cooperation.completed(1).await?;
        }
        for block in &blocks {
            for source in 0..block.input.count(policy)? {
                let row = source / block.chunk * output_chunk + block.base + source % block.chunk;
                values[row] = block.input.offset.at(source, policy)?;
                cooperation.completed(1).await?;
            }
        }
        Offset::Dense(
            Tensor::from_numeric_storage(runmat_value::NumericStorage::F64(values), shape.clone())
                .map_err(|e| policy.invalid(e))?,
        )
    };
    let assigned = if all_assigned {
        Assigned::All
    } else {
        let mut values = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        for _ in 0..count {
            values.push(false);
            cooperation.completed(1).await?;
        }
        for block in &blocks {
            for source in 0..block.input.count(policy)? {
                let row = source / block.chunk * output_chunk + block.base + source % block.chunk;
                values[row] = block.input.assigned.at(source, policy)?;
                cooperation.completed(1).await?;
            }
        }
        Assigned::Partial(std::rc::Rc::new(values))
    };
    let mut entries = Vec::new();
    for block in &blocks {
        entries
            .try_reserve(block.input.coefficients.nonzeros())
            .map_err(|e| policy.reserve(e))?;
        for term in block.input.coefficients.entries() {
            let row = term.row / block.chunk * output_chunk + block.base + term.row % block.chunk;
            entries.push(Term { row, ..term });
            cooperation.completed(1).await?;
        }
    }
    let coefficients =
        match Coefficients::from_entries(entries, count, columns, policy, cooperation).await? {
            Ok(values) => values,
            Err(stop) => return Ok(Err(stop)),
        };
    Ok(Ok(Array {
        shape,
        offset,
        assigned,
        coefficients,
        degree,
    }))
}
