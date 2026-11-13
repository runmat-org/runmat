//! Exact affine products of fully assigned original sparse operands.
//! Traverse actual stored nonzeros, never a dense m-by-k coefficient grid.
use super::*;
fn stored(
    array: &runmat_value::SparseTensor,
    index: usize,
    policy: Policy,
) -> BuiltinResult<Rational> {
    let Some(runmat_value::NumericScalar::F64(value)) = array.numeric_value_at(index) else {
        return Err(policy.invalid("original sparse constant lost its real double storage"));
    };
    Rational::from_float(value)
        .ok_or_else(|| policy.invalid("original sparse constant is not finite"))
}
pub(super) async fn multiply(
    left: Data,
    right: Data,
    shape: SharedShape,
    m: usize,
    k: usize,
    n: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let count = evaluation::element_count(&shape, policy)?;
    let mut values = data::buffer(count, policy)?;
    for _ in 0..count {
        values.push(Some(Rational::zero()));
        cooperation.completed(1).await?;
    }
    if let Some(matrix) = left.sparse() {
        for column in 0..n {
            for axis in 0..k {
                let value = right
                    .at(axis + column * k, policy)?
                    .ok_or_else(|| policy.invalid("complete exact product lost a source slot"))?;
                if !value.is_zero() {
                    for entry in matrix.col_ptrs[axis]..matrix.col_ptrs[axis + 1] {
                        let row = matrix.row_indices[entry] + column * m;
                        let target = values[row].as_mut().expect("complete exact output");
                        *target = target
                            .add(
                                &stored(matrix, entry, policy)?
                                    .mul(&value, policy, cooperation)
                                    .await?,
                                policy,
                                cooperation,
                            )
                            .await?;
                        cooperation.completed(1).await?;
                    }
                }
                cooperation.completed(1).await?;
            }
        }
    } else if let Some(matrix) = right.sparse() {
        for column in 0..n {
            for entry in matrix.col_ptrs[column]..matrix.col_ptrs[column + 1] {
                let axis = matrix.row_indices[entry];
                let value = stored(matrix, entry, policy)?;
                for row in 0..m {
                    let source = left.at(row + axis * m, policy)?.ok_or_else(|| {
                        policy.invalid("complete exact product lost a source slot")
                    })?;
                    let target = values[row + column * m]
                        .as_mut()
                        .expect("complete exact output");
                    *target = target
                        .add(
                            &source.mul(&value, policy, cooperation).await?,
                            policy,
                            cooperation,
                        )
                        .await?;
                    cooperation.completed(1).await?;
                }
            }
            cooperation.completed(1).await?;
        }
    } else {
        return Err(policy.invalid("sparse exact product has no original sparse operand"));
    }
    Data::explicit(shape, values, policy)
}
