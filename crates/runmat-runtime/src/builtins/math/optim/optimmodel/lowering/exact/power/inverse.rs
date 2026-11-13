//! Terminal proof-only exact inverse of an original constant square matrix.
//! This is not a numerical optimizer or a variable/global-column Jacobian.
use super::*;
pub(super) async fn evaluate(
    input: Data,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let n = input.shape.first().copied().unwrap_or(1);
    let count = n
        .checked_mul(n)
        .ok_or_else(|| policy.invalid("constant inverse extent overflow"))?;
    let mut left = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    let mut inverse = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    for column in 0..n {
        for row in 0..n {
            let Some(value) = input.at(row + column * n, policy)? else {
                return Ok(Data::uniform(input.shape, None));
            };
            left.push(value);
            inverse.push(Rational::from_usize(usize::from(row == column)));
            c.completed(1).await?;
        }
    }
    for pivot in 0..n {
        let mut selected = None;
        for row in pivot..n {
            if !left[row + pivot * n].is_zero() {
                selected = Some(row);
                break;
            }
            c.completed(1).await?;
        }
        let Some(selected) = selected else {
            return Ok(Data::uniform(input.shape, None));
        };
        for column in 0..n {
            left.swap(pivot + column * n, selected + column * n);
            inverse.swap(pivot + column * n, selected + column * n);
            c.completed(1).await?;
        }
        let diagonal = left[pivot + pivot * n].clone();
        for column in 0..n {
            let index = pivot + column * n;
            left[index] = left[index].div(&diagonal, policy, c).await?;
            inverse[index] = inverse[index].div(&diagonal, policy, c).await?;
            c.completed(1).await?;
        }
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = left[row + pivot * n].clone();
            if factor.is_zero() {
                c.completed(1).await?;
                continue;
            }
            for column in 0..n {
                let target = row + column * n;
                left[target] = left[target]
                    .sub(
                        &factor.mul(&left[pivot + column * n], policy, c).await?,
                        policy,
                        c,
                    )
                    .await?;
                inverse[target] = inverse[target]
                    .sub(
                        &factor.mul(&inverse[pivot + column * n], policy, c).await?,
                        policy,
                        c,
                    )
                    .await?;
                c.completed(1).await?;
            }
        }
    }
    let mut values = data::buffer(count, policy)?;
    for value in inverse {
        values.push(Some(value));
        c.completed(1).await?;
    }
    Data::explicit(input.shape, values, policy)
}
