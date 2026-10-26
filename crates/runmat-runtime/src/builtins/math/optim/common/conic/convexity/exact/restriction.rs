//! Exact equality congruence before the existing sparse Schur certificate.
//! Coordinates remain original; eliminated coordinates become empty zero rows.
use super::*;
type Row = BTreeMap<usize, Rational>;

pub(in crate::builtins::math::optim) async fn certify_restricted(
    mut hessian: OriginalHessian,
    mut equations: Vec<Row>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for selected in 0..equations.len() {
        let mut equation = std::mem::take(&mut equations[selected]);
        let Some((pivot, coefficient)) = equation.pop_first() else {
            // Redundancy and inconsistent RHS are feasibility questions, not
            // grounds to assume full rank or reject a zero-dimensional kernel.
            c.completed(1).await?;
            continue;
        };
        if pivot >= hessian.rows.len() || coefficient.is_zero() {
            return Err(policy.invalid("invalid original equality curvature coordinate"));
        }
        for (&column, value) in &mut equation {
            if column >= hessian.rows.len() {
                return Err(policy.invalid("original equality exceeds Hessian dimension"));
            }
            *value = value.div(&coefficient, policy, c).await?.negated();
            c.completed(1).await?;
        }
        restrict(&mut hessian, pivot, &equation, policy, c).await?;
        for remaining in &mut equations[selected + 1..] {
            if let Some(value) = remaining.remove(&pivot) {
                for (&column, factor) in &equation {
                    let delta = value.mul(factor, policy, c).await?;
                    add(remaining, column, delta, policy, c).await?;
                }
            }
            c.completed(1).await?;
        }
    }
    super::certify_original(hessian, policy, c).await
}

async fn add(
    row: &mut Row,
    column: usize,
    delta: Rational,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let value = match row.remove(&column) {
        Some(previous) => previous.add(&delta, policy, c).await?,
        None => delta,
    };
    if !value.is_zero() {
        row.insert(column, value);
    }
    c.completed(1).await
}

async fn restrict(
    hessian: &mut OriginalHessian,
    pivot: usize,
    substitution: &Row,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let mut cross = std::mem::take(&mut hessian.rows[pivot]);
    let diagonal = cross.remove(&pivot).unwrap_or_else(Rational::zero);
    for row in 0..pivot {
        if let Some(value) = hessian.rows[row].remove(&pivot) {
            cross.insert(row, value);
        }
        c.completed(1).await?;
    }
    // H_ij t_k + t_i H_jk; diagonal receives both identical terms.
    for (&i, value) in &cross {
        for (&k, factor) in substitution {
            let mut delta = value.mul(factor, policy, c).await?;
            if i == k {
                delta = delta.add(&delta, policy, c).await?;
            }
            hessian.add(i.min(k), i.max(k), delta, policy, c).await?;
        }
    }
    // H_jj t_i t_k, stored once in the symmetric upper triangle.
    for (&i, a) in substitution {
        let scaled = diagonal.mul(a, policy, c).await?;
        for (&k, b) in substitution.range(i..) {
            let delta = scaled.mul(b, policy, c).await?;
            hessian.add(i, k, delta, policy, c).await?;
        }
    }
    Ok(())
}
