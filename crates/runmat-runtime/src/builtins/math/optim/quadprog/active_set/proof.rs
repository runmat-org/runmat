//! A feasible anchor plus an exactly flat, strictly descending original ray.
//! Recovery only proposes rationals; no numerical tolerance proves Unbounded.
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational;
use num_rational::BigRational;

pub(super) async fn certify<V: compiled::Validation>(
    model: &Model,
    point: &[f64],
    direction: &[f64],
    validation: &mut V,
    c: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    Ok(conic::terminal::verified_point_satisfying(
        &model.view(),
        point,
        c,
        async |point: &[BigRational], c: &mut ScalarCooperation| {
            if !validation
                .certificate(compiled::CertificateCandidate::Feasible(point), c)
                .await?
            {
                return Ok(false);
            }
            conic::propose_exact_candidates(
                direction,
                OWNER,
                c,
                async |direction: &[BigRational], c: &mut ScalarCooperation| {
                    if !original(model, point, direction, c).await? {
                        return Ok(false);
                    }
                    validation.recession(point, direction, c).await
                },
            )
            .await
        },
    )
    .await?
    .is_some())
}
async fn import(
    values: &[BigRational],
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<Rational>>> {
    let mut out = capacity(values.len())?;
    for value in values {
        let Some(value) = Rational::import(value, curvature::POLICY, c).await? else {
            return Ok(None);
        };
        out.push(value);
        c.completed(1).await?;
    }
    Ok(Some(out))
}
async fn exact_zeros(count: usize, c: &mut ScalarCooperation) -> BuiltinResult<Vec<Rational>> {
    let mut out = capacity(count)?;
    for _ in 0..count {
        out.push(Rational::zero());
        c.completed(1).await?;
    }
    Ok(out)
}
async fn original(
    model: &Model,
    point: &[BigRational],
    direction: &[BigRational],
    c: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let policy = curvature::POLICY;
    if point.len() != model.f.len() || direction.len() != point.len() {
        return Ok(false);
    }
    let Some(point) = import(point, c).await? else {
        return Ok(false);
    };
    let Some(direction) = import(direction, c).await? else {
        return Ok(false);
    };
    let mut activity = exact_zeros(model.b.len(), c).await?;
    let mut hd = exact_zeros(point.len(), c).await?;
    let mut gradient = capacity(point.len())?;
    for &value in &model.f {
        gradient.push(
            Rational::from_float(value).ok_or_else(|| invalid("nonfinite original QP gradient"))?,
        );
        c.completed(1).await?;
    }
    for column in 0..point.len() {
        for entry in model.a.colptr[column]..model.a.colptr[column + 1] {
            let row = model.a.rowval[entry];
            let coefficient = Rational::from_float(model.a.nzval[entry])
                .ok_or_else(|| invalid("nonfinite original constraint"))?;
            activity[row] = activity[row]
                .add(
                    &coefficient.mul(&direction[column], policy, c).await?,
                    policy,
                    c,
                )
                .await?;
            c.completed(1).await?;
        }
        for entry in model.p.colptr[column]..model.p.colptr[column + 1] {
            let row = model.p.rowval[entry];
            let coefficient = Rational::from_float(model.p.nzval[entry])
                .ok_or_else(|| invalid("nonfinite original Hessian"))?;
            hd[row] = hd[row]
                .add(
                    &coefficient.mul(&direction[column], policy, c).await?,
                    policy,
                    c,
                )
                .await?;
            gradient[row] = gradient[row]
                .add(
                    &coefficient.mul(&point[column], policy, c).await?,
                    policy,
                    c,
                )
                .await?;
            if row != column {
                hd[column] = hd[column]
                    .add(
                        &coefficient.mul(&direction[row], policy, c).await?,
                        policy,
                        c,
                    )
                    .await?;
                gradient[column] = gradient[column]
                    .add(&coefficient.mul(&point[row], policy, c).await?, policy, c)
                    .await?;
            }
            c.completed(1).await?;
        }
    }
    for (row, value) in activity.into_iter().enumerate() {
        if (row < model.eq && !value.is_zero()) || (row >= model.eq && value.is_positive()) {
            return Ok(false);
        }
        c.completed(1).await?;
    }
    let mut curvature = Rational::zero();
    let mut slope = Rational::zero();
    for ((direction, hd), gradient) in direction.iter().zip(hd).zip(gradient) {
        curvature = curvature
            .add(&direction.mul(&hd, policy, c).await?, policy, c)
            .await?;
        slope = slope
            .add(&direction.mul(&gradient, policy, c).await?, policy, c)
            .await?;
        c.completed(1).await?;
    }
    Ok(curvature.is_zero() && slope.is_negative())
}
