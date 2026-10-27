//! Exact original-model terminal proofs for zero, nonnegative and SOC blocks.
use super::super::ScalarCooperation;
use super::{dimension, Model};
use crate::BuiltinResult;
use clarabel::solver::SupportedConeT;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
/// Private-field evidence constructed only by the exact terminal validators.
pub(crate) struct VerifiedPoint(Vec<BigRational>);
pub(crate) struct VerifiedRay(Vec<BigRational>);
impl VerifiedPoint {
    pub(crate) fn values(&self) -> &[BigRational] {
        &self.0
    }
    pub(crate) fn into_values(self) -> Vec<BigRational> {
        self.0
    }
}
impl VerifiedRay {
    pub(crate) fn values(&self) -> &[BigRational] {
        &self.0
    }
}

pub(crate) async fn primal_infeasible(
    model: &Model<'_>,
    z: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    primal_infeasible_satisfying(
        model,
        z,
        cooperation,
        async |_: &[BigRational], _: &mut ScalarCooperation| Ok(true),
    )
    .await
}
/// Additional original-domain evidence is tested on each recovered candidate.
/// Rejection continues recovery; it proves no opposite terminal.
pub(crate) async fn primal_infeasible_satisfying<F>(
    model: &Model<'_>,
    z: &[f64],
    cooperation: &mut ScalarCooperation,
    mut domain: F,
) -> BuiltinResult<bool>
where
    F: for<'a> std::ops::AsyncFnMut(
        &'a [BigRational],
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<bool>,
{
    if !model.valid_layout() || z.len() != model.b.len() {
        return Ok(false);
    }
    // An exact row contradiction is a proposal, not source authority. It
    // passes the same full-model and original-domain checks as backend duals.
    let initial = match super::linear::LinearCertificates::borrow(model, cooperation).await? {
        Some(certificates) => certificates.infeasibility_certificate(cooperation).await?,
        None => None,
    };
    Ok(super::recovery::checked_owned(
        z,
        model.identity,
        cooperation,
        initial,
        async |candidate: &[BigRational], cooperation: &mut ScalarCooperation| {
            let mut offset = 0;
            for cone in model.cones {
                let size = dimension(cone).expect("validated layout");
                if !contains(
                    cone,
                    candidate[offset..offset + size].iter(),
                    true,
                    cooperation,
                )
                .await?
                {
                    return Ok(false);
                }
                offset += size;
            }
            for col in 0..model.f.len() {
                let entries = model.a.colptr[col]..model.a.colptr[col + 1];
                let Some(sum) = dot(
                    entries.map(|entry| (model.a.nzval[entry], &candidate[model.a.rowval[entry]])),
                    cooperation,
                )
                .await?
                else {
                    return Ok(false);
                };
                if !sum.is_zero() {
                    return Ok(false);
                }
                cooperation.completed(1).await?;
            }
            let Some(rhs) = dot(model.b.iter().copied().zip(candidate), cooperation).await? else {
                return Ok(false);
            };
            if !rhs.is_negative() {
                return Ok(false);
            }
            domain(candidate, cooperation).await
        },
    )
    .await?
    .is_some())
}

pub(crate) async fn descent_ray(
    model: &Model<'_>,
    direction: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    Ok(verified_descent_ray(model, direction, cooperation)
        .await?
        .is_some())
}
pub(crate) async fn verified_descent_ray(
    model: &Model<'_>,
    direction: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<VerifiedRay>> {
    verified_descent_ray_satisfying(
        model,
        direction,
        cooperation,
        async |_: &[BigRational], _: &mut ScalarCooperation| Ok(true),
    )
    .await
}
pub(crate) async fn verified_descent_ray_satisfying<F>(
    model: &Model<'_>,
    direction: &[f64],
    cooperation: &mut ScalarCooperation,
    mut domain: F,
) -> BuiltinResult<Option<VerifiedRay>>
where
    F: for<'a> std::ops::AsyncFnMut(
        &'a [BigRational],
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<bool>,
{
    if !model.valid_layout() || direction.len() != model.f.len() {
        return Ok(None);
    }
    super::recovery::checked_owned(
        direction,
        model.identity,
        cooperation,
        None,
        async |candidate: &[BigRational], cooperation: &mut ScalarCooperation| {
            if !rational_descent_ray(model, candidate, cooperation).await? {
                return Ok(false);
            }
            domain(candidate, cooperation).await
        },
    )
    .await
    .map(|point| point.map(VerifiedRay))
}
/// Exact acceptance owner shared with retained domain-specific ray certificates.
pub(crate) async fn rational_descent_ray(
    model: &Model<'_>,
    candidate: &[BigRational],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if !model.valid_layout() || candidate.len() != model.f.len() {
        return Ok(false);
    }
    if let Some(p) = model.p {
        let mut product = vec![BigRational::zero(); candidate.len()];
        for col in 0..p.n {
            for (work, entry) in (p.colptr[col]..p.colptr[col + 1]).enumerate() {
                let row = p.rowval[entry];
                let Some(coefficient) = BigRational::from_float(p.nzval[entry]) else {
                    return Ok(false);
                };
                product[row] += &coefficient * &candidate[col];
                if row != col {
                    product[col] += coefficient * &candidate[row];
                }
                if work % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
            cooperation.completed(1).await?;
        }
        if product.iter().any(|value| !value.is_zero()) {
            return Ok(false);
        }
    }
    let Some(slack) = negative_activity(model, candidate, cooperation).await? else {
        return Ok(false);
    };
    if !slack_in_cones(model, &slack, cooperation).await? {
        return Ok(false);
    }
    let Some(cost) = dot(model.f.iter().copied().zip(candidate), cooperation).await? else {
        return Ok(false);
    };
    Ok(cost.is_negative())
}

pub(crate) async fn feasible_point(
    model: &Model<'_>,
    point: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    Ok(verified_point(model, point, cooperation).await?.is_some())
}
pub(crate) async fn verified_point(
    model: &Model<'_>,
    point: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<VerifiedPoint>> {
    verified_point_satisfying(
        model,
        point,
        cooperation,
        async |_: &[BigRational], _: &mut ScalarCooperation| Ok(true),
    )
    .await
}

/// Domain constraints participate in each recovery proposal's acceptance, so
/// a continuous feasible proposal cannot suppress a later discrete witness.
/// Every accepted proposal still passes the exact original conic model.
pub(crate) async fn verified_point_satisfying<F>(
    model: &Model<'_>,
    point: &[f64],
    cooperation: &mut ScalarCooperation,
    mut domain: F,
) -> BuiltinResult<Option<VerifiedPoint>>
where
    F: for<'a> std::ops::AsyncFnMut(
        &'a [BigRational],
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<bool>,
{
    if !model.valid_layout() || point.len() != model.f.len() {
        return Ok(None);
    }
    super::recovery::checked_owned(
        point,
        model.identity,
        cooperation,
        None,
        async |candidate: &[BigRational], cooperation: &mut ScalarCooperation| {
            if !domain(candidate, cooperation).await? {
                return Ok(false);
            }
            feasible_rational_point(model, candidate, cooperation).await
        },
    )
    .await
    .map(|point| point.map(VerifiedPoint))
}

pub(crate) async fn feasible_rational_point(
    model: &Model<'_>,
    candidate: &[BigRational],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if !model.valid_layout() || candidate.len() != model.f.len() {
        return Ok(false);
    }
    let Some(mut slack) = negative_activity(model, candidate, cooperation).await? else {
        return Ok(false);
    };
    for (row, &rhs) in model.b.iter().enumerate() {
        let Some(rhs) = BigRational::from_float(rhs) else {
            return Ok(false);
        };
        if !rhs.is_zero() {
            match &mut slack[row] {
                Some(sum) => *sum += rhs,
                slot @ None => *slot = Some(rhs),
            }
        }
        if row % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    slack_in_cones(model, &slack, cooperation).await
}

async fn slack_in_cones(
    model: &Model<'_>,
    values: &[Option<BigRational>],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let zero = BigRational::zero();
    let mut offset = 0;
    for cone in model.cones {
        let size = dimension(cone).expect("validated layout");
        let entries = values[offset..offset + size]
            .iter()
            .map(|value| value.as_ref().unwrap_or(&zero));
        if !contains(cone, entries, false, cooperation).await? {
            return Ok(false);
        }
        offset += size;
    }
    Ok(true)
}

pub(super) async fn contains<'a>(
    cone: &SupportedConeT<f64>,
    mut values: impl Iterator<Item = &'a BigRational>,
    dual: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    match cone {
        SupportedConeT::ZeroConeT(_) if dual => {} // dual of {0} is the whole space
        SupportedConeT::ZeroConeT(_) | SupportedConeT::NonnegativeConeT(_) => {
            for (index, value) in values.enumerate() {
                if matches!(cone, SupportedConeT::ZeroConeT(_)) && !value.is_zero()
                    || matches!(cone, SupportedConeT::NonnegativeConeT(_)) && value.is_negative()
                {
                    return Ok(false);
                }
                if index % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
        }
        SupportedConeT::SecondOrderConeT(_) => {
            let Some(head) = values.next() else {
                return Ok(false);
            };
            if head.is_negative() {
                return Ok(false);
            }
            let mut margin = head * head;
            for (index, value) in values.enumerate() {
                margin -= value * value;
                if margin.is_negative() {
                    return Ok(false);
                }
                if index % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
        }
        _ => return Ok(false),
    }
    cooperation.completed(1).await?;
    Ok(true)
}

pub(super) async fn negative_activity(
    model: &Model<'_>,
    values: &[BigRational],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<Option<BigRational>>>> {
    // One exact accumulator per original row; no CSR/model/kernel copies.
    let mut result: Vec<Option<BigRational>> = vec![None; model.b.len()];
    for (col, value) in values.iter().enumerate() {
        if !value.is_zero() {
            for first in (model.a.colptr[col]..model.a.colptr[col + 1]).step_by(1024) {
                let end = first.saturating_add(1024).min(model.a.colptr[col + 1]);
                for entry in first..end {
                    // Negation is an exact sign change of the original finite
                    // coefficient, not a normalized/rounded matrix product.
                    let Some(coefficient) = BigRational::from_float(-model.a.nzval[entry]) else {
                        return Ok(None);
                    };
                    if !coefficient.is_zero() {
                        let product = coefficient * value;
                        match &mut result[model.a.rowval[entry]] {
                            Some(sum) => *sum += product,
                            slot @ None => *slot = Some(product),
                        }
                    }
                }
                cooperation.completed((end - first) as u64).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(Some(result))
}

pub(super) async fn dot<'a>(
    pairs: impl Iterator<Item = (f64, &'a BigRational)>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<BigRational>> {
    let mut sum = BigRational::zero();
    let mut pending = 0;
    for (left, right) in pairs {
        let Some(left) = BigRational::from_float(left) else {
            return Ok(None);
        };
        sum += left * right;
        pending += 1;
        if pending == 1024 {
            cooperation.completed(pending).await?;
            pending = 0;
        }
    }
    cooperation.completed(pending).await?;
    Ok(Some(sum))
}
