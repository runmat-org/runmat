//! Exact original row preparation retires before stationarity and gap checks.
use super::*;
use futures::future::Either;
pub(super) struct Data {
    pub(super) weighted: super::super::farkas::Weighted,
    pub(super) weights: Vec<Rational>,
    pub(super) candidate: Vec<Rational>,
}
pub(super) struct Prepared<'a, 'source> {
    authority: &'a mut Authority<'source>,
    data: Data,
    roles: &'a [RowRole],
    tolerance: f64,
    cooperation: &'a mut ScalarCooperation,
}
pub(super) async fn prepare<'a, 'source>(
    authority: &'a mut Authority<'source>,
    point: &'a [f64],
    roles: &'a [RowRole],
    duals: &'a [f64],
    tolerance: f64,
    cooperation: &'a mut ScalarCooperation,
) -> BuiltinResult<Option<Prepared<'a, 'source>>> {
    let policy = authority.policy;
    if point.len() != authority.program.layout.columns || roles.len() != duals.len() {
        return Ok(None);
    }
    let mut candidate =
        runmat_value::exact_capacity_vec(point.len()).map_err(|e| policy.storage(e))?;
    for &value in point {
        let Some(value) = Rational::from_float(value) else {
            return Ok(None);
        };
        candidate.push(value);
        cooperation.completed(1).await?;
    }
    let mut weights =
        runmat_value::exact_capacity_vec(duals.len()).map_err(|e| policy.storage(e))?;
    for &value in duals {
        let Some(value) = Rational::from_float(value) else {
            return Ok(None);
        };
        weights.push(value);
        cooperation.completed(1).await?;
    }
    let Some(weighted) = authority
        .weighted_rows(roles, &weights, cooperation)
        .await?
    else {
        return Ok(None);
    };
    Ok(Some(Prepared {
        authority,
        data: Data {
            candidate,
            weights,
            weighted,
        },
        roles,
        tolerance,
        cooperation,
    }))
}
pub(super) fn select<'a, 'source: 'a>(
    prepared: Option<Prepared<'a, 'source>>,
) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a + use<'a, 'source> {
    match prepared {
        None => Either::Left(std::future::ready(Ok(false))),
        Some(Prepared {
            authority,
            data,
            roles,
            tolerance,
            cooperation,
        }) => Either::Right(authority.certify(data, roles, tolerance, cooperation)),
    }
}
