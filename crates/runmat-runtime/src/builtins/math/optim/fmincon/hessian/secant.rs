//! Only O(n) differences cross an accepted-state ownership transition. The
//! controller decides whether/when another numerical model is actually needed.
use super::super::{model::Linearization, validation};
use super::{storage, BuiltinCatalogIdentity, BuiltinResult, ScalarCooperation};
pub(in crate::builtins::math::optim::fmincon) struct LagrangianPoint<'a> {
    pub x: &'a [f64],
    pub gradient: &'a [f64],
    pub constraints: &'a Linearization,
}
pub(in crate::builtins::math::optim::fmincon) struct Secant {
    pub displacement: Vec<f64>,
    pub change: Vec<f64>,
}
pub(in crate::builtins::math::optim::fmincon) async fn between(
    old: LagrangianPoint<'_>,
    new: LagrangianPoint<'_>,
    multipliers: &[f64],
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Secant>> {
    if old.x.len() != new.x.len() {
        return Ok(None);
    }
    let Some(previous) = validation::lagrangian(
        old.gradient,
        old.constraints,
        multipliers,
        owner,
        cooperation,
    )
    .await?
    else {
        return Ok(None);
    };
    let Some(next) = validation::lagrangian(
        new.gradient,
        new.constraints,
        multipliers,
        owner,
        cooperation,
    )
    .await?
    else {
        return Ok(None);
    };
    let mut displacement = storage(old.x.len(), owner)?;
    let mut change = storage(old.x.len(), owner)?;
    for index in 0..old.x.len() {
        let dx = new.x[index] - old.x[index];
        let dy = next[index] - previous[index];
        if !dx.is_finite() || !dy.is_finite() {
            return Ok(None);
        }
        displacement.push(dx);
        change.push(dy);
        cooperation.completed(1).await?;
    }
    Ok(Some(Secant {
        displacement,
        change,
    }))
}
