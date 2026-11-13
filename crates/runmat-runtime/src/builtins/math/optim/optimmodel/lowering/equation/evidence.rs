//! Original equation differences provide affine residual proof data directly.
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational;
impl Evidence {
    pub(super) async fn seeds(
        &self,
        point: &exact::Evaluated,
        origin: &exact::Evaluated,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<(Vec<exact::Seed>, Rational)> {
        let mut seeds = Vec::new();
        let mut cost = Rational::zero();
        for (entry, geometry) in self.formulation.entries.iter().zip(&self.geometries) {
            for row in 0..geometry.plan.len() {
                let (left, right) = geometry
                    .indices(row)
                    .ok_or_else(|| policy.invalid("equation source geometry changed"))?;
                let value = difference(point, entry, left, right, policy, c).await?;
                let offset = difference(origin, entry, left, right, policy, c).await?;
                let weight = Rational::from_usize(2).mul(&value, policy, c).await?;
                cost = cost
                    .add(
                        &weight
                            .mul(&value.sub(&offset, policy, c).await?, policy, c)
                            .await?,
                        policy,
                        c,
                    )
                    .await?;
                seeds.try_reserve(2).map_err(|e| policy.reserve(e))?;
                seeds.push(exact::Seed {
                    root: entry.left,
                    row: left,
                    value: weight.clone(),
                });
                seeds.push(exact::Seed {
                    root: entry.right,
                    row: right,
                    value: -weight,
                });
                c.completed(1).await?;
            }
        }
        Ok((seeds, cost))
    }
    pub(super) async fn scale(
        &self,
        program: &Program,
        origin: &exact::Evaluated,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<proof::ObjectiveScale> {
        let mut linear = runmat_value::exact_capacity_vec(program.layout.columns)
            .map_err(|e| policy.storage(e))?;
        let mut curvature = runmat_value::exact_capacity_vec(program.layout.columns)
            .map_err(|e| policy.storage(e))?;
        for _ in 0..program.layout.columns {
            linear.push(Rational::zero());
            curvature.push(Rational::zero());
            c.completed(1).await?;
        }
        let mut reverse = exact::SparsePlan::new(program, policy, c).await?;
        for (entry, geometry) in self.formulation.entries.iter().zip(&self.geometries) {
            for row in 0..geometry.plan.len() {
                let (left, right) = geometry
                    .indices(row)
                    .ok_or_else(|| policy.invalid("equation source geometry changed"))?;
                let offset = difference(origin, entry, left, right, policy, c).await?;
                let mut gradient = reverse
                    .row(program, origin, entry.left, left, policy, c)
                    .await?;
                for (column, coefficient) in reverse
                    .row(program, origin, entry.right, right, policy, c)
                    .await?
                {
                    let value = match gradient.remove(&column) {
                        Some(previous) => previous.sub(&coefficient, policy, c).await?,
                        None => -coefficient,
                    };
                    if !value.is_zero() {
                        gradient.try_reserve(1).map_err(|e| policy.reserve(e))?;
                        gradient.insert(column, value);
                    }
                    c.completed(1).await?;
                }
                let mut norm = Rational::zero();
                for coefficient in gradient.values() {
                    norm = norm.add(&coefficient.abs(), policy, c).await?;
                    c.completed(1).await?;
                }
                for (column, coefficient) in gradient {
                    let coefficient = Rational::from_usize(2)
                        .mul(&coefficient.abs(), policy, c)
                        .await?;
                    linear[column] = linear[column]
                        .add(&coefficient.mul(&offset.abs(), policy, c).await?, policy, c)
                        .await?;
                    curvature[column] = curvature[column]
                        .add(&coefficient.mul(&norm, policy, c).await?, policy, c)
                        .await?;
                    c.completed(1).await?;
                }
            }
        }
        proof::ObjectiveScale::from_parts(linear, curvature, policy, c).await
    }
}
async fn difference(
    values: &exact::Evaluated,
    entry: &formulation::Entry,
    left: usize,
    right: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Rational> {
    let left = values
        .value(entry.left, left, policy)?
        .ok_or_else(|| policy.invalid("original equation left value is unassigned"))?;
    let right = values
        .value(entry.right, right, policy)?
        .ok_or_else(|| policy.invalid("original equation right value is unassigned"))?;
    left.sub(&right, policy, c).await
}
