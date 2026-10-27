//! Original-row stationarity/gap/complementarity at the supplied floating
//! candidate. Exact arithmetic prevents rounded coefficient erasure from
//! authorizing optimality; ordinary primal tolerance remains a separate check.
use super::super::quadratic::directional_cost;
use super::*;
mod phases;
mod certificate;
impl<'source> Authority<'source> {
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn kkt<'a>(
        &'a mut self,
        point: &'a [f64],
        roles: &'a [RowRole],
        duals: &'a [f64],
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a + use<'a, 'source> {
        self.kkt_at_tolerance(
            point,
            roles,
            duals,
            super::super::super::super::linprog::compiled::validation_tolerance(),
            cooperation,
        )
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn kkt_at_tolerance<'a>(
        &'a mut self,
        point: &'a [f64],
        roles: &'a [RowRole],
        duals: &'a [f64],
        tolerance: f64,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a + use<'a, 'source> {
        crate::execution::after_result(
            phases::prepare(self, point, roles, duals, tolerance, cooperation),
            phases::select,
        )
    }
    fn certify<'a>(
        &'a mut self,
        data: phases::Data,
        roles: &'a [RowRole],
        tolerance: f64,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a + use<'a, 'source> {
        crate::execution::after_result(
            certificate::prepare(self, data, cooperation),
            move |(certificate, authority, cooperation)| {
                certificate::check(authority, certificate, roles, tolerance, cooperation)
            },
        )
    }
    async fn residual(
        &self,
        role: &RowRole,
        point: &[Rational],
        evaluated: &exact::Evaluated,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Rational>> {
        let policy = self.policy;
        match *role {
            RowRole::Equality(row) | RowRole::Inequality(row) => {
                let ranges = if matches!(role, RowRole::Equality(_)) {
                    &self.equalities
                } else {
                    &self.inequalities
                };
                let index = ranges.partition_point(|range| range.end <= row);
                let Some(range) = ranges.get(index).filter(|range| row >= range.start) else {
                    return Ok(None);
                };
                let constraint = &self.formulation.constraints[range.constraint];
                let local = row - range.start;
                let Some((a, b)) = self.geometries[range.constraint].indices(local) else {
                    return Ok(None);
                };
                let Some(left) = evaluated.value(constraint.left, a, policy)? else {
                    return Ok(None);
                };
                let Some(right) = evaluated.value(constraint.right, b, policy)? else {
                    return Ok(None);
                };
                Ok(Some(if constraint.relation == Relation::GreaterEqual {
                    right.sub(&left, policy, cooperation).await?
                } else {
                    left.sub(&right, policy, cooperation).await?
                }))
            }
            RowRole::Lower(column) | RowRole::Upper(column) => {
                let Some((variable, row)) = self.program.layout.coordinate(column) else {
                    return Ok(None);
                };
                let lower = matches!(role, RowRole::Lower(_));
                let Some(bound) = variable
                    .view
                    .definition()
                    .bound_value(if lower { Bound::Lower } else { Bound::Upper }, row)
                    .filter(|value| value.is_finite())
                else {
                    return Ok(None);
                };
                let bound = rational(bound, policy)?;
                Ok(Some(if lower {
                    bound.sub(&point[column], policy, cooperation).await?
                } else {
                    point[column].sub(&bound, policy, cooperation).await?
                }))
            }
        }
    }
}
