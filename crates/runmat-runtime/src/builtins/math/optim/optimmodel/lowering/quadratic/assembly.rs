//! One upper-triangle Hessian and linear term from exact degree-two entries.
//! Constants remain exclusively in the original public objective evaluation.
use super::*;
use crate::builtins::math::optim::common::conic::convexity::{certify_original, OriginalHessian};
use runmat_value::optimization_model::OptimizationObjectiveSense;
impl Compiled {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn assemble(
        self,
        columns: usize,
        sense: OptimizationObjectiveSense,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> Lowered<Prepared> {
        let mut linear =
            runmat_value::exact_capacity_vec(columns).map_err(|e| policy.storage(e))?;
        let mut linear_abs =
            runmat_value::exact_capacity_vec(columns).map_err(|e| policy.storage(e))?;
        let mut curvature_abs =
            runmat_value::exact_capacity_vec(columns).map_err(|e| policy.storage(e))?;
        for _ in 0..columns {
            linear.push(0.);
            linear_abs.push(Rational::zero());
            curvature_abs.push(Rational::zero());
            c.completed(1).await?;
        }
        let sign = Weight::constant(
            if sense == OptimizationObjectiveSense::Minimize {
                1.
            } else {
                -1.
            },
            policy,
        )?;
        let two = Weight::constant(2., policy)?;
        let mut exact = OriginalHessian::new(columns, policy, c).await?;
        let mut entries = Vec::new();
        for (term, value) in self.cost.entries() {
            let value = value.mul(&sign, policy, c).await?;
            match term {
                Monomial::Constant => {}
                Monomial::Linear(column) => {
                    let target = linear.get_mut(column).ok_or_else(|| {
                        policy.invalid("quadratic cost column exceeds original layout")
                    })?;
                    if !value.proposal.is_finite() {
                        return Ok(Err(Stop::NonFinite));
                    }
                    *target = value.proposal;
                    linear_abs[column] = value.exact.abs();
                }
                Monomial::Quadratic(row, column) => {
                    if row > column || column >= columns {
                        return Err(
                            policy.invalid("quadratic Hessian coordinate exceeds original layout")
                        );
                    }
                    // 1/2*x'H*x: diagonal entries double; each upper off-diagonal
                    // entry appears twice in the symmetric quadratic product.
                    let value = if row == column {
                        value.mul(&two, policy, c).await?
                    } else {
                        value
                    };
                    if !value.proposal.is_finite() {
                        return Ok(Err(Stop::NonFinite));
                    }
                    let magnitude = value.exact.abs();
                    curvature_abs[row] = curvature_abs[row].add(&magnitude, policy, c).await?;
                    if row != column {
                        curvature_abs[column] =
                            curvature_abs[column].add(&magnitude, policy, c).await?;
                    }
                    exact.add(row, column, value.exact, policy, c).await?;
                    if value.proposal != 0. {
                        entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                        entries.push(terms::Term {
                            column,
                            row,
                            value: value.proposal,
                        });
                    }
                }
            }
            c.completed(1).await?;
        }
        let curvature = if certify_original(exact, policy, c).await? {
            Curvature::Convex
        } else {
            Curvature::Nonconvex
        };
        let hessian =
            match super::super::assembly::matrix(entries, columns, columns, policy, c).await? {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            };
        Ok(Ok(Prepared {
            cost: Cost { hessian, linear },
            evidence: Evidence {
                restricted: (curvature == Curvature::Nonconvex).then_some(self.cost),
                selection: self.selection,
                atoms: self.atoms,
                curvature,
                projection: self.projection,
                scale: super::super::proof::ObjectiveScale::from_parts(
                    linear_abs,
                    curvature_abs,
                    policy,
                    c,
                )
                .await?,
            },
            facts: self.facts,
        }))
    }
}
