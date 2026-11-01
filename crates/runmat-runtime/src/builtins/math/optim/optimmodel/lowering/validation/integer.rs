//! Original-source decisions for the one existing MILP tree and its node boxes.
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational;
use crate::builtins::math::optim::{
    common::conic::linear::{ExactLowerBound, LinearCertificates},
    intlinprog::compiled as mip,
};
use num_rational::BigRational;
mod publication;
mod callbacks;
pub(super) struct Rows {
    roles: Vec<RowRole>,
    original: usize,
    equalities: usize,
}
impl Validator<'_> {
    async fn integer_node(
        &mut self,
        node: &mip::Node<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let policy = self.policy;
        let n = self.program.layout.columns;
        if self.integer_rows.is_none() {
            let mut equalities = 0usize;
            let mut inequalities = 0usize;
            for constraint in &self.formulation.constraints {
                let count = evaluation::element_count(&constraint.shape, policy)?;
                let total = if constraint.relation
                    == runmat_value::optimization_model::OptimizationConstraintRelation::Equal
                {
                    &mut equalities
                } else {
                    &mut inequalities
                };
                *total = total
                    .checked_add(count)
                    .ok_or_else(|| policy.invalid("original integer row count overflow"))?;
                c.completed(1).await?;
            }
            let original = equalities
                .checked_add(inequalities)
                .ok_or_else(|| policy.invalid("integer row count overflow"))?;
            let count = n
                .checked_mul(2)
                .and_then(|v| v.checked_add(original))
                .ok_or_else(|| policy.invalid("integer bound row count overflow"))?;
            let mut roles =
                runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
            for row in 0..equalities {
                roles.push(RowRole::Equality(row));
                c.completed(1).await?;
            }
            for row in 0..inequalities {
                roles.push(RowRole::Inequality(row));
                c.completed(1).await?;
            }
            for column in 0..n {
                roles.push(RowRole::Lower(column));
                roles.push(RowRole::Upper(column));
                c.completed(1).await?;
            }
            self.integer_rows = Some(Rows {
                roles,
                original,
                equalities,
            });
        }
        let rows = self
            .integer_rows
            .as_ref()
            .expect("original integer row roles");
        if node.original_rows != rows.original
            || node.equalities != rows.equalities
            || node.model.f.len() != n
            || node.lower.len() != n
            || node.upper.len() != n
            || node.model.b.len() != rows.roles.len()
        {
            return Err(policy.invalid("MILP node does not match its original model layout"));
        }
        for (&lower, &upper) in node.lower.iter().zip(node.upper) {
            if lower.is_nan() || upper.is_nan() {
                return Err(policy.invalid("MILP node bound is NaN"));
            }
            c.completed(1).await?;
        }
        Ok(())
    }
    async fn integer_weights(
        &self,
        node: &mip::Node<'_>,
        values: &[BigRational],
        project: bool,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<Rational>>> {
        let policy = self.policy;
        let rows = self.integer_rows.as_ref().expect("validated integer node");
        if values.len() != rows.roles.len() {
            return Ok(None);
        }
        let mut weights =
            runmat_value::exact_capacity_vec(values.len()).map_err(|e| policy.storage(e))?;
        for (role, value) in rows.roles.iter().zip(values) {
            use num_traits::Signed;
            let disabled = match *role {
                RowRole::Lower(column) => !node.lower[column].is_finite(),
                RowRole::Upper(column) => !node.upper[column].is_finite(),
                _ => false,
            };
            if disabled || project && !matches!(role, RowRole::Equality(_)) && value.is_negative() {
                weights.push(Rational::zero());
            } else {
                if !matches!(role, RowRole::Equality(_)) && value.is_negative() {
                    return Ok(None);
                }
                let Some(value) = Rational::import(value, policy, c).await? else {
                    return Ok(None);
                };
                weights.push(value);
            }
            c.completed(1).await?;
        }
        Ok(Some(weights))
    }
}
impl Validator<'_> {
    async fn original_lower_bound(
        &mut self,
        node: &mip::Node<'_>,
        _: &LinearCertificates<'_, '_>,
        dual: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<ExactLowerBound>> {
        self.integer_node(node, c).await?;
        let policy = self.policy;
        let mut best: Option<BigRational> = None;
        let _ = crate::builtins::math::optim::common::conic::propose_exact_candidates(
            dual,
            policy.identity,
            c,
            async |candidate: &[BigRational], c: &mut ScalarCooperation| {
                if let Some(weights) = self.integer_weights(node, candidate, true, c).await? {
                    let rows = self
                        .integer_rows
                        .as_ref()
                        .expect("original integer row roles");
                    if let Some(value) = self
                        .authority
                        .integer_lower_bound(
                            &rows.roles,
                            &weights,
                            proof::BoundValues {
                                lower: node.lower,
                                upper: node.upper,
                            },
                            c,
                        )
                        .await?
                    {
                        let value = value.export(policy, c).await?;
                        if best.as_ref().is_none_or(|old| old < &value) {
                            best = Some(value);
                        }
                    }
                }
                Ok(false)
            },
        )
        .await?;
        Ok(best.map(ExactLowerBound::from_verified_original))
    }
    async fn original_farkas(
        &mut self,
        node: &mip::Node<'_>,
        candidate: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.integer_node(node, c).await?;
        let Some(weights) = self.integer_weights(node, candidate, false, c).await? else {
            return Ok(false);
        };
        let rows = self
            .integer_rows
            .as_ref()
            .expect("original integer row roles");
        self.authority
            .integer_farkas(
                &rows.roles,
                &weights,
                proof::BoundValues {
                    lower: node.lower,
                    upper: node.upper,
                },
                c,
            )
            .await
    }
    async fn original_ray(
        &mut self,
        node: &mip::Node<'_>,
        direction: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.integer_node(node, c).await?;
        self.authority
            .certificate(CertificateCandidate::Descent(direction), c)
            .await
    }
    async fn original_feasible(
        &mut self,
        node: &mip::Node<'_>,
        point: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.integer_node(node, c).await?;
        let Some(point) = proof::imported(point, self.policy, c).await? else {
            return Ok(false);
        };
        // This is a relaxation witness, not an incumbent. The existing tree
        // still owns integer/semi disjunctions; use its actual convex node box.
        self.authority
            .feasible_at(
                &point,
                Some(proof::BoundValues {
                    lower: node.lower,
                    upper: node.upper,
                }),
                c,
            )
            .await
    }
    async fn original_incumbent(
        &mut self,
        node: &mip::Node<'_>,
        point: &[BigRational],
        objective: &mut BigRational,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.integer_node(node, c).await?;
        if !self
            .authority
            .certificate(CertificateCandidate::Feasible(point), c)
            .await?
        {
            return Ok(false);
        }
        let policy = self.policy;
        let Some(imported) = proof::imported(point, policy, c).await? else {
            return Ok(false);
        };
        *objective = self
            .authority
            .integer_objective(&imported, c)
            .await?
            .export(policy, c)
            .await?;
        Ok(true)
    }
    async fn original_publication(
        &mut self,
        point: mip::Publication<'_, '_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<crate::builtins::math::optim::intlinprog::compiled::Evaluation> {
        let mip::Publication {
            node,
            values,
            exact_values,
            domains_admitted,
            tolerance,
            cutoff,
        } = point;
        self.integer_node(node, c).await?;
        self.integer_publication(values, exact_values, domains_admitted, tolerance, cutoff, c)
            .await
    }
}
