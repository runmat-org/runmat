//! Supplementary original-source authority for the existing MILP engine.
use super::*;
use common::conic::linear::{ExactLowerBound, LinearCertificates};
use num_rational::BigRational;

/// Borrowed actual node generation, including temporary integer-fixing repairs.
/// Original source roles and generated bound rows share one immutable ordering.
pub(in crate::builtins::math::optim) struct Node<'a> {
    pub model: common::conic::Model<'a>,
    pub original_rows: usize,
    pub equalities: usize,
    pub lower: &'a [f64],
    pub upper: &'a [f64],
}

pub(in crate::builtins::math::optim) struct Publication<'view, 'node> {
    pub node: &'view Node<'node>,
    pub values: &'view [f64],
    pub exact_values: &'view [BigRational],
    pub domains_admitted: bool,
    pub tolerance: f64,
    pub cutoff: Option<&'view BigRational>,
}
pub(in crate::builtins::math::optim) trait Validation {
    /// True only when supplied binary64 matrices ARE the original model.
    /// Original DAG models cannot consume a rounded singleton contradiction.
    fn matrix_is_original(&self) -> bool;
    async fn lower_bound(
        &mut self,
        node: &Node<'_>,
        supplied: &LinearCertificates<'_, '_>,
        dual: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<ExactLowerBound>>;
    async fn farkas(
        &mut self,
        node: &Node<'_>,
        weights: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn ray(
        &mut self,
        node: &Node<'_>,
        direction: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn feasible(
        &mut self,
        node: &Node<'_>,
        point: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    /// A verified original feasible incumbent may replace only its exact
    /// ranking objective. Coordinates remain the same moved exact witness.
    async fn incumbent(
        &mut self,
        node: &Node<'_>,
        point: &[BigRational],
        objective: &mut BigRational,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn publication(
        &mut self,
        point: Publication<'_, '_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<super::super::publication::Evaluation>;
}

pub(in crate::builtins::math::optim::intlinprog) struct Supplied;
impl Validation for Supplied {
    fn matrix_is_original(&self) -> bool {
        true
    }
    async fn lower_bound(
        &mut self,
        _: &Node<'_>,
        supplied: &LinearCertificates<'_, '_>,
        dual: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<ExactLowerBound>> {
        supplied.lower_bound(dual, c).await
    }
    async fn farkas(
        &mut self,
        _: &Node<'_>,
        _: &[BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn ray(
        &mut self,
        _: &Node<'_>,
        _: &[BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn feasible(
        &mut self,
        _: &Node<'_>,
        _: &[BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn incumbent(
        &mut self,
        _: &Node<'_>,
        _: &[BigRational],
        _: &mut BigRational,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn publication(
        &mut self,
        point: Publication<'_, '_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<super::super::publication::Evaluation> {
        let Publication {
            node,
            values: _,
            exact_values: values,
            domains_admitted,
            tolerance,
            cutoff,
        } = point;
        let (objective, violation) = common::conic::linear::evaluate(&node.model, values, c)
            .await?
            .ok_or_else(|| numerical("cannot evaluate the public point in the original model"))?;
        let tolerance = BigRational::from_float(tolerance).expect("finite admitted tolerance");
        let admitted = domains_admitted
            && violation <= tolerance
            && cutoff.is_none_or(|cutoff| &objective <= cutoff);
        Ok(super::super::publication::Evaluation {
            objective: super::super::publication::number(&objective),
            violation: super::super::publication::nonnegative(&violation),
            admitted,
        })
    }
}

pub(in crate::builtins::math::optim::intlinprog) struct Terminal<'a, 'node, V> {
    pub node: &'a Node<'node>,
    pub validation: &'a mut V,
}
impl<V: Validation> common::conic::solve::TerminalValidation for Terminal<'_, '_, V> {
    async fn farkas(
        &mut self,
        weights: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.validation.farkas(self.node, weights, c).await
    }
    async fn ray(
        &mut self,
        direction: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.validation.ray(self.node, direction, c).await
    }
    async fn feasible(
        &mut self,
        point: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.validation.feasible(self.node, point, c).await
    }
}
