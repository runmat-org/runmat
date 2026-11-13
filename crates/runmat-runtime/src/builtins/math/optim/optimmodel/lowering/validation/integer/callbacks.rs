//! Admitted original-source callbacks for the numerical integer engine.
//!
//! Original-source validation owns its cooperative proof state. Keep that concrete
//! future layout at this borrowed domain boundary rather than propagating it through
//! the conic node solver and integer tree. The existing continuation owner checks
//! cancellation and charges the selected layout before its factory runs. Source
//! roots stay with the validator; numerical buffers are borrowed, never cloned.
use super::*;
use crate::execution::admit_borrowed_operation_with_contextual_error;

impl mip::Validation for Validator<'_> {
    fn matrix_is_original(&self) -> bool {
        false
    }

    async fn lower_bound(
        &mut self,
        node: &mip::Node<'_>,
        supplied: &LinearCertificates<'_, '_>,
        dual: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<ExactLowerBound>> {
        let runtime = c.runtime().cloned();
        admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
            self.original_lower_bound(node, supplied, dual, c)
        })?
        .await
    }

    async fn farkas(
        &mut self,
        node: &mip::Node<'_>,
        candidate: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let runtime = c.runtime().cloned();
        admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
            self.original_farkas(node, candidate, c)
        })?
        .await
    }

    async fn ray(
        &mut self,
        node: &mip::Node<'_>,
        direction: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let runtime = c.runtime().cloned();
        admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
            self.original_ray(node, direction, c)
        })?
        .await
    }

    async fn feasible(
        &mut self,
        node: &mip::Node<'_>,
        point: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let runtime = c.runtime().cloned();
        admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
            self.original_feasible(node, point, c)
        })?
        .await
    }

    async fn incumbent(
        &mut self,
        node: &mip::Node<'_>,
        point: &[BigRational],
        objective: &mut BigRational,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let runtime = c.runtime().cloned();
        admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
            self.original_incumbent(node, point, objective, c)
        })?
        .await
    }

    async fn publication(
        &mut self,
        point: mip::Publication<'_, '_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<mip::Evaluation> {
        let runtime = c.runtime().cloned();
        admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
            self.original_publication(point, c)
        })?
        .await
    }
}
