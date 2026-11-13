//! Additional original-model evidence, after canonical residual/KKT checks.
use super::*;
use num_rational::BigRational;
use std::future::Future;
/// A borrowed projection of the authenticated lifted model, not copied rows.
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Rows<'a>(Source<'a>);
#[derive(Clone, Copy)]
enum Source<'a> {
    Lifted(&'a model::Model),
    Unlifted(&'a [conic::LinearRowRole]),
}
impl<'a> Rows<'a> {
    pub(in crate::builtins::math::optim::lsqlin) fn new(model: &'a model::Model) -> Self {
        Self(Source::Lifted(model))
    }
    pub(in crate::builtins::math::optim::lsqlin) fn bounds(
        rows: &'a [conic::LinearRowRole],
    ) -> Self {
        Self(Source::Unlifted(rows))
    }
    pub(in crate::builtins::math::optim::lsqlin) fn original(
        rows: &'a [conic::LinearRowRole],
    ) -> Self {
        Self(Source::Unlifted(rows))
    }
    pub(in crate::builtins::math::optim) fn iter(
        self,
    ) -> impl Iterator<Item = conic::LinearRowRole> + 'a {
        let count = match self.0 {
            Source::Lifted(model) => model.external_role_count(),
            Source::Unlifted(rows) => rows.len(),
        };
        (0..count).filter_map(move |index| match self.0 {
            Source::Lifted(model) => model.external_role(index),
            Source::Unlifted(rows) => rows.get(index).copied(),
        })
    }
    pub(in crate::builtins::math::optim) fn external<T>(self, values: &[T]) -> Option<&[T]> {
        match self.0 {
            Source::Lifted(model) => (values.len() == model.b.len()).then(|| &values[model.c.m..]),
            Source::Unlifted(rows) => (values.len() == rows.len()).then_some(values),
        }
    }
}
pub(in crate::builtins::math::optim) enum Certificate<'a> {
    InconsistentBounds {
        column: usize,
        lower: f64,
        upper: f64,
    },
    Farkas {
        rows: Rows<'a>,
        multipliers: &'a [BigRational],
    },
}
pub(in crate::builtins::math::optim) trait Validation {
    // Plain algorithm data only: these hooks confer no feasible/KKT authority.
    fn saved_working_set(&self) -> Option<&runmat_value::warm_start::LinearWorkingSetData> {
        None
    }
    fn wants_working_set(&self) -> bool {
        false
    }
    fn retain_working_set(&mut self, _data: runmat_value::warm_start::LinearWorkingSetData) {}
    /// Retain actual evaluated-only coordinates separately from feasible proof.
    fn stopped_candidate<'a>(
        &'a mut self,
        point: &'a [f64],
        c: &'a mut ScalarCooperation,
    ) -> impl Future<Output = BuiltinResult<bool>> + 'a;
    /// Only an actual original feasible objective may trigger the threshold.
    fn objective_limit<'a>(
        &'a mut self,
        point: &'a [f64],
        cost: f64,
        limit: f64,
        feasibility: f64,
        c: &'a mut ScalarCooperation,
    ) -> impl Future<Output = BuiltinResult<Option<bool>>> + 'a;

    fn solution<'a>(
        &'a mut self,
        point: &'a [f64],
        rows: Rows<'a>,
        duals: &'a [f64],
        feasibility: f64,
        optimality: f64,
        c: &'a mut ScalarCooperation,
    ) -> impl Future<Output = BuiltinResult<bool>> + 'a;
    fn incumbent<'a>(
        &'a mut self,
        point: &'a [f64],
        feasibility: f64,
        c: &'a mut ScalarCooperation,
    ) -> impl Future<Output = BuiltinResult<bool>> + 'a;
    fn certificate<'a>(
        &'a mut self,
        candidate: Certificate<'a>,
        c: &'a mut ScalarCooperation,
    ) -> impl Future<Output = BuiltinResult<bool>> + 'a;
}
pub(in crate::builtins::math::optim::lsqlin) struct Supplied;
mod warm;
pub(in crate::builtins::math::optim::lsqlin) use warm::WarmSupplied;
impl Validation for Supplied {
    async fn stopped_candidate(
        &mut self,
        _: &[f64],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn objective_limit(
        &mut self,
        _: &[f64],
        cost: f64,
        limit: f64,
        _: f64,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>> {
        Ok(Some(cost < limit))
    }

    async fn solution(
        &mut self,
        _: &[f64],
        _: Rows<'_>,
        _: &[f64],
        _: f64,
        _: f64,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn incumbent(
        &mut self,
        _: &[f64],
        _: f64,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn certificate(
        &mut self,
        _: Certificate<'_>,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
}
pub(in crate::builtins::math::optim::lsqlin) struct Terminal<'a, V> {
    pub model: &'a model::Model,
    pub validation: &'a mut V,
}
impl<V: Validation> conic::solve::TerminalValidation for Terminal<'_, V> {
    async fn farkas(
        &mut self,
        z: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        use num_traits::Zero;
        if z.len() != self.model.b.len() {
            return Ok(false);
        }
        // Residual equalities introduce no original feasibility restriction.
        // Their multipliers must vanish exactly before original row projection.
        for value in &z[..self.model.c.m] {
            if !value.is_zero() {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        self.validation
            .certificate(
                Certificate::Farkas {
                    rows: Rows::new(self.model),
                    multipliers: z,
                },
                c,
            )
            .await
    }
    async fn ray(&mut self, _: &[BigRational], _: &mut ScalarCooperation) -> BuiltinResult<bool> {
        Ok(false)
    }
    async fn feasible(
        &mut self,
        _: &[BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(false)
    }
}
