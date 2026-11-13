//! One retained sparse QR/COD projector; all feasibility and multiplier
//! claims are rechecked against the original supplied equality rows.
use super::super::model::LinearConstraints;
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::equality_factor::EqualityFactor;
use trust_quadratic::Operator;
pub(super) struct Equality<'a> {
    pub original: &'a LinearConstraints,
    pub factor: EqualityFactor,
    pub free: usize,
}
impl<'a> Equality<'a> {
    pub(super) async fn new(
        original: &'a LinearConstraints,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let a = &original.matrix;
        let Some(factor) = EqualityFactor::factor(
            a.m,
            a.n,
            &a.colptr,
            &a.rowval,
            &a.nzval,
            cooperation.runtime(),
            IDENTITY,
        )
        .await?
        else {
            return Ok(None);
        };
        let free = a.n.saturating_sub(factor.rank());
        Ok(Some(Self {
            original,
            factor,
            free,
        }))
    }
    async fn product(
        &self,
        x: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        for value in output.iter_mut() {
            *value = 0.0;
            cooperation.completed(1).await?;
        }
        let a = &self.original.matrix;
        for (column, &x) in x.iter().enumerate() {
            for index in a.colptr[column]..a.colptr[column + 1] {
                let slot = &mut output[a.rowval[index]];
                *slot += a.nzval[index] * x;
                if !slot.is_finite() {
                    return Ok(None);
                }
                cooperation.completed(1).await?;
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(()))
    }
    pub(super) async fn feasible(
        &self,
        x: &[f64],
        tolerance: f64,
        rhs: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if self.product(x, rhs, cooperation).await?.is_none() {
            return Ok(false);
        }
        for (value, &b) in rhs.iter_mut().zip(&self.original.rhs) {
            *value -= b;
            if !value.is_finite() || value.abs() > tolerance {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        Ok(true)
    }
    pub(super) async fn correct(
        &self,
        x: &mut [f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let mut rhs = zeros(self.original.matrix.m, cooperation).await?;
        let mut correction = zeros(x.len(), cooperation).await?;
        // Iterative refinement reuses the SAME factor, never a restarted
        // optimizer. Every acceptance uses the original absolute row tolerance.
        for _ in 0..2 {
            if self.product(x, &mut rhs, cooperation).await?.is_none() {
                return Ok(false);
            }
            let mut feasible = true;
            for (value, &b) in rhs.iter_mut().zip(&self.original.rhs) {
                *value -= b;
                if !value.is_finite() {
                    return Ok(false);
                }
                feasible &= value.abs() <= tolerance;
                cooperation.completed(1).await?;
            }
            if feasible {
                return Ok(true);
            }
            if self
                .factor
                .solve_into(&rhs, false, &mut correction)
                .await?
                .is_none()
            {
                return Ok(false);
            }
            let mut moved = false;
            for (x, &delta) in x.iter_mut().zip(&correction) {
                let next = *x - delta;
                if !next.is_finite() {
                    return Ok(false);
                }
                moved |= next != *x;
                *x = next;
                cooperation.completed(1).await?;
            }
            if !moved {
                break;
            }
        }
        self.feasible(x, tolerance, &mut rhs, cooperation).await
    }
    async fn project(
        &self,
        values: &mut [f64],
        rhs: &mut [f64],
        correction: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        if self.product(values, rhs, cooperation).await?.is_none() {
            return Ok(None);
        }
        if self
            .factor
            .solve_into(rhs, false, correction)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        for (value, &subtract) in values.iter_mut().zip(correction.iter()) {
            *value -= subtract;
            if !value.is_finite() {
                return Ok(None);
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(()))
    }
    pub(super) async fn gradient(
        &self,
        original: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        let mut values = copy(original, cooperation).await?;
        let mut rhs = zeros(self.original.matrix.m, cooperation).await?;
        let mut correction = zeros(values.len(), cooperation).await?;
        Ok(self
            .project(&mut values, &mut rhs, &mut correction, cooperation)
            .await?
            .map(|()| values))
    }
    pub(super) async fn multipliers(
        &self,
        gradient: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        let mut rhs = workspace::capacity(gradient.len(), IDENTITY)?;
        for &value in gradient {
            rhs.push(-value);
            cooperation.completed(1).await?;
        }
        self.factor.solve(&rhs, true).await
    }
}
pub(super) struct Projected<'a, 'matrix, 'model> {
    pub base: &'a mut model::Products<'model>,
    pub equality: &'a Equality<'matrix>,
    pub input: Vec<f64>,
    pub rhs: Vec<f64>,
    pub correction: Vec<f64>,
}
impl Operator for Projected<'_, '_, '_> {
    fn policy(&self) -> trust_quadratic::Policy {
        model::policy()
    }
    async fn apply(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        for (to, &from) in self.input.iter_mut().zip(input) {
            *to = from;
            cooperation.completed(1).await?;
        }
        if self
            .equality
            .project(
                &mut self.input,
                &mut self.rhs,
                &mut self.correction,
                cooperation,
            )
            .await?
            .is_none()
        {
            return Ok(None);
        }
        if self
            .base
            .apply_numerical(&self.input, output, cooperation)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        self.equality
            .project(output, &mut self.rhs, &mut self.correction, cooperation)
            .await
    }
    async fn project(
        &mut self,
        values: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        self.equality
            .project(values, &mut self.rhs, &mut self.correction, cooperation)
            .await
    }
}
