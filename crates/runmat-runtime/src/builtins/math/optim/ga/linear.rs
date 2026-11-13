//! Sequential ownership of original sparse rows across creation and repair.
use super::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
mod rows;
mod geometry;
pub(in crate::builtins::math::optim) use super::numeric::linear_tolerance;
pub(in crate::builtins::math::optim) use geometry::Geometry;
#[cfg(test)]
mod tests;
pub(super) mod step;
use super::super::intlinprog::feasibility as mip;

/// One original model plus coefficient-free row access for feasible genetic
/// directions. The metadata cannot be swapped onto a different model by a
/// user callback or by public state editing.
pub(super) struct Domain {
    model: mip::Domain,
    rows: rows::Rows,
}
impl Domain {
    pub async fn new(
        model: mip::Domain,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let rows = rows::Rows::new(model.original_linear(), cooperation).await?;
        Ok(Self { model, rows })
    }
}
impl Geometry for Domain {
    fn bounds(&self) -> (&[f64], &[f64]) {
        self.model.bounds()
    }
    fn integers(&self) -> &[usize] {
        self.model.integers()
    }
    fn rhs(&self) -> &[f64] {
        self.model.original_linear().rhs()
    }
    fn equalities(&self) -> usize {
        self.model.original_linear().equalities()
    }
    fn repair_nodes(&self) -> usize {
        self.model.repair_nodes()
    }
    fn retune(
        &mut self,
        maximum: super::common::global_time::TimeLimit,
        tolerance: f64,
    ) -> BuiltinResult<()> {
        self.model
            .retune(maximum, super::numeric::linear_tolerance(tolerance))
    }
    async fn repair(
        &mut self,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<mip::Repair> {
        self.model.repair(point, cooperation).await
    }
    async fn classify(
        &mut self,
        point: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<mip::Membership> {
        self.model
            .classify(
                point,
                super::numeric::linear_tolerance(tolerance),
                cooperation,
            )
            .await
    }
    fn row_scratch(&self, cooperation: &ScalarCooperation) -> BuiltinResult<Vec<(usize, f64)>> {
        self.rows.scratch(cooperation)
    }
    async fn original_row(
        &self,
        row: usize,
        scratch: &mut Vec<(usize, f64)>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        self.rows
            .fill(self.model.original_linear(), row, scratch, cooperation)
            .await
    }
}

pub(super) struct Data {
    pub inequality: (CscMatrix<f64>, Vec<f64>),
    pub equality: (CscMatrix<f64>, Vec<f64>),
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub integers: Vec<usize>,
}
impl Data {
    pub async fn into_repair(
        self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<super::super::intlinprog::feasibility::Input> {
        super::super::intlinprog::feasibility::Input::from_pairs(
            self.inequality,
            self.equality,
            self.lower,
            self.upper,
            self.integers,
            cooperation,
        )
        .await
    }
    pub fn into_nonlinear_creation(
        self,
    ) -> (super::super::fmincon::feasibility::Input, Vec<usize>) {
        (
            super::super::fmincon::feasibility::Input {
                inequality: self.inequality,
                equality: self.equality,
                lower: self.lower,
                upper: self.upper,
            },
            self.integers,
        )
    }
    pub fn after_nonlinear_creation(
        input: super::super::fmincon::feasibility::Input,
        integers: Vec<usize>,
    ) -> Self {
        let super::super::fmincon::feasibility::Input {
            inequality,
            equality,
            lower,
            upper,
        } = input;
        Self {
            inequality,
            equality,
            lower,
            upper,
            integers,
        }
    }
}
