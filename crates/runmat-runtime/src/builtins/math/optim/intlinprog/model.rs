//! One mutable node workspace; the tree stores only parent links and bound deltas.
use super::*;
use clarabel::{algebra::CscMatrix, solver::SupportedConeT};
use common::conic;

pub(super) struct Problem {
    pub a: CscMatrix<f64>,
    pub b: Vec<f64>,
    pub f: Vec<f64>,
    pub equalities: usize,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub integers: Vec<usize>,
    pub semis: Vec<domains::Semi>,
    pub initial: Option<Vec<f64>>,
}
#[derive(Clone, Copy)]
pub(super) enum Bound {
    Lower { column: usize, value: f64 },
    Upper { column: usize, value: f64 },
}
pub(super) struct Workspace {
    a: CscMatrix<f64>,
    b: Vec<f64>,
    f: Vec<f64>,
    cones: Vec<SupportedConeT<f64>>,
    lower_entries: Vec<usize>,
    base_lower: Vec<f64>,
    base_upper: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    original_rows: usize,
    scale: f64,
    pub integers: Vec<usize>,
    pub semis: Vec<domains::Semi>,
    pub initial: Option<Vec<f64>>,
}
impl Workspace {
    pub async fn new(problem: Problem, cooperation: &mut ScalarCooperation) -> BuiltinResult<Self> {
        let Problem {
            a,
            b,
            f,
            equalities,
            lower,
            upper,
            integers,
            semis,
            initial,
        } = problem;
        let n = f.len();
        if lower.len() != n
            || upper.len() != n
            || a.n != n
            || a.m != b.len()
            || equalities > b.len()
            || a.colptr.len()
                != n.checked_add(1)
                    .ok_or_else(|| invalid("column count overflow"))?
            || a.rowval.len() != a.nzval.len()
            || a.colptr.first() != Some(&0)
            || a.colptr.last() != Some(&a.nzval.len())
            || integers.iter().any(|&j| j >= n)
            || semis.iter().any(|domain| domain.column >= n)
            || initial.as_ref().is_some_and(|v| v.len() != n)
        {
            return Err(invalid("inconsistent original linear model geometry"));
        }
        let original_rows = b.len();
        let extra = n
            .checked_mul(2)
            .ok_or_else(|| invalid("bound row count overflow"))?;
        let rows = original_rows
            .checked_add(extra)
            .ok_or_else(|| invalid("row count overflow"))?;
        let nz = a
            .nzval
            .len()
            .checked_add(extra)
            .ok_or_else(|| invalid("sparse size overflow"))?;
        let mut colptr = runmat_value::exact_capacity_vec(n + 1).map_err(allocation)?;
        let mut rowval = runmat_value::exact_capacity_vec(nz).map_err(allocation)?;
        let mut nzval = runmat_value::exact_capacity_vec(nz).map_err(allocation)?;
        let mut lower_entries = runmat_value::exact_capacity_vec(n).map_err(allocation)?;
        colptr.push(0);
        for j in 0..n {
            let first = a.colptr[j];
            let end = a.colptr[j + 1];
            if first > end || end > a.nzval.len() {
                return Err(invalid("invalid CSC pointers"));
            }
            let mut previous = None;
            for k in first..end {
                let row = a.rowval[k];
                let value = a.nzval[k];
                if row >= original_rows || previous.is_some_and(|p| p >= row) || !value.is_finite()
                {
                    return Err(invalid("constraint matrix must be finite canonical CSC"));
                }
                previous = Some(row);
                rowval.push(row);
                nzval.push(value);
                if k % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
            lower_entries.push(nzval.len());
            rowval.push(original_rows + 2 * j);
            nzval.push(0.0);
            rowval.push(original_rows + 2 * j + 1);
            nzval.push(0.0);
            colptr.push(nzval.len());
            cooperation.completed(1).await?;
        }
        let mut rhs = runmat_value::exact_capacity_vec(rows).map_err(allocation)?;
        for chunk in b.chunks(1024) {
            if chunk.iter().any(|v| !v.is_finite()) {
                return Err(invalid("constraint right sides must be finite"));
            }
            rhs.extend_from_slice(chunk);
            cooperation.completed(chunk.len() as u64).await?;
        }
        rhs.resize(rows, 0.0);
        let mut scale = 0.0_f64;
        for chunk in f.chunks(1024) {
            for &v in chunk {
                if !v.is_finite() {
                    return Err(invalid("objective must be finite"));
                }
                scale = scale.max(v.abs());
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        for j in 0..n {
            if lower[j].is_nan() || upper[j].is_nan() {
                return Err(invalid("variable bounds cannot be NaN"));
            }
            cooperation.completed(1).await?;
        }
        let mut active_lower = runmat_value::exact_capacity_vec(n).map_err(allocation)?;
        active_lower.extend_from_slice(&lower);
        let mut active_upper = runmat_value::exact_capacity_vec(n).map_err(allocation)?;
        active_upper.extend_from_slice(&upper);
        let mut cones = Vec::new();
        if equalities > 0 {
            cones.push(SupportedConeT::ZeroConeT(equalities));
        }
        if rows > equalities {
            cones.push(SupportedConeT::NonnegativeConeT(rows - equalities));
        }
        let mut result = Self {
            a: CscMatrix::new(rows, n, colptr, rowval, nzval),
            b: rhs,
            f,
            cones,
            lower_entries,
            base_lower: lower,
            base_upper: upper,
            lower: active_lower,
            upper: active_upper,
            original_rows,
            scale: if scale == 0.0 { 1.0 } else { scale },
            integers,
            semis,
            initial,
        };
        result.publish_bounds(cooperation).await?;
        Ok(result)
    }
    pub fn view(&self) -> conic::Model<'_> {
        conic::Model {
            p: None,
            a: &self.a,
            b: &self.b,
            f: &self.f,
            cones: &self.cones,
            identity: runmat_builtins::INTLINPROG_CATALOG_ENTRY.identity,
            objective_scale: self.scale,
        }
    }
    pub(super) fn node(&self) -> compiled::Node<'_> {
        compiled::Node {
            model: self.view(),
            original_rows: self.original_rows,
            equalities: match self.cones.first() {
                Some(SupportedConeT::ZeroConeT(count)) => *count,
                _ => 0,
            },
            lower: &self.lower,
            upper: &self.upper,
        }
    }
    pub fn n(&self) -> usize {
        self.f.len()
    }
    pub fn is_mixed(&self) -> bool {
        !self.integers.is_empty() || !self.semis.is_empty()
    }
    pub(super) fn original_bounds(&self) -> (&[f64], &[f64]) {
        (&self.base_lower, &self.base_upper)
    }
    /// Retarget only the final paired absolute-distance rows in an internal
    /// feasibility lift. Original constraint rows, integer domains and bounds
    /// remain unchanged. Every repair obtains fresh certificates for this RHS.
    pub(super) async fn retarget_distance(
        &mut self,
        first_row: usize,
        target: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if target.len().checked_mul(2) != Some(self.n())
            || first_row.checked_add(self.n()) != Some(self.original_rows)
            || target.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid internal distance-repair geometry"));
        }
        for (column, &value) in target.iter().enumerate() {
            self.b[first_row + 2 * column] = value;
            self.b[first_row + 2 * column + 1] = -value;
            cooperation.completed(1).await?;
        }
        Ok(())
    }

    pub async fn reset(&mut self, cooperation: &mut ScalarCooperation) -> BuiltinResult<()> {
        for j in 0..self.n() {
            self.lower[j] = self.base_lower[j];
            self.upper[j] = self.base_upper[j];
            cooperation.completed(1).await?;
        }
        Ok(())
    }
    pub fn tighten(&mut self, bound: Bound) {
        match bound {
            Bound::Lower { column, value } => self.lower[column] = self.lower[column].max(value),
            Bound::Upper { column, value } => self.upper[column] = self.upper[column].min(value),
        }
    }
    pub fn consistent(&self) -> bool {
        self.lower
            .iter()
            .zip(&self.upper)
            .all(|(l, u)| l <= u && *l != f64::INFINITY && *u != f64::NEG_INFINITY)
    }
    /// Both children must tighten the current node. An out-of-bounds backend
    /// proposal cannot justify repeatedly adding an already-present bound.
    pub fn branch_progress(&self, column: usize, floor: f64, ceil: f64) -> bool {
        floor.is_finite()
            && ceil.is_finite()
            && floor < ceil
            && self.lower[column] < ceil
            && floor < self.upper[column]
    }
    pub async fn publish_bounds(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        for j in 0..self.n() {
            let entry = self.lower_entries[j];
            let row = self.original_rows + 2 * j;
            if self.lower[j] > self.upper[j]
                || self.lower[j] == f64::INFINITY
                || self.upper[j] == f64::NEG_INFINITY
            {
                self.a.nzval[entry] = 0.0;
                self.b[row] = -1.0;
                self.a.nzval[entry + 1] = 0.0;
                self.b[row + 1] = 0.0;
                cooperation.completed(1).await?;
                continue;
            }
            self.a.nzval[entry] = if self.lower[j].is_finite() { -1.0 } else { 0.0 };
            self.b[row] = if self.lower[j].is_finite() {
                -self.lower[j]
            } else {
                0.0
            };
            self.a.nzval[entry + 1] = if self.upper[j].is_finite() { 1.0 } else { 0.0 };
            self.b[row + 1] = if self.upper[j].is_finite() {
                self.upper[j]
            } else {
                0.0
            };
            cooperation.completed(1).await?;
        }
        Ok(())
    }
    pub async fn seed(&self, cooperation: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
        let mut result = runmat_value::exact_capacity_vec(self.n()).map_err(allocation)?;
        for j in 0..self.n() {
            result.push(0.0_f64.max(self.lower[j]).min(self.upper[j]));
            cooperation.completed(1).await?;
        }
        Ok(result)
    }
}
