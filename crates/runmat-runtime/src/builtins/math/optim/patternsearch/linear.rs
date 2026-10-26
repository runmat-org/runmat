//! Original sparse linear geometry. Poll proposals are checked against these
//! rows, never merely against a rounded/projection surrogate.
use super::{buffer, cone::Cone, invalid, normal::Normal, push, scaling::Scaling};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::{algebra::CscMatrix, solver::SupportedConeT};

pub(super) struct Linear {
    pub matrix: CscMatrix<f64>,
    pub rhs: Vec<f64>,
    pub equalities: usize,
    pub cones: Vec<SupportedConeT<f64>>,
    // Sparse transpose METADATA only. Coefficients stay in the original CSC
    // backing. Row access for active normals does not densify the model.
    rows: Vec<Vec<(usize, usize)>>,
    widest: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProposalStop {
    Blocked,
    Unrepresentable,
    Numerical,
}
pub(super) struct Prefix {
    pub point: Vec<f64>,
    pub length: f64,
}

impl Linear {
    pub fn axis_aligned(&self) -> bool {
        self.rows.iter().all(|row| row.len() <= 1)
    }
    pub async fn new(
        matrix: CscMatrix<f64>,
        rhs: Vec<f64>,
        equalities: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if rhs.len() != matrix.m
            || equalities > rhs.len()
            || rhs.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("linear row geometry mismatch"));
        }
        let mut rows = buffer(matrix.m)?;
        for _ in 0..matrix.m {
            rows.push(Vec::new());
            cooperation.completed(1).await?;
        }
        for column in 0..matrix.n {
            for slot in matrix.colptr[column]..matrix.colptr[column + 1] {
                let row = matrix.rowval[slot];
                if row >= matrix.m || !matrix.nzval[slot].is_finite() {
                    return Err(invalid("invalid original linear coefficient"));
                }
                push(&mut rows[row], (column, slot))?;
                cooperation.completed(1).await?;
            }
        }
        let mut cones = buffer(2)?;
        if equalities > 0 {
            cones.push(SupportedConeT::ZeroConeT(equalities));
        }
        if rhs.len() > equalities {
            cones.push(SupportedConeT::NonnegativeConeT(rhs.len() - equalities));
        }
        let widest = rows.iter().map(Vec::len).max().unwrap_or(0);
        Ok(Self {
            matrix,
            rhs,
            equalities,
            cones,
            rows,
            widest,
        })
    }

    pub fn row_capacity(&self) -> usize {
        self.widest
    }
    /// Original sorted columns and coefficients, with one caller-owned row
    /// scratch. No row matrix or coefficient cache is created.
    pub async fn original_row(
        &self,
        row: usize,
        scratch: &mut Vec<(usize, f64)>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let entries = self
            .rows
            .get(row)
            .ok_or_else(|| invalid("original row is out of range"))?;
        if scratch.capacity() < entries.len() {
            return Err(invalid("original row scratch has insufficient capacity"));
        }
        scratch.clear();
        for &(column, slot) in entries {
            scratch.push((column, self.matrix.nzval[slot]));
            cooperation.completed(1).await?;
        }
        Ok(())
    }

    pub async fn activity(
        &self,
        x: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        if x.len() != self.matrix.n {
            return Err(invalid("linear point extent mismatch"));
        }
        let mut result = buffer(self.matrix.m)?;
        for row in &self.rows {
            let mut value = 0.0;
            for &(column, slot) in row {
                value += self.matrix.nzval[slot] * x[column];
                if !value.is_finite() {
                    return Ok(None);
                }
                cooperation.completed(1).await?;
            }
            result.push(value);
            cooperation.completed(1).await?;
        }
        Ok(Some(result))
    }

    pub async fn violation(
        &self,
        x: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>> {
        let Some(activity) = self.activity(x, cooperation).await? else {
            return Ok(None);
        };
        let mut maximum = 0.0f64;
        for (row, (&value, &rhs)) in activity.iter().zip(&self.rhs).enumerate() {
            let residual = value - rhs;
            if !residual.is_finite() {
                return Ok(None);
            }
            maximum = maximum.max(if row < self.equalities {
                residual.abs()
            } else {
                residual.max(0.0)
            });
            cooperation.completed(1).await?;
        }
        Ok(Some(maximum))
    }

    pub async fn feasible(
        &self,
        x: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let Some(activity) = self.activity(x, cooperation).await? else {
            return Ok(false);
        };
        for (row, (&value, &rhs)) in activity.iter().zip(&self.rhs).enumerate() {
            let residual = value - rhs;
            if !residual.is_finite()
                || if row < self.equalities {
                    residual.abs() > tolerance
                } else {
                    residual > tolerance
                }
            {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        Ok(true)
    }

    pub async fn cone(
        &self,
        x: &[f64],
        mesh: f64,
        binding: f64,
        scaling: &Scaling,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Cone>> {
        Ok(self
            .binding_cone(x, mesh, binding, scaling, cooperation)
            .await?
            .map(|(cone, _)| cone))
    }
    pub async fn binding_cone(
        &self,
        x: &[f64],
        mesh: f64,
        binding: f64,
        scaling: &Scaling,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<(Cone, bool)>> {
        if scaling.dimension() != self.matrix.n {
            return Err(invalid("linear mesh scale dimension mismatch"));
        }
        let Some(activity) = self.activity(x, cooperation).await? else {
            return Ok(None);
        };
        let mut cone = Cone::full(self.matrix.n, cooperation).await?;
        let mut constrained = false;
        let epsilon = binding.min(mesh);
        for (row, entries) in self.rows.iter().enumerate() {
            let mut normal = buffer(entries.len())?;
            for &(column, slot) in entries {
                let value = self.matrix.nzval[slot];
                normal.push((column, value));
                cooperation.completed(1).await?;
            }
            let equality = row < self.equalities;
            let normal = Normal::new(&normal, scaling, cooperation).await?;
            let Some(distance) = normal.distance(self.rhs[row] - activity[row]) else {
                return Ok(None);
            };
            let active = equality || distance <= epsilon;
            if active {
                constrained = true;
                cone.intersect_normal(&normal, false, cooperation).await?;
                if equality {
                    cone.intersect_normal(&normal, true, cooperation).await?;
                }
            }
            cooperation.completed(1).await?;
        }
        Ok(Some((cone, constrained)))
    }

    /// Independent affine-tail axes for the ORIGINAL equality subspace.
    /// Bounds fixed at one value add true coordinate equalities; ordinary
    /// active inequalities never reduce radial distances or polynomial rank.
    pub async fn rbf_axes(
        &self,
        lower: &[f64],
        upper: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<crate::builtins::math::optim::common::feasible_cone::IndependentAxes> {
        let n = self.matrix.n;
        if lower.len() != n || upper.len() != n {
            return Err(invalid("RBF bound geometry extent mismatch"));
        }
        if self.equalities == 0 {
            let mut axes = buffer(n)?;
            for (axis, (&l, &u)) in lower.iter().zip(upper).enumerate() {
                if l != u {
                    axes.push(axis);
                }
                cooperation.completed(1).await?;
            }
            return Ok(if axes.is_empty() {
                crate::builtins::math::optim::common::feasible_cone::IndependentAxes::Unique
            } else {
                crate::builtins::math::optim::common::feasible_cone::IndependentAxes::Axes(axes)
            });
        }
        let scaling = Scaling::unit(n);
        let mut cone = Cone::full(n, cooperation).await?;
        for entries in self.rows.iter().take(self.equalities) {
            let mut row = buffer(entries.len())?;
            for &(column, slot) in entries {
                row.push((column, self.matrix.nzval[slot]));
                cooperation.completed(1).await?;
            }
            let normal = Normal::new(&row, &scaling, cooperation).await?;
            cone.intersect_normal(&normal, false, cooperation).await?;
            cone.intersect_normal(&normal, true, cooperation).await?;
        }
        for (axis, (&l, &u)) in lower.iter().zip(upper).enumerate() {
            if l == u && l.is_finite() {
                let normal = Normal::new(&[(axis, 1.0)], &scaling, cooperation).await?;
                cone.intersect_normal(&normal, false, cooperation).await?;
                cone.intersect_normal(&normal, true, cooperation).await?;
            }
            cooperation.completed(1).await?;
        }
        cone.independent_axes(cooperation).await
    }

    /// Largest numerically feasible prefix of a proposed feasible-cone ray.
    /// Original equalities and every inactive inequality are still checked at
    /// the actual floating candidate, so conversion cannot authorize drift.
    /// The prefix length describes the SAME checked FMA point, not a second
    /// projection or an approximate direction-identity claim.
    pub async fn candidate_with_step(
        &self,
        x: &[f64],
        ray: &[f64],
        mesh: f64,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Result<Prefix, ProposalStop>> {
        let Some(activity) = self.activity(x, cooperation).await? else {
            return Ok(Err(ProposalStop::Numerical));
        };
        let Some(products) = self.activity(ray, cooperation).await? else {
            return Ok(Err(ProposalStop::Numerical));
        };
        let mut length = mesh;
        for row in self.equalities..self.rhs.len() {
            if products[row] > 0.0 {
                let slack = self.rhs[row] - activity[row];
                if !slack.is_finite() {
                    return Ok(Err(ProposalStop::Numerical));
                }
                length = length.min(slack.max(0.0) / products[row]);
            }
            cooperation.completed(1).await?;
        }
        if !length.is_finite() {
            return Ok(Err(ProposalStop::Numerical));
        }
        if length <= 0.0 {
            return Ok(Err(ProposalStop::Blocked));
        }
        let mut candidate = buffer(x.len())?;
        let mut changed = false;
        for (&origin, &direction) in x.iter().zip(ray) {
            let value = length.mul_add(direction, origin);
            if !value.is_finite() {
                return Ok(Err(ProposalStop::Numerical));
            }
            changed |= value != origin;
            candidate.push(value);
            cooperation.completed(1).await?;
        }
        if !changed {
            return Ok(Err(ProposalStop::Unrepresentable));
        }
        if !self.feasible(&candidate, tolerance, cooperation).await? {
            return Ok(Err(ProposalStop::Numerical));
        }
        Ok(Ok(Prefix {
            point: candidate,
            length,
        }))
    }

    /// A lattice method tests the whole mesh point. It must not continuously
    /// shorten its ray and silently leave the selected integer mesh.
    pub async fn mesh_candidate(
        &self,
        x: &[f64],
        ray: &[f64],
        mesh: f64,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Result<Prefix, ProposalStop>> {
        if x.len() != self.matrix.n || ray.len() != x.len() {
            return Err(invalid("mesh point/ray extent mismatch"));
        }
        let mut point = buffer(x.len())?;
        let mut changed = false;
        for (&origin, &direction) in x.iter().zip(ray) {
            let value = mesh.mul_add(direction, origin);
            if !value.is_finite() {
                return Ok(Err(ProposalStop::Numerical));
            }
            changed |= value != origin;
            point.push(value);
            cooperation.completed(1).await?;
        }
        if !changed {
            return Ok(Err(ProposalStop::Unrepresentable));
        }
        let Some(violation) = self.violation(&point, cooperation).await? else {
            return Ok(Err(ProposalStop::Numerical));
        };
        if violation > tolerance {
            return Ok(Err(ProposalStop::Blocked));
        }
        Ok(Ok(Prefix {
            point,
            length: mesh,
        }))
    }
}
