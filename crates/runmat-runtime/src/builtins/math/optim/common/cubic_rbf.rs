//! Cubic radial-basis interpolation with a linear polynomial tail.
//! Training coordinates/measurements stay borrowed from the single trial
//! history. Only coefficients and row-index metadata outlive a fit.
use crate::builtins::math::linalg::ops::dense_factor::system::{
    NumericalFailure, RealSquare, SystemError,
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::execution::ExecutionServiceError;
use crate::BuiltinResult;
use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};
use std::rc::Rc;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Policy {
    pub identity: BuiltinCatalogIdentity,
    pub invalid: &'static BuiltinErrorDescriptor,
}
impl Policy {
    fn invalid(self, message: impl std::fmt::Display) -> crate::RuntimeError {
        let mut builder = crate::build_runtime_error(format!("{}: {message}", self.identity.name))
            .with_builtin(self.identity.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
        if let Some(identifier) = self.invalid.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

/// One marker belongs to one canonical append-only training history.
/// Cloning it retains identity metadata, never coordinates or measurements.
#[derive(Clone)]
pub(in crate::builtins::math::optim) struct TrainingOwner(Rc<()>);
impl TrainingOwner {
    pub(in crate::builtins::math::optim) fn new() -> Self {
        Self(Rc::new(()))
    }
    pub(in crate::builtins::math::optim) fn same_as(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::optim) struct Generation(pub(in crate::builtins::math::optim) u64);

#[cfg(test)]
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Data<'a> {
    /// Complete row-major trial buffers; rejected trials are not in rows.
    pub owner: &'a TrainingOwner,
    pub points: &'a [f64],
    pub values: &'a [f64],
    pub rows: &'a [usize],
    pub dimensions: usize,
    pub lanes: usize,
    pub generation: Generation,
}
#[cfg(test)]
impl Data<'_> {
    fn valid(self) -> bool {
        if self.dimensions == 0 || !self.points.len().is_multiple_of(self.dimensions) {
            return false;
        }
        let trials = self.points.len() / self.dimensions;
        trials.checked_mul(self.lanes) == Some(self.values.len())
            && self.rows.iter().all(|&row| row < trials)
    }
    fn coordinate(self, row: usize, axis: usize) -> f64 {
        self.points[self.rows[row] * self.dimensions + axis]
    }
    fn measurement(self, row: usize, lane: usize) -> f64 {
        self.values[self.rows[row] * self.lanes + lane]
    }
}

/// A numerical view over the single canonical trial history. Domain mapping
/// and selected-row membership are validated by that history owner; no second
/// population or normalized-coordinate payload is required to fit a model.
pub(in crate::builtins::math::optim) trait Training {
    /// Constant-time validity over metadata checked cooperatively when the
    /// caller created this borrowed view; never acquire a provider here.
    fn valid(&self) -> bool;
    fn owner(&self) -> &TrainingOwner;
    fn generation(&self) -> Generation;
    fn rows(&self) -> &[usize];
    fn dimensions(&self) -> usize;
    fn lanes(&self) -> usize;
    /// Independent linear-tail coordinates; None selects all physical axes.
    /// Some axes must be strictly increasing. Radial distances still use
    /// every physical dimension, including dependent equality coordinates.
    fn affine_axes(&self) -> Option<&[usize]> {
        None
    }
    fn coordinate(&self, selected_row: usize, axis: usize) -> f64;
    fn measurement(&self, selected_row: usize, lane: usize) -> f64;
}
#[cfg(test)]
impl Training for Data<'_> {
    fn owner(&self) -> &TrainingOwner {
        self.owner
    }
    fn valid(&self) -> bool {
        Data::valid(*self)
    }
    fn generation(&self) -> Generation {
        self.generation
    }
    fn rows(&self) -> &[usize] {
        self.rows
    }
    fn dimensions(&self) -> usize {
        self.dimensions
    }
    fn lanes(&self) -> usize {
        self.lanes
    }
    fn coordinate(&self, row: usize, axis: usize) -> f64 {
        Data::coordinate(*self, row, axis)
    }
    fn measurement(&self, row: usize, lane: usize) -> f64 {
        Data::measurement(*self, row, lane)
    }
}

pub(in crate::builtins::math::optim) enum Fit {
    Poised(Model),
    Unpoised,
    Undefined,
    NoFunctions,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::optim) enum Prediction {
    Finite,
    Undefined,
}
pub(in crate::builtins::math::optim) struct Model {
    policy: Policy,
    owner: TrainingOwner,
    generation: Generation,
    rows: Vec<usize>,
    dimensions: usize,
    lanes: usize,
    affine_axes: Option<Vec<usize>>,
    /// One column-major coefficient vector per objective/constraint lane.
    coefficients: Vec<f64>,
}

async fn tick(cooperation: &mut ScalarCooperation) -> Result<(), ExecutionServiceError> {
    if let (_, Some(budget)) = cooperation.runtime_budget() {
        budget.checkpoint(1).await?;
    }
    Ok(())
}
async fn distance(
    dimensions: usize,
    mut pair: impl FnMut(usize) -> (f64, f64),
    cooperation: &mut ScalarCooperation,
) -> Result<f64, ExecutionServiceError> {
    let mut norm = 0.0_f64;
    for axis in 0..dimensions {
        let (left, right) = pair(axis);
        norm = norm.hypot(left - right);
        tick(cooperation).await?;
    }
    Ok(norm)
}
fn cubic(distance: f64) -> f64 {
    distance * distance * distance
}
async fn buffer(
    policy: Policy,
    length: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|source| {
        crate::build_runtime_error(format!(
            "{}: model workspace allocation failed",
            policy.identity.name
        ))
        .with_builtin(policy.identity.name)
        .with_source(source)
        .build()
    })?;
    while values.len() < length {
        let next = length.min(values.len().saturating_add(1024));
        let work = next - values.len();
        values.resize(next, 0.0);
        cooperation.completed(work as u64).await?;
    }
    Ok(values)
}
fn failed(
    policy: Policy,
    error: SystemError,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<Fit> {
    match error.numerical() {
        Some(NumericalFailure::Singular) => Ok(Fit::Unpoised),
        Some(NumericalFailure::NonFinite) => Ok(Fit::Undefined),
        None => Err(error.into_runtime_error(policy.identity, cooperation.runtime())),
    }
}

/// The full saddle system [Phi P; P' 0] is factored directly. A rank-deficient
/// affine tail or duplicate site returns Unpoised; it never becomes a zero
/// model, a normal-equation approximation, or an invented success.
pub(in crate::builtins::math::optim) async fn fit(
    policy: Policy,
    data: &impl Training,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Fit> {
    cooperation.completed(0).await?;
    if !data.valid() {
        return Err(policy.invalid("invalid radial-basis training geometry"));
    }
    if data.lanes() == 0 {
        return Ok(Fit::NoFunctions);
    }
    let mut previous = None;
    if let Some(axes) = data.affine_axes() {
        for &axis in axes {
            if axis >= data.dimensions() || previous.is_some_and(|before| before >= axis) {
                return Err(policy.invalid("invalid independent radial-basis tail axes"));
            }
            previous = Some(axis);
            cooperation.completed(1).await?;
        }
    }
    let affine_dimensions = data.affine_axes().map_or(data.dimensions(), <[usize]>::len);
    let tail = affine_dimensions
        .checked_add(1)
        .ok_or_else(|| policy.invalid("radial-basis affine extent overflow"))?;
    let n = data.rows().len();
    if n < tail {
        return Ok(Fit::Unpoised);
    }
    let order = n
        .checked_add(tail)
        .ok_or_else(|| policy.invalid("radial-basis factor extent overflow"))?;
    let coefficient_count = order
        .checked_mul(data.lanes())
        .ok_or_else(|| policy.invalid("radial-basis coefficient extent overflow"))?;
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        policy.identity.name,
        "finite host-memory execution has no complete radial-basis model resource bound",
    )?;
    for row in 0..n {
        for axis in 0..data.dimensions() {
            if !data.coordinate(row, axis).is_finite() {
                return Ok(Fit::Undefined);
            }
            cooperation.completed(1).await?;
        }
        for lane in 0..data.lanes() {
            if !data.measurement(row, lane).is_finite() {
                return Ok(Fit::Undefined);
            }
            cooperation.completed(1).await?;
        }
    }
    let runtime = cooperation.runtime().cloned();
    let factor = RealSquare::factor(order, runtime.as_ref(), async |row, column| {
        if row < n && column < n {
            return Ok(cubic(
                distance(
                    data.dimensions(),
                    |axis| (data.coordinate(row, axis), data.coordinate(column, axis)),
                    cooperation,
                )
                .await?,
            ));
        }
        if row >= n && column >= n {
            return Ok(0.0);
        }
        let (site, affine) = if row < n {
            (row, column - n)
        } else {
            (column, row - n)
        };
        Ok(if affine == 0 {
            1.0
        } else {
            let axis = data
                .affine_axes()
                .map_or(affine - 1, |axes| axes[affine - 1]);
            data.coordinate(site, axis)
        })
    })
    .await;
    let mut factor = match factor {
        Ok(factor) => factor,
        Err(error) => return failed(policy, error, cooperation),
    };
    let mut rhs = buffer(policy, order, cooperation).await?;
    let mut coefficients = buffer(policy, coefficient_count, cooperation).await?;
    for lane in 0..data.lanes() {
        for (row, target) in rhs[..n].iter_mut().enumerate() {
            *target = data.measurement(row, lane);
            cooperation.completed(1).await?;
        }
        let column = &mut coefficients[lane * order..(lane + 1) * order];
        if let Err(error) = factor.solve_into(&rhs, column, runtime.as_ref()).await {
            return failed(policy, error, cooperation);
        }
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(n).map_err(|source| {
        crate::build_runtime_error(format!(
            "{}: model row-index allocation failed",
            policy.identity.name
        ))
        .with_builtin(policy.identity.name)
        .with_source(source)
        .build()
    })?;
    for chunk in data.rows().chunks(1024) {
        rows.extend_from_slice(chunk);
        cooperation.completed(chunk.len() as u64).await?;
    }
    let affine_axes = if let Some(axes) = data.affine_axes() {
        let mut selected = runmat_value::exact_capacity_vec(axes.len()).map_err(|source| {
            crate::build_runtime_error(format!(
                "{}: tail-index allocation failed",
                policy.identity.name
            ))
            .with_builtin(policy.identity.name)
            .with_source(source)
            .build()
        })?;
        for &axis in axes {
            selected.push(axis);
            cooperation.completed(1).await?;
        }
        Some(selected)
    } else {
        None
    };
    Ok(Fit::Poised(Model {
        policy,
        owner: data.owner().clone(),
        generation: data.generation(),
        rows,
        dimensions: data.dimensions(),
        lanes: data.lanes(),
        affine_axes,
        coefficients,
    }))
}

fn add(total: &mut f64, correction: &mut f64, term: f64) -> bool {
    let next = *total + term;
    if !term.is_finite() || !next.is_finite() {
        return false;
    }
    *correction += if total.abs() >= term.abs() {
        (*total - next) + term
    } else {
        (term - next) + *total
    };
    *total = next;
    correction.is_finite()
}
impl Model {
    /// The caller's committed generation and exact owned row set are checked
    /// before prediction; pending append-only rows need not change that set.
    pub(in crate::builtins::math::optim) async fn predict_into(
        &self,
        data: &impl Training,
        point: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Prediction> {
        self.predict_lanes_into(data, point, 0..self.lanes, output, cooperation)
            .await
    }
    pub(in crate::builtins::math::optim) async fn predict_lanes_into(
        &self,
        data: &impl Training,
        point: &[f64],
        lanes: std::ops::Range<usize>,
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Prediction> {
        cooperation.completed(0).await?;
        let policy = self.policy;
        if !data.valid()
            || !data.owner().same_as(&self.owner)
            || data.generation() != self.generation
            || data.rows().len() != self.rows.len()
            || data.dimensions() != self.dimensions
            || data.lanes() != self.lanes
            || point.len() != self.dimensions
            || lanes.start > lanes.end
            || lanes.end > self.lanes
            || output.len() != lanes.len()
        {
            return Err(policy.invalid("stale or mismatched radial-basis model generation"));
        }
        for (&actual, &expected) in data.rows().iter().zip(&self.rows) {
            if actual != expected {
                return Err(policy.invalid("stale or mismatched radial-basis model rows"));
            }
            cooperation.completed(1).await?;
        }
        match (data.affine_axes(), self.affine_axes.as_deref()) {
            (None, None) => {}
            (Some(actual), Some(expected)) if actual.len() == expected.len() => {
                for (&actual, &expected) in actual.iter().zip(expected) {
                    if actual != expected {
                        return Err(policy.invalid("mismatched radial-basis affine tail"));
                    }
                    cooperation.completed(1).await?;
                }
            }
            _ => return Err(policy.invalid("mismatched radial-basis affine tail")),
        }
        if point.iter().any(|x| !x.is_finite()) {
            return Ok(Prediction::Undefined);
        }
        let n = self.rows.len();
        let affine_dimensions = self.affine_axes.as_ref().map_or(self.dimensions, Vec::len);
        let order = n + affine_dimensions + 1;
        for (lane, target) in lanes.zip(output.iter_mut()) {
            let coefficients = &self.coefficients[lane * order..(lane + 1) * order];
            let (mut total, mut correction) = (coefficients[n], 0.0);
            for affine in 0..affine_dimensions {
                let axis = self
                    .affine_axes
                    .as_ref()
                    .map_or(affine, |axes| axes[affine]);
                if !add(
                    &mut total,
                    &mut correction,
                    coefficients[n + affine + 1] * point[axis],
                ) {
                    return Ok(Prediction::Undefined);
                }
                cooperation.completed(1).await?;
            }
            for (row, &weight) in coefficients[..n].iter().enumerate() {
                let radius = distance(
                    self.dimensions,
                    |axis| (point[axis], data.coordinate(row, axis)),
                    cooperation,
                )
                .await
                .map_err(|source| {
                    crate::builtins::common::map_control_flow_with_builtin(
                        source.into_runtime_error_in_context(cooperation.runtime()),
                        policy.identity.name,
                    )
                })?;
                if !add(&mut total, &mut correction, weight * cubic(radius)) {
                    return Ok(Prediction::Undefined);
                }
            }
            *target = total + correction;
            if !target.is_finite() {
                return Ok(Prediction::Undefined);
            }
        }
        Ok(Prediction::Finite)
    }
}
