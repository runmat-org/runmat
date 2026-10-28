//! Rigorous lower bounds for original linear zero/nonnegative-cone models.
//! Numerical duals are proposals; weak duality and residual corrections are exact.
use super::super::ScalarCooperation;
use super::{dimension, terminal, Model};
use crate::{BuiltinResult, RuntimeError};
use clarabel::solver::SupportedConeT;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
#[cfg(test)]
mod tests;
mod contradiction;
use contradiction::Contradiction;

#[derive(Clone)]
pub(crate) struct ExactLowerBound(BigRational);
impl ExactLowerBound {
    /// Internal original-source proof owners issue this only after exact
    /// weak-duality/residual correction. Numerical proposals cannot issue it.
    pub(in crate::builtins::math::optim) fn from_verified_original(value: BigRational) -> Self {
        Self(value)
    }
    pub(crate) fn value(&self) -> &BigRational {
        &self.0
    }
}
#[derive(Default)]
struct Interval {
    lower: Option<BigRational>,
    upper: Option<BigRational>,
    lower_entry: Option<usize>,
    upper_entry: Option<usize>,
}
#[derive(Clone, Copy)]
enum Singleton {
    Empty,
    One { column: usize, entry: usize },
    Many,
}
/// One immutable node model; singleton bounds/cone roles are derived once and
/// reused for every recovery proposal. No second matrix/row authority is accepted.
pub(crate) struct LinearCertificates<'view, 'data> {
    model: &'view Model<'data>,
    equalities: Vec<bool>,
    bounds: Vec<Interval>,
    consistent: bool,
    contradiction: Option<Contradiction>,
}
fn allocation(
    source: runmat_value::ExactHostNumericAllocationError,
    model: &Model<'_>,
) -> RuntimeError {
    crate::build_runtime_error(format!(
        "{}: exact linear certificate allocation failed",
        model.identity.name
    ))
    .with_builtin(model.identity.name)
    .with_source(source)
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}
impl<'view, 'data> LinearCertificates<'view, 'data> {
    pub(crate) async fn borrow(
        model: &'view Model<'data>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        crate::context::host_allocation::require_unbounded_workspace(cooperation.runtime(),model.identity.name,
            "finite host-memory execution is unsupported for exact linear certificate and rational-growth workspace")?;
        if model.p.is_some() || !model.valid_layout() {
            return Ok(None);
        }
        let n = model.f.len();
        let m = model.b.len();
        let a = model.a;
        if n.checked_add(1) != Some(a.colptr.len())
            || a.rowval.len() != a.nzval.len()
            || a.colptr.first() != Some(&0)
            || a.colptr.last() != Some(&a.nzval.len())
        {
            return Ok(None);
        }
        let mut equalities =
            runmat_value::exact_capacity_vec(m).map_err(|e| allocation(e, model))?;
        for cone in model.cones {
            let (count, equality) = match cone {
                SupportedConeT::ZeroConeT(count) => (*count, true),
                SupportedConeT::NonnegativeConeT(count) => (*count, false),
                _ => return Ok(None),
            };
            equalities.extend(std::iter::repeat_n(equality, count));
            cooperation.completed(count as u64).await?;
        }
        for chunk in model.f.chunks(1024).chain(model.b.chunks(1024)) {
            if chunk.iter().any(|v| !v.is_finite()) {
                return Ok(None);
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        let mut singletons =
            runmat_value::exact_capacity_vec(m).map_err(|e| allocation(e, model))?;
        singletons.resize(m, Singleton::Empty);
        for col in 0..n {
            let first = a.colptr[col];
            let end = a.colptr[col + 1];
            if first > end || end > a.nzval.len() {
                return Ok(None);
            }
            let mut previous = None;
            for entry in first..end {
                let row = a.rowval[entry];
                let coefficient = a.nzval[entry];
                if row >= m || previous.is_some_and(|p| p >= row) || !coefficient.is_finite() {
                    return Ok(None);
                }
                previous = Some(row);
                if coefficient != 0.0 {
                    singletons[row] = match singletons[row] {
                        Singleton::Empty => Singleton::One { column: col, entry },
                        _ => Singleton::Many,
                    };
                }
                if entry % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
            cooperation.completed(1).await?;
        }
        let mut bounds = runmat_value::exact_capacity_vec(n).map_err(|e| allocation(e, model))?;
        bounds.resize_with(n, Interval::default);
        let mut consistent = true;
        let mut contradiction = None;
        for (row, singleton) in singletons.into_iter().enumerate() {
            let rhs = BigRational::from_float(model.b[row]).expect("finite RHS checked");
            match singleton {
                Singleton::Empty => {
                    if equalities[row] && !rhs.is_zero() || !equalities[row] && rhs.is_negative() {
                        consistent = false;
                        contradiction.get_or_insert(Contradiction::EmptyRow(row));
                    }
                }
                Singleton::One { column, entry } => {
                    let coefficient = BigRational::from_float(a.nzval[entry])
                        .expect("finite nonzero coefficient");
                    let positive = coefficient.is_positive();
                    let value = rhs / coefficient;
                    let interval = &mut bounds[column];
                    if (equalities[row] || !positive)
                        && interval.lower.as_ref().is_none_or(|bound| bound < &value)
                    {
                        interval.lower = Some(value.clone());
                        interval.lower_entry = Some(entry);
                    }
                    if (equalities[row] || positive)
                        && interval.upper.as_ref().is_none_or(|bound| bound > &value)
                    {
                        interval.upper = Some(value);
                        interval.upper_entry = Some(entry);
                    }
                    if interval
                        .lower
                        .as_ref()
                        .zip(interval.upper.as_ref())
                        .is_some_and(|(l, u)| l > u)
                    {
                        consistent = false;
                        contradiction.get_or_insert(Contradiction::SingletonBounds {
                            lower_entry: interval.lower_entry.expect("retained lower bound"),
                            upper_entry: interval.upper_entry.expect("retained upper bound"),
                        });
                    }
                }
                Singleton::Many => {}
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(Self {
            model,
            equalities,
            bounds,
            consistent,
            contradiction,
        }))
    }
    /// Exact contradictions from zero rows or intersected singleton bounds.
    pub(crate) fn is_infeasible(&self) -> bool {
        !self.consistent
    }
    fn effective<'a>(
        &self,
        values: &'a [BigRational],
        zero: &'a BigRational,
        row: usize,
    ) -> &'a BigRational {
        // Projection is only a dual proposal. Every correction below retains
        // the resulting stationarity residual; no primal point is clamped.
        if !self.equalities[row] && values[row].is_negative() {
            zero
        } else {
            &values[row]
        }
    }
    async fn corrected(
        &self,
        values: &[BigRational],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<BigRational>> {
        if !self.consistent || values.len() != self.model.b.len() {
            return Ok(None);
        }
        let model = self.model;
        let zero = BigRational::zero();
        let mut offset = 0;
        for cone in model.cones {
            let size = dimension(cone).expect("linear layout checked");
            let entries = (offset..offset + size).map(|row| self.effective(values, &zero, row));
            if !terminal::contains(cone, entries, true, cooperation).await? {
                return Ok(None);
            }
            offset += size;
        }
        let Some(rhs) = terminal::dot(
            model
                .b
                .iter()
                .copied()
                .enumerate()
                .map(|(row, b)| (b, self.effective(values, &zero, row))),
            cooperation,
        )
        .await?
        else {
            return Ok(None);
        };
        let mut bound = -rhs;
        for col in 0..model.f.len() {
            let Some(cost) = BigRational::from_float(model.f[col]) else {
                return Ok(None);
            };
            let entries = model.a.colptr[col]..model.a.colptr[col + 1];
            let Some(product) = terminal::dot(
                entries.map(|entry| {
                    (
                        model.a.nzval[entry],
                        self.effective(values, &zero, model.a.rowval[entry]),
                    )
                }),
                cooperation,
            )
            .await?
            else {
                return Ok(None);
            };
            let residual = cost + product;
            if residual.is_positive() {
                let Some(lower) = &self.bounds[col].lower else {
                    return Ok(None);
                };
                bound += residual * lower;
            } else if residual.is_negative() {
                let Some(upper) = &self.bounds[col].upper else {
                    return Ok(None);
                };
                bound += residual * upper;
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(bound))
    }
    /// None is a missing finite proof (-infinity), not an unboundedness claim.
    /// All existing proposals are inspected: a weak first correction must not
    /// hide a later exactly stationary or stronger recovered dual.
    pub(crate) async fn lower_bound(
        &self,
        dual: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<ExactLowerBound>> {
        if dual.len() != self.model.b.len() || !self.consistent {
            return Ok(None);
        }
        let mut best: Option<BigRational> = None;
        let _ = super::propose_exact_candidates(
            dual,
            self.model.identity,
            cooperation,
            async |candidate: &[BigRational], cooperation: &mut ScalarCooperation| {
                if let Some(bound) = self.corrected(candidate, cooperation).await? {
                    if best.as_ref().is_none_or(|old| old < &bound) {
                        best = Some(bound);
                    }
                }
                Ok(false)
            },
        )
        .await?;
        Ok(best.map(ExactLowerBound))
    }
}
/// Exact original objective evaluation, not a primal-feasibility assertion.
pub(crate) async fn objective(
    model: &Model<'_>,
    point: &[BigRational],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<BigRational>> {
    if model.p.is_some() || point.len() != model.f.len() {
        return Ok(None);
    }
    terminal::dot(model.f.iter().copied().zip(point), cooperation).await
}
/// Exact diagnostics on the supplied point, including a rounded public point;
/// this does not grant feasibility or numerical-success authority.
pub(crate) async fn evaluate(
    model: &Model<'_>,
    point: &[BigRational],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<(BigRational, BigRational)>> {
    if model.p.is_some() || !model.valid_layout() || point.len() != model.f.len() {
        return Ok(None);
    }
    let Some(objective) = objective(model, point, cooperation).await? else {
        return Ok(None);
    };
    let Some(mut activity) = terminal::negative_activity(model, point, cooperation).await? else {
        return Ok(None);
    };
    let mut offset = 0;
    let mut violation = BigRational::zero();
    for cone in model.cones {
        let (count, equality) = match cone {
            SupportedConeT::ZeroConeT(n) => (*n, true),
            SupportedConeT::NonnegativeConeT(n) => (*n, false),
            _ => return Ok(None),
        };
        for row in offset..offset + count {
            let Some(rhs) = BigRational::from_float(model.b[row]) else {
                return Ok(None);
            };
            let slack = activity[row].take().unwrap_or_else(BigRational::zero) + rhs;
            let error = if equality {
                slack.abs()
            } else {
                (-slack).max(BigRational::zero())
            };
            violation = violation.max(error);
            cooperation.completed(1).await?;
        }
        offset += count;
    }
    Ok(Some((objective, violation)))
}
