//! Exact Double-A/integer-B selection. Output values are read only from A;
//! integer B is indexed in its authoritative class, never materialized as f64.
use super::super::set_policy::{ElementKey, Operand};
use super::{
    setdiff_error_with, setdiff_internal_error, SetdiffEvaluation,
    SETDIFF_ERROR_ROWS_COLUMN_MISMATCH,
};
use crate::builtins::logical::rel::integer_comparison::compare_numeric_scalars_exact;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::{SetdiffOptions, SetdiffOrder};
use runmat_value::{NumericDType, NumericScalar, Tensor, Value};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy)]
pub(super) struct Execution<'a> {
    runtime: Option<&'a RuntimeContext>,
    double_left: bool,
}

/// Borrowed A-record identity for the existing same-double deduplication
/// policy. Keys retain indices, not copied workspace values or row payloads.
#[derive(Clone, Copy)]
struct ARecord<'a> {
    source: Records<'a>,
    index: usize,
}
impl PartialEq for ARecord<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.source.columns == other.source.columns
            && ((std::ptr::eq(self.source.tensor, other.source.tensor)
                && self.index == other.index)
                || (!self.source.missing(self.index)
                    && !other.source.missing(other.index)
                    && self.source.compare(self.index, other.source, other.index)
                        == Some(Ordering::Equal)))
    }
}
impl Eq for ARecord<'_> {}
impl Hash for ARecord<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.source.columns.hash(state);
        for column in 0..self.source.columns {
            let NumericScalar::F64(value) = self.source.at(self.index, column) else {
                unreachable!("A is native double")
            };
            ElementKey::real(value, Operand::A, self.index).hash(state);
        }
    }
}
impl<'a> Execution<'a> {
    pub(super) fn new(
        runtime: Option<&'a RuntimeContext>,
        a: &Value,
        b: &Value,
    ) -> BuiltinResult<Self> {
        let double_left = match a {
            Value::Num(_) => true,
            Value::Tensor(a) => a.numeric_dtype() == NumericDType::F64,
            Value::GpuTensor(a) => {
                runmat_accelerate_api::handle_integer_type(a).is_none()
                    && !runmat_accelerate_api::handle_is_logical(a)
                    && runmat_accelerate_api::handle_storage(a)
                        == runmat_accelerate_api::GpuTensorStorage::Real
                    && runmat_accelerate_api::handle_precision(a)
                        == Some(runmat_accelerate_api::ProviderPrecision::F64)
            }
            _ => false,
        };
        let integer_right = match b {
            Value::Int(_) => true,
            Value::Tensor(b) => b.integer_storage().is_some(),
            Value::GpuTensor(b) => runmat_accelerate_api::handle_integer_type(b).is_some(),
            _ => false,
        };
        let execution = Self {
            runtime,
            double_left,
        };
        if double_left && integer_right {
            execution.admit()?;
        }
        Ok(execution)
    }
    /// Provider numeric helpers have no Runtime authority and may not select a
    /// new mixed-class route on behalf of an implicit finite caller.
    pub(super) const fn provider() -> Self {
        Self {
            runtime: None,
            double_left: false,
        }
    }
    pub(super) fn selects(self, a: &Tensor, b: &Tensor) -> bool {
        self.double_left && a.numeric_dtype() == NumericDType::F64 && b.integer_storage().is_some()
    }
    fn admit(self) -> BuiltinResult<()> {
        if HostAllocationReceipt::policy(self.runtime)
            .map_err(|cause| allocation(self.runtime, cause))?
            == HostAllocationPolicy::Managed
        {
            return Err(allocation(
                self.runtime,
                HostAllocationError::MissingManagedOwner,
            ));
        }
        self.checkpoint()
    }
    fn checkpoint(self) -> BuiltinResult<()> {
        if self.runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(allocation(self.runtime, HostAllocationError::Cancelled));
        }
        Ok(())
    }
}
fn allocation(runtime: Option<&RuntimeContext>, cause: HostAllocationError) -> RuntimeError {
    let mut error = host_allocation_runtime_error(
        runtime,
        "RunMat:setdiff:HostAllocation",
        format!("setdiff: exact mixed-input selector allocation: {cause}"),
        cause,
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

#[derive(Clone, Copy)]
struct Records<'a> {
    tensor: &'a Tensor,
    count: usize,
    columns: usize,
    rows: bool,
}
impl<'a> Records<'a> {
    fn new(tensor: &'a Tensor, rows: bool) -> BuiltinResult<Self> {
        tensor.validate_shape().map_err(setdiff_internal_error)?;
        let (count, columns) = if rows {
            let [count, columns] = tensor.shape() else {
                return Err(setdiff_internal_error(
                    "setdiff: 'rows' requires 2-D numeric matrices",
                ));
            };
            (*count, *columns)
        } else {
            (tensor.len(), 1)
        };
        Ok(Self {
            tensor,
            count,
            columns,
            rows,
        })
    }
    fn at(self, record: usize, column: usize) -> NumericScalar {
        let index = if self.rows {
            record + column * self.count
        } else {
            record
        };
        self.tensor
            .numeric_value_at(index)
            .expect("validated native record index")
    }
    fn missing(self, record: usize) -> bool {
        (0..self.columns).any(|column| self.at(record, column).is_nan())
    }
    fn compare(self, left: usize, other: Self, right: usize) -> Option<Ordering> {
        for column in 0..self.columns {
            let ordering =
                compare_numeric_scalars_exact(self.at(left, column), other.at(right, column))?;
            if ordering != Ordering::Equal {
                return Some(ordering);
            }
        }
        Some(Ordering::Equal)
    }
    fn ordered(self, left: usize, right: usize) -> Ordering {
        for column in 0..self.columns {
            let a = self.at(left, column);
            let b = self.at(right, column);
            let order = match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => compare_numeric_scalars_exact(a, b).expect("nonmissing native real order"),
            };
            if order != Ordering::Equal {
                return order;
            }
        }
        left.cmp(&right)
    }
}
fn indices(count: usize) -> BuiltinResult<Vec<usize>> {
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(count)
        .map_err(|_| setdiff_internal_error("setdiff: index scratch allocation failed"))?;
    indices.extend(0..count);
    Ok(indices)
}
fn contains(a: Records<'_>, record: usize, b: Records<'_>, order: &[usize]) -> bool {
    if a.missing(record) {
        return false;
    }
    order
        .binary_search_by(|&index| {
            b.compare(index, a, record)
                .expect("integer B and nonmissing A have an exact order")
        })
        .is_ok()
}

pub(super) fn evaluate(
    execution: Execution<'_>,
    a: Tensor,
    b: Tensor,
    options: &SetdiffOptions,
) -> BuiltinResult<SetdiffEvaluation> {
    // This leaf owns all newly added scratch. Finite execution refuses before
    // geometry allocation, sorting, or output construction until complete
    // selector AND retained-output ownership is available.
    execution.admit()?;
    let a = Records::new(&a, options.rows)?;
    let b = Records::new(&b, options.rows)?;
    if a.columns != b.columns {
        return Err(setdiff_error_with(
            &SETDIFF_ERROR_ROWS_COLUMN_MISMATCH,
            "setdiff: row inputs must have matching columns",
        ));
    }
    if a.count as u128 > 1u128 << 53 {
        return Err(setdiff_internal_error(
            "setdiff: index exceeds exact double range",
        ));
    }
    let mut b_order = indices(b.count)?;
    b_order.sort_unstable_by(|&left, &right| {
        b.compare(left, b, right)
            .expect("integer records have total exact order")
    });
    execution.checkpoint()?;
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(a.count)
        .map_err(|_| setdiff_internal_error("setdiff: selected index allocation failed"))?;
    let mut seen = HashSet::new();
    seen.try_reserve(a.count)
        .map_err(|_| setdiff_internal_error("setdiff: record key allocation failed"))?;
    for record in 0..a.count {
        if record.is_multiple_of(256) {
            execution.checkpoint()?;
        }
        if !contains(a, record, b, &b_order)
            && seen.insert(ARecord {
                source: a,
                index: record,
            })
        {
            selected.push(record);
        }
    }
    if matches!(options.order, SetdiffOrder::Sorted) {
        selected.sort_unstable_by(|&left, &right| a.ordered(left, right));
    }
    execution.checkpoint()?;
    let count = selected.len();
    let len = count
        .checked_mul(a.columns)
        .ok_or_else(|| setdiff_internal_error("setdiff: output size overflow"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| setdiff_internal_error("setdiff: output allocation failed"))?;
    for column in 0..a.columns {
        for &record in &selected {
            if values.len().is_multiple_of(4096) {
                execution.checkpoint()?;
            }
            let NumericScalar::F64(value) = a.at(record, column) else {
                unreachable!("selector admits native double A only")
            };
            values.push(value);
        }
    }
    let mut ia = Vec::new();
    ia.try_reserve_exact(count)
        .map_err(|_| setdiff_internal_error("setdiff: output index allocation failed"))?;
    for (position, index) in selected.into_iter().enumerate() {
        if position.is_multiple_of(4096) {
            execution.checkpoint()?;
        }
        ia.push(index as f64 + 1.0);
    }
    let shape = if options.rows {
        vec![count, a.columns]
    } else {
        vec![count, 1]
    };
    let values = Tensor::new(values, shape).map_err(setdiff_internal_error)?;
    let ia = Tensor::new(ia, vec![count, 1]).map_err(setdiff_internal_error)?;
    Ok(SetdiffEvaluation::new(
        crate::builtins::common::tensor::tensor_into_value(values),
        ia,
    ))
}
