//! Shared native-four sparse publication for managed and ordinary execution.

use super::managed_storage::Fence;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use runmat_execution::resource::RetainedHostAllocationClaim;
use runmat_value::{
    ComplexElement, ExactHostSparseClass, ExactHostSparseTensorAllocation, HostNumericBuffer,
    NumericDType, NumericScalar, NumericStorage, SparseCscValidation, SparseTensor,
    ValidatedSparseCscParts, Value,
};
use std::num::NonZeroUsize;
use std::sync::Arc;

#[derive(Debug)]
pub(in crate::builtins::math::linalg) enum SparsePublicationError {
    Admission(HostAllocationError),
    Execution(ExecutionServiceError),
    Invalid(String),
    Allocation,
}

#[cfg(test)]
mod tests;
impl From<HostAllocationError> for SparsePublicationError {
    fn from(v: HostAllocationError) -> Self {
        Self::Admission(v)
    }
}
impl From<ExecutionServiceError> for SparsePublicationError {
    fn from(v: ExecutionServiceError) -> Self {
        Self::Execution(v)
    }
}
impl std::fmt::Display for SparsePublicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admission(e) => write!(f, "sparse output admission failed: {e}"),
            Self::Execution(e) => write!(f, "sparse output execution failed: {e}"),
            Self::Invalid(e) => write!(f, "invalid sparse output: {e}"),
            Self::Allocation => f.write_str("sparse output allocation failed"),
        }
    }
}
impl std::error::Error for SparsePublicationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(e) => Some(e),
            Self::Execution(e) => Some(e),
            _ => None,
        }
    }
}

pub(in crate::builtins::math::linalg) enum Values {
    F32(Vec<f32>),
    F64(Vec<f64>),
    C32(Vec<ComplexElement<f32>>),
    C64(Vec<ComplexElement<f64>>),
}
impl Values {
    fn new(class: ExactHostSparseClass, n: usize) -> Result<Self, SparsePublicationError> {
        fn v<T>(n: usize) -> Result<Vec<T>, SparsePublicationError> {
            let mut v = Vec::new();
            v.try_reserve_exact(n)
                .map_err(|_| SparsePublicationError::Allocation)?;
            Ok(v)
        }
        Ok(match class {
            ExactHostSparseClass::Numeric(NumericDType::F32) => Self::F32(v(n)?),
            ExactHostSparseClass::Numeric(NumericDType::F64) => Self::F64(v(n)?),
            ExactHostSparseClass::Complex(NumericDType::F32) => Self::C32(v(n)?),
            ExactHostSparseClass::Complex(NumericDType::F64) => Self::C64(v(n)?),
            ExactHostSparseClass::Logical
            | ExactHostSparseClass::Numeric(_)
            | ExactHostSparseClass::Complex(_) => {
                return Err(SparsePublicationError::Invalid(
                    "sparse publication class is not native-four".into(),
                ))
            }
        })
    }
    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::C32(v) => v.len(),
            Self::C64(v) => v.len(),
        }
    }
    fn push(&mut self, r: f64, i: f64) {
        match self {
            Self::F32(v) => v.push(r as f32),
            Self::F64(v) => v.push(r),
            Self::C32(v) => v.push(ComplexElement(r as f32, i as f32)),
            Self::C64(v) => v.push(ComplexElement(r, i)),
        }
    }
    fn finish(
        self,
        parts: ValidatedSparseCscParts,
    ) -> Result<SparseTensor, SparsePublicationError> {
        match self {
            Self::F32(v) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::F32(v)),
            ),
            Self::F64(v) => SparseTensor::from_validated_numeric(
                parts,
                HostNumericBuffer::from_numeric_storage(NumericStorage::F64(v)),
            ),
            Self::C32(v) => SparseTensor::from_validated_complex_f32(parts, v.into()),
            Self::C64(v) => SparseTensor::from_validated_complex_f64(parts, v.into()),
        }
        .map_err(SparsePublicationError::Invalid)
    }
}

pub(in crate::builtins::math::linalg) trait PublicationReceiptOwner {
    fn receipt(&self) -> &HostAllocationReceipt;
}
impl PublicationReceiptOwner for HostAllocationReceipt {
    fn receipt(&self) -> &HostAllocationReceipt {
        self
    }
}
impl PublicationReceiptOwner for &HostAllocationReceipt {
    fn receipt(&self) -> &HostAllocationReceipt {
        self
    }
}
pub(in crate::builtins::math::linalg) struct ManagedPublication<R> {
    output: Option<ExactHostSparseTensorAllocation>,
    receipt: R,
    class: ExactHostSparseClass,
    columns: usize,
    entries: usize,
    next_pointer: usize,
    next_entry: usize,
    _control_claim: Arc<dyn RetainedHostAllocationClaim>,
}

pub(in crate::builtins::math::linalg) enum SparsePublication<'a> {
    ManagedOwned(Box<ManagedPublication<HostAllocationReceipt>>),
    ManagedBorrowed(Box<ManagedPublication<&'a HostAllocationReceipt>>),
    Ordinary {
        ptr: Vec<usize>,
        rows: Vec<usize>,
        values: Values,
        m: usize,
        n: usize,
        nnz: usize,
    },
}

impl<R: PublicationReceiptOwner> ManagedPublication<R> {
    fn pointer(&mut self, index: usize, value: usize) -> Result<(), SparsePublicationError> {
        if index != self.next_pointer || index > self.columns || value > self.entries {
            return Err(SparsePublicationError::Invalid(
                "nonsequential pointer".into(),
            ));
        }
        self.output
            .as_mut()
            .expect("managed sparse output is present before finish")
            .set_col_ptr(index, value)
            .map_err(SparsePublicationError::Invalid)?;
        self.next_pointer += 1;
        Ok(())
    }

    fn entry(
        &mut self,
        index: usize,
        row: usize,
        value: (f64, f64),
    ) -> Result<(), SparsePublicationError> {
        if index != self.next_entry || index >= self.entries {
            return Err(SparsePublicationError::Invalid(
                "nonsequential entry".into(),
            ));
        }
        let output = self
            .output
            .as_mut()
            .expect("managed sparse output is present before finish");
        output
            .set_row_index(index, row)
            .map_err(SparsePublicationError::Invalid)?;
        match self.class {
            ExactHostSparseClass::Complex(NumericDType::F32 | NumericDType::F64) => {
                output.set_complex_value(index, value.0, value.1)
            }
            ExactHostSparseClass::Numeric(NumericDType::F32) => {
                output.set_numeric_value(index, NumericScalar::F32(value.0 as f32))
            }
            ExactHostSparseClass::Numeric(NumericDType::F64) => {
                output.set_numeric_value(index, NumericScalar::F64(value.0))
            }
            ExactHostSparseClass::Logical
            | ExactHostSparseClass::Numeric(_)
            | ExactHostSparseClass::Complex(_) => unreachable!(),
        }
        .map_err(SparsePublicationError::Invalid)?;
        self.next_entry += 1;
        Ok(())
    }
}

async fn finish_managed<R: PublicationReceiptOwner, E>(
    mut control: Box<ManagedPublication<R>>,
    runtime: &RuntimeContext,
    check: &mut impl FnMut() -> Result<(), E>,
    map_error: &mut impl FnMut(SparsePublicationError) -> E,
) -> Result<Value, E> {
    if control.next_pointer != control.columns.saturating_add(1)
        || control.next_entry != control.entries
    {
        return Err(map_error(SparsePublicationError::Invalid(
            "sparse output does not match its admitted dimensions".into(),
        )));
    }
    let output = control
        .output
        .take()
        .expect("managed sparse output is consumed once");
    let mut validation = output
        .into_validation()
        .map_err(SparsePublicationError::Invalid)
        .map_err(&mut *map_error)?;
    let mut work = runtime.cpu_cooperation();
    loop {
        check()?;
        control
            .receipt
            .receipt()
            .checkpoint(runtime)
            .map_err(SparsePublicationError::Admission)
            .map_err(&mut *map_error)?;
        let done = validation
            .advance(NonZeroUsize::new(256).unwrap())
            .map_err(SparsePublicationError::Invalid)
            .map_err(&mut *map_error)?;
        work.checkpoint(256)
            .await
            .map_err(SparsePublicationError::Execution)
            .map_err(&mut *map_error)?;
        check()?;
        control
            .receipt
            .receipt()
            .checkpoint(runtime)
            .map_err(SparsePublicationError::Admission)
            .map_err(&mut *map_error)?;
        if done {
            break;
        }
    }
    check()?;
    validation
        .into_tensor()
        .map(Value::SparseTensor)
        .map_err(SparsePublicationError::Invalid)
        .map_err(map_error)
}
pub(in crate::builtins::math::linalg) fn sparse_publication_bytes(
    class: ExactHostSparseClass,
    m: usize,
    n: usize,
    nnz: usize,
) -> Result<u64, HostAllocationError> {
    let raw = ExactHostSparseTensorAllocation::requested_bytes_for_dimensions(class, m, n, nnz)?;
    let output = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
        class.retained_claim_count(),
    )?;
    let control_size =
        std::mem::size_of::<ManagedPublication<HostAllocationReceipt>>().max(std::mem::size_of::<
            ManagedPublication<&'static HostAllocationReceipt>,
        >());
    let control = HostAllocationPlan::retained_metadata_bytes(
        u64::try_from(control_size).map_err(|_| HostAllocationError::Overflow)?,
    )?;
    output
        .checked_add(control)
        .ok_or(HostAllocationError::Overflow)
}

fn validate_native_four(class: ExactHostSparseClass) -> Result<(), SparsePublicationError> {
    match class {
        ExactHostSparseClass::Numeric(NumericDType::F32 | NumericDType::F64)
        | ExactHostSparseClass::Complex(NumericDType::F32 | NumericDType::F64) => Ok(()),
        _ => Err(SparsePublicationError::Invalid(
            "sparse publication class is not native-four".into(),
        )),
    }
}

impl<'a> SparsePublication<'a> {
    pub fn new(
        runtime: &RuntimeContext,
        class: ExactHostSparseClass,
        m: usize,
        n: usize,
        nnz: usize,
    ) -> Result<Self, SparsePublicationError> {
        validate_native_four(class)?;
        let retained = sparse_publication_bytes(class, m, n, nnz)?;
        if let Some(receipt) =
            HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, retained)?)?
        {
            return Self::managed(runtime, receipt, class, m, n, nnz).map(Self::ManagedOwned);
        }
        Self::ordinary(runtime, class, m, n, nnz)
    }
    pub fn with_receipt(
        runtime: &RuntimeContext,
        receipt: Option<&'a HostAllocationReceipt>,
        class: ExactHostSparseClass,
        m: usize,
        n: usize,
        nnz: usize,
    ) -> Result<Self, SparsePublicationError> {
        validate_native_four(class)?;
        match receipt {
            Some(receipt) => {
                Self::managed(runtime, receipt, class, m, n, nnz).map(Self::ManagedBorrowed)
            }
            None => Self::ordinary(runtime, class, m, n, nnz),
        }
    }
    fn managed<R: PublicationReceiptOwner>(
        runtime: &RuntimeContext,
        receipt: R,
        class: ExactHostSparseClass,
        m: usize,
        n: usize,
        nnz: usize,
    ) -> Result<Box<ManagedPublication<R>>, SparsePublicationError> {
        let output = receipt
            .receipt()
            .with_output_allocation(runtime, |permit| {
                ExactHostSparseTensorAllocation::new(class, m, n, nnz, permit)
            })
            .map_err(SparsePublicationError::Admission)?;
        let control_bytes = u64::try_from(std::mem::size_of::<ManagedPublication<R>>())
            .map_err(|_| HostAllocationError::Overflow)?;
        let control_claim = receipt
            .receipt()
            .claim_retained_metadata(runtime, control_bytes)?;
        receipt.receipt().checkpoint(runtime)?;
        Ok(Box::new(ManagedPublication {
            output: Some(output),
            receipt,
            class,
            columns: n,
            entries: nnz,
            next_pointer: 0,
            next_entry: 0,
            _control_claim: control_claim,
        }))
    }
    fn ordinary(
        runtime: &RuntimeContext,
        class: ExactHostSparseClass,
        m: usize,
        n: usize,
        nnz: usize,
    ) -> Result<Self, SparsePublicationError> {
        Fence::Unbounded.checkpoint(runtime)?;
        let mut ptr = Vec::new();
        ptr.try_reserve_exact(n.checked_add(1).ok_or(HostAllocationError::Overflow)?)
            .map_err(|_| SparsePublicationError::Allocation)?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(nnz)
            .map_err(|_| SparsePublicationError::Allocation)?;
        Ok(Self::Ordinary {
            ptr,
            rows,
            values: Values::new(class, nnz)?,
            m,
            n,
            nnz,
        })
    }
    pub fn pointer(&mut self, index: usize, value: usize) -> Result<(), SparsePublicationError> {
        match self {
            Self::ManagedOwned(control) => control.pointer(index, value),
            Self::ManagedBorrowed(control) => control.pointer(index, value),
            Self::Ordinary { ptr, n, nnz, .. } => {
                if index > *n || value > *nnz || ptr.len() != index {
                    return Err(SparsePublicationError::Invalid(
                        "nonsequential pointer".into(),
                    ));
                }
                ptr.push(value);
                Ok(())
            }
        }
    }
    pub fn entry(
        &mut self,
        index: usize,
        row: usize,
        value: (f64, f64),
    ) -> Result<(), SparsePublicationError> {
        match self {
            Self::ManagedOwned(control) => control.entry(index, row, value),
            Self::ManagedBorrowed(control) => control.entry(index, row, value),
            Self::Ordinary {
                rows,
                values,
                m,
                nnz,
                ..
            } => {
                if index >= *nnz || row >= *m || rows.len() != index || values.len() != index {
                    return Err(SparsePublicationError::Invalid(
                        "nonsequential entry".into(),
                    ));
                }
                rows.push(row);
                values.push(value.0, value.1);
                Ok(())
            }
        }
    }
    pub fn checkpoint(&self, runtime: &RuntimeContext) -> Result<(), SparsePublicationError> {
        match self {
            Self::ManagedOwned(control) => control.receipt.receipt().checkpoint(runtime)?,
            Self::ManagedBorrowed(control) => control.receipt.receipt().checkpoint(runtime)?,
            Self::Ordinary { .. } => Fence::Unbounded.checkpoint(runtime)?,
        }
        Ok(())
    }
    pub async fn finish(self, runtime: &RuntimeContext) -> Result<Value, SparsePublicationError> {
        self.finish_with(runtime, || Ok(()), |error| error).await
    }
    pub async fn finish_with<E>(
        self,
        runtime: &RuntimeContext,
        mut check: impl FnMut() -> Result<(), E>,
        mut map_error: impl FnMut(SparsePublicationError) -> E,
    ) -> Result<Value, E> {
        match self {
            Self::ManagedOwned(control) => {
                finish_managed(control, runtime, &mut check, &mut map_error).await
            }
            Self::ManagedBorrowed(control) => {
                finish_managed(control, runtime, &mut check, &mut map_error).await
            }
            Self::Ordinary {
                ptr,
                rows,
                values,
                m,
                n,
                nnz,
            } => {
                Fence::Unbounded
                    .checkpoint(runtime)
                    .map_err(SparsePublicationError::Admission)
                    .map_err(&mut map_error)?;
                if ptr.len() != n.saturating_add(1)
                    || rows.len() != nnz
                    || values.len() != nnz
                    || ptr.last().copied() != Some(nnz)
                {
                    return Err(map_error(SparsePublicationError::Invalid(
                        "sparse output does not match its admitted dimensions".into(),
                    )));
                }
                let mut v = SparseCscValidation::new(m, n, ptr.into(), rows.into(), values.len())
                    .map_err(SparsePublicationError::Invalid)
                    .map_err(&mut map_error)?;
                let mut work = runtime.cpu_cooperation();
                loop {
                    check()?;
                    Fence::Unbounded
                        .checkpoint(runtime)
                        .map_err(SparsePublicationError::Admission)
                        .map_err(&mut map_error)?;
                    let done = v
                        .advance(NonZeroUsize::new(256).unwrap())
                        .map_err(SparsePublicationError::Invalid)
                        .map_err(&mut map_error)?;
                    work.checkpoint(256)
                        .await
                        .map_err(SparsePublicationError::Execution)
                        .map_err(&mut map_error)?;
                    check()?;
                    Fence::Unbounded
                        .checkpoint(runtime)
                        .map_err(SparsePublicationError::Admission)
                        .map_err(&mut map_error)?;
                    if done {
                        break;
                    }
                }
                let parts = v
                    .into_validated()
                    .map_err(|_| SparsePublicationError::Invalid("incomplete validation".into()))
                    .map_err(&mut map_error)?;
                Fence::Unbounded
                    .checkpoint(runtime)
                    .map_err(SparsePublicationError::Admission)
                    .map_err(&mut map_error)?;
                values
                    .finish(parts)
                    .map(Value::SparseTensor)
                    .map_err(&mut map_error)
            }
        }
    }
}
