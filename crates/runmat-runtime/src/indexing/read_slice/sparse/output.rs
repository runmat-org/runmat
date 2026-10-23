//! Same final CSC owner for sparse read, merge, growth and deletion. Count is
//! established from borrowed input before any final backing is allocated.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::indexing::plan::admission::Owner;
use crate::RuntimeError;
use runmat_value::{
    ComplexElement, ExactHostSparseClass, ExactHostSparsePublicationError,
    ExactHostSparseTensorAllocation, NumericDType, NumericScalar, NumericStorage, SparseTensor,
};
use std::num::NonZeroUsize;

#[derive(Clone, Copy)]
pub(in crate::indexing) enum Element {
    Logical(bool),
    Numeric(NumericScalar),
    Complex32(ComplexElement<f32>),
    Complex64(ComplexElement<f64>),
}
impl Element {
    pub(in crate::indexing) fn is_zero(self) -> bool {
        match self {
            Self::Logical(value) => !value,
            Self::Numeric(value) => value.is_zero(),
            Self::Complex32(ComplexElement(real, imag)) => real == 0.0 && imag == 0.0,
            Self::Complex64(ComplexElement(real, imag)) => real == 0.0 && imag == 0.0,
        }
    }
}
pub(in crate::indexing) fn element(source: &SparseTensor, entry: usize) -> Element {
    if source.is_logical() {
        Element::Logical(true)
    } else if let Some(values) = source.as_complex_f32_slice() {
        Element::Complex32(values[entry])
    } else if let Some(values) = source.as_complex_f64_slice() {
        Element::Complex64(values[entry])
    } else {
        Element::Numeric(
            source
                .numeric_value_at(entry)
                .expect("validated sparse numeric storage"),
        )
    }
}
fn class(source: &SparseTensor) -> ExactHostSparseClass {
    if source.is_logical() {
        ExactHostSparseClass::Logical
    } else if source.is_complex() {
        ExactHostSparseClass::Complex(source.numeric_dtype().expect("floating complex"))
    } else {
        ExactHostSparseClass::Numeric(source.numeric_dtype().expect("numeric sparse"))
    }
}
pub(in crate::indexing) fn allocation(
    context: Option<&RuntimeContext>,
    cause: HostAllocationError,
) -> RuntimeError {
    host_allocation_runtime_error(
        context,
        "RunMat:Indexing:PayloadAllocation",
        "sparse CSC storage",
        cause,
    )
}
pub(in crate::indexing) fn invalid(detail: impl std::fmt::Display) -> RuntimeError {
    super::super::map_slice_shape_error(detail)
}

/// Same 4096-work quantum used by the dense/primitive indexing loops. This
/// batches only pure numeric/metadata work; effect and publication boundaries
/// still perform mandatory checks on the original Owner.
pub(in crate::indexing) fn step(owner: &Owner<'_>, work: &mut usize) -> Result<(), RuntimeError> {
    *work += 1;
    if *work == 4096 {
        *work = 0;
        owner.check()?;
    }
    Ok(())
}

mod index;
pub(in crate::indexing) use index::indices;

enum OrdinaryValues {
    Logical,
    Numeric(NumericStorage),
    Complex32(Vec<(f32, f32)>),
    Complex64(Vec<(f64, f64)>),
}
pub(in crate::indexing) struct Ordinary {
    pointers: Vec<usize>,
    indices: Vec<usize>,
    values: OrdinaryValues,
}
pub(in crate::indexing) enum Destination<'a> {
    Exact(&'a mut ExactHostSparseTensorAllocation),
    Ordinary(&'a mut Ordinary),
}
impl Destination<'_> {
    pub(in crate::indexing) fn pointer(
        &mut self,
        column: usize,
        count: usize,
    ) -> Result<(), RuntimeError> {
        match self {
            Self::Exact(output) => output.set_col_ptr(column, count).map_err(invalid),
            Self::Ordinary(output) => {
                output.pointers[column] = count;
                Ok(())
            }
        }
    }
    pub(in crate::indexing) fn set(
        &mut self,
        index: usize,
        row: usize,
        value: Element,
    ) -> Result<(), RuntimeError> {
        match self {
            Self::Exact(output) => {
                output.set_row_index(index, row).map_err(invalid)?;
                match value {
                    Element::Logical(true) => Ok(()),
                    Element::Logical(false) => Err(invalid("sparse logical zero must be elided")),
                    Element::Numeric(value) => {
                        output.set_numeric_value(index, value).map_err(invalid)
                    }
                    Element::Complex32(value) => {
                        output.set_complex_f32_value(index, value).map_err(invalid)
                    }
                    Element::Complex64(ComplexElement(real, imag)) => {
                        output.set_complex_value(index, real, imag).map_err(invalid)
                    }
                }
            }
            Self::Ordinary(output) => {
                output.indices[index] = row;
                match (&mut output.values, value) {
                    (OrdinaryValues::Logical, Element::Logical(true)) => Ok(()),
                    (OrdinaryValues::Numeric(values), Element::Numeric(value)) => {
                        values.set_value(index, value).map_err(invalid)
                    }
                    (
                        OrdinaryValues::Complex32(values),
                        Element::Complex32(ComplexElement(real, imag)),
                    ) => {
                        values[index] = (real, imag);
                        Ok(())
                    }
                    (
                        OrdinaryValues::Complex64(values),
                        Element::Complex64(ComplexElement(real, imag)),
                    ) => {
                        values[index] = (real, imag);
                        Ok(())
                    }
                    _ => Err(invalid("sparse output class mismatch")),
                }
            }
        }
    }
}

pub(in crate::indexing) fn build(
    context: Option<&RuntimeContext>,
    prototype: &SparseTensor,
    rows: usize,
    cols: usize,
    nnz: usize,
    fill: impl FnOnce(&mut Destination<'_>) -> Result<(), RuntimeError>,
) -> Result<SparseTensor, RuntimeError> {
    let owner = Owner::new(context)?;
    let class = class(prototype);
    let raw =
        ExactHostSparseTensorAllocation::requested_bytes_for_dimensions(class, rows, cols, nnz)
            .map_err(|cause| allocation(context, cause.into()))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| allocation(context, HostAllocationError::Overflow))?,
        class.retained_claim_count(),
    )
    .map_err(|cause| allocation(context, cause))?;
    let receipt = HostAllocationReceipt::admit(
        context,
        HostAllocationPlan::new(0, retained).map_err(|cause| allocation(context, cause))?,
    )
    .map_err(|cause| allocation(context, cause))?;
    let output = if let Some(receipt) = &receipt {
        let runtime = context.expect("managed sparse output has an owner");
        receipt
            .with_output_construction(runtime, |permit| {
                let mut destination =
                    ExactHostSparseTensorAllocation::new(class, rows, cols, nnz, permit)
                        .map_err(|cause| allocation(context, cause.into()))?;
                fill(&mut Destination::Exact(&mut destination))?;
                owner.check()?;
                let mut ready =
                    destination
                        .into_validation_fallible()
                        .map_err(|cause| match cause {
                            ExactHostSparsePublicationError::Allocation(cause) => {
                                allocation(context, cause.into())
                            }
                            ExactHostSparsePublicationError::Invalid(detail) => invalid(detail),
                        })?;
                while !ready
                    .advance(NonZeroUsize::new(4096).expect("nonzero validation quantum"))
                    .map_err(invalid)?
                {
                    owner.check()?;
                }
                owner.check()?;
                ready.into_tensor().map_err(invalid)
            })
            .map_err(|cause| allocation(context, cause))??
    } else {
        let values = match class {
            ExactHostSparseClass::Logical => OrdinaryValues::Logical,
            ExactHostSparseClass::Numeric(dtype) => {
                OrdinaryValues::Numeric(NumericStorage::zeros(dtype, nnz))
            }
            ExactHostSparseClass::Complex(NumericDType::F32) => {
                OrdinaryValues::Complex32(vec![(0.0, 0.0); nnz])
            }
            ExactHostSparseClass::Complex(NumericDType::F64) => {
                OrdinaryValues::Complex64(vec![(0.0, 0.0); nnz])
            }
            _ => unreachable!("validated floating complex class"),
        };
        let pointer_count = cols
            .checked_add(1)
            .ok_or_else(|| allocation(context, HostAllocationError::Overflow))?;
        let mut destination = Ordinary {
            pointers: vec![0; pointer_count],
            indices: vec![0; nnz],
            values,
        };
        fill(&mut Destination::Ordinary(&mut destination))?;
        owner.check()?;
        let Ordinary {
            pointers,
            indices,
            values,
        } = destination;
        match values {
            OrdinaryValues::Logical => SparseTensor::new_logical(rows, cols, pointers, indices),
            OrdinaryValues::Numeric(values) => SparseTensor::from_host_numeric_buffers(
                rows,
                cols,
                pointers.into(),
                indices.into(),
                runmat_value::HostNumericBuffer::from_numeric_storage(values),
            ),
            OrdinaryValues::Complex32(values) => {
                SparseTensor::new_complex_f32(rows, cols, pointers, indices, values)
            }
            OrdinaryValues::Complex64(values) => {
                SparseTensor::new_complex(rows, cols, pointers, indices, values)
            }
        }
        .map_err(invalid)?
    };
    owner.check()?;
    Ok(output)
}
