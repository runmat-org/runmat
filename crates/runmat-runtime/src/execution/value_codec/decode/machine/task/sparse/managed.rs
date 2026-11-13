//! Decode directly into the admitted final CSC buffers, with bounded validation.

use super::super::dense::{numeric_dtype, scalar};
use super::*;
use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use crate::context::RuntimeContext;
use runmat_execution::value::ElementType;
use runmat_value::{
    ComplexElement, ExactHostSparseClass, ExactHostSparseTensorAllocation, NumericScalar,
    PreparedExactSparseTensor,
};

pub(super) struct ManagedSparse<'a> {
    output: Option<ExactHostSparseTensorAllocation>,
    validation: Option<PreparedExactSparseTensor>,
    pointers: usize,
    rows: usize,
    elements: usize,
    receipt: HostAllocationReceipt,
    runtime: &'a RuntimeContext,
}

impl<'a> ManagedSparse<'a> {
    pub(super) fn new(
        source: &SparseValue,
        rows: usize,
        columns: usize,
        runtime: Option<&'a RuntimeContext>,
        path: &str,
    ) -> Result<Option<Self>, ValueCodecError> {
        let allocation = |error| ValueCodecError::allocation(path, error);
        if HostAllocationReceipt::policy(runtime).map_err(allocation)?
            == crate::context::host_allocation::HostAllocationPolicy::Unbounded
        {
            return Ok(None);
        }
        let dtype = numeric_dtype(source.element_type);
        let class = match source.element_type {
            ElementType::Logical => ExactHostSparseClass::Logical,
            ElementType::ComplexF32 | ElementType::ComplexF64 => {
                ExactHostSparseClass::Complex(dtype)
            }
            _ => ExactHostSparseClass::Numeric(dtype),
        };
        let raw = ExactHostSparseTensorAllocation::requested_bytes_for_dimensions(
            class,
            rows,
            columns,
            source.row_indices.len(),
        )
        .map_err(|error| allocation(error.into()))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| {
                allocation(crate::context::host_allocation::HostAllocationError::Overflow)
            })?,
            class.retained_claim_count(),
        )
        .map_err(allocation)?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, retained).map_err(allocation)?,
        )
        .map_err(allocation)?
        .ok_or_else(|| {
            allocation(crate::context::host_allocation::HostAllocationError::StaleAuthority)
        })?;
        let runtime = runtime.expect("managed sparse receipt has an invocation");
        let output = receipt
            .with_output_allocation(runtime, |permit| {
                ExactHostSparseTensorAllocation::new(
                    class,
                    rows,
                    columns,
                    source.row_indices.len(),
                    permit,
                )
            })
            .map_err(allocation)?;
        Ok(Some(Self {
            output: Some(output),
            validation: None,
            pointers: 0,
            rows: 0,
            elements: 0,
            receipt,
            runtime,
        }))
    }

    pub(super) fn advance(
        &mut self,
        source: &SparseValue,
        maximum_work: usize,
        path: &str,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        self.checkpoint(path)?;
        let maximum_work = maximum_work.max(1);
        let invalid = |error| ValueCodecError::invalid(path, error);
        if let Some(validation) = &mut self.validation {
            let done = validation
                .advance(NonZeroUsize::new(maximum_work).unwrap())
                .map_err(invalid)?;
            self.checkpoint(path)?;
            let value = if done {
                Some(Value::SparseTensor(
                    self.validation
                        .take()
                        .expect("completed validation")
                        .into_tensor()
                        .map_err(invalid)?,
                ))
            } else {
                None
            };
            self.checkpoint(path)?;
            return Ok((maximum_work, value));
        }
        let output = self.output.as_mut().expect("active sparse output");
        let completed;
        if self.pointers < source.column_offsets.len() {
            let end = source
                .column_offsets
                .len()
                .min(self.pointers.saturating_add(maximum_work));
            for index in self.pointers..end {
                let value = usize::try_from(source.column_offsets[index])
                    .map_err(|_| invalid("column offset exceeds platform limits".to_owned()))?;
                output.set_col_ptr(index, value).map_err(invalid)?;
            }
            completed = end - self.pointers;
            self.pointers = end;
        } else if self.rows < source.row_indices.len() {
            let end = source
                .row_indices
                .len()
                .min(self.rows.saturating_add(maximum_work));
            for index in self.rows..end {
                let value = usize::try_from(source.row_indices[index])
                    .map_err(|_| invalid("row index exceeds platform limits".to_owned()))?;
                output.set_row_index(index, value).map_err(invalid)?;
            }
            completed = end - self.rows;
            self.rows = end;
        } else {
            let width = element_width(source.element_type);
            let end = source
                .row_indices
                .len()
                .min(self.elements.saturating_add(maximum_work));
            for index in self.elements..end {
                let bytes = &source.little_endian_data[index * width..(index + 1) * width];
                match source.element_type {
                    ElementType::Logical => {
                        if bytes[0] != 1 {
                            return Err(invalid(
                                "sparse logical payload must match its true pattern".to_owned(),
                            ));
                        }
                    }
                    ElementType::ComplexF32 | ElementType::ComplexF64 => {
                        let lane = width / 2;
                        let dtype = numeric_dtype(source.element_type);
                        match (scalar(dtype, &bytes[..lane]), scalar(dtype, &bytes[lane..])) {
                            (NumericScalar::F32(re), NumericScalar::F32(im)) => {
                                output.set_complex_f32_value(index, ComplexElement(re, im))
                            }
                            (NumericScalar::F64(re), NumericScalar::F64(im)) => {
                                output.set_complex_value(index, re, im)
                            }
                            _ => unreachable!("floating complex wire class"),
                        }
                        .map_err(invalid)?;
                    }
                    _ => output
                        .set_numeric_value(index, scalar(numeric_dtype(source.element_type), bytes))
                        .map_err(invalid)?,
                }
            }
            completed = (end - self.elements).max(1);
            self.elements = end;
            if end == source.row_indices.len() {
                self.validation = Some(
                    self.output
                        .take()
                        .expect("completed sparse payload")
                        .into_validation()
                        .map_err(invalid)?,
                );
            }
        }
        self.checkpoint(path)?;
        Ok((completed.max(1), None))
    }

    fn checkpoint(&self, path: &str) -> Result<(), ValueCodecError> {
        self.receipt
            .checkpoint(self.runtime)
            .map_err(|error| ValueCodecError::allocation(path, error))
    }
}
