use super::super::*;
use super::dense::{element_width, is_complex_integer, DenseOutput};
use runmat_execution::value::SparseValue;
use runmat_value::SparseCscValidation;
use std::num::NonZeroUsize;

#[path = "sparse/managed.rs"]
mod managed;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use managed::ManagedSparse;

enum Phase {
    ColumnOffsets,
    RowIndices,
    Values,
    Structure,
}

pub(in crate::execution::value_codec::decode::machine) struct SparseTask<'a> {
    source: &'a SparseValue,
    rows: usize,
    columns: usize,
    column_offsets: Vec<usize>,
    row_indices: Vec<usize>,
    byte_offset: usize,
    values: Option<DenseOutput>,
    validation: Option<SparseCscValidation>,
    phase: Phase,
    path: ControlPath<'a>,
    managed: Option<ManagedSparse<'a>>,
    runtime: Option<&'a crate::context::RuntimeContext>,
}

impl<'a> SparseTask<'a> {
    pub(super) fn new(
        source: &'a SparseValue,
        path: ControlPath<'a>,
        runtime: Option<&'a crate::context::RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let rows = usize::try_from(source.rows)
            .map_err(|_| ValueCodecError::invalid(&path, "row count exceeds platform limits"))?;
        let columns = usize::try_from(source.columns)
            .map_err(|_| ValueCodecError::invalid(&path, "column count exceeds platform limits"))?;
        if is_complex_integer(source.element_type) {
            return Err(ValueCodecError::unsupported(
                &path,
                "the runtime has no matching sparse value class",
            ));
        }
        let pointers = columns.checked_add(1).ok_or_else(|| {
            ValueCodecError::invalid(&path, "column count exceeds platform limits")
        })?;
        let bytes = source
            .row_indices
            .len()
            .checked_mul(element_width(source.element_type))
            .ok_or_else(|| {
                ValueCodecError::invalid(&path, "sparse payload exceeds platform limits")
            })?;
        if source.column_offsets.len() != pointers || source.little_endian_data.len() != bytes {
            return Err(ValueCodecError::invalid(
                &path,
                "sparse topology and payload lengths disagree",
            ));
        }
        let managed = ManagedSparse::new(source, rows, columns, runtime, &path)?;
        let (column_offsets, row_indices) = if managed.is_some() {
            (Vec::new(), Vec::new())
        } else {
            (
                Vec::with_capacity(source.column_offsets.len()),
                Vec::with_capacity(source.row_indices.len()),
            )
        };
        Ok(Self {
            source,
            rows,
            columns,
            column_offsets,
            row_indices,
            byte_offset: 0,
            values: None,
            validation: None,
            phase: Phase::ColumnOffsets,
            path,
            managed,
            runtime,
        })
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        if let Some(managed) = &mut self.managed {
            return managed.advance(self.source, maximum_work, &self.path);
        }
        if HostAllocationReceipt::policy(self.runtime)
            .map_err(|error| ValueCodecError::allocation(&self.path, error))?
            != HostAllocationPolicy::Unbounded
        {
            return Err(ValueCodecError::allocation(
                &self.path,
                HostAllocationError::StaleAuthority,
            ));
        }
        match self.phase {
            Phase::ColumnOffsets => {
                let completed = copy_indices(
                    &self.source.column_offsets,
                    &mut self.column_offsets,
                    maximum_work,
                    &self.path,
                )?;
                if self.column_offsets.len() == self.source.column_offsets.len() {
                    self.phase = Phase::RowIndices;
                }
                Ok((completed, None))
            }
            Phase::RowIndices => {
                let completed = copy_indices(
                    &self.source.row_indices,
                    &mut self.row_indices,
                    maximum_work,
                    &self.path,
                )?;
                if self.row_indices.len() == self.source.row_indices.len() {
                    if is_complex_integer(self.source.element_type) {
                        return Err(ValueCodecError::unsupported(
                            &self.path,
                            "the runtime has no matching sparse value class",
                        ));
                    }
                    let width = element_width(self.source.element_type);
                    if !self.source.little_endian_data.len().is_multiple_of(width) {
                        return Err(ValueCodecError::invalid(
                            &self.path,
                            "encoded data is not aligned to its element width",
                        ));
                    }
                    self.values = Some(DenseOutput::new(
                        self.source.element_type,
                        self.source.little_endian_data.len() / width,
                    ));
                    self.phase = Phase::Values;
                }
                Ok((completed, None))
            }
            Phase::Values => {
                let width = element_width(self.source.element_type);
                let available = (self.source.little_endian_data.len() - self.byte_offset) / width;
                let elements = available.min(maximum_work);
                let end = self.byte_offset + elements * width;
                for bytes in
                    self.source.little_endian_data[self.byte_offset..end].chunks_exact(width)
                {
                    self.values.as_mut().expect("value decoder").push(bytes);
                }
                self.byte_offset = end;
                if self.byte_offset == self.source.little_endian_data.len() {
                    self.phase = Phase::Structure;
                    let values_len = self.source.row_indices.len();
                    self.validation = Some(
                        SparseCscValidation::new(
                            self.rows,
                            self.columns,
                            std::mem::take(&mut self.column_offsets).into(),
                            std::mem::take(&mut self.row_indices).into(),
                            values_len,
                        )
                        .map_err(|error| ValueCodecError::invalid(&self.path, error))?,
                    );
                }
                Ok((elements.max(1), None))
            }
            Phase::Structure => {
                let completed = maximum_work;
                let complete = self
                    .validation
                    .as_mut()
                    .expect("sparse validation")
                    .advance(NonZeroUsize::new(maximum_work).expect("machine work is nonzero"))
                    .map_err(|error| ValueCodecError::invalid(&self.path, error))?;
                if !complete {
                    return Ok((completed, None));
                }
                let parts = self
                    .validation
                    .take()
                    .expect("sparse validation")
                    .into_validated()
                    .map_err(|_| {
                        ValueCodecError::invalid(&self.path, "sparse validation incomplete")
                    })?;
                let value = self
                    .values
                    .take()
                    .expect("sparse values")
                    .into_sparse(parts, &self.path)?;
                Ok((completed, Some(value)))
            }
        }
    }

    pub(super) fn path(&self) -> &ControlPath<'a> {
        &self.path
    }
}

fn copy_indices(
    source: &[u64],
    output: &mut Vec<usize>,
    maximum_work: usize,
    path: &str,
) -> Result<usize, ValueCodecError> {
    let count = (source.len() - output.len()).min(maximum_work);
    for value in &source[output.len()..output.len() + count] {
        output.push(
            usize::try_from(*value)
                .map_err(|_| ValueCodecError::invalid(path, "value exceeds platform limits"))?,
        );
    }
    Ok(count.max(1))
}
