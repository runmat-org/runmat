use super::*;

impl<'a> ValuePayloadValidationCursor<'a> {
    pub(super) fn sparse(&mut self, value: &'a SparseValue) -> Result<(), ContractError> {
        if value.column_offsets.len() as u64 != value.columns.saturating_add(1)
            || value.column_offsets.first().copied() != Some(0)
        {
            return Err(invalid(
                "sparse payload",
                "column offsets are not canonical CSC offsets",
            ));
        }
        self.tasks.push(Task::SparseOffsets(value, 0));
        Ok(())
    }

    pub(super) fn sparse_offset(
        &mut self,
        value: &'a SparseValue,
        index: usize,
    ) -> Result<(), ContractError> {
        self.budget.add_bytes(8)?;
        if index + 1 < value.column_offsets.len() {
            if value.column_offsets[index] > value.column_offsets[index + 1] {
                return Err(invalid(
                    "sparse payload",
                    "column offsets are not canonical CSC offsets",
                ));
            }
            self.tasks.push(Task::SparseOffsets(value, index + 1));
        } else {
            let nonzero = value.row_indices.len() as u64;
            if value.column_offsets.last().copied() != Some(nonzero)
                || value.little_endian_data.len() as u64
                    != nonzero.saturating_mul(value.element_type.byte_width())
            {
                return Err(invalid(
                    "sparse payload",
                    "indices or data length do not match nonzero count",
                ));
            }
            self.tasks.push(Task::SparseRows(value, 0));
        }
        Ok(())
    }

    pub(super) fn sparse_row(
        &mut self,
        value: &'a SparseValue,
        index: usize,
    ) -> Result<(), ContractError> {
        if let Some(row) = value.row_indices.get(index) {
            self.budget.add_bytes(8)?;
            if *row >= value.rows {
                return Err(invalid(
                    "sparse payload",
                    "indices or data length do not match nonzero count",
                ));
            }
            self.tasks.push(Task::SparseRows(value, index + 1));
        } else {
            self.budget.add_elements(value.row_indices.len() as u64)?;
            self.budget
                .add_bytes(value.little_endian_data.len() as u64)?;
        }
        Ok(())
    }
}
