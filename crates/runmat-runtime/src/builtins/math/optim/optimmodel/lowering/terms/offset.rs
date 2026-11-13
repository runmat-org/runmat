//! Move a newly reduced scalar-offset column into its real numerical owner.
use super::super::array::Offset;
use super::*;
impl Coefficients {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn into_offset(
        self,
        shape: SharedShape,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Offset> {
        let Self::Sparse(table) = self else {
            return Ok(Offset::Uniform(0.));
        };
        let table = std::rc::Rc::try_unwrap(table)
            .map_err(|_| policy.invalid("new offset column unexpectedly has shared ownership"))?;
        if table.columns != [0] || table.matrix.n != 1 {
            return Err(policy.invalid("scalar offset reduction has non-scalar column roles"));
        }
        let mut matrix = table.matrix;
        if matrix.m != evaluation::element_count(&shape, policy)? {
            return Err(policy.invalid("scalar offset differs from original output extent"));
        }
        if shape.iter().skip(2).all(|&d| d == 1) {
            let rows = shape.first().copied().unwrap_or(1);
            let columns = shape.get(1).copied().unwrap_or(1);
            let count = columns.checked_add(1).ok_or_else(|| {
                policy.storage(runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
            })?;
            let mut pointers =
                runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
            pointers.push(0);
            let mut cursor = 0;
            for column in 0..columns {
                while cursor < matrix.rowval.len()
                    && rows != 0
                    && matrix.rowval[cursor] / rows == column
                {
                    matrix.rowval[cursor] %= rows;
                    cursor += 1;
                    c.completed(1).await?;
                }
                pointers.push(cursor);
                c.completed(1).await?;
            }
            if cursor != matrix.rowval.len() {
                return Err(policy.invalid("offset coordinates exceed original output shape"));
            }

            let mut validation = runmat_value::SparseCscValidation::new(
                rows,
                columns,
                pointers.into(),
                matrix.rowval.into(),
                matrix.nzval.len(),
            )
            .map_err(|e| policy.invalid(e))?;
            let quantum = std::num::NonZeroUsize::new(1).expect("one CSC work item");
            loop {
                let done = validation.advance(quantum).map_err(|e| policy.invalid(e))?;
                c.completed(1).await?;
                if done {
                    break;
                }
            }
            let parts = validation
                .into_validated()
                .map_err(|_| policy.invalid("completed scalar-offset CSC did not seal"))?;
            let values = runmat_value::HostNumericBuffer::from_numeric_storage(
                runmat_value::NumericStorage::F64(matrix.nzval),
            );
            let sparse = runmat_value::SparseTensor::from_validated_numeric(parts, values)
                .map_err(|e| policy.invalid(e))?;
            return Ok(Offset::Sparse(sparse));
        }
        let count = evaluation::element_count(&shape, policy)?;
        let mut values = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        for _ in 0..count {
            values.push(0.);
            c.completed(1).await?;
        }
        for (row, value) in matrix.rowval.into_iter().zip(matrix.nzval) {
            values[row] = value;
            c.completed(1).await?;
        }
        let tensor = runmat_value::Tensor::new(values, shape).map_err(|e| policy.invalid(e))?;
        Ok(Offset::Dense(tensor))
    }
}
