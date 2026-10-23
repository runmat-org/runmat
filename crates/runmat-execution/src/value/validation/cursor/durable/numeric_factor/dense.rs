//! Bounded validation of canonical packed dense-factor words and maps.
use super::*;
use crate::value::DurableDenseOrthogonalFactor;

impl<'a> ValuePayloadValidationCursor<'a> {
    pub(super) fn dense_factor(
        &mut self,
        factor: &'a DurableNumericFactor,
    ) -> Result<(), ContractError> {
        self.tasks.push(Task::FactorMap(factor, 0, 0));
        match factor {
            DurableNumericFactor::DenseLu { rows, .. } => self.factor_permutation(rows),
            DurableNumericFactor::DenseCholesky { .. } => {}
            DurableNumericFactor::DenseQr(first) => self.factor_permutation(&first.permutation),
            DurableNumericFactor::DenseCod { first, completion } => {
                self.factor_permutation(&first.permutation);
                if let Some(second) = completion {
                    self.factor_permutation(&second.permutation);
                }
            }
            _ => unreachable!("only dense factor reaches dense validation"),
        }
        for index in 0..4 {
            if let Some(block) = factor_dense_block(factor, index) {
                self.tasks.push(Task::FactorDense(factor, index, 0));
                self.shape(
                    &block.shape,
                    AfterShape::Dense(block.element_type, &block.little_endian_data),
                );
            }
        }
        Ok(())
    }

    pub(in crate::value::validation::cursor) fn factor_dense(
        &mut self,
        factor: &'a DurableNumericFactor,
        block_index: usize,
        offset: usize,
    ) -> Result<(), ContractError> {
        let block = factor_dense_block(factor, block_index)
            .ok_or_else(|| invalid("numerical factor", "dense factor block is missing"))?;
        let width = usize::try_from(block.element_type.byte_width())
            .map_err(|_| invalid("numerical factor", "dense factor width is invalid"))?;
        let count = block.little_endian_data.len() / width;
        if offset == count {
            return Ok(());
        }
        if offset > count {
            return Err(invalid(
                "numerical factor",
                "dense factor offset exceeds payload",
            ));
        }
        let (real, imaginary) = factor_word(block.element_type, &block.little_endian_data, offset)?;
        let rows = block.shape[0];
        let word_index = offset;
        let offset = u64::try_from(word_index)
            .map_err(|_| invalid("numerical factor", "dense factor offset overflows"))?;
        let row = offset % rows;
        let column = offset / rows;
        let coefficient = coefficient(factor, block_index);
        let identity_reflector = coefficient.is_some()
            && row == column % rows
            && real == f64::INFINITY
            && imaginary == 0.0;
        if !(real.is_finite() && imaginary.is_finite() || identity_reflector) {
            return Err(invalid(
                "numerical factor",
                "dense factor contains invalid nonfinite data",
            ));
        }
        if matches!(factor, DurableNumericFactor::DenseCholesky { .. })
            && ((row < column && (real != 0.0 || imaginary != 0.0))
                || (row == column && (real <= 0.0 || imaginary != 0.0)))
        {
            return Err(invalid(
                "numerical factor",
                "dense Cholesky lower factor is invalid",
            ));
        }
        if let Some(rank) = leading_rank(factor, block_index) {
            if row == column && column < rank && real == 0.0 && imaginary == 0.0 {
                return Err(invalid(
                    "numerical factor",
                    "dense QR leading diagonal is zero",
                ));
            }
        }
        self.tasks.push(Task::FactorDense(
            factor,
            block_index,
            word_index
                .checked_add(1)
                .ok_or_else(|| invalid("numerical factor", "dense factor offset overflows"))?,
        ));
        Ok(())
    }
}

fn coefficient(
    value: &DurableNumericFactor,
    index: usize,
) -> Option<&DurableDenseOrthogonalFactor> {
    match value {
        DurableNumericFactor::DenseQr(first) | DurableNumericFactor::DenseCod { first, .. }
            if index == 1 =>
        {
            Some(first)
        }
        DurableNumericFactor::DenseCod {
            completion: Some(second),
            ..
        } if index == 3 => Some(second),
        _ => None,
    }
}

fn leading_rank(value: &DurableNumericFactor, index: usize) -> Option<u64> {
    match value {
        DurableNumericFactor::DenseQr(first) | DurableNumericFactor::DenseCod { first, .. }
            if index == 0 =>
        {
            Some(first.rank)
        }
        DurableNumericFactor::DenseCod {
            completion: Some(second),
            ..
        } if index == 2 => Some(second.rank),
        _ => None,
    }
}
