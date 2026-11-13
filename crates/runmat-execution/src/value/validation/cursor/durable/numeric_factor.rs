//! Bounded durable-factor envelope validation. The Runtime's canonical
//! NumericFactorValidation checks numerical topology after exact import.

use super::*;
use crate::value::{DurableFactorPermutation, DurableOrthogonalFactor};
#[path = "numeric_factor/dense.rs"]
mod dense;

impl<'a> ValuePayloadValidationCursor<'a> {
    pub(super) fn numeric_factor(
        &mut self,
        factor: &'a DurableNumericFactor,
    ) -> Result<(), ContractError> {
        geometry(factor)?;
        if matches!(factor, DurableNumericFactor::Empty { .. }) {
            return Ok(());
        }
        if matches!(
            factor,
            DurableNumericFactor::DenseLu { .. }
                | DurableNumericFactor::DenseCholesky { .. }
                | DurableNumericFactor::DenseQr(_)
                | DurableNumericFactor::DenseCod { .. }
        ) {
            return self.dense_factor(factor);
        }
        self.tasks.push(Task::FactorScale(factor, 0));
        self.tasks.push(Task::FactorMap(factor, 0, 0));
        self.tasks.push(Task::FactorBlock(factor, 0, 0, 0, false));
        match factor {
            DurableNumericFactor::Empty { .. } => unreachable!("handled empty factor"),
            DurableNumericFactor::DenseLu { .. }
            | DurableNumericFactor::DenseCholesky { .. }
            | DurableNumericFactor::DenseQr(_)
            | DurableNumericFactor::DenseCod { .. } => unreachable!("handled dense factor"),
            DurableNumericFactor::Lu {
                lower,
                upper,
                rows,
                columns,
                row_scale,
            } => {
                self.factor_sparse(lower)?;
                self.factor_sparse(upper)?;
                self.factor_permutation(rows);
                self.factor_permutation(columns);
                if let Some(scale) = row_scale {
                    if !matches!(scale.element_type, ElementType::F32 | ElementType::F64)
                        || scale.shape.len() != 2
                        || scale.shape[1] != 1
                    {
                        return Err(invalid(
                            "numerical factor",
                            "invalid row-scale class or shape",
                        ));
                    }
                    self.shape(
                        &scale.shape,
                        AfterShape::Dense(scale.element_type, &scale.little_endian_data),
                    );
                }
            }
            DurableNumericFactor::Cholesky { lower, permutation } => {
                self.factor_sparse(lower)?;
                self.factor_permutation(permutation);
            }
            DurableNumericFactor::Qr(first) => self.factor_orthogonal(first)?,
            DurableNumericFactor::Cod { first, completion } => {
                self.factor_orthogonal(first)?;
                if let Some(completion) = completion {
                    self.factor_orthogonal(completion)?;
                }
            }
        }
        Ok(())
    }

    fn factor_orthogonal(
        &mut self,
        value: &'a DurableOrthogonalFactor,
    ) -> Result<(), ContractError> {
        self.factor_sparse(&value.r)?;
        self.factor_sparse(&value.reflectors)?;
        self.factor_permutation(&value.permutation);
        Ok(())
    }

    fn factor_sparse(&mut self, value: &'a SparseValue) -> Result<(), ContractError> {
        if !matches!(
            value.element_type,
            ElementType::F32 | ElementType::F64 | ElementType::ComplexF32 | ElementType::ComplexF64
        ) {
            return Err(invalid(
                "numerical factor",
                "factor blocks require native floating storage",
            ));
        }
        self.sparse(value)
    }

    fn factor_permutation(&mut self, value: &'a DurableFactorPermutation) {
        self.tasks.push(Task::FactorIndices(&value.forward, 0));
        self.tasks.push(Task::FactorIndices(&value.inverse, 0));
    }

    pub(in crate::value::validation::cursor) fn factor_index(
        &mut self,
        values: &'a [u64],
        index: usize,
    ) -> Result<(), ContractError> {
        if values.get(index).is_some() {
            self.tasks.push(Task::FactorIndices(values, index + 1));
        } else {
            let count = u64::try_from(values.len())
                .map_err(|_| invalid("numerical factor", "permutation length overflows"))?;
            self.budget.add_elements(count)?;
            self.budget.add_bytes(count.checked_mul(8).ok_or_else(|| {
                invalid("numerical factor", "permutation byte count overflows")
            })?)?;
        }
        Ok(())
    }
}

#[path = "numeric_factor/geometry.rs"]
mod geometry;
use geometry::{
    factor_block, factor_dense_block, factor_permutation, geometry, orthogonal_for_block,
};

impl<'a> ValuePayloadValidationCursor<'a> {
    pub(in crate::value::validation::cursor) fn factor_block(
        &mut self,
        factor: &'a DurableNumericFactor,
        block_index: usize,
        column: usize,
        edge: usize,
        diagonal: bool,
    ) -> Result<(), ContractError> {
        let Some(block) = factor_block(factor, block_index) else {
            return Ok(());
        };
        let columns = usize::try_from(block.columns)
            .map_err(|_| invalid("numerical factor", "factor columns exceed platform limits"))?;
        if column >= columns {
            self.tasks
                .push(Task::FactorBlock(factor, block_index + 1, 0, 0, false));
            return Ok(());
        }
        let end = usize::try_from(
            *block
                .column_offsets
                .get(column + 1)
                .ok_or_else(|| invalid("numerical factor", "missing factor column offset"))?,
        )
        .map_err(|_| invalid("numerical factor", "factor offset exceeds platform limits"))?;
        let start = usize::try_from(
            *block
                .column_offsets
                .get(column)
                .ok_or_else(|| invalid("numerical factor", "missing factor column offset"))?,
        )
        .map_err(|_| invalid("numerical factor", "factor offset exceeds platform limits"))?;
        if edge == 0 && start > end {
            return Err(invalid(
                "numerical factor",
                "factor offsets are not ordered",
            ));
        }
        let absolute = start
            .checked_add(edge)
            .ok_or_else(|| invalid("numerical factor", "factor edge offset overflows"))?;
        if absolute < end {
            let row = *block
                .row_indices
                .get(absolute)
                .ok_or_else(|| invalid("numerical factor", "factor row is missing"))?;
            if row >= block.rows
                || (edge > 0
                    && block
                        .row_indices
                        .get(absolute - 1)
                        .is_some_and(|prior| *prior >= row))
            {
                return Err(invalid(
                    "numerical factor",
                    "factor rows must be sorted and unique",
                ));
            }
            let (real, imaginary) = factor_value(block, absolute)?;
            if !real.is_finite() || !imaginary.is_finite() {
                return Err(invalid("numerical factor", "factor has nonfinite data"));
            }
            if let Some((orthogonal, reflector)) = orthogonal_for_block(factor, block_index) {
                if reflector {
                    if column as u64 >= orthogonal.rank || row < column as u64 {
                        return Err(invalid(
                            "numerical factor",
                            "reflector support exceeds rank",
                        ));
                    }
                } else if row < orthogonal.rank && row > column as u64 {
                    return Err(invalid(
                        "numerical factor",
                        "QR leading block is not upper triangular",
                    ));
                }
            }
            let lower = matches!(
                factor,
                DurableNumericFactor::Lu { .. } | DurableNumericFactor::Cholesky { .. }
            ) && block_index == 0;
            let upper = matches!(factor, DurableNumericFactor::Lu { .. }) && block_index == 1;
            if (lower && row < column as u64) || (upper && row > column as u64) {
                return Err(invalid(
                    "numerical factor",
                    "factor has an off-triangle entry",
                ));
            }
            let mut diagonal = diagonal;
            if row == column as u64 {
                diagonal = true;
                if let Some((orthogonal, false)) = orthogonal_for_block(factor, block_index) {
                    if (column as u64) < orthogonal.rank && real == 0.0 && imaginary == 0.0 {
                        return Err(invalid("numerical factor", "QR leading diagonal is zero"));
                    }
                }
                if lower
                    && matches!(factor, DurableNumericFactor::Lu { .. })
                    && (real != 1.0 || imaginary != 0.0)
                {
                    return Err(invalid("numerical factor", "LU lower diagonal is not one"));
                }
                if lower
                    && matches!(factor, DurableNumericFactor::Cholesky { .. })
                    && (real <= 0.0 || imaginary != 0.0)
                {
                    return Err(invalid(
                        "numerical factor",
                        "Cholesky diagonal is not positive real",
                    ));
                }
            }
            self.tasks.push(Task::FactorBlock(
                factor,
                block_index,
                column,
                edge + 1,
                diagonal,
            ));
        } else {
            let requires_diagonal = (block_index == 0
                && matches!(
                    factor,
                    DurableNumericFactor::Lu { .. } | DurableNumericFactor::Cholesky { .. }
                ))
                || orthogonal_for_block(factor, block_index)
                    .is_some_and(|(qr, _)| (column as u64) < qr.rank);
            if requires_diagonal && !diagonal {
                return Err(invalid(
                    "numerical factor",
                    "factor is missing a required diagonal",
                ));
            }
            self.tasks
                .push(Task::FactorBlock(factor, block_index, column + 1, 0, false));
        }
        Ok(())
    }

    pub(in crate::value::validation::cursor) fn factor_map(
        &mut self,
        factor: &'a DurableNumericFactor,
        map_index: usize,
        offset: usize,
    ) -> Result<(), ContractError> {
        let Some((map, extent)) = factor_permutation(factor, map_index) else {
            return Ok(());
        };
        if let Some(&forward) = map.forward.get(offset) {
            let inverse_index = usize::try_from(forward)
                .map_err(|_| invalid("numerical factor", "factor map exceeds platform limits"))?;
            if forward >= extent
                || map.inverse.get(inverse_index).copied() != u64::try_from(offset).ok()
            {
                return Err(invalid(
                    "numerical factor",
                    "factor maps are not inverse permutations",
                ));
            }
            self.tasks
                .push(Task::FactorMap(factor, map_index, offset + 1));
        } else {
            self.tasks.push(Task::FactorMap(factor, map_index + 1, 0));
        }
        Ok(())
    }

    pub(in crate::value::validation::cursor) fn factor_scale(
        &mut self,
        factor: &'a DurableNumericFactor,
        offset: usize,
    ) -> Result<(), ContractError> {
        let DurableNumericFactor::Lu {
            row_scale: Some(scale),
            ..
        } = factor
        else {
            return Ok(());
        };
        let width = usize::try_from(scale.element_type.byte_width())
            .map_err(|_| invalid("numerical factor", "factor scale width is invalid"))?;
        let start = offset
            .checked_mul(width)
            .ok_or_else(|| invalid("numerical factor", "factor scale offset overflows"))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| invalid("numerical factor", "factor scale offset overflows"))?;
        let Some(bytes) = scale.little_endian_data.get(start..end) else {
            return Ok(());
        };
        let value = match scale.element_type {
            ElementType::F32 => f64::from(f32::from_bits(u32::from_le_bytes(
                bytes.try_into().expect("F32 scale width"),
            ))),
            ElementType::F64 => f64::from_bits(u64::from_le_bytes(
                bytes.try_into().expect("F64 scale width"),
            )),
            _ => return Err(invalid("numerical factor", "factor scale is not floating")),
        };
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid(
                "numerical factor",
                "LU row scaling is not finite positive",
            ));
        }
        self.tasks.push(Task::FactorScale(factor, offset + 1));
        Ok(())
    }
}

fn factor_value(value: &SparseValue, index: usize) -> Result<(f64, f64), ContractError> {
    factor_word(value.element_type, &value.little_endian_data, index)
}

fn factor_word(
    element_type: ElementType,
    payload: &[u8],
    index: usize,
) -> Result<(f64, f64), ContractError> {
    let width = usize::try_from(element_type.byte_width())
        .map_err(|_| invalid("numerical factor", "factor value width is invalid"))?;
    let start = index
        .checked_mul(width)
        .ok_or_else(|| invalid("numerical factor", "factor value offset overflows"))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| invalid("numerical factor", "factor value offset overflows"))?;
    let bytes = payload
        .get(start..end)
        .ok_or_else(|| invalid("numerical factor", "factor value is missing"))?;
    Ok(match element_type {
        ElementType::F32 => (
            f64::from(f32::from_bits(u32::from_le_bytes(
                bytes.try_into().expect("F32 width"),
            ))),
            0.0,
        ),
        ElementType::F64 => (
            f64::from_bits(u64::from_le_bytes(bytes.try_into().expect("F64 width"))),
            0.0,
        ),
        ElementType::ComplexF32 => (
            f64::from(f32::from_bits(u32::from_le_bytes(
                bytes[..4].try_into().expect("real F32"),
            ))),
            f64::from(f32::from_bits(u32::from_le_bytes(
                bytes[4..].try_into().expect("imag F32"),
            ))),
        ),
        ElementType::ComplexF64 => (
            f64::from_bits(u64::from_le_bytes(bytes[..8].try_into().expect("real F64"))),
            f64::from_bits(u64::from_le_bytes(bytes[8..].try_into().expect("imag F64"))),
        ),
        _ => return Err(invalid("numerical factor", "factor block is not floating")),
    })
}
