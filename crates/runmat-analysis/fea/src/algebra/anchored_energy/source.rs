//! Bounded transient capture of original assembly contributions. This is input
//! to a mathematical check, never a certificate or a deserializable proof.
use super::*;

struct Block {
    node_start: usize,
    value_start: usize,
    width: usize,
}

pub(crate) struct AnchoredEnergySource {
    blocks: Vec<Block>,
    nodes: Vec<usize>,
    values: Vec<f64>,
    captured: usize,
    resident_bytes: u64,
}

impl AnchoredEnergySource {
    /// Exact inventory with conservative allocation/header allowance. Shared by
    /// task planning and capture, before allocating any source payload.
    pub fn resident_bytes_for_widths(
        mut widths: impl Iterator<Item = usize>,
    ) -> Result<u64, EnergyError> {
        widths.try_fold(256_u64, |bytes, width| {
            if !(2..=32).contains(&width) {
                return Err(EnergyError::InvalidBlock);
            }
            let width = width as u64;
            bytes
                .checked_add(32 + width * 8 + width * width * 8)
                .ok_or(EnergyError::Arithmetic)
        })
    }

    pub fn new(
        widths: impl Iterator<Item = usize> + Clone,
        byte_limit: u64,
    ) -> Result<Self, EnergyError> {
        check_cancelled()?;
        let resident_bytes = Self::resident_bytes_for_widths(widths.clone())?;
        if resident_bytes > byte_limit {
            return Err(EnergyError::Admission {
                operations: 0,
                operation_limit: 0,
                resident: resident_bytes,
                resident_limit: byte_limit,
            });
        }
        let mut node_count = 0_usize;
        let mut value_count = 0_usize;
        let mut block_count = 0_usize;
        for width in widths.clone() {
            node_count = node_count
                .checked_add(width)
                .ok_or(EnergyError::Arithmetic)?;
            value_count = value_count
                .checked_add(width * width)
                .ok_or(EnergyError::Arithmetic)?;
            block_count = block_count.checked_add(1).ok_or(EnergyError::Arithmetic)?;
        }
        let mut blocks = Vec::with_capacity(block_count);
        let mut node_start = 0;
        let mut value_start = 0;
        for width in widths {
            blocks.push(Block {
                node_start,
                value_start,
                width,
            });
            node_start += width;
            value_start += width * width;
        }
        Ok(Self {
            blocks,
            nodes: vec![0; node_count],
            values: vec![0.; value_count],
            captured: 0,
            resident_bytes,
        })
    }

    /// Ordered once-only access keeps the original source inventory complete.
    pub fn capture_next(&mut self, nodes: &[usize]) -> Result<&mut [f64], EnergyError> {
        check_cancelled()?;
        let block = self
            .blocks
            .get(self.captured)
            .ok_or(EnergyError::InvalidBlock)?;
        if block.width != nodes.len() {
            return Err(EnergyError::InvalidBlock);
        }
        self.nodes[block.node_start..block.node_start + block.width].copy_from_slice(nodes);
        self.captured += 1;
        Ok(&mut self.values[block.value_start..block.value_start + block.width * block.width])
    }

    pub(crate) fn prove(
        mut self,
        matrix: &RealSparseMatrix,
        prescribed: &[bool],
        normalization: f64,
    ) -> Result<EnergyVerdict, EnergyError> {
        if self.captured != self.blocks.len() || !normalization.is_finite() || normalization <= 0. {
            return Err(EnergyError::InvalidBlock);
        }
        let envelope = CoercivityWorkEnvelope::derive(matrix.pattern())
            .map_err(|_| EnergyError::Arithmetic)?;
        let mut check = AnchoredEnergyCheck::new_with_retained(
            matrix,
            prescribed,
            envelope,
            self.resident_bytes
                .checked_add(prescribed.len() as u64)
                .ok_or(EnergyError::Arithmetic)?,
            self.values.len() as u64,
        )?;
        for block in &self.blocks {
            check_cancelled()?;
            let values =
                &mut self.values[block.value_start..block.value_start + block.width * block.width];
            for value in values.iter_mut() {
                *value *= normalization;
            }
            check.push_block(
                &self.nodes[block.node_start..block.node_start + block.width],
                values,
            )?;
        }
        check.finish()
    }
}

#[cfg(test)]
mod tests;
