//! Bounded transient capture of original assembly contributions. This is input
//! to a mathematical check, never a certificate or a deserializable proof.
use super::*;

pub(super) const SOURCE_HEADER_BYTES: u64 = 256;

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
        widths.try_fold(SOURCE_HEADER_BYTES, |bytes, width| {
            bytes
                .checked_add(block_storage_bytes(width)?)
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
        self,
        matrix: &RealSparseMatrix,
        prescribed: &[bool],
        normalization: f64,
    ) -> Result<EnergyVerdict, EnergyError> {
        self.prove_projected(
            matrix,
            EnergyProjection::Identity(prescribed),
            prescribed.len() as u64,
            normalization,
        )
    }

    pub(crate) fn prove_selection(
        self,
        matrix: &RealSparseMatrix,
        rows: &[Option<usize>],
    ) -> Result<EnergyVerdict, EnergyError> {
        self.prove_projected(
            matrix,
            EnergyProjection::Selection(rows),
            std::mem::size_of_val(rows) as u64,
            1.0,
        )
    }

    fn prove_projected(
        mut self,
        matrix: &RealSparseMatrix,
        projection: EnergyProjection<'_>,
        projection_bytes: u64,
        normalization: f64,
    ) -> Result<EnergyVerdict, EnergyError> {
        if self.captured != self.blocks.len() || !normalization.is_finite() || normalization <= 0. {
            return Err(EnergyError::InvalidBlock);
        }
        let envelope = CoercivityWorkEnvelope::derive(matrix.pattern())
            .map_err(|_| EnergyError::Arithmetic)?;
        // Multiple original elements can contribute to the same assembled pair.
        // Neither projection kind may infer source work from the deduplicated CSR.
        let admitted = SourceEnergyDemand::derive(
            envelope,
            projection.len(),
            projection_bytes,
            self.blocks.iter().map(|block| block.width),
        )?
        .admission();
        let mut check = AnchoredEnergyCheck::new_projected(
            matrix,
            projection,
            admitted,
            self.resident_bytes
                .checked_add(projection_bytes)
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

/// Original capture owner supplies the block/header inventory to preparation and proof planning.
pub(super) fn block_storage_bytes(width: usize) -> Result<u64, EnergyError> {
    if !(2..=32).contains(&width) {
        return Err(EnergyError::InvalidBlock);
    }
    let width = width as u64;
    Ok(32 + width * 8 + width * width * 8)
}
