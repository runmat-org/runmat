use anyhow::{anyhow, ensure};
use serde::{Deserialize, Serialize};

use super::{SparseFormat, SparseIndexWidth};

#[derive(Clone, Copy, Debug)]
pub struct HostCsrPatternView<'a> {
    pub rows: u64,
    pub columns: u64,
    pub row_offsets: &'a [u64],
    pub column_indices: &'a [u64],
    pub index_width: SparseIndexWidth,
}

impl HostCsrPatternView<'_> {
    pub fn validate(self) -> anyhow::Result<SparsePatternDescriptor> {
        let row_offset_count = self
            .rows
            .checked_add(1)
            .ok_or_else(|| anyhow!("sparse CSR row count overflows its offset array"))?;
        ensure!(
            u64::try_from(self.row_offsets.len()).ok() == Some(row_offset_count),
            "sparse CSR requires rows + 1 row offsets"
        );
        ensure!(
            self.row_offsets.first().copied() == Some(0),
            "sparse CSR row offsets must start at zero"
        );
        ensure!(
            self.row_offsets.windows(2).all(|pair| pair[0] <= pair[1]),
            "sparse CSR row offsets must be nondecreasing"
        );
        let nnz = u64::try_from(self.column_indices.len())
            .map_err(|_| anyhow!("sparse CSR nonzero count does not fit u64"))?;
        ensure!(
            self.row_offsets.last().copied() == Some(nnz),
            "sparse CSR final row offset must equal its nonzero count"
        );
        ensure!(
            self.column_indices
                .iter()
                .all(|column| *column < self.columns),
            "sparse CSR column index is outside the matrix"
        );
        ensure!(
            self.row_offsets
                .iter()
                .chain(self.column_indices)
                .all(|value| self.index_width.accepts(*value)),
            "sparse CSR indices exceed the requested provider index width"
        );
        for range in self.row_offsets.windows(2) {
            let start = usize::try_from(range[0])
                .map_err(|_| anyhow!("sparse CSR row offset does not fit usize"))?;
            let end = usize::try_from(range[1])
                .map_err(|_| anyhow!("sparse CSR row offset does not fit usize"))?;
            ensure!(
                self.column_indices[start..end]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
                "sparse CSR columns must be strictly increasing within each row"
            );
        }
        Ok(SparsePatternDescriptor {
            format: SparseFormat::Csr,
            index_width: self.index_width,
            rows: self.rows,
            columns: self.columns,
            nnz,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparsePatternDescriptor {
    pub format: SparseFormat,
    pub index_width: SparseIndexWidth,
    pub rows: u64,
    pub columns: u64,
    pub nnz: u64,
}

impl SparsePatternDescriptor {
    pub fn validate(self) -> anyhow::Result<()> {
        let entry_capacity = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| anyhow!("sparse matrix dimensions exceed addressable entry count"))?;
        ensure!(
            self.nnz <= entry_capacity,
            "sparse nonzero count exceeds matrix dimensions"
        );
        ensure!(
            self.checked_storage_bytes().is_some(),
            "sparse pattern storage size overflows"
        );
        Ok(())
    }

    pub fn checked_storage_bytes(self) -> Option<u64> {
        self.rows
            .checked_add(1)?
            .checked_add(self.nnz)?
            .checked_mul(self.index_width.byte_width())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparsePatternHandle {
    pub device_id: u32,
    pub pattern_id: u64,
    pub descriptor: SparsePatternDescriptor,
}

pub fn validate_pattern_owner(
    pattern: &SparsePatternHandle,
    provider_device_id: u32,
) -> anyhow::Result<()> {
    ensure!(
        pattern.device_id == provider_device_id,
        "sparse pattern belongs to a different provider"
    );
    Ok(())
}
