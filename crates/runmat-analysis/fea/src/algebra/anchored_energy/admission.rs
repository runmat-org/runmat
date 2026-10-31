//! Inventory for proving an operator from original local energies.
//! Sparse factor caps are unchanged; original-domain proof work has its own owner.
use super::{
    check_cancelled,
    source::{block_storage_bytes, SOURCE_HEADER_BYTES},
    CoercivityWorkEnvelope, EnergyError,
};
pub(super) const LOCAL_PROOF_CACHE_CAPACITY: usize = 32;

#[derive(Clone, Copy)]
pub(super) struct EnergyAdmission {
    dimension: u64,
    operations: u64,
    resident: u64,
}
impl From<CoercivityWorkEnvelope> for EnergyAdmission {
    fn from(value: CoercivityWorkEnvelope) -> Self {
        Self {
            dimension: value.dimension(),
            operations: value.energy_operation_limit(),
            resident: value.energy_resident_limit(),
        }
    }
}
impl EnergyAdmission {
    pub fn dimension(self) -> u64 {
        self.dimension
    }
    pub fn update_limit(self) -> u64 {
        self.operations
    }
    pub fn resident_bytes_limit(self) -> u64 {
        self.resident
    }
}

/// Count-only component demand, not a mathematical certificate or measured allocation.
#[derive(Clone, Copy)]
pub(crate) struct SourceEnergyDemand {
    admission: EnergyAdmission,
    additional_resident_bytes: u64,
}
impl SourceEnergyDemand {
    pub fn derive(
        matrix: CoercivityWorkEnvelope,
        original_vertices: usize,
        projection_bytes: u64,
        widths: impl Iterator<Item = usize>,
    ) -> Result<Self, EnergyError> {
        let mut source = SOURCE_HEADER_BYTES;
        let mut graph = 0_u64;
        let mut missing = 0_u64;
        let mut scratch = 0_u64;
        let mut cache_entry = 0_u64;
        let mut blocks = 0_u64;
        let mut operations = matrix.energy_operation_limit();
        for width in widths {
            check_cancelled()?;
            let block_bytes = block_storage_bytes(width)?;
            let w = width as u64;
            let q = w - 1;
            source = add(source, block_bytes)?;
            graph = add(graph, 64 * q)?;
            // Every original pair may be absent from the numeric CSR, even when its
            // reconstructed coefficient is zero. Duplicate pairs only over-reserve.
            missing = add(missing, 512 * w * w)?;
            scratch = scratch.max(64 * w * w + 128 * w);
            cache_entry = cache_entry.max(8 * w * w + 256);
            blocks = add(blocks, 1)?;
            // Initial normalization; comparisons against at most32 cached blocks;
            // perturbation verification; fresh local LDL/outward proof; reconstruction;
            // finishing all possible missing coefficients. The cache never proves equality.
            operations = add(
                operations,
                w * w
                    + 3 * LOCAL_PROOF_CACHE_CAPACITY as u64 * q * q
                    + 1
                    + 12 * q * q
                    + 16 * q
                    + 32
                    + 6 * w * w * w
                    + 12 * q * q
                    + 4 * w * w
                    + 16 * w * w,
            )?;
        }
        let vertices = u64::try_from(original_vertices).map_err(|_| EnergyError::Arithmetic)?;
        operations = add(operations, mul(vertices, 16)?)?;
        let retained = add(add(source, projection_bytes)?, mul(vertices, 80)?)?;
        let additional_resident_bytes = [
            retained,
            graph,
            missing,
            scratch,
            mul(cache_entry, blocks.min(LOCAL_PROOF_CACHE_CAPACITY as u64))?,
        ]
        .into_iter()
        .try_fold(0, add)?;
        Ok(Self {
            admission: EnergyAdmission {
                dimension: matrix.dimension(),
                operations,
                resident: add(matrix.energy_resident_limit(), additional_resident_bytes)?,
            },
            additional_resident_bytes,
        })
    }
    pub fn additional_resident_bytes(self) -> u64 {
        self.additional_resident_bytes
    }
    pub fn resident_bytes(self) -> u64 {
        self.admission.resident
    }
    pub fn scalar_operations(self) -> u64 {
        self.admission.operations
    }
    pub(super) fn admission(self) -> EnergyAdmission {
        self.admission
    }
}
fn add(a: u64, b: u64) -> Result<u64, EnergyError> {
    a.checked_add(b).ok_or(EnergyError::Arithmetic)
}
fn mul(a: u64, b: u64) -> Result<u64, EnergyError> {
    a.checked_mul(b).ok_or(EnergyError::Arithmetic)
}

#[cfg(test)]
mod tests;
