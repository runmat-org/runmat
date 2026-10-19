//! Bounded union for canonical dense topology indices, preserving streaming admission.
use super::{add, check_cancelled, reserve, slots, TopologyStorageError};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DenseUnionError {
    Storage(TopologyStorageError),
    Domain,
    Limit,
}
impl From<TopologyStorageError> for DenseUnionError {
    fn from(error: TopologyStorageError) -> Self {
        Self::Storage(error)
    }
}

pub(crate) struct DenseIndexUnion {
    domain: usize,
    maximum: usize,
    bits: Vec<u64>,
    indices: Vec<usize>,
}
impl DenseIndexUnion {
    pub(crate) fn resident_bytes(
        domain: usize,
        maximum: usize,
    ) -> Result<u64, TopologyStorageError> {
        add(
            slots::<u64>(domain.div_ceil(64))?,
            slots::<usize>(domain.min(maximum))?,
        )
    }
    pub(crate) fn new(domain: usize, maximum: usize) -> Result<Self, TopologyStorageError> {
        Self::resident_bytes(domain, maximum)?;
        let mut bits = reserve(domain.div_ceil(64))?;
        bits.resize(domain.div_ceil(64), 0);
        Ok(Self {
            domain,
            maximum: domain.min(maximum),
            bits,
            indices: reserve(domain.min(maximum))?,
        })
    }
    pub(crate) fn collect(
        mut self,
        source: impl Iterator<Item = usize>,
    ) -> Result<Vec<usize>, DenseUnionError> {
        for (ordinal, index) in source.enumerate() {
            if ordinal.is_multiple_of(1024) {
                check_cancelled()?;
            }
            if index >= self.domain {
                return Err(DenseUnionError::Domain);
            }
            let mask = 1u64 << (index % 64);
            let word = &mut self.bits[index / 64];
            if *word & mask == 0 {
                if self.indices.len() == self.maximum {
                    return Err(DenseUnionError::Limit);
                }
                *word |= mask;
                self.indices.push(index);
            }
        }
        check_cancelled()?;
        self.indices.sort_unstable();
        check_cancelled()?;
        Ok(self.indices)
    }
}
#[cfg(test)]
mod tests;
