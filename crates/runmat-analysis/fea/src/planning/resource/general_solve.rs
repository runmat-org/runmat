//! Complete original GENERAL controls and shared solver/proof resource demand.
//! Decoded inventories are claims; planning and actual execution rederive them.
use super::{ByteCount, Count};
use crate::{
    algebra::{RealSparseMatrix, SparsePatternStatistics},
    solve::general::{GeneralSolveEnvelope, GeneralSolveError, GeneralSolvePolicy},
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct GeneralSolveResourceControls(GeneralSolvePolicy);
// Construction and decoding exclude NaN/infinite controls, so float equality
// is reflexive in this immutable type. Raw GeneralSolvePolicy stays unchanged.
impl Eq for GeneralSolveResourceControls {}
impl TryFrom<GeneralSolvePolicy> for GeneralSolveResourceControls {
    type Error = GeneralSolveError;
    fn try_from(policy: GeneralSolvePolicy) -> Result<Self, Self::Error> {
        policy.validate()?;
        Ok(Self(policy))
    }
}
impl<'de> Deserialize<'de> for GeneralSolveResourceControls {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::try_from(GeneralSolvePolicy::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}
impl GeneralSolveResourceControls {
    pub fn policy(self) -> GeneralSolvePolicy {
        self.0
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralSparseSolveResourceDemand {
    dimension: Count,
    nonzero_upper_bound: Count,
    maximum_row_nonzeros: Count,
    controls: GeneralSolveResourceControls,
    maximum_solves: Count,
    admission: GeneralSolveEnvelope,
}
impl GeneralSparseSolveResourceDemand {
    pub fn derive(
        support: SparsePatternStatistics,
        policy: GeneralSolvePolicy,
        maximum_solves: Count,
        admission: GeneralSolveEnvelope,
    ) -> Result<Self, GeneralSolveError> {
        if support.rows != support.columns {
            return Err(GeneralSolveError::Invalid);
        }
        let value = Self {
            dimension: Count(support.rows as u64),
            nonzero_upper_bound: Count(support.nonzeros as u64),
            maximum_row_nonzeros: Count(support.maximum_row_nonzeros as u64),
            controls: policy.try_into()?,
            maximum_solves,
            admission,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(self) -> Result<(), GeneralSolveError> {
        if self.maximum_solves.0 == 0
            || self.dimension.0 == 0
            || self.nonzero_upper_bound.0 == 0
            || self.maximum_row_nonzeros.0 == 0
            || self.maximum_row_nonzeros.0 > self.dimension.0
            || self.maximum_row_nonzeros.0 > self.nonzero_upper_bound.0
            || self.nonzero_upper_bound.0
                > self
                    .dimension
                    .0
                    .checked_mul(self.dimension.0)
                    .ok_or(GeneralSolveError::Overflow)?
        {
            return Err(GeneralSolveError::Invalid);
        }
        let required = GeneralSolveEnvelope::for_counts(
            usize::try_from(self.dimension.0).map_err(|_| GeneralSolveError::Overflow)?,
            usize::try_from(self.nonzero_upper_bound.0).map_err(|_| GeneralSolveError::Overflow)?,
            self.controls.policy(),
        )?;
        if required.resident_bytes > self.admission.resident_bytes
            || required.scalar_operations > self.admission.scalar_operations
        {
            return Err(GeneralSolveError::Admission);
        }
        self.scalar_operations_upper_bound()?;
        Ok(())
    }
    /// Recheck original numerical controls and actual CSR against planned support;
    /// the solver separately constructs a fresh factorization certificate.
    pub fn admit_operator(
        self,
        matrix: &RealSparseMatrix,
        policy: GeneralSolvePolicy,
    ) -> Result<GeneralSolveEnvelope, GeneralSolveError> {
        self.validate()?;
        let width = matrix
            .pattern()
            .row_offsets()
            .windows(2)
            .map(|r| r[1] - r[0])
            .max()
            .unwrap_or(0);
        if policy != self.controls.policy()
            || matrix.rows() != matrix.columns()
            || matrix.rows() as u64 != self.dimension.0
            || matrix.values().len() as u64 > self.nonzero_upper_bound.0
            || width > self.maximum_row_nonzeros.0
        {
            return Err(GeneralSolveError::Admission);
        }
        let required = GeneralSolveEnvelope::derive(matrix, policy)?;
        if required.resident_bytes > self.admission.resident_bytes
            || required.scalar_operations > self.admission.scalar_operations
        {
            return Err(GeneralSolveError::Admission);
        }
        Ok(self.admission)
    }
    pub fn dimension(self) -> Count {
        self.dimension
    }
    pub const fn workspace_bytes(self) -> ByteCount {
        ByteCount(self.admission.resident_bytes)
    }
    pub fn scalar_operations_upper_bound(self) -> Result<u64, GeneralSolveError> {
        self.maximum_solves
            .0
            .checked_mul(self.admission.scalar_operations)
            .ok_or(GeneralSolveError::Overflow)
    }
    pub fn maximum_solves(self) -> Count {
        self.maximum_solves
    }
    pub fn controls(self) -> GeneralSolveResourceControls {
        self.controls
    }
}
#[cfg(test)]
mod tests;
