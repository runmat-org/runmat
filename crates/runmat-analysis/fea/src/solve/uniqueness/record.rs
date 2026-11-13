//! Portable diagnostic claims, deliberately distinct from a construction-only proof.
//! Decoding validates shape/controls, not the inequality: discarded candidate columns
//! cannot be reconstructed from digests. Governed consumers must bind this record to
//! the exact producing result/receipt; independent cold algebra replay needs a witness.
use super::{SparseUniquenessError as Error, SparseUniquenessPolicy};
use crate::algebra::RightInverseCertificate;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum Claim {
    Verified {
        inverse_identity: Digest,
        defect_infinity_upper_bits: u64,
    },
    NoFreeUnknowns,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordData {
    operator_identity: Digest,
    dimension: usize,
    policy: SparseUniquenessPolicy,
    iterations: u64,
    claim: Claim,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SparseUniquenessRecord(RecordData);

impl<'de> Deserialize<'de> for SparseUniquenessRecord {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let record = Self(RecordData::deserialize(deserializer)?);
        record.validate().map_err(serde::de::Error::custom)?;
        Ok(record)
    }
}

impl SparseUniquenessRecord {
    pub fn from_certificate(
        policy: SparseUniquenessPolicy,
        certificate: &RightInverseCertificate,
        iterations: u64,
    ) -> Result<Self, Error> {
        let record = Self(RecordData {
            operator_identity: certificate.operator_identity(),
            dimension: certificate.dimension(),
            policy,
            iterations,
            claim: Claim::Verified {
                inverse_identity: certificate.inverse_identity(),
                defect_infinity_upper_bits: certificate.defect_infinity_upper().to_bits(),
            },
        });
        record.validate()?;
        Ok(record)
    }

    /// An empty original reduced operator has no free equations to certify.
    pub fn no_free_unknowns(
        operator_identity: Digest,
        policy: SparseUniquenessPolicy,
    ) -> Result<Self, Error> {
        let record = Self(RecordData {
            operator_identity,
            dimension: 0,
            policy,
            iterations: 0,
            claim: Claim::NoFreeUnknowns,
        });
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), Error> {
        self.0.policy.validate()?;
        let maximum = u64::try_from(self.0.dimension)
            .map_err(|_| Error::Overflow)?
            .checked_mul(
                u64::try_from(self.0.policy.maximum_iterations_per_column)
                    .map_err(|_| Error::Overflow)?,
            )
            .ok_or(Error::Overflow)?;
        if self.0.iterations > maximum {
            return Err(Error::Invalid);
        }
        match self.0.claim {
            Claim::NoFreeUnknowns if self.0.dimension == 0 && self.0.iterations == 0 => Ok(()),
            Claim::Verified {
                defect_infinity_upper_bits,
                ..
            } if self.0.dimension > 0 && {
                let q = f64::from_bits(defect_infinity_upper_bits);
                q.is_finite() && (0.0..1.0).contains(&q)
            } =>
            {
                Ok(())
            }
            _ => Err(Error::Invalid),
        }
    }

    pub const fn policy(&self) -> SparseUniquenessPolicy {
        self.0.policy
    }
    pub const fn dimension(&self) -> usize {
        self.0.dimension
    }
    pub const fn operator_identity(&self) -> Digest {
        self.0.operator_identity
    }
    pub const fn iterations(&self) -> u64 {
        self.0.iterations
    }
    pub fn defect_infinity_upper(&self) -> Option<f64> {
        match self.0.claim {
            Claim::Verified {
                defect_infinity_upper_bits,
                ..
            } => Some(f64::from_bits(defect_infinity_upper_bits)),
            Claim::NoFreeUnknowns => None,
        }
    }
}
