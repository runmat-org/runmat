use runmat_meshing_core::StableDigest;
use sha2::{Digest as _, Sha256};

use super::{error, DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind};

const STEINER_IDENTITY_DOMAIN: &[u8] = b"runmat-meshing-cdt-segment-steiner/v1\0";

#[derive(Clone, Copy)]
pub(super) struct SegmentContext {
    pub(super) constraint_index: u32,
    pub(super) first_identity: StableDigest,
    pub(super) last_identity: StableDigest,
    pub(super) first_coordinates: [f64; 3],
    pub(super) last_coordinates: [f64; 3],
}

#[derive(Clone, Copy)]
pub(super) struct DyadicNode {
    pub(super) identity: StableDigest,
    pub(super) numerator: u64,
    pub(super) exponent: u8,
}

impl DyadicNode {
    pub(super) fn endpoint(identity: StableDigest, last: bool) -> Self {
        Self {
            identity,
            numerator: u64::from(last),
            exponent: 0,
        }
    }

    pub(super) fn midpoint(
        self,
        right: Self,
        context: SegmentContext,
    ) -> Result<Self, DelaunaySegmentRecoveryError> {
        let exponent = self.exponent.max(right.exponent);
        if exponent >= 62 {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
                Some(context.constraint_index),
                "dyadic segment parameter exhausted its exact integer range",
            ));
        }
        let left = self.numerator << (exponent - self.exponent);
        let right = right.numerator << (exponent - right.exponent);
        let mut numerator = left + right;
        let mut exponent = exponent + 1;
        while exponent > 0 && numerator.is_multiple_of(2) {
            numerator /= 2;
            exponent -= 1;
        }
        Ok(Self {
            identity: StableDigest::ZERO,
            numerator,
            exponent,
        })
    }

    pub(super) fn with_identity(mut self, identity: StableDigest) -> Self {
        self.identity = identity;
        self
    }

    pub(super) fn parameter(self) -> Result<f64, DelaunaySegmentRecoveryError> {
        if self.exponent >= 63 || self.numerator > (1_u64 << self.exponent) {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
                None,
                "dyadic segment parameter is outside [0, 1]",
            ));
        }
        Ok(self.numerator as f64 / (1_u64 << self.exponent) as f64)
    }
}

pub(super) fn interpolate(first: [f64; 3], last: [f64; 3], parameter: f64) -> [f64; 3] {
    std::array::from_fn(|axis| first[axis] * (1.0 - parameter) + last[axis] * parameter)
}

pub(super) fn steiner_identity(context: SegmentContext, node: DyadicNode) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(STEINER_IDENTITY_DOMAIN);
    hasher.update(context.first_identity.bytes());
    hasher.update(context.last_identity.bytes());
    hasher.update(node.numerator.to_be_bytes());
    hasher.update([node.exponent]);
    StableDigest::from_bytes(hasher.finalize().into())
}
