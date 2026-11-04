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
        if exponent >= 63 {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
                Some(context.constraint_index),
                "dyadic segment parameter exhausted its exact integer range",
            ));
        }
        // The unreduced sum may need one extra bit even when the resulting
        // parameter fits the admitted representation. Reduce before admission.
        let left = u128::from(self.numerator) << (exponent - self.exponent);
        let right = u128::from(right.numerator) << (exponent - right.exponent);
        let mut numerator = left + right;
        let mut exponent = exponent + 1;
        while exponent > 0 && numerator.is_multiple_of(2) {
            numerator /= 2;
            exponent -= 1;
        }
        if exponent >= 63 {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
                Some(context.constraint_index),
                format!("dyadic midpoint requires exponent {exponent} after exact reduction"),
            ));
        }
        Ok(Self {
            identity: StableDigest::ZERO,
            numerator: u64::try_from(numerator).map_err(|_| {
                error(
                    DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
                    Some(context.constraint_index),
                    "dyadic midpoint numerator is out of range",
                )
            })?,
            exponent,
        })
    }

    #[cfg(test)]
    pub(super) fn near_parameter(
        self,
        right: Self,
        parameter: f64,
        context: SegmentContext,
    ) -> Result<Self, DelaunaySegmentRecoveryError> {
        match self.near_parameter_with_clearance(right, parameter, &mut |_| Ok(true))? {
            Some(candidate) => Ok(candidate),
            None => self.midpoint(right, context),
        }
    }

    #[cfg(test)]
    pub(super) fn near_parameter_with_clearance(
        self,
        right: Self,
        parameter: f64,
        admit: &mut impl FnMut(Self) -> Result<bool, DelaunaySegmentRecoveryError>,
    ) -> Result<Option<Self>, DelaunaySegmentRecoveryError> {
        self.search_near_parameter(right, parameter, false, admit)
    }

    /// Keep a verified coarse bracket, then approach the geometric preference
    /// on the admitted exact integer grid. Clearance alone must not turn a
    /// coarse rounding hint into a drifting acute-corner protection radius.
    pub(super) fn closest_parameter_with_clearance(
        self,
        right: Self,
        parameter: f64,
        neighbors: bool,
        admit: &mut impl FnMut(Self) -> Result<bool, DelaunaySegmentRecoveryError>,
    ) -> Result<Option<Self>, DelaunaySegmentRecoveryError> {
        let Some(seed) = self.search_near_parameter(right, parameter, neighbors, admit)? else {
            return Ok(None);
        };
        let grid = |node: Self| node.numerator << (62 - node.exponent);
        let canonical = |mut numerator: u64| {
            let mut exponent = 62;
            while exponent > 0 && numerator.is_multiple_of(2) {
                numerator /= 2;
                exponent -= 1;
            }
            Self {
                identity: StableDigest::ZERO,
                numerator,
                exponent,
            }
        };
        let target = (parameter * (1_u64 << 62) as f64).round() as u64;
        let mut accepted = grid(seed);
        if target <= grid(self) || target >= grid(right) || target == accepted {
            return Ok(Some(seed));
        }
        if admit(canonical(target))? {
            return Ok(Some(canonical(target)));
        }
        let mut rejected = target;
        while accepted.abs_diff(rejected) > 1 {
            let midpoint = accepted.min(rejected) + accepted.abs_diff(rejected) / 2;
            if admit(canonical(midpoint))? {
                accepted = midpoint;
            } else {
                rejected = midpoint;
            }
        }
        Ok(Some(canonical(accepted)))
    }

    pub(super) fn search_near_parameter(
        self,
        right: Self,
        parameter: f64,
        neighbors: bool,
        admit: &mut impl FnMut(Self) -> Result<bool, DelaunaySegmentRecoveryError>,
    ) -> Result<Option<Self>, DelaunaySegmentRecoveryError> {
        let interval = right.parameter()? - self.parameter()?;
        if parameter.is_finite() && (0.0..=1.0).contains(&parameter) && interval > 0.0 {
            // Begin with the inexpensive coarse geometric hint. Finer grids
            // are admitted only by the caller's exact clearance predicate;
            // fixed rounding precision never confers physical acceptance.
            let mut previous = None;
            for exponent in 0..63_u8 {
                let denominator = 1_u64 << exponent;
                if 1.0 / denominator as f64 > interval / 256.0 {
                    continue;
                }
                let rounded = (parameter * denominator as f64).round() as u64;
                // A radius bound may exclude the nearest grid point. Test its
                // two adjacent representable points before refining the grid.
                for mut numerator in [rounded, rounded.saturating_sub(1), rounded + 1]
                    .into_iter()
                    .take(if neighbors { 3 } else { 1 })
                {
                    let common = exponent.max(self.exponent).max(right.exponent);
                    let aligned = numerator << (common - exponent);
                    let left = self.numerator << (common - self.exponent);
                    let right = right.numerator << (common - right.exponent);
                    if aligned > left && aligned < right {
                        let mut exponent = exponent;
                        while exponent > 0 && numerator.is_multiple_of(2) {
                            numerator /= 2;
                            exponent -= 1;
                        }
                        let candidate = Self {
                            identity: StableDigest::ZERO,
                            numerator,
                            exponent,
                        };
                        if previous != Some((numerator, exponent)) {
                            previous = Some((numerator, exponent));
                            if admit(candidate)? {
                                return Ok(Some(candidate));
                            }
                        }
                    }
                }
            }
        }
        Ok(None)
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
