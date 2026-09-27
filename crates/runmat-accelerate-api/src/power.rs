//! Numerical-domain contract for power of real floating operands.
use serde::{Deserialize, Serialize};

/// Result storage selected from the actual paired operands, not a residual
/// tolerance or the fact that both inputs have real storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerResultDomain {
    Real,
    Complex,
}

impl PowerResultDomain {
    /// A finite negative base with a finite noninteger exponent uses the
    /// principal complex branch. NaN, infinities, and either signed zero do
    /// not establish that finite-domain condition.
    pub fn for_real_pair(base: f64, exponent: f64) -> Self {
        if base.is_finite() && base < 0.0 && exponent.is_finite() && exponent.fract() != 0.0 {
            Self::Complex
        } else {
            Self::Real
        }
    }

    pub fn join(self, other: Self) -> Self {
        if self == Self::Complex || other == Self::Complex {
            Self::Complex
        } else {
            Self::Real
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PowerResultDomain::{self, Complex, Real};
    use crate::NumericalDomainEvidence;

    #[test]
    fn older_workloads_decode_unknown_and_domains_round_trip() {
        let old: crate::ProviderWorkload =
            serde_json::from_str(r#"{"elements":4,"flops":null,"batch":null}"#).unwrap();
        assert_eq!(old.numerical_domain, NumericalDomainEvidence::Unknown);
        assert_eq!(old.requested_outputs, None);
        for requested_outputs in [None, Some(0), Some(1), Some(3)] {
            let workload = crate::ProviderWorkload {
                requested_outputs,
                ..old.clone()
            };
            let encoded = serde_json::to_string(&workload).unwrap();
            assert_eq!(
                serde_json::from_str::<crate::ProviderWorkload>(&encoded).unwrap(),
                workload
            );
        }
        for domain in [Real, Complex] {
            let workload = crate::ProviderWorkload {
                numerical_domain: NumericalDomainEvidence::RealPower(domain),
                ..old.clone()
            };
            let encoded = serde_json::to_string(&workload).unwrap();
            assert_eq!(
                serde_json::from_str::<crate::ProviderWorkload>(&encoded).unwrap(),
                workload
            );
        }
    }

    #[test]
    fn domain_uses_inputs_not_underflow_or_roundoff() {
        for (base, exponent, expected) in [
            (-1e-300, 0.5, Complex),
            (-1e-300, 1.5, Complex),
            (-4.0, 2.0, Real),
            (-4.0, -0.5, Complex),
            (-0.0, 0.5, Real),
            (0.0, -0.5, Real),
            (f64::NEG_INFINITY, 0.5, Real),
            (-1.0, f64::INFINITY, Real),
            (f64::NAN, 0.0, Real),
            (1.0, f64::NAN, Real),
        ] {
            assert_eq!(PowerResultDomain::for_real_pair(base, exponent), expected);
        }
    }
}
