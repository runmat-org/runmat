//! Numerical-domain contract for power of real floating operands.
use runmat_types::NumericClass;
use serde::{Deserialize, Serialize};

/// Result storage selected from the actual paired operands, not a residual
/// tolerance or the fact that both inputs have real storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerResultDomain {
    Real,
    Complex,
}

/// Physical arithmetic and publication precision for a real-input power.
/// Mixed operands evaluate in Double, then convert each real or complex
/// component once to the public Single result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealPowerArithmeticMode {
    NativeSingle,
    NativeDouble,
    MixedDoubleThenSingle { single_operand: RealPowerOperand },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealPowerOperand {
    Base,
    Exponent,
}

/// Original scalar payload at a POWER boundary. In particular, an exactly
/// representable Double is not interchangeable with a Single control.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerScalarControl {
    SingleBits(u32),
    DoubleBits(u64),
}

impl PowerScalarControl {
    pub fn precision(self) -> crate::ProviderPrecision {
        match self {
            Self::SingleBits(_) => crate::ProviderPrecision::F32,
            Self::DoubleBits(_) => crate::ProviderPrecision::F64,
        }
    }

    pub fn value(self) -> f64 {
        match self {
            Self::SingleBits(bits) => f64::from(f32::from_bits(bits)),
            Self::DoubleBits(bits) => f64::from_bits(bits),
        }
    }
}

impl RealPowerArithmeticMode {
    pub fn for_classes(base: NumericClass, exponent: NumericClass) -> Option<Self> {
        match (base, exponent) {
            (NumericClass::Single, NumericClass::Single) => Some(Self::NativeSingle),
            (NumericClass::Double, NumericClass::Double) => Some(Self::NativeDouble),
            (NumericClass::Single, NumericClass::Double) => Some(Self::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Base,
            }),
            (NumericClass::Double, NumericClass::Single) => Some(Self::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Exponent,
            }),
            _ => None,
        }
    }

    pub fn output_class(self) -> NumericClass {
        match self {
            Self::NativeSingle | Self::MixedDoubleThenSingle { .. } => NumericClass::Single,
            Self::NativeDouble => NumericClass::Double,
        }
    }
}

impl PowerResultDomain {
    /// A finite integral exponent keeps every real floating base in the real
    /// result domain, including negative finite bases. This is a universal
    /// fact about the operation, independent of any sampled base.
    pub fn real_for_any_real_base(exponent: f64) -> bool {
        exponent.is_finite() && exponent.fract() == 0.0
    }

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
    fn mixed_publication_is_single_for_either_operand_position() {
        use super::{RealPowerArithmeticMode as Mode, RealPowerOperand};
        use runmat_types::NumericClass::{Double, Single};
        assert_eq!(
            Mode::for_classes(Single, Double),
            Some(Mode::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Base
            })
        );
        assert_eq!(
            Mode::for_classes(Double, Single),
            Some(Mode::MixedDoubleThenSingle {
                single_operand: RealPowerOperand::Exponent
            })
        );
        assert_eq!(
            Mode::for_classes(Double, Single).unwrap().output_class(),
            Single
        );
        assert_eq!(
            Mode::for_classes(Double, Double).unwrap().output_class(),
            Double
        );
    }

    #[test]
    fn older_workloads_decode_unknown_and_domains_round_trip() {
        let old: crate::ProviderWorkload =
            serde_json::from_str(r#"{"elements":4,"flops":null,"batch":null}"#).unwrap();
        assert_eq!(old.numerical_domain, NumericalDomainEvidence::Unknown);
        assert_eq!(old.requested_outputs, None);
        assert_eq!(old.real_power_mode, None);
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
        for mode in [
            super::RealPowerArithmeticMode::NativeSingle,
            super::RealPowerArithmeticMode::NativeDouble,
            super::RealPowerArithmeticMode::MixedDoubleThenSingle {
                single_operand: super::RealPowerOperand::Base,
            },
            super::RealPowerArithmeticMode::MixedDoubleThenSingle {
                single_operand: super::RealPowerOperand::Exponent,
            },
        ] {
            let workload = crate::ProviderWorkload {
                real_power_mode: Some(mode),
                numerical_domain: NumericalDomainEvidence::RealPower(Complex),
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

    #[test]
    fn universal_real_exponent_is_integral_and_finite() {
        for exponent in [-3.0, -0.0, 0.0, 2.0, 16_777_217.0] {
            assert!(PowerResultDomain::real_for_any_real_base(exponent));
            assert_eq!(PowerResultDomain::for_real_pair(-2.0, exponent), Real);
        }
        for exponent in [0.5, -1.5, f64::NAN, f64::INFINITY] {
            assert!(!PowerResultDomain::real_for_any_real_base(exponent));
        }
    }
}
