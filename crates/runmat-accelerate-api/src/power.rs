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

/// Cost/query description, never allocation or execution authority. A live
/// preparation compares this description with its independently owned handle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerInputDescription {
    Live {
        physical: crate::GpuPhysicalDescriptor,
        view: crate::GpuTensorView,
    },
    /// Future dense contiguous allocation extent supplied by the owning
    /// allocator projection. This is not a physical descriptor or a lease.
    PlannedDense { requested_bytes: u64 },
}

impl PowerInputDescription {
    pub(crate) fn from_handle(handle: &crate::GpuTensorHandle) -> Self {
        Self::Live {
            physical: *handle.physical(),
            view: handle.view().clone(),
        }
    }
    pub(crate) fn matches(&self, handle: &crate::GpuTensorHandle) -> bool {
        match self {
            Self::Live { physical, view } => {
                *physical == *handle.physical() && *view == *handle.view()
            }
            Self::PlannedDense { requested_bytes } => {
                *requested_bytes == handle.physical().requested_bytes()
                    && matches!(handle.view().layout(), crate::GpuTensorLayout::Contiguous)
            }
        }
    }
    pub fn requested_bytes(&self) -> u64 {
        match self {
            Self::Live { physical, .. } => physical.requested_bytes(),
            Self::PlannedDense { requested_bytes } => *requested_bytes,
        }
    }
}

/// Exact scalar program plus optional physical source description. Unknown
/// backing remains None, especially for empty views retaining a large buffer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerScalarWorkload {
    pub control: PowerScalarControl,
    pub input: Option<PowerInputDescription>,
}

/// Certified algebraic cases of a Double exponent applied to a real Single
/// base. This describes an implementation, not permission to narrow a control
/// or to execute general mixed POWER in Single arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExactSingleBasePower {
    Zero,
    Identity,
    Square,
}

impl ExactSingleBasePower {
    pub fn for_control(control: PowerScalarControl) -> Option<Self> {
        let PowerScalarControl::DoubleBits(bits) = control else {
            return None;
        };
        match bits {
            0 | 0x8000_0000_0000_0000 => Some(Self::Zero),
            0x3ff0_0000_0000_0000 => Some(Self::Identity),
            0x4000_0000_0000_0000 => Some(Self::Square),
            _ => None,
        }
    }

    /// Binary64 numerical policy before Single publication. Promoting any
    /// finite binary32 base is exact; its square has at most 48 significand
    /// bits and stays within binary64's exponent range. No binary64 rounding
    /// occurs in this multiplication. NaNs precede exponent-zero semantics.
    pub fn evaluate(self, base: f32) -> f32 {
        if base.is_nan() {
            return f32::NAN;
        }
        match self {
            Self::Zero => 1.0,
            Self::Identity => base,
            Self::Square => (f64::from(base) * f64::from(base)) as f32,
        }
    }
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
    fn certified_single_base_family_retains_double_control_and_exception_policy() {
        use super::{ExactSingleBasePower as Exact, PowerScalarControl as Control};
        for (value, expected) in [
            (0.0_f64, Exact::Zero),
            (-0.0, Exact::Zero),
            (1.0, Exact::Identity),
            (2.0, Exact::Square),
        ] {
            assert_eq!(
                Exact::for_control(Control::DoubleBits(value.to_bits())),
                Some(expected)
            );
            assert!(Exact::for_control(Control::SingleBits((value as f32).to_bits())).is_none());
        }
        for value in [
            -2.0_f64,
            -1.0,
            0.5,
            3.0,
            f64::INFINITY,
            f64::NAN,
            f64::from_bits(2.0_f64.to_bits() + 1),
        ] {
            assert!(Exact::for_control(Control::DoubleBits(value.to_bits())).is_none());
        }
        for operation in [Exact::Zero, Exact::Identity, Exact::Square] {
            assert!(operation.evaluate(f32::NAN).is_nan());
        }
        assert_eq!(
            Exact::Identity.evaluate(-0.0).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(Exact::Identity.evaluate(f32::from_bits(1)).to_bits(), 1);
        assert_eq!(Exact::Square.evaluate(-0.0).to_bits(), 0);
        assert_eq!(Exact::Square.evaluate(f32::NEG_INFINITY), f32::INFINITY);
        // Square straddles the half-min-subnormal tie. The binary64 product
        // is exact, so this oracle does not duplicate the word shader.
        assert_eq!(
            Exact::Square.evaluate(f32::from_bits(0x1a000000)).to_bits(),
            0
        );
        assert_eq!(
            Exact::Square.evaluate(f32::from_bits(0x1a000001)).to_bits(),
            1
        );
    }

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
