//! Validated time domain and polynomial variable for SISO transfer functions.
//!
//! The public `tf` object continues to serialize `Ts` as a scalar and
//! `Variable` as text. These types are the sole in-memory authority after
//! that boundary has been parsed; a model cannot hold contradictory numeric
//! sample-time and variable-domain fields.

use crate::BuiltinResult;

use super::{control_error, invalid_sample_time_identifier, invalid_variable_identifier};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TfVariable {
    S,
    P,
    Z,
    Q,
    ZInverse,
    QInverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TfPolynomialBasis {
    Continuous,
    DiscreteForward,
    DiscreteInverse,
}

impl TfVariable {
    pub fn parse(value: &str, builtin: &'static str) -> BuiltinResult<Self> {
        match value.trim() {
            "s" => Ok(Self::S),
            "p" => Ok(Self::P),
            "z" => Ok(Self::Z),
            "q" => Ok(Self::Q),
            "z^-1" => Ok(Self::ZInverse),
            "q^-1" => Ok(Self::QInverse),
            _ => Err(control_error(
                builtin,
                invalid_variable_identifier(builtin),
                format!(
                    "{builtin}: invalid Variable option: must be one of 's', 'p', 'z', 'q', 'z^-1', or 'q^-1'"
                ),
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::S => "s",
            Self::P => "p",
            Self::Z => "z",
            Self::Q => "q",
            Self::ZInverse => "z^-1",
            Self::QInverse => "q^-1",
        }
    }

    pub const fn basis(self) -> TfPolynomialBasis {
        match self {
            Self::S | Self::P => TfPolynomialBasis::Continuous,
            Self::Z | Self::Q => TfPolynomialBasis::DiscreteForward,
            Self::ZInverse | Self::QInverse => TfPolynomialBasis::DiscreteInverse,
        }
    }

    pub const fn is_discrete(self) -> bool {
        !matches!(self.basis(), TfPolynomialBasis::Continuous)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositiveSamplePeriod(f64);

impl PositiveSamplePeriod {
    pub const fn value(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TfSampleDomain {
    Continuous,
    Discrete(PositiveSamplePeriod),
    DiscreteUnspecified,
}

impl TfSampleDomain {
    pub fn parse(
        sample_time: f64,
        variable: TfVariable,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        if !sample_time.is_finite() || (sample_time < 0.0 && sample_time != -1.0) {
            return Err(control_error(
                builtin,
                invalid_sample_time_identifier(builtin),
                format!("{builtin}: sample time must be -1 or a finite non-negative scalar"),
            ));
        }
        if variable.basis() == TfPolynomialBasis::Continuous {
            return if sample_time == 0.0 {
                Ok(Self::Continuous)
            } else {
                Err(control_error(
                    builtin,
                    invalid_variable_identifier(builtin),
                    format!("{builtin}: continuous transfer-function variables require Ts = 0"),
                ))
            };
        }
        if sample_time == -1.0 {
            Ok(Self::DiscreteUnspecified)
        } else if sample_time > 0.0 {
            Ok(Self::Discrete(PositiveSamplePeriod(sample_time)))
        } else {
            Err(control_error(
                builtin,
                invalid_sample_time_identifier(builtin),
                format!("{builtin}: discrete transfer-function variables require a positive sample time"),
            ))
        }
    }

    pub const fn sample_time(self) -> f64 {
        match self {
            Self::Continuous => 0.0,
            Self::Discrete(period) => period.value(),
            Self::DiscreteUnspecified => -1.0,
        }
    }

    pub const fn is_discrete(self) -> bool {
        !matches!(self, Self::Continuous)
    }

    pub const fn period(self) -> Option<PositiveSamplePeriod> {
        match self {
            Self::Discrete(period) => Some(period),
            Self::Continuous | Self::DiscreteUnspecified => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_domain_is_derived_once_from_validated_variable_and_ts() {
        assert_eq!(
            TfSampleDomain::parse(0.0, TfVariable::S, "tf").unwrap(),
            TfSampleDomain::Continuous
        );
        assert_eq!(
            TfSampleDomain::parse(0.25, TfVariable::Z, "tf")
                .unwrap()
                .period()
                .unwrap()
                .value(),
            0.25
        );
        assert_eq!(
            TfSampleDomain::parse(-1.0, TfVariable::ZInverse, "tf").unwrap(),
            TfSampleDomain::DiscreteUnspecified
        );
        assert!(TfSampleDomain::parse(-1.0, TfVariable::S, "tf").is_err());
        assert!(TfSampleDomain::parse(0.0, TfVariable::Q, "tf").is_err());
    }
}
