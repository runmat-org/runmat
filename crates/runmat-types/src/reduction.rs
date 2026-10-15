/// Treatment of missing numeric observations in a reduction.
///
/// Public option spellings are decoded only at the language boundary. This
/// identity also crosses the provider workload boundary without re-parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReductionNaN {
    Include,
    Omit,
}

mod fusion_contract;
pub use fusion_contract::*;
mod fusion_work;
pub use fusion_work::*;
mod sum_work;
pub use sum_work::*;

/// Public entrypoint whose already-admitted numerical work uses the sum
/// reduction. The alias retains its own binding, errors and feedback identity.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SumInvocation {
    #[default]
    Sum,
    NanSum,
}

impl SumInvocation {
    pub fn is_sum(&self) -> bool {
        matches!(self, Self::Sum)
    }
}

/// Requested output-class policy for the ordinary numerical sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SumOutputClass {
    Default,
    Double,
    Native,
}

impl SumOutputClass {
    pub fn resolve_numeric_class(self, input: crate::NumericClass) -> crate::NumericClass {
        match self {
            Self::Native => input,
            Self::Double => crate::NumericClass::Double,
            Self::Default if input == crate::NumericClass::Single => crate::NumericClass::Single,
            Self::Default => crate::NumericClass::Double,
        }
    }
}

impl ReductionNaN {
    pub fn from_public_option(option: &str) -> Option<Self> {
        if option.eq_ignore_ascii_case("omitnan") || option.eq_ignore_ascii_case("omitmissing") {
            Some(Self::Omit)
        } else if option.eq_ignore_ascii_case("includenan")
            || option.eq_ignore_ascii_case("includemissing")
        {
            Some(Self::Include)
        } else {
            None
        }
    }
}
