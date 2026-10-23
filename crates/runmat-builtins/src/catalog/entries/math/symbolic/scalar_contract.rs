//! Shared scalar symbolic identities and public control grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicScalarOperation {
    Precision,
    Integrate,
    Limit,
    Piecewise,
    Decimal,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicScalarOperand {
    Algebra,
    DecimalInput,
    Variable,
    VariableOrPoint,
    Point,
    Direction,
    Digits,
    Precision,
}
impl SymbolicScalarOperation {
    pub fn catalog_entry(self) -> &'static crate::BuiltinCatalogEntry {
        match self {
            Self::Precision => &super::DIGITS_CATALOG_ENTRY,
            Self::Integrate => &super::INT_CATALOG_ENTRY,
            Self::Limit => &super::LIMIT_CATALOG_ENTRY,
            Self::Piecewise => &super::PIECEWISE_CATALOG_ENTRY,
            Self::Decimal => &super::VPA_CATALOG_ENTRY,
        }
    }
    pub fn accepts_arity(self, count: usize) -> bool {
        match self {
            Self::Precision => count <= 1,
            Self::Integrate | Self::Limit => (1..=4).contains(&count),
            Self::Piecewise => count >= 2,
            Self::Decimal => (1..=2).contains(&count),
        }
    }
    pub fn operand(self, index: usize, count: usize) -> SymbolicScalarOperand {
        use SymbolicScalarOperand::*;
        match self {
            Self::Precision => Digits,
            Self::Decimal if index == 1 => Precision,
            Self::Decimal => DecimalInput,
            Self::Piecewise => Algebra,
            Self::Integrate if (count == 2 || count == 4) && index == 1 => Variable,
            Self::Limit if index == 3 => Direction,
            Self::Limit if index == 1 && count == 2 => VariableOrPoint,
            Self::Limit if index == 1 => Variable,
            Self::Limit if index == 2 => Point,
            _ => Algebra,
        }
    }
    pub fn arity_error(self) -> &'static crate::BuiltinErrorDescriptor {
        let index = match self {
            Self::Integrate => 4,
            Self::Limit => 5,
            _ => 0,
        };
        &self.catalog_entry().descriptor.errors[index]
    }
    pub fn operand_error(
        self,
        index: usize,
        count: usize,
    ) -> &'static crate::BuiltinErrorDescriptor {
        use SymbolicScalarOperand::*;
        let error = match self {
            Self::Precision => 1,
            Self::Decimal => {
                if index == 0 {
                    1
                } else {
                    2
                }
            }
            Self::Piecewise => 0,
            Self::Integrate => {
                if index == 0 {
                    0
                } else if self.operand(index, count) == Variable {
                    1
                } else {
                    2
                }
            }
            Self::Limit => match self.operand(index, count) {
                Variable => 1,
                Point | VariableOrPoint => 2,
                Direction => 3,
                _ => 0,
            },
        };
        &self.catalog_entry().descriptor.errors[error]
    }
}
pub const SYMBOLIC_DEFAULT_DIGITS: usize = 32;
pub const SYMBOLIC_MAX_DIGITS: usize = 100_000_000;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicPrecisionKind {
    Default,
    Decimal,
}
impl SymbolicPrecisionKind {
    pub const fn minimum(self) -> usize {
        match self {
            Self::Default => 2,
            Self::Decimal => 1,
        }
    }
    pub fn accepts(self, value: usize) -> bool {
        (self.minimum()..=SYMBOLIC_MAX_DIGITS).contains(&value)
    }
    pub fn rounded(self, value: f64) -> Option<usize> {
        let value = value.round();
        (value.is_finite() && value >= self.minimum() as f64 && value <= SYMBOLIC_MAX_DIGITS as f64)
            .then_some(value as usize)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicLimitDirection {
    TwoSided,
    Left,
    Right,
}
impl SymbolicLimitDirection {
    pub fn from_public_option(text: &str) -> Option<Self> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("left") || text == "-" {
            Some(Self::Left)
        } else if text.eq_ignore_ascii_case("right") || text == "+" {
            Some(Self::Right)
        } else if text.is_empty()
            || text.eq_ignore_ascii_case("both")
            || text.eq_ignore_ascii_case("two-sided")
            || text.eq_ignore_ascii_case("twosided")
        {
            Some(Self::TwoSided)
        } else {
            None
        }
    }
}
/// Recognized MATLAB options currently rejected by RunMat's scalar integrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicIntegrationOption {
    IgnoreAnalyticConstraints,
    PrincipalValue,
    IgnoreSpecialCases,
    Hold,
}
impl SymbolicIntegrationOption {
    pub fn from_public_characters(
        value: impl DoubleEndedIterator<Item = char> + Clone,
    ) -> Option<Self> {
        Self::from_public_units(value.map(u32::from))
    }
    /// Recognize borrowed UTF-16 controls without allocating or normalizing raw units.
    pub fn from_public_code_units(
        value: impl DoubleEndedIterator<Item = u16> + Clone,
    ) -> Option<Self> {
        Self::from_public_units(value.map(u32::from))
    }
    fn from_public_units(mut value: impl DoubleEndedIterator<Item = u32> + Clone) -> Option<Self> {
        while value
            .clone()
            .next()
            .is_some_and(|unit| char::from_u32(unit).is_some_and(char::is_whitespace))
        {
            value.next();
        }
        while value
            .clone()
            .next_back()
            .is_some_and(|unit| char::from_u32(unit).is_some_and(char::is_whitespace))
        {
            value.next_back();
        }
        [
            ("ignoreanalyticconstraints", Self::IgnoreAnalyticConstraints),
            ("principalvalue", Self::PrincipalValue),
            ("ignorespecialcases", Self::IgnoreSpecialCases),
            ("hold", Self::Hold),
        ]
        .into_iter()
        .find_map(|(name, option)| {
            value
                .clone()
                .map(|unit| {
                    if (65..=90).contains(&unit) {
                        unit + 32
                    } else {
                        unit
                    }
                })
                .eq(name.chars().map(u32::from))
                .then_some(option)
        })
    }
    pub fn from_public_option(text: &str) -> Option<Self> {
        Self::from_public_characters(text.chars())
    }
}
