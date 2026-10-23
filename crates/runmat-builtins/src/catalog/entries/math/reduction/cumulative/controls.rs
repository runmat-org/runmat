use runmat_types::LiteralValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CumulativeKind {
    Sum,
    Product,
    Minimum,
    Maximum,
}
impl CumulativeKind {
    pub const fn has_indices(self) -> bool {
        matches!(self, Self::Minimum | Self::Maximum)
    }
    pub fn catalog_entry(self) -> &'static crate::BuiltinCatalogEntry {
        match self {
            Self::Sum => &super::CUMSUM_CATALOG_ENTRY,
            Self::Product => &super::CUMPROD_CATALOG_ENTRY,
            Self::Minimum => &super::CUMMIN_CATALOG_ENTRY,
            Self::Maximum => &super::CUMMAX_CATALOG_ENTRY,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CumulativeScanDirection {
    Forward,
    Reverse,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CumulativeMissingMode {
    Include,
    Omit,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CumulativeControl {
    Direction(CumulativeScanDirection),
    Missing(CumulativeMissingMode),
}
impl CumulativeControl {
    /// The public text boundary preserves the established trimmed,
    /// case-insensitive grammar. Execution retains the decoded identity.
    pub fn from_public_option(value: &str) -> Option<Self> {
        Self::from_public_characters(value.chars())
    }

    /// Decode borrowed UTF-8 or character storage without copying a control
    /// into an unadmitted String. Both public representations share one grammar.
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
            ("forward", Self::Direction(CumulativeScanDirection::Forward)),
            ("reverse", Self::Direction(CumulativeScanDirection::Reverse)),
            ("includenan", Self::Missing(CumulativeMissingMode::Include)),
            (
                "includemissing",
                Self::Missing(CumulativeMissingMode::Include),
            ),
            ("omitnan", Self::Missing(CumulativeMissingMode::Omit)),
            ("omitmissing", Self::Missing(CumulativeMissingMode::Omit)),
        ]
        .into_iter()
        .find_map(|(name, control)| {
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
                .then_some(control)
        })
    }
    pub fn from_literal(value: &LiteralValue) -> Option<Self> {
        match value {
            LiteralValue::String(text)
            | LiteralValue::Character(text)
            | LiteralValue::Keyword(text) => Self::from_public_option(text),
            _ => None,
        }
    }
}
