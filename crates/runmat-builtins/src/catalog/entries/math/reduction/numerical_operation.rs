//! Identities used by sampled-data and scalar product/count execution owners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalReductionKind {
    Product,
    CountNonzero,
    Difference,
    Gradient,
    Trapezoid,
    CumulativeTrapezoid,
    NonNanMaximum,
}
impl NumericalReductionKind {
    pub fn catalog_entry(self) -> &'static crate::BuiltinCatalogEntry {
        match self {
            Self::Product => &super::PROD_CATALOG_ENTRY,
            Self::CountNonzero => &super::NNZ_CATALOG_ENTRY,
            Self::Difference => &super::DIFF_CATALOG_ENTRY,
            Self::Gradient => &super::GRADIENT_CATALOG_ENTRY,
            Self::Trapezoid => &super::TRAPZ_CATALOG_ENTRY,
            Self::CumulativeTrapezoid => &super::CUMTRAPZ_CATALOG_ENTRY,
            Self::NonNanMaximum => &super::NANMAX_CATALOG_ENTRY,
        }
    }
    pub fn internal_error(self) -> &'static crate::BuiltinErrorDescriptor {
        match self {
            Self::Product => &super::PROD_ERROR_INTERNAL,
            Self::CountNonzero => &super::NNZ_ERROR_INTERNAL,
            Self::Difference => &super::DIFF_ERROR_INTERNAL,
            Self::Gradient => &super::GRADIENT_ERROR_INTERNAL,
            Self::Trapezoid => &super::TRAPZ_ERROR_INTERNAL,
            Self::CumulativeTrapezoid => &super::CUMTRAPZ_ERROR_INTERNAL,
            Self::NonNanMaximum => &super::MAX_ERROR_INTERNAL,
        }
    }
    pub fn argument_error(self) -> &'static crate::BuiltinErrorDescriptor {
        match self {
            Self::Product => &super::PROD_ERROR_INVALID_ARGUMENT,
            Self::CountNonzero => &super::NNZ_ERROR_INVALID_ARGUMENT,
            Self::Difference => &super::DIFF_ERROR_INVALID_ARGUMENT,
            Self::Gradient => &super::GRADIENT_ERROR_INVALID_ARGUMENT,
            Self::Trapezoid => &super::TRAPZ_ERROR_INVALID_ARGUMENT,
            Self::CumulativeTrapezoid => &super::CUMTRAPZ_ERROR_INVALID_ARGUMENT,
            Self::NonNanMaximum => &super::MAX_ERROR_INVALID_ARGUMENT,
        }
    }
    pub fn input_error(self) -> &'static crate::BuiltinErrorDescriptor {
        match self {
            Self::Product => &super::PROD_ERROR_INVALID_INPUT,
            Self::CountNonzero => &super::NNZ_ERROR_INVALID_INPUT,
            Self::Difference => &super::DIFF_ERROR_INVALID_INPUT,
            Self::Gradient => &super::GRADIENT_ERROR_INVALID_INPUT,
            Self::Trapezoid => &super::TRAPZ_ERROR_INVALID_INPUT,
            Self::CumulativeTrapezoid => &super::CUMTRAPZ_ERROR_INVALID_INPUT,
            Self::NonNanMaximum => &super::MAX_ERROR_INVALID_INPUT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductOutputClass {
    Default,
    Double,
    Native,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductControl {
    All,
    Missing(super::ReductionNaN),
    Output(ProductOutputClass),
}
impl ProductControl {
    pub fn from_public_option(text: &str) -> Option<Self> {
        Self::from_public_characters(text.chars())
    }
    pub fn from_public_characters(
        text: impl DoubleEndedIterator<Item = char> + Clone,
    ) -> Option<Self> {
        Self::from_public_units(text.map(u32::from))
    }
    /// Recognize borrowed UTF-16 controls without allocating or normalizing raw units.
    pub fn from_public_code_units(
        text: impl DoubleEndedIterator<Item = u16> + Clone,
    ) -> Option<Self> {
        Self::from_public_units(text.map(u32::from))
    }
    fn from_public_units(mut text: impl DoubleEndedIterator<Item = u32> + Clone) -> Option<Self> {
        while text
            .clone()
            .next()
            .is_some_and(|unit| char::from_u32(unit).is_some_and(char::is_whitespace))
        {
            text.next();
        }
        while text
            .clone()
            .next_back()
            .is_some_and(|unit| char::from_u32(unit).is_some_and(char::is_whitespace))
        {
            text.next_back();
        }
        [
            ("all", Self::All),
            ("omitnan", Self::Missing(super::ReductionNaN::Omit)),
            ("omitmissing", Self::Missing(super::ReductionNaN::Omit)),
            ("includenan", Self::Missing(super::ReductionNaN::Include)),
            (
                "includemissing",
                Self::Missing(super::ReductionNaN::Include),
            ),
            ("default", Self::Output(ProductOutputClass::Default)),
            ("double", Self::Output(ProductOutputClass::Double)),
            ("native", Self::Output(ProductOutputClass::Native)),
        ]
        .into_iter()
        .find_map(|(name, control)| {
            text.clone()
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
}
