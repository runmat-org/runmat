//! Canonical statistics-option field identities shared by catalog inference and runtime parsing.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatOptionField {
    Display,
    MaxFunEvals,
    MaxIter,
    TolBnd,
    TolFun,
    TolTypeFun,
    TolX,
    TolTypeX,
    GradObj,
    Jacobian,
    DerivStep,
    FunValCheck,
    Robust,
    RobustWgtFun,
    WgtFun,
    Tune,
    UseParallel,
    UseSubstreams,
    Streams,
    OutputFcn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatOptionValueKind {
    Display,
    OnOff,
    ToleranceType,
    RobustWeight,
    PositiveCount,
    PositiveScalar,
    PositiveArray,
    Boolean,
    Opaque,
}

impl StatOptionField {
    pub const ALL: [Self; 20] = [
        Self::Display,
        Self::MaxFunEvals,
        Self::MaxIter,
        Self::TolBnd,
        Self::TolFun,
        Self::TolTypeFun,
        Self::TolX,
        Self::TolTypeX,
        Self::GradObj,
        Self::Jacobian,
        Self::DerivStep,
        Self::FunValCheck,
        Self::Robust,
        Self::RobustWgtFun,
        Self::WgtFun,
        Self::Tune,
        Self::UseParallel,
        Self::UseSubstreams,
        Self::Streams,
        Self::OutputFcn,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Display => "Display",
            Self::MaxFunEvals => "MaxFunEvals",
            Self::MaxIter => "MaxIter",
            Self::TolBnd => "TolBnd",
            Self::TolFun => "TolFun",
            Self::TolTypeFun => "TolTypeFun",
            Self::TolX => "TolX",
            Self::TolTypeX => "TolTypeX",
            Self::GradObj => "GradObj",
            Self::Jacobian => "Jacobian",
            Self::DerivStep => "DerivStep",
            Self::FunValCheck => "FunValCheck",
            Self::Robust => "Robust",
            Self::RobustWgtFun => "RobustWgtFun",
            Self::WgtFun => "WgtFun",
            Self::Tune => "Tune",
            Self::UseParallel => "UseParallel",
            Self::UseSubstreams => "UseSubstreams",
            Self::Streams => "Streams",
            Self::OutputFcn => "OutputFcn",
        }
    }

    pub const fn value_kind(self) -> StatOptionValueKind {
        match self {
            Self::Display => StatOptionValueKind::Display,
            Self::FunValCheck | Self::GradObj | Self::Jacobian => StatOptionValueKind::OnOff,
            Self::TolTypeFun | Self::TolTypeX => StatOptionValueKind::ToleranceType,
            Self::RobustWgtFun => StatOptionValueKind::RobustWeight,
            Self::MaxFunEvals | Self::MaxIter => StatOptionValueKind::PositiveCount,
            Self::TolBnd | Self::TolFun | Self::TolX | Self::Tune => {
                StatOptionValueKind::PositiveScalar
            }
            Self::DerivStep => StatOptionValueKind::PositiveArray,
            Self::UseParallel | Self::UseSubstreams => StatOptionValueKind::Boolean,
            Self::Streams | Self::OutputFcn | Self::Robust | Self::WgtFun => {
                StatOptionValueKind::Opaque
            }
        }
    }

    pub fn resolve(name: &str) -> Option<Self> {
        if let Some(exact) = Self::ALL
            .into_iter()
            .find(|field| field.name().eq_ignore_ascii_case(name))
        {
            return Some(exact);
        }
        let needle = name.to_ascii_lowercase();
        let mut candidates = Self::ALL
            .into_iter()
            .filter(|field| field.name().to_ascii_lowercase().starts_with(&needle));
        let first = candidates.next()?;
        candidates.next().is_none().then_some(first)
    }
}

#[cfg(test)]
mod tests {
    use super::StatOptionField;

    #[test]
    fn exact_and_unique_prefix_names_are_typed() {
        assert_eq!(
            StatOptionField::resolve("MAXI"),
            Some(StatOptionField::MaxIter)
        );
        assert_eq!(
            StatOptionField::resolve("tolx"),
            Some(StatOptionField::TolX)
        );
        assert_eq!(StatOptionField::resolve("tol"), None);
        assert_eq!(StatOptionField::resolve(""), None);
    }

    #[test]
    fn names_remain_unique_case_insensitively() {
        let names = StatOptionField::ALL
            .into_iter()
            .map(|field| field.name().to_ascii_lowercase())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), StatOptionField::ALL.len());
        for field in StatOptionField::ALL {
            assert_eq!(StatOptionField::resolve(field.name()), Some(field));
        }
    }
}
