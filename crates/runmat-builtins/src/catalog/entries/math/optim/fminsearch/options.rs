/// Legacy fminsearch controls, deliberately separate from optimoptions solver identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FminsearchOptionName {
    Display,
    FunValCheck,
    MaxFunEvals,
    MaxIter,
    OutputFcn,
    PlotFcns,
    TolFun,
    TolX,
}
impl FminsearchOptionName {
    pub const ALL: [Self; 8] = [
        Self::Display,
        Self::FunValCheck,
        Self::MaxFunEvals,
        Self::MaxIter,
        Self::OutputFcn,
        Self::PlotFcns,
        Self::TolFun,
        Self::TolX,
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Display => "Display",
            Self::FunValCheck => "FunValCheck",
            Self::MaxFunEvals => "MaxFunEvals",
            Self::MaxIter => "MaxIter",
            Self::OutputFcn => "OutputFcn",
            Self::PlotFcns => "PlotFcns",
            Self::TolFun => "TolFun",
            Self::TolX => "TolX",
        }
    }
    pub fn from_public(name: &str) -> Option<Self> {
        if name.is_empty() {
            return None;
        }
        if let Some(field) = Self::ALL
            .into_iter()
            .find(|field| field.name().eq_ignore_ascii_case(name))
        {
            return Some(field);
        }
        let mut matches = Self::ALL.into_iter().filter(|field| {
            field
                .name()
                .get(..name.len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(name))
        });
        let value = matches.next()?;
        matches.next().is_none().then_some(value)
    }
    pub const fn is_callback(self) -> bool {
        matches!(self, Self::OutputFcn | Self::PlotFcns)
    }
    pub fn ambiguous(name: &str) -> bool {
        !name.is_empty()
            && !Self::ALL
                .into_iter()
                .any(|field| field.name().eq_ignore_ascii_case(name))
            && Self::ALL
                .into_iter()
                .filter(|field| {
                    field
                        .name()
                        .get(..name.len())
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(name))
                })
                .take(2)
                .count()
                == 2
    }
}
/// The documented dimension-dependent default, not a general expression evaluator.
pub const FMINSEARCH_DEFAULT_BUDGET: &str = "200*numberOfVariables";
