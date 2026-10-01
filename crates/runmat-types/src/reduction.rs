/// Treatment of missing numeric observations in a reduction.
///
/// Public option spellings are decoded only at the language boundary. This
/// identity also crosses the provider workload boundary without re-parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReductionNaN {
    Include,
    Omit,
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
