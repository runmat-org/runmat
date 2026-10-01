/// The numerical reduction's treatment of missing numeric observations.
///
/// This is the owner shared by catalog admission and runtime execution. Public
/// option spellings are decoded at the argument boundary; aliases supply the
/// typed default instead of constructing a synthetic source argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
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

#[cfg(test)]
mod tests {
    use super::ReductionNaN;

    #[test]
    fn public_missing_policy_spellings_share_one_typed_identity() {
        for option in ["omitnan", "omitmissing", "OMITNAN"] {
            assert_eq!(
                ReductionNaN::from_public_option(option),
                Some(ReductionNaN::Omit)
            );
        }
        for option in ["includenan", "includemissing", "INCLUDEMISSING"] {
            assert_eq!(
                ReductionNaN::from_public_option(option),
                Some(ReductionNaN::Include)
            );
        }
        assert_eq!(ReductionNaN::from_public_option("native"), None);
    }
}
