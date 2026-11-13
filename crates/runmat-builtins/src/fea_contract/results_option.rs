//! One semantic decoder for fea.results selector names. Public spelling is
//! normalized once at the boundary; both catalog inference and runtime query
//! construction retain the typed option thereafter.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResultsOption {
    IncludeFields,
    IncludeFieldValues,
    IncludeDiagnostics,
    DiagnosticCodes,
    IncludeModalResults,
    ModeIndices,
    IncludeTransientResults,
    TransientSnapshotIndices,
    IncludeNonlinearResults,
}

impl ResultsOption {
    pub fn decode(name: &str) -> Option<Self> {
        Self::from_normalized(&Self::normalized_key(name))
    }

    pub fn normalized_key(name: &str) -> String {
        super::options::normalize_option_key(name)
    }

    /// Use when the caller already applied the shared FEA option-key parser.
    pub fn from_normalized(name: &str) -> Option<Self> {
        Some(match name {
            "includefields" | "fields" => Self::IncludeFields,
            "includefieldvalues" | "fieldvalues" => Self::IncludeFieldValues,
            "includediagnostics" => Self::IncludeDiagnostics,
            "diagnosticcodes" => Self::DiagnosticCodes,
            "includemodalresults" => Self::IncludeModalResults,
            "modeindices" => Self::ModeIndices,
            "includetransientresults" => Self::IncludeTransientResults,
            "transientsnapshotindices" => Self::TransientSnapshotIndices,
            "includenonlinearresults" => Self::IncludeNonlinearResults,
            _ => return None,
        })
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::IncludeFields => "includefields",
            Self::IncludeFieldValues => "includefieldvalues",
            Self::IncludeDiagnostics => "includediagnostics",
            Self::DiagnosticCodes => "diagnosticcodes",
            Self::IncludeModalResults => "includemodalresults",
            Self::ModeIndices => "modeindices",
            Self::IncludeTransientResults => "includetransientresults",
            Self::TransientSnapshotIndices => "transientsnapshotindices",
            Self::IncludeNonlinearResults => "includenonlinearresults",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ResultsOption;

    #[test]
    fn aliases_and_punctuation_share_canonical_identity() {
        assert_eq!(
            ResultsOption::decode("FIELDS"),
            Some(ResultsOption::IncludeFields)
        );
        assert_eq!(
            ResultsOption::decode("Include-Fields"),
            Some(ResultsOption::IncludeFields)
        );
        assert_eq!(
            ResultsOption::decode("Field_Values"),
            Some(ResultsOption::IncludeFieldValues)
        );
        assert_eq!(
            ResultsOption::decode("include.field.values")
                .unwrap()
                .canonical_name(),
            "includefieldvalues"
        );
        assert_eq!(ResultsOption::decode("not-an-option"), None);
    }
}
