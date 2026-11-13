//! Public normalization keyword identities shared by inference and execution.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NormalizeKeyword {
    ZScore,
    Norm,
    Scale,
    Range,
    Center,
    MedianIqr,
    DataVariables,
    ReplaceValues,
    Std,
    Robust,
    Mean,
    Median,
    Mad,
    Iqr,
    First,
    Other(String),
}

impl NormalizeKeyword {
    /// Decode one source-facing spelling; all later semantic decisions use the enum.
    pub fn from_public_text(text: &str) -> Self {
        match text.to_ascii_lowercase().as_str() {
            "zscore" => Self::ZScore,
            "norm" => Self::Norm,
            "scale" => Self::Scale,
            "range" => Self::Range,
            "center" => Self::Center,
            "medianiqr" => Self::MedianIqr,
            "datavariables" => Self::DataVariables,
            "replacevalues" => Self::ReplaceValues,
            "std" => Self::Std,
            "robust" => Self::Robust,
            "mean" => Self::Mean,
            "median" => Self::Median,
            "mad" => Self::Mad,
            "iqr" => Self::Iqr,
            "first" => Self::First,
            _ => Self::Other(text.to_ascii_lowercase()),
        }
    }

    pub fn public_name(&self) -> &str {
        match self {
            Self::ZScore => "zscore",
            Self::Norm => "norm",
            Self::Scale => "scale",
            Self::Range => "range",
            Self::Center => "center",
            Self::MedianIqr => "medianiqr",
            Self::DataVariables => "datavariables",
            Self::ReplaceValues => "replacevalues",
            Self::Std => "std",
            Self::Robust => "robust",
            Self::Mean => "mean",
            Self::Median => "median",
            Self::Mad => "mad",
            Self::Iqr => "iqr",
            Self::First => "first",
            Self::Other(text) => text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_keyword_decoder_preserves_supported_spellings_and_unknown_text() {
        for (name, expected) in [
            ("ZSCORE", NormalizeKeyword::ZScore),
            ("norm", NormalizeKeyword::Norm),
            ("scale", NormalizeKeyword::Scale),
            ("range", NormalizeKeyword::Range),
            ("center", NormalizeKeyword::Center),
            ("medianiqr", NormalizeKeyword::MedianIqr),
            ("DataVariables", NormalizeKeyword::DataVariables),
            ("ReplaceValues", NormalizeKeyword::ReplaceValues),
            ("STD", NormalizeKeyword::Std),
            ("ROBUST", NormalizeKeyword::Robust),
            ("MEAN", NormalizeKeyword::Mean),
            ("Median", NormalizeKeyword::Median),
            ("MAD", NormalizeKeyword::Mad),
            ("IQR", NormalizeKeyword::Iqr),
            ("First", NormalizeKeyword::First),
        ] {
            assert_eq!(NormalizeKeyword::from_public_text(name), expected);
        }
        assert_eq!(
            NormalizeKeyword::from_public_text("Mystery"),
            NormalizeKeyword::Other("mystery".into())
        );
    }
}
