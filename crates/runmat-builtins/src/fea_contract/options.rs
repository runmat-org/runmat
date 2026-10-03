//! Canonical FEA option identities shared by runtime admission and inference.

pub fn normalize_option_key(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaStructuralOption {
    Profile,
    FailFast,
    WindowSize,
}

impl FeaStructuralOption {
    pub fn decode(name: &str) -> Option<Self> {
        Self::from_normalized(&normalize_option_key(name))
    }

    pub fn from_normalized(name: &str) -> Option<Self> {
        Some(match name {
            "profile" => Self::Profile,
            "failfast" => Self::FailFast,
            "windowsize" => Self::WindowSize,
            _ => return None,
        })
    }
}

/// Plot historically lowercases ASCII but does not erase punctuation except
/// for the explicitly admitted `field_id` alias.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaPlotOption {
    Field,
    Mesh,
    Overlay,
    Deformed,
}

impl FeaPlotOption {
    pub fn decode(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "field" | "fieldid" | "field_id" => Self::Field,
            "mesh" => Self::Mesh,
            "overlay" => Self::Overlay,
            "deformed" => Self::Deformed,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_spellings_normalize_but_plot_spellings_do_not_widen() {
        assert_eq!(
            FeaStructuralOption::decode("P_rofile"),
            Some(FeaStructuralOption::Profile)
        );
        assert_eq!(
            FeaStructuralOption::decode("Window-Size"),
            Some(FeaStructuralOption::WindowSize)
        );
        assert_eq!(
            FeaStructuralOption::decode("FAIL_FAST"),
            Some(FeaStructuralOption::FailFast)
        );
        assert_eq!(
            FeaPlotOption::decode("FIELD_ID"),
            Some(FeaPlotOption::Field)
        );
        assert_eq!(FeaPlotOption::decode("field-id"), None);
    }
}
