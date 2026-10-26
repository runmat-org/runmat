//! Canonical FEA option identities shared by runtime admission and inference.

/// The runtime's option-key normalization: ASCII alphanumerics only, lowercased.
pub fn normalize_option_key(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Options accepted by the existing model constructor. Decoding does not
/// impose uniqueness: model options retain their ordered last-wins behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaModelOption {
    Frame,
    Materials,
    MaterialAssignments,
    Constraints,
    Boundaries,
    Signals,
    Excitations,
    ExcitationCases,
    InitialConditions,
    ResponseSpectra,
    Steps,
    Sequence,
    Domains,
    Interfaces,
}

impl FeaModelOption {
    pub fn decode(name: &str) -> Option<Self> {
        Self::from_normalized(&normalize_option_key(name))
    }

    pub fn from_normalized(name: &str) -> Option<Self> {
        Some(match name {
            "frame" => Self::Frame,
            "materials" => Self::Materials,
            "materialassignments" => Self::MaterialAssignments,
            "constraints" => Self::Constraints,
            "boundaries" => Self::Boundaries,
            "signals" => Self::Signals,
            "excitations" => Self::Excitations,
            "excitationcases" => Self::ExcitationCases,
            "initialconditions" => Self::InitialConditions,
            "responsespectra" => Self::ResponseSpectra,
            "steps" => Self::Steps,
            "sequence" => Self::Sequence,
            "domains" => Self::Domains,
            "interfaces" => Self::Interfaces,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaStudyOption {
    Execution,
    Mesh,
}

impl FeaStudyOption {
    pub fn decode(name: &str) -> Option<Self> {
        Self::from_normalized(&normalize_option_key(name))
    }

    pub fn from_normalized(name: &str) -> Option<Self> {
        Some(match name {
            "execution" => Self::Execution,
            "mesh" => Self::Mesh,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaMaterialAssignmentOption {
    ExpectedMaterial,
    Confidence,
}

impl FeaMaterialAssignmentOption {
    pub fn decode(name: &str) -> Option<Self> {
        Self::from_normalized(&normalize_option_key(name))
    }

    pub fn from_normalized(name: &str) -> Option<Self> {
        Some(match name {
            "expectedmaterial" | "expectedmaterialid" => Self::ExpectedMaterial,
            "confidence" => Self::Confidence,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaSweepOption {
    FailFast,
}

impl FeaSweepOption {
    pub fn decode(name: &str) -> Option<Self> {
        Self::from_normalized(&normalize_option_key(name))
    }

    pub fn from_normalized(name: &str) -> Option<Self> {
        match name {
            "failfast" => Some(Self::FailFast),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaTrendsOption {
    WindowSize,
}

impl FeaTrendsOption {
    pub fn decode(name: &str) -> Option<Self> {
        Self::from_normalized(&normalize_option_key(name))
    }

    pub fn from_normalized(name: &str) -> Option<Self> {
        Some(match name {
            "windowsize" => Self::WindowSize,
            _ => return None,
        })
    }
}

/// `plot(field, ...)` options. The runtime lowercases ASCII but does not
/// erase punctuation, so `"Edges"` matches and `"show-edges"` does not.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaPlotOption {
    Edges,
    Overlay,
    Deformed,
}

impl FeaPlotOption {
    pub fn decode(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "edges" => Self::Edges,
            "overlay" => Self::Overlay,
            "deformed" => Self::Deformed,
            _ => return None,
        })
    }
}

/// `mesh(runOrResultsOrField, ...)` options, decoded with the shared normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaMeshOption {
    Source,
    Edges,
}

impl FeaMeshOption {
    pub fn decode(name: &str) -> Option<Self> {
        Some(match normalize_option_key(name).as_str() {
            "source" => Self::Source,
            "edges" => Self::Edges,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_spellings_normalize_but_plot_spellings_do_not_widen() {
        assert_eq!(FeaModelOption::decode("P_rofile"), None);
        assert_eq!(
            FeaTrendsOption::decode("Window-Size"),
            Some(FeaTrendsOption::WindowSize)
        );
        assert_eq!(
            FeaSweepOption::decode("FAIL_FAST"),
            Some(FeaSweepOption::FailFast)
        );
        assert_eq!(
            FeaModelOption::decode("S_te-ps"),
            Some(FeaModelOption::Steps)
        );
        assert_eq!(
            FeaModelOption::decode("Material_Assignments"),
            Some(FeaModelOption::MaterialAssignments)
        );
        assert_eq!(FeaModelOption::decode("Assignments"), None);
        assert_eq!(FeaModelOption::decode("Cases"), None);
        assert_eq!(FeaStudyOption::decode("P_rofile"), None);
        assert_eq!(FeaStudyOption::decode("Mesh"), Some(FeaStudyOption::Mesh));
        assert_eq!(
            FeaMaterialAssignmentOption::decode("Expected_Material-Id"),
            Some(FeaMaterialAssignmentOption::ExpectedMaterial)
        );
        assert_eq!(FeaPlotOption::decode("EDGES"), Some(FeaPlotOption::Edges));
        assert_eq!(FeaPlotOption::decode("show-edges"), None);
        assert_eq!(FeaMeshOption::decode("Source"), Some(FeaMeshOption::Source));
    }
}
