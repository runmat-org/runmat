//! Canonical FEA option identities shared by runtime admission and inference.

/// The runtime's option-key normalization: ASCII alphanumerics only, lowercased.
pub fn normalize_option_key(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Name/value options with a single structural meaning across FEA operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaStructuralOption {
    /// `fea.study`: a typed `fea.ExecutionPlan`.
    Execution,
    /// `fea.study`: a validated `meshing.Mesh`.
    Mesh,
    /// `fea.sweep`: logical stop-on-first-failure policy.
    FailFast,
    /// `fea.trends`: positive window size.
    WindowSize,
}

impl FeaStructuralOption {
    pub fn decode(name: &str) -> Option<Self> {
        Self::from_normalized(&normalize_option_key(name))
    }

    pub fn from_normalized(name: &str) -> Option<Self> {
        Some(match name {
            "execution" => Self::Execution,
            "mesh" => Self::Mesh,
            "failfast" => Self::FailFast,
            "windowsize" => Self::WindowSize,
            _ => return None,
        })
    }
}

/// `fea.model` component options. Each name is accepted at most once by the
/// runtime and decoded with the shared normalization.
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
        Some(match normalize_option_key(name).as_str() {
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
        assert_eq!(
            FeaStructuralOption::decode("Window-Size"),
            Some(FeaStructuralOption::WindowSize)
        );
        assert_eq!(
            FeaStructuralOption::decode("FAIL_FAST"),
            Some(FeaStructuralOption::FailFast)
        );
        assert_eq!(FeaStructuralOption::decode("Profile"), None);
        assert_eq!(
            FeaModelOption::decode("Material_Assignments"),
            Some(FeaModelOption::MaterialAssignments)
        );
        assert_eq!(FeaModelOption::decode("Assignments"), None);
        assert_eq!(FeaPlotOption::decode("EDGES"), Some(FeaPlotOption::Edges));
        assert_eq!(FeaPlotOption::decode("show-edges"), None);
        assert_eq!(FeaMeshOption::decode("Source"), Some(FeaMeshOption::Source));
    }
}
