//! Contextual authoring names recognized by semantic compilation, not reserved
//! lexical keywords. Editors consume these same typed identities and roles.
use runmat_analysis_core::AssessmentEvidenceRequirement;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeaAuthoringWordRole {
    Block,
    Property,
    EnumValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeaContextualWord {
    Assessment,
    Minimum,
    Maximum,
    Evidence,
    EvidenceValue(AssessmentEvidenceRequirement),
    /// `display <output> { unit; range; precision; hide; context; deformation }`: how a study
    /// shows a field output.
    Display,
    Unit,
    Range,
    Precision,
    Hide,
    Context,
    Deformation,
}

impl FeaContextualWord {
    pub const ALL: &'static [Self] = &[
        Self::Assessment,
        Self::Minimum,
        Self::Maximum,
        Self::Evidence,
        Self::EvidenceValue(AssessmentEvidenceRequirement::AuthoredSamples),
        Self::EvidenceValue(AssessmentEvidenceRequirement::CertifiedBound),
        Self::Display,
        Self::Unit,
        Self::Range,
        Self::Precision,
        Self::Hide,
        Self::Context,
        Self::Deformation,
    ];

    pub const fn as_source(self) -> &'static str {
        match self {
            Self::Assessment => "assessment",
            Self::Minimum => "minimum",
            Self::Maximum => "maximum",
            Self::Evidence => "evidence",
            Self::EvidenceValue(AssessmentEvidenceRequirement::AuthoredSamples) => {
                "authored_samples"
            }
            Self::EvidenceValue(AssessmentEvidenceRequirement::CertifiedBound) => "certified_bound",
            Self::Display => "display",
            Self::Unit => "unit",
            Self::Range => "range",
            Self::Precision => "precision",
            Self::Hide => "hide",
            Self::Context => "context",
            Self::Deformation => "deformation",
        }
    }

    pub fn from_source(source: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|word| word.as_source() == source)
    }

    pub const fn role(self) -> FeaAuthoringWordRole {
        match self {
            Self::Assessment | Self::Display => FeaAuthoringWordRole::Block,
            Self::Minimum
            | Self::Maximum
            | Self::Evidence
            | Self::Unit
            | Self::Range
            | Self::Precision
            | Self::Hide
            | Self::Context
            | Self::Deformation => FeaAuthoringWordRole::Property,
            Self::EvidenceValue(_) => FeaAuthoringWordRole::EnumValue,
        }
    }
}
