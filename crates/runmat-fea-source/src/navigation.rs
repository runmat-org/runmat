use runmat_analysis_language::{FeaDiagnosticLocation, FeaSymbolKey, LogicalSourcePath};
use serde::{Deserialize, Serialize};

/// Source navigation metadata kept beside, rather than embedded into, canonical FEA objects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSourceIndex {
    pub entries: Vec<FeaSourceIndexEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSourceIndexEntry {
    pub identity: FeaSourceIdentity,
    pub location: FeaDiagnosticLocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeaSourceIdentity {
    Declaration {
        module_path: LogicalSourcePath,
        symbol: FeaSymbolKey,
    },
    ModelMember {
        module_path: LogicalSourcePath,
        model_id: String,
        member_kind: FeaModelMemberKind,
        member_id: String,
    },
    StudyMember {
        module_path: LogicalSourcePath,
        study_id: String,
        member_kind: FeaStudyMemberKind,
        member_id: String,
    },
    Function {
        module_path: LogicalSourcePath,
        function_id: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum FeaStudyMemberKind {
    Output,
    Assessment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum FeaModelMemberKind {
    Section,
    Layup,
    Orientation,
    Constraint,
    Load,
    LoadCase,
    Interface,
    Connector,
    Excitation,
    ResponseSpectrum,
    RandomExcitation,
    InitialCondition,
    Signal,
}
