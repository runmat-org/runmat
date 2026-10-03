use super::super::identities::{
    FEA_BOUNDARY_CONDITION_CLASS, FEA_DOMAIN_CLASS, FEA_INTERFACE_CLASS, FEA_LOAD_CASE_CLASS,
    FEA_MATERIAL_ASSIGNMENT_CLASS, FEA_MATERIAL_CLASS, FEA_MODEL_CLASS, FEA_RUN_OPTIONS_CLASS,
    FEA_STEP_CLASS, FEA_STUDY_CLASS, FEA_SWEEP_CLASS,
};
use runmat_types::StaticClassIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaConstructorKind {
    AuthorStudy,
    Study,
    Sweep,
    Step,
    RunOptions,
    Model,
    Material,
    MaterialAssignment,
    BoundaryCondition,
    LoadCase,
    Domain,
    Interface,
    LoadDocument,
}

impl FeaConstructorKind {
    pub(super) const fn positional(self) -> usize {
        match self {
            Self::AuthorStudy => 3,
            Self::Study | Self::Sweep | Self::Step | Self::Model | Self::MaterialAssignment => 2,
            Self::BoundaryCondition | Self::LoadCase | Self::Interface => 3,
            Self::RunOptions | Self::Material | Self::Domain | Self::LoadDocument => 1,
        }
    }

    pub(super) const fn class(self) -> Option<StaticClassIdentity> {
        Some(match self {
            Self::AuthorStudy | Self::Study => FEA_STUDY_CLASS,
            Self::Sweep => FEA_SWEEP_CLASS,
            Self::Step => FEA_STEP_CLASS,
            Self::RunOptions => FEA_RUN_OPTIONS_CLASS,
            Self::Model => FEA_MODEL_CLASS,
            Self::Material => FEA_MATERIAL_CLASS,
            Self::MaterialAssignment => FEA_MATERIAL_ASSIGNMENT_CLASS,
            Self::BoundaryCondition => FEA_BOUNDARY_CONDITION_CLASS,
            Self::LoadCase => FEA_LOAD_CASE_CLASS,
            Self::Domain => FEA_DOMAIN_CLASS,
            Self::Interface => FEA_INTERFACE_CLASS,
            Self::LoadDocument => return None,
        })
    }

    pub(super) const fn is_document_load(self) -> bool {
        matches!(self, Self::LoadDocument)
    }
}
