use super::super::identities::{
    FEA_MATERIAL_ASSIGNMENT_CLASS, FEA_MATERIAL_CLASS, FEA_MODEL_CLASS, FEA_STUDY_CLASS,
    FEA_SWEEP_CLASS,
};
use runmat_types::StaticClassIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaConstructorKind {
    Study,
    Sweep,
    Model,
    Material,
    MaterialAssignment,
}

impl FeaConstructorKind {
    pub(super) const fn positional(self) -> usize {
        match self {
            Self::Study => 3,
            Self::Sweep | Self::Model | Self::MaterialAssignment => 2,
            Self::Material => 1,
        }
    }

    pub(super) const fn class(self) -> StaticClassIdentity {
        match self {
            Self::Study => FEA_STUDY_CLASS,
            Self::Sweep => FEA_SWEEP_CLASS,
            Self::Model => FEA_MODEL_CLASS,
            Self::Material => FEA_MATERIAL_CLASS,
            Self::MaterialAssignment => FEA_MATERIAL_ASSIGNMENT_CLASS,
        }
    }
}
