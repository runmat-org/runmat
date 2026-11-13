use super::super::identities::{
    FEA_MATERIAL_ASSIGNMENT_CLASS, FEA_MATERIAL_CLASS, FEA_MODEL_CLASS, FEA_STUDY_CLASS,
    FEA_SWEEP_CLASS,
};
use runmat_types::StaticClassIdentity;

/// FEA object constructors whose static facts this module owns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaConstructorKind {
    /// `fea.study(id, geometry, model, Name, Value, ...)`
    Study,
    /// `fea.sweep(id, studies, Name, Value, ...)`
    Sweep,
    /// `fea.model(id, geometry, Name, Value, ...)`
    Model,
    /// `fea.material(id, Name, Value, ...)`
    Material,
    /// `fea.materialAssignment(target, material, Name, Value, ...)`
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

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Study => "fea.study",
            Self::Sweep => "fea.sweep",
            Self::Model => "fea.model",
            Self::Material => "fea.material",
            Self::MaterialAssignment => "fea.materialAssignment",
        }
    }
}
