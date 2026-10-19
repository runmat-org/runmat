use runmat_analysis_fea::execution::NumericalMaterializationIdentity;
use runmat_execution::Digest;
use runmat_execution_artifact::{ArtifactAccess, ProgramExecutionObject};

use crate::{
    AnalysisBridgeResult, NumericalMaterializationExpectation, PreparedGeneratorProgram,
    PreparedNumericalMaterialization, PreparedSignalMaterialization,
    SignalMaterializationExpectation,
};

#[derive(Clone, Debug, PartialEq)]
pub enum PreparedInputMaterialization {
    Signal(Box<PreparedSignalMaterialization>),
    Numerical(Box<PreparedNumericalMaterialization>),
}

impl PreparedInputMaterialization {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        match self {
            Self::Signal(value) => value.validate(),
            Self::Numerical(value) => value.validate(),
        }
    }

    pub fn validate_closed(&self) -> AnalysisBridgeResult<()> {
        match self {
            Self::Signal(value) => value.validate_closed(),
            Self::Numerical(value) => value.validate_closed(),
        }
    }

    pub fn rebind(self, access: &ArtifactAccess) -> AnalysisBridgeResult<Self> {
        match self {
            Self::Signal(value) => value.rebind(access).map(Box::new).map(Self::Signal),
            Self::Numerical(value) => value.rebind(access).map(Box::new).map(Self::Numerical),
        }
    }

    pub fn expectation(&self) -> AnalysisBridgeResult<InputMaterializationExpectation> {
        match self {
            Self::Signal(value) => value
                .expectation()
                .map(Box::new)
                .map(InputMaterializationExpectation::Signal),
            Self::Numerical(value) => value
                .expectation()
                .map(Box::new)
                .map(InputMaterializationExpectation::Numerical),
        }
    }

    pub fn program(&self) -> &PreparedGeneratorProgram {
        match self {
            Self::Signal(value) => &value.program,
            Self::Numerical(value) => &value.program,
        }
    }

    pub fn input_objects(&self) -> &[ProgramExecutionObject] {
        match self {
            Self::Signal(value) => &value.input_objects,
            Self::Numerical(value) => &value.input_objects,
        }
    }

    pub fn materialization_digest(&self) -> Digest {
        match self {
            Self::Signal(value) => value.admitted.materialization_identity.digest(),
            Self::Numerical(value) => value.admitted.materialization_identity.digest(),
        }
    }

    pub fn numerical_identity(&self) -> Option<NumericalMaterializationIdentity> {
        match self {
            Self::Signal(_) => None,
            Self::Numerical(value) => Some(value.admitted.materialization_identity),
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMaterializationExpectation {
    Signal(Box<SignalMaterializationExpectation>),
    Numerical(Box<NumericalMaterializationExpectation>),
}

impl InputMaterializationExpectation {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        match self {
            Self::Signal(value) => value.validate(),
            Self::Numerical(value) => value.validate(),
        }
    }
}
