//! Transport representation shared by geometry-derived and pre-meshed study roots.

use runmat_analysis_execution::{
    PreparedGeneratorChunk, PreparedGeneratorProgram, PreparedInputMaterialization,
    PreparedNumericalMaterialization, PreparedSignalMaterialization,
};
use runmat_execution::value::ValueRef;
use runmat_execution_artifact::{ProgramExecutionDescriptor, ProgramExecutionRequest};
use serde::{Deserialize, Serialize};

use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InputMaterializationRequest {
    Signal(Box<SignalMaterializationRequest>),
    Numerical(Box<NumericalMaterializationRequest>),
}

impl InputMaterializationRequest {
    pub(crate) fn from_prepared(
        prepared: &PreparedInputMaterialization,
    ) -> NativeExecutionResult<Self> {
        match prepared {
            PreparedInputMaterialization::Signal(value) => {
                SignalMaterializationRequest::from_prepared(value)
                    .map(Box::new)
                    .map(Self::Signal)
            }
            PreparedInputMaterialization::Numerical(value) => {
                NumericalMaterializationRequest::from_prepared(value)
                    .map(Box::new)
                    .map(Self::Numerical)
            }
        }
    }

    pub(crate) fn to_prepared(&self) -> NativeExecutionResult<PreparedInputMaterialization> {
        match self {
            Self::Signal(value) => value
                .to_prepared()
                .map(Box::new)
                .map(PreparedInputMaterialization::Signal),
            Self::Numerical(value) => value
                .to_prepared()
                .map(Box::new)
                .map(PreparedInputMaterialization::Numerical),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SignalMaterializationRequest {
    authored: runmat_analysis_core::PhysicalSignal,
    admitted: runmat_analysis_execution::AdmittedSignalGenerator,
    descriptor: ProgramExecutionDescriptor,
    chunks: Vec<runmat_analysis_execution::GeneratorChunk>,
    input_references: Vec<ValueRef>,
    input_admissions: Vec<runmat_analysis_execution::SignalInputAdmission>,
}

impl SignalMaterializationRequest {
    fn from_prepared(prepared: &PreparedSignalMaterialization) -> NativeExecutionResult<Self> {
        prepared.validate().map_err(protocol)?;
        Ok(Self {
            authored: prepared.authored.clone(),
            admitted: prepared.admitted.clone(),
            descriptor: prepared.program.descriptor.clone(),
            chunks: prepared
                .program
                .chunks
                .iter()
                .map(|chunk| chunk.plan.clone())
                .collect(),
            input_references: prepared.program.input_references.clone(),
            input_admissions: prepared.input_admissions.clone(),
        })
    }

    fn to_prepared(&self) -> NativeExecutionResult<PreparedSignalMaterialization> {
        let prepared = PreparedSignalMaterialization {
            authored: self.authored.clone(),
            admitted: self.admitted.clone(),
            program: prepared_program(&self.descriptor, &self.chunks, &self.input_references),
            input_objects: Vec::new(),
            input_admissions: self.input_admissions.clone(),
        };
        prepared.validate().map_err(protocol)?;
        Ok(prepared)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NumericalMaterializationRequest {
    authored: runmat_analysis_execution::NumericalGeneratorSource,
    admitted: runmat_analysis_execution::AdmittedNumericalGenerator,
    descriptor: ProgramExecutionDescriptor,
    chunks: Vec<runmat_analysis_execution::GeneratorChunk>,
    input_references: Vec<ValueRef>,
    input_admissions: Vec<runmat_analysis_execution::SignalInputAdmission>,
}

impl NumericalMaterializationRequest {
    fn from_prepared(prepared: &PreparedNumericalMaterialization) -> NativeExecutionResult<Self> {
        prepared.validate().map_err(protocol)?;
        Ok(Self {
            authored: prepared.authored.clone(),
            admitted: prepared.admitted.clone(),
            descriptor: prepared.program.descriptor.clone(),
            chunks: prepared
                .program
                .chunks
                .iter()
                .map(|chunk| chunk.plan.clone())
                .collect(),
            input_references: prepared.program.input_references.clone(),
            input_admissions: prepared.input_admissions.clone(),
        })
    }

    fn to_prepared(&self) -> NativeExecutionResult<PreparedNumericalMaterialization> {
        let prepared = PreparedNumericalMaterialization {
            authored: self.authored.clone(),
            admitted: self.admitted.clone(),
            program: prepared_program(&self.descriptor, &self.chunks, &self.input_references),
            input_objects: Vec::new(),
            input_admissions: self.input_admissions.clone(),
        };
        prepared.validate().map_err(protocol)?;
        Ok(prepared)
    }
}

fn prepared_program(
    descriptor: &ProgramExecutionDescriptor,
    chunks: &[runmat_analysis_execution::GeneratorChunk],
    input_references: &[ValueRef],
) -> PreparedGeneratorProgram {
    PreparedGeneratorProgram {
        descriptor: descriptor.clone(),
        chunks: chunks
            .iter()
            .cloned()
            .map(|plan| PreparedGeneratorChunk {
                request: ProgramExecutionRequest {
                    schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
                    recipe: descriptor.recipe.clone(),
                    artifact: descriptor.artifact.clone(),
                    function: descriptor.function,
                    arguments: plan.arguments.clone(),
                    requested_outputs: descriptor.requested_outputs,
                },
                plan,
            })
            .collect(),
        input_references: input_references.to_vec(),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
