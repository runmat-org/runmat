//! Transport representation shared by geometry-derived and pre-meshed study roots.

use std::collections::BTreeSet;

use runmat_analysis_core::RealizedStudy;
use runmat_analysis_execution::{
    PreparedGeneratorChunk, PreparedGeneratorProgram, PreparedInputMaterialization,
    PreparedNumericalMaterialization, PreparedSignalMaterialization,
};
use runmat_execution::value::ValueRef;
use runmat_execution::ProgramRevision;
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
    topology_binding: Option<runmat_analysis_execution::SignalTopologyBinding>,
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
            topology_binding: prepared.topology_binding.clone(),
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
            topology_binding: self.topology_binding.clone(),
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

pub(crate) fn validate_input_materialization_inventory(
    study: &RealizedStudy,
    materializations: &[PreparedInputMaterialization],
    revision: &ProgramRevision,
) -> NativeExecutionResult<()> {
    let models = study
        .included_models
        .iter()
        .chain(std::iter::once(&study.model))
        .collect::<Vec<_>>();
    let mut signals = models
        .iter()
        .flat_map(|model| &model.signals)
        .filter(|signal| signal.generator().is_some());
    let mut responses = models
        .iter()
        .flat_map(|model| &model.response_spectra)
        .filter(|spectrum| spectrum.ordinates.generator().is_some());
    let mut random = models
        .iter()
        .flat_map(|model| &model.random_excitations)
        .filter(|spectrum| spectrum.spectral_density.generator().is_some());
    let expected_count =
        signals.clone().count() + responses.clone().count() + random.clone().count();
    if expected_count != materializations.len() {
        return Err(protocol(
            "study input materializations differ from its authored generator inventory",
        ));
    }
    let mut identities = BTreeSet::new();
    for prepared in materializations {
        prepared.validate().map_err(protocol)?;
        let (materialization_revision, identity, authored_matches) = match prepared {
            PreparedInputMaterialization::Signal(prepared) => (
                &prepared.admitted.program_revision,
                prepared.admitted.materialization_identity.digest(),
                signals.next() == Some(&prepared.authored),
            ),
            PreparedInputMaterialization::Numerical(prepared) => {
                let authored_matches = match &prepared.authored {
                    runmat_analysis_execution::NumericalGeneratorSource::ResponseSpectrum(
                        spectrum,
                    ) => responses.next() == Some(spectrum.as_ref()),
                    runmat_analysis_execution::NumericalGeneratorSource::RandomSpectrum(
                        spectrum,
                    ) => random.next() == Some(spectrum.as_ref()),
                };
                (
                    &prepared.admitted.invocation.program_revision,
                    prepared.admitted.materialization_identity.digest(),
                    authored_matches,
                )
            }
        };
        if materialization_revision != revision || !authored_matches || !identities.insert(identity)
        {
            return Err(protocol(
                "study input materialization inventory is inconsistent or duplicated",
            ));
        }
    }
    Ok(())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
