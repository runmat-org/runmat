//! Transport representation shared by geometry-derived and pre-meshed study roots.

use std::collections::BTreeSet;

use runmat_analysis_core::RealizedStudy;
use runmat_analysis_execution::{
    PreparedGeneratorChunk, PreparedGeneratorProgram, PreparedInputMaterialization,
    PreparedNumericalMaterialization, PreparedSignalMaterialization,
};
use runmat_execution::value::ValueRef;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{
    ProgramExecutionDescriptor, ProgramExecutionRequest, WorkerProgramRequest,
};
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    invocations: Option<Vec<WorkerProgramRequest>>,
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
            invocations: retained_graph_invocations(&prepared.program),
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
            program: prepared_program(
                &self.descriptor,
                &self.chunks,
                &self.input_references,
                self.invocations.as_deref(),
            )?,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    invocations: Option<Vec<WorkerProgramRequest>>,
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
            invocations: retained_graph_invocations(&prepared.program),
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
            program: prepared_program(
                &self.descriptor,
                &self.chunks,
                &self.input_references,
                self.invocations.as_deref(),
            )?,
            input_objects: Vec::new(),
            input_admissions: self.input_admissions.clone(),
        };
        prepared.validate().map_err(protocol)?;
        Ok(prepared)
    }
}

fn retained_graph_invocations(
    program: &PreparedGeneratorProgram,
) -> Option<Vec<WorkerProgramRequest>> {
    program
        .chunks
        .iter()
        .any(|chunk| matches!(&chunk.request, WorkerProgramRequest::Graph(_)))
        .then(|| {
            program
                .chunks
                .iter()
                .map(|chunk| chunk.request.clone())
                .collect()
        })
}

fn prepared_program(
    descriptor: &ProgramExecutionDescriptor,
    chunks: &[runmat_analysis_execution::GeneratorChunk],
    input_references: &[ValueRef],
    invocations: Option<&[WorkerProgramRequest]>,
) -> NativeExecutionResult<PreparedGeneratorProgram> {
    // The producer retains an invocation inventory only for homogeneous Graph
    // mode. Direct requests are reconstructed from the original descriptor;
    // an optional field must not become a second Direct invocation authority.
    if invocations.is_some_and(|requests| {
        requests.is_empty()
            || requests.len() != chunks.len()
            || requests
                .iter()
                .any(|request| !matches!(request, WorkerProgramRequest::Graph(_)))
    }) {
        return Err(protocol(
            "retained generator invocations must be a complete nonempty Graph inventory",
        ));
    }
    Ok(PreparedGeneratorProgram {
        descriptor: descriptor.clone(),
        chunks: chunks
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, plan)| PreparedGeneratorChunk {
                request: invocations
                    .map(|requests| requests[index].clone())
                    .unwrap_or_else(|| {
                        WorkerProgramRequest::direct(ProgramExecutionRequest {
                            console: descriptor.console,
                            environment: descriptor.environment.clone(),
                            schema_version:
                                runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                            recipe: descriptor.recipe.clone(),
                            artifact: descriptor.artifact.clone(),
                            callable: descriptor.callable.clone(),
                            context: runmat_execution::ProgramInvocationContext::Direct,
                            assignment: None,
                            job_id: None,
                            arguments: plan.arguments.clone(),
                            requested_outputs: descriptor.requested_outputs,
                        })
                    }),
                plan,
            })
            .collect(),
        input_references: input_references.to_vec(),
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InputMaterializationPlanRequest {
    Ready(Box<InputMaterializationRequest>),
    DeferredSignal(Box<runmat_analysis_execution::FrozenDeferredSignal>),
}

impl InputMaterializationPlanRequest {
    pub(crate) fn from_plan(
        plan: &runmat_analysis_execution::InputMaterializationPlan,
    ) -> NativeExecutionResult<Self> {
        plan.validate_closed().map_err(protocol)?;
        match plan {
            runmat_analysis_execution::InputMaterializationPlan::Ready(value) => {
                InputMaterializationRequest::from_prepared(value)
                    .map(Box::new)
                    .map(Self::Ready)
            }
            runmat_analysis_execution::InputMaterializationPlan::DeferredSignal(value) => {
                Ok(Self::DeferredSignal(value.clone()))
            }
        }
    }

    pub(crate) fn to_plan(
        &self,
    ) -> NativeExecutionResult<runmat_analysis_execution::InputMaterializationPlan> {
        let plan = match self {
            Self::Ready(value) => value.to_prepared()?.into(),
            Self::DeferredSignal(value) => {
                runmat_analysis_execution::InputMaterializationPlan::DeferredSignal(value.clone())
            }
        };
        plan.validate().map_err(protocol)?;
        Ok(plan)
    }
}

pub(crate) fn validate_input_materialization_plan_inventory(
    study: &RealizedStudy,
    plans: &[runmat_analysis_execution::InputMaterializationPlan],
    revision: &ProgramRevision,
) -> NativeExecutionResult<()> {
    let mut identities = BTreeSet::new();
    for plan in plans {
        let identity = match plan {
            runmat_analysis_execution::InputMaterializationPlan::Ready(value) => {
                value.materialization_digest()
            }
            runmat_analysis_execution::InputMaterializationPlan::DeferredSignal(value) => {
                value.identity().map_err(protocol)?
            }
        };
        if !identities.insert(identity) {
            return Err(protocol(
                "study input plan inventory contains duplicated identities",
            ));
        }
    }
    let expectations = plans
        .iter()
        .map(|plan| plan.expectation().map_err(protocol))
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    validate_input_plan_expectation_inventory(study, &expectations, revision)
}

pub(crate) fn validate_input_plan_expectation_inventory(
    study: &RealizedStudy,
    expectations: &[runmat_analysis_execution::InputMaterializationPlanExpectation],
    revision: &ProgramRevision,
) -> NativeExecutionResult<()> {
    for expectation in expectations {
        expectation.validate().map_err(protocol)?;
    }
    let models = study
        .included_models
        .iter()
        .chain(std::iter::once(&study.model))
        .collect::<Vec<_>>();
    let expected = models
        .iter()
        .flat_map(|model| &model.signals)
        .filter(|signal| signal.generator().is_some())
        .map(InputExpectationSource::Signal)
        .chain(
            models
                .iter()
                .flat_map(|model| &model.response_spectra)
                .filter(|spectrum| spectrum.ordinates.generator().is_some())
                .map(InputExpectationSource::Response),
        )
        .chain(
            models
                .iter()
                .flat_map(|model| &model.random_excitations)
                .filter(|spectrum| spectrum.spectral_density.generator().is_some())
                .map(InputExpectationSource::Random),
        )
        .collect::<Vec<_>>();
    let actual = expectations
        .iter()
        .map(|expectation| match expectation {
            runmat_analysis_execution::InputMaterializationPlanExpectation::DeferredSignal(
                value,
            ) => (
                &value.admitted().program_revision,
                InputExpectationSource::Signal(value.authored()),
            ),
            runmat_analysis_execution::InputMaterializationPlanExpectation::Ready(value) => {
                match value.as_ref() {
                    runmat_analysis_execution::InputMaterializationExpectation::Signal(value) => (
                        &value.admitted.program_revision,
                        InputExpectationSource::Signal(&value.authored),
                    ),
                    runmat_analysis_execution::InputMaterializationExpectation::Numerical(
                        value,
                    ) => {
                        let source = match &value.authored {
                    runmat_analysis_execution::NumericalGeneratorSource::ResponseSpectrum(
                        value,
                    ) => InputExpectationSource::Response(value),
                    runmat_analysis_execution::NumericalGeneratorSource::RandomSpectrum(value) => {
                        InputExpectationSource::Random(value)
                    }
                };
                        (&value.admitted.invocation.program_revision, source)
                    }
                }
            }
        })
        .collect::<Vec<_>>();
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|((actual_revision, actual), expected)| {
                *actual_revision != revision || *actual != expected
            })
    {
        return Err(protocol(
            "study input expectations differ from its authored generator inventory",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum InputExpectationSource<'a> {
    Signal(&'a runmat_analysis_core::PhysicalSignal),
    Response(&'a runmat_analysis_core::ResponseSpectrum),
    Random(&'a runmat_analysis_core::RandomExcitationSpectrum),
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
