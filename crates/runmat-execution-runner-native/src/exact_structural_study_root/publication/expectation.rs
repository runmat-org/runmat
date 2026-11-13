//! Caller-bound import expectations and workload lineage checks.
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralStudyResultExpectation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub geometry_source_digest: Digest,
    pub workloads: StructuralStudyWorkloads,
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub analysis_accelerator_requests: Vec<runmat_execution::resource::AcceleratorRequest>,
    pub input_materializations: Vec<runmat_analysis_execution::InputMaterializationPlanExpectation>,
    pub expected_meshing_request: MeshingRequest,
}

impl ExactStructuralStudyResultExpectation {
    pub fn study_identity(&self) -> NativeExecutionResult<StudyIdentity> {
        StudyIdentity::from_authored_digest(self.authored_study_digest, None).map_err(protocol)
    }

    pub fn validate(&self) -> NativeExecutionResult<()> {
        self.artifact_access.validate().map_err(protocol)?;
        self.workloads.validate().map_err(protocol)?;
        validate_analysis_accelerator_requests(
            &self.workloads,
            &self.analysis_accelerator_requests,
        )?;
        self.expected_meshing_request.validate().map_err(protocol)?;
        let model = self.workloads.authored_model().map_err(protocol)?;
        let authored_signals = model
            .signals
            .iter()
            .filter(|signal| signal.generator().is_some())
            .collect::<Vec<_>>();
        let authored_responses = model
            .response_spectra
            .iter()
            .filter(|spectrum| spectrum.ordinates.generator().is_some())
            .collect::<Vec<_>>();
        let authored_random = model
            .random_excitations
            .iter()
            .filter(|spectrum| spectrum.spectral_density.generator().is_some())
            .collect::<Vec<_>>();
        if authored_signals.len() + authored_responses.len() + authored_random.len()
            != self.input_materializations.len()
        {
            return Err(protocol(
                "whole-study input expectations differ from its authored workload",
            ));
        }
        let mut signals = authored_signals.into_iter();
        let mut responses = authored_responses.into_iter();
        let mut random = authored_random.into_iter();
        for materialization in &self.input_materializations {
            materialization.validate().map_err(protocol)?;
            let (revision, matches) = match materialization {
                runmat_analysis_execution::InputMaterializationPlanExpectation::DeferredSignal(
                    frozen,
                ) => (
                    &frozen.admitted().program_revision,
                    signals.next() == Some(frozen.authored()),
                ),
                runmat_analysis_execution::InputMaterializationPlanExpectation::Ready(
                    materialization,
                ) => match materialization.as_ref() {
                    runmat_analysis_execution::InputMaterializationExpectation::Signal(
                        expected,
                    ) => (
                        &expected.admitted.program_revision,
                        signals.next() == Some(&expected.authored),
                    ),
                    runmat_analysis_execution::InputMaterializationExpectation::Numerical(
                        expected,
                    ) => {
                        let matches = match &expected.authored {
                        runmat_analysis_execution::NumericalGeneratorSource::ResponseSpectrum(
                            spectrum,
                        ) => responses.next() == Some(spectrum.as_ref()),
                        runmat_analysis_execution::NumericalGeneratorSource::RandomSpectrum(
                            spectrum,
                        ) => random.next() == Some(spectrum.as_ref()),
                    };
                        (&expected.admitted.invocation.program_revision, matches)
                    }
                },
            };
            if revision != &self.program_revision || !matches {
                return Err(protocol(
                    "whole-study input expectation differs from its workload or program revision",
                ));
            }
        }
        if self.authored_study_digest.bytes() == &[0; 32]
            || self.geometry_source_digest.bytes() == &[0; 32]
            || self
                .frozen_project_digest
                .is_some_and(|digest| digest.bytes() == &[0; 32])
        {
            return Err(protocol("whole-study result expectation is invalid"));
        }
        Ok(())
    }
}
