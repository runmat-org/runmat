//! Validate original workloads, materialization order and accelerator intent.
use super::*;

pub(super) fn validate_input_materialization_inventory(
    workloads: &StructuralStudyWorkloads,
    materializations: &[InputMaterializationPlan],
    program_revision: &ProgramRevision,
) -> NativeExecutionResult<()> {
    let model = workloads.authored_model().map_err(protocol)?;
    let authored_signal_count = model
        .signals
        .iter()
        .filter(|signal| signal.generator().is_some())
        .count();
    let authored_response_count = model
        .response_spectra
        .iter()
        .filter(|spectrum| spectrum.ordinates.generator().is_some())
        .count();
    let authored_random_count = model
        .random_excitations
        .iter()
        .filter(|spectrum| spectrum.spectral_density.generator().is_some())
        .count();
    if authored_signal_count + authored_response_count + authored_random_count
        != materializations.len()
    {
        return Err(protocol(
            "study input materializations differ from its authored generator inventory",
        ));
    }
    let mut signals = model
        .signals
        .iter()
        .filter(|signal| signal.generator().is_some());
    let mut responses = model
        .response_spectra
        .iter()
        .filter(|spectrum| spectrum.ordinates.generator().is_some());
    let mut random = model
        .random_excitations
        .iter()
        .filter(|spectrum| spectrum.spectral_density.generator().is_some());
    let mut identities = BTreeSet::new();
    for prepared in materializations {
        prepared.validate().map_err(protocol)?;
        let (revision, identity, authored_matches) = match prepared {
            InputMaterializationPlan::DeferredSignal(frozen) => (
                &frozen.admitted().program_revision,
                frozen.identity().map_err(protocol)?,
                signals.next() == Some(frozen.authored()),
            ),
            InputMaterializationPlan::Ready(prepared) => match prepared.as_ref() {
                runmat_analysis_execution::PreparedInputMaterialization::Signal(prepared) => (
                    &prepared.admitted.program_revision,
                    prepared.admitted.materialization_identity.digest(),
                    signals.next() == Some(&prepared.authored),
                ),
                runmat_analysis_execution::PreparedInputMaterialization::Numerical(prepared) => {
                    let matches = match &prepared.authored {
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
                        matches,
                    )
                }
            },
        };
        if revision != program_revision || !authored_matches || !identities.insert(identity) {
            return Err(protocol(
                "study input materialization inventory is inconsistent or duplicated",
            ));
        }
    }
    if signals.next().is_some() || responses.next().is_some() || random.next().is_some() {
        return Err(protocol(
            "study input materialization inventory is incomplete",
        ));
    }
    Ok(())
}

pub(in crate::exact_structural_study_root) fn validate_analysis_accelerator_requests(
    workloads: &StructuralStudyWorkloads,
    requests: &[AcceleratorRequest],
) -> NativeExecutionResult<()> {
    if requests.len() > 16 {
        return Err(protocol(
            "exact structural study exceeds the accelerator-request limit",
        ));
    }
    for request in requests {
        request.validate().map_err(protocol)?;
    }
    let backends = workloads.backends().map_err(protocol)?;
    let requests_accelerator = backends.contains(&runmat_analysis_fea::ComputeBackend::Gpu);
    let requests_host = backends.contains(&runmat_analysis_fea::ComputeBackend::Cpu);
    if requests_accelerator == requests.is_empty() {
        return Err(protocol(
            "exact structural study accelerator requests differ from its authored backends",
        ));
    }
    if requests_accelerator && requests_host {
        return Err(protocol(
            "one exact structural study cannot mix host and accelerator backends",
        ));
    }
    Ok(())
}

pub(crate) fn structural_workload_inventory_digest(
    workloads: &StructuralStudyWorkloads,
) -> NativeExecutionResult<Digest> {
    workloads.canonical_digest().map_err(protocol)
}
