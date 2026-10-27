use std::collections::BTreeSet;

use runmat_analysis_core::{GeneratorCallable, ImportedFunctionIdentity, RealizedStudy};
use runmat_analysis_execution::{
    admit_named_random_spectrum_generator, admit_named_response_spectrum_generator,
    admit_named_signal_generator, plan_random_spectrum_generator_chunks,
    plan_response_spectrum_generator_chunks, plan_signal_generator_chunks,
    prepare_numerical_materialization, prepare_signal_materialization, FrozenDeferredSignal,
    InputMaterializationPlan, NumericalGeneratorSource, PreparedInputMaterialization,
};
use runmat_execution::value::ValueLimits;

pub(super) async fn prepare_input_materializations(
    source: &runmat_core::CheckedFeaSessionSource,
    study: &RealizedStudy,
) -> Result<Vec<InputMaterializationPlan>, String> {
    let models = study
        .included_models
        .iter()
        .chain(std::iter::once(&study.model))
        .collect::<Vec<_>>();
    let maximum_inline_result_bytes = usize::try_from(ValueLimits::default().max_inline_bytes)
        .map_err(|_| "generator inline-result bound exceeds this host".to_string())?;
    let mut prepared = Vec::new();

    for signal in models
        .iter()
        .flat_map(|model| &model.signals)
        .filter(|signal| signal.generator().is_some())
    {
        let identity = imported_identity(
            &signal
                .generator()
                .expect("generator-backed signal was filtered")
                .invocation
                .callable,
        )?;
        let compiled = super::callable::compile_imported_function(
            source,
            &identity.module_path,
            &identity.symbol,
            Some(&identity.source_digest),
        )
        .await?;
        if matches!(
            signal.shape(),
            runmat_analysis_core::SignalShape::DeferredTopologyField { .. }
        ) {
            let frozen = FrozenDeferredSignal::prepare(
                signal.clone(),
                compiled.revision,
                compiled.callable,
                compiled.executable,
                BTreeSet::new(),
                maximum_inline_result_bytes,
            )
            .map_err(|error| error.to_string())?;
            prepared.push(InputMaterializationPlan::DeferredSignal(Box::new(frozen)));
            continue;
        }
        let admitted = admit_named_signal_generator(
            signal,
            compiled.revision,
            compiled.callable,
            compiled.compiled_binding,
            &compiled.function,
            &BTreeSet::new(),
        )
        .map_err(|error| error.to_string())?;
        let plan = plan_signal_generator_chunks(
            signal,
            &admitted,
            None,
            maximum_inline_result_bytes,
            reject_unbound_artifact,
        )
        .map_err(|error| error.to_string())?;
        let materialization = prepare_signal_materialization(
            signal.clone(),
            admitted,
            compiled.executable,
            plan.chunks,
            Vec::new(),
            plan.input_admissions,
            None,
        )
        .map_err(|error| error.to_string())?;
        prepared.push(PreparedInputMaterialization::Signal(Box::new(materialization)).into());
    }

    for spectrum in models
        .iter()
        .flat_map(|model| &model.response_spectra)
        .filter(|spectrum| spectrum.ordinates.generator().is_some())
    {
        let identity = imported_identity(
            &spectrum
                .ordinates
                .generator()
                .expect("generator-backed response spectrum was filtered")
                .callable,
        )?;
        let compiled = super::callable::compile_imported_function(
            source,
            &identity.module_path,
            &identity.symbol,
            Some(&identity.source_digest),
        )
        .await?;
        let admitted = admit_named_response_spectrum_generator(
            spectrum,
            compiled.revision,
            compiled.callable,
            compiled.compiled_binding,
            &compiled.function,
            &BTreeSet::new(),
        )
        .map_err(|error| error.to_string())?;
        let plan = plan_response_spectrum_generator_chunks(
            spectrum,
            &admitted,
            maximum_inline_result_bytes,
            reject_unbound_artifact,
        )
        .map_err(|error| error.to_string())?;
        let materialization = prepare_numerical_materialization(
            NumericalGeneratorSource::ResponseSpectrum(Box::new(spectrum.clone())),
            admitted,
            compiled.executable,
            plan.chunks,
            Vec::new(),
            plan.input_admissions,
        )
        .map_err(|error| error.to_string())?;
        prepared.push(PreparedInputMaterialization::Numerical(Box::new(materialization)).into());
    }

    for spectrum in models
        .iter()
        .flat_map(|model| &model.random_excitations)
        .filter(|spectrum| spectrum.spectral_density.generator().is_some())
    {
        let identity = imported_identity(
            &spectrum
                .spectral_density
                .generator()
                .expect("generator-backed random spectrum was filtered")
                .callable,
        )?;
        let compiled = super::callable::compile_imported_function(
            source,
            &identity.module_path,
            &identity.symbol,
            Some(&identity.source_digest),
        )
        .await?;
        let admitted = admit_named_random_spectrum_generator(
            spectrum,
            compiled.revision,
            compiled.callable,
            compiled.compiled_binding,
            &compiled.function,
            &BTreeSet::new(),
        )
        .map_err(|error| error.to_string())?;
        let plan = plan_random_spectrum_generator_chunks(
            spectrum,
            &admitted,
            maximum_inline_result_bytes,
            reject_unbound_artifact,
        )
        .map_err(|error| error.to_string())?;
        let materialization = prepare_numerical_materialization(
            NumericalGeneratorSource::RandomSpectrum(Box::new(spectrum.clone())),
            admitted,
            compiled.executable,
            plan.chunks,
            Vec::new(),
            plan.input_admissions,
        )
        .map_err(|error| error.to_string())?;
        prepared.push(PreparedInputMaterialization::Numerical(Box::new(materialization)).into());
    }

    Ok(prepared)
}

fn imported_identity(callable: &GeneratorCallable) -> Result<&ImportedFunctionIdentity, String> {
    match callable {
        GeneratorCallable::ImportedFunction(identity) => Ok(identity),
        GeneratorCallable::ProjectFunction { qualified_name } => Err(format!(
            "typed FEA generator `{qualified_name}` is not bound to an imported source identity"
        )),
        GeneratorCallable::BoundFunction { display_name, .. } => Err(format!(
            "typed FEA generator `{display_name}` contains a runtime-only function binding"
        )),
    }
}

fn reject_unbound_artifact(
    artifact: &runmat_analysis_core::SignalArtifactRef,
) -> Result<
    runmat_analysis_execution::ResolvedSignalArtifact,
    runmat_analysis_execution::AnalysisBridgeError,
> {
    Err(runmat_analysis_execution::AnalysisBridgeError::Invalid(
        format!(
            "generator artifact `{}` has no immutable project binding",
            artifact.artifact_id
        ),
    ))
}
