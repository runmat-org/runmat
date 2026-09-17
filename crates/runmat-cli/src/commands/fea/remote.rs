use std::path::Path;

use anyhow::{Context, Result};
use base64::Engine as _;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::ExecutionBundleBuilder;
use runmat_runtime::analysis::{
    analysis_study_sweep_definition, AnalysisStudyRunData, AnalysisStudySpec,
    AnalysisStudySweepData, AnalysisStudySweepSpec,
};
use runmat_runtime::operations::{OperationContext, OperationEnvelope};

use crate::cli::RemoteExecutionTarget;
use crate::commands::job::secret::SavedOutputIntent;
use crate::commands::job::submit::RemoteSubmissionOptions;

mod materialization;
mod preparation;

use super::input_materialization::{
    prepare_numerical_materializations_in_frozen, prepare_signal_materializations_in_frozen,
};
pub(crate) use materialization::{materialize_remote_outputs, materialize_remote_sweep_outputs};
use preparation::{
    prepare_remote_execution, prepare_remote_sweep_execution, prepare_root, prepare_study_source,
    prepare_sweep_root,
};

pub(super) async fn execute_study(
    source: &Path,
    spec: AnalysisStudySpec,
    context: OperationContext,
    target: RemoteExecutionTarget,
    cli: &crate::cli::Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<Option<OperationEnvelope<AnalysisStudyRunData>>> {
    let resolved = crate::commands::package::resolve_for_source(source, cli)
        .await?
        .context("remote FEA requires a runmat.toml project containing the study document")?;
    let signals = spec
        .model
        .as_ref()
        .map_or(&[][..], |model| model.signals.as_slice());
    let signal_materializations = prepare_signal_materializations_in_frozen(
        source,
        signals,
        &spec.immutable_values,
        &resolved.resolved.frozen,
        config,
    )
    .await?;
    let (response_spectra, random_spectra) =
        spec.model.as_ref().map_or((&[][..], &[][..]), |model| {
            (
                model.response_spectra.as_slice(),
                model.random_excitations.as_slice(),
            )
        });
    let numerical_materializations = prepare_numerical_materializations_in_frozen(
        source,
        response_spectra,
        random_spectra,
        &spec.immutable_values,
        &resolved.resolved.frozen,
        config,
    )
    .await?;
    let input_materializations = signal_materializations
        .into_iter()
        .map(runmat_analysis_execution::PreparedInputMaterialization::Signal)
        .chain(
            numerical_materializations
                .into_iter()
                .map(runmat_analysis_execution::PreparedInputMaterialization::Numerical),
        )
        .collect::<Vec<_>>();
    let input_expectations = input_materializations
        .iter()
        .map(runmat_analysis_execution::PreparedInputMaterialization::expectation)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let source_study = prepare_study_source(&spec, context, input_materializations)?;
    let study = &source_study.intent;
    let continuation = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(study.continuation_bytes().map_err(anyhow::Error::msg)?);
    let project_identity = resolved.resolved.frozen.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_identity.graph_digest.bytes()),
        Digest::from_bytes(*project_identity.source_revision.bytes()),
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let placeholder = prepare_root(
        study,
        revision.clone(),
        "bundle-sizing-authority",
        &source_study.geometry_name,
        &source_study.geometry_bytes,
        source_study.input_materializations.clone(),
    )?;
    let mut bundle_builder =
        ExecutionBundleBuilder::native(&resolved.resolved.frozen, revision.clone())?
            .with_compiled_package_closure();
    for program in placeholder.required_programs()? {
        bundle_builder = bundle_builder.with_materialized_program(
            program.recipe,
            program.artifact.form,
            program.artifact.executable_bytes,
        );
    }
    let bundle = bundle_builder.build()?;
    let mut bundle_archive = Vec::new();
    write_bundle(&bundle, &mut bundle_archive, ArchiveLimits::default())?;
    let bundle_size = bundle_archive.len();
    let revision_for_scope = revision.clone();
    let committed = crate::commands::job::submit::submit_scoped(
        revision.clone(),
        bundle_size,
        |run_id| {
            let prepared = prepare_root(
                study,
                revision_for_scope,
                run_id,
                &source_study.geometry_name,
                &source_study.geometry_bytes,
                source_study.input_materializations.clone(),
            )?;
            prepare_remote_execution(prepared, revision.clone(), bundle_archive)
        },
        RemoteSubmissionOptions {
            project: target.project,
            cluster: target.cluster,
            queue: target.queue,
            trust_identity: target.trust_identity,
            idempotency_key: target.idempotency_key,
            workers: target.workers.max(1),
            on_run_created: None,
            output_intent: SavedOutputIntent::StructuralStudy {
                continuation: continuation.clone(),
                input_materializations: input_expectations.clone(),
                revision: revision.clone(),
                materialization: crate::commands::job::secret::OutputMaterialization::Pending,
                result: None,
            },
        },
    )
    .await?;
    if target.detach {
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "run": committed,
                    "materialization": "deferred",
                    "output": "structural_study",
                }))?
            );
        } else {
            println!(
                "remote FEA job {} submitted ({})",
                committed.id, committed.state
            );
            println!("  detached; verified study results will materialize on job attach");
        }
        return Ok(None);
    }
    let (outputs, objects) =
        crate::commands::job::attach::await_downloaded_outputs(&committed.id).await?;
    let envelope = materialize_remote_outputs(
        &committed.id,
        outputs.response,
        objects,
        &continuation,
        &input_expectations,
        revision,
    )?;
    crate::commands::job::secret::mark_structural_study_materialized(
        &committed.id,
        envelope.clone(),
    )?;
    Ok(Some(envelope))
}

pub(super) async fn execute_sweep(
    source: &Path,
    spec: AnalysisStudySweepSpec,
    context: OperationContext,
    target: RemoteExecutionTarget,
    cli: &crate::cli::Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<Option<OperationEnvelope<AnalysisStudySweepData>>> {
    let definition =
        analysis_study_sweep_definition(&spec, &context).map_err(super::report_operation_error)?;
    let resolved = crate::commands::package::resolve_for_source(source, cli)
        .await?
        .context("remote FEA requires a runmat.toml project containing the sweep document")?;
    let signal_counts = spec
        .studies
        .iter()
        .map(|study| {
            study.model.as_ref().map_or(0, |model| {
                model
                    .signals
                    .iter()
                    .filter(|signal| signal.generator().is_some())
                    .count()
            })
        })
        .collect::<Vec<_>>();
    let signals = spec
        .studies
        .iter()
        .filter_map(|study| study.model.as_ref())
        .flat_map(|model| model.signals.iter().cloned())
        .collect::<Vec<_>>();
    let immutable_values = spec
        .studies
        .iter()
        .flat_map(|study| study.immutable_values.iter().cloned())
        .collect::<Vec<_>>();
    let mut materializations = prepare_signal_materializations_in_frozen(
        source,
        &signals,
        &immutable_values,
        &resolved.resolved.frozen,
        config,
    )
    .await?
    .into_iter();
    let mut studies = Vec::with_capacity(spec.studies.len());
    for (study, count) in spec.studies.iter().zip(signal_counts) {
        let signal_inputs = materializations
            .by_ref()
            .take(count)
            .map(runmat_analysis_execution::PreparedInputMaterialization::Signal);
        let (response_spectra, random_spectra) =
            study.model.as_ref().map_or((&[][..], &[][..]), |model| {
                (
                    model.response_spectra.as_slice(),
                    model.random_excitations.as_slice(),
                )
            });
        let numerical_inputs = prepare_numerical_materializations_in_frozen(
            source,
            response_spectra,
            random_spectra,
            &study.immutable_values,
            &resolved.resolved.frozen,
            config,
        )
        .await?
        .into_iter()
        .map(runmat_analysis_execution::PreparedInputMaterialization::Numerical);
        studies.push(prepare_study_source(
            study,
            context.clone(),
            signal_inputs.chain(numerical_inputs).collect(),
        )?);
    }
    if materializations.next().is_some() {
        anyhow::bail!("remote sweep signal materialization inventory is oversized");
    }
    let continuations = studies
        .iter()
        .map(|study| {
            study
                .intent
                .continuation_bytes()
                .map(|bytes| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
                .map_err(anyhow::Error::msg)
        })
        .collect::<Result<Vec<_>>>()?;
    let input_expectations = studies
        .iter()
        .map(|study| {
            study
                .input_materializations
                .iter()
                .map(runmat_analysis_execution::PreparedInputMaterialization::expectation)
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let project_identity = resolved.resolved.frozen.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_identity.graph_digest.bytes()),
        Digest::from_bytes(*project_identity.source_revision.bytes()),
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let placeholder = prepare_sweep_root(
        &studies,
        revision.clone(),
        "bundle-sizing-authority",
        definition.clone(),
    )?;
    let mut bundle_builder =
        ExecutionBundleBuilder::native(&resolved.resolved.frozen, revision.clone())?
            .with_compiled_package_closure();
    for program in placeholder.required_programs() {
        bundle_builder = bundle_builder.with_materialized_program(
            program.recipe,
            program.artifact.form,
            program.artifact.executable_bytes,
        );
    }
    let bundle = bundle_builder.build()?;
    let mut bundle_archive = Vec::new();
    write_bundle(&bundle, &mut bundle_archive, ArchiveLimits::default())?;
    let bundle_size = bundle_archive.len();
    let revision_for_scope = revision.clone();
    let committed = crate::commands::job::submit::submit_scoped(
        revision.clone(),
        bundle_size,
        |run_id| {
            let prepared = prepare_sweep_root(&studies, revision_for_scope, run_id, definition)?;
            prepare_remote_sweep_execution(prepared, revision.clone(), bundle_archive)
        },
        RemoteSubmissionOptions {
            project: target.project,
            cluster: target.cluster,
            queue: target.queue,
            trust_identity: target.trust_identity,
            idempotency_key: target.idempotency_key,
            workers: target.workers.max(1),
            on_run_created: None,
            output_intent: SavedOutputIntent::StructuralSweep {
                spec: Box::new(spec.clone()),
                continuations: continuations.clone(),
                input_materializations: input_expectations.clone(),
                revision: revision.clone(),
                materialization: crate::commands::job::secret::OutputMaterialization::Pending,
                result: None,
            },
        },
    )
    .await?;
    if target.detach {
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "run": committed,
                    "materialization": "deferred",
                    "output": "structural_sweep",
                }))?
            );
        } else {
            println!(
                "remote FEA sweep {} submitted ({})",
                committed.id, committed.state
            );
            println!("  detached; verified sweep results will materialize on job attach");
        }
        return Ok(None);
    }
    let (outputs, objects) =
        crate::commands::job::attach::await_downloaded_outputs(&committed.id).await?;
    let envelope = materialize_remote_sweep_outputs(
        &committed.id,
        outputs.response,
        objects,
        &spec,
        &continuations,
        &input_expectations,
        revision,
    )?;
    crate::commands::job::secret::mark_structural_sweep_materialized(
        &committed.id,
        envelope.clone(),
    )?;
    Ok(Some(envelope))
}
