//! Remote authored study submission remains restricted to portable structural roots.
use super::*;

pub(super) async fn execute_study_remotely(
    source: &NativeCheckedFeaSource,
    study: ResolvedAnalysisStudy,
    inputs: Vec<AuthoredStudyInputBinding>,
    target: RemoteExecutionTarget,
    compute: FeaComputeAllocation,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    ensure_remote_structural(&study.study.sequence)?;
    let frozen = source
        .frozen_project
        .as_ref()
        .context("remote typed FEA requires an enclosing frozen RunMat project")?;
    let placeholder = prepare_study(
        source,
        study.clone(),
        &inputs,
        "bundle-sizing-authority",
        &compute,
        config,
    )
    .await?;
    let prepared_root = require_remote_root(placeholder.root)?;
    let saved_expectation = prepared_root.result_expectation();
    let assessment_functions = runmat_execution_runner_native::freeze_fea_assessment_functions(
        &source.session_source,
        &saved_expectation.study.assessments,
    )
    .await
    .map_err(anyhow::Error::msg)?;
    let study_id = placeholder.realized.study_id.clone();
    let revision = placeholder.revision.clone();
    let mut builder =
        ExecutionBundleBuilder::native(frozen, revision.clone())?.with_compiled_package_closure();
    for program in prepared_root.required_programs() {
        builder = builder.with_materialized_program(
            program.recipe.clone(),
            program.artifact.form,
            program.artifact.executable_bytes.clone(),
        );
    }
    let mut bundle_archive = Vec::new();
    write_bundle(
        &builder.build()?,
        &mut bundle_archive,
        ArchiveLimits::default(),
    )?;
    let bundle_size = bundle_archive.len();
    let revision_for_root = revision.clone();
    let committed = crate::commands::job::submit::submit_scoped(
        revision.clone(),
        bundle_size,
        |run_id| {
            let root = prepared_root
                .rebind_artifact_access(super::super::preparation::artifact_access(run_id))?;
            let remote = super::super::remote::prepare_typed_remote_execution(
                root,
                revision_for_root,
                bundle_archive,
            )?;
            Ok::<PreparedRemoteExecution, anyhow::Error>(remote)
        },
        RemoteSubmissionOptions {
            project: target.project,
            cluster: target.cluster,
            queue: target.queue,
            trust_identity: target.trust_identity,
            idempotency_key: target.idempotency_key,
            workers: target.workers.max(1),
            accelerator: compute.accelerators.first().cloned(),
            on_run_created: None,
            output_intent: SavedOutputIntent::FeaStudy {
                study_id,
                expectation: Box::new(saved_expectation),
                assessment_functions,
            },
        },
    )
    .await?;
    if target.detach {
        if json {
            println!("{}", serde_json::to_string(&committed)?);
        } else {
            println!(
                "{} {} ({})",
                presentation::stdout().success("FEA job"),
                committed.id,
                committed.state
            );
        }
        return Ok(());
    }
    crate::commands::job::attach::attach(&committed.id, false, json).await
}
