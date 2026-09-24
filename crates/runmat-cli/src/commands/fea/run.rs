use anyhow::{bail, Context, Result};
use runmat_analysis_core::AuthoredStudyInputBinding;
use runmat_analysis_language::{ResolvedAnalysisStudy, ResolvedAnalysisSweep};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::ExecutionBundleBuilder;
use runmat_runtime::analysis::ResolvedFeaRoot;
use std::path::PathBuf;

use crate::cli::{Cli, ExecutionTarget, ExecutionTargetArgs, RemoteExecutionTarget};
use crate::commands::job::secret::SavedOutputIntent;
use crate::commands::job::submit::{PreparedRemoteExecution, RemoteSubmissionOptions};
use crate::presentation;

use super::preparation::{prepare_study, FeaComputeAllocation};
use super::result_presentation::{local_summary, print_fea_result, print_sweep_result};
use super::source::NativeCheckedFeaSource;

pub(super) async fn execute(
    _path: PathBuf,
    source: NativeCheckedFeaSource,
    root: ResolvedFeaRoot,
    execution: ExecutionTargetArgs,
    _cli: &Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    let execution = resolve_execution(&source, &root, &execution)?;
    match root {
        ResolvedFeaRoot::Study(study) => {
            execute_study(&source, study, Vec::new(), execution, config, json).await
        }
        ResolvedFeaRoot::Sweep(sweep) => {
            execute_sweep(&source, sweep, execution, config, json).await
        }
    }
}

async fn execute_study(
    source: &NativeCheckedFeaSource,
    study: ResolvedAnalysisStudy,
    inputs: Vec<AuthoredStudyInputBinding>,
    execution: ResolvedFeaExecution,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    match execution.target {
        ExecutionTarget::Local => {
            if !inputs.is_empty() {
                bail!("direct study execution cannot supply sweep case inputs");
            }
            let selector = runmat_runtime::analysis::FeaRootSelector::Study {
                id: study.study.study_id.clone(),
            };
            let completed = runmat_execution_runner_native::execute_checked_fea_locally(
                &source.session_source,
                runmat_execution_runner_native::LocalFeaExecutionOptions {
                    root: Some(selector),
                    workers: None,
                    backend: execution.compute.backend,
                    accelerators: execution.compute.accelerators.clone(),
                    ..Default::default()
                },
            )
            .await
            .map_err(anyhow::Error::msg)?;
            let runmat_execution_runner_native::LocalFeaExecution::Study(completed) = completed
            else {
                bail!("selected study resolved to a sweep during native execution");
            };
            print_fea_result(
                &completed.case_id,
                &completed.result,
                &completed.publication,
                json,
            )
        }
        ExecutionTarget::Remote(target) => {
            execute_study_remotely(
                source,
                study,
                inputs,
                target,
                execution.compute,
                config,
                json,
            )
            .await
        }
    }
}

async fn execute_sweep(
    source: &NativeCheckedFeaSource,
    sweep: ResolvedAnalysisSweep,
    execution: ResolvedFeaExecution,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    let cases = sweep.sweep.expand_cases(&sweep.study.study)?;
    if cases.is_empty() {
        bail!("typed FEA sweep contains no cases");
    }
    match execution.target {
        ExecutionTarget::Local => {
            let selector = runmat_runtime::analysis::FeaRootSelector::Sweep {
                id: sweep.sweep.sweep_id.clone(),
            };
            let completed = runmat_execution_runner_native::execute_checked_fea_locally(
                &source.session_source,
                runmat_execution_runner_native::LocalFeaExecutionOptions {
                    root: Some(selector),
                    workers: None,
                    backend: execution.compute.backend,
                    accelerators: execution.compute.accelerators.clone(),
                    ..Default::default()
                },
            )
            .await
            .map_err(anyhow::Error::msg)?;
            let runmat_execution_runner_native::LocalFeaExecution::Sweep {
                cases: completed, ..
            } = completed
            else {
                bail!("selected sweep resolved to a study during native execution");
            };
            let summaries = completed
                .iter()
                .map(|case| local_summary(&case.case_id, &case.result, &case.publication))
                .collect::<Result<Vec<_>>>()?;
            print_sweep_result(&sweep.sweep.sweep_id, &summaries, json)
        }
        ExecutionTarget::Remote(target) => {
            execute_sweep_remotely(
                source,
                sweep,
                cases,
                target,
                execution.compute,
                config,
                json,
            )
            .await
        }
    }
}

async fn execute_sweep_remotely(
    source: &NativeCheckedFeaSource,
    sweep: ResolvedAnalysisSweep,
    cases: Vec<runmat_analysis_core::AuthoredSweepCase>,
    target: RemoteExecutionTarget,
    compute: FeaComputeAllocation,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    let frozen = source
        .frozen_project
        .as_ref()
        .context("remote typed FEA requires an enclosing frozen RunMat project")?;
    let placeholder = prepare_sweep_root(
        source,
        &sweep,
        &cases,
        "bundle-sizing-authority",
        &compute,
        config,
    )
    .await?;
    let saved_expectation = placeholder.root.result_expectation();
    let sweep_id = sweep.sweep.sweep_id.clone();
    let case_ids = cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect::<Vec<_>>();
    let revision = placeholder.revision.clone();
    let mut builder =
        ExecutionBundleBuilder::native(frozen, revision.clone())?.with_compiled_package_closure();
    for program in placeholder.root.required_programs() {
        builder = builder.with_materialized_program(
            program.recipe,
            program.artifact.form,
            program.artifact.executable_bytes,
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
            let root = placeholder
                .root
                .rebind_artifact_access(super::preparation::artifact_access(run_id))?;
            let remote = super::remote::prepare_typed_remote_sweep_execution(
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
            output_intent: SavedOutputIntent::FeaSweep {
                sweep_id,
                case_ids,
                expectation: Box::new(saved_expectation),
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
                presentation::stdout().success("FEA sweep job"),
                committed.id,
                committed.state
            );
        }
        return Ok(());
    }
    crate::commands::job::attach::attach(&committed.id, false, json).await
}

struct PreparedTypedSweep {
    root: runmat_execution_runner_native::PreparedStructuralSweepRoot,
    revision: runmat_execution::ProgramRevision,
}

async fn prepare_sweep_root(
    source: &NativeCheckedFeaSource,
    sweep: &ResolvedAnalysisSweep,
    cases: &[runmat_analysis_core::AuthoredSweepCase],
    authority: &str,
    compute: &FeaComputeAllocation,
    config: &RunMatRuntimeConfig,
) -> Result<PreparedTypedSweep> {
    let mut studies = Vec::with_capacity(cases.len());
    for case in cases {
        let inputs = case
            .bindings
            .iter()
            .cloned()
            .map(|binding| AuthoredStudyInputBinding {
                input_id: binding.input_id,
                value: binding.value,
            })
            .collect::<Vec<_>>();
        let prepared = prepare_study(
            source,
            sweep.study.clone(),
            &inputs,
            authority,
            compute,
            config,
        )
        .await?;
        studies.push(prepared.root);
    }
    let revision = studies
        .first()
        .map(|study| study.request().recipe.program_revision.clone())
        .context("typed FEA sweep contains no prepared studies")?;
    let identities = studies
        .iter()
        .map(|study| {
            runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(
                study.authored_study_digest(),
                None,
            )
        })
        .collect::<runmat_analysis_fea::execution::AnalysisExecutionResult<Vec<_>>>()?;
    let authored = serde_json::to_vec(&sweep.sweep)?;
    let definition = runmat_analysis_fea::execution::SweepDefinition::new(
        runmat_execution::Digest::sha256(
            [
                b"runmat-typed-fea-sweep/v1\0".as_slice(),
                authored.as_slice(),
            ]
            .concat(),
        ),
        identities,
        runmat_analysis_fea::execution::SweepCompositionPolicy::CollectAll,
    )?;
    let root = runmat_execution_runner_native::prepare_structural_sweep_root(
        runmat_execution_runner_native::StructuralSweepRootPreparation {
            program_revision: revision.clone(),
            artifact_access: super::preparation::artifact_access(authority),
            definition,
            studies,
        },
    )?;
    Ok(PreparedTypedSweep { root, revision })
}

async fn execute_study_remotely(
    source: &NativeCheckedFeaSource,
    study: ResolvedAnalysisStudy,
    inputs: Vec<AuthoredStudyInputBinding>,
    target: RemoteExecutionTarget,
    compute: FeaComputeAllocation,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
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
    let saved_expectation = placeholder.root.result_expectation();
    let study_id = placeholder.realized.study_id.clone();
    let revision = placeholder.revision.clone();
    let mut builder =
        ExecutionBundleBuilder::native(frozen, revision.clone())?.with_compiled_package_closure();
    for program in placeholder.root.required_programs() {
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
            let root = placeholder
                .root
                .rebind_artifact_access(super::preparation::artifact_access(run_id))?;
            let remote = super::remote::prepare_typed_remote_execution(
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

struct ResolvedFeaExecution {
    target: ExecutionTarget,
    compute: FeaComputeAllocation,
}

fn resolve_execution(
    source: &NativeCheckedFeaSource,
    root: &ResolvedFeaRoot,
    args: &ExecutionTargetArgs,
) -> Result<ResolvedFeaExecution> {
    let policy = match root {
        ResolvedFeaRoot::Study(root) => root.execution_policy.as_ref(),
        ResolvedFeaRoot::Sweep(root) => root.study.execution_policy.as_ref(),
    };
    let profile_name = policy.and_then(|policy| match &policy.placement {
        runmat_analysis_core::PlacementPreference::Cluster { profile } => profile.as_deref(),
        runmat_analysis_core::PlacementPreference::Auto
        | runmat_analysis_core::PlacementPreference::Local => None,
    });
    let profile = profile_name.and_then(|name| {
        source
            .project_manifest
            .as_ref()
            .and_then(|manifest| manifest.execution_profiles.get(name))
    });
    if profile_name.is_some() && profile.is_none() {
        bail!(
            "execution profile `{}` is unavailable",
            profile_name.expect("checked above")
        );
    }
    let target = if args.cluster.is_some() {
        args.resolve()?
    } else if matches!(
        policy.map(|policy| &policy.placement),
        Some(runmat_analysis_core::PlacementPreference::Cluster { .. })
    ) {
        if args.project.is_some()
            || args.queue.is_some()
            || args.trust_identity.is_some()
            || args.idempotency_key.is_some()
            || args.workers.is_some()
            || args.detach
        {
            bail!("remote CLI options require an explicit --cluster override");
        }
        let profile_name = profile_name.context(
            "cluster placement without an execution profile requires an explicit --cluster override",
        )?;
        let route = profile.context("execution profile disappeared during placement")?;
        let trust_identity = route.trust_identity.clone().with_context(|| {
            format!("execution profile `{profile_name}` has no pinned trust identity")
        })?;
        let workers = policy
            .and_then(|policy| policy.resources.workers)
            .map_or(1, |range| range.maximum.max(1));
        ExecutionTarget::Remote(RemoteExecutionTarget {
            cluster: route.cluster.clone(),
            project: None,
            queue: route.queue.clone().unwrap_or_else(|| "default".into()),
            trust_identity,
            idempotency_key: None,
            workers,
            detach: false,
        })
    } else {
        args.resolve()?
    };
    let allocation_profile = match (&target, profile) {
        (ExecutionTarget::Remote(target), Some(profile)) if target.cluster == profile.cluster => {
            Some(profile)
        }
        (ExecutionTarget::Remote(_), _) => None,
        (ExecutionTarget::Local, profile) => profile,
    };
    let compute = resolve_compute_allocation(policy, &target, allocation_profile)?;
    Ok(ResolvedFeaExecution { target, compute })
}

fn resolve_compute_allocation(
    policy: Option<&runmat_analysis_core::AuthoredExecutionPolicy>,
    target: &ExecutionTarget,
    profile: Option<&runmat_config::project::ProjectExecutionProfile>,
) -> Result<FeaComputeAllocation> {
    use runmat_analysis_core::{AcceleratorPreference, BackendPreference};

    let backend = policy.map_or(BackendPreference::Auto, |policy| policy.backend);
    let preference = policy.map_or(AcceleratorPreference::Disabled, |policy| {
        policy.resources.accelerators
    });
    if backend == BackendPreference::Cpu || preference == AcceleratorPreference::Disabled {
        return Ok(FeaComputeAllocation {
            backend: runmat_analysis_fea::ComputeBackend::Cpu,
            accelerators: Vec::new(),
        });
    }
    let required =
        backend == BackendPreference::Accelerator || preference == AcceleratorPreference::Required;
    let allocation = match target {
        ExecutionTarget::Local => runmat_execution_runner_native::active_provider_resource()
            .map_err(anyhow::Error::new),
        ExecutionTarget::Remote(_) => profile
            .and_then(|profile| profile.accelerator.as_ref())
            .map(|accelerator| runmat_execution::resource::AcceleratorRequest {
                class: accelerator.class.clone(),
                count: 1,
                memory_bytes_each: accelerator.memory_bytes,
            })
            .context(
                "remote accelerator execution requires an exact accelerator allocation in the selected execution profile",
            ),
    };
    match allocation {
        Ok(request) => {
            request.validate().map_err(anyhow::Error::msg)?;
            Ok(FeaComputeAllocation {
                backend: runmat_analysis_fea::ComputeBackend::Gpu,
                accelerators: vec![request],
            })
        }
        Err(_) if !required => Ok(FeaComputeAllocation {
            backend: runmat_analysis_fea::ComputeBackend::Cpu,
            accelerators: Vec::new(),
        }),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests;
