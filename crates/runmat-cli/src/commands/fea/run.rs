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

use super::preparation::{portable_root, prepare_study, FeaComputeAllocation};
use super::result_presentation::{print_completed_fea_result, print_fea_sweep_result};
use super::source::NativeCheckedFeaSource;

mod remote_study;
use remote_study::execute_study_remotely;

pub(super) async fn execute(
    path: PathBuf,
    mut source: NativeCheckedFeaSource,
    mut root: ResolvedFeaRoot,
    execution: ExecutionTargetArgs,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    let mut resolved_execution = resolve_execution(&source, &root, &execution)?;
    let project_policy = if matches!(resolved_execution.target, ExecutionTarget::Remote(_)) {
        let frozen = source
            .frozen_project
            .as_ref()
            .context("cluster FEA requires a frozen project")?;
        Some(
            runmat_config::runtime::ClusterRuntimePolicy::from_project_file(&frozen.manifest_path)?,
        )
    } else {
        None
    };
    let project_runtime = project_policy
        .as_ref()
        .map(|policy| policy.runtime_config())
        .transpose()?;
    let config = project_runtime.as_ref().unwrap_or(config);
    if project_policy.is_some() {
        let selector = match &root {
            ResolvedFeaRoot::Study(study) => runmat_runtime::analysis::FeaRootSelector::Study {
                id: study.study.study_id.clone(),
            },
            ResolvedFeaRoot::Sweep(sweep) => runmat_runtime::analysis::FeaRootSelector::Sweep {
                id: sweep.sweep.sweep_id.clone(),
            },
        };
        source = super::source::check_source_for_cluster(&path, cli, config).await?;
        if !source.check_result().status.is_valid() {
            bail!("FEA source is invalid under project-only cluster language policy");
        }
        root = source
            .resolve_root(Some(&selector))
            .map_err(anyhow::Error::new)?;
        resolved_execution = resolve_execution(&source, &root, &execution)?;
    }
    resolved_execution.project_runtime_policy = project_policy;
    match root {
        ResolvedFeaRoot::Study(study) => {
            execute_study(&source, study, Vec::new(), resolved_execution, config, json).await
        }
        ResolvedFeaRoot::Sweep(sweep) => {
            execute_sweep(&source, sweep, resolved_execution, config, json).await
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
                    accelerator_placement: execution.compute.placement.clone(),
                    ..Default::default()
                },
            )
            .await
            .map_err(anyhow::Error::msg)?;
            match completed {
                runmat_execution_runner_native::LocalFeaExecution::Study(completed) => {
                    print_completed_fea_result(
                        &completed.case_id,
                        &completed.result,
                        &completed.publication,
                        json,
                    )
                }
                // A multi-frequency harmonic study runs one case per frequency.
                runmat_execution_runner_native::LocalFeaExecution::Sweep { sweep_id, cases } => {
                    print_local_sweep(&sweep_id, cases, json)
                }
            }
        }
        ExecutionTarget::Remote(target) => {
            execute_study_remotely(
                source,
                study,
                inputs,
                target,
                execution.compute,
                execution
                    .project_runtime_policy
                    .context("cluster FEA policy was not captured before compilation")?,
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
                    accelerator_placement: execution.compute.placement.clone(),
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
            print_local_sweep(&sweep.sweep.sweep_id, completed, json)
        }
        ExecutionTarget::Remote(target) => {
            execute_sweep_remotely(
                source,
                sweep,
                cases,
                target,
                execution.compute,
                execution
                    .project_runtime_policy
                    .context("cluster FEA policy was not captured before compilation")?,
                config,
                json,
            )
            .await
        }
    }
}

/// Print locally executed cases, each with its own terminal status, in authored order.
fn print_local_sweep(
    sweep_id: &str,
    completed: Vec<runmat_execution_runner_native::LocalSweepCase>,
    json: bool,
) -> Result<()> {
    let case_ids = completed
        .iter()
        .map(|case| case.case_id.clone())
        .collect::<Vec<_>>();
    let materialized = super::remote::MaterializedFeaSweep {
        entries: completed
            .into_iter()
            .enumerate()
            .map(
                |(study_index, case)| super::remote::MaterializedFeaSweepEntry {
                    study_index,
                    terminal: match case.outcome {
                        Ok(completed) => super::remote::MaterializedFeaSweepTerminal::Succeeded(
                            Box::new(super::remote::MaterializedFeaStudy {
                                result: completed.result,
                                publication: completed.publication,
                            }),
                        ),
                        Err(message) => {
                            super::remote::MaterializedFeaSweepTerminal::Failed(message)
                        }
                    },
                },
            )
            .collect(),
    };
    print_fea_sweep_result(sweep_id, &case_ids, &materialized, json)
}

#[expect(
    clippy::too_many_arguments,
    reason = "FEA alpha execution-context structure is deferred under the approved 1.0 hardening scope"
)]
async fn execute_sweep_remotely(
    source: &NativeCheckedFeaSource,
    sweep: ResolvedAnalysisSweep,
    cases: Vec<runmat_analysis_core::AuthoredSweepCase>,
    target: RemoteExecutionTarget,
    compute: FeaComputeAllocation,
    project_policy: runmat_config::runtime::ClusterRuntimePolicy,
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
    let mut saved_expectation = placeholder.root.result_expectation();
    for study in &mut saved_expectation.studies {
        study.accelerator_placement = compute.placement.clone();
    }
    let mut assessment_functions = Vec::with_capacity(saved_expectation.studies.len());
    for study in &saved_expectation.studies {
        assessment_functions.push(
            runmat_execution_runner_native::freeze_fea_assessment_functions(
                &source.session_source,
                &study.study.assessments,
            )
            .await
            .map_err(anyhow::Error::msg)?,
        );
    }
    let sweep_id = sweep.sweep.sweep_id.clone();
    let case_ids = cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect::<Vec<_>>();
    let revision = placeholder.revision.clone();
    runmat_core::cluster_settings::validate_program_environment(&project_policy, &revision)
        .map_err(anyhow::Error::msg)?;
    let runtime_policy = runmat_core::cluster_settings::capture_project_policy(&project_policy)
        .map_err(anyhow::Error::msg)?;
    let mut builder = ExecutionBundleBuilder::native(frozen, revision.clone())?
        .with_compiled_package_closure()
        .with_project_runtime_policy_object(runtime_policy)?;
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
            console: None,
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
        studies.push(portable_root(prepared.root)?);
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

struct ResolvedFeaExecution {
    target: ExecutionTarget,
    compute: FeaComputeAllocation,
    project_runtime_policy: Option<runmat_config::runtime::ClusterRuntimePolicy>,
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
        profile_name.context(
            "cluster placement without an execution profile requires an explicit --cluster override",
        )?;
        let route = profile.context("execution profile disappeared during placement")?;
        let trust_identity = crate::cli::resolve_endpoint_trust(route.trust_identity.as_deref())?;
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
    let steps = match root {
        ResolvedFeaRoot::Study(root) => &root.study.sequence.steps,
        ResolvedFeaRoot::Sweep(root) => &root.study.study.sequence.steps,
    };
    let compute = resolve_compute_allocation(policy, steps, &target, allocation_profile)?;
    Ok(ResolvedFeaExecution {
        target,
        compute,
        project_runtime_policy: None,
    })
}

/// Resolves the authored placement into one compute allocation through the shared accelerator
/// admission: a study whose families cannot run on the accelerator, or a host without an
/// allocatable accelerator, falls back to `cpu_reference` with the fallback diagnostic unless the
/// policy says `accelerators = required`, which fails closed.
fn resolve_compute_allocation(
    policy: Option<&runmat_analysis_core::AuthoredExecutionPolicy>,
    steps: &[runmat_analysis_core::ProcedureStep],
    target: &ExecutionTarget,
    profile: Option<&runmat_config::project::ProjectExecutionProfile>,
) -> Result<FeaComputeAllocation> {
    use runmat_analysis_core::{AcceleratorPreference, BackendPreference};
    use runmat_analysis_fea::accelerator_admission::{
        admit_study_accelerator, AcceleratorPlacement,
    };

    let cpu = |placement| FeaComputeAllocation {
        backend: runmat_analysis_fea::ComputeBackend::Cpu,
        accelerators: Vec::new(),
        placement,
    };
    let backend = policy.map_or(BackendPreference::Auto, |policy| policy.backend);
    let preference = policy.map_or(AcceleratorPreference::Disabled, |policy| {
        policy.resources.accelerators
    });
    let admission_policy = policy
        .map(|policy| policy.resources.accelerator_admission())
        .unwrap_or_default();
    if backend == BackendPreference::Cpu || preference == AcceleratorPreference::Disabled {
        return Ok(cpu(AcceleratorPlacement::default()));
    }
    let mut allocation = None;
    let admission = admit_study_accelerator(
        steps,
        runmat_analysis_fea::ComputeBackend::Gpu,
        admission_policy,
        || {
            let request = accelerator_request(target, profile)
                .and_then(|request| {
                    request.validate().map_err(anyhow::Error::msg)?;
                    Ok(request)
                })
                .map_err(|error| format!("no accelerator can be allocated ({error:#})"))?;
            allocation = Some(request);
            Ok(())
        },
    )
    .map_err(anyhow::Error::new)?;
    if let Some(fallback) = admission.fallback {
        eprintln!("warning: {}: {}", fallback.code, fallback.message);
        // The published study carries the same warning.
        return Ok(cpu(AcceleratorPlacement {
            policy: admission_policy,
            fallback: Some(fallback),
        }));
    }
    let request = allocation
        .context("accelerator admission admitted the device without an allocation request")?;
    Ok(FeaComputeAllocation {
        backend: runmat_analysis_fea::ComputeBackend::Gpu,
        accelerators: vec![request],
        placement: AcceleratorPlacement {
            policy: admission_policy,
            fallback: None,
        },
    })
}

fn accelerator_request(
    target: &ExecutionTarget,
    profile: Option<&runmat_config::project::ProjectExecutionProfile>,
) -> Result<runmat_execution::resource::AcceleratorRequest> {
    match target {
        ExecutionTarget::Local => runmat_execution_runner_native::active_provider_resource()
            .map_err(anyhow::Error::new),
        ExecutionTarget::Remote(_) => profile
            .and_then(|profile| profile.accelerator.as_ref())
            .context(
                "remote accelerator execution requires an exact accelerator allocation in the selected execution profile",
            )
            .and_then(|accelerator| {
                Ok(runmat_execution::resource::AcceleratorRequest {
                    class: runmat_execution::resource::AcceleratorClass::new(&accelerator.class)?,
                    count: 1,
                    minimum_allocation_bytes: accelerator.memory_bytes,
                    provider: None,
                    required_features: std::collections::BTreeSet::from([
                        runmat_execution::resource::AcceleratorFeature::Compute,
                    ]),
                })
            }),
    }
}

#[cfg(test)]
mod tests;
