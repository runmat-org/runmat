use anyhow::{bail, Context, Result};
use futures::StreamExt as _;
use runmat_analysis_core::AuthoredStudyInputBinding;
use runmat_analysis_language::{ResolvedAnalysisStudy, ResolvedAnalysisSweep};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::ExecutionBundleBuilder;
use serde::Serialize;
use std::path::PathBuf;

use crate::cli::{Cli, ExecutionTarget, ExecutionTargetArgs, RemoteExecutionTarget};
use crate::commands::job::secret::SavedOutputIntent;
use crate::commands::job::submit::{PreparedRemoteExecution, RemoteSubmissionOptions};
use crate::presentation;

use super::preparation::{prepare_study, FeaComputeAllocation};
use super::selection::SelectedFeaRoot;
use super::source::CheckedFeaSource;

pub(super) async fn execute(
    path: PathBuf,
    source: CheckedFeaSource,
    root: SelectedFeaRoot,
    execution: ExecutionTargetArgs,
    _cli: &Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    let execution = resolve_execution(&source, &root, &execution)?;
    match root {
        SelectedFeaRoot::Study(study) => {
            execute_study(&path, &source, study, Vec::new(), execution, config, json).await
        }
        SelectedFeaRoot::Sweep(sweep) => {
            execute_sweep(&path, &source, sweep, execution, config, json).await
        }
    }
}

async fn execute_study(
    path: &PathBuf,
    source: &CheckedFeaSource,
    study: ResolvedAnalysisStudy,
    inputs: Vec<AuthoredStudyInputBinding>,
    execution: ResolvedFeaExecution,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    match execution.target {
        ExecutionTarget::Local => {
            let authority = format!("typed-fea-local:{}", uuid::Uuid::new_v4());
            let prepared = prepare_study(
                source,
                study,
                &inputs,
                &authority,
                &execution.compute,
                config,
            )?;
            let study_id = prepared.realized.study_id.clone();
            let completed =
                runmat_execution_runner_native::execute_prepared_structural_study_locally(
                    runmat_execution_runner_native::NativeExecutionConfig::for_current_executable(
                    )?,
                    prepared.root,
                    prepared.workers,
                )
                .await?;
            print_local_result(&study_id, &completed, json)
        }
        ExecutionTarget::Remote(target) => {
            execute_study_remotely(
                path,
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
    _path: &PathBuf,
    source: &CheckedFeaSource,
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
            let case_count = cases.len();
            let mut prepared_cases = Vec::with_capacity(case_count);
            for (index, case) in cases.into_iter().enumerate() {
                let inputs = case
                    .bindings
                    .into_iter()
                    .map(|binding| AuthoredStudyInputBinding {
                        input_id: binding.input_id,
                        value: binding.value,
                    })
                    .collect::<Vec<_>>();
                let authority = format!(
                    "typed-fea-sweep:{}:{}",
                    sweep.sweep.sweep_id, case.case_id
                );
                let prepared =
                    prepare_study(
                        source,
                        sweep.study.clone(),
                        &inputs,
                        &authority,
                        &execution.compute,
                        config,
                    )?;
                prepared_cases.push((index, case.case_id, prepared));
            }
            let concurrent_cases = sweep_case_concurrency(&sweep, &prepared_cases);
            let mut completed = futures::stream::iter(prepared_cases.into_iter().map(
                |(index, case_id, prepared)| async move {
                    let result =
                        runmat_execution_runner_native::execute_prepared_structural_study_locally(
                            runmat_execution_runner_native::NativeExecutionConfig::for_current_executable(
                            )?,
                            prepared.root,
                            prepared.workers,
                        )
                        .await
                        .with_context(|| format!("execute sweep case `{case_id}`"))?;
                    Ok::<_, anyhow::Error>((index, local_summary(case_id, &result)))
                },
            ))
            .buffer_unordered(concurrent_cases)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
            completed.sort_by_key(|(index, _)| *index);
            let summaries = completed
                .into_iter()
                .map(|(_, summary)| summary)
                .collect::<Vec<_>>();
            print_sweep_result(&sweep.sweep.sweep_id, &summaries, json)
        }
        ExecutionTarget::Remote(_) => bail!(
            "typed remote sweeps require one atomic portable sweep root; per-case job submission is deliberately not substituted"
        ),
    }
}

fn sweep_case_concurrency(
    sweep: &ResolvedAnalysisSweep,
    cases: &[(usize, String, super::preparation::PreparedTypedStudy)],
) -> usize {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let requested = sweep
        .study
        .execution_policy
        .as_ref()
        .map(|policy| policy.fanout.sweep_cases)
        .and_then(|limit| match limit {
            runmat_analysis_core::ParallelismLimit::Auto => None,
            runmat_analysis_core::ParallelismLimit::Fixed(value) => Some(value as usize),
        })
        .unwrap_or(available);
    let workers_per_case = cases
        .iter()
        .map(|(_, _, prepared)| prepared.workers as usize)
        .max()
        .unwrap_or(1)
        .max(1);
    requested
        .min(available / workers_per_case.max(1))
        .max(1)
        .min(cases.len().max(1))
}

async fn execute_study_remotely(
    _path: &PathBuf,
    source: &CheckedFeaSource,
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
    )?;
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
            let prepared = prepare_study(source, study, &inputs, run_id, &compute, config)?;
            let remote = super::remote::prepare_typed_remote_execution(
                prepared.root,
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
            output_intent: SavedOutputIntent::Generic,
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
    source: &CheckedFeaSource,
    root: &SelectedFeaRoot,
    args: &ExecutionTargetArgs,
) -> Result<ResolvedFeaExecution> {
    let policy = match root {
        SelectedFeaRoot::Study(root) => root.execution_policy.as_ref(),
        SelectedFeaRoot::Sweep(root) => root.study.execution_policy.as_ref(),
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

#[derive(Serialize)]
struct LocalRunSummary {
    root_id: String,
    procedure: &'static str,
    mesh_digest: String,
    nodes: usize,
    elements: usize,
    boundary_faces: usize,
}

fn local_summary(
    root_id: impl Into<String>,
    completed: &runmat_execution_runner_native::CompletedLocalStructuralStudy,
) -> LocalRunSummary {
    LocalRunSummary {
        root_id: root_id.into(),
        procedure: numerical_kind(&completed.numerical),
        mesh_digest: stable_digest_hex(&completed.meshing.artifact.canonical_digest),
        nodes: completed.meshing.artifact.topology.nodes.len(),
        elements: completed.meshing.artifact.topology.volume_elements.len(),
        boundary_faces: completed.meshing.artifact.topology.boundary_faces.len(),
    }
}

fn print_local_result(
    study_id: &str,
    completed: &runmat_execution_runner_native::CompletedLocalStructuralStudy,
    json: bool,
) -> Result<()> {
    let summary = local_summary(study_id, completed);
    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    let styles = presentation::stdout();
    println!(
        "{} {}",
        styles.success("FEA run complete"),
        styles.identifier(study_id)
    );
    println!("  {} {}", styles.label("procedure:"), summary.procedure);
    println!("  {} {}", styles.label("mesh:"), summary.mesh_digest);
    println!(
        "  {} {} nodes, {} tetrahedra, {} boundary faces",
        styles.label("topology:"),
        summary.nodes,
        summary.elements,
        summary.boundary_faces
    );
    Ok(())
}

fn print_sweep_result(id: &str, summaries: &[LocalRunSummary], json: bool) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "sweep_id": id,
                "cases": summaries,
            }))?
        );
    } else {
        let styles = presentation::stdout();
        println!(
            "{} {} ({} cases)",
            styles.success("FEA sweep complete"),
            styles.identifier(id),
            summaries.len()
        );
        for summary in summaries {
            println!(
                "  {}: {} tetrahedra, {}",
                summary.root_id, summary.elements, summary.procedure
            );
        }
    }
    Ok(())
}

fn stable_digest_hex(digest: &runmat_meshing_core::StableDigest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn numerical_kind(
    result: &runmat_runtime::analysis::StructuralStudyNumericalResult,
) -> &'static str {
    use runmat_runtime::analysis::StructuralStudyNumericalResult;
    match result {
        StructuralStudyNumericalResult::LinearSequence(_) => "structural_linear_sequence",
        StructuralStudyNumericalResult::StructuralModal(_) => "structural_modal",
        StructuralStudyNumericalResult::StructuralModalTransient(_) => "structural_modal_transient",
        StructuralStudyNumericalResult::StructuralModalResponseSpectrum(_) => {
            "structural_modal_response_spectrum"
        }
        StructuralStudyNumericalResult::StructuralRandomResponse(_) => "structural_random_response",
        StructuralStudyNumericalResult::StructuralBuckling(_) => "structural_buckling",
        StructuralStudyNumericalResult::StructuralPrestressedModal(_) => {
            "structural_prestressed_modal"
        }
        StructuralStudyNumericalResult::StructuralPrestressedFrequency(_) => {
            "structural_prestressed_frequency"
        }
        StructuralStudyNumericalResult::StructuralTransient(_) => "structural_transient",
        StructuralStudyNumericalResult::StructuralFrequency(_) => "structural_frequency",
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AcceleratorPreference, AuthoredExecutionPolicy, BackendPreference, CooperativePolicy,
        ExecutionResourceLimits, FanoutPolicy, PlacementPreference, ResiliencePolicy,
    };
    use runmat_config::project::{ProjectAcceleratorProfile, ProjectExecutionProfile};

    use super::*;

    fn remote_target() -> ExecutionTarget {
        ExecutionTarget::Remote(RemoteExecutionTarget {
            cluster: "gpu-cluster".into(),
            project: None,
            queue: "gpu".into(),
            trust_identity: "sha256:worker".into(),
            idempotency_key: None,
            workers: 2,
            detach: false,
        })
    }

    fn policy(accelerators: AcceleratorPreference) -> AuthoredExecutionPolicy {
        AuthoredExecutionPolicy {
            module_path: "project/main.fea".into(),
            policy_id: "scale".into(),
            placement: PlacementPreference::Cluster {
                profile: Some("gpu".into()),
            },
            backend: BackendPreference::Accelerator,
            resources: ExecutionResourceLimits {
                workers: None,
                maximum_memory_bytes_per_worker: None,
                accelerators,
            },
            fanout: FanoutPolicy::default(),
            cooperative: CooperativePolicy::default(),
            resilience: ResiliencePolicy::default(),
        }
    }

    #[test]
    fn remote_accelerator_planning_requires_and_retains_exact_profile_allocation() {
        let profile = ProjectExecutionProfile {
            cluster: "gpu-cluster".into(),
            queue: Some("gpu".into()),
            trust_identity: Some("sha256:worker".into()),
            accelerator: Some(ProjectAcceleratorProfile {
                class: "metal".into(),
                memory_bytes: 16 << 30,
            }),
        };
        let allocation = resolve_compute_allocation(
            Some(&policy(AcceleratorPreference::Required)),
            &remote_target(),
            Some(&profile),
        )
        .unwrap();
        assert_eq!(allocation.backend, runmat_analysis_fea::ComputeBackend::Gpu);
        assert_eq!(allocation.accelerators.len(), 1);
        assert_eq!(allocation.accelerators[0].class, "metal");
        assert_eq!(allocation.accelerators[0].memory_bytes_each, 16 << 30);

        assert!(resolve_compute_allocation(
            Some(&policy(AcceleratorPreference::Required)),
            &remote_target(),
            None,
        )
        .is_err());
    }
}
