//! Explicit selected-pack compilation. The ordinary embedded archive route is
//! unchanged; this path never downgrades a failed selected selection to it.

use runmat_aot::selected_admission as admission;

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};
use runmat_aot::archive::{build_selected_runtime_archive, RuntimeArchiveEncoding};
use runmat_aot::compile::build_program_link_plan_with_interop;
use runmat_aot::link::{link_selected_standalone, LinkRequest};
use runmat_core::foreign_artifacts::PreparedForeignArtifacts;
use runmat_execution::host::NativeDeploymentServiceKind;
use runmat_execution_artifact::target_pack::TargetPackRequirements;
use runmat_native_codegen::aot::{NativeOptimization, SelectedNativeProduct};
use runmat_types::InteropManifest;

use crate::cli::AotOptLevel;
use crate::presentation;

pub(super) use admission::SelectedPaths;

pub(super) struct SelectedCompileRequest {
    pub paths: SelectedPaths,
    pub output: PathBuf,
    pub optimization: AotOptLevel,
    pub policy: runmat_aot::compile::CompilationPolicy,
    pub explain_link: bool,
    pub link_plan_json: Option<PathBuf>,
    pub keep_temps: bool,
    pub force: bool,
}

pub(super) fn execute(
    unit: &runmat_core::ExecutableUnit,
    interop: InteropManifest,
    foreign_artifacts: PreparedForeignArtifacts,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    request: SelectedCompileRequest,
) -> Result<()> {
    let host = admission::SelectedHostAdmission::load(&request.paths.host_admission)?;
    let manifest = admission::load_manifest(&request.paths.manifest)?;
    let numerical_qualification = host.numerical_qualification();
    let optimization = match request.optimization {
        AotOptLevel::None => NativeOptimization::None,
        AotOptLevel::Size => NativeOptimization::SpeedAndSize,
        AotOptLevel::Speed => NativeOptimization::Speed,
    };
    let probe_bytes = admission::read_component(
        &request.paths.components,
        &manifest,
        &manifest.abi_probe_component,
        &host.link.storage_grant(),
    )?;
    let probe = runmat_execution::host::NativeAbiProbe::from_canonical_bytes(&probe_bytes)?;
    let expected_target = runmat_native_codegen::NativeTarget::Native {
        target: Box::new(host.expected_target.clone()),
        probe: Box::new(probe),
    };
    expected_target.validate()?;
    let (_, retained) = unit.reachability_analysis();
    let retained = retained.into_iter().collect::<BTreeSet<_>>();
    let assembly = unit
        .prepare_native_compilation_for_with_interop(None, interop.clone())?
        .retain_functions(&retained)?
        .lower(expected_target)?;
    let effects = assembly.required_effects();
    let required_services = assembly.required_runtime_services();
    if required_services.contains(&NativeDeploymentServiceKind::SourceEvaluation)
        && !host.runtime_capabilities.dynamic_source_loading
    {
        bail!("reached SourceEvaluation requires the independently admitted dynamic-source runtime capability");
    }
    let requirements = TargetPackRequirements {
        profile: host.profile,
        target: host.expected_target.clone(),
        abi: host.expected_abi.clone(),
        runtime: unit
            .revision()
            .program_revision
            .environment()
            .runtime_fingerprint,
        catalog: unit
            .revision()
            .program_revision
            .environment()
            .catalog_fingerprint,
        numerical: host.numerical.clone(),
        optimization,
        reached_capabilities: assembly.requirements.capabilities.clone(),
        authorized_runtime_capabilities: host.authorized_runtime_capabilities.clone(),
        effects,
        required_services,
        components: host.additional_components.clone(),
        platform_service_abi: host.platform_service_abi,
        service_abis: host.service_abis(),
    };
    let selection = manifest.select(&requirements, numerical_qualification.as_ref())?;
    let components = admission::read_selected_components(
        &request.paths.components,
        &manifest,
        selection.components(),
        &host.link.storage_grant(),
    )?;
    let verified = manifest.verify_selected_bytes(&selection, &components)?;
    let selected = SelectedNativeProduct::from_verified(&verified)?;
    // Nothing beyond exact selected byte admission can choose the runtime or
    // target from here. The archive, object, and link share this borrow.
    let environment = unit.revision().program_revision.environment();
    let runtime = build_selected_runtime_archive(
        &selected,
        &environment,
        RuntimeArchiveEncoding::Zstd,
        host.runtime_capabilities.clone(),
    )?;
    request.policy.validate(&runtime.manifest.capabilities)?;
    let plan = build_program_link_plan_with_interop(unit, &runtime, request.policy, interop)?;
    if request.explain_link {
        super::print_link_explanation(&plan);
    }
    if let Some(path) = request.link_plan_json.as_deref() {
        super::write_link_plan(path, &plan, request.force)?;
    }
    if plan.retained_functions() != retained {
        bail!("selected compiler retention differs from the canonical link plan");
    }
    let options = runmat_aot::NativeObjectOptions {
        clock_inputs: host.clock_inputs.clone(),
        local_service_inputs: host.local_service_inputs.clone(),
        optimization,
        retained_functions: Some(retained),
        runtime_binding_mode: plan.runtime_composition.binding_mode(),
        retained_builtin_bindings: plan.runtime_composition.bindings().to_vec(),
        interop: plan.interop.clone(),
        foreign_policy: Some(
            runmat_core::foreign_artifacts::capture_foreign_policy_object(&config.foreign)
                .map_err(anyhow::Error::msg)?,
        ),
        native_interfaces: foreign_artifacts.native_interfaces,
        mex_artifacts: foreign_artifacts.mex_artifacts,
        java_artifacts: foreign_artifacts.java_artifacts,
        python_artifacts: foreign_artifacts.python_artifacts,
    };
    let object = runmat_aot::emit_selected_native_object(unit, &selected, options)?;
    let product = selected
        .product()
        .selected()
        .ok_or_else(|| anyhow!("selected target-pack bytes lost selected product mode"))?
        .fingerprint()?;
    let link = host.load_link(product)?;
    let linked = link_selected_standalone(
        &selected,
        &object,
        &runtime,
        &link.admission,
        link.hosted(),
        &LinkRequest {
            output: request.output,
            linker: None,
            keep_temps: request.keep_temps,
            overwrite: request.force,
        },
    )?;
    println!(
        "{} {}",
        presentation::stdout().success("Compiled"),
        linked.output.display()
    );
    if let Some(temporary) = linked.retained_temps {
        println!("Temporary link inputs: {}", temporary.display());
    }
    Ok(())
}
