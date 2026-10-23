//! File-backed operator inputs for the production paired compiler.
//! The independent host admission is never generated from the pack.

use anyhow::{bail, Result};
use runmat_execution::host::{NativeAbiProbe, NativeDeploymentServiceKind};
use runmat_execution_artifact::target_pack::TargetPackRequirements;

use super::admission::{self, SelectedHostAdmission, SelectedPaths};
use super::{
    CompiledSelectedNativePair, NativePairArtifactRecipes, PersistedNativePair,
    ReopenSelectedPairedRequest, SelectedPairedLinkRequest, SelectedTargetPack,
};
use crate::link::PairedLinkRequests;
use crate::NativeObjectOptions;
use std::path::{Path, PathBuf};

pub struct OperatorPairReopenRequest<'a> {
    pub application_installation: &'a Path,
    pub test_installation: &'a Path,
    pub application_image: PathBuf,
    pub test_image: PathBuf,
}

/// Load exact selected bytes and independently granted build trees, derive
/// reached requirements from the whole frozen application/test inventory,
/// then use the ordinary paired compiler and qualified linker.
pub fn compile_operator_selected_pair(
    unit: &runmat_core::ExecutableUnit,
    tests: &runmat_test::plan::TestPlan,
    paths: &SelectedPaths,
    mut options: NativeObjectOptions,
    recipes: NativePairArtifactRecipes,
    publication: PairedLinkRequests<'_>,
) -> Result<CompiledSelectedNativePair> {
    let host = SelectedHostAdmission::load(&paths.host_admission)?;
    if options.clock_inputs.is_some() && options.clock_inputs != host.clock_inputs {
        bail!("paired clock inputs differ from the independent host admission");
    }
    options.clock_inputs.clone_from(&host.clock_inputs);
    if options.local_service_inputs.is_some()
        && options.local_service_inputs != host.local_service_inputs
    {
        bail!("paired local service inputs differ from the independent host admission");
    }
    options
        .local_service_inputs
        .clone_from(&host.local_service_inputs);
    let manifest = admission::load_manifest(&paths.manifest)?;
    let probe = admission::read_component(
        &paths.components,
        &manifest,
        &manifest.abi_probe_component,
        &host.link.storage_grant(),
    )?;
    let target = runmat_native_codegen::NativeTarget::Native {
        target: Box::new(host.expected_target.clone()),
        probe: Box::new(NativeAbiProbe::from_canonical_bytes(&probe)?),
    };
    target.validate()?;
    let assembly = unit
        .prepare_native_compilation_for_with_interop(None, options.interop.clone())?
        .lower(target)?;
    let required_services = assembly.required_runtime_services();
    if required_services.contains(&NativeDeploymentServiceKind::SourceEvaluation)
        && !host.runtime_capabilities.dynamic_source_loading
    {
        bail!("reached SourceEvaluation requires independently admitted dynamic-source capability");
    }
    let environment = unit.revision().program_revision.environment();
    let requirements = TargetPackRequirements {
        profile: host.profile,
        target: host.expected_target.clone(),
        abi: host.expected_abi.clone(),
        runtime: environment.runtime_fingerprint,
        catalog: environment.catalog_fingerprint,
        numerical: host.numerical.clone(),
        optimization: options.optimization,
        reached_capabilities: assembly.requirements.capabilities.clone(),
        authorized_runtime_capabilities: host.authorized_runtime_capabilities.clone(),
        effects: assembly.required_effects(),
        required_services,
        components: host.additional_components.clone(),
        platform_service_abi: host.platform_service_abi,
        service_abis: host.service_abis(),
    };
    let numerical = host.numerical_qualification();
    let selection = manifest.select(&requirements, numerical.as_ref())?;
    let contents = admission::read_selected_components(
        &paths.components,
        &manifest,
        selection.components(),
        &host.link.storage_grant(),
    )?;
    let verified = manifest.verify_selected_bytes(&selection, &contents)?;
    let selected = runmat_native_codegen::aot::SelectedNativeProduct::from_verified(&verified)?;
    let product = selected
        .product()
        .selected()
        .expect("verified selected constructor")
        .fingerprint()?;
    let link = host.load_link(product)?;
    super::compile_selected_target_pack_pair(
        unit,
        tests,
        SelectedTargetPack {
            manifest: &manifest,
            requirements: &requirements,
            numerical_admission: numerical.as_ref(),
            contents: &contents,
        },
        options,
        host.runtime_capabilities.clone(),
        recipes,
        SelectedPairedLinkRequest {
            admission: &link.admission,
            hosted: link.hosted(),
            publication,
        },
    )
    .map_err(Into::into)
}

/// Rebuild the exact portable products and reopen already-linked role images.
/// Each image requires a separately issued installation admission; the
/// persisted pair cannot authorize its own physical bytes.
pub fn reopen_operator_selected_pair(
    unit: &runmat_core::ExecutableUnit,
    tests: &runmat_test::plan::TestPlan,
    paths: &SelectedPaths,
    mut options: NativeObjectOptions,
    recipes: NativePairArtifactRecipes,
    persisted: &PersistedNativePair,
    reopen: OperatorPairReopenRequest<'_>,
) -> Result<CompiledSelectedNativePair> {
    let host = SelectedHostAdmission::load(&paths.host_admission)?;
    if options.clock_inputs.is_some() && options.clock_inputs != host.clock_inputs {
        bail!("paired clock inputs differ from the independent host admission");
    }
    options.clock_inputs.clone_from(&host.clock_inputs);
    if options.local_service_inputs.is_some()
        && options.local_service_inputs != host.local_service_inputs
    {
        bail!("paired local service inputs differ from the independent host admission");
    }
    options
        .local_service_inputs
        .clone_from(&host.local_service_inputs);
    let manifest = admission::load_manifest(&paths.manifest)?;
    let probe = admission::read_component(
        &paths.components,
        &manifest,
        &manifest.abi_probe_component,
        &host.link.storage_grant(),
    )?;
    let target = runmat_native_codegen::NativeTarget::Native {
        target: Box::new(host.expected_target.clone()),
        probe: Box::new(NativeAbiProbe::from_canonical_bytes(&probe)?),
    };
    target.validate()?;
    let assembly = unit
        .prepare_native_compilation_for_with_interop(None, options.interop.clone())?
        .lower(target)?;
    let required_services = assembly.required_runtime_services();
    if required_services.contains(&NativeDeploymentServiceKind::SourceEvaluation)
        && !host.runtime_capabilities.dynamic_source_loading
    {
        bail!("reached SourceEvaluation requires independently admitted dynamic-source capability");
    }
    let environment = unit.revision().program_revision.environment();
    let requirements = TargetPackRequirements {
        profile: host.profile,
        target: host.expected_target.clone(),
        abi: host.expected_abi.clone(),
        runtime: environment.runtime_fingerprint,
        catalog: environment.catalog_fingerprint,
        numerical: host.numerical.clone(),
        optimization: options.optimization,
        reached_capabilities: assembly.requirements.capabilities.clone(),
        authorized_runtime_capabilities: host.authorized_runtime_capabilities.clone(),
        effects: assembly.required_effects(),
        required_services,
        components: host.additional_components.clone(),
        platform_service_abi: host.platform_service_abi,
        service_abis: host.service_abis(),
    };
    let numerical = host.numerical_qualification();
    let selection = manifest.select(&requirements, numerical.as_ref())?;
    let contents = admission::read_selected_components(
        &paths.components,
        &manifest,
        selection.components(),
        &host.link.storage_grant(),
    )?;
    let verified = manifest.verify_selected_bytes(&selection, &contents)?;
    let selected = runmat_native_codegen::aot::SelectedNativeProduct::from_verified(&verified)?;
    let product = selected
        .product()
        .selected()
        .expect("verified selected constructor")
        .fingerprint()?;
    let link = host.load_link(product)?;
    let application_installation =
        admission::load_installation_admission(reopen.application_installation)?;
    let test_installation = admission::load_installation_admission(reopen.test_installation)?;
    super::reopen_selected_target_pack_pair(
        unit,
        tests,
        SelectedTargetPack {
            manifest: &manifest,
            requirements: &requirements,
            numerical_admission: numerical.as_ref(),
            contents: &contents,
        },
        options,
        host.runtime_capabilities.clone(),
        recipes,
        ReopenSelectedPairedRequest {
            admission: &link.admission,
            persisted,
            application_installation: &application_installation,
            test_installation: &test_installation,
            application_path: reopen.application_image,
            test_path: reopen.test_image,
        },
    )
    .map_err(Into::into)
}
