//! One selected program handoff: canonical source product, authenticated
//! target-pack bytes, selected object, selected archive, and qualified link.
//! The caller supplies independent target qualification and link permissions;
//! no decoded manifest or ambient tool path can mint those rights here.

use std::collections::BTreeMap;

use runmat_execution_artifact::target_pack::{
    ComponentId, ExternalNumericalAdmission, TargetPackManifest, TargetPackRequirements,
};
use runmat_native_codegen::aot::SelectedNativeProduct;

use crate::{
    archive::{build_selected_runtime_archive, RuntimeArchiveCapabilities, RuntimeArchiveEncoding},
    emit_selected_native_object,
    link::{
        link_selected_standalone, ExternalNativeLinkAdmission, HostedNativeBuildTrees, LinkRequest,
        LinkedProgram,
    },
    AotError, AotResult, NativeObjectOptions,
};

pub mod admission;
mod operator_pair;
pub use operator_pair::{
    compile_operator_selected_pair, reopen_operator_selected_pair, OperatorPairReopenRequest,
};
mod paired;
pub use paired::{
    compile_selected_target_pack_pair, CompiledSelectedNativePair, NativePairArtifactRecipes,
    PersistedNativePair, SelectedPairedLinkRequest, PERSISTED_NATIVE_PAIR_SCHEMA_VERSION,
};
pub(crate) use paired::{reopen_selected_target_pack_pair, ReopenSelectedPairedRequest};

pub struct SelectedTargetPack<'a> {
    pub manifest: &'a TargetPackManifest,
    pub requirements: &'a TargetPackRequirements,
    pub numerical_admission: Option<&'a ExternalNumericalAdmission>,
    pub contents: &'a BTreeMap<ComponentId, Vec<u8>>,
}

pub struct SelectedLinkRequest<'a> {
    pub admission: &'a ExternalNativeLinkAdmission,
    pub hosted: HostedNativeBuildTrees<'a>,
    pub publication: &'a LinkRequest,
}

/// Qualify one exact target-pack byte closure before compiling or linking.
/// Requirements, numerical admission, and link admission are separate trusted
/// host inputs; no field in the pack manifest grants any of those rights.
pub fn compile_selected_target_pack_standalone(
    unit: &runmat_core::ExecutableUnit,
    pack: SelectedTargetPack<'_>,
    options: NativeObjectOptions,
    capabilities: RuntimeArchiveCapabilities,
    link: SelectedLinkRequest<'_>,
) -> AotResult<LinkedProgram> {
    with_verified_product(pack, |selected| {
        compile_selected_standalone(
            unit,
            selected,
            options,
            capabilities,
            link.admission,
            link.hosted,
            link.publication,
        )
    })
}

fn with_verified_product<T>(
    pack: SelectedTargetPack<'_>,
    operation: impl FnOnce(&SelectedNativeProduct<'_>) -> AotResult<T>,
) -> AotResult<T> {
    let selection = pack
        .manifest
        .select(pack.requirements, pack.numerical_admission)
        .map_err(|error| AotError::contract("aot.selected.select", error.to_string()))?;
    let verified = pack
        .manifest
        .verify_selected_bytes(&selection, pack.contents)
        .map_err(|error| AotError::contract("aot.selected.bytes", error.to_string()))?;
    let selected = SelectedNativeProduct::from_verified(&verified)
        .map_err(|error| AotError::contract("aot.selected.product", error.to_string()))?;
    operation(&selected)
}

pub fn compile_selected_standalone(
    unit: &runmat_core::ExecutableUnit,
    selected: &SelectedNativeProduct<'_>,
    options: NativeObjectOptions,
    capabilities: RuntimeArchiveCapabilities,
    admission: &ExternalNativeLinkAdmission,
    hosted: HostedNativeBuildTrees<'_>,
    request: &LinkRequest,
) -> AotResult<LinkedProgram> {
    // The archive and object must be derived from the *same* verified pack
    // borrow. Neither independently decoded product metadata nor a caller
    // supplied target can be substituted between compilation and linking.
    let environment = unit.revision().program_revision.environment();
    let runtime = build_selected_runtime_archive(
        selected,
        &environment,
        RuntimeArchiveEncoding::Zstd,
        capabilities,
    )?;
    let object = emit_selected_native_object(unit, selected, options)?;
    link_selected_standalone(selected, &object, &runtime, admission, hosted, request)
}
