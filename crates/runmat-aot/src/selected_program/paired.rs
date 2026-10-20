//! Compose frozen compilation, portable artifacts and qualified paired links.
//! Individual owners retain their validation and publication responsibilities.

use runmat_execution_artifact::target_test::NativePairedArtifactManifest;
use runmat_execution_artifact::{ProgramArtifact, ProgramBuildRecipe};
use runmat_test::plan::TestPlan;

use super::{with_verified_product, SelectedTargetPack};
use crate::archive::{
    build_selected_runtime_archive, RuntimeArchiveCapabilities, RuntimeArchiveEncoding,
};
use crate::link::{
    ExternalNativeLinkAdmission, HostedNativeBuildTrees, LinkedNativePair, PairedLinkRequests,
};
use crate::{
    emit_selected_paired_native_objects, materialize_selected_paired_native_artifacts, AotError,
    AotResult, NativeObjectOptions, PairedNativeArtifacts,
};

pub struct NativePairArtifactRecipes {
    pub application: ProgramBuildRecipe,
    pub tests: ProgramBuildRecipe,
}

pub struct SelectedPairedLinkRequest<'a> {
    pub admission: &'a ExternalNativeLinkAdmission,
    pub hosted: HostedNativeBuildTrees<'a>,
    pub publication: PairedLinkRequests<'a>,
}

/// Exact portable object artifacts and actual linked images from one frozen
/// selected build. Construction does not execute or qualify either image.
pub struct CompiledSelectedNativePair {
    artifacts: PairedNativeArtifacts,
    linked: LinkedNativePair,
}

impl CompiledSelectedNativePair {
    pub fn manifest(&self) -> &NativePairedArtifactManifest {
        &self.artifacts.manifest
    }

    pub fn application_artifact(&self) -> &ProgramArtifact {
        &self.artifacts.application
    }

    pub fn test_artifact(&self) -> &ProgramArtifact {
        &self.artifacts.tests
    }

    pub fn linked(&self) -> &LinkedNativePair {
        &self.linked
    }
}

/// Build both roles using one verified pack and one frozen frontend product.
/// The same runtime archive and link admission are reused for both images.
/// As with the paired linker, a second-image failure may leave the completed
/// first image on disk; no atomic multi-file publication is claimed.
pub fn compile_selected_target_pack_pair(
    unit: &runmat_core::ExecutableUnit,
    tests: &TestPlan,
    pack: SelectedTargetPack<'_>,
    options: NativeObjectOptions,
    capabilities: RuntimeArchiveCapabilities,
    recipes: NativePairArtifactRecipes,
    link: SelectedPairedLinkRequest<'_>,
) -> AotResult<CompiledSelectedNativePair> {
    for recipe in [&recipes.application, &recipes.tests] {
        recipe.validate().map_err(invalid)?;
        if recipe.program_revision != unit.revision().program_revision {
            return Err(invalid(
                "paired recipe differs from frozen program revision",
            ));
        }
    }
    with_verified_product(pack, |selected| {
        let runtime = build_selected_runtime_archive(
            selected,
            &unit.revision().program_revision.environment(),
            RuntimeArchiveEncoding::Zstd,
            capabilities,
        )?;
        let objects = emit_selected_paired_native_objects(unit, selected, tests, options)?;
        let artifacts = materialize_selected_paired_native_artifacts(
            &objects,
            recipes.application,
            recipes.tests,
        )?;
        let linked = crate::link::link_selected_native_pair(
            selected,
            &objects,
            &runtime,
            link.admission,
            link.hosted,
            link.publication,
        )?;
        artifacts
            .manifest
            .validate_header_declarations(linked.application_header(), linked.test_header())
            .map_err(invalid)?;
        Ok(CompiledSelectedNativePair { artifacts, linked })
    })
}

fn invalid(message: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.selected.paired", message.to_string())
}
