//! Compose frozen compilation, portable artifacts and qualified paired links.
//! Individual owners retain their validation and publication responsibilities.

use runmat_execution_artifact::target_test::NativePairedArtifactManifest;
use runmat_execution_artifact::{ProgramArtifact, ProgramBuildRecipe};
use runmat_test::plan::TestPlan;

use super::{with_verified_product, SelectedTargetPack};
use crate::archive::{
    build_selected_runtime_archive, RuntimeArchiveCapabilities, RuntimeArchiveEncoding,
};
use crate::installation::ExternalNativeInstallationAdmission;
use crate::link::{
    ExternalNativeLinkAdmission, HostedNativeBuildTrees, LinkedNativePair, PairedLinkRequests,
    PersistedSelectedImage,
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

pub const PERSISTED_NATIVE_PAIR_SCHEMA_VERSION: u16 = 1;
const MAX_PERSISTED_NATIVE_PAIR_BYTES: usize = 32 * 1024 * 1024;

/// Durable physical-link receipt around the existing portable paired
/// manifest. It contains no execution, target-pack, or link authority.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistedNativePair {
    pub schema_version: u16,
    pub manifest: NativePairedArtifactManifest,
    pub application: PersistedSelectedImage,
    pub tests: PersistedSelectedImage,
}

impl PersistedNativePair {
    /// Ingress readers must enforce this bound before materializing the wire.
    pub const MAX_ENCODED_BYTES: usize = MAX_PERSISTED_NATIVE_PAIR_BYTES;

    pub fn validate(&self) -> AotResult<()> {
        self.manifest.validate_declaration().map_err(invalid)?;
        self.application.validate()?;
        self.tests.validate()?;
        if self.schema_version != PERSISTED_NATIVE_PAIR_SCHEMA_VERSION
            || self.application.product != self.manifest.binding.selected_product
            || self.tests.product != self.manifest.binding.selected_product
            || self.application.output_digest == self.tests.output_digest
        {
            return Err(invalid(
                "persisted native pair disagrees with its portable paired manifest",
            ));
        }
        Ok(())
    }

    pub fn to_canonical_bytes(&self) -> AotResult<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(invalid)?;
        if bytes.len() > MAX_PERSISTED_NATIVE_PAIR_BYTES {
            return Err(invalid("persisted native pair exceeds its wire bound"));
        }
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AotResult<Self> {
        if bytes.len() > MAX_PERSISTED_NATIVE_PAIR_BYTES {
            return Err(invalid("persisted native pair exceeds its wire bound"));
        }
        let value: Self = serde_json::from_slice(bytes).map_err(invalid)?;
        if value.to_canonical_bytes()?.as_slice() != bytes {
            return Err(invalid("persisted native pair is not canonically encoded"));
        }
        Ok(value)
    }
}

pub(crate) struct ReopenSelectedPairedRequest<'a> {
    pub admission: &'a ExternalNativeLinkAdmission,
    pub persisted: &'a PersistedNativePair,
    pub application_installation: &'a ExternalNativeInstallationAdmission,
    pub test_installation: &'a ExternalNativeInstallationAdmission,
    pub application_path: std::path::PathBuf,
    pub test_path: std::path::PathBuf,
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

    pub fn persisted(&self) -> AotResult<PersistedNativePair> {
        let persisted = PersistedNativePair {
            schema_version: PERSISTED_NATIVE_PAIR_SCHEMA_VERSION,
            manifest: self.artifacts.manifest.clone(),
            application: self.linked.application().persisted_selected_image()?,
            tests: self.linked.tests().persisted_selected_image()?,
        };
        persisted.validate()?;
        Ok(persisted)
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
    validate_recipes(unit, &recipes)?;
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

/// Rebuild the exact portable pair from the frozen source and verified target
/// pack, then reopen physical image bytes without invoking the linker again.
/// Neither the persisted envelope nor its paths can mint target/link authority.
pub(crate) fn reopen_selected_target_pack_pair(
    unit: &runmat_core::ExecutableUnit,
    tests: &TestPlan,
    pack: SelectedTargetPack<'_>,
    options: NativeObjectOptions,
    capabilities: RuntimeArchiveCapabilities,
    recipes: NativePairArtifactRecipes,
    reopen: ReopenSelectedPairedRequest<'_>,
) -> AotResult<CompiledSelectedNativePair> {
    validate_recipes(unit, &recipes)?;
    let persisted = reopen.persisted;
    persisted.validate()?;
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
        if artifacts.manifest != persisted.manifest {
            return Err(invalid(
                "rebuilt portable pair differs from persisted paired manifest",
            ));
        }
        persisted
            .manifest
            .validate_header_declarations(objects.application_header(), objects.test_header())
            .map_err(invalid)?;
        let application_commitment = crate::link::persisted::admit_completed_image(
            reopen.application_installation,
            selected,
            reopen.admission,
            objects.application(),
            &runtime,
            &persisted.application,
        )?;
        let test_commitment = crate::link::persisted::admit_completed_image(
            reopen.test_installation,
            selected,
            reopen.admission,
            objects.tests(),
            &runtime,
            &persisted.tests,
        )?;
        let linked = crate::link::paired::reopen_selected_native_pair(
            selected,
            &objects,
            &runtime,
            reopen.admission,
            crate::link::paired::PersistedPairReopenRequest {
                application_header: objects.application_header().clone(),
                test_header: objects.test_header().clone(),
                application_receipt: &persisted.application,
                test_receipt: &persisted.tests,
                application_path: reopen.application_path,
                test_path: reopen.test_path,
                application_commitment,
                test_commitment,
            },
        )?;
        Ok(CompiledSelectedNativePair { artifacts, linked })
    })
}

fn validate_recipes(
    unit: &runmat_core::ExecutableUnit,
    recipes: &NativePairArtifactRecipes,
) -> AotResult<()> {
    for recipe in [&recipes.application, &recipes.tests] {
        recipe.validate().map_err(invalid)?;
        if recipe.program_revision != unit.revision().program_revision {
            return Err(invalid(
                "paired recipe differs from frozen program revision",
            ));
        }
    }
    Ok(())
}

fn invalid(message: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.selected.paired", message.to_string())
}
