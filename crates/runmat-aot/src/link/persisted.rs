//! Durable selected-image evidence and independently admitted reopening.
//!
//! Decoded evidence never authorizes an image. A reopening commitment is
//! issued only from the existing external installation-admission owner.

use std::path::PathBuf;

use runmat_execution::Digest;
use runmat_native_codegen::aot::{RelocatableNativeObject, SelectedNativeProduct};

use super::{
    CanonicalSelectedLinkPlan, ExternalNativeLinkAdmission, LinkedProgram, NativeLinkBuildIdentity,
    NativeProgramLinkIdentity, VerifiedSelectedImage,
};
use crate::installation::ExternalNativeInstallationAdmission;
use crate::{AotError, AotResult};

pub const PERSISTED_SELECTED_IMAGE_SCHEMA_VERSION: u16 = 1;
const MAX_PERSISTED_SELECTED_IMAGE_BYTES: usize = 16 * 1024 * 1024;

/// Portable evidence describing one completed selected link. Its output
/// digest is observational; reopening also requires an independent trusted
/// installation admission carrying the same completed-image commitment.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistedSelectedImage {
    pub schema_version: u16,
    pub canonical_plan: CanonicalSelectedLinkPlan,
    pub canonical_plan_identity: Digest,
    pub output_digest: Digest,
    pub output_bytes: u64,
    pub product: runmat_execution::host::NativeSelectedProductBinding,
    pub selected_build: NativeLinkBuildIdentity,
    pub inputs: NativeProgramLinkIdentity,
}

impl PersistedSelectedImage {
    pub fn validate(&self) -> AotResult<()> {
        if self.schema_version != PERSISTED_SELECTED_IMAGE_SCHEMA_VERSION
            || self.output_bytes == 0
            || self.canonical_plan.identity()? != self.canonical_plan_identity
            || self.canonical_plan.product != self.product
            || self.canonical_plan.build != self.selected_build
            || self.canonical_plan.inputs != self.inputs
            || self
                .product
                .fingerprint()
                .map_err(|error| AotError::contract("aot.link.persisted", error.to_string()))?
                != self.selected_build.product
        {
            return Err(invalid("persisted selected image fields disagree"));
        }
        Ok(())
    }

    pub fn to_canonical_bytes(&self) -> AotResult<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| AotError::contract("aot.link.persisted", error.to_string()))?;
        if bytes.len() > MAX_PERSISTED_SELECTED_IMAGE_BYTES {
            return Err(invalid("persisted selected image exceeds its wire bound"));
        }
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AotResult<Self> {
        if bytes.len() > MAX_PERSISTED_SELECTED_IMAGE_BYTES {
            return Err(invalid("persisted selected image exceeds its wire bound"));
        }
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|error| AotError::contract("aot.link.persisted", error.to_string()))?;
        if value.to_canonical_bytes()?.as_slice() != bytes {
            return Err(invalid(
                "persisted selected image is not canonically encoded",
            ));
        }
        Ok(value)
    }
}

pub(crate) struct CompletedImageCommitment {
    digest: Digest,
    max_bytes: u64,
}

pub(crate) fn admit_completed_image(
    admission: &ExternalNativeInstallationAdmission,
    selected: &SelectedNativeProduct<'_>,
    link_admission: &ExternalNativeLinkAdmission,
    object: &RelocatableNativeObject,
    runtime: &crate::archive::RuntimeArchive,
    evidence: &PersistedSelectedImage,
) -> AotResult<CompletedImageCommitment> {
    evidence.validate()?;
    let expected_inputs = NativeProgramLinkIdentity::from_products(object, runtime);
    let expected_product = selected
        .product()
        .selected()
        .ok_or_else(|| invalid("selected product required"))?
        .fingerprint()
        .map_err(|error| AotError::contract("aot.link.persisted", error.to_string()))?;
    let expected_build = link_admission.identity();
    if admission.product != expected_product
        || admission.image != evidence.output_digest
        || admission.inputs != expected_inputs
        || admission.build != expected_build
        || admission
            .deployment_qualification
            .bytes()
            .iter()
            .all(|byte| *byte == 0)
    {
        return Err(invalid(
            "installation admission differs from completed selected image evidence",
        ));
    }
    admission.storage.file(evidence.output_bytes)?;
    Ok(CompletedImageCommitment {
        digest: admission.image,
        max_bytes: admission.storage.archive.max_file_bytes,
    })
}

impl LinkedProgram {
    pub fn persisted_selected_image(&self) -> AotResult<PersistedSelectedImage> {
        let image = self
            .verified_selected_image()?
            .ok_or_else(|| invalid("selected image required"))?;
        let (output, output_bytes) =
            crate::tree::physical::open_bounded_regular(&self.output, u64::MAX)?;
        if crate::tree::physical::hash_open_file(&output, output_bytes, &self.output)?
            != image.output_digest
        {
            return Err(invalid("selected image changed after completed link"));
        }
        let persisted = PersistedSelectedImage {
            schema_version: PERSISTED_SELECTED_IMAGE_SCHEMA_VERSION,
            canonical_plan: image.canonical_plan.record().clone(),
            canonical_plan_identity: image.canonical_plan.identity(),
            output_digest: image.output_digest,
            output_bytes,
            product: image.product.clone(),
            selected_build: image.selected_build.clone(),
            inputs: image.inputs.clone(),
        };
        persisted.validate()?;
        Ok(persisted)
    }

    pub(crate) fn reopen_selected(
        output: PathBuf,
        persisted: &PersistedSelectedImage,
        selected: &SelectedNativeProduct<'_>,
        admission: &ExternalNativeLinkAdmission,
        expected_inputs: &NativeProgramLinkIdentity,
        binding_mode: runmat_native_codegen::aot::AotRuntimeBindingMode,
        commitment: CompletedImageCommitment,
    ) -> AotResult<Self> {
        persisted.validate()?;
        let product = selected
            .product()
            .selected()
            .ok_or_else(|| invalid("selected product required"))?;
        if &persisted.product != product
            || persisted.selected_build != admission.identity()
            || &persisted.inputs != expected_inputs
            || persisted.output_digest != commitment.digest
            || persisted.output_bytes > commitment.max_bytes
        {
            return Err(invalid(
                "persisted image differs from verified reopening authority",
            ));
        }
        let expected_plan = super::canonical::selected_link_plan(
            selected,
            admission,
            expected_inputs.clone(),
            binding_mode,
        )?;
        if persisted.canonical_plan != expected_plan {
            return Err(invalid(
                "persisted canonical plan differs from verified reopening inputs",
            ));
        }
        let (file, actual_bytes) =
            crate::tree::physical::open_bounded_regular(&output, commitment.max_bytes)?;
        if actual_bytes != persisted.output_bytes
            || actual_bytes > commitment.max_bytes
            || crate::tree::physical::hash_open_file(&file, actual_bytes, &output)?
                != commitment.digest
        {
            return Err(invalid(
                "persisted image bytes differ from trusted completed-image commitment",
            ));
        }
        let canonical = super::CanonicalSelectedLinkReceipt::new(persisted.canonical_plan.clone())?;
        let selected_image = VerifiedSelectedImage {
            canonical_plan: canonical.clone(),
            output_digest: commitment.digest,
            product: persisted.product.clone(),
            selected_build: persisted.selected_build.clone(),
            inputs: persisted.inputs.clone(),
        };
        let linked = Self {
            output,
            retained_temps: None,
            product: runmat_execution::host::NativeProductMode::Selected {
                binding: Box::new(persisted.product.clone()),
            },
            output_digest: commitment.digest,
            selected_build: Some(persisted.selected_build.clone()),
            inputs: persisted.inputs.clone(),
            canonical_selected_plan: Some(canonical),
            selected_image: Some(selected_image),
        };
        linked.verified_selected_image()?;
        Ok(linked)
    }
}

fn invalid(message: impl Into<String>) -> AotError {
    AotError::contract("aot.link.persisted", message.into())
}
