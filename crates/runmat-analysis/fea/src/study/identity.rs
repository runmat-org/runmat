use std::collections::BTreeMap;

use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-structural-study-source\0";
const DOMAIN: &str = "structural-study-source";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 96);

/// One exact material binding in the immutable source identity of a structural study.
///
/// A sequence is used at the serialization boundary because material references are structured
/// keys rather than JSON/CBOR text-map keys. Construction from a `BTreeMap` provides canonical
/// ordering without giving serialization details authority over the analysis model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralStudyMaterialBinding {
    pub reference: AuthoredMaterialRef,
    pub material: MaterialModel,
}

#[derive(Debug, thiserror::Error)]
pub enum StructuralStudySourceIdentityError {
    #[error("structural study source contract is invalid: {0}")]
    Invalid(String),
    #[error("structural study source contract cannot be encoded: {0}")]
    Encoding(#[from] runmat_canonical_codec::CanonicalCodecError),
}

/// Computes the placement-neutral identity of a realized study and its exact material catalog.
pub fn structural_study_source_digest(
    study: &RealizedStudy,
    materials: &BTreeMap<AuthoredMaterialRef, MaterialModel>,
) -> Result<Digest, StructuralStudySourceIdentityError> {
    if study.module_path.trim().is_empty() || study.study_id.trim().is_empty() {
        return Err(StructuralStudySourceIdentityError::Invalid(
            "study identity is empty".into(),
        ));
    }
    runmat_analysis_core::validate_material_definitions(
        &materials.values().cloned().collect::<Vec<_>>(),
    )
    .map_err(|error| StructuralStudySourceIdentityError::Invalid(error.to_string()))?;
    let materials = materials
        .iter()
        .map(|(reference, material)| StructuralStudyMaterialBinding {
            reference: reference.clone(),
            material: material.clone(),
        })
        .collect::<Vec<_>>();
    Ok(Digest::sha256(&encode_contract(
        PREFIX,
        DOMAIN,
        &(study, materials),
        LIMITS,
    )?))
}
