use std::collections::BTreeSet;

use super::{
    GeometryDocument, GeometryModel, GeometryObjectRef, GeometrySourceFormat,
    DISPLAY_TESSELLATION_MEDIA_TYPE, EXACT_BREP_MEDIA_TYPE, FACETED_SOLID_MEDIA_TYPE,
    GEOMETRY_DOCUMENT_SCHEMA_VERSION, GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
};
use crate::{GeometryContractError, UnitSystem};

const MAX_ARTIFACT_BYTES: u64 = 1 << 40;
const MAX_DISPLAY_TESSELLATIONS: usize = 16;

pub(super) fn validate_document(document: &GeometryDocument) -> Result<(), GeometryContractError> {
    if document.schema_version != GEOMETRY_DOCUMENT_SCHEMA_VERSION {
        return Err(invalid("geometry document schema", "unsupported version"));
    }
    document
        .source
        .content_digest
        .validate_nonzero("geometry source digest")?;
    validate_token(
        "geometry importer version",
        &document.source.importer_version,
        128,
    )?;
    if document.source.source_units == UnitSystem::Unspecified
        || !document.source.meters_per_source_unit.is_finite()
        || document.source.meters_per_source_unit <= 0.0
    {
        return Err(invalid(
            "geometry source units",
            "units and positive finite normalization must be explicit",
        ));
    }
    if document.revision.revision == 0 || document.revision.persistent_mapping_version == 0 {
        return Err(invalid(
            "geometry revision",
            "revision and persistent mapping version must be non-zero",
        ));
    }
    if let Some(parent) = document.revision.parent_document_digest {
        parent.validate_nonzero("parent geometry document digest")?;
    }
    document.tolerance.validate()?;
    validate_token(
        "geometry healing algorithm version",
        &document.healing.algorithm_version,
        128,
    )?;

    match &document.model {
        GeometryModel::ExactBRep { model } => {
            require_source_class(document.source.format, true)?;
            let kernel_version = document.source.kernel_version.as_deref().ok_or_else(|| {
                invalid(
                    "geometry kernel version",
                    "exact sources require an importer kernel version",
                )
            })?;
            validate_token("geometry kernel version", kernel_version, 128)?;
            validate_token("exact geometry kernel ABI", &model.kernel_abi, 128)?;
            validate_object(&model.artifact, EXACT_BREP_MEDIA_TYPE)?;
            if !model.capabilities.complete_for_meshing() {
                return Err(invalid(
                    "exact geometry capabilities",
                    "every exact curve, surface, classifier, and mass-property capability is required",
                ));
            }
            if model.assembly_count == 0
                || model.body_count == 0
                || model.shell_count == 0
                || model.face_count == 0
                || model.wire_count == 0
                || model.coedge_count == 0
                || model.edge_count == 0
                || model.vertex_count == 0
            {
                return Err(invalid(
                    "exact geometry topology counts",
                    "an exact model must expose non-empty assembly, body, shell, and boundary topology",
                ));
            }
        }
        GeometryModel::FacetedSolid { model } => {
            require_source_class(document.source.format, false)?;
            if document.source.kernel_version.is_some() {
                return Err(invalid(
                    "geometry kernel version",
                    "faceted sources cannot claim an exact CAD kernel",
                ));
            }
            validate_object(&model.artifact, FACETED_SOLID_MEDIA_TYPE)?;
            if model.vertex_count < 4
                || model.triangle_count < 4
                || model.shell_count == 0
                || !model.is_watertight
                || !model.is_oriented
            {
                return Err(invalid(
                    "faceted solid topology",
                    "a faceted solid must be non-empty, oriented, and watertight",
                ));
            }
        }
    }

    if document.display_tessellations.len() > MAX_DISPLAY_TESSELLATIONS {
        return Err(invalid(
            "display tessellations",
            "too many derived display cache entries",
        ));
    }
    let mut profiles = BTreeSet::new();
    for display in &document.display_tessellations {
        validate_token("display tessellation profile", &display.profile_id, 128)?;
        if !profiles.insert(display.profile_id.as_str())
            || display.geometry_revision != document.revision.revision
            || display.derived_from_primary_digest != document.primary_artifact().digest
        {
            return Err(invalid(
                "display tessellation identity",
                "profiles must be unique and bind the exact document revision and primary payload",
            ));
        }
        validate_object(&display.artifact, DISPLAY_TESSELLATION_MEDIA_TYPE)?;
        if display.artifact.digest == document.primary_artifact().digest {
            return Err(invalid(
                "display tessellation artifact",
                "display and authoritative geometry must be distinct objects",
            ));
        }
    }
    Ok(())
}

fn require_source_class(
    format: GeometrySourceFormat,
    exact: bool,
) -> Result<(), GeometryContractError> {
    if format.is_exact() != exact {
        return Err(invalid(
            "geometry source format",
            "source format and authoritative payload class disagree",
        ));
    }
    Ok(())
}

fn validate_object(
    object: &GeometryObjectRef,
    required_media_type: &str,
) -> Result<(), GeometryContractError> {
    object.digest.validate_nonzero("geometry artifact digest")?;
    if object.encoded_length == 0
        || object.encoded_length > MAX_ARTIFACT_BYTES
        || object.media_type != required_media_type
        || object.schema_version != GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION
    {
        return Err(invalid(
            "geometry artifact reference",
            "artifact length, media type, or schema is invalid for its role",
        ));
    }
    Ok(())
}

fn validate_token(
    field: &str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), GeometryContractError> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.trim() != value
    {
        return Err(invalid(
            field,
            format!("must be 1..={maximum_bytes} printable ASCII bytes without surrounding space"),
        ));
    }
    Ok(())
}

fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
