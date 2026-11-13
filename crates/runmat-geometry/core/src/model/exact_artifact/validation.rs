use super::{
    AdmittedExactGeometry, ExactGeometryManifest, EXACT_EVALUATOR_MEDIA_TYPE,
    EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION, EXACT_TOPOLOGY_MEDIA_TYPE, GEOMETRY_HEALING_MEDIA_TYPE,
    KERNEL_REPRESENTATION_MEDIA_TYPE,
};
use crate::{
    model::analysis_identity::validate_token, GeometryContractError, GeometryDocument,
    GeometryModel, GeometryObjectRef, GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
};

const MAX_COMPONENT_BYTES: u64 = 512 * 1024 * 1024;

impl ExactGeometryManifest {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        if self.schema_version != EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION {
            return Err(invalid(
                "exact geometry manifest schema",
                "unsupported version",
            ));
        }
        self.source_digest
            .validate_nonzero("exact geometry source digest")?;
        if self.revision.revision == 0 || self.revision.persistent_mapping_version == 0 {
            return Err(invalid(
                "exact geometry revision",
                "revision and persistent mapping version must be non-zero",
            ));
        }
        validate_token("exact geometry kernel ABI", &self.kernel_abi, 128)?;
        validate_component(&self.topology, EXACT_TOPOLOGY_MEDIA_TYPE)?;
        validate_component(&self.evaluators, EXACT_EVALUATOR_MEDIA_TYPE)?;
        if let Some(representation) = &self.kernel_representation {
            validate_component(representation, KERNEL_REPRESENTATION_MEDIA_TYPE)?;
        }
        if let Some(healing) = &self.healing_report {
            validate_component(healing, GEOMETRY_HEALING_MEDIA_TYPE)?;
        }
        let mut digests = [None; 4];
        for (index, component) in [&self.topology, &self.evaluators]
            .into_iter()
            .chain(self.kernel_representation.iter())
            .chain(self.healing_report.iter())
            .enumerate()
        {
            if digests[..index].contains(&Some(component.digest)) {
                return Err(invalid(
                    "exact geometry component identity",
                    "component objects must have distinct content identities",
                ));
            }
            digests[index] = Some(component.digest);
        }
        Ok(())
    }

    pub fn validate_against_document(
        &self,
        document: &GeometryDocument,
    ) -> Result<(), GeometryContractError> {
        self.validate()?;
        document.validate()?;
        let GeometryModel::ExactBRep { model } = &document.model else {
            return Err(invalid(
                "exact geometry document",
                "an exact manifest cannot bind a faceted document",
            ));
        };
        if self.source_digest != document.source.content_digest
            || self.revision != document.revision
            || self.kernel_abi != model.kernel_abi
        {
            return Err(invalid(
                "exact geometry manifest identity",
                "source, revision, mapping version, and kernel ABI must match the document",
            ));
        }
        Ok(())
    }
}

/// Independently hashes, bounded-decodes, and semantically validates an exact closure. Callers
/// supply bytes obtained from any storage or transport; physical location is never trusted.
pub fn admit_exact_geometry_closure(
    document: &GeometryDocument,
    manifest_bytes: &[u8],
    topology_bytes: &[u8],
    evaluator_bytes: &[u8],
    kernel_representation_bytes: Option<&[u8]>,
    healing_bytes: Option<&[u8]>,
) -> Result<AdmittedExactGeometry, GeometryContractError> {
    document.validate()?;
    let GeometryModel::ExactBRep { model } = &document.model else {
        return Err(invalid(
            "exact geometry closure",
            "faceted geometry has no exact closure",
        ));
    };
    verify_object(document.primary_artifact(), manifest_bytes)?;
    let manifest = ExactGeometryManifest::canonical_decode(manifest_bytes)?;
    manifest.validate_against_document(document)?;
    verify_object(&manifest.topology, topology_bytes)?;
    verify_object(&manifest.evaluators, evaluator_bytes)?;
    match (&manifest.healing_report, healing_bytes) {
        (Some(reference), Some(bytes)) => verify_object(reference, bytes)?,
        (None, None) => {}
        _ => {
            return Err(invalid(
                "exact geometry healing component",
                "manifest reference and supplied healing bytes must agree",
            ))
        }
    }
    let super::DecodedExactGeometryComponents {
        topology,
        evaluators,
        healing_report,
    } = super::decode_exact_geometry_components(
        super::ExactGeometryComponentBytes {
            topology: topology_bytes,
            evaluators: evaluator_bytes,
            healing: healing_bytes,
        },
        model,
    )?;
    super::bindings::validate_exact_geometry_component_bindings(
        document,
        &manifest,
        &topology,
        &evaluators,
        kernel_representation_bytes,
        healing_report.as_ref(),
    )?;
    let kernel_representation = kernel_representation_bytes.map(<[u8]>::to_vec);
    Ok(AdmittedExactGeometry {
        manifest,
        topology,
        evaluators,
        kernel_representation,
        healing_report,
    })
}

pub(super) fn topology_contains(
    topology: &crate::ExactBRepTopology,
    target: &crate::PersistentEntityId,
) -> bool {
    use crate::PersistentEntityKind;

    match target.kind {
        PersistentEntityKind::Assembly => topology
            .assemblies
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Instance => topology
            .instances
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Body => topology
            .bodies
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Lump => topology
            .lumps
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Solid => topology
            .solids
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Shell => topology
            .shells
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Face => topology
            .faces
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Wire => topology
            .wires
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Coedge => topology
            .coedges
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Edge => topology
            .edges
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Vertex => topology
            .vertices
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Contact => topology
            .contacts
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
        PersistentEntityKind::Region => topology
            .regions
            .binary_search_by(|value| value.id.cmp(target))
            .is_ok(),
    }
}

fn validate_component(
    object: &GeometryObjectRef,
    media_type: &str,
) -> Result<(), GeometryContractError> {
    object
        .digest
        .validate_nonzero("geometry component digest")?;
    if object.encoded_length == 0
        || object.encoded_length > MAX_COMPONENT_BYTES
        || object.media_type != media_type
        || object.schema_version != GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION
    {
        return Err(invalid(
            "geometry component reference",
            "component length, media type, or schema is invalid for its role",
        ));
    }
    Ok(())
}

pub(super) fn verify_object(
    reference: &GeometryObjectRef,
    bytes: &[u8],
) -> Result<(), GeometryContractError> {
    if reference.encoded_length != bytes.len() as u64
        || reference.digest != crate::model::canonical::digest(bytes)?
    {
        return Err(invalid(
            "geometry component bytes",
            "encoded length or SHA-256 content identity does not match its reference",
        ));
    }
    Ok(())
}

pub(super) fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
