use std::collections::BTreeSet;

use super::{
    AdmittedExactGeometry, ExactGeometryManifest, EXACT_EVALUATOR_MEDIA_TYPE,
    EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION, EXACT_TOPOLOGY_MEDIA_TYPE, GEOMETRY_HEALING_MEDIA_TYPE,
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
        if let Some(healing) = &self.healing_report {
            validate_component(healing, GEOMETRY_HEALING_MEDIA_TYPE)?;
        }
        let mut digests = BTreeSet::new();
        for component in [&self.topology, &self.evaluators]
            .into_iter()
            .chain(self.healing_report.iter())
        {
            if !digests.insert(component.digest) {
                return Err(invalid(
                    "exact geometry component identity",
                    "component objects must have distinct content identities",
                ));
            }
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
    let topology = super::decode_exact_topology(topology_bytes, model)?;
    let evaluators = super::decode_exact_evaluators(evaluator_bytes, &topology, model)?;
    let healing_report = match (&manifest.healing_report, healing_bytes) {
        (Some(reference), Some(bytes)) => {
            verify_object(reference, bytes)?;
            let report = super::decode_geometry_healing_report(bytes)?;
            if report.healed_topology_digest != manifest.topology.digest
                || report.revision_map.target_revision != manifest.revision
            {
                return Err(invalid(
                    "exact geometry healing binding",
                    "healing must produce this topology and target revision",
                ));
            }
            for target in report
                .revision_map
                .operations
                .iter()
                .flat_map(|operation| operation.targets())
            {
                if !topology_contains(&topology, target) {
                    return Err(invalid(
                        "exact geometry healed topology mapping",
                        "every mapped target must exist in the admitted healed topology",
                    ));
                }
            }
            Some(report)
        }
        (None, None) => None,
        _ => {
            return Err(invalid(
                "exact geometry healing component",
                "manifest reference and supplied healing bytes must agree",
            ));
        }
    };
    Ok(AdmittedExactGeometry {
        manifest,
        topology,
        evaluators,
        healing_report,
    })
}

fn topology_contains(
    topology: &crate::ExactBRepTopology,
    target: &crate::PersistentEntityId,
) -> bool {
    use crate::PersistentEntityKind;

    match target.kind {
        PersistentEntityKind::Assembly => {
            topology.assemblies.iter().any(|value| &value.id == target)
        }
        PersistentEntityKind::Instance => {
            topology.instances.iter().any(|value| &value.id == target)
        }
        PersistentEntityKind::Body => topology.bodies.iter().any(|value| &value.id == target),
        PersistentEntityKind::Lump => topology.lumps.iter().any(|value| &value.id == target),
        PersistentEntityKind::Solid => topology.solids.iter().any(|value| &value.id == target),
        PersistentEntityKind::Shell => topology.shells.iter().any(|value| &value.id == target),
        PersistentEntityKind::Face => topology.faces.iter().any(|value| &value.id == target),
        PersistentEntityKind::Wire => topology.wires.iter().any(|value| &value.id == target),
        PersistentEntityKind::Coedge => topology.coedges.iter().any(|value| &value.id == target),
        PersistentEntityKind::Edge => topology.edges.iter().any(|value| &value.id == target),
        PersistentEntityKind::Vertex => topology.vertices.iter().any(|value| &value.id == target),
        PersistentEntityKind::Contact => topology.contacts.iter().any(|value| &value.id == target),
        PersistentEntityKind::Region => topology.interfaces.iter().any(|interface| {
            &interface.side_a_region_id == target || &interface.side_b_region_id == target
        }),
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

fn verify_object(reference: &GeometryObjectRef, bytes: &[u8]) -> Result<(), GeometryContractError> {
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

fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
