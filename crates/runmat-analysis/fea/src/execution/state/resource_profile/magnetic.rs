//! Borrow the retained result and physical arrays to project the exact state wire.
use super::*;
use crate::execution::magnetic_static::fields::{
    native_magnetic_entity_counts, native_magnetic_field_definition, native_magnetic_fields,
    NativeMagneticValues,
};
use crate::execution::{FieldRecoveryRequest, FieldRecoverySource, MagnetostaticSolverResult};
use crate::results::field::{
    RecoveredFieldDefinition, RecoveredFieldEntityCounts, RecoveredFieldSample,
};

#[derive(Serialize)]
struct MagneticProjection<'a> {
    result: &'a MagnetostaticSolverResult,
    recovery_request: &'a FieldRecoveryRequest,
    fields: ArtifactProjection<'a>,
}
#[derive(Serialize)]
struct ArtifactProjection<'a> {
    schema_version: u16,
    identity: Digest,
    recovery_request: crate::execution::FieldRecoveryRequestIdentity,
    source: FieldRecoverySource,
    topology_digest: Digest,
    entity_counts: RecoveredFieldEntityCounts,
    fields: Vec<FieldProjection<'a>>,
}
struct FieldProjection<'a> {
    definition: RecoveredFieldDefinition,
    values: NativeMagneticValues<'a>,
    shape: Vec<usize>,
}
impl Serialize for FieldProjection<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("identity", &Digest::sha256([]))?;
        map.serialize_entry("definition", &self.definition)?;
        map.serialize_entry("sample", &RecoveredFieldSample::Final)?;
        map.serialize_entry(
            "storage",
            &AnalysisField::host_f64_serialization(
                self.definition.field_id(),
                &self.definition.layout(),
                &self.shape,
                &self.values,
            ),
        )?;
        map.end()
    }
}
pub(super) fn profile(
    request: &SemanticStatePublicationRequest,
    result: &MagnetostaticSolverResult,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    result.validate_publication_request(request)?;
    let recovery_request = result.native_field_request_from_result()?;
    let fields = native_magnetic_fields(result)
        .into_iter()
        .zip(&recovery_request.plan.requirements)
        .map(|(field, requirement)| {
            Ok(FieldProjection {
                definition: native_magnetic_field_definition(field.field_id, requirement)?,
                values: field.values,
                shape: field.shape,
            })
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let payload = MagneticProjection {
        result,
        recovery_request: &recovery_request,
        fields: ArtifactProjection {
            schema_version: 1,
            identity: Digest::sha256([]),
            recovery_request: recovery_request.identity(),
            source: recovery_request.source,
            topology_digest: request.topology_digest,
            entity_counts: native_magnetic_entity_counts(result),
            fields,
        },
    };
    profile_state_payload(
        request,
        &Tagged {
            kind: "magnetostatic",
            key: "state",
            value: &payload,
        },
    )
}
