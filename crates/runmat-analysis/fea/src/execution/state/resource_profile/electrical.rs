//! Borrow retained arrays to project the exact shared state wire; no recovered-field copies.
use super::*;
use crate::execution::electrical_static::{
    fields::{native_electrical_fields, NativeElectricalValues},
    native_electrical_entity_counts,
};
use crate::execution::{ElectricalStaticSolverResult, FieldRecoveryRequest, FieldRecoverySource};
use crate::results::field::{
    RecoveredFieldDefinition, RecoveredFieldEntityCounts, RecoveredFieldSample,
};

#[derive(Serialize)]
struct ElectricalProjection<'a> {
    equation: runmat_analysis_core::StaticElectricalEquation,
    element_order: runmat_meshing_core::ElementOrder,
    ordered_node_ids: &'a [u64],
    ordered_element_ids: &'a [u64],
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
    values: NativeElectricalValues<'a>,
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
    result: &ElectricalStaticSolverResult,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    result.validate_publication_request(request)?;
    let recovery_request = result.native_field_request_from_result()?;
    let fields = native_electrical_fields(result)
        .into_iter()
        .zip(&recovery_request.plan.requirements)
        .map(|(field, requirement)| {
            Ok(FieldProjection {
                definition:
                    crate::execution::electrical_static::native_electrical_field_definition(
                        field.field_id,
                        requirement,
                    )?,
                values: field.values,
                shape: field.shape,
            })
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let payload = ElectricalProjection {
        equation: result.response().equation,
        element_order: result.element_order(),
        ordered_node_ids: &result.response().ordered_node_ids,
        ordered_element_ids: &result.response().ordered_element_ids,
        recovery_request: &recovery_request,
        fields: ArtifactProjection {
            schema_version: 1,
            identity: Digest::sha256([]),
            recovery_request: recovery_request.identity(),
            source: recovery_request.source,
            topology_digest: request.topology_digest,
            entity_counts: native_electrical_entity_counts(result),
            fields,
        },
    };
    profile_state_payload(
        request,
        &Tagged {
            kind: "electrical_static",
            key: "state",
            value: &payload,
        },
    )
}
