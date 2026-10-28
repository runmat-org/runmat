use std::collections::HashMap;

use runmat_meshing_core::{CanonicalMeshingContract, MeshingEvidence, SolverMeshArtifact};
use runmat_types::MethodName;
use runmat_value::{IntegerStorage, ObjectInstance, Tensor, UnboundedObjectBuilder, Value};

use super::{
    error, BOUNDARY_FACES_NAME, ELEMENTS_NAME, EVIDENCE_NAME, GENERATE_NAME, MESH_METHOD_NAME,
    NODES_NAME, QUALITY_NAME, REGIONS_NAME, SAVE_NAME,
};
use crate::class_registry::declaration::{instance_methods, register_value_class};
use crate::class_registry::RuntimeMethod;
use crate::BuiltinResult;

/// Derived from the catalog class identity.
pub const MESH_CLASS: &str = runmat_builtins::MESHING_MESH_OBJECT_CLASS.display_name();
/// Derived from the catalog class identity.
pub const EVIDENCE_CLASS: &str = runmat_builtins::MESHING_EVIDENCE_CLASS.display_name();
const ARTIFACT_PROPERTY: &str = "__runmat_meshing_artifact_cbor";
const EVIDENCE_PROPERTY: &str = "__runmat_meshing_evidence_cbor";

static MESHING_CLASSES_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(runmat_builtins::MESHING_MESH_OBJECT_CLASS);

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MeshPayload {
    pub artifact: SolverMeshArtifact,
    pub evidence: MeshingEvidence,
}

pub(super) fn mesh_object(payload: MeshPayload) -> BuiltinResult<Value> {
    ensure_classes_registered();
    payload
        .evidence
        .validate(&payload.artifact)
        .map_err(|cause| error(GENERATE_NAME, cause.to_string()))?;
    let artifact_bytes = payload
        .artifact
        .canonical_encode()
        .map_err(|cause| error(GENERATE_NAME, cause.to_string()))?;
    let evidence_bytes = payload
        .evidence
        .canonical_encode()
        .map_err(|cause| error(GENERATE_NAME, cause.to_string()))?;
    let mut object = UnboundedObjectBuilder::new(MESH_CLASS.to_string());
    object.insert(
        "canonical_digest".to_string(),
        Value::String(digest_hex(payload.artifact.canonical_digest.bytes()).into()),
    );
    object.insert(
        "node_count".to_string(),
        Value::Int(runmat_value::IntValue::U64(
            payload.artifact.topology.nodes.len() as u64,
        )),
    );
    object.insert(
        "element_count".to_string(),
        Value::Int(runmat_value::IntValue::U64(
            payload.artifact.topology.volume_elements.len() as u64,
        )),
    );
    object.insert(
        "boundary_face_count".to_string(),
        Value::Int(runmat_value::IntValue::U64(
            payload.artifact.topology.boundary_faces.len() as u64,
        )),
    );
    object.insert(
        ARTIFACT_PROPERTY.to_string(),
        byte_tensor(artifact_bytes, GENERATE_NAME)?,
    );
    object.insert(
        EVIDENCE_PROPERTY.to_string(),
        byte_tensor(evidence_bytes, GENERATE_NAME)?,
    );
    Ok(Value::Object(object.finish()))
}

pub(super) fn evidence_object(evidence: MeshingEvidence) -> BuiltinResult<Value> {
    ensure_classes_registered();
    let evidence_bytes = evidence
        .canonical_encode()
        .map_err(|cause| error(EVIDENCE_NAME, cause.to_string()))?;
    let mut object = UnboundedObjectBuilder::new(EVIDENCE_CLASS.to_string());
    object.insert(
        "artifact_digest".to_string(),
        Value::String(digest_hex(evidence.artifact_digest.bytes()).into()),
    );
    object.insert(
        "stage_count".to_string(),
        Value::Int(runmat_value::IntValue::U64(evidence.stages.len() as u64)),
    );
    object.insert(
        "generated_nodes".to_string(),
        Value::Int(runmat_value::IntValue::U64(
            evidence.resources.generated_nodes,
        )),
    );
    object.insert(
        "generated_elements".to_string(),
        Value::Int(runmat_value::IntValue::U64(
            evidence.resources.generated_elements,
        )),
    );
    object.insert(
        "wall_time_ms".to_string(),
        Value::Int(runmat_value::IntValue::U64(evidence.resources.wall_time_ms)),
    );
    object.insert(
        "deterministic_seed".to_string(),
        exact(evidence.deterministic_seed),
    );
    object.insert(
        "geometry_revision".to_string(),
        exact(u64::from(evidence.geometry.geometry_revision)),
    );
    object.insert("resources".to_string(), resources_value(&evidence));
    object.insert("stages".to_string(), stages_value(&evidence)?);
    object.insert(
        EVIDENCE_PROPERTY.to_string(),
        byte_tensor(evidence_bytes, EVIDENCE_NAME)?,
    );
    Ok(Value::Object(object.finish()))
}

fn resources_value(evidence: &MeshingEvidence) -> Value {
    let resources = &evidence.resources;
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    for (name, count) in [
        ("generated_nodes", resources.generated_nodes),
        ("generated_elements", resources.generated_elements),
        ("peak_memory_bytes", resources.peak_memory_bytes),
        ("peak_scratch_bytes", resources.peak_scratch_bytes),
        ("wall_time_ms", resources.wall_time_ms),
        ("artifact_bytes", resources.artifact_bytes),
        ("search_work", resources.search_work),
        (
            "maximum_recursion_depth",
            u64::from(resources.maximum_recursion_depth),
        ),
        ("iterations", resources.iterations),
    ] {
        value.insert(name, exact(count));
    }
    Value::Struct(value.finish())
}

fn stages_value(evidence: &MeshingEvidence) -> BuiltinResult<Value> {
    let stages = evidence
        .stages
        .iter()
        .map(|stage| {
            let mut value = runmat_value::UnboundedStructValueBuilder::new();
            let encoded_stage = serde_json::to_value(stage.stage)
                .map_err(|cause| error(EVIDENCE_NAME, cause.to_string()))?;
            let name = encoded_stage.as_str().ok_or_else(|| {
                error(
                    EVIDENCE_NAME,
                    "meshing evidence stage did not serialize as a canonical name",
                )
            })?;
            value.insert("stage", Value::String(name.to_string().into()));
            for (name, count) in [
                ("completed_work", stage.completed_work),
                ("estimated_work", stage.estimated_work),
                ("peak_memory_bytes", stage.peak_memory_bytes),
                ("peak_scratch_bytes", stage.peak_scratch_bytes),
                ("search_work", stage.search_work),
                ("iterations", stage.iterations),
                ("elapsed_time_ms", stage.elapsed_time_ms),
                ("cancellation_checkpoints", stage.cancellation_checkpoints),
            ] {
                value.insert(name, exact(count));
            }
            let invariants = stage
                .invariants
                .iter()
                .map(|invariant| {
                    let mut value = runmat_value::UnboundedStructValueBuilder::new();
                    value.insert("id", Value::String(invariant.invariant_id.clone().into()));
                    value.insert("passed", Value::Bool(invariant.passed));
                    if let Some(measured) = invariant.measured {
                        value.insert("measured", Value::Num(measured));
                    }
                    if let Some(required) = invariant.required {
                        value.insert("required", Value::Num(required));
                    }
                    if let Some(unit) = &invariant.unit {
                        value.insert("unit", Value::String(unit.clone().into()));
                    }
                    Value::Struct(value.finish())
                })
                .collect::<Vec<_>>();
            value.insert(
                "invariants",
                crate::make_cell_with_shape(invariants, vec![1, stage.invariants.len()])
                    .map_err(|message| error(EVIDENCE_NAME, message))?,
            );
            Ok(Value::Struct(value.finish()))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    crate::make_cell_with_shape(stages, vec![1, evidence.stages.len()])
        .map_err(|message| error(EVIDENCE_NAME, message))
}

fn exact(value: u64) -> Value {
    Value::Int(runmat_value::IntValue::U64(value))
}

pub(super) fn payload_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<MeshPayload> {
    let Value::Object(object) = value else {
        return Err(error(builtin, "expected a meshing.Mesh"));
    };
    if !object
        .class_name()
        .is(runmat_builtins::MESHING_MESH_OBJECT_CLASS)
    {
        return Err(error(builtin, format!("expected {MESH_CLASS}")));
    }
    let artifact =
        SolverMeshArtifact::canonical_decode(&bytes_property(object, ARTIFACT_PROPERTY, builtin)?)
            .map_err(|cause| error(builtin, cause.to_string()))?;
    let evidence =
        MeshingEvidence::canonical_decode(&bytes_property(object, EVIDENCE_PROPERTY, builtin)?)
            .map_err(|cause| error(builtin, cause.to_string()))?;
    evidence
        .validate(&artifact)
        .map_err(|cause| error(builtin, cause.to_string()))?;
    Ok(MeshPayload { artifact, evidence })
}

fn byte_tensor(bytes: Vec<u8>, builtin: &'static str) -> BuiltinResult<Value> {
    let count = bytes.len();
    Tensor::new_integer(IntegerStorage::U8(bytes), vec![count, 1])
        .map(Value::Tensor)
        .map_err(|cause| error(builtin, cause))
}

fn bytes_property(
    object: &ObjectInstance,
    property: &str,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    let Some(Value::Tensor(tensor)) = object.properties().get(property) else {
        return Err(error(
            builtin,
            "meshing.Mesh is missing its canonical payload",
        ));
    };
    match tensor.integer_storage() {
        Some(IntegerStorage::U8(bytes)) => Ok(bytes.clone()),
        _ => Err(error(
            builtin,
            "meshing.Mesh canonical payload is not uint8",
        )),
    }
}

impl From<crate::meshing::MeshingExecutionOutput> for MeshPayload {
    fn from(output: crate::meshing::MeshingExecutionOutput) -> Self {
        Self {
            artifact: output.artifact,
            evidence: output.evidence,
        }
    }
}

pub(crate) fn execution_output_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<crate::meshing::MeshingExecutionOutput> {
    let payload = payload_from_value(builtin, value)?;
    Ok(crate::meshing::MeshingExecutionOutput {
        artifact: payload.artifact,
        evidence: payload.evidence,
    })
}

fn ensure_classes_registered() {
    MESHING_CLASSES_REGISTERED.ensure(|| {
        register_value_class(MESH_CLASS, HashMap::new(), mesh_methods());
        register_value_class(EVIDENCE_CLASS, HashMap::new(), HashMap::new());
    });
}

fn mesh_methods() -> HashMap<MethodName, RuntimeMethod> {
    instance_methods([
        ("mesh", MESH_METHOD_NAME),
        ("save", SAVE_NAME),
        ("nodes", NODES_NAME),
        ("elements", ELEMENTS_NAME),
        ("boundaryFaces", BOUNDARY_FACES_NAME),
        ("regions", REGIONS_NAME),
        ("quality", QUALITY_NAME),
        ("evidence", EVIDENCE_NAME),
    ])
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{fixtures, ElementOrder};

    use super::*;

    fn payload() -> MeshPayload {
        let artifact = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let evidence = fixtures::canonical_tetrahedron_meshing_evidence(&artifact);
        MeshPayload { artifact, evidence }
    }

    #[test]
    fn mesh_object_round_trips_canonical_contracts() {
        let expected = payload();
        let value = mesh_object(expected.clone()).expect("mesh object");
        let actual = payload_from_value("test", &value).expect("canonical payload");

        assert_eq!(actual, expected);
    }

    #[test]
    fn mesh_object_rejects_corrupted_canonical_bytes() {
        let mut value = mesh_object(payload()).expect("mesh object");
        let Value::Object(object) = std::mem::replace(&mut value, Value::Num(0.0)) else {
            panic!("expected meshing.Mesh object");
        };
        let mut object = object
            .into_unbounded_builder()
            .expect("ordinary mesh object can enter an explicit edit");
        object.insert(
            ARTIFACT_PROPERTY.to_string(),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![0xff]), vec![1, 1])
                    .expect("invalid canonical bytes tensor"),
            ),
        );
        value = Value::Object(object.finish());

        let error = payload_from_value("test", &value).expect_err("corrupt payload must fail");
        assert!(error.to_string().contains("test"));
    }
}
