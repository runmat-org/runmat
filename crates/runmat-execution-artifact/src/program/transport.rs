use std::collections::BTreeSet;

use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
pub use runmat_execution::value::{EXTERNALIZED_VALUE_MEDIA_TYPE, EXTERNALIZED_VALUE_SCHEMA};
use serde::{Deserialize, Serialize};

use super::request::validate_externalized_inventory;
use super::{
    ProgramExecutionResponse, MAX_PROGRAM_EXECUTION_RESULT_OBJECTS,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use crate::{ArtifactError, ArtifactResult};

pub const MAX_PROGRAM_EXECUTION_OBJECT_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES: u64 =
    MAX_PROGRAM_EXECUTION_OBJECT_BYTES * MAX_PROGRAM_EXECUTION_RESULT_OBJECTS as u64;
pub const PROGRAM_EXECUTION_OBJECT_CLOSURE_SCHEMA_VERSION: u16 = 2;
pub const PROGRAM_EXECUTION_OBJECT_CLOSURE_MEDIA_TYPE: &str =
    "application/vnd.runmat.execution-object-closure+json";
pub const PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA: &str =
    "runmat.execution-object-closure.v2";
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramExecutionObject {
    pub reference: ValueRef,
    pub bytes: Vec<u8>,
}

impl ProgramExecutionObject {
    pub fn externalized_value(
        access: &crate::ArtifactAccess,
        kind: ValueRefKind,
        value: ValuePayload,
    ) -> ArtifactResult<Self> {
        access.validate()?;
        value
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        if !matches!(value, ValuePayload::Inline(_))
            || !collect_references(std::slice::from_ref(&value)).is_empty()
        {
            return Err(ArtifactError::Invalid(
                "externalized values must be closed inline value graphs".into(),
            ));
        }
        let logical_digest = value
            .logical_digest()
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        let bytes = serde_json::to_vec(&value)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        let reference = ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
            id: access.value_id(logical_digest),
            logical_digest,
            encoded_length: bytes.len() as u64,
            media_type: EXTERNALIZED_VALUE_MEDIA_TYPE.into(),
            value_schema: EXTERNALIZED_VALUE_SCHEMA.into(),
            encryption_context: access.encryption_context,
            kind,
            authorization_scope: access.authorization_scope.clone(),
            resident_fence: None,
        };
        let object = Self { reference, bytes };
        object.validate()?;
        Ok(object)
    }

    pub fn is_externalized_value(&self) -> bool {
        self.reference.media_type == EXTERNALIZED_VALUE_MEDIA_TYPE
            && self.reference.value_schema == EXTERNALIZED_VALUE_SCHEMA
    }

    pub fn rebind_externalized_value(
        &self,
        access: &crate::ArtifactAccess,
    ) -> ArtifactResult<Self> {
        let value = self
            .decode_externalized_value()?
            .ok_or_else(|| ArtifactError::Invalid("program object is not a RunMat value".into()))?;
        Self::externalized_value(access, self.reference.kind, value)
    }

    pub fn decode_externalized_value(&self) -> ArtifactResult<Option<ValuePayload>> {
        if !self.is_externalized_value() {
            return Ok(None);
        }
        let value: ValuePayload = serde_json::from_slice(&self.bytes)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        value
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        let canonical_bytes = serde_json::to_vec(&value)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        if !matches!(value, ValuePayload::Inline(_))
            || !collect_references(std::slice::from_ref(&value)).is_empty()
            || canonical_bytes != self.bytes
            || value
                .logical_digest()
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?
                != self.reference.logical_digest
        {
            return Err(ArtifactError::Identity(
                "externalized value bytes differ from their logical identity".into(),
            ));
        }
        Ok(Some(value))
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        ValuePayload::Object(Box::new(self.reference.clone()))
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        let externalized_media = self.reference.media_type == EXTERNALIZED_VALUE_MEDIA_TYPE;
        let externalized_schema = self.reference.value_schema == EXTERNALIZED_VALUE_SCHEMA;
        if !matches!(
            self.reference.kind,
            ValueRefKind::DriverObject | ValueRefKind::ResultObject
        ) || self.reference.resident_fence.is_some()
            || self.reference.encoded_length == 0
            || self.reference.encoded_length > MAX_PROGRAM_EXECUTION_OBJECT_BYTES
            || self.reference.encoded_length != self.bytes.len() as u64
            || externalized_media != externalized_schema
        {
            return Err(ArtifactError::Invalid(
                "program object differs from its portable reference".into(),
            ));
        }
        if externalized_media {
            let access = crate::ArtifactAccess {
                authorization_scope: self.reference.authorization_scope.clone(),
                encryption_context: self.reference.encryption_context,
            };
            if self.reference.id != access.value_id(self.reference.logical_digest) {
                return Err(ArtifactError::Identity(
                    "externalized value reference differs from its access authority".into(),
                ));
            }
            self.decode_externalized_value()?;
        } else if self.reference.logical_digest != runmat_execution::Digest::sha256(&self.bytes) {
            return Err(ArtifactError::Identity(
                "program object bytes differ from their logical identity".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionObjectLocation {
    pub reference: ValueRef,
    pub artifact_id: String,
}

impl ProgramExecutionObjectLocation {
    pub fn validate(&self) -> ArtifactResult<()> {
        validate_reference(&self.reference)?;
        if self.artifact_id.is_empty()
            || self.artifact_id.len() > 128
            || !self
                .artifact_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(ArtifactError::Invalid(
                "program object artifact identity is malformed or unbounded".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionObjectClosure {
    pub schema_version: u16,
    pub root: ValueRef,
    pub objects: Vec<ProgramExecutionObjectLocation>,
}

impl ProgramExecutionObjectClosure {
    pub fn new(
        root: ValueRef,
        mut objects: Vec<ProgramExecutionObjectLocation>,
    ) -> ArtifactResult<Self> {
        objects.sort_by_key(|object| object.reference.id);
        let closure = Self {
            schema_version: PROGRAM_EXECUTION_OBJECT_CLOSURE_SCHEMA_VERSION,
            root,
            objects,
        };
        closure.validate()?;
        Ok(closure)
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_OBJECT_CLOSURE_SCHEMA_VERSION
            || self.root.kind != ValueRefKind::ResultObject
        {
            return Err(ArtifactError::Invalid(
                "program execution object closure uses an unsupported schema or root kind".into(),
            ));
        }
        validate_location_inventory(&self.objects, LocationKinds::PortableClosure, "closure")?;
        if self.objects.is_empty()
            || self
                .objects
                .windows(2)
                .any(|pair| pair[0].reference.id >= pair[1].reference.id)
            || !self
                .objects
                .iter()
                .any(|object| object.reference == self.root)
        {
            return Err(ArtifactError::Invalid(
                "program execution object closure is empty, unordered, or omits its root".into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> ArtifactResult<Vec<u8>> {
        self.validate()?;
        let bytes =
            serde_json::to_vec(self).map_err(|error| ArtifactError::Encoding(error.to_string()))?;
        if bytes.len() as u64 > MAX_PROGRAM_EXECUTION_OBJECT_BYTES {
            return Err(ArtifactError::Limit(
                "program execution object closure exceeds its byte bound".into(),
            ));
        }
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> ArtifactResult<Self> {
        if bytes.is_empty() || bytes.len() as u64 > MAX_PROGRAM_EXECUTION_OBJECT_BYTES {
            return Err(ArtifactError::Limit(
                "program execution object closure is empty or exceeds its byte bound".into(),
            ));
        }
        let closure: Self = serde_json::from_slice(bytes)
            .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
        closure.validate()?;
        if closure.canonical_bytes()? != bytes {
            return Err(ArtifactError::Identity(
                "program execution object closure is not canonically encoded".into(),
            ));
        }
        Ok(closure)
    }
}

/// Every external program argument must be a portable driver object listed
/// exactly once in the input inventory.
pub(super) fn validate_input_objects(
    arguments: &[ValuePayload],
    objects: &[ProgramExecutionObjectLocation],
) -> ArtifactResult<()> {
    validate_locations(objects, ValueRefKind::DriverObject, "input")?;
    for reference in collect_references(arguments) {
        if reference.kind != ValueRefKind::DriverObject {
            return Err(ArtifactError::Invalid(
                "program inputs contain a non-portable external argument".into(),
            ));
        }
        if !objects.iter().any(|object| object.reference == *reference) {
            return Err(ArtifactError::Invalid(
                "program input object inventory omits an external argument".into(),
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionOutputs {
    pub schema_version: u16,
    pub response: ProgramExecutionResponse,
    pub objects: Vec<ProgramExecutionObjectLocation>,
}

impl ProgramExecutionOutputs {
    pub fn new(
        response: ProgramExecutionResponse,
        objects: Vec<ProgramExecutionObjectLocation>,
    ) -> ArtifactResult<Self> {
        let outputs = Self {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            response,
            objects,
        };
        outputs.validate()?;
        Ok(outputs)
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION {
            return Err(ArtifactError::Invalid(
                "program outputs use an unsupported schema".into(),
            ));
        }
        validate_locations(&self.objects, ValueRefKind::ResultObject, "result")?;
        match &self.response {
            ProgramExecutionResponse::Success { outputs } => {
                for value in outputs {
                    value
                        .validate(ValueLimits::default())
                        .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                }
                if !self.objects.is_empty() {
                    return Err(ArtifactError::Invalid(
                        "inline program output carries external objects".into(),
                    ));
                }
            }
            ProgramExecutionResponse::ExternalizedSuccess {
                outputs,
                result_objects,
            } => {
                let requested_outputs = u16::try_from(outputs.len()).map_err(|_| {
                    ArtifactError::Limit("program output count exceeds its bound".into())
                })?;
                if result_objects.len() != self.objects.len()
                    || result_objects
                        .iter()
                        .zip(&self.objects)
                        .any(|(reference, object)| reference != &object.reference)
                {
                    return Err(ArtifactError::Invalid(
                        "program response and result object inventory differ".into(),
                    ));
                }
                validate_externalized_inventory(outputs, result_objects, requested_outputs, None)?;
            }
            ProgramExecutionResponse::Failure { message } => {
                if message.is_empty() || message.len() > 1024 * 1024 || !self.objects.is_empty() {
                    return Err(ArtifactError::Invalid(
                        "failed program output is malformed".into(),
                    ));
                }
            }
            ProgramExecutionResponse::RuntimeFailure { failure } => {
                failure
                    .validate()
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                if !self.objects.is_empty() {
                    return Err(ArtifactError::Invalid(
                        "failed program output carries external objects".into(),
                    ));
                }
            }
            ProgramExecutionResponse::GraphSuccess { .. }
            | ProgramExecutionResponse::SpmdSuccess { .. } => {
                return Err(ArtifactError::Invalid(
                    "object-backed program outputs carry only direct value results".into(),
                ));
            }
        }
        Ok(())
    }
}

fn validate_locations(
    objects: &[ProgramExecutionObjectLocation],
    expected_kind: ValueRefKind,
    inventory_name: &str,
) -> ArtifactResult<()> {
    validate_location_inventory(objects, LocationKinds::Exact(expected_kind), inventory_name)
}

enum LocationKinds {
    Exact(ValueRefKind),
    PortableClosure,
}

impl LocationKinds {
    fn permits(&self, kind: ValueRefKind) -> bool {
        match self {
            Self::Exact(expected) => kind == *expected,
            Self::PortableClosure => {
                matches!(
                    kind,
                    ValueRefKind::DriverObject | ValueRefKind::ResultObject
                )
            }
        }
    }
}

fn validate_location_inventory(
    objects: &[ProgramExecutionObjectLocation],
    kinds: LocationKinds,
    inventory_name: &str,
) -> ArtifactResult<()> {
    if objects.len() > MAX_PROGRAM_EXECUTION_RESULT_OBJECTS {
        return Err(ArtifactError::Invalid(format!(
            "program {inventory_name} object inventory exceeds its count bound"
        )));
    }
    let mut ids = BTreeSet::new();
    let mut digests = BTreeSet::new();
    let mut artifact_ids = BTreeSet::new();
    let mut authorization_scope = None;
    let mut total_bytes = 0_u64;
    for object in objects {
        object.validate()?;
        if !kinds.permits(object.reference.kind)
            || !ids.insert(object.reference.id)
            || !digests.insert(object.reference.logical_digest)
            || !artifact_ids.insert(object.artifact_id.as_str())
        {
            return Err(ArtifactError::Invalid(format!(
                "program {inventory_name} object inventory contains a wrong-kind or duplicate object"
            )));
        }
        match authorization_scope {
            Some(scope) if scope != object.reference.authorization_scope => {
                return Err(ArtifactError::Invalid(format!(
                    "program {inventory_name} objects span authorization scopes"
                )))
            }
            None => authorization_scope = Some(object.reference.authorization_scope.as_str()),
            _ => {}
        }
        total_bytes = total_bytes
            .checked_add(object.reference.encoded_length)
            .ok_or_else(|| {
                ArtifactError::Invalid(format!(
                    "program {inventory_name} object byte total overflowed"
                ))
            })?;
    }
    if total_bytes > MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES {
        return Err(ArtifactError::Invalid(format!(
            "program {inventory_name} object inventory exceeds its byte bound"
        )));
    }
    Ok(())
}

fn validate_reference(reference: &ValueRef) -> ArtifactResult<()> {
    ValuePayload::Object(Box::new(reference.clone()))
        .validate(ValueLimits::default())
        .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
    if !matches!(
        reference.kind,
        ValueRefKind::DriverObject | ValueRefKind::ResultObject
    ) || reference.resident_fence.is_some()
        || reference.encoded_length == 0
        || reference.encoded_length > MAX_PROGRAM_EXECUTION_OBJECT_BYTES
    {
        return Err(ArtifactError::Invalid(
            "program object reference is not portable or exceeds its bound".into(),
        ));
    }
    Ok(())
}

fn collect_references(values: &[ValuePayload]) -> Vec<&ValueRef> {
    fn visit<'a>(value: &'a ValuePayload, references: &mut Vec<&'a ValueRef>) {
        match value {
            ValuePayload::Object(reference) => references.push(reference),
            // Pool-resident partitions are addressed by their pool handle, not
            // by transported objects.
            ValuePayload::Distributed(_) | ValuePayload::Composite(_) => {}
            ValuePayload::Inline(value) => match value.as_ref() {
                runmat_execution::value::InlineValue::Cell { values, .. } => {
                    values.iter().for_each(|value| visit(value, references));
                }
                runmat_execution::value::InlineValue::Struct(fields) => fields
                    .iter()
                    .for_each(|field| visit(&field.value, references)),
                runmat_execution::value::InlineValue::Symbolic(data)
                | runmat_execution::value::InlineValue::ImmutableValueClass(data) => data
                    .fields
                    .iter()
                    .for_each(|field| visit(&field.value, references)),
                runmat_execution::value::InlineValue::Callable(callable) => callable
                    .captures
                    .iter()
                    .for_each(|value| visit(value, references)),
                _ => {}
            },
        }
    }

    let mut references = Vec::new();
    values
        .iter()
        .for_each(|value| visit(value, &mut references));
    references
}

#[cfg(test)]
mod tests {
    use runmat_execution::identity::ValueId;
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
    use runmat_execution::value::InlineValue;
    use runmat_execution::Digest;

    use super::*;
    use crate::ProgramExecutionInputs;

    fn result_reference(bytes: &[u8]) -> ValueRef {
        let logical_digest = Digest::sha256(bytes);
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
            id: ValueId::derive(&[b"program-transport-test", logical_digest.bytes()]),
            logical_digest,
            encoded_length: bytes.len() as u64,
            media_type: "application/vnd.runmat.test-object".into(),
            value_schema: "runmat.test-object.v1".into(),
            encryption_context: Digest::sha256(b"test-encryption-context"),
            kind: ValueRefKind::ResultObject,
            authorization_scope: "test-scope".into(),
            resident_fence: None,
        }
    }

    fn input_reference(bytes: &[u8]) -> ValueRef {
        let mut reference = result_reference(bytes);
        reference.kind = ValueRefKind::DriverObject;
        reference
    }

    #[test]
    fn externalized_values_bind_logical_value_identity_not_json_encoding_identity() {
        let access = crate::ArtifactAccess {
            authorization_scope: "test-scope".into(),
            encryption_context: Digest::sha256(b"test-value-encryption"),
        };
        let value = ValuePayload::Inline(Box::new(InlineValue::Dense(
            runmat_execution::value::DenseValue {
                element_type: runmat_execution::value::ElementType::F64,
                shape: vec![2, 1],
                little_endian_data: [1.0_f64, 2.0]
                    .into_iter()
                    .flat_map(|item| item.to_bits().to_le_bytes())
                    .collect(),
            },
        )));

        let object = ProgramExecutionObject::externalized_value(
            &access,
            ValueRefKind::DriverObject,
            value.clone(),
        )
        .unwrap();

        assert_ne!(
            Digest::sha256(&object.bytes),
            object.reference.logical_digest
        );
        assert_eq!(object.decode_externalized_value().unwrap(), Some(value));
        let rebound = object
            .rebind_externalized_value(&crate::ArtifactAccess {
                authorization_scope: "other-scope".into(),
                encryption_context: Digest::sha256(b"other-value-encryption"),
            })
            .unwrap();
        assert_eq!(
            rebound.reference.logical_digest,
            object.reference.logical_digest
        );
        assert_ne!(rebound.reference.id, object.reference.id);
        let mut substituted = object;
        substituted.bytes.push(b' ');
        substituted.reference.encoded_length += 1;
        assert!(substituted.validate().is_err());
    }

    #[test]
    fn externalized_inputs_require_a_complete_verified_inventory() {
        let bytes = b"canonical input object".to_vec();
        let root = input_reference(&bytes);
        let inputs = ProgramExecutionInputs {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: vec![ValuePayload::Object(Box::new(root.clone()))],
            objects: vec![ProgramExecutionObjectLocation {
                reference: root.clone(),
                artifact_id: "input-artifact".into(),
            }],
        };
        inputs.validate().unwrap();

        let mut missing = inputs.clone();
        missing.objects.clear();
        assert!(missing.validate().is_err());

        let mut malformed = inputs.clone();
        malformed.objects[0].artifact_id.clear();
        assert!(malformed.validate().is_err());

        let mut duplicate = inputs;
        duplicate.objects.push(ProgramExecutionObjectLocation {
            reference: root,
            artifact_id: "different-artifact".into(),
        });
        assert!(duplicate.validate().is_err());

        let mut resident = ProgramExecutionInputs {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: vec![ValuePayload::Object(Box::new(input_reference(b"resident")))],
            objects: Vec::new(),
        };
        if let ValuePayload::Object(reference) = &mut resident.arguments[0] {
            reference.kind = ValueRefKind::ResidentObject;
        }
        assert!(resident.validate().is_err());
    }

    #[test]
    fn terminal_outputs_bind_externalized_references_to_unique_artifact_locations() {
        let bytes = b"canonical result bytes".to_vec();
        let root = result_reference(&bytes);
        let response = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
            result_objects: vec![root.clone()],
        };
        let outputs = ProgramExecutionOutputs::new(
            response.clone(),
            vec![ProgramExecutionObjectLocation {
                reference: root.clone(),
                artifact_id: "result-artifact".into(),
            }],
        )
        .unwrap();
        outputs.validate().unwrap();
        let encoded = serde_json::to_vec(&outputs).unwrap();
        let encoded_text = String::from_utf8(encoded.clone()).unwrap();
        assert!(!encoded_text.contains("canonical result bytes"));
        assert!(encoded_text.contains("result-artifact"));
        assert_eq!(
            serde_json::from_slice::<ProgramExecutionOutputs>(&encoded).unwrap(),
            outputs
        );

        let mut tampered = outputs.clone();
        tampered.objects[0].reference.logical_digest = Digest::sha256(b"substituted");
        assert!(tampered.validate().is_err());
        let mut missing = outputs;
        missing.objects.clear();
        assert!(missing.validate().is_err());

        ProgramExecutionOutputs::new(
            ProgramExecutionResponse::Success {
                outputs: vec![ValuePayload::Inline(Box::new(
                    runmat_execution::value::InlineValue::String("done".into()),
                ))],
            },
            Vec::new(),
        )
        .unwrap();
    }

    #[test]
    fn object_closure_is_canonical_complete_and_scope_bound() {
        let first = result_reference(b"first");
        let second = result_reference(b"second");
        let closure = ProgramExecutionObjectClosure::new(
            second.clone(),
            vec![
                ProgramExecutionObjectLocation {
                    reference: second.clone(),
                    artifact_id: "second-artifact".into(),
                },
                ProgramExecutionObjectLocation {
                    reference: first,
                    artifact_id: "first-artifact".into(),
                },
            ],
        )
        .unwrap();
        let bytes = closure.canonical_bytes().unwrap();
        assert_eq!(
            ProgramExecutionObjectClosure::from_canonical_bytes(&bytes).unwrap(),
            closure
        );

        let mut missing_root = closure.clone();
        missing_root
            .objects
            .retain(|object| object.reference != second);
        assert!(missing_root.validate().is_err());

        let mut wrong_scope = closure.clone();
        wrong_scope.objects[0].reference.authorization_scope = "other-run".into();
        assert!(wrong_scope.validate().is_err());

        let mut reordered = closure;
        reordered.objects.reverse();
        assert!(reordered.validate().is_err());
    }

    fn mixed_closure() -> ProgramExecutionObjectClosure {
        let root = result_reference(b"checkpoint root");
        ProgramExecutionObjectClosure::new(
            root.clone(),
            vec![
                ProgramExecutionObjectLocation {
                    reference: root,
                    artifact_id: "root-artifact".into(),
                },
                ProgramExecutionObjectLocation {
                    reference: input_reference(b"retained driver input"),
                    artifact_id: "input-artifact".into(),
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn mixed_portable_closure_has_canonical_versioned_identity() {
        let closure = mixed_closure();
        assert_eq!(closure.schema_version, 2);
        assert_eq!(
            PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA,
            "runmat.execution-object-closure.v2"
        );
        let bytes = closure.canonical_bytes().unwrap();
        assert_eq!(
            ProgramExecutionObjectClosure::from_canonical_bytes(&bytes).unwrap(),
            closure
        );
        let mut reverse = closure.objects.clone();
        reverse.reverse();
        assert_eq!(
            ProgramExecutionObjectClosure::new(closure.root.clone(), reverse)
                .unwrap()
                .canonical_bytes()
                .unwrap(),
            bytes
        );
        let mut noncanonical = bytes.clone();
        noncanonical.push(b' ');
        assert!(ProgramExecutionObjectClosure::from_canonical_bytes(&noncanonical).is_err());
        for schema in [0, 1, 3] {
            let mut invalid = closure.clone();
            invalid.schema_version = schema;
            assert!(invalid.validate().is_err());
            assert!(ProgramExecutionObjectClosure::from_canonical_bytes(
                &serde_json::to_vec(&invalid).unwrap()
            )
            .is_err());
        }
    }

    #[test]
    fn mixed_closure_rejects_foreign_nonportable_duplicate_and_unbounded_members() {
        use runmat_execution::identity::{NodeLeaseId, WorkerId};
        use runmat_execution::value::ResidentFence;
        let closure = mixed_closure();
        let input = closure
            .objects
            .iter()
            .position(|object| object.reference.kind == ValueRefKind::DriverObject)
            .unwrap();
        for case in 0..9 {
            let mut invalid = closure.clone();
            match case {
                0 => invalid.objects[input].reference.kind = ValueRefKind::WorkerObject,
                1 => {
                    invalid.objects[input].reference.resident_fence = Some(ResidentFence {
                        worker_id: WorkerId::derive(&[b"worker"]),
                        node_lease_id: NodeLeaseId::derive(&[b"lease"]),
                        process_generation: 1,
                        device_identity: None,
                    })
                }
                2 => invalid.objects[input].reference.authorization_scope = "foreign".into(),
                3 => {
                    let mut alias = invalid.objects[input].clone();
                    alias.reference.kind = ValueRefKind::ResultObject;
                    alias.artifact_id = "alias-artifact".into();
                    invalid.objects.push(alias);
                    invalid.objects.sort_by_key(|object| object.reference.id);
                }
                4 => {
                    invalid.objects[input].reference.encoded_length =
                        MAX_PROGRAM_EXECUTION_OBJECT_BYTES + 1
                }
                5 => invalid
                    .objects
                    .retain(|object| object.reference != closure.root),
                6 => invalid.root.kind = ValueRefKind::DriverObject,
                7 => {
                    invalid.objects[input].artifact_id =
                        invalid.objects[1 - input].artifact_id.clone()
                }
                _ => invalid.objects[input].reference.logical_digest = invalid.root.logical_digest,
            }
            assert!(invalid.validate().is_err(), "corruption {case}");
        }
        let mut too_many = closure;
        too_many.objects =
            vec![too_many.objects[0].clone(); MAX_PROGRAM_EXECUTION_RESULT_OBJECTS + 1];
        assert!(too_many.validate().is_err());
    }

    #[test]
    fn mixed_closure_does_not_relax_input_or_output_kind_contracts() {
        let closure = mixed_closure();
        let input = closure
            .objects
            .iter()
            .find(|object| object.reference.kind == ValueRefKind::DriverObject)
            .unwrap()
            .clone();
        let result = closure
            .objects
            .iter()
            .find(|object| object.reference.kind == ValueRefKind::ResultObject)
            .unwrap()
            .clone();
        let inputs = |object: ProgramExecutionObjectLocation| ProgramExecutionInputs {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: vec![ValuePayload::Object(Box::new(object.reference.clone()))],
            objects: vec![object],
        };
        assert!(inputs(input.clone()).validate().is_ok());
        assert!(inputs(result.clone()).validate().is_err());
        let outputs = |object: ProgramExecutionObjectLocation| {
            ProgramExecutionOutputs::new(
                ProgramExecutionResponse::ExternalizedSuccess {
                    outputs: vec![ValuePayload::Object(Box::new(object.reference.clone()))],
                    result_objects: vec![object.reference.clone()],
                },
                vec![object],
            )
        };
        assert!(outputs(result).is_ok());
        assert!(outputs(input).is_err());
    }
}
