use std::collections::BTreeSet;

use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use serde::{Deserialize, Serialize};

use super::request::validate_externalized_inventory;
use super::{
    ProgramExecutionResponse, MAX_PROGRAM_EXECUTION_ARGUMENTS,
    MAX_PROGRAM_EXECUTION_RESULT_OBJECTS, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use crate::{ArtifactError, ArtifactResult};

pub const MAX_PROGRAM_EXECUTION_OBJECT_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionObject {
    pub reference: ValueRef,
    #[serde(with = "base64_bytes")]
    pub bytes: Vec<u8>,
}

impl ProgramExecutionObject {
    pub fn validate(&self) -> ArtifactResult<()> {
        ValuePayload::Object(Box::new(self.reference.clone()))
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        if !matches!(
            self.reference.kind,
            ValueRefKind::DriverObject | ValueRefKind::ResultObject
        ) || self.reference.resident_fence.is_some()
            || self.reference.encoded_length == 0
            || self.reference.encoded_length > MAX_PROGRAM_EXECUTION_OBJECT_BYTES
            || self.reference.encoded_length != self.bytes.len() as u64
            || self.reference.logical_digest != runmat_execution::Digest::sha256(&self.bytes)
        {
            return Err(ArtifactError::Invalid(
                "program object differs from its portable reference".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionInputs {
    pub schema_version: u16,
    pub arguments: Vec<ValuePayload>,
    pub objects: Vec<ProgramExecutionObject>,
}

impl ProgramExecutionInputs {
    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V1
            || self.arguments.len() > MAX_PROGRAM_EXECUTION_ARGUMENTS
        {
            return Err(ArtifactError::Invalid(
                "program inputs use an unsupported schema or exceed their bound".into(),
            ));
        }
        for argument in &self.arguments {
            argument
                .validate(ValueLimits::default())
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        }
        validate_objects(&self.objects, ValueRefKind::DriverObject, "input")?;
        for reference in collect_references(&self.arguments) {
            if reference.kind != ValueRefKind::DriverObject {
                return Err(ArtifactError::Invalid(
                    "program inputs contain a non-portable external argument".into(),
                ));
            }
            if !self
                .objects
                .iter()
                .any(|object| object.reference == *reference)
            {
                return Err(ArtifactError::Invalid(
                    "program input object inventory omits an external argument".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionOutputs {
    pub schema_version: u16,
    pub response: ProgramExecutionResponse,
    pub objects: Vec<ProgramExecutionObject>,
}

impl ProgramExecutionOutputs {
    pub fn from_response<F>(response: ProgramExecutionResponse, mut read: F) -> ArtifactResult<Self>
    where
        F: FnMut(&ValueRef) -> ArtifactResult<Option<Vec<u8>>>,
    {
        let references: &[ValueRef] = match &response {
            ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } => result_objects,
            _ => &[],
        };
        let mut objects = Vec::with_capacity(references.len());
        for reference in references {
            let bytes = read(reference)?.ok_or_else(|| {
                ArtifactError::Invalid("program result object is absent from its store".into())
            })?;
            objects.push(ProgramExecutionObject {
                reference: reference.clone(),
                bytes,
            });
        }
        let outputs = Self {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            response,
            objects,
        };
        outputs.validate()?;
        Ok(outputs)
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V1 {
            return Err(ArtifactError::Invalid(
                "program outputs use an unsupported schema".into(),
            ));
        }
        validate_objects(&self.objects, ValueRefKind::ResultObject, "result")?;
        match &self.response {
            ProgramExecutionResponse::Success { value } => {
                value
                    .validate(ValueLimits::default())
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
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
                validate_externalized_inventory(outputs, result_objects, requested_outputs)?;
            }
            ProgramExecutionResponse::Failure { message } => {
                if message.is_empty() || message.len() > 1024 * 1024 || !self.objects.is_empty() {
                    return Err(ArtifactError::Invalid(
                        "failed program output is malformed".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn validate_objects(
    objects: &[ProgramExecutionObject],
    expected_kind: ValueRefKind,
    inventory_name: &str,
) -> ArtifactResult<()> {
    if objects.len() > MAX_PROGRAM_EXECUTION_RESULT_OBJECTS {
        return Err(ArtifactError::Invalid(format!(
            "program {inventory_name} object inventory exceeds its count bound"
        )));
    }
    let mut ids = BTreeSet::new();
    let mut digests = BTreeSet::new();
    let mut authorization_scope = None;
    let mut total_bytes = 0_u64;
    for object in objects {
        object.validate()?;
        if object.reference.kind != expected_kind
            || !ids.insert(object.reference.id)
            || !digests.insert(object.reference.logical_digest)
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

fn collect_references(values: &[ValuePayload]) -> Vec<&ValueRef> {
    fn visit<'a>(value: &'a ValuePayload, references: &mut Vec<&'a ValueRef>) {
        match value {
            ValuePayload::Object(reference) => references.push(reference),
            ValuePayload::Inline(value) => match value.as_ref() {
                runmat_execution::value::InlineValue::Cell { values, .. }
                | runmat_execution::value::InlineValue::OutputList(values) => {
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

mod base64_bytes {
    use base64::Engine as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use runmat_execution::identity::ValueId;
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;
    use runmat_execution::Digest;

    use super::*;

    fn result_reference(bytes: &[u8]) -> ValueRef {
        let logical_digest = Digest::sha256(bytes);
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_V1,
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
    fn externalized_inputs_require_a_complete_verified_inventory() {
        let bytes = b"canonical input object".to_vec();
        let root = input_reference(&bytes);
        let inputs = ProgramExecutionInputs {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            arguments: vec![ValuePayload::Object(Box::new(root.clone()))],
            objects: vec![ProgramExecutionObject {
                reference: root.clone(),
                bytes: bytes.clone(),
            }],
        };
        inputs.validate().unwrap();

        let mut missing = inputs.clone();
        missing.objects.clear();
        assert!(missing.validate().is_err());

        let mut corrupt = inputs.clone();
        corrupt.objects[0].bytes.push(0);
        assert!(corrupt.validate().is_err());

        let mut duplicate = inputs;
        duplicate.objects.push(ProgramExecutionObject {
            reference: root,
            bytes,
        });
        assert!(duplicate.validate().is_err());

        let mut resident = ProgramExecutionInputs {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            arguments: vec![ValuePayload::Object(Box::new(input_reference(b"resident")))],
            objects: Vec::new(),
        };
        if let ValuePayload::Object(reference) = &mut resident.arguments[0] {
            reference.kind = ValueRefKind::ResidentObject;
        }
        assert!(resident.validate().is_err());
    }

    #[test]
    fn terminal_outputs_bind_externalized_references_to_verified_bytes() {
        let bytes = b"canonical result bytes".to_vec();
        let root = result_reference(&bytes);
        let response = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
            result_objects: vec![root.clone()],
        };
        let outputs = ProgramExecutionOutputs::from_response(response.clone(), |reference| {
            Ok((reference == &root).then_some(bytes.clone()))
        })
        .unwrap();
        outputs.validate().unwrap();
        let encoded = serde_json::to_vec(&outputs).unwrap();
        let encoded_text = String::from_utf8(encoded.clone()).unwrap();
        assert!(!encoded_text.contains("\"bytes\":["));
        assert!(encoded_text.contains("Y2Fub25pY2FsIHJlc3VsdCBieXRlcw"));
        assert_eq!(
            serde_json::from_slice::<ProgramExecutionOutputs>(&encoded).unwrap(),
            outputs
        );

        assert!(ProgramExecutionOutputs::from_response(response, |_| Ok(None)).is_err());
        let mut tampered = outputs.clone();
        tampered.objects[0].bytes.push(0);
        assert!(tampered.validate().is_err());
        let mut missing = outputs;
        missing.objects.clear();
        assert!(missing.validate().is_err());

        ProgramExecutionOutputs::from_response(
            ProgramExecutionResponse::Success {
                value: ValuePayload::Inline(Box::new(
                    runmat_execution::value::InlineValue::String("done".into()),
                )),
            },
            |_| unreachable!(),
        )
        .unwrap();
    }
}
