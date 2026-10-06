use std::borrow::Borrow;
use std::collections::BTreeSet;

use minicbor::Encoder;
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionInputs};
use crate::{ArtifactError, ArtifactResult};

pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_V1: u16 = 1;
pub const MAX_PROGRAM_EXECUTION_ARGUMENTS: usize = 4096;
pub const MAX_PROGRAM_EXECUTION_RESULT_OBJECTS: usize = 65_539;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionDescriptor {
    pub schema_version: u16,
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
    pub function: usize,
    pub requested_outputs: u16,
}

impl ProgramExecutionDescriptor {
    pub fn validate(&self) -> ArtifactResult<()> {
        self.artifact.validate_against(&self.recipe)?;
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V1
            || !entrypoint_matches(self.artifact.form, self.function, &self.recipe.entrypoint)
            || self.requested_outputs != self.recipe.outputs.requested_outputs
        {
            return Err(ArtifactError::Invalid(
                "program descriptor has an inconsistent callable or output contract".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_for_portable_host(&self) -> ArtifactResult<()> {
        self.validate()?;
        self.artifact.target.validate_for_portable_host()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionRequest<
    R = ProgramBuildRecipe,
    A = ProgramArtifact,
    V = Vec<ValuePayload>,
> {
    pub schema_version: u16,
    pub recipe: R,
    pub artifact: A,
    pub function: usize,
    pub arguments: V,
    pub requested_outputs: u16,
}

/// Original request fields borrowed for validation, identity and transport encoding.
/// Owned deserialization retains the same default request type and wire schema.
pub type BorrowedProgramExecutionRequest<'a> =
    ProgramExecutionRequest<&'a ProgramBuildRecipe, &'a ProgramArtifact, &'a [ValuePayload]>;

impl ProgramExecutionRequest {
    pub fn from_parts(
        descriptor: ProgramExecutionDescriptor,
        inputs: ProgramExecutionInputs,
    ) -> ArtifactResult<Self> {
        descriptor.validate()?;
        inputs.validate()?;
        let request = Self {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe: descriptor.recipe,
            artifact: descriptor.artifact,
            function: descriptor.function,
            arguments: inputs.arguments,
            requested_outputs: descriptor.requested_outputs,
        };
        request.validate()?;
        Ok(request)
    }
}

impl<R: Borrow<ProgramBuildRecipe>, A: Borrow<ProgramArtifact>, V: AsRef<[ValuePayload]>>
    ProgramExecutionRequest<R, A, V>
{
    pub fn as_borrowed(&self) -> BorrowedProgramExecutionRequest<'_> {
        ProgramExecutionRequest {
            schema_version: self.schema_version,
            recipe: self.recipe.borrow(),
            artifact: self.artifact.borrow(),
            function: self.function,
            arguments: self.arguments.as_ref(),
            requested_outputs: self.requested_outputs,
        }
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        let recipe = self.recipe.borrow();
        let artifact = self.artifact.borrow();
        let arguments = self.arguments.as_ref();
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V1 {
            return Err(ArtifactError::Invalid(
                "unsupported program execution request schema".into(),
            ));
        }
        artifact.validate_against(recipe)?;
        if !entrypoint_matches(artifact.form, self.function, &recipe.entrypoint)
            || self.requested_outputs != recipe.outputs.requested_outputs
            || arguments.len() > MAX_PROGRAM_EXECUTION_ARGUMENTS
            || (matches!(
                artifact.form,
                ExecutableForm::InterpreterScriptV1 | ExecutableForm::TestAttemptV1
            ) && !arguments.is_empty())
        {
            return Err(ArtifactError::Invalid(
                "program execution request has an inconsistent callable, output contract, or argument count".into(),
            ));
        }
        if artifact.form == ExecutableForm::ExecutableUnitV4 {
            let envelope = artifact
                .executable_unit()?
                .expect("executable-unit form returns its validated envelope");
            if usize::try_from(envelope.manifest.identity.entrypoint_function.0).ok()
                != Some(self.function)
                || (envelope.manifest.identity.entrypoint_kind
                    == runmat_execution::ExecutableEntrypointKind::Script
                    && !arguments.is_empty())
            {
                return Err(ArtifactError::Invalid(
                    "executable unit request does not match its declared entrypoint".into(),
                ));
            }
        }
        for argument in arguments {
            argument
                .validate(ValueLimits::default())
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        }
        Ok(())
    }

    pub fn validate_for_portable_host(&self) -> ArtifactResult<()> {
        self.validate()?;
        self.artifact.borrow().target.validate_for_portable_host()
    }

    pub fn canonical_digest(&self) -> ArtifactResult<Digest> {
        self.validate()?;
        let recipe_id = self.recipe.borrow().id()?;
        let function = u64::try_from(self.function)
            .map_err(|_| ArtifactError::Limit("program function index exceeds u64".into()))?;
        let argument_count = u64::try_from(self.arguments.as_ref().len())
            .map_err(|_| ArtifactError::Limit("program argument count exceeds u64".into()))?;
        let mut bytes = b"runmat-program-execution-request\0".to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .array(6)
            .and_then(|encoder| encoder.u16(self.schema_version))
            .and_then(|encoder| encoder.bytes(recipe_id.0.bytes()))
            .and_then(|encoder| encoder.bytes(self.artifact.borrow().id.0.bytes()))
            .and_then(|encoder| encoder.u64(function))
            .and_then(|encoder| encoder.u16(self.requested_outputs))
            .and_then(|encoder| encoder.array(argument_count))
            .map_err(encoding)?;
        for argument in self.arguments.as_ref() {
            let argument = serde_json::to_vec(argument)
                .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
            encoder
                .bytes(Digest::sha256(argument).bytes())
                .map_err(encoding)?;
        }
        Ok(Digest::sha256(bytes))
    }
}

fn encoding(error: minicbor::encode::Error<std::convert::Infallible>) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}

fn entrypoint_matches(form: ExecutableForm, function: usize, entrypoint: &str) -> bool {
    match form {
        ExecutableForm::InterpreterBytecodeV1 => function.to_string() == entrypoint,
        ExecutableForm::InterpreterScriptV1 => function == 0 && entrypoint == "script",
        ExecutableForm::TestAttemptV1 => function == 0 && entrypoint == "test_attempt",
        ExecutableForm::MeshingWorkload => function == 0 && entrypoint == "meshing_workload",
        ExecutableForm::AnalysisWorkload => function == 0 && entrypoint == "analysis_workload",
        ExecutableForm::GeometryWorkload => function == 0 && entrypoint == "geometry_workload",
        ExecutableForm::ExecutableUnitV4 => function.to_string() == entrypoint,
        ExecutableForm::NativeObjectV1 => function.to_string() == entrypoint,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramExecutionResponse {
    Success {
        value: ValuePayload,
    },
    ExternalizedSuccess {
        outputs: Vec<ValuePayload>,
        result_objects: Vec<ValueRef>,
    },
    Failure {
        message: String,
    },
}

impl ProgramExecutionResponse {
    /// Preserve inline versus object-backed delivery when projecting admitted task outputs.
    /// The resulting response still requires validation against its original request.
    pub fn from_success_outputs(outputs: Vec<ValuePayload>, result_objects: Vec<ValueRef>) -> Self {
        if result_objects.is_empty() && outputs.len() == 1 {
            Self::Success {
                value: outputs
                    .into_iter()
                    .next()
                    .expect("one output was established"),
            }
        } else {
            Self::ExternalizedSuccess {
                outputs,
                result_objects,
            }
        }
    }

    pub fn validate_against<
        R: Borrow<ProgramBuildRecipe>,
        A: Borrow<ProgramArtifact>,
        V: AsRef<[ValuePayload]>,
    >(
        &self,
        request: &ProgramExecutionRequest<R, A, V>,
    ) -> ArtifactResult<()> {
        request.validate()?;
        match self {
            Self::Success { value } => {
                if request.requested_outputs != 1 {
                    return Err(ArtifactError::Invalid(
                        "single-value response differs from its output contract".into(),
                    ));
                }
                validate_externalized_inventory(
                    std::slice::from_ref(value),
                    &[],
                    request.requested_outputs,
                )
            }
            Self::ExternalizedSuccess {
                outputs,
                result_objects,
            } => {
                validate_externalized_inventory(outputs, result_objects, request.requested_outputs)
            }
            Self::Failure { message } => {
                if message.is_empty() || message.len() > 1024 * 1024 {
                    return Err(ArtifactError::Limit(
                        "program failure message is empty or exceeds its byte bound".into(),
                    ));
                }
                Ok(())
            }
        }
    }
}

pub(super) fn validate_externalized_inventory(
    outputs: &[ValuePayload],
    result_objects: &[ValueRef],
    requested_outputs: u16,
) -> ArtifactResult<()> {
    if outputs.len() != usize::from(requested_outputs)
        || result_objects.len() > MAX_PROGRAM_EXECUTION_RESULT_OBJECTS
    {
        return Err(ArtifactError::Limit(
            "externalized program response exceeds its output or object inventory contract".into(),
        ));
    }
    for output in outputs {
        output
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
    }
    let mut value_ids = BTreeSet::new();
    let mut logical_digests = BTreeSet::new();
    for object in result_objects {
        ValuePayload::Object(Box::new(object.clone()))
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        if object.kind != ValueRefKind::ResultObject
            || object.resident_fence.is_some()
            || !value_ids.insert(object.id)
            || !logical_digests.insert(object.logical_digest)
        {
            return Err(ArtifactError::Invalid(
                "externalized program response inventory is invalid or duplicated".into(),
            ));
        }
    }
    if outputs.iter().any(|output| match output {
        ValuePayload::Object(reference) if reference.kind == ValueRefKind::ResultObject => {
            !result_objects.contains(reference.as_ref())
        }
        _ => false,
    }) {
        return Err(ArtifactError::Invalid(
            "externalized program response inventory omits a result root".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use runmat_execution::identity::ValueId;
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;
    use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};

    use super::*;
    use crate::ExecutableForm;

    fn request() -> ProgramExecutionRequest {
        let revision = ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: crate::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "7".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target: crate::ProgramTarget::portable("portable-test"),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV1,
            b"program".to_vec(),
        )
        .unwrap();
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe,
            artifact,
            function: 7,
            arguments: Vec::new(),
            requested_outputs: 1,
        }
    }

    fn result_reference(bytes: &[u8]) -> ValueRef {
        let logical_digest = Digest::sha256(bytes);
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_V1,
            id: ValueId::derive(&[b"program-response-test", logical_digest.bytes()]),
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

    #[test]
    fn exact_program_request_validates_every_identity_boundary() {
        request().validate().unwrap();
        let mut mismatched = request();
        mismatched.function = 8;
        assert!(mismatched.validate().is_err());
        let mut tampered = request();
        tampered.artifact.executable_bytes.push(0);
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn exact_program_request_rejects_unknown_schemas_and_output_drift() {
        let mut unknown = request();
        unknown.schema_version += 1;
        assert!(unknown.validate().is_err());
        let mut outputs = request();
        outputs.requested_outputs = 2;
        assert!(outputs.validate().is_err());
    }

    #[test]
    fn canonical_request_digest_is_exact_without_a_small_envelope_limit() {
        let mut large = request();
        large.arguments.push(ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::String("x".repeat(300_000)),
        )));
        let digest = large.canonical_digest().unwrap();
        let mut changed = large;
        changed.arguments[0] = ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::String(format!("{}y", "x".repeat(299_999))),
        ));
        assert_ne!(digest, changed.canonical_digest().unwrap());

        let mut referenced = request();
        let mut reference = result_reference(b"argument");
        reference.kind = ValueRefKind::DriverObject;
        referenced
            .arguments
            .push(ValuePayload::Object(Box::new(reference.clone())));
        let digest = referenced.canonical_digest().unwrap();
        if let ValuePayload::Object(reference) = &mut referenced.arguments[0] {
            reference.authorization_scope = "another-scope".into();
        }
        assert_ne!(digest, referenced.canonical_digest().unwrap());
    }

    #[test]
    fn borrowed_request_preserves_original_payload_wire_identity_and_cold_decode() {
        let mut owned = request();
        owned.arguments.push(ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::String("λ\\\"\n".repeat(25_000)),
        )));
        let borrowed = owned.as_borrowed();
        assert!(std::ptr::eq(borrowed.recipe, &owned.recipe));
        assert!(std::ptr::eq(borrowed.artifact, &owned.artifact));
        assert_eq!(borrowed.arguments.as_ptr(), owned.arguments.as_ptr());
        borrowed.validate_for_portable_host().unwrap();
        let bytes = serde_json::to_vec(&borrowed).unwrap();
        assert_eq!(bytes, serde_json::to_vec(&owned).unwrap());
        assert_eq!(
            borrowed.canonical_digest().unwrap(),
            owned.canonical_digest().unwrap()
        );
        let cold: ProgramExecutionRequest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(cold, owned);
        cold.validate_for_portable_host().unwrap();
        let response = ProgramExecutionResponse::Success {
            value: ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::Logical(
                true,
            ))),
        };
        response.validate_against(&borrowed).unwrap();
        response.validate_against(&cold).unwrap();
        let mut foreign = borrowed;
        foreign.function += 1;
        assert!(foreign.validate().is_err());
        assert!(response.validate_against(&foreign).is_err());
    }

    #[test]
    fn script_form_has_an_explicit_argument_free_entrypoint() {
        let mut script = request();
        script.recipe.entrypoint = "script".into();
        script.function = 0;
        script.artifact = ProgramArtifact::materialize(
            &script.recipe,
            ExecutableForm::InterpreterScriptV1,
            b"script-bytecode".to_vec(),
        )
        .unwrap();
        script.validate().unwrap();
        script
            .arguments
            .push(runmat_execution::value::ValuePayload::Inline(Box::new(
                runmat_execution::value::InlineValue::String("unexpected".into()),
            )));
        assert!(script.validate().is_err());
    }

    #[test]
    fn task_success_projection_preserves_inline_delivery_and_request_arity() {
        let request = request();
        let value = ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::Logical(
            true,
        )));
        let response = ProgramExecutionResponse::from_success_outputs(vec![value.clone()], vec![]);
        assert_eq!(
            response,
            ProgramExecutionResponse::Success {
                value: value.clone()
            }
        );
        response.validate_against(&request).unwrap();
        for outputs in [vec![], vec![value.clone(), value]] {
            assert!(
                ProgramExecutionResponse::from_success_outputs(outputs, vec![])
                    .validate_against(&request)
                    .is_err()
            );
        }
        let root = result_reference(b"root");
        let response = ProgramExecutionResponse::from_success_outputs(
            vec![ValuePayload::Object(Box::new(root.clone()))],
            vec![root],
        );
        assert!(matches!(
            response,
            ProgramExecutionResponse::ExternalizedSuccess { .. }
        ));
        response.validate_against(&request).unwrap();
    }

    #[test]
    fn inline_output_lists_preserve_exact_arity_through_cold_transport() {
        for count in [0, 2, 3] {
            let mut request = request();
            request.requested_outputs = count;
            request.recipe.outputs.requested_outputs = count;
            request.artifact = ProgramArtifact::materialize(
                &request.recipe,
                ExecutableForm::InterpreterBytecodeV1,
                b"program".to_vec(),
            )
            .unwrap();
            let outputs = (0..count)
                .map(|index| {
                    ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::F64Bits(
                        f64::from(index).to_bits(),
                    )))
                })
                .collect();
            let response = ProgramExecutionResponse::from_success_outputs(outputs, vec![]);
            response.validate_against(&request).unwrap();
            let cold: ProgramExecutionResponse =
                serde_json::from_slice(&serde_json::to_vec(&response).unwrap()).unwrap();
            assert_eq!(cold, response);
            cold.validate_against(&request).unwrap();
            request.requested_outputs = 1;
            request.recipe.outputs.requested_outputs = 1;
            request.artifact = ProgramArtifact::materialize(
                &request.recipe,
                ExecutableForm::InterpreterBytecodeV1,
                b"program".to_vec(),
            )
            .unwrap();
            assert!(cold.validate_against(&request).is_err());
        }
        let missing = ProgramExecutionResponse::from_success_outputs(
            vec![ValuePayload::Object(Box::new(result_reference(b"root")))],
            vec![],
        );
        assert!(missing.validate_against(&request()).is_err());
    }

    #[test]
    fn externalized_response_requires_a_complete_unique_result_inventory() {
        let request = request();
        let root = result_reference(b"root");
        let response = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
            result_objects: vec![root.clone(), result_reference(b"chunk")],
        };
        response.validate_against(&request).unwrap();

        let missing_root = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
            result_objects: vec![result_reference(b"chunk")],
        };
        assert!(missing_root.validate_against(&request).is_err());

        let duplicate = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
            result_objects: vec![root.clone(), root],
        };
        assert!(duplicate.validate_against(&request).is_err());
    }
}
