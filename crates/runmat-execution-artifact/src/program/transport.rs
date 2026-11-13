use std::collections::BTreeSet;

use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
pub use runmat_execution::value::{EXTERNALIZED_VALUE_MEDIA_TYPE, EXTERNALIZED_VALUE_SCHEMA};
use serde::{Deserialize, Serialize};

use super::request::validate_externalized_inventory;
#[cfg(test)]
use super::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION;
use super::{ProgramExecutionResponse, MAX_PROGRAM_EXECUTION_RESULT_OBJECTS};
use crate::{ArtifactError, ArtifactResult};

pub const MAX_PROGRAM_EXECUTION_OBJECT_BYTES: u64 = 64 * 1024 * 1024;
pub const PROGRAM_EXECUTION_OUTPUTS_SCHEMA_VERSION: u16 = 9;
pub const MAX_PROGRAM_EXECUTION_CONSOLE_SCOPES: u32 = 1024;
pub const PROGRAM_EXECUTION_CONSOLE_SCOPE_BYTES: u64 =
    runmat_execution::console::CONSOLE_SCOPE_BYTES;

pub use runmat_execution::console::ConsoleBudget as ProgramExecutionConsoleBudget;

pub use runmat_execution::console::{
    ScopedConsoleTranscript as ProgramExecutionConsoleScope,
    ScopedConsoleTranscriptRef as ProgramExecutionConsoleScopeRef,
};

/// A source calls the visitor while its original retention owner is borrowed.
pub trait ProgramExecutionConsoleSource {
    fn visit(
        &self,
        visitor: &mut dyn FnMut(ProgramExecutionConsoleScopeRef<'_>) -> ArtifactResult<()>,
    ) -> ArtifactResult<()>;
}

/// Reusable borrowed sequence for terminal artifact and private worker frames.
/// Every visit serializes before its original owner releases the borrow.
pub struct ProgramExecutionConsoleScopeSequence<'a> {
    root: Option<&'a runmat_execution::console::ConsoleTranscript>,
    source: &'a dyn ProgramExecutionConsoleSource,
    budget: Option<ProgramExecutionConsoleBudget>,
    delegated_root: Option<runmat_execution::console::ConsoleSourceIdentity>,
    require_complete_root: bool,
}
impl<'a> ProgramExecutionConsoleScopeSequence<'a> {
    pub fn new(
        root: Option<&'a runmat_execution::console::ConsoleTranscript>,
        source: &'a dyn ProgramExecutionConsoleSource,
        budget: Option<ProgramExecutionConsoleBudget>,
    ) -> Self {
        Self {
            root,
            source,
            budget,
            delegated_root: None,
            require_complete_root: false,
        }
    }

    fn with_delegated_root(
        mut self,
        identity: Option<runmat_execution::console::ConsoleSourceIdentity>,
        require_complete: bool,
    ) -> Self {
        self.delegated_root = identity;
        self.require_complete_root = require_complete;
        self
    }
}
impl Serialize for ProgramExecutionConsoleScopeSequence<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut inventory = ConsoleEncodingInventory::new(self.root, self.budget)
            .map_err(serde::ser::Error::custom)?;
        let mut sequence = serializer.serialize_seq(None)?;
        let mut found_root = false;
        if self.root.is_some() && self.delegated_root.is_some() {
            return Err(serde::ser::Error::custom("console has two root owners"));
        }
        self.source
            .visit(&mut |scope| {
                inventory.scope(&scope)?;
                let identity = runmat_execution::console::ConsoleSourceIdentity {
                    scope_id: scope.scope_id,
                    pool_id: scope.pool_id,
                    task_id: scope.task_id,
                    attempt_id: scope.attempt_id,
                };
                if Some(identity) == self.delegated_root {
                    found_root = true;
                    if self.require_complete_root
                        && scope.transcript.completion()
                            != runmat_execution::console::ConsoleCompletion::Complete
                    {
                        return Err(ArtifactError::Invalid(
                            "successful delegated root console is incomplete".into(),
                        ));
                    }
                }
                sequence
                    .serialize_element(&scope)
                    .map_err(|error| ArtifactError::Encoding(error.to_string()))
            })
            .map_err(serde::ser::Error::custom)?;
        inventory.finish().map_err(serde::ser::Error::custom)?;
        if self.delegated_root.is_some() && !found_root {
            return Err(serde::ser::Error::custom(
                "delegated root console source is missing",
            ));
        }
        sequence.end()
    }
}

struct ConsoleEncodingInventory {
    budget: Option<ProgramExecutionConsoleBudget>,
    scopes: u64,
    records: u64,
    text: u64,
    identities: BTreeSet<(
        runmat_execution::ExecutionScopeId,
        runmat_execution::PoolId,
        runmat_execution::TaskId,
        runmat_execution::identity::AttemptId,
    )>,
}

impl ConsoleEncodingInventory {
    fn new(
        root: Option<&runmat_execution::console::ConsoleTranscript>,
        budget: Option<ProgramExecutionConsoleBudget>,
    ) -> ArtifactResult<Self> {
        if let Some(budget) = budget {
            budget.validate().map_err(crate::error::contract_error)?;
        }
        let mut inventory = Self {
            budget,
            scopes: 0,
            records: 0,
            text: 0,
            identities: BTreeSet::new(),
        };
        if let Some(root) = root {
            inventory.admit(root)?;
        }
        Ok(inventory)
    }
    fn scope(&mut self, scope: &ProgramExecutionConsoleScopeRef<'_>) -> ArtifactResult<()> {
        if !self.identities.insert((
            scope.scope_id,
            scope.pool_id,
            scope.task_id,
            scope.attempt_id,
        )) {
            return Err(ArtifactError::Invalid(
                "duplicate original console source".into(),
            ));
        }
        self.admit(scope.transcript)
    }
    fn admit(
        &mut self,
        transcript: &runmat_execution::console::ConsoleTranscript,
    ) -> ArtifactResult<()> {
        transcript
            .validate()
            .map_err(crate::error::contract_error)?;
        self.scopes += 1;
        self.records += transcript.records().len() as u64;
        self.text += transcript.text_bytes();
        let budget = self.budget.ok_or_else(|| {
            ArtifactError::Invalid("captured console omits its admitted budget".into())
        })?;
        let retained = runmat_execution::console::CONSOLE_RETAINED_COPY_FACTOR
            * (self.records
                * std::mem::size_of::<runmat_execution::console::ConsoleRecord>() as u64
                + self.text
                + self.scopes * PROGRAM_EXECUTION_CONSOLE_SCOPE_BYTES);
        let encoded = 6 * self.text
            + 256 * self.records
            + self.scopes * PROGRAM_EXECUTION_CONSOLE_SCOPE_BYTES
            + 1024;
        if self.scopes > u64::from(budget.max_scopes)
            || self.records > budget.max_records
            || self.text > budget.max_text_bytes
            || retained > budget.max_retained_bytes
            || encoded > budget.max_encoded_bytes
        {
            return Err(ArtifactError::Limit(
                "original console sources exceed their admitted aggregate budget".into(),
            ));
        }
        Ok(())
    }
    fn finish(&self) -> ArtifactResult<()> {
        if self.scopes == 0 && self.budget.is_some() {
            return Err(ArtifactError::Invalid(
                "console budget without captured evidence".into(),
            ));
        }
        Ok(())
    }
}

struct ConsoleEncodingWriter {
    bytes: Vec<u8>,
    limit: u64,
}
impl std::io::Write for ConsoleEncodingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = (self.bytes.len() as u64)
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| std::io::Error::other("terminal console encoding overflow"))?;
        if next > self.limit {
            return Err(std::io::Error::other(
                "terminal output exceeds admitted encoding budget",
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(std::io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
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
    #[serde(deserialize_with = "runmat_execution::console::deserialize_transcript")]
    pub console: Option<Box<runmat_execution::console::ConsoleTranscript>>,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_scopes")]
    pub console_scopes: Vec<ProgramExecutionConsoleScope>,
    #[serde(deserialize_with = "required_console_budget")]
    pub console_budget: Option<ProgramExecutionConsoleBudget>,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_source_identity")]
    pub delegated_console_root: Option<runmat_execution::console::ConsoleSourceIdentity>,
}

impl ProgramExecutionOutputs {
    pub fn new(
        response: ProgramExecutionResponse,
        objects: Vec<ProgramExecutionObjectLocation>,
    ) -> ArtifactResult<Self> {
        Self::with_console(response, objects, None)
    }

    pub fn with_console(
        response: ProgramExecutionResponse,
        objects: Vec<ProgramExecutionObjectLocation>,
        console: Option<Box<runmat_execution::console::ConsoleTranscript>>,
    ) -> ArtifactResult<Self> {
        let budget = console
            .as_ref()
            .map(|console| ProgramExecutionConsoleBudget::single(console.limits()))
            .transpose()
            .map_err(crate::error::contract_error)?;
        Self::with_console_scopes(response, objects, console, Vec::new(), budget)
    }

    pub fn with_console_scopes(
        response: ProgramExecutionResponse,
        objects: Vec<ProgramExecutionObjectLocation>,
        console: Option<Box<runmat_execution::console::ConsoleTranscript>>,
        console_scopes: Vec<ProgramExecutionConsoleScope>,
        console_budget: Option<ProgramExecutionConsoleBudget>,
    ) -> ArtifactResult<Self> {
        let outputs = Self {
            schema_version: PROGRAM_EXECUTION_OUTPUTS_SCHEMA_VERSION,
            response,
            objects,
            console,
            console_scopes,
            console_budget,
            delegated_console_root: None,
        };
        outputs.validate()?;
        Ok(outputs)
    }

    /// Encode original sources during their owner's synchronous visit. The
    /// borrowed inventory never survives a driver lock or copies its text.
    pub fn encode_with_console_source(
        &self,
        console: Option<&runmat_execution::console::ConsoleTranscript>,
        sources: &dyn ProgramExecutionConsoleSource,
        console_budget: Option<ProgramExecutionConsoleBudget>,
        delegated_console_root: Option<runmat_execution::console::ConsoleSourceIdentity>,
    ) -> ArtifactResult<Vec<u8>> {
        self.validate()?;
        validate_root_completion(&self.response, console)?;
        if self.console.is_some()
            || !self.console_scopes.is_empty()
            || self.delegated_console_root.is_some()
        {
            return Err(ArtifactError::Invalid(
                "output already owns console evidence".into(),
            ));
        }
        #[derive(Serialize)]
        struct View<'a> {
            schema_version: u16,
            response: &'a ProgramExecutionResponse,
            objects: &'a [ProgramExecutionObjectLocation],
            console: Option<&'a runmat_execution::console::ConsoleTranscript>,
            console_scopes: ProgramExecutionConsoleScopeSequence<'a>,
            console_budget: Option<ProgramExecutionConsoleBudget>,
            delegated_console_root: Option<runmat_execution::console::ConsoleSourceIdentity>,
        }
        let view = View {
            schema_version: self.schema_version,
            response: &self.response,
            objects: &self.objects,
            console,
            console_scopes: ProgramExecutionConsoleScopeSequence::new(
                console,
                sources,
                console_budget,
            )
            .with_delegated_root(
                delegated_console_root,
                matches!(
                    &self.response,
                    ProgramExecutionResponse::Success { .. }
                        | ProgramExecutionResponse::ExternalizedSuccess { .. }
                ),
            ),
            console_budget,
            delegated_console_root,
        };
        // Program values retain their transport allowance. The independently
        // admitted console component extends the envelope, while its visitor
        // enforces console-only counters and encoding bounds.
        let limit = MAX_PROGRAM_EXECUTION_OBJECT_BYTES
            .checked_add(console_budget.map_or(0, |budget| budget.max_encoded_bytes))
            .ok_or_else(|| ArtifactError::Limit("terminal envelope bound overflow".into()))?;
        let mut output = ConsoleEncodingWriter {
            bytes: Vec::new(),
            limit,
        };
        serde_json::to_writer(&mut output, &view)
            .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
        Ok(output.bytes)
    }

    pub fn encode_with_retained_console(
        &self,
        console: Option<&runmat_execution::console::ConsoleTranscript>,
        scopes: &[ProgramExecutionConsoleScopeRef<'_>],
        budget: Option<ProgramExecutionConsoleBudget>,
        delegated_console_root: Option<runmat_execution::console::ConsoleSourceIdentity>,
    ) -> ArtifactResult<Vec<u8>> {
        struct Slice<'a, 'b>(&'a [ProgramExecutionConsoleScopeRef<'b>]);
        impl ProgramExecutionConsoleSource for Slice<'_, '_> {
            fn visit(
                &self,
                visitor: &mut dyn FnMut(ProgramExecutionConsoleScopeRef<'_>) -> ArtifactResult<()>,
            ) -> ArtifactResult<()> {
                for scope in self.0 {
                    visitor(ProgramExecutionConsoleScopeRef {
                        scope_id: scope.scope_id,
                        pool_id: scope.pool_id,
                        task_id: scope.task_id,
                        attempt_id: scope.attempt_id,
                        transcript: scope.transcript,
                    })?;
                }
                Ok(())
            }
        }
        self.encode_with_console_source(console, &Slice(scopes), budget, delegated_console_root)
    }

    /// The original physical root remains in its source inventory when delegated.
    pub fn root_console(&self) -> Option<&runmat_execution::console::ConsoleTranscript> {
        runmat_execution::console::original_root(
            self.console.as_deref(),
            &self.console_scopes,
            self.delegated_console_root,
        )
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_OUTPUTS_SCHEMA_VERSION {
            return Err(ArtifactError::Invalid(
                "program outputs use an unsupported schema".into(),
            ));
        }
        if let Some(console) = &self.console {
            console
                .validate()
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        }
        validate_root_completion(&self.response, self.console.as_deref())?;
        runmat_execution::console::validate_scopes(&self.console_scopes)
            .map_err(crate::error::contract_error)?;
        runmat_execution::console::validate_delegated_root(
            self.console.as_deref(),
            &self.console_scopes,
            self.delegated_console_root,
            matches!(
                &self.response,
                ProgramExecutionResponse::Success { .. }
                    | ProgramExecutionResponse::ExternalizedSuccess { .. }
            ),
        )
        .map_err(crate::error::contract_error)?;
        validate_console_batch(
            self.console.as_deref(),
            self.console_scopes
                .iter()
                .map(|scope| scope.transcript.as_ref()),
            self.console_budget,
        )?;
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

fn validate_root_completion(
    response: &ProgramExecutionResponse,
    root: Option<&runmat_execution::console::ConsoleTranscript>,
) -> ArtifactResult<()> {
    if matches!(
        response,
        ProgramExecutionResponse::Success { .. }
            | ProgramExecutionResponse::ExternalizedSuccess { .. }
    ) && root.is_some_and(|root| {
        root.completion() != runmat_execution::console::ConsoleCompletion::Complete
    }) {
        return Err(ArtifactError::Invalid(
            "successful program output carries an incomplete root console".into(),
        ));
    }
    Ok(())
}

fn validate_locations(
    objects: &[ProgramExecutionObjectLocation],
    expected_kind: ValueRefKind,
    inventory_name: &str,
) -> ArtifactResult<()> {
    validate_location_inventory(objects, LocationKinds::Exact(expected_kind), inventory_name)
}

fn required_console_budget<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ProgramExecutionConsoleBudget>, D::Error> {
    let budget = Option::<ProgramExecutionConsoleBudget>::deserialize(deserializer)?;
    if let Some(budget) = budget {
        budget.validate().map_err(serde::de::Error::custom)?;
    }
    Ok(budget)
}

fn validate_console_batch<'a>(
    root: Option<&'a runmat_execution::console::ConsoleTranscript>,
    scopes: impl Iterator<Item = &'a runmat_execution::console::ConsoleTranscript>,
    budget: Option<ProgramExecutionConsoleBudget>,
) -> ArtifactResult<()> {
    let mut count = 0_u64;
    let mut records = 0_u64;
    let mut text = 0_u64;
    for transcript in root.into_iter().chain(scopes) {
        transcript
            .validate()
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        count = count
            .checked_add(1)
            .ok_or_else(|| ArtifactError::Limit("console scope count overflow".into()))?;
        records = records
            .checked_add(transcript.records().len() as u64)
            .ok_or_else(|| ArtifactError::Limit("console record count overflow".into()))?;
        text = text
            .checked_add(transcript.text_bytes())
            .ok_or_else(|| ArtifactError::Limit("console text count overflow".into()))?;
        if count > u64::from(MAX_PROGRAM_EXECUTION_CONSOLE_SCOPES) {
            return Err(ArtifactError::Limit(
                "console scope inventory exceeds its bound".into(),
            ));
        }
    }
    if count == 0 {
        if budget.is_some() {
            return Err(ArtifactError::Invalid(
                "console budget without captured evidence".into(),
            ));
        }
        return Ok(());
    }
    let budget = budget.ok_or_else(|| {
        ArtifactError::Invalid("captured console omits its aggregate budget".into())
    })?;
    budget.validate().map_err(crate::error::contract_error)?;
    let retained = runmat_execution::console::CONSOLE_RETAINED_COPY_FACTOR
        * (records * std::mem::size_of::<runmat_execution::console::ConsoleRecord>() as u64
            + text
            + count * PROGRAM_EXECUTION_CONSOLE_SCOPE_BYTES);
    let encoded = text * 6 + records * 256 + count * PROGRAM_EXECUTION_CONSOLE_SCOPE_BYTES + 1024;
    if count > u64::from(budget.max_scopes)
        || records > budget.max_records
        || text > budget.max_text_bytes
        || retained > budget.max_retained_bytes
        || encoded > budget.max_encoded_bytes
    {
        return Err(ArtifactError::Limit(
            "terminal console inventory exceeds its admitted aggregate budget".into(),
        ));
    }
    Ok(())
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

    #[test]
    fn successful_outputs_require_complete_root_but_failure_keeps_prefix() {
        use runmat_execution::console::{ConsoleCapture, ConsoleLimits, ConsoleStream};
        let limits = ConsoleLimits {
            max_records: 4,
            max_text_bytes: 64,
        };
        let mut capture = ConsoleCapture::new(limits).unwrap();
        capture
            .push_borrowed(ConsoleStream::Stdout, "before failure\n")
            .unwrap();
        let interrupted = Box::new(capture.finish(false));
        let success = ProgramExecutionOutputs::new(
            ProgramExecutionResponse::Success { outputs: vec![] },
            vec![],
        )
        .unwrap();
        let budget = ProgramExecutionConsoleBudget::single(limits).unwrap();
        let returned = ProgramExecutionOutputs::new(
            ProgramExecutionResponse::Success {
                outputs: vec![ValuePayload::Inline(Box::new(InlineValue::String(
                    "r".repeat(8192).into(),
                )))],
            },
            vec![],
        )
        .unwrap();
        let complete = ConsoleCapture::new(limits).unwrap().finish(true);
        let encoded = returned
            .encode_with_retained_console(Some(&complete), &[], Some(budget), None)
            .unwrap();
        assert!(encoded.len() as u64 > budget.max_encoded_bytes);
        let decoded: ProgramExecutionOutputs = serde_json::from_slice(&encoded).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.response, returned.response);
        assert!(success
            .encode_with_retained_console(Some(&interrupted), &[], Some(budget), None)
            .is_err());
        let mut failed = ProgramExecutionOutputs::with_console(
            ProgramExecutionResponse::Failure {
                message: "original failure".into(),
            },
            vec![],
            Some(interrupted),
        )
        .unwrap();
        assert!(failed.validate().is_ok());
        failed.response = ProgramExecutionResponse::Success { outputs: vec![] };
        assert!(failed.validate().is_err());
    }

    #[test]
    fn scoped_failed_attempt_prefix_is_independent_of_successful_root() {
        use runmat_execution::console::{ConsoleCapture, ConsoleLimits};
        let limits = ConsoleLimits {
            max_records: 4,
            max_text_bytes: 64,
        };
        let root = Box::new(ConsoleCapture::new(limits).unwrap().finish(true));
        let scope = ProgramExecutionConsoleScope {
            scope_id: runmat_execution::ExecutionScopeId::derive(&[b"child"]),
            pool_id: runmat_execution::PoolId::derive(&[b"child"]),
            task_id: runmat_execution::TaskId::derive(&[b"child"]),
            attempt_id: runmat_execution::identity::AttemptId::derive(&[b"child"]),
            transcript: Box::new(ConsoleCapture::new(limits).unwrap().finish(false)),
        };
        let budget = ProgramExecutionConsoleBudget::for_driver_allocation(1024 * 1024 * 1024);
        let mut outputs = ProgramExecutionOutputs::with_console_scopes(
            ProgramExecutionResponse::Success { outputs: vec![] },
            vec![],
            Some(root),
            vec![scope],
            Some(budget),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&outputs).unwrap();
        let decoded: ProgramExecutionOutputs = serde_json::from_slice(&bytes).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, outputs);
        outputs.console_budget.as_mut().unwrap().max_scopes = 1;
        assert!(outputs.validate().is_err());
        let mut value = serde_json::to_value(decoded).unwrap();
        value.as_object_mut().unwrap().remove("console_scopes");
        assert!(serde_json::from_value::<ProgramExecutionOutputs>(value).is_err());
    }

    #[test]
    fn terminal_wire_preserves_delegated_root_identity_and_completion() {
        use runmat_execution::console::{ConsoleCapture, ConsoleLimits, ConsoleStream};
        let limits = ConsoleLimits {
            max_records: 4,
            max_text_bytes: 64,
        };
        let identity = runmat_execution::console::ConsoleSourceIdentity {
            scope_id: runmat_execution::ExecutionScopeId::derive(&[b"physical-root"]),
            pool_id: runmat_execution::PoolId::derive(&[b"physical-root"]),
            task_id: runmat_execution::TaskId::derive(&[b"physical-root"]),
            attempt_id: runmat_execution::identity::AttemptId::derive(&[b"physical-root"]),
        };
        let budget = ProgramExecutionConsoleBudget::for_driver_allocation(1024 * 1024 * 1024);
        for complete in [false, true] {
            let mut capture = ConsoleCapture::new(limits).unwrap();
            capture
                .push(ConsoleStream::Stdout, "original output".into())
                .unwrap();
            let capture = capture.finish(complete);
            let scope = ProgramExecutionConsoleScopeRef {
                scope_id: identity.scope_id,
                pool_id: identity.pool_id,
                task_id: identity.task_id,
                attempt_id: identity.attempt_id,
                transcript: &capture,
            };
            let outputs = ProgramExecutionOutputs::new(
                ProgramExecutionResponse::Success { outputs: vec![] },
                vec![],
            )
            .unwrap();
            let encoded =
                outputs.encode_with_retained_console(None, &[scope], Some(budget), Some(identity));
            if !complete {
                assert!(encoded.is_err());
                continue;
            }
            let mut decoded: ProgramExecutionOutputs =
                serde_json::from_slice(&encoded.unwrap()).unwrap();
            decoded.validate().unwrap();
            assert!(decoded.console.is_none());
            assert_eq!(decoded.delegated_console_root, Some(identity));
            assert_eq!(decoded.root_console(), Some(&capture));
            decoded.delegated_console_root.as_mut().unwrap().attempt_id =
                runmat_execution::identity::AttemptId::derive(&[b"unrelated-source"]);
            assert!(decoded.validate().is_err());
        }
    }

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

        assert_eq!(
            object.reference.schema_version,
            VALUE_PAYLOAD_SCHEMA_VERSION
        );
        object.validate().unwrap();
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
            rebound.reference.schema_version,
            VALUE_PAYLOAD_SCHEMA_VERSION
        );
        rebound.validate().unwrap();
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
                        accelerator: None,
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
