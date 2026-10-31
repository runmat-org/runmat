use serde::{Deserialize, Serialize};

use crate::identity::{Digest, NodeLeaseId, ValueId, WorkerId};
use crate::ContractError;

mod admission;
mod catalog;
mod durable_aggregate;
mod exception;
mod graph;
mod identity;
mod limits;
mod origins;
mod selected_code;
mod selected_json;
pub use selected_code::PortableSelectedCode;
pub use selected_json::{
    SelectedGraphJsonCursor, SelectedGraphJsonError, SelectedGraphJsonProgress,
};
mod storage;
mod validation;
pub use exception::{DetachedDiagnosticFrame, ExceptionCallFrame, ExceptionFrameSource};
pub use graph::{
    GraphActivation, GraphBinding, GraphBindingKind, GraphBoundOwner, GraphCallable,
    GraphCallableOwner, GraphClassDependency, GraphClassMethodDependency, GraphField,
    GraphInvocationRoots, GraphLexicalLayout, GraphNode, GraphNodeRef, GraphProgramUnitId,
    GraphRegistered, GraphRegisteredKind, GraphRequiredSchema,
    GraphSelectedDynamicPropertyMetadata, GraphSelectedObject, GraphSelectedObjectArray,
    GraphSelectedObjectKind, GraphStaticCaller, GraphStaticDependency, GraphStaticReferenceSite,
    PayloadGraph, GRAPH_MAX_SELECTED_UNITS, GRAPH_REQUEST_SCHEMA_V10, GRAPH_REQUEST_SCHEMA_V11,
    GRAPH_REQUEST_SCHEMA_V12, GRAPH_REQUEST_SCHEMA_V13, GRAPH_REQUEST_SCHEMA_V14,
};
pub use limits::ValueLimitBudget;
pub use validation::{ValuePayloadValidationCursor, ValueValidationProgress};
#[cfg(test)]
mod callable_tests;
pub use durable_aggregate::{
    DurableDecompositionPayload, DurableDenseOrthogonalFactor, DurableDynamicProperty,
    DurableEmptyFactorKind, DurableFactorPermutation, DurableNumericFactor, DurableObjectArray,
    DurableOrthogonalFactor, DurablePlainObject, DurableStructArray, DurableStructArrayField,
    DurableStructuralDecomposition,
};

pub use admission::{
    ValueAdmissionOwnership, ValueAdmissionReceipt, ValueAdmissionTransfer,
    ValueConversionFidelity, ValueCopyAccounting, ValueNumericConversion,
    VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION,
};
pub use catalog::{artifact_object_value_id, PublishedObjectCatalog, PublishedObjectSource};

pub const EXTERNALIZED_VALUE_MEDIA_TYPE: &str = "application/vnd.runmat.value+json";
pub const EXTERNALIZED_VALUE_SCHEMA: &str = "runmat-value-payload-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValueLimits {
    pub max_depth: u16,
    pub max_nodes: u64,
    pub max_inline_bytes: u64,
    pub max_elements: u64,
    pub max_fields: u32,
    pub max_text_bytes: u64,
}

impl Default for ValueLimits {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_nodes: 1_000_000,
            max_inline_bytes: 1024 * 1024,
            max_elements: 100_000_000,
            max_fields: 100_000,
            max_text_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "form", content = "value")]
pub enum ValuePayload {
    Inline(Box<InlineValue>),
    Object(Box<ValueRef>),
    Distributed(Box<crate::DistributedValueHandle>),
    Composite(Box<crate::CompositeHandle>),
}

impl ValuePayload {
    pub fn validate(&self, limits: ValueLimits) -> Result<(), ContractError> {
        validation::validate(self, limits)
    }

    /// Validate a complete aggregate under one shared budget. Validating each
    /// member independently would multiply every limit by the member count.
    pub fn validate_all(values: &[Self], limits: ValueLimits) -> Result<(), ContractError> {
        validation::validate_all(values, limits)
    }

    /// Validate one aggregate under the supplied shared limits, then return
    /// every immutable bound-function origin reachable from it. Callers use
    /// these origins to select the exact executable artifacts required to
    /// reconstruct callable values.
    pub fn binding_origins(
        values: &[Self],
        limits: ValueLimits,
    ) -> Result<std::collections::HashSet<crate::FunctionBindingOrigin>, ContractError> {
        Self::validate_all(values, limits)?;
        Ok(origins::binding_origins(values))
    }

    /// Validate portable exception source ordinals against the exact selected
    /// artifact map, including nested cells, structures, and callable captures.
    /// A source-less artifact admits no source-bearing exception frame.
    pub fn validate_for_program_sources(
        &self,
        limits: ValueLimits,
        source_map: Option<&crate::ExecutableSourceMapV2>,
    ) -> Result<(), ContractError> {
        validation::validate_for_program_sources(self, limits, source_map)
    }

    /// V8 graph leaves may carry bounded detached diagnostic presentation.
    /// The selected source map still exclusively authenticates admitted code.
    pub fn validate_for_graph_sources(
        &self,
        limits: ValueLimits,
        source_map: Option<&crate::ExecutableSourceMapV2>,
    ) -> Result<(), ContractError> {
        validation::validate_for_graph_sources(self, limits, source_map)
    }

    /// Durable-storage payloads may contain aggregate tags that
    /// the V2 execution transport deliberately rejects.
    pub fn validate_for_durable_storage(&self, limits: ValueLimits) -> Result<(), ContractError> {
        validation::validate_for_durable_storage(self, limits)
    }

    /// Validate a durable aggregate under one shared budget. This is the
    /// durable counterpart to [`Self::validate_all`]; validating members
    /// independently would multiply each admission limit by the batch size.
    pub fn validate_all_for_durable_storage(
        values: &[Self],
        limits: ValueLimits,
    ) -> Result<(), ContractError> {
        validation::validate_all_for_durable_storage(values, limits)
    }

    pub fn logical_digest(&self) -> Result<Digest, ContractError> {
        identity::logical_digest(self)
    }

    /// Validates both the value encoding and whether references may cross the
    /// requested transport boundary. Portable transport rejects process- or
    /// device-resident references; resident transport requires an exact
    /// worker, node lease, process generation, attempt, and device lease.
    pub fn validate_for_transport(
        &self,
        limits: ValueLimits,
        context: ValueTransportContext<'_>,
    ) -> Result<(), ContractError> {
        validation::validate_for_transport(self, limits, context)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ValueTransportContext<'a> {
    Portable,
    Resident(ResidentValueAuthority<'a>),
}

#[derive(Clone, Copy, Debug)]
pub struct ResidentValueAuthority<'a> {
    pub worker_id: WorkerId,
    pub node_lease_id: NodeLeaseId,
    pub process_generation: u64,
    pub attempt_id: crate::identity::AttemptId,
    pub accelerator_leases: &'a [crate::resource::AcceleratorDeviceLease],
}

impl ResidentValueAuthority<'_> {
    fn authorizes(&self, fence: &ResidentFence) -> bool {
        if self.process_generation == 0
            || fence.worker_id != self.worker_id
            || fence.node_lease_id != self.node_lease_id
            || fence.process_generation != self.process_generation
        {
            return false;
        }
        match &fence.accelerator {
            None => true,
            Some(accelerator) => {
                accelerator.attempt_id == self.attempt_id
                    && self
                        .accelerator_leases
                        .iter()
                        .any(|lease| lease == &accelerator.lease)
            }
        }
    }
}

/// Portable string element: JSON strings are literal text and null is missing.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StringElement(pub Option<String>);

impl StringElement {
    pub fn as_str(&self) -> Option<&str> {
        self.0.as_deref()
    }
}
impl From<String> for StringElement {
    fn from(value: String) -> Self {
        Self(Some(value))
    }
}
impl From<&str> for StringElement {
    fn from(value: &str) -> Self {
        Self(Some(value.to_owned()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum InlineValue {
    Null,
    Logical(bool),
    F64Bits(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    ComplexF64Bits {
        real: u64,
        imaginary: u64,
    },
    String(StringElement),
    Char {
        encoding: CharacterEncoding,
        shape: Vec<u64>,
        code_units: Vec<u16>,
    },
    StringArray {
        shape: Vec<u64>,
        values: Vec<StringElement>,
    },
    Dense(DenseValue),
    Sparse(SparseValue),
    Symbolic(RegisteredData),
    Cell {
        shape: Vec<u64>,
        values: Vec<ValuePayload>,
    },
    Struct(Vec<StructField>),
    DurableOrderedStruct(Vec<StructField>),
    Exception(ExceptionValue),
    Callable(CallableValue),
    ImmutableValueClass(RegisteredData),
    DurablePlainObject(DurablePlainObject),
    DurableStructArray(DurableStructArray),
    DurableObjectArray(DurableObjectArray),
}

/// Versioned physical representation for a portable character payload.
/// There is intentionally no legacy scalar-value variant: pre-1.0 payloads
/// must be regenerated rather than ambiguously changing array geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CharacterEncoding {
    Utf16CodeUnitsV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DenseValue {
    pub element_type: ElementType,
    pub shape: Vec<u64>,
    pub little_endian_data: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SparseValue {
    pub element_type: ElementType,
    pub rows: u64,
    pub columns: u64,
    pub column_offsets: Vec<u64>,
    pub row_indices: Vec<u64>,
    pub little_endian_data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ElementType {
    Logical = 0,
    F32 = 1,
    F64 = 2,
    ComplexF64 = 3,
    I8 = 4,
    I16 = 5,
    I32 = 6,
    I64 = 7,
    U8 = 8,
    U16 = 9,
    U32 = 10,
    U64 = 11,
    ComplexF32 = 12,
    ComplexI8 = 13,
    ComplexI16 = 14,
    ComplexI32 = 15,
    ComplexI64 = 16,
    ComplexU8 = 17,
    ComplexU16 = 18,
    ComplexU32 = 19,
    ComplexU64 = 20,
}

impl ElementType {
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::Logical | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::ComplexI8 | Self::ComplexU8 => 2,
            Self::ComplexF32 | Self::ComplexI32 | Self::ComplexU32 => 8,
            Self::ComplexI16 | Self::ComplexU16 => 4,
            Self::ComplexI64 | Self::ComplexU64 => 16,
            Self::ComplexF64 => 16,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StructField {
    pub name: String,
    pub value: ValuePayload,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RegisteredData {
    pub type_identity: String,
    pub schema_version: u32,
    pub fields: Vec<RegisteredField>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RegisteredField {
    pub name: String,
    pub value: ValuePayload,
}

impl RegisteredData {
    pub(crate) fn validate_identity_parts(
        type_identity: &str,
        schema_version: u32,
    ) -> Result<(), ContractError> {
        if schema_version == 0 || type_identity.trim().is_empty() || type_identity.contains('\0') {
            return Err(ContractError::invalid(
                "registered value",
                "type identity must be nonempty and its schema version nonzero",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExceptionValue {
    pub identifier: String,
    pub message: String,
    pub stack: Vec<String>,
    pub causes: Vec<ExceptionValue>,
    /// Source-less caught runtime errors retain structured metadata across
    /// portable execution transport. Opaque live causes are not serialized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<ExceptionRuntimeMetadata>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExceptionGpuGatherRetry {
    Legacy,
    Never,
    Requested,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExceptionFailureOrigin {
    Unclassified,
    LanguageEvaluation,
    ExecutionControlCancellation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExceptionRuntimeMetadata {
    pub span: Option<(u64, u64)>,
    pub builtin: Option<String>,
    pub task_id: Option<String>,
    pub call_frames: Vec<ExceptionCallFrame>,
    pub call_frames_elided: u64,
    pub call_stack: Vec<String>,
    pub phase: Option<String>,
    pub gpu_gather_retry: ExceptionGpuGatherRetry,
    pub failure_origin: ExceptionFailureOrigin,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CallableValue {
    pub owner: CallableOwner,
    pub qualified_name: String,
    pub callable_digest: Digest,
    /// Required callable identity fields. Historical payloads cannot be
    /// restored with invented function provenance.
    pub form: CallableForm,
    pub source_role: runmat_types::FunctionSourceRole,
    /// Original registered display path, if the callable was created from a file-backed source.
    #[serde(deserialize_with = "deserialize_required_source_file")]
    pub source_file: Option<String>,
    pub captures: Vec<ValuePayload>,
}

/// Portable callable resolution never interprets a destination-local number
/// or display spelling as a semantic function identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "binding")]
pub enum CallableOwner {
    Workspace,
    External,
    Method,
    Bound(crate::FunctionBindingOrigin),
}

// Custom field deserialization makes a missing Option field an error while
// retaining explicit null for callables without file-backed provenance.
fn deserialize_required_source_file<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    Option::<String>::deserialize(deserializer)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallableForm {
    Handle,
    Closure,
}

impl CallableValue {
    pub fn identity_digest(
        owner: CallableOwner,
        qualified_name: &str,
        form: CallableForm,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<&str>,
    ) -> Digest {
        let mut identity = b"runmat-callable-v4\0".to_vec();
        match owner {
            CallableOwner::Workspace => identity.push(0),
            CallableOwner::External => identity.push(1),
            CallableOwner::Method => identity.push(2),
            CallableOwner::Bound(origin) => {
                identity.push(3);
                identity.extend_from_slice(origin.program.bytes());
                identity.extend_from_slice(&origin.function.0.to_be_bytes());
            }
        }
        identity.push(0);
        identity.extend_from_slice(qualified_name.as_bytes());
        identity.push(0);
        identity.push(match form {
            CallableForm::Handle => 0,
            CallableForm::Closure => 1,
        });
        identity.push(source_role as u8);
        match source_file {
            Some(file) => {
                identity.push(1);
                identity.extend_from_slice(&(file.len() as u64).to_be_bytes());
                identity.extend_from_slice(file.as_bytes());
            }
            None => identity.push(0),
        }
        Digest::sha256(identity)
    }

    pub fn validate_identity(&self) -> Result<(), ContractError> {
        Self::validate_identity_parts(
            self.owner,
            &self.qualified_name,
            self.callable_digest,
            self.form,
            self.source_role,
            self.source_file.as_deref(),
            self.captures.len(),
        )
    }

    pub(crate) fn validate_identity_parts(
        owner: CallableOwner,
        qualified_name: &str,
        callable_digest: Digest,
        form: CallableForm,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<&str>,
        capture_count: usize,
    ) -> Result<(), ContractError> {
        if form == CallableForm::Handle && capture_count != 0 {
            return Err(ContractError::invalid(
                "callable value",
                "plain function handles cannot carry closure captures",
            ));
        }
        if qualified_name.trim().is_empty()
            || qualified_name.contains('\0')
            || source_file.is_some_and(|path| path.contains('\0'))
        {
            return Err(ContractError::invalid(
                "callable value",
                "callable name or source path is invalid",
            ));
        }
        if callable_digest
            != Self::identity_digest(owner, qualified_name, form, source_role, source_file)
        {
            return Err(ContractError::invalid(
                "callable value",
                "callable digest does not match its owner and qualified name",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValueRef {
    pub schema_version: u16,
    pub id: ValueId,
    pub logical_digest: Digest,
    pub encoded_length: u64,
    pub media_type: String,
    pub value_schema: String,
    pub encryption_context: Digest,
    pub kind: ValueRefKind,
    pub authorization_scope: String,
    pub resident_fence: Option<ResidentFence>,
}

impl ValueRef {
    /// Validate reference fields through the shared payload validation cursor.
    pub fn validate(&self, limits: ValueLimits) -> Result<(), crate::ContractError> {
        validation::validate_reference(self, limits)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueRefKind {
    DriverObject,
    WorkerObject,
    ProjectObject,
    BroadcastObject,
    SlicedObject,
    ResultObject,
    CheckpointObject,
    ResidentObject,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResidentFence {
    pub worker_id: WorkerId,
    pub node_lease_id: NodeLeaseId,
    pub process_generation: u64,
    pub accelerator: Option<ResidentAcceleratorFence>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidentAcceleratorFence {
    pub attempt_id: crate::identity::AttemptId,
    pub lease: crate::resource::AcceleratorDeviceLease,
}

impl ResidentFence {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.process_generation == 0 {
            return Err(ContractError::invalid(
                "resident value",
                "process generation must be non-zero",
            ));
        }
        if let Some(accelerator) = &self.accelerator {
            accelerator
                .lease
                .validate_for_attempt(accelerator.attempt_id)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod exception_protocol_tests {
    use super::*;

    #[test]
    fn optional_runtime_metadata_changes_logical_identity_without_breaking_legacy_payload() {
        let old_json = serde_json::json!({
            "identifier": "RunMat:old",
            "message": "failure",
            "stack": ["origin"],
            "causes": []
        });
        let old: ExceptionValue = serde_json::from_value(old_json).unwrap();
        assert!(old.runtime.is_none());
        let old_payload = ValuePayload::Inline(Box::new(InlineValue::Exception(old.clone())));
        let old_digest = old_payload.logical_digest().unwrap();

        let mut newer = old;
        newer.runtime = Some(ExceptionRuntimeMetadata {
            span: Some((2, 5)),
            builtin: Some("error".into()),
            task_id: None,
            call_frames: vec![ExceptionCallFrame {
                function: "origin".into(),
                source: ExceptionFrameSource::Admitted(runmat_types::ProgramSourceId(7)),
                span: Some((2, 5)),
            }],
            call_frames_elided: 0,
            call_stack: vec!["origin".into()],
            phase: Some("execution".into()),
            gpu_gather_retry: ExceptionGpuGatherRetry::Never,
            failure_origin: ExceptionFailureOrigin::LanguageEvaluation,
        });
        let newer_payload = ValuePayload::Inline(Box::new(InlineValue::Exception(newer)));
        assert_ne!(old_digest, newer_payload.logical_digest().unwrap());
    }
}
