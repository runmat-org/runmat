use serde::{Deserialize, Serialize};

use crate::identity::{Digest, NodeLeaseId, ValueId, WorkerId};
use crate::ContractError;

mod identity;
mod validation;

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
}

impl ValuePayload {
    pub fn validate(&self, limits: ValueLimits) -> Result<(), ContractError> {
        validation::validate(self, limits)
    }

    pub fn logical_digest(&self) -> Result<Digest, ContractError> {
        identity::logical_digest(self)
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
    String(String),
    Char {
        shape: Vec<u64>,
        code_points: Vec<u32>,
    },
    StringArray {
        shape: Vec<u64>,
        values: Vec<String>,
    },
    Dense(DenseValue),
    Sparse(SparseValue),
    Symbolic(RegisteredData),
    Cell {
        shape: Vec<u64>,
        values: Vec<ValuePayload>,
    },
    Struct(Vec<StructField>),
    OutputList(Vec<ValuePayload>),
    Exception(ExceptionValue),
    Callable(CallableValue),
    ImmutableValueClass(RegisteredData),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
}

impl ElementType {
    const fn byte_width(self) -> u64 {
        match self {
            Self::Logical | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExceptionValue {
    pub identifier: String,
    pub message: String,
    pub stack: Vec<String>,
    pub causes: Vec<ExceptionValue>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CallableValue {
    pub owner_identity: String,
    pub qualified_name: String,
    pub callable_digest: Digest,
    pub captures: Vec<ValuePayload>,
}

impl CallableValue {
    pub fn identity_digest(owner_identity: &str, qualified_name: &str) -> Digest {
        let mut identity = b"runmat-callable-v1\0".to_vec();
        identity.extend_from_slice(owner_identity.as_bytes());
        identity.push(0);
        identity.extend_from_slice(qualified_name.as_bytes());
        Digest::sha256(identity)
    }

    pub fn validate_identity(&self) -> Result<(), ContractError> {
        if self.callable_digest != Self::identity_digest(&self.owner_identity, &self.qualified_name)
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
    pub device_identity: Option<String>,
}
