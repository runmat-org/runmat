use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::PythonObjectHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PythonDType {
    Float64,
    Float32,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
    Complex64,
    Complex128,
    DateTime64Micros,
    TimeDelta64Micros,
}

impl PythonDType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::Float64
            | Self::Int64
            | Self::Uint64
            | Self::DateTime64Micros
            | Self::TimeDelta64Micros => 8,
            Self::Float32 | Self::Int32 | Self::Uint32 => 4,
            Self::Int16 | Self::Uint16 => 2,
            Self::Int8 | Self::Uint8 | Self::Bool => 1,
            Self::Complex64 => 8,
            Self::Complex128 => 16,
        }
    }

    pub const fn numpy_typestr(self) -> &'static str {
        match self {
            Self::Float64 => "<f8",
            Self::Float32 => "<f4",
            Self::Int8 => "|i1",
            Self::Int16 => "<i2",
            Self::Int32 => "<i4",
            Self::Int64 => "<i8",
            Self::Uint8 => "|u1",
            Self::Uint16 => "<u2",
            Self::Uint32 => "<u4",
            Self::Uint64 => "<u8",
            Self::Bool => "|b1",
            Self::Complex64 => "<c8",
            Self::Complex128 => "<c16",
            Self::DateTime64Micros => "<M8[us]",
            Self::TimeDelta64Micros => "<m8[us]",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub microsecond: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonTimeDelta {
    pub days: i64,
    pub seconds: u32,
    pub microseconds: u32,
}

pub trait PythonBufferOwner: std::fmt::Debug + Send + Sync {
    fn address(&self) -> usize;
    fn byte_length(&self) -> usize;
    fn copy_bytes(&self) -> Vec<u8>;
}

#[derive(Debug)]
pub(crate) struct OwnedPythonBytes(pub Vec<u8>);

impl PythonBufferOwner for OwnedPythonBytes {
    fn address(&self) -> usize {
        self.0.as_ptr() as usize
    }

    fn byte_length(&self) -> usize {
        self.0.len()
    }

    fn copy_bytes(&self) -> Vec<u8> {
        self.0.clone()
    }
}

#[derive(Clone)]
pub struct PythonArray {
    pub dtype: PythonDType,
    pub shape: Vec<usize>,
    pub column_major: bool,
    pub read_only: bool,
    pub owner: Arc<dyn PythonBufferOwner>,
}

impl PythonArray {
    pub fn from_owned_bytes(
        dtype: PythonDType,
        shape: Vec<usize>,
        column_major: bool,
        read_only: bool,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            dtype,
            shape,
            column_major,
            read_only,
            owner: Arc::new(OwnedPythonBytes(bytes)),
        }
    }
}

impl std::fmt::Debug for PythonArray {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PythonArray")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("column_major", &self.column_major)
            .field("read_only", &self.read_only)
            .field("byte_length", &self.owner.byte_length())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub enum PythonValue {
    None,
    Bool(bool),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
    Complex { real: f64, imaginary: f64 },
    String(String),
    Bytes(Vec<u8>),
    DateTime(PythonDateTime),
    TimeDelta(PythonTimeDelta),
    List(Vec<PythonValue>),
    Tuple(Vec<PythonValue>),
    Dict(Vec<(PythonValue, PythonValue)>),
    Array(PythonArray),
    Object(PythonObjectHandle),
    Callback(u64),
    Keywords(Vec<(String, PythonValue)>),
}
