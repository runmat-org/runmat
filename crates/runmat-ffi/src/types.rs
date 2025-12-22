//! FFI type definitions for function signatures.

/// Supported FFI types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiType {
    /// 64-bit floating point (double)
    F64,
    /// 32-bit floating point (float)
    F32,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// Pointer to f64 array with dimensions
    ArrayF64,
    /// Void (for return type only)
    Void,
}

impl FfiType {
    /// Parse a type from a string representation.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f64" | "double" => Some(FfiType::F64),
            "f32" | "float" => Some(FfiType::F32),
            "i32" | "int" | "int32" => Some(FfiType::I32),
            "i64" | "int64" => Some(FfiType::I64),
            "array_f64" | "matrix" => Some(FfiType::ArrayF64),
            "void" => Some(FfiType::Void),
            _ => None,
        }
    }
}

/// A function signature for FFI calls.
#[derive(Debug, Clone)]
pub struct FfiSignature {
    /// Function name in the native library
    pub name: String,
    /// Argument types
    pub args: Vec<FfiType>,
    /// Return type
    pub ret: FfiType,
}

impl FfiSignature {
    /// Create a new FFI signature.
    pub fn new(name: impl Into<String>, args: Vec<FfiType>, ret: FfiType) -> Self {
        Self {
            name: name.into(),
            args,
            ret,
        }
    }

    /// Create a signature for a simple scalar function: (f64, f64) -> f64
    pub fn scalar_binary(name: impl Into<String>) -> Self {
        Self::new(name, vec![FfiType::F64, FfiType::F64], FfiType::F64)
    }

    /// Create a signature for a unary scalar function: f64 -> f64
    pub fn scalar_unary(name: impl Into<String>) -> Self {
        Self::new(name, vec![FfiType::F64], FfiType::F64)
    }
}
