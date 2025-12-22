//! FFI type definitions for function signatures.

use std::fmt;

/// Supported FFI types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfiType {
    /// 64-bit floating point (double)
    F64,
    /// 32-bit floating point (float)
    F32,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer / size_t
    Usize,
    /// Pointer to another type
    Ptr(Box<FfiType>),
    /// Mutable pointer to another type
    PtrMut(Box<FfiType>),
    /// Void (for return type only)
    Void,
}

impl FfiType {
    /// Parse a type from a string representation.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();

        // Handle pointer types: ptr<T> or *T or *mut T
        if let Some(inner) = s.strip_prefix("ptr<").and_then(|s| s.strip_suffix('>')) {
            return Some(FfiType::Ptr(Box::new(FfiType::parse(inner)?)));
        }
        if let Some(inner) = s.strip_prefix("ptr_mut<").and_then(|s| s.strip_suffix('>')) {
            return Some(FfiType::PtrMut(Box::new(FfiType::parse(inner)?)));
        }
        if let Some(inner) = s.strip_prefix("*mut ") {
            return Some(FfiType::PtrMut(Box::new(FfiType::parse(inner)?)));
        }
        if let Some(inner) = s.strip_prefix('*') {
            return Some(FfiType::Ptr(Box::new(FfiType::parse(inner)?)));
        }

        match s.to_lowercase().as_str() {
            "f64" | "double" => Some(FfiType::F64),
            "f32" | "float" => Some(FfiType::F32),
            "i32" | "int" | "int32" => Some(FfiType::I32),
            "i64" | "int64" => Some(FfiType::I64),
            "u32" | "uint32" => Some(FfiType::U32),
            "usize" | "size_t" => Some(FfiType::Usize),
            "void" | "()" => Some(FfiType::Void),
            _ => None,
        }
    }

    /// Check if this type is a scalar numeric type.
    pub fn is_scalar(&self) -> bool {
        matches!(self, FfiType::F64 | FfiType::F32 | FfiType::I32 | FfiType::I64 | FfiType::U32 | FfiType::Usize)
    }

    /// Check if this type is a pointer type.
    pub fn is_pointer(&self) -> bool {
        matches!(self, FfiType::Ptr(_) | FfiType::PtrMut(_))
    }
}

impl fmt::Display for FfiType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FfiType::F64 => write!(f, "f64"),
            FfiType::F32 => write!(f, "f32"),
            FfiType::I32 => write!(f, "i32"),
            FfiType::I64 => write!(f, "i64"),
            FfiType::U32 => write!(f, "u32"),
            FfiType::Usize => write!(f, "usize"),
            FfiType::Ptr(inner) => write!(f, "ptr<{}>", inner),
            FfiType::PtrMut(inner) => write!(f, "ptr_mut<{}>", inner),
            FfiType::Void => write!(f, "void"),
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

    /// Create a signature for a nullary function: () -> f64
    pub fn scalar_nullary(name: impl Into<String>) -> Self {
        Self::new(name, vec![], FfiType::F64)
    }

    /// Check if all arguments and return type are scalar f64.
    pub fn is_all_f64_scalar(&self) -> bool {
        self.args.iter().all(|t| *t == FfiType::F64) && self.ret == FfiType::F64
    }
}

impl fmt::Display for FfiSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: (", self.name)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", arg)?;
        }
        write!(f, ") -> {}", self.ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_types() {
        assert_eq!(FfiType::parse("f64"), Some(FfiType::F64));
        assert_eq!(FfiType::parse("double"), Some(FfiType::F64));
        assert_eq!(FfiType::parse("i32"), Some(FfiType::I32));
        assert_eq!(FfiType::parse("usize"), Some(FfiType::Usize));
        assert_eq!(FfiType::parse("size_t"), Some(FfiType::Usize));
        assert_eq!(FfiType::parse("void"), Some(FfiType::Void));
    }

    #[test]
    fn test_parse_pointer_types() {
        assert_eq!(
            FfiType::parse("ptr<f64>"),
            Some(FfiType::Ptr(Box::new(FfiType::F64)))
        );
        assert_eq!(
            FfiType::parse("ptr_mut<f64>"),
            Some(FfiType::PtrMut(Box::new(FfiType::F64)))
        );
        assert_eq!(
            FfiType::parse("*f64"),
            Some(FfiType::Ptr(Box::new(FfiType::F64)))
        );
        assert_eq!(
            FfiType::parse("*mut f64"),
            Some(FfiType::PtrMut(Box::new(FfiType::F64)))
        );
    }

    #[test]
    fn test_signature_display() {
        let sig = FfiSignature::scalar_binary("add");
        assert_eq!(sig.to_string(), "add: (f64, f64) -> f64");

        let sig = FfiSignature::scalar_nullary("get_pi");
        assert_eq!(sig.to_string(), "get_pi: () -> f64");
    }
}
