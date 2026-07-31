pub use inventory;
pub mod symbolic;
use runmat_gc_api::{GcHandle, Trace, Tracer};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::thread::ThreadId;
pub use symbolic::{SymbolicExpr, SymbolicFunction};

use indexmap::IndexMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

#[cfg(target_arch = "wasm32")]
pub mod wasm_registry {
    use super::{BuiltinDoc, BuiltinFunction, Constant};
    use std::cell::{Cell, RefCell};

    thread_local! {
        static FUNCTIONS: RefCell<Vec<&'static BuiltinFunction>> = const { RefCell::new(Vec::new()) };
        static CONSTANTS: RefCell<Vec<&'static Constant>> = const { RefCell::new(Vec::new()) };
        static DOCS: RefCell<Vec<&'static BuiltinDoc>> = const { RefCell::new(Vec::new()) };
        static REGISTERED: Cell<bool> = const { Cell::new(false) };
    }

    fn leak<T>(value: T) -> &'static T {
        Box::leak(Box::new(value))
    }

    pub fn submit_builtin_function(func: BuiltinFunction) {
        let leaked = leak(func);
        FUNCTIONS.with_borrow_mut(|functions| functions.push(leaked));
    }

    pub fn submit_constant(constant: Constant) {
        let leaked = leak(constant);
        CONSTANTS.with_borrow_mut(|constants| constants.push(leaked));
    }

    pub fn submit_builtin_doc(doc: BuiltinDoc) {
        let leaked = leak(doc);
        DOCS.with_borrow_mut(|docs| docs.push(leaked));
    }

    pub fn builtin_functions() -> Vec<&'static BuiltinFunction> {
        FUNCTIONS.with_borrow(Clone::clone)
    }

    pub fn constants() -> Vec<&'static Constant> {
        CONSTANTS.with_borrow(Clone::clone)
    }

    pub fn builtin_docs() -> Vec<&'static BuiltinDoc> {
        DOCS.with_borrow(Clone::clone)
    }

    pub fn mark_registered() {
        REGISTERED.set(true);
    }

    pub fn is_registered() -> bool {
        REGISTERED.get()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(IntValue),
    Num(f64),
    /// Complex scalar value represented as (re, im)
    Complex(f64, f64),
    Bool(bool),
    // Logical array (N-D of booleans). Scalars use Bool.
    LogicalArray(LogicalArray),
    String(String),
    // String array (R2016b+): N-D array of string scalars
    StringArray(StringArray),
    // Char array (single-quoted): 2-D character array (rows x cols)
    CharArray(CharArray),
    Tensor(Tensor),
    /// Real sparse matrix in compressed sparse column form.
    SparseTensor(SparseTensor),
    /// Complex numeric array; same column-major shape semantics as `Tensor`
    ComplexTensor(ComplexTensor),
    /// Scalar symbolic expression used by `sym`, `syms`, and symbolic math builtins.
    Symbolic(SymbolicExpr),
    Cell(CellArray),
    // Struct (scalar or nested). Struct arrays are represented in higher layers;
    // this variant holds a single struct's fields.
    Struct(StructValue),
    // GPU-resident tensor handle (opaque; buffer managed by backend)
    GpuTensor(runmat_accelerate_api::GpuTensorHandle),
    // Simple object instance until full class system lands
    Object(ObjectInstance),
    /// Handle-object wrapper providing identity semantics and validity tracking
    HandleObject(HandleRef),
    /// Event listener handle for events
    Listener(Listener),
    /// Multiple outputs captured as a list (internal destructuring helper)
    OutputList(Vec<Value>),
    // Function handle pointing to a named function (builtin or user)
    FunctionHandle(String),
    // Function handle whose resolution must stay at the external boundary.
    ExternalFunctionHandle(String),
    // Function handle preserving typed method identity.
    MethodFunctionHandle(String),
    // Function handle with compiler/session semantic identity.
    BoundFunctionHandle {
        name: String,
        function: usize,
    },
    Closure(Closure),
    ClassRef(String),
    MException(MException),
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl IntValue {
    pub fn to_i64(&self) -> i64 {
        match self {
            IntValue::I8(v) => *v as i64,
            IntValue::I16(v) => *v as i64,
            IntValue::I32(v) => *v as i64,
            IntValue::I64(v) => *v,
            IntValue::U8(v) => *v as i64,
            IntValue::U16(v) => *v as i64,
            IntValue::U32(v) => *v as i64,
            IntValue::U64(v) => {
                if *v > i64::MAX as u64 {
                    i64::MAX
                } else {
                    *v as i64
                }
            }
        }
    }

    /// Returns the signed representation when it is exactly representable.
    ///
    /// Unlike [`Self::to_i64`], this never saturates an out-of-range `uint64`.
    pub fn try_to_i64(&self) -> Option<i64> {
        match self {
            IntValue::I8(v) => Some(*v as i64),
            IntValue::I16(v) => Some(*v as i64),
            IntValue::I32(v) => Some(*v as i64),
            IntValue::I64(v) => Some(*v),
            IntValue::U8(v) => Some(*v as i64),
            IntValue::U16(v) => Some(*v as i64),
            IntValue::U32(v) => Some(*v as i64),
            IntValue::U64(v) => i64::try_from(*v).ok(),
        }
    }

    /// Returns the `int32` representation when it is exactly representable.
    pub fn try_to_i32(&self) -> Option<i32> {
        self.try_to_i64()
            .and_then(|value| i32::try_from(value).ok())
    }

    /// Returns the platform signed representation when it is exactly
    /// representable.
    ///
    /// This is intended for signed offsets and shifts. In particular, it
    /// rejects `uint64` values above `int64::MAX` instead of saturating them.
    pub fn try_to_isize(&self) -> Option<isize> {
        self.try_to_i64()
            .and_then(|value| isize::try_from(value).ok())
    }

    /// Returns the unsigned representation when it is exactly representable.
    pub fn try_to_u64(&self) -> Option<u64> {
        match self {
            IntValue::I8(v) => u64::try_from(*v).ok(),
            IntValue::I16(v) => u64::try_from(*v).ok(),
            IntValue::I32(v) => u64::try_from(*v).ok(),
            IntValue::I64(v) => u64::try_from(*v).ok(),
            IntValue::U8(v) => Some(*v as u64),
            IntValue::U16(v) => Some(*v as u64),
            IntValue::U32(v) => Some(*v as u64),
            IntValue::U64(v) => Some(*v),
        }
    }

    /// Returns the platform dimension representation when it is exactly
    /// representable and non-negative.
    pub fn try_to_usize(&self) -> Option<usize> {
        self.try_to_u64()
            .and_then(|value| usize::try_from(value).ok())
    }
    pub fn to_f64(&self) -> f64 {
        match self {
            // `uint64` has a wider positive range than `int64`. Converting it
            // through `to_i64` incorrectly clamps every value above i64::MAX
            // before normal IEEE-754 rounding can occur.
            IntValue::U64(value) => *value as f64,
            _ => self.to_i64() as f64,
        }
    }
    pub fn is_zero(&self) -> bool {
        match self {
            IntValue::I8(value) => *value == 0,
            IntValue::I16(value) => *value == 0,
            IntValue::I32(value) => *value == 0,
            IntValue::I64(value) => *value == 0,
            IntValue::U8(value) => *value == 0,
            IntValue::U16(value) => *value == 0,
            IntValue::U32(value) => *value == 0,
            IntValue::U64(value) => *value == 0,
        }
    }
    pub fn class_name(&self) -> &'static str {
        match self {
            IntValue::I8(_) => "int8",
            IntValue::I16(_) => "int16",
            IntValue::I32(_) => "int32",
            IntValue::I64(_) => "int64",
            IntValue::U8(_) => "uint8",
            IntValue::U16(_) => "uint16",
            IntValue::U32(_) => "uint32",
            IntValue::U64(_) => "uint64",
        }
    }

    /// Returns the exact base-10 representation without narrowing through a
    /// signed integer or floating-point compatibility path.
    pub fn decimal_string(&self) -> String {
        match self {
            IntValue::I8(value) => value.to_string(),
            IntValue::I16(value) => value.to_string(),
            IntValue::I32(value) => value.to_string(),
            IntValue::I64(value) => value.to_string(),
            IntValue::U8(value) => value.to_string(),
            IntValue::U16(value) => value.to_string(),
            IntValue::U32(value) => value.to_string(),
            IntValue::U64(value) => value.to_string(),
        }
    }

    /// Add two values of the same MATLAB integer class with saturating
    /// semantics. Sparse triplet construction uses this for duplicate entries.
    pub fn saturating_add(&self, rhs: &Self) -> Result<Self, String> {
        match (self, rhs) {
            (Self::I8(lhs), Self::I8(rhs)) => Ok(Self::I8(lhs.saturating_add(*rhs))),
            (Self::I16(lhs), Self::I16(rhs)) => Ok(Self::I16(lhs.saturating_add(*rhs))),
            (Self::I32(lhs), Self::I32(rhs)) => Ok(Self::I32(lhs.saturating_add(*rhs))),
            (Self::I64(lhs), Self::I64(rhs)) => Ok(Self::I64(lhs.saturating_add(*rhs))),
            (Self::U8(lhs), Self::U8(rhs)) => Ok(Self::U8(lhs.saturating_add(*rhs))),
            (Self::U16(lhs), Self::U16(rhs)) => Ok(Self::U16(lhs.saturating_add(*rhs))),
            (Self::U32(lhs), Self::U32(rhs)) => Ok(Self::U32(lhs.saturating_add(*rhs))),
            (Self::U64(lhs), Self::U64(rhs)) => Ok(Self::U64(lhs.saturating_add(*rhs))),
            (lhs, rhs) => Err(format!(
                "cannot add {} and {} integer values",
                lhs.class_name(),
                rhs.class_name()
            )),
        }
    }
}

#[cfg(test)]
mod int_value_tests {
    use super::{IntValue, Value};

    #[test]
    fn uint64_to_f64_does_not_clamp_through_int64() {
        let value = IntValue::U64(u64::MAX);
        assert_eq!(value.to_f64(), u64::MAX as f64);
        assert!(value.to_f64() > i64::MAX as f64);
    }

    #[test]
    fn decimal_string_preserves_full_signed_and_unsigned_range() {
        assert_eq!(
            IntValue::I64(i64::MIN).decimal_string(),
            "-9223372036854775808"
        );
        assert_eq!(
            IntValue::U64(u64::MAX).decimal_string(),
            "18446744073709551615"
        );
        assert_eq!(
            Value::Int(IntValue::U64(u64::MAX)).to_string(),
            "18446744073709551615"
        );
        assert_eq!(
            String::try_from(&Value::Int(IntValue::U64(u64::MAX))).expect("string conversion"),
            "18446744073709551615"
        );
    }

    #[test]
    fn checked_integer_conversions_do_not_saturate_or_change_sign() {
        assert_eq!(IntValue::I64(i64::MIN).try_to_i64(), Some(i64::MIN));
        assert_eq!(IntValue::U64(i64::MAX as u64).try_to_i64(), Some(i64::MAX));
        assert_eq!(IntValue::U64(u64::MAX).try_to_i64(), None);
        assert_eq!(IntValue::I32(i32::MIN).try_to_i32(), Some(i32::MIN));
        assert_eq!(IntValue::U64(i32::MAX as u64).try_to_i32(), Some(i32::MAX));
        assert_eq!(IntValue::U64(u64::MAX).try_to_i32(), None);
        assert_eq!(IntValue::I64(-1).try_to_u64(), None);
        assert_eq!(IntValue::U64(u64::MAX).try_to_u64(), Some(u64::MAX));
        assert_eq!(
            IntValue::U64(u64::MAX).try_to_usize(),
            usize::try_from(u64::MAX).ok()
        );
        assert_eq!(
            IntValue::I64(isize::MIN as i64).try_to_isize(),
            Some(isize::MIN)
        );
        assert_eq!(IntValue::U64(u64::MAX).try_to_isize(), None);
    }

    #[test]
    fn is_zero_checks_integer_storage_exactly() {
        let zeroes = [
            IntValue::I8(0),
            IntValue::I16(0),
            IntValue::I32(0),
            IntValue::I64(0),
            IntValue::U8(0),
            IntValue::U16(0),
            IntValue::U32(0),
            IntValue::U64(0),
        ];
        for value in zeroes {
            assert!(value.is_zero(), "{value:?} should be zero");
        }

        let nonzeroes = [
            IntValue::I8(-1),
            IntValue::I16(1),
            IntValue::I32(-1),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ];
        for value in nonzeroes {
            assert!(!value.is_zero(), "{value:?} should be nonzero");
        }
    }

    #[test]
    fn bool_conversion_uses_exact_integer_zero_test() {
        assert!(!bool::try_from(&Value::Int(IntValue::U64(0))).expect("zero bool"));
        assert!(bool::try_from(&Value::Int(IntValue::U64(u64::MAX))).expect("max bool"));
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructValue {
    pub fields: IndexMap<String, Value>,
}

impl StructValue {
    pub fn new() -> Self {
        Self {
            fields: IndexMap::new(),
        }
    }

    /// Insert a field, preserving insertion order when the name is new.
    pub fn insert(&mut self, name: impl Into<String>, value: Value) -> Option<Value> {
        self.fields.insert(name.into(), value)
    }

    /// Remove a field while preserving the relative order of remaining fields.
    pub fn remove(&mut self, name: &str) -> Option<Value> {
        self.fields.shift_remove(name)
    }

    /// Returns an iterator over field names in their stored order.
    pub fn field_names(&self) -> impl Iterator<Item = &String> {
        self.fields.keys()
    }
}

impl Default for StructValue {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericDType {
    F64,
    F32,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

/// Exact homogeneous backing storage for MATLAB integer arrays.
///
/// This deliberately stores each class in its native Rust representation so
/// `int64` and `uint64` values never round through `f64` before an
/// integer-aware runtime path consumes them.
#[derive(Debug, Clone, PartialEq)]
pub enum IntegerStorage {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl IntegerStorage {
    /// Returns the logical MATLAB class represented by this exact buffer.
    pub fn numeric_dtype(&self) -> NumericDType {
        match self {
            Self::I8(_) => NumericDType::I8,
            Self::I16(_) => NumericDType::I16,
            Self::I32(_) => NumericDType::I32,
            Self::I64(_) => NumericDType::I64,
            Self::U8(_) => NumericDType::U8,
            Self::U16(_) => NumericDType::U16,
            Self::U32(_) => NumericDType::U32,
            Self::U64(_) => NumericDType::U64,
        }
    }

    /// Construct a one-element buffer preserving the scalar's MATLAB integer
    /// class.
    pub fn from_scalar(value: IntValue) -> Self {
        match value {
            IntValue::I8(value) => Self::I8(vec![value]),
            IntValue::I16(value) => Self::I16(vec![value]),
            IntValue::I32(value) => Self::I32(vec![value]),
            IntValue::I64(value) => Self::I64(vec![value]),
            IntValue::U8(value) => Self::U8(vec![value]),
            IntValue::U16(value) => Self::U16(vec![value]),
            IntValue::U32(value) => Self::U32(vec![value]),
            IntValue::U64(value) => Self::U64(vec![value]),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::I8(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn class_name(&self) -> &'static str {
        match self {
            Self::I8(_) => "int8",
            Self::I16(_) => "int16",
            Self::I32(_) => "int32",
            Self::I64(_) => "int64",
            Self::U8(_) => "uint8",
            Self::U16(_) => "uint16",
            Self::U32(_) => "uint32",
            Self::U64(_) => "uint64",
        }
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Self::I8(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I16(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I32(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I64(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U8(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U16(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U32(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U64(values) => values.iter().map(|&value| value as f64).collect(),
        }
    }

    /// Returns an exact scalar from this homogeneous buffer.
    pub fn value_at(&self, index: usize) -> Option<IntValue> {
        match self {
            Self::I8(values) => values.get(index).copied().map(IntValue::I8),
            Self::I16(values) => values.get(index).copied().map(IntValue::I16),
            Self::I32(values) => values.get(index).copied().map(IntValue::I32),
            Self::I64(values) => values.get(index).copied().map(IntValue::I64),
            Self::U8(values) => values.get(index).copied().map(IntValue::U8),
            Self::U16(values) => values.get(index).copied().map(IntValue::U16),
            Self::U32(values) => values.get(index).copied().map(IntValue::U32),
            Self::U64(values) => values.get(index).copied().map(IntValue::U64),
        }
    }

    /// Returns exact values in storage order.
    pub fn exact_values(&self) -> Vec<IntValue> {
        (0..self.len())
            .map(|index| {
                self.value_at(index)
                    .expect("integer storage index is valid")
            })
            .collect()
    }

    /// Converts an exact integer scalar to this storage class using the
    /// round-and-saturate assignment semantics used by MATLAB integer arrays.
    pub fn cast_exact_assignment(&self, value: &IntValue) -> IntValue {
        match self {
            Self::I8(_) => IntValue::I8(value.to_i64().clamp(i8::MIN as i64, i8::MAX as i64) as i8),
            Self::I16(_) => {
                IntValue::I16(value.to_i64().clamp(i16::MIN as i64, i16::MAX as i64) as i16)
            }
            Self::I32(_) => {
                IntValue::I32(value.to_i64().clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            }
            Self::I64(_) => IntValue::I64(value.to_i64()),
            Self::U8(_) => IntValue::U8(cast_exact_unsigned(value, u8::MAX as u64) as u8),
            Self::U16(_) => IntValue::U16(cast_exact_unsigned(value, u16::MAX as u64) as u16),
            Self::U32(_) => IntValue::U32(cast_exact_unsigned(value, u32::MAX as u64) as u32),
            Self::U64(_) => IntValue::U64(cast_exact_unsigned(value, u64::MAX)),
        }
    }

    /// Converts a floating scalar to this storage class using the
    /// round-and-saturate assignment semantics used by MATLAB integer arrays.
    pub fn cast_f64_assignment(&self, value: f64) -> IntValue {
        match self {
            Self::I8(_) => {
                IntValue::I8(cast_f64_signed(value, i8::MIN as i64, i8::MAX as i64) as i8)
            }
            Self::I16(_) => {
                IntValue::I16(cast_f64_signed(value, i16::MIN as i64, i16::MAX as i64) as i16)
            }
            Self::I32(_) => {
                IntValue::I32(cast_f64_signed(value, i32::MIN as i64, i32::MAX as i64) as i32)
            }
            Self::I64(_) => IntValue::I64(cast_f64_signed(value, i64::MIN, i64::MAX)),
            Self::U8(_) => IntValue::U8(cast_f64_unsigned(value, u8::MAX as u64) as u8),
            Self::U16(_) => IntValue::U16(cast_f64_unsigned(value, u16::MAX as u64) as u16),
            Self::U32(_) => IntValue::U32(cast_f64_unsigned(value, u32::MAX as u64) as u32),
            Self::U64(_) => IntValue::U64(cast_f64_unsigned(value, u64::MAX)),
        }
    }

    /// Rebuilds this homogeneous storage class from exact values.
    pub fn from_exact_values_like(&self, values: Vec<IntValue>) -> Result<Self, String> {
        macro_rules! rebuild {
            ($variant:ident, $value_variant:ident) => {{
                let mut output = Vec::with_capacity(values.len());
                for value in values {
                    let IntValue::$value_variant(value) = value else {
                        return Err("integer storage class mismatch".into());
                    };
                    output.push(value);
                }
                Ok(Self::$variant(output))
            }};
        }
        match self {
            Self::I8(_) => rebuild!(I8, I8),
            Self::I16(_) => rebuild!(I16, I16),
            Self::I32(_) => rebuild!(I32, I32),
            Self::I64(_) => rebuild!(I64, I64),
            Self::U8(_) => rebuild!(U8, U8),
            Self::U16(_) => rebuild!(U16, U16),
            Self::U32(_) => rebuild!(U32, U32),
            Self::U64(_) => rebuild!(U64, U64),
        }
    }

    /// Applies a structural reorder while preserving this exact storage class.
    pub fn reorder(
        &self,
        reorder: impl Fn(&[IntValue]) -> Result<Vec<IntValue>, String>,
    ) -> Result<Self, String> {
        self.from_exact_values_like(reorder(&self.exact_values())?)
    }

    /// Allocates zeros while preserving this integer class.
    pub fn zeros_like(&self, len: usize) -> Self {
        match self {
            Self::I8(_) => Self::I8(vec![0; len]),
            Self::I16(_) => Self::I16(vec![0; len]),
            Self::I32(_) => Self::I32(vec![0; len]),
            Self::I64(_) => Self::I64(vec![0; len]),
            Self::U8(_) => Self::U8(vec![0; len]),
            Self::U16(_) => Self::U16(vec![0; len]),
            Self::U32(_) => Self::U32(vec![0; len]),
            Self::U64(_) => Self::U64(vec![0; len]),
        }
    }

    /// Allocates ones while preserving this integer class.
    pub fn ones_like(&self, len: usize) -> Self {
        match self {
            Self::I8(_) => Self::I8(vec![1; len]),
            Self::I16(_) => Self::I16(vec![1; len]),
            Self::I32(_) => Self::I32(vec![1; len]),
            Self::I64(_) => Self::I64(vec![1; len]),
            Self::U8(_) => Self::U8(vec![1; len]),
            Self::U16(_) => Self::U16(vec![1; len]),
            Self::U32(_) => Self::U32(vec![1; len]),
            Self::U64(_) => Self::U64(vec![1; len]),
        }
    }

    /// Stores a same-class exact scalar without floating-point conversion.
    pub fn set_value(&mut self, index: usize, value: IntValue) -> Result<(), String> {
        match (self, value) {
            (Self::I8(values), IntValue::I8(value)) => set_integer_element(values, index, value),
            (Self::I16(values), IntValue::I16(value)) => set_integer_element(values, index, value),
            (Self::I32(values), IntValue::I32(value)) => set_integer_element(values, index, value),
            (Self::I64(values), IntValue::I64(value)) => set_integer_element(values, index, value),
            (Self::U8(values), IntValue::U8(value)) => set_integer_element(values, index, value),
            (Self::U16(values), IntValue::U16(value)) => set_integer_element(values, index, value),
            (Self::U32(values), IntValue::U32(value)) => set_integer_element(values, index, value),
            (Self::U64(values), IntValue::U64(value)) => set_integer_element(values, index, value),
            (storage, value) => Err(format!(
                "cannot store {} in {} integer storage",
                value.class_name(),
                storage.class_name()
            )),
        }
    }

    /// Converts and stores an exact scalar without materializing a floating
    /// compatibility value or rebuilding the backing buffer.
    pub fn set_exact_assignment(&mut self, index: usize, value: &IntValue) -> Result<(), String> {
        let value = self.cast_exact_assignment(value);
        self.set_value(index, value)
    }

    /// Converts and stores a floating scalar using integer assignment
    /// semantics without rebuilding the backing buffer.
    pub fn set_f64_assignment(&mut self, index: usize, value: f64) -> Result<(), String> {
        let value = self.cast_f64_assignment(value);
        self.set_value(index, value)
    }

    /// Builds storage with this buffer's class from same-class exact values.
    pub fn from_same_class_values(&self, values: Vec<IntValue>) -> Result<Self, String> {
        macro_rules! collect_values {
            ($variant:ident, $type:ty) => {
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::$variant(value) => Ok(value),
                        value => Err(format!(
                            "cannot store {} in {} integer storage",
                            value.class_name(),
                            self.class_name()
                        )),
                    })
                    .collect::<Result<Vec<$type>, String>>()
                    .map(Self::$variant)
            };
        }
        match self {
            Self::I8(_) => collect_values!(I8, i8),
            Self::I16(_) => collect_values!(I16, i16),
            Self::I32(_) => collect_values!(I32, i32),
            Self::I64(_) => collect_values!(I64, i64),
            Self::U8(_) => collect_values!(U8, u8),
            Self::U16(_) => collect_values!(U16, u16),
            Self::U32(_) => collect_values!(U32, u32),
            Self::U64(_) => collect_values!(U64, u64),
        }
    }
}

fn cast_exact_unsigned(value: &IntValue, max: u64) -> u64 {
    match value {
        IntValue::U64(value) => (*value).min(max),
        value => (value.to_i64().max(0) as u64).min(max),
    }
}

fn cast_f64_signed(value: f64, min: i64, max: i64) -> i64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            min
        } else {
            max
        }
    } else {
        value.round().clamp(min as f64, max as f64) as i64
    }
}

fn cast_f64_unsigned(value: f64, max: u64) -> u64 {
    if value.is_nan() || value.is_sign_negative() {
        0
    } else if value.is_infinite() {
        max
    } else {
        value.round().clamp(0.0, max as f64) as u64
    }
}

fn set_integer_element<T>(values: &mut [T], index: usize, value: T) -> Result<(), String> {
    let slot = values
        .get_mut(index)
        .ok_or_else(|| format!("integer storage index {index} is out of bounds"))?;
    *slot = value;
    Ok(())
}

impl NumericDType {
    pub fn class_name(self) -> &'static str {
        match self {
            NumericDType::F64 => "double",
            NumericDType::F32 => "single",
            NumericDType::I8 => "int8",
            NumericDType::I16 => "int16",
            NumericDType::I32 => "int32",
            NumericDType::I64 => "int64",
            NumericDType::U8 => "uint8",
            NumericDType::U16 => "uint16",
            NumericDType::U32 => "uint32",
            NumericDType::U64 => "uint64",
        }
    }

    pub fn byte_size(self) -> usize {
        match self {
            NumericDType::F64 => 8,
            NumericDType::F32 => 4,
            NumericDType::I8 => 1,
            NumericDType::I16 => 2,
            NumericDType::I32 => 4,
            NumericDType::I64 => 8,
            NumericDType::U8 => 1,
            NumericDType::U16 => 2,
            NumericDType::U32 => 4,
            NumericDType::U64 => 8,
        }
    }
}

#[cfg(test)]
mod integer_storage_tests {
    use super::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, NumericDType, Tensor,
    };

    #[test]
    fn uint64_tensor_keeps_exact_backing_values() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2])
            .expect("integer tensor");

        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![0, u64::MAX]))
        );
        assert_eq!(tensor.data[1], u64::MAX as f64);
    }

    #[test]
    fn integer_tensor_reports_its_exact_matlab_dtype() {
        let cases = [
            (IntegerStorage::I8(vec![0]), NumericDType::I8),
            (IntegerStorage::I16(vec![0]), NumericDType::I16),
            (IntegerStorage::I32(vec![0]), NumericDType::I32),
            (IntegerStorage::I64(vec![0]), NumericDType::I64),
            (IntegerStorage::U8(vec![0]), NumericDType::U8),
            (IntegerStorage::U16(vec![0]), NumericDType::U16),
            (IntegerStorage::U32(vec![0]), NumericDType::U32),
            (IntegerStorage::U64(vec![0]), NumericDType::U64),
        ];

        for (storage, dtype) in cases {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
            assert_eq!(tensor.dtype, dtype);
            assert_eq!(
                tensor.dtype.class_name(),
                tensor.integer_storage().unwrap().class_name()
            );
        }
    }

    #[test]
    fn single_constructor_preserves_f32_values_and_dtype_across_storage_migration() {
        let source = vec![f32::MIN_POSITIVE, 1.0 / 10.0, f32::MAX];
        let expected: Vec<f64> = source.iter().map(|&value| f64::from(value)).collect();
        let tensor = Tensor::from_f32(source, vec![1, 3]).expect("single tensor");

        assert_eq!(tensor.dtype, NumericDType::F32);
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.data, expected);
        assert!(tensor.integer_storage().is_none());
    }

    #[test]
    fn typed_constructor_materializes_exact_integer_storage() {
        let tensor =
            Tensor::new_with_dtype(vec![-2.2, 12.8, 99_999.0], vec![1, 3], NumericDType::I16)
                .expect("typed tensor");

        assert_eq!(tensor.dtype, NumericDType::I16);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::I16(vec![-2, 13, i16::MAX]))
        );
    }

    #[test]
    fn integer_tensor_supports_empty_typed_arrays() {
        let tensor = Tensor::new_integer(IntegerStorage::I64(Vec::new()), vec![0, 1])
            .expect("empty integer tensor");

        assert_eq!(
            tensor.integer_storage().map(IntegerStorage::class_name),
            Some("int64")
        );
        assert!(tensor.data.is_empty());
    }

    #[test]
    fn assignment_conversion_preserves_wide_exact_values_and_mutates_in_place() {
        let large = 9_007_199_254_740_993_u64;
        let mut storage = IntegerStorage::U64(vec![0, 1]);

        storage
            .set_exact_assignment(0, &IntValue::U64(large))
            .unwrap();
        storage.set_f64_assignment(1, -4.2).unwrap();

        assert_eq!(storage, IntegerStorage::U64(vec![large, 0]));
        assert_eq!(
            IntegerStorage::I8(vec![0]).cast_f64_assignment(200.6),
            IntValue::I8(i8::MAX)
        );
    }

    #[test]
    fn tensor_get2_reads_every_integer_class_from_exact_storage() {
        let cases = [
            IntegerStorage::I8(vec![-8, 7]),
            IntegerStorage::I16(vec![-16, 15]),
            IntegerStorage::I32(vec![-32, 31]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![8, u8::MAX]),
            IntegerStorage::U16(vec![16, u16::MAX]),
            IntegerStorage::U32(vec![32, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in cases {
            let expected = storage.to_f64_vec();
            let mut tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
            tensor.data.clear();

            assert_eq!(tensor.get2(0, 0), Ok(expected[0]));
            assert_eq!(tensor.get2(0, 1), Ok(expected[1]));
        }
    }

    #[test]
    fn tensor_set2_updates_exact_integer_storage_and_repairs_poisoned_mirror() {
        let cases = [
            IntegerStorage::I8(vec![0]),
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I32(vec![0]),
            IntegerStorage::I64(vec![0]),
            IntegerStorage::U8(vec![0]),
            IntegerStorage::U16(vec![0]),
            IntegerStorage::U32(vec![0]),
            IntegerStorage::U64(vec![0]),
        ];

        for storage in cases {
            let expected = storage.cast_f64_assignment(-2.6);
            let mut tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
            tensor.data.fill(f64::NAN);

            tensor.set2(0, 0, -2.6).expect("integer assignment");

            assert_eq!(
                tensor
                    .integer_storage()
                    .and_then(|storage| storage.value_at(0)),
                Some(expected)
            );
            assert_eq!(tensor.data, tensor.integer_storage().unwrap().to_f64_vec());
        }
    }

    #[test]
    fn integer_tensor_rejects_shape_length_mismatches() {
        let err = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![3, 1])
            .expect_err("shape mismatch");
        assert!(err.contains("doesn't match shape"));
    }

    #[test]
    fn reshape_preserves_exact_integer_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![-1, i64::MAX]), vec![1, 2])
            .expect("integer tensor")
            .reshape(vec![2, 1])
            .expect("reshape");

        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::I64(vec![-1, i64::MAX]))
        );
    }

    #[test]
    fn reshape_and_display_use_integer_storage_when_mirror_is_missing() {
        let mut tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2],
        )
        .expect("integer tensor");
        tensor.data.clear();

        let tensor = tensor.reshape(vec![1, 2]).expect("reshape");
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
        );

        let mut vector = tensor.clone();
        vector.shape = vec![2];
        vector.rows = 1;
        vector.cols = 2;
        assert_eq!(
            vector.to_string(),
            "[9007199254740993 18446744073709551615]"
        );
    }

    #[test]
    fn integer_complex_storage_preserves_paired_uint64_values() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_809]),
        )
        .expect("matching uint64 components");
        let tensor = ComplexTensor::new_integer(storage.clone(), vec![1, 2])
            .expect("integer complex tensor");

        assert_eq!(tensor.integer_data, Some(storage));
        assert_eq!(
            tensor
                .integer_data
                .as_ref()
                .map(IntegerComplexStorage::class_name),
            Some("uint64")
        );
    }

    #[test]
    fn integer_complex_display_uses_storage_when_mirror_is_missing() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
        )
        .expect("matching storage");
        let mut tensor = ComplexTensor::new_integer(storage, vec![2]).expect("complex tensor");
        tensor.data.clear();

        assert_eq!(
            tensor.to_string(),
            "[9007199254740993+18446744073709551615i 18446744073709551615+9007199254740993i]"
        );
    }

    #[test]
    fn integer_complex_storage_rejects_mismatched_components() {
        let class_mismatch =
            IntegerComplexStorage::new(IntegerStorage::I64(vec![1]), IntegerStorage::U64(vec![1]))
                .expect_err("integer classes must match");
        assert!(class_mismatch.contains("matching class"));

        let length_mismatch = IntegerComplexStorage::new(
            IntegerStorage::I64(vec![1]),
            IntegerStorage::I64(vec![1, 2]),
        )
        .expect_err("component lengths must match");
        assert!(length_mismatch.contains("matching class and length"));
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f64>,
    /// Exact homogeneous integer backing storage during the integer-dtype
    /// migration. Floating tensors leave this unset.
    pub integer_data: Option<IntegerStorage>,
    pub shape: Vec<usize>, // Column-major layout
    pub rows: usize,       // Compatibility for 2D usage
    pub cols: usize,       // Compatibility for 2D usage
    /// Logical numeric class of this tensor; host storage remains f64.
    pub dtype: NumericDType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensor {
    pub rows: usize,
    pub cols: usize,
    /// Column pointers into `row_indices`/`values`; length is `cols + 1`.
    pub col_ptrs: Vec<usize>,
    /// Zero-based row indices, sorted within each column.
    pub row_indices: Vec<usize>,
    /// Floating compatibility view for legacy sparse consumers.
    pub values: Vec<f64>,
    /// Exact homogeneous backing storage for typed integer sparse values.
    pub integer_data: Option<IntegerStorage>,
}

type SparseCscParts<T> = (Vec<usize>, Vec<usize>, Vec<T>);

#[derive(Debug, Clone, PartialEq)]
pub struct ComplexTensor {
    pub data: Vec<(f64, f64)>,
    pub integer_data: Option<IntegerComplexStorage>,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntegerComplexStorage {
    pub real: IntegerStorage,
    pub imag: IntegerStorage,
}

impl IntegerComplexStorage {
    pub fn new(real: IntegerStorage, imag: IntegerStorage) -> Result<Self, String> {
        if real.class_name() != imag.class_name() || real.len() != imag.len() {
            return Err("complex integer components must have matching class and length".into());
        }
        Ok(Self { real, imag })
    }

    pub fn len(&self) -> usize {
        self.real.len()
    }

    pub fn is_empty(&self) -> bool {
        self.real.is_empty()
    }

    pub fn class_name(&self) -> &'static str {
        self.real.class_name()
    }

    /// Tests a paired complex integer element without consulting its lossy
    /// floating compatibility representation.
    pub fn is_nonzero_at(&self, index: usize) -> Option<bool> {
        let real = self.real.value_at(index)?;
        let imag = self.imag.value_at(index)?;
        Some(!real.is_zero() || !imag.is_zero())
    }

    /// Applies the same structural reorder independently to both exact components.
    pub fn reorder(
        &self,
        reorder: impl Fn(&[IntValue]) -> Result<Vec<IntValue>, String>,
    ) -> Result<Self, String> {
        let real = self
            .real
            .from_exact_values_like(reorder(&self.real.exact_values())?)?;
        let imag = self
            .imag
            .from_exact_values_like(reorder(&self.imag.exact_values())?)?;
        Self::new(real, imag)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StringArray {
    pub data: Vec<String>,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalArray {
    pub data: Vec<u8>, // 0 or 1 values; compact bitset can come later
    pub shape: Vec<usize>,
}

impl LogicalArray {
    pub fn new(data: Vec<u8>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "LogicalArray data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        // Normalize to 0/1
        let mut d = data;
        for v in &mut d {
            *v = if *v != 0 { 1 } else { 0 };
        }
        Ok(LogicalArray { data: d, shape })
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        LogicalArray {
            data: vec![0u8; expected],
            shape,
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CharArray {
    pub data: Vec<char>,
    pub rows: usize,
    pub cols: usize,
}

impl CharArray {
    pub fn new_row(s: &str) -> Self {
        CharArray {
            data: s.chars().collect(),
            rows: 1,
            cols: s.chars().count(),
        }
    }
    pub fn new(data: Vec<char>, rows: usize, cols: usize) -> Result<Self, String> {
        if rows * cols != data.len() {
            return Err(format!(
                "Char data length {} doesn't match dimensions {}x{}",
                data.len(),
                rows,
                cols
            ));
        }
        Ok(CharArray { data, rows, cols })
    }
}

impl StringArray {
    pub fn new(data: Vec<String>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "StringArray data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        Ok(StringArray {
            data,
            shape,
            rows,
            cols,
        })
    }
    pub fn new_2d(data: Vec<String>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }
    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }
}

// GpuTensorHandle now lives in runmat-accel-api

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "Tensor data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        Ok(Tensor {
            data,
            integer_data: None,
            shape,
            rows,
            cols,
            dtype: NumericDType::F64,
        })
    }

    pub fn new_2d(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }

    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        let converted: Vec<f64> = data.into_iter().map(|v| v as f64).collect();
        Self::new_with_dtype(converted, shape, NumericDType::F32)
    }

    pub fn from_f32_slice(data: &[f32], shape: &[usize]) -> Result<Self, String> {
        let converted: Vec<f64> = data.iter().map(|&v| v as f64).collect();
        Self::new_with_dtype(converted, shape.to_vec(), NumericDType::F32)
    }

    pub fn new_with_dtype(
        data: Vec<f64>,
        shape: Vec<usize>,
        dtype: NumericDType,
    ) -> Result<Self, String> {
        if let Some(prototype) = integer_storage_prototype(dtype) {
            let values = data
                .into_iter()
                .map(|value| prototype.cast_f64_assignment(value))
                .collect();
            return Self::new_integer(prototype.from_same_class_values(values)?, shape);
        }
        let mut t = Self::new(data, shape)?;
        t.dtype = dtype;
        Ok(t)
    }

    /// Construct a tensor backed by an exact homogeneous integer buffer.
    ///
    /// The floating `data` member is retained only as a compatibility view for
    /// legacy numeric consumers. Integer-aware code must use `integer_data`.
    pub fn new_integer(storage: IntegerStorage, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if storage.len() != expected {
            return Err(format!(
                "integer tensor data length {} doesn't match shape {:?} ({} elements)",
                storage.len(),
                shape,
                expected
            ));
        }

        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        let dtype = storage.numeric_dtype();
        Ok(Tensor {
            data: storage.to_f64_vec(),
            integer_data: Some(storage),
            shape,
            rows,
            cols,
            dtype,
        })
    }

    pub fn integer_storage(&self) -> Option<&IntegerStorage> {
        self.integer_data.as_ref()
    }

    /// Change only shape metadata while retaining the underlying numeric storage.
    pub fn reshape(mut self, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if let Some(storage) = &self.integer_data {
            if storage.len() != expected {
                return Err(format!(
                    "integer tensor data length {} doesn't match shape {:?} ({} elements)",
                    storage.len(),
                    shape,
                    expected
                ));
            }
        } else if self.data.len() != expected {
            return Err(format!(
                "Tensor data length {} doesn't match shape {:?} ({} elements)",
                self.data.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        self.shape = shape;
        self.rows = rows;
        self.cols = cols;
        Ok(self)
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        Tensor {
            data: vec![0.0; size],
            integer_data: None,
            shape,
            rows,
            cols,
            dtype: NumericDType::F64,
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        Tensor {
            data: vec![1.0; size],
            integer_data: None,
            shape,
            rows,
            cols,
            dtype: NumericDType::F64,
        }
    }

    // 2D helpers for transitional call sites
    pub fn zeros2(rows: usize, cols: usize) -> Self {
        Self::zeros(vec![rows, cols])
    }
    pub fn ones2(rows: usize, cols: usize) -> Self {
        Self::ones(vec![rows, cols])
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }
    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }

    pub fn get2(&self, row: usize, col: usize) -> Result<f64, String> {
        let rows = self.rows();
        let cols = self.cols();
        if row >= rows || col >= cols {
            return Err(format!(
                "Index ({row}, {col}) out of bounds for {rows}x{cols} tensor"
            ));
        }
        // Column-major linearization: lin = row + col*rows
        let index = row + col * rows;
        // This legacy API deliberately returns f64, but typed tensors must
        // source that conversion from their authoritative integer buffer.
        // The f64 mirror can be lossy (or absent in migration tests).
        Ok(self
            .integer_data
            .as_ref()
            .and_then(|storage| storage.value_at(index))
            .map_or_else(|| self.data[index], |value| value.to_f64()))
    }

    pub fn set2(&mut self, row: usize, col: usize, value: f64) -> Result<(), String> {
        let rows = self.rows();
        let cols = self.cols();
        if row >= rows || col >= cols {
            return Err(format!(
                "Index ({row}, {col}) out of bounds for {rows}x{cols} tensor"
            ));
        }
        // Column-major linearization
        let index = row + col * rows;
        if let Some(storage) = &mut self.integer_data {
            // Assignment to a MATLAB integer array rounds and saturates to
            // its existing class. Rebuild the compatibility view from the
            // exact result so it cannot become authoritative by accident.
            storage.set_f64_assignment(index, value)?;
            self.data = storage.to_f64_vec();
        } else {
            self.data[index] = value;
        }
        Ok(())
    }

    pub fn scalar_to_tensor2(scalar: f64, rows: usize, cols: usize) -> Tensor {
        Tensor {
            data: vec![scalar; rows * cols],
            integer_data: None,
            shape: vec![rows, cols],
            rows,
            cols,
            dtype: NumericDType::F64,
        }
    }
    // No-compat constructors: prefer new/new_2d/zeros/zeros2/ones/ones2
}

fn integer_storage_prototype(dtype: NumericDType) -> Option<IntegerStorage> {
    match dtype {
        NumericDType::I8 => Some(IntegerStorage::I8(Vec::new())),
        NumericDType::I16 => Some(IntegerStorage::I16(Vec::new())),
        NumericDType::I32 => Some(IntegerStorage::I32(Vec::new())),
        NumericDType::I64 => Some(IntegerStorage::I64(Vec::new())),
        NumericDType::U8 => Some(IntegerStorage::U8(Vec::new())),
        NumericDType::U16 => Some(IntegerStorage::U16(Vec::new())),
        NumericDType::U32 => Some(IntegerStorage::U32(Vec::new())),
        NumericDType::U64 => Some(IntegerStorage::U64(Vec::new())),
        NumericDType::F64 | NumericDType::F32 => None,
    }
}

impl SparseTensor {
    pub fn new(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, String> {
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, values.len())?;
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
            integer_data: None,
        })
    }

    /// Constructs a sparse matrix backed by an exact integer value buffer.
    pub fn new_integer(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        integer_data: IntegerStorage,
    ) -> Result<Self, String> {
        Self::validate_structure(rows, cols, &col_ptrs, &row_indices, integer_data.len())?;
        let values = integer_data.to_f64_vec();
        Ok(Self {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
            integer_data: Some(integer_data),
        })
    }

    fn validate_structure(
        rows: usize,
        cols: usize,
        col_ptrs: &[usize],
        row_indices: &[usize],
        values_len: usize,
    ) -> Result<(), String> {
        if col_ptrs.len() != cols.saturating_add(1) {
            return Err(format!(
                "SparseTensor col_ptrs length {} doesn't match cols {}",
                col_ptrs.len(),
                cols
            ));
        }
        if row_indices.len() != values_len {
            return Err(format!(
                "SparseTensor row index length {} doesn't match value length {}",
                row_indices.len(),
                values_len
            ));
        }
        if col_ptrs.first().copied().unwrap_or(usize::MAX) != 0 {
            return Err("SparseTensor col_ptrs must start at 0".to_string());
        }
        if col_ptrs.last().copied().unwrap_or(usize::MAX) != values_len {
            return Err("SparseTensor final col_ptr must equal nnz".to_string());
        }
        for window in col_ptrs.windows(2) {
            if window[0] > window[1] {
                return Err("SparseTensor col_ptrs must be nondecreasing".to_string());
            }
        }
        for col in 0..cols {
            let start = col_ptrs[col];
            let end = col_ptrs[col + 1];
            let mut prev: Option<usize> = None;
            for &row in &row_indices[start..end] {
                if row >= rows {
                    return Err(format!("SparseTensor row index {row} exceeds rows {rows}"));
                }
                if prev.is_some_and(|p| p >= row) {
                    return Err("SparseTensor row indices must be sorted and unique".to_string());
                }
                prev = Some(row);
            }
        }
        Ok(())
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)],
            row_indices: Vec::new(),
            values: Vec::new(),
            integer_data: None,
        }
    }

    /// Creates an all-zero sparse matrix retaining an exact integer class.
    pub fn zeros_with_integer_storage(rows: usize, cols: usize, storage: &IntegerStorage) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)],
            row_indices: Vec::new(),
            values: Vec::new(),
            integer_data: Some(storage.zeros_like(0)),
        }
    }

    /// Creates a typed sparse matrix using the class of `prototype`.
    pub fn new_integer_like(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<IntValue>,
        prototype: &IntegerStorage,
    ) -> Result<Self, String> {
        Self::new_integer(
            rows,
            cols,
            col_ptrs,
            row_indices,
            prototype.from_same_class_values(values)?,
        )
    }

    pub fn nnz(&self) -> usize {
        self.integer_data
            .as_ref()
            .map_or_else(|| self.values.len(), IntegerStorage::len)
    }

    pub fn shape(&self) -> Vec<usize> {
        vec![self.rows, self.cols]
    }

    pub fn to_dense(&self) -> Result<Tensor, String> {
        let len = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| "SparseTensor dense dimensions overflow usize".to_string())?;
        if let Some(integer_data) = &self.integer_data {
            let mut data = integer_data.zeros_like(len);
            for col in 0..self.cols {
                for idx in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                    let row = self.row_indices[idx];
                    let value = integer_data.value_at(idx).ok_or_else(|| {
                        "SparseTensor integer storage is inconsistent".to_string()
                    })?;
                    data.set_value(row + col * self.rows, value)?;
                }
            }
            return Tensor::new_integer(data, self.shape());
        }
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|err| format!("SparseTensor dense allocation failed: {err}"))?;
        data.resize(len, 0.0);
        for col in 0..self.cols {
            for idx in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                let row = self.row_indices[idx];
                data[row + col * self.rows] = self.values[idx];
            }
        }
        Tensor::new(data, self.shape())
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        self.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .map(|offset| {
                let index = start + offset;
                self.integer_data
                    .as_ref()
                    .and_then(|storage| storage.value_at(index))
                    // `get` is a legacy f64 API; typed sparse consumers should
                    // use `integer_at` when exact wide integer values matter.
                    .map_or_else(|| self.values[index], |value| value.to_f64())
            })
    }

    /// Returns an exact stored integer value when this sparse matrix is typed.
    pub fn integer_at(&self, row: usize, col: usize) -> Option<IntValue> {
        let integer_data = self.integer_data.as_ref()?;
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        self.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .and_then(|offset| integer_data.value_at(start + offset))
    }

    pub fn integer_storage(&self) -> Option<&IntegerStorage> {
        self.integer_data.as_ref()
    }

    fn merged_linear_updates<T: Clone>(
        &self,
        updates: &[(usize, T)],
        mut stored_value: impl FnMut(usize) -> Result<T, String>,
        is_zero: impl Fn(&T) -> bool,
    ) -> Result<SparseCscParts<T>, String> {
        let total = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| "SparseTensor assignment dimensions overflow usize".to_string())?;
        let mut latest = BTreeMap::new();
        for (index, value) in updates {
            if *index >= total {
                return Err(format!(
                    "SparseTensor assignment linear index {} exceeds {} elements",
                    index, total
                ));
            }
            latest.insert(*index, value.clone());
        }

        let capacity = self
            .nnz()
            .checked_add(latest.len())
            .ok_or_else(|| "SparseTensor assignment nnz overflow".to_string())?;
        let mut col_ptrs = Vec::with_capacity(self.cols.saturating_add(1));
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        row_indices
            .try_reserve_exact(capacity)
            .map_err(|error| format!("SparseTensor assignment allocation failed: {error}"))?;
        values
            .try_reserve_exact(capacity)
            .map_err(|error| format!("SparseTensor assignment allocation failed: {error}"))?;
        col_ptrs.push(0);

        for col in 0..self.cols {
            let column_start = col * self.rows;
            let column_end = column_start + self.rows;
            let mut stored = self.col_ptrs[col];
            let stored_end = self.col_ptrs[col + 1];
            for (&linear, value) in latest.range(column_start..column_end) {
                let row = linear - column_start;
                while stored < stored_end && self.row_indices[stored] < row {
                    row_indices.push(self.row_indices[stored]);
                    values.push(stored_value(stored)?);
                    stored += 1;
                }
                if stored < stored_end && self.row_indices[stored] == row {
                    stored += 1;
                }
                if !is_zero(value) {
                    row_indices.push(row);
                    values.push(value.clone());
                }
            }
            while stored < stored_end {
                row_indices.push(self.row_indices[stored]);
                values.push(stored_value(stored)?);
                stored += 1;
            }
            col_ptrs.push(values.len());
        }
        Ok((col_ptrs, row_indices, values))
    }

    /// Applies floating updates in one CSC merge. Repeated indices use the
    /// final assignment value and zeros are elided without densifying.
    pub fn with_updated_linear_values(&self, updates: &[(usize, f64)]) -> Result<Self, String> {
        if self.integer_data.is_some() {
            return Err("cannot assign floating sparse value to typed integer storage".to_string());
        }
        let (col_ptrs, row_indices, values) = self.merged_linear_updates(
            updates,
            |index| Ok(self.values[index]),
            |value| *value == 0.0,
        )?;
        Self::new(self.rows, self.cols, col_ptrs, row_indices, values)
    }

    /// Applies exact integer updates in one CSC merge. Values must already be
    /// in this sparse matrix's class; coercion belongs to the VM layer.
    pub fn with_updated_integer_linear_values(
        &self,
        updates: &[(usize, IntValue)],
    ) -> Result<Self, String> {
        let storage = self
            .integer_data
            .as_ref()
            .ok_or_else(|| "cannot assign integer sparse value to floating storage".to_string())?;
        let (col_ptrs, row_indices, values) = self.merged_linear_updates(
            updates,
            |index| {
                storage
                    .value_at(index)
                    .ok_or_else(|| "SparseTensor integer storage is inconsistent".to_string())
            },
            IntValue::is_zero,
        )?;
        Self::new_integer_like(self.rows, self.cols, col_ptrs, row_indices, values, storage)
    }

    pub fn with_updated_value(&self, row: usize, col: usize, value: f64) -> Result<Self, String> {
        let index = self.checked_assignment_linear_index(row, col)?;
        self.with_updated_linear_values(&[(index, value)])
    }

    pub fn with_updated_integer_value(
        &self,
        row: usize,
        col: usize,
        value: IntValue,
    ) -> Result<Self, String> {
        let index = self.checked_assignment_linear_index(row, col)?;
        self.with_updated_integer_linear_values(&[(index, value)])
    }

    /// Expands sparse dimensions without materializing implicit zero entries.
    pub fn with_expanded_shape(&self, rows: usize, cols: usize) -> Result<Self, String> {
        if rows < self.rows || cols < self.cols {
            return Err(format!(
                "SparseTensor cannot shrink shape ({}, {}) to ({rows}, {cols})",
                self.rows, self.cols
            ));
        }
        let mut col_ptrs = self.col_ptrs.clone();
        col_ptrs.resize(
            cols.checked_add(1)
                .ok_or_else(|| "SparseTensor expanded column count overflow".to_string())?,
            self.nnz(),
        );
        if let Some(storage) = self.integer_storage() {
            return Self::new_integer(
                rows,
                cols,
                col_ptrs,
                self.row_indices.clone(),
                storage.clone(),
            );
        }
        Self::new(
            rows,
            cols,
            col_ptrs,
            self.row_indices.clone(),
            self.values.clone(),
        )
    }

    fn checked_assignment_linear_index(&self, row: usize, col: usize) -> Result<usize, String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!(
                "SparseTensor assignment index ({}, {}) exceeds shape ({}, {})",
                row, col, self.rows, self.cols
            ));
        }
        col.checked_mul(self.rows)
            .and_then(|base| base.checked_add(row))
            .ok_or_else(|| "SparseTensor assignment linear index overflow".to_string())
    }

    fn checked_deletion_indices(
        indices: &[usize],
        bound: usize,
        axis: &str,
    ) -> Result<Vec<usize>, String> {
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            if pair[0] == pair[1] {
                return Err(format!(
                    "SparseTensor {axis} deletion indices must be unique"
                ));
            }
        }
        if sorted.iter().any(|&index| index >= bound) {
            return Err(format!(
                "SparseTensor {axis} deletion index exceeds dimension"
            ));
        }
        Ok(sorted)
    }

    fn rebuilt_csc<T: Clone>(
        &self,
        source_columns: &[usize],
        mut map_row: impl FnMut(usize) -> Option<usize>,
        mut stored_value: impl FnMut(usize) -> Result<T, String>,
    ) -> Result<SparseCscParts<T>, String> {
        let mut col_ptrs = Vec::new();
        col_ptrs
            .try_reserve_exact(
                source_columns
                    .len()
                    .checked_add(1)
                    .ok_or_else(|| "SparseTensor deletion column count overflow".to_string())?,
            )
            .map_err(|error| format!("SparseTensor deletion allocation failed: {error}"))?;
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        row_indices
            .try_reserve_exact(self.nnz())
            .map_err(|error| format!("SparseTensor deletion allocation failed: {error}"))?;
        values
            .try_reserve_exact(self.nnz())
            .map_err(|error| format!("SparseTensor deletion allocation failed: {error}"))?;
        col_ptrs.push(0);
        for &source_column in source_columns {
            let start = self.col_ptrs[source_column];
            let end = self.col_ptrs[source_column + 1];
            for index in start..end {
                if let Some(row) = map_row(self.row_indices[index]) {
                    row_indices.push(row);
                    values.push(stored_value(index)?);
                }
            }
            col_ptrs.push(values.len());
        }
        Ok((col_ptrs, row_indices, values))
    }

    /// Deletes complete sparse matrix rows without materializing dense storage.
    pub fn with_deleted_rows(&self, rows: &[usize]) -> Result<Self, String> {
        let rows = Self::checked_deletion_indices(rows, self.rows, "row")?;
        let source_columns = (0..self.cols).collect::<Vec<_>>();
        let output_rows = self
            .rows
            .checked_sub(rows.len())
            .ok_or_else(|| "SparseTensor deletion row count underflow".to_string())?;
        if let Some(storage) = self.integer_storage() {
            let (col_ptrs, row_indices, values) = self.rebuilt_csc(
                &source_columns,
                |row| match rows.binary_search(&row) {
                    Ok(_) => None,
                    Err(removed_before) => Some(row - removed_before),
                },
                |index| {
                    storage
                        .value_at(index)
                        .ok_or_else(|| "SparseTensor integer storage is inconsistent".to_string())
                },
            )?;
            return Self::new_integer_like(
                output_rows,
                self.cols,
                col_ptrs,
                row_indices,
                values,
                storage,
            );
        }
        let (col_ptrs, row_indices, values) = self.rebuilt_csc(
            &source_columns,
            |row| match rows.binary_search(&row) {
                Ok(_) => None,
                Err(removed_before) => Some(row - removed_before),
            },
            |index| Ok(self.values[index]),
        )?;
        Self::new(output_rows, self.cols, col_ptrs, row_indices, values)
    }

    /// Deletes complete sparse matrix columns without materializing dense storage.
    pub fn with_deleted_columns(&self, columns: &[usize]) -> Result<Self, String> {
        let columns = Self::checked_deletion_indices(columns, self.cols, "column")?;
        let source_columns = (0..self.cols)
            .filter(|column| columns.binary_search(column).is_err())
            .collect::<Vec<_>>();
        if let Some(storage) = self.integer_storage() {
            let (col_ptrs, row_indices, values) =
                self.rebuilt_csc(&source_columns, Some, |index| {
                    storage
                        .value_at(index)
                        .ok_or_else(|| "SparseTensor integer storage is inconsistent".to_string())
                })?;
            return Self::new_integer_like(
                self.rows,
                source_columns.len(),
                col_ptrs,
                row_indices,
                values,
                storage,
            );
        }
        let (col_ptrs, row_indices, values) =
            self.rebuilt_csc(&source_columns, Some, |index| Ok(self.values[index]))?;
        Self::new(
            self.rows,
            source_columns.len(),
            col_ptrs,
            row_indices,
            values,
        )
    }

    pub fn class_name(&self) -> &'static str {
        self.integer_data
            .as_ref()
            .map_or("double", IntegerStorage::class_name)
    }
}

#[cfg(test)]
mod sparse_tensor_tests {
    use super::*;

    #[test]
    fn typed_sparse_scalar_updates_preserve_exact_values_and_zero_elision() {
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 1],
            vec![0],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("sparse");
        let inserted = sparse
            .with_updated_integer_value(1, 1, IntValue::U64(9_223_372_036_854_775_808))
            .expect("insert");
        assert_eq!(inserted.col_ptrs, vec![0, 1, 2]);
        assert_eq!(inserted.row_indices, vec![0, 1]);
        assert_eq!(
            inserted.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                u64::MAX,
                9_223_372_036_854_775_808
            ]))
        );

        let removed = inserted
            .with_updated_integer_value(0, 0, IntValue::U64(0))
            .expect("remove");
        assert_eq!(removed.col_ptrs, vec![0, 0, 1]);
        assert_eq!(removed.row_indices, vec![1]);
        assert_eq!(
            removed.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808]))
        );
    }

    #[test]
    fn floating_sparse_scalar_updates_keep_csc_order_and_elide_zero() {
        let sparse =
            SparseTensor::new(3, 1, vec![0, 2], vec![0, 2], vec![1.0, 3.0]).expect("sparse");
        let inserted = sparse.with_updated_value(1, 0, 2.0).expect("insert");
        assert_eq!(inserted.row_indices, vec![0, 1, 2]);
        assert_eq!(inserted.values, vec![1.0, 2.0, 3.0]);

        let removed = inserted.with_updated_value(1, 0, 0.0).expect("remove");
        assert_eq!(removed.row_indices, vec![0, 2]);
        assert_eq!(removed.values, vec![1.0, 3.0]);
    }

    #[test]
    fn sparse_structural_deletion_preserves_csc_and_exact_integer_values() {
        let sparse = SparseTensor::new_integer(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            IntegerStorage::U64(vec![1, u64::MAX, 9_223_372_036_854_775_808, 4, 5]),
        )
        .expect("sparse");

        let without_middle_row = sparse.with_deleted_rows(&[1]).expect("delete row");
        assert_eq!(without_middle_row.shape(), vec![2, 3]);
        assert_eq!(without_middle_row.col_ptrs, vec![0, 2, 2, 4]);
        assert_eq!(without_middle_row.row_indices, vec![0, 1, 0, 1]);
        assert_eq!(
            without_middle_row.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, u64::MAX, 4, 5]))
        );

        let without_outer_columns = sparse
            .with_deleted_columns(&[0, 2])
            .expect("delete columns");
        assert_eq!(without_outer_columns.shape(), vec![3, 1]);
        assert_eq!(without_outer_columns.col_ptrs, vec![0, 1]);
        assert_eq!(without_outer_columns.row_indices, vec![1]);
        assert_eq!(
            without_outer_columns.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808]))
        );

        assert!(sparse.with_deleted_rows(&[1, 1]).is_err());
        assert!(sparse.with_deleted_columns(&[3]).is_err());
    }

    #[test]
    fn sparse_expansion_preserves_csc_and_integer_storage() {
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 0],
            IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]),
        )
        .expect("sparse");
        let expanded = sparse.with_expanded_shape(4, 4).expect("expand");
        assert_eq!(expanded.shape(), vec![4, 4]);
        assert_eq!(expanded.col_ptrs, vec![0, 1, 2, 2, 2]);
        assert_eq!(expanded.row_indices, vec![1, 0]);
        assert_eq!(expanded.integer_storage(), sparse.integer_storage());
        assert!(expanded.with_expanded_shape(1, 4).is_err());
    }

    #[test]
    fn sparse_display_reports_exact_integer_class_and_values() {
        let mut sparse = SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![1],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("uint64 sparse");
        sparse.values.fill(f64::NAN);
        let text = sparse.to_string();

        assert!(text.contains("2x1 uint64 sparse matrix with 1 nonzero entries"));
        assert!(text.contains("18446744073709551615"));
        assert!(!text.contains("18446744073709552000"));
    }

    #[test]
    fn sparse_legacy_reads_keep_integer_storage_authoritative_when_mirrors_are_poisoned() {
        let mut unsigned = SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![1],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("uint64 sparse");
        unsigned.values.fill(f64::NAN);
        assert_eq!(unsigned.get(1, 0), Some(u64::MAX as f64));
        let dense = unsigned.to_dense().expect("dense uint64 sparse");
        assert_eq!(
            dense.integer_storage(),
            Some(&IntegerStorage::U64(vec![0, u64::MAX]))
        );

        let mut signed = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        )
        .expect("int64 sparse");
        signed.values.fill(f64::NAN);
        assert_eq!(signed.get(0, 0), Some(i64::MIN as f64));
        assert_eq!(signed.get(1, 1), Some(i64::MAX as f64));
        let text = signed.to_string();
        assert!(text.contains("-9223372036854775808"));
        assert!(text.contains("9223372036854775807"));
    }

    #[test]
    fn to_dense_rejects_overflowing_dimensions() {
        let sparse = SparseTensor {
            rows: usize::MAX,
            cols: 2,
            col_ptrs: vec![0, 0, 0],
            row_indices: Vec::new(),
            values: Vec::new(),
            integer_data: None,
        };

        let err = sparse.to_dense().unwrap_err();
        assert!(err.contains("overflow"));
    }
}

impl ComplexTensor {
    pub fn new(data: Vec<(f64, f64)>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "ComplexTensor data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        Ok(ComplexTensor {
            data,
            integer_data: None,
            shape,
            rows,
            cols,
        })
    }
    pub fn new_integer(storage: IntegerComplexStorage, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if storage.len() != expected {
            return Err("complex integer storage length does not match shape".into());
        }
        let data = storage
            .real
            .to_f64_vec()
            .into_iter()
            .zip(storage.imag.to_f64_vec())
            .collect();
        let mut tensor = Self::new(data, shape)?;
        tensor.integer_data = Some(storage);
        Ok(tensor)
    }
    pub fn new_2d(data: Vec<(f64, f64)>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        ComplexTensor {
            data: vec![(0.0, 0.0); size],
            integer_data: None,
            shape,
            rows,
            cols,
        }
    }

    /// Formats one element using its exact integer components when present.
    ///
    /// `data` remains a floating compatibility view during the integer-dtype
    /// migration, so it must not be used to render typed values such as
    /// `uint64` above the exact IEEE-754 range.
    pub fn format_element(&self, index: usize) -> String {
        if let Some(storage) = &self.integer_data {
            let real = storage
                .real
                .value_at(index)
                .expect("complex integer real storage must match tensor shape");
            let imag = storage
                .imag
                .value_at(index)
                .expect("complex integer imaginary storage must match tensor shape");
            return format_integer_complex_value(&real, &imag);
        }

        let (real, imag) = self.data[index];
        Value::Complex(real, imag).to_string()
    }
}

fn format_integer_complex_value(real: &IntValue, imag: &IntValue) -> String {
    if imag.is_zero() {
        return format_integer_value(real);
    }
    if real.is_zero() {
        return format!("{}i", format_integer_value(imag));
    }
    if integer_value_is_negative(imag) {
        return format!(
            "{}-{}i",
            format_integer_value(real),
            format_integer_magnitude(imag)
        );
    }
    format!(
        "{}+{}i",
        format_integer_value(real),
        format_integer_value(imag)
    )
}

fn format_integer_value(value: &IntValue) -> String {
    match value {
        IntValue::I8(value) => value.to_string(),
        IntValue::I16(value) => value.to_string(),
        IntValue::I32(value) => value.to_string(),
        IntValue::I64(value) => value.to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

fn integer_value_is_negative(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value < 0,
        IntValue::I16(value) => *value < 0,
        IntValue::I32(value) => *value < 0,
        IntValue::I64(value) => *value < 0,
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => false,
    }
}

fn format_integer_magnitude(value: &IntValue) -> String {
    match value {
        IntValue::I8(value) => value.unsigned_abs().to_string(),
        IntValue::I16(value) => value.unsigned_abs().to_string(),
        IntValue::I32(value) => value.unsigned_abs().to_string(),
        IntValue::I64(value) => value.unsigned_abs().to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

const MAX_ND_DISPLAY_ELEMENTS: usize = 4096;

fn should_expand_nd_display(shape: &[usize]) -> bool {
    shape.len() > 2
        && matches!(
            total_len(shape),
            Some(total) if total > 0 && total <= MAX_ND_DISPLAY_ELEMENTS
        )
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn decode_page_coords(mut page_index: usize, page_shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(page_shape.len());
    for &dim in page_shape {
        if dim == 0 {
            coords.push(0);
        } else {
            coords.push(page_index % dim);
            page_index /= dim;
        }
    }
    coords
}

fn write_nd_pages(
    f: &mut fmt::Formatter<'_>,
    shape: &[usize],
    mut write_element: impl FnMut(&mut fmt::Formatter<'_>, usize) -> fmt::Result,
) -> fmt::Result {
    if shape.len() <= 2 {
        return Ok(());
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows == 0 || cols == 0 {
        return write!(f, "[]");
    }
    let Some(page_count) = total_len(&shape[2..]) else {
        return write!(f, "Tensor(shape={shape:?})");
    };
    if page_count == 0 {
        return write!(f, "[]");
    }
    let strides = column_major_strides(shape);
    for page_index in 0..page_count {
        if page_index > 0 {
            write!(f, "\n\n")?;
        }
        let coords = decode_page_coords(page_index, &shape[2..]);
        write!(f, "(:, :")?;
        for &coord in &coords {
            write!(f, ", {}", coord + 1)?;
        }
        write!(f, ") =")?;

        let mut page_base = 0usize;
        for (offset, &coord) in coords.iter().enumerate() {
            page_base += coord * strides[offset + 2];
        }
        for r in 0..rows {
            writeln!(f)?;
            write!(f, "  ")?;
            for c in 0..cols {
                if c > 0 {
                    write!(f, "  ")?;
                }
                let linear = page_base + r + c * rows;
                write_element(f, linear)?;
            }
        }
    }
    Ok(())
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format_element = |idx: usize| {
            self.integer_data
                .as_ref()
                .and_then(|storage| storage.value_at(idx))
                .map(|value| value.decimal_string())
                .unwrap_or_else(|| format_number(self.data[idx]))
        };

        match self.shape.len() {
            0 | 1 => {
                // Treat as row vector for display
                write!(f, "[")?;
                let len = self
                    .integer_data
                    .as_ref()
                    .map_or(self.data.len(), IntegerStorage::len);
                for i in 0..len {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", format_element(i))?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.rows();
                let cols = self.cols();
                // Display as matrix
                for r in 0..rows {
                    writeln!(f)?;
                    write!(f, "  ")?; // Indent
                    for c in 0..cols {
                        if c > 0 {
                            write!(f, "  ")?;
                        }
                        write!(f, "{}", format_element(r + c * rows))?;
                    }
                }
                Ok(())
            }
            _ => {
                if should_expand_nd_display(&self.shape) {
                    write_nd_pages(f, &self.shape, |f, idx| {
                        write!(f, "{}", format_element(idx))
                    })
                } else {
                    write!(f, "Tensor(shape={:?})", self.shape)
                }
            }
        }
    }
}

impl fmt::Display for SparseTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}x{} {} sparse matrix with {} nonzero entries",
            self.rows,
            self.cols,
            self.class_name(),
            self.nnz()
        )?;
        if self.nnz() == 0 {
            return Ok(());
        }
        for col in 0..self.cols {
            for idx in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                let row = self.row_indices[idx];
                let value = self
                    .integer_data
                    .as_ref()
                    .and_then(|storage| storage.value_at(idx).map(format_int_value));
                writeln!(
                    f,
                    "  ({},{})  {}",
                    row + 1,
                    col + 1,
                    value.unwrap_or_else(|| format_number(self.values[idx]))
                )?;
            }
        }
        Ok(())
    }
}

fn format_int_value(value: IntValue) -> String {
    match value {
        IntValue::I8(value) => value.to_string(),
        IntValue::I16(value) => value.to_string(),
        IntValue::I32(value) => value.to_string(),
        IntValue::I64(value) => value.to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

impl fmt::Display for StringArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = match self.shape.len() {
            0 => (0, 0),
            1 => (1, self.shape[0]),
            _ => (self.shape[0], self.shape[1]),
        };
        let count = self.data.len();
        if count == 1 && rows == 1 && cols == 1 {
            let v = &self.data[0];
            if v == "<missing>" {
                return write!(f, "<missing>");
            }
            let escaped = v.replace('"', "\\\"");
            return write!(f, "\"{escaped}\"");
        }
        if self.shape.len() > 2 {
            let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
            return write!(f, "{} string array", dims.join("x"));
        }
        write!(f, "{rows}x{cols} string array")?;
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        for r in 0..rows {
            writeln!(f)?;
            write!(f, "  ")?;
            for c in 0..cols {
                if c > 0 {
                    write!(f, "  ")?;
                }
                let v = &self.data[r + c * rows];
                if v == "<missing>" {
                    write!(f, "<missing>")?;
                } else {
                    let escaped = v.replace('"', "\\\"");
                    write!(f, "\"{escaped}\"")?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for LogicalArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.data.len() == 1 {
            return write!(f, "{}", if self.data[0] != 0 { 1 } else { 0 });
        }
        match self.shape.len() {
            0 => write!(f, "[]"),
            1 => {
                write!(f, "[")?;
                for (i, v) in self.data.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", if *v != 0 { 1 } else { 0 })?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                // Display as matrix
                for r in 0..rows {
                    writeln!(f)?;
                    write!(f, "  ")?; // Indent
                    for c in 0..cols {
                        if c > 0 {
                            write!(f, "  ")?;
                        }
                        let idx = r + c * rows;
                        write!(f, "{}", if self.data[idx] != 0 { 1 } else { 0 })?;
                    }
                }
                Ok(())
            }
            _ => {
                if should_expand_nd_display(&self.shape) {
                    write_nd_pages(f, &self.shape, |f, idx| {
                        write!(f, "{}", if self.data[idx] != 0 { 1 } else { 0 })
                    })
                } else {
                    let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
                    write!(f, "{} logical array", dims.join("x"))
                }
            }
        }
    }
}

impl fmt::Display for CharArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..self.rows {
            writeln!(f)?;
            write!(f, "  ")?; // Indent
            for c in 0..self.cols {
                let ch = self.data[r * self.cols + c];
                write!(f, "{ch}")?;
            }
        }
        Ok(())
    }
}

// From implementations for Value
impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Int(IntValue::I32(i))
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(IntValue::I64(i))
    }
}
impl From<u32> for Value {
    fn from(i: u32) -> Self {
        Value::Int(IntValue::U32(i))
    }
}
impl From<u64> for Value {
    fn from(i: u64) -> Self {
        Value::Int(IntValue::U64(i))
    }
}
impl From<i16> for Value {
    fn from(i: i16) -> Self {
        Value::Int(IntValue::I16(i))
    }
}
impl From<i8> for Value {
    fn from(i: i8) -> Self {
        Value::Int(IntValue::I8(i))
    }
}
impl From<u16> for Value {
    fn from(i: u16) -> Self {
        Value::Int(IntValue::U16(i))
    }
}
impl From<u8> for Value {
    fn from(i: u8) -> Self {
        Value::Int(IntValue::U8(i))
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Num(f)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<Tensor> for Value {
    fn from(m: Tensor) -> Self {
        Value::Tensor(m)
    }
}

// Remove blanket From<Vec<Value>> to avoid losing shape information

// TryFrom implementations for extracting native types
impl TryFrom<&Value> for i32 {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Int(i) => Ok(i.to_i64() as i32),
            Value::Num(n) => Ok(*n as i32),
            _ => Err(format!("cannot convert {v:?} to i32")),
        }
    }
}

impl TryFrom<&Value> for f64 {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Num(n) => Ok(*n),
            Value::Int(i) => Ok(i.to_f64()),
            _ => Err(format!("cannot convert {v:?} to f64")),
        }
    }
}

impl TryFrom<&Value> for bool {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Bool(b) => Ok(*b),
            Value::Int(i) => Ok(!i.is_zero()),
            Value::Num(n) => Ok(*n != 0.0),
            _ => Err(format!("cannot convert {v:?} to bool")),
        }
    }
}

impl TryFrom<&Value> for String {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::String(s) => Ok(s.clone()),
            Value::StringArray(sa) => {
                if sa.data.len() == 1 {
                    Ok(sa.data[0].clone())
                } else {
                    Err("cannot convert string array to scalar string".to_string())
                }
            }
            Value::CharArray(ca) => {
                // Convert full char array to one string if it is a single row; else error
                if ca.rows == 1 {
                    Ok(ca.data.iter().collect())
                } else {
                    Err("cannot convert multi-row char array to scalar string".to_string())
                }
            }
            Value::Int(i) => Ok(i.decimal_string()),
            Value::Num(n) => Ok(n.to_string()),
            Value::Bool(b) => Ok(b.to_string()),
            _ => Err(format!("cannot convert {v:?} to String")),
        }
    }
}

impl TryFrom<&Value> for Tensor {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tensor(m) => Ok(m.clone()),
            _ => Err(format!("cannot convert {v:?} to Tensor")),
        }
    }
}

impl TryFrom<&Value> for Value {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        Ok(v.clone())
    }
}

impl TryFrom<&Value> for Vec<Value> {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Cell(c) => Ok(c.data.clone()),
            _ => Err(format!("cannot convert {v:?} to Vec<Value>")),
        }
    }
}

use serde::{Deserialize, Serialize};

/// Enhanced type system used throughout RunMat for HIR and builtin functions
/// Designed to mirror Value variants for better type inference and LSP support
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub enum Type {
    /// Integer number type
    Int,
    /// Floating-point number type  
    Num,
    /// Boolean type
    Bool,
    /// Logical array type (N-D boolean array) with optional shape information
    Logical {
        /// Optional full shape; None means unknown/dynamic; individual dims can be omitted by using None
        shape: Option<Vec<Option<usize>>>,
    },
    /// String type
    String,
    /// Tensor type with optional shape information (column-major semantics in runtime)
    Tensor {
        /// Optional full shape; None means unknown/dynamic; individual dims can be omitted by using None
        shape: Option<Vec<Option<usize>>>,
    },
    /// Scalar symbolic expression type.
    Symbolic,
    /// Cell array type with optional element type information
    Cell {
        /// Optional element type (None means mixed/unknown)
        element_type: Option<Box<Type>>,
        /// Optional length (None means unknown/dynamic)
        length: Option<usize>,
    },
    /// Function type with parameter and return types
    Function {
        /// Parameter types
        params: Vec<Type>,
        /// Return type
        returns: Box<Type>,
    },
    /// Void type (no value)
    Void,
    /// Unknown type (for type inference)
    Unknown,
    /// Union type (multiple possible types)
    Union(Vec<Type>),
    /// Struct-like type with optional known field set (purely for inference)
    Struct {
        /// Optional set of known field names observed via control-flow (None = unknown fields)
        known_fields: Option<Vec<String>>, // kept sorted unique for deterministic Eq
    },
    /// Multiple return values captured as a list (internal destructuring helper)
    OutputList(Vec<Type>),
}

impl Type {
    /// Create a tensor type with unknown shape
    pub fn tensor() -> Self {
        Type::Tensor { shape: None }
    }

    /// Create a logical type with unknown shape
    pub fn logical() -> Self {
        Type::Logical { shape: None }
    }

    /// Create a logical type with known shape
    pub fn logical_with_shape(shape: Vec<usize>) -> Self {
        Type::Logical {
            shape: Some(shape.into_iter().map(Some).collect()),
        }
    }

    /// Create a tensor type with known shape
    pub fn tensor_with_shape(shape: Vec<usize>) -> Self {
        Type::Tensor {
            shape: Some(shape.into_iter().map(Some).collect()),
        }
    }

    /// Create a cell array type with unknown element type
    pub fn cell() -> Self {
        Type::Cell {
            element_type: None,
            length: None,
        }
    }

    /// Create a cell array type with known element type
    pub fn cell_of(element_type: Type) -> Self {
        Type::Cell {
            element_type: Some(Box::new(element_type)),
            length: None,
        }
    }

    /// Check if this type is compatible with another type
    pub fn is_compatible_with(&self, other: &Type) -> bool {
        match (self, other) {
            (Type::Unknown, _) | (_, Type::Unknown) => true,
            (Type::Int, Type::Num) | (Type::Num, Type::Int) => true, // Number compatibility
            (Type::Tensor { .. }, Type::Tensor { .. }) => true, // Tensor compatibility regardless of dims for now
            (Type::OutputList(a), Type::OutputList(b)) => a.len() == b.len(),
            (a, b) => a == b,
        }
    }

    /// Get the most specific common type between two types
    pub fn unify(&self, other: &Type) -> Type {
        match (self, other) {
            (Type::Unknown, t) | (t, Type::Unknown) => t.clone(),
            (Type::Int, Type::Num) | (Type::Num, Type::Int) => Type::Num,
            (Type::Tensor { shape: a }, Type::Tensor { shape: b }) => {
                let a_norm = match a {
                    Some(dims) if dims.is_empty() => None,
                    _ => a.clone(),
                };
                let b_norm = match b {
                    Some(dims) if dims.is_empty() => None,
                    _ => b.clone(),
                };
                let a_unknown = a_norm
                    .as_ref()
                    .map(|dims| dims.iter().all(|d| d.is_none()))
                    .unwrap_or(true);
                let b_unknown = b_norm
                    .as_ref()
                    .map(|dims| dims.iter().all(|d| d.is_none()))
                    .unwrap_or(true);
                if a_norm == b_norm
                    || (!a_unknown && b_unknown)
                    || (a_norm.is_some() && b_norm.is_none())
                {
                    Type::Tensor { shape: a_norm }
                } else if (a_unknown && !b_unknown) || (a_norm.is_none() && b_norm.is_some()) {
                    Type::Tensor { shape: b_norm }
                } else {
                    Type::tensor()
                }
            }
            (Type::Logical { shape: a }, Type::Logical { shape: b }) => {
                let a_norm = match a {
                    Some(dims) if dims.is_empty() => None,
                    _ => a.clone(),
                };
                let b_norm = match b {
                    Some(dims) if dims.is_empty() => None,
                    _ => b.clone(),
                };
                let a_unknown = a_norm
                    .as_ref()
                    .map(|dims| dims.iter().all(|d| d.is_none()))
                    .unwrap_or(true);
                let b_unknown = b_norm
                    .as_ref()
                    .map(|dims| dims.iter().all(|d| d.is_none()))
                    .unwrap_or(true);
                if a_norm == b_norm
                    || (!a_unknown && b_unknown)
                    || (a_norm.is_some() && b_norm.is_none())
                {
                    Type::Logical { shape: a_norm }
                } else if (a_unknown && !b_unknown) || (a_norm.is_none() && b_norm.is_some()) {
                    Type::Logical { shape: b_norm }
                } else {
                    Type::logical()
                }
            }
            (Type::Struct { known_fields: a }, Type::Struct { known_fields: b }) => match (a, b) {
                (None, None) => Type::Struct { known_fields: None },
                (Some(ka), None) | (None, Some(ka)) => Type::Struct {
                    known_fields: Some(ka.clone()),
                },
                (Some(ka), Some(kb)) => {
                    let mut set: std::collections::BTreeSet<String> = ka.iter().cloned().collect();
                    set.extend(kb.iter().cloned());
                    Type::Struct {
                        known_fields: Some(set.into_iter().collect()),
                    }
                }
            },
            (Type::OutputList(a), Type::OutputList(b)) => {
                if a.len() == b.len() {
                    let items = a
                        .iter()
                        .zip(b.iter())
                        .map(|(lhs, rhs)| lhs.unify(rhs))
                        .collect();
                    Type::OutputList(items)
                } else {
                    Type::OutputList(vec![Type::Unknown; a.len().max(b.len())])
                }
            }
            (a, b) if a == b => a.clone(),
            _ => Type::Union(vec![self.clone(), other.clone()]),
        }
    }

    /// Infer type from a Value
    pub fn from_value(value: &Value) -> Type {
        match value {
            Value::Int(_) => Type::Int,
            Value::Num(_) => Type::Num,
            Value::Complex(_, _) => Type::Num, // treat as numeric double (complex) in type system for now
            Value::Bool(_) => Type::Bool,
            Value::LogicalArray(arr) => Type::Logical {
                shape: Some(arr.shape.iter().map(|&d| Some(d)).collect()),
            },
            Value::String(_) => Type::String,
            Value::StringArray(_sa) => {
                // Model as Cell of String for type system for now
                Type::cell_of(Type::String)
            }
            Value::Tensor(t) => Type::Tensor {
                shape: Some(t.shape.iter().map(|&d| Some(d)).collect()),
            },
            Value::SparseTensor(t) => Type::Tensor {
                shape: Some(vec![Some(t.rows), Some(t.cols)]),
            },
            Value::ComplexTensor(t) => Type::Tensor {
                shape: Some(t.shape.iter().map(|&d| Some(d)).collect()),
            },
            Value::Symbolic(_) => Type::Symbolic,
            Value::Cell(cells) => {
                if cells.data.is_empty() {
                    Type::cell()
                } else {
                    // Infer element type from first element
                    let element_type = Type::from_value(&cells.data[0]);
                    Type::Cell {
                        element_type: Some(Box::new(element_type)),
                        length: Some(cells.data.len()),
                    }
                }
            }
            Value::GpuTensor(h) => Type::Tensor {
                shape: Some(h.shape.iter().map(|&d| Some(d)).collect()),
            },
            Value::Object(_) => Type::Unknown,
            Value::HandleObject(_) => Type::Unknown,
            Value::Listener(_) => Type::Unknown,
            Value::Struct(_) => Type::Struct { known_fields: None },
            Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. } => Type::Function {
                params: vec![Type::Unknown],
                returns: Box::new(Type::Unknown),
            },
            Value::Closure(_) => Type::Function {
                params: vec![Type::Unknown],
                returns: Box::new(Type::Unknown),
            },
            Value::ClassRef(_) => Type::Unknown,
            Value::MException(_) => Type::Unknown,
            Value::CharArray(ca) => {
                // Treat as cell of char for type purposes; or a 2-D char matrix conceptually
                Type::Cell {
                    element_type: Some(Box::new(Type::String)),
                    length: Some(ca.rows * ca.cols),
                }
            }
            Value::OutputList(values) => {
                Type::OutputList(values.iter().map(Type::from_value).collect())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    pub function_name: String,
    pub bound_function: Option<usize>,
    pub captures: Vec<Value>,
}

/// Acceleration metadata describing GPU-friendly characteristics of a builtin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelTag {
    Unary,
    Elementwise,
    Reduction,
    MatMul,
    Transpose,
    ArrayConstruct,
}

/// Control-flow type for builtins that may suspend or error.
pub type BuiltinControlFlow = runmat_async::RuntimeError;

/// Async result type for builtins.
pub type BuiltinFuture = Pin<Box<dyn Future<Output = Result<Value, BuiltinControlFlow>> + 'static>>;

#[derive(Clone, Debug, Default)]
pub struct ResolveContext {
    pub literal_args: Vec<LiteralValue>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LiteralValue {
    Number(f64),
    Bool(bool),
    String(String),
    Vector(Vec<LiteralValue>),
    Unknown,
}

impl ResolveContext {
    pub fn new(literal_args: Vec<LiteralValue>) -> Self {
        Self { literal_args }
    }

    pub fn numeric_dims(&self) -> Vec<Option<usize>> {
        self.numeric_dims_from(0)
    }

    pub fn numeric_dims_from(&self, start: usize) -> Vec<Option<usize>> {
        let slice = self.literal_args.get(start..).unwrap_or(&[]);
        if let Some(LiteralValue::Vector(values)) = slice.first() {
            return values
                .iter()
                .map(Self::numeric_dimension_from_literal)
                .collect();
        }
        slice
            .iter()
            .map(Self::numeric_dimension_from_literal)
            .collect()
    }

    pub fn literal_string_at(&self, index: usize) -> Option<String> {
        match self.literal_args.get(index) {
            Some(LiteralValue::String(value)) => Some(value.to_ascii_lowercase()),
            _ => None,
        }
    }

    pub fn literal_bool_at(&self, index: usize) -> Option<bool> {
        match self.literal_args.get(index) {
            Some(LiteralValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn literal_vector_at(&self, index: usize) -> Option<Vec<LiteralValue>> {
        match self.literal_args.get(index) {
            Some(LiteralValue::Vector(values)) => Some(values.clone()),
            _ => None,
        }
    }

    pub fn numeric_vector_at(&self, index: usize) -> Option<Vec<Option<usize>>> {
        let values = match self.literal_args.get(index) {
            Some(LiteralValue::Vector(values)) => values,
            _ => return None,
        };
        if values
            .iter()
            .any(|value| matches!(value, LiteralValue::Vector(_)))
        {
            return None;
        }
        Some(
            values
                .iter()
                .map(Self::numeric_dimension_from_literal)
                .collect(),
        )
    }

    fn numeric_dimension_from_literal(value: &LiteralValue) -> Option<usize> {
        match value {
            LiteralValue::Number(num) => {
                if num.is_finite() {
                    let rounded = num.round();
                    if (num - rounded).abs() <= 1e-9 && rounded >= 0.0 {
                        return Some(rounded as usize);
                    }
                }
                None
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod resolve_context_tests {
    use super::{LiteralValue, ResolveContext};

    #[test]
    fn numeric_dims_reads_vector_literal() {
        let ctx = ResolveContext::new(vec![LiteralValue::Vector(vec![
            LiteralValue::Number(2.0),
            LiteralValue::Number(3.0),
        ])]);
        assert_eq!(ctx.numeric_dims(), vec![Some(2), Some(3)]);
    }

    #[test]
    fn numeric_dims_skips_non_numeric_entries() {
        let ctx = ResolveContext::new(vec![
            LiteralValue::Number(4.0),
            LiteralValue::String("like".to_string()),
            LiteralValue::Unknown,
        ]);
        assert_eq!(ctx.numeric_dims(), vec![Some(4), None, None]);
    }

    #[test]
    fn numeric_dims_prefers_vector_even_with_trailing_args() {
        let ctx = ResolveContext::new(vec![
            LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(5.0)]),
            LiteralValue::String("like".to_string()),
        ]);
        assert_eq!(ctx.numeric_dims(), vec![Some(1), Some(5)]);
    }

    #[test]
    fn literal_string_is_lowercased() {
        let ctx = ResolveContext::new(vec![LiteralValue::String("OmItNaN".to_string())]);
        assert_eq!(ctx.literal_string_at(0), Some("omitnan".to_string()));
    }

    #[test]
    fn literal_bool_is_available() {
        let ctx = ResolveContext::new(vec![LiteralValue::Bool(true)]);
        assert_eq!(ctx.literal_bool_at(0), Some(true));
    }

    #[test]
    fn literal_vector_at_returns_clone() {
        let ctx = ResolveContext::new(vec![LiteralValue::Vector(vec![
            LiteralValue::Number(7.0),
            LiteralValue::Unknown,
        ])]);
        assert_eq!(
            ctx.literal_vector_at(0),
            Some(vec![LiteralValue::Number(7.0), LiteralValue::Unknown])
        );
    }

    #[test]
    fn numeric_vector_at_rejects_nested_vectors() {
        let ctx = ResolveContext::new(vec![LiteralValue::Vector(vec![LiteralValue::Vector(
            vec![LiteralValue::Number(1.0)],
        )])]);
        assert_eq!(ctx.numeric_vector_at(0), None);
    }
}

pub type TypeResolver = fn(args: &[Type]) -> Type;
pub type TypeResolverWithContext = fn(args: &[Type], ctx: &ResolveContext) -> Type;

#[derive(Clone, Copy, Debug)]
pub enum TypeResolverKind {
    Simple(TypeResolver),
    WithContext(TypeResolverWithContext),
}

pub fn type_resolver_kind(resolver: TypeResolver) -> TypeResolverKind {
    TypeResolverKind::Simple(resolver)
}

pub fn type_resolver_kind_ctx(resolver: TypeResolverWithContext) -> TypeResolverKind {
    TypeResolverKind::WithContext(resolver)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinOutputMode {
    Fixed,
    ByRequestedOutputCount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinCompletionPolicy {
    Public,
    MethodOnly,
    HiddenInternal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinParamArity {
    Required,
    Optional,
    Variadic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinParamType {
    Any,
    NumericScalar,
    IntegerScalar,
    StringScalar,
    NumericArray,
    LogicalArray,
    SizeArg,
    LikePrototype,
    AxesHandle,
    StyleSpec,
    PropertyName,
    PropertyValue,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinParamDescriptor {
    pub name: &'static str,
    pub ty: BuiltinParamType,
    pub arity: BuiltinParamArity,
    pub default: Option<&'static str>,
    pub description: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinSignatureDescriptor {
    pub label: &'static str,
    pub inputs: &'static [BuiltinParamDescriptor],
    pub outputs: &'static [BuiltinParamDescriptor],
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinErrorDescriptor {
    pub code: &'static str,
    pub identifier: Option<&'static str>,
    pub when: &'static str,
    pub message: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinDescriptor {
    pub signatures: &'static [BuiltinSignatureDescriptor],
    pub output_mode: BuiltinOutputMode,
    pub completion_policy: BuiltinCompletionPolicy,
    pub errors: &'static [BuiltinErrorDescriptor],
}

/// Simple builtin function definition using the unified type system
#[derive(Debug, Clone)]
pub struct BuiltinFunction {
    pub name: &'static str,
    pub description: &'static str,
    pub category: &'static str,
    pub doc: &'static str,
    pub examples: &'static str,
    pub param_types: Vec<Type>,
    pub return_type: Type,
    pub type_resolver: Option<TypeResolverKind>,
    pub implementation: fn(&[Value]) -> BuiltinFuture,
    pub accel_tags: &'static [AccelTag],
    pub is_sink: bool,
    pub suppress_auto_output: bool,
    pub descriptor: Option<&'static BuiltinDescriptor>,
}

impl BuiltinFunction {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &'static str,
        description: &'static str,
        category: &'static str,
        doc: &'static str,
        examples: &'static str,
        param_types: Vec<Type>,
        return_type: Type,
        type_resolver: Option<TypeResolverKind>,
        implementation: fn(&[Value]) -> BuiltinFuture,
        accel_tags: &'static [AccelTag],
        is_sink: bool,
        suppress_auto_output: bool,
    ) -> Self {
        Self {
            name,
            description,
            category,
            doc,
            examples,
            param_types,
            return_type,
            type_resolver,
            implementation,
            accel_tags,
            is_sink,
            suppress_auto_output,
            descriptor: None,
        }
    }

    pub fn with_descriptor(mut self, descriptor: &'static BuiltinDescriptor) -> Self {
        self.descriptor = Some(descriptor);
        self
    }

    pub fn with_descriptor_option(
        mut self,
        descriptor: Option<&'static BuiltinDescriptor>,
    ) -> Self {
        self.descriptor = descriptor;
        self
    }

    pub fn infer_return_type(&self, args: &[Type]) -> Type {
        self.infer_return_type_with_context(args, &ResolveContext::default())
    }

    pub fn infer_return_type_with_context(&self, args: &[Type], ctx: &ResolveContext) -> Type {
        if let Some(resolver) = self.type_resolver {
            return match resolver {
                TypeResolverKind::Simple(resolver) => resolver(args),
                TypeResolverKind::WithContext(resolver) => resolver(args, ctx),
            };
        }
        self.return_type.clone()
    }

    pub fn semantics(&self) -> BuiltinSemantics {
        semantics::builtin_semantics_for(self)
    }
}

/// A constant value that can be accessed as a variable
#[derive(Clone)]
pub struct Constant {
    pub name: &'static str,
    pub value: Value,
}

pub mod semantics;
pub mod shape_rules;

pub use semantics::{
    builtin_semantics_for, builtin_semantics_for_name, BuiltinAsyncBehavior, BuiltinCompatibility,
    BuiltinEffects, BuiltinEnvironmentEffect, BuiltinPurity, BuiltinSemanticKind, BuiltinSemantics,
    BuiltinWorkspaceEffect, ConcatKind, ShapeTransformKind,
};

impl std::fmt::Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Constant {{ name: {:?}, value: {:?} }}",
            self.name, self.value
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(BuiltinFunction);
#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(Constant);

#[cfg(not(target_arch = "wasm32"))]
pub fn builtin_functions() -> Vec<&'static BuiltinFunction> {
    inventory::iter::<BuiltinFunction>().collect()
}

#[cfg(target_arch = "wasm32")]
pub fn builtin_functions() -> Vec<&'static BuiltinFunction> {
    wasm_registry::builtin_functions()
}

#[cfg(not(target_arch = "wasm32"))]
static BUILTIN_LOOKUP: OnceLock<HashMap<String, &'static BuiltinFunction>> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn builtin_lookup_map() -> &'static HashMap<String, &'static BuiltinFunction> {
    BUILTIN_LOOKUP.get_or_init(|| {
        let mut map = HashMap::new();
        for func in builtin_functions() {
            map.insert(func.name.to_ascii_lowercase(), func);
        }
        map
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn builtin_function_by_name(name: &str) -> Option<&'static BuiltinFunction> {
    builtin_lookup_map()
        .get(&name.to_ascii_lowercase())
        .copied()
}

#[cfg(target_arch = "wasm32")]
pub fn builtin_function_by_name(name: &str) -> Option<&'static BuiltinFunction> {
    wasm_registry::builtin_functions()
        .into_iter()
        .find(|f| f.name.eq_ignore_ascii_case(name))
}

pub fn suppresses_auto_output(name: &str) -> bool {
    builtin_function_by_name(name)
        .map(|f| f.suppress_auto_output)
        .unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn constants() -> Vec<&'static Constant> {
    inventory::iter::<Constant>().collect()
}

#[cfg(target_arch = "wasm32")]
pub fn constants() -> Vec<&'static Constant> {
    wasm_registry::constants()
}

// ----------------------
// Builtin documentation metadata (optional, registered by macros)
// ----------------------

#[derive(Debug)]
pub struct BuiltinDoc {
    pub name: &'static str,
    pub category: Option<&'static str>,
    pub summary: Option<&'static str>,
    pub keywords: Option<&'static str>,
    pub errors: Option<&'static str>,
    pub related: Option<&'static str>,
    pub introduced: Option<&'static str>,
    pub status: Option<&'static str>,
    pub examples: Option<&'static str>,
}

#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(BuiltinDoc);

#[cfg(not(target_arch = "wasm32"))]
pub fn builtin_docs() -> Vec<&'static BuiltinDoc> {
    inventory::iter::<BuiltinDoc>().collect()
}

#[cfg(target_arch = "wasm32")]
pub fn builtin_docs() -> Vec<&'static BuiltinDoc> {
    wasm_registry::builtin_docs()
}

// ----------------------
// Display implementations
// ----------------------

/// Controls how numeric values are displayed in the console, mirroring MATLAB's `format` command.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FormatMode {
    /// 4 decimal places, fixed or scientific (MATLAB default).
    #[default]
    Short,
    /// 15 decimal places, fixed or scientific.
    Long,
    /// Always scientific notation, 4 decimal places.
    ShortE,
    /// Always scientific notation, 14 decimal places.
    LongE,
    /// Compact: shorter of fixed/scientific, 5 significant digits.
    ShortG,
    /// Compact: shorter of fixed/scientific, 15 significant digits.
    LongG,
    /// Rational approximation (p/q).
    Rational,
    /// IEEE 754 hexadecimal representation.
    Hex,
}

runmat_thread_local! {
    static DISPLAY_FORMAT: RefCell<FormatMode> = const { RefCell::new(FormatMode::Short) };
}

pub fn set_display_format(mode: FormatMode) {
    DISPLAY_FORMAT.with(|c| *c.borrow_mut() = mode);
}

pub fn get_display_format() -> FormatMode {
    DISPLAY_FORMAT.with(|c| *c.borrow())
}

/// Format a number using the current thread-local display format.
pub fn format_number(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf"
        } else {
            "Inf"
        }
        .to_string();
    }
    let mode = get_display_format();
    if mode == FormatMode::Hex {
        return fmt_hex(value);
    }
    let v = if value == 0.0 { 0.0 } else { value };
    match mode {
        FormatMode::Short => fmt_short(v),
        FormatMode::Long => fmt_long(v),
        FormatMode::ShortE => fmt_sci(v, 4),
        FormatMode::LongE => fmt_sci(v, 14),
        FormatMode::ShortG => fmt_compact(v, 5),
        FormatMode::LongG => fmt_compact(v, 15),
        FormatMode::Rational => fmt_rational(v),
        FormatMode::Hex => unreachable!("hex mode handled before zero normalization"),
    }
}

/// Reformat Rust's `e`-notation exponent into MATLAB style (`e+02`, `e-03`).
fn matlab_exp(s: &str) -> String {
    if let Some(e_pos) = s.find('e') {
        let mantissa = &s[..e_pos];
        let exp: i32 = s[e_pos + 1..].parse().unwrap_or(0);
        let sign = if exp >= 0 { '+' } else { '-' };
        format!("{mantissa}e{sign}{:02}", exp.unsigned_abs())
    } else {
        s.to_string()
    }
}

fn fmt_sci(v: f64, dec: usize) -> String {
    if v == 0.0 {
        return format!("0.{:0>dec$}e+00", 0, dec = dec);
    }
    let s = format!("{v:.dec$e}");
    matlab_exp(&s)
}

fn fmt_short(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        return "0".to_string();
    }
    if v.fract() == 0.0 && abs < 1e15 {
        return format!("{}", v as i64);
    }
    if (0.001..10000.0).contains(&abs) {
        format!("{:.4}", v)
    } else {
        fmt_sci(v, 4)
    }
}

fn fmt_long(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        return "0".to_string();
    }
    if v.fract() == 0.0 && abs < 1e15 {
        return format!("{}", v as i64);
    }
    if (0.001..10000.0).contains(&abs) {
        format!("{:.15}", v)
    } else {
        fmt_sci(v, 14)
    }
}

fn fmt_compact(v: f64, sig_digits: usize) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        return "0".to_string();
    }
    let use_scientific = !(1e-4..1e6).contains(&abs);
    if use_scientific {
        let dec = sig_digits - 1;
        let s = format!("{v:.dec$e}");
        // trim trailing zeros in mantissa then reformat exponent
        if let Some(e_pos) = s.find('e') {
            let exp_part = &s[e_pos..];
            let mut mantissa = s[..e_pos].to_string();
            if let Some(dot) = mantissa.find('.') {
                let mut end = mantissa.len();
                while end > dot + 1 && mantissa.as_bytes()[end - 1] == b'0' {
                    end -= 1;
                }
                if mantissa.as_bytes()[end - 1] == b'.' {
                    end -= 1;
                }
                mantissa.truncate(end);
            }
            return matlab_exp(&format!("{mantissa}{exp_part}"));
        }
        return matlab_exp(&s);
    }
    let exp10 = abs.log10().floor() as i32;
    let decimals = ((sig_digits as i32 - 1 - exp10).max(0)) as usize;
    let pow = 10f64.powi(decimals as i32);
    let rounded = (v * pow).round() / pow;
    let mut s = format!("{rounded:.decimals$}");
    if let Some(dot) = s.find('.') {
        let mut end = s.len();
        while end > dot + 1 && s.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if s.as_bytes()[end - 1] == b'.' {
            end -= 1;
        }
        s.truncate(end);
    }
    if s.is_empty() || s == "-0" {
        s = "0".to_string();
    }
    s
}

fn fmt_rational(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let negative = v < 0.0;
    let abs = v.abs();
    if v.fract() == 0.0 && abs < 1e15 {
        return format!("{}", v as i64);
    }
    // Continued fraction convergents; stop at the first one within MATLAB's
    // 5e-7 relative tolerance (matches `format rational` behaviour for pi → 355/113).
    let tol = 5e-7 * abs;
    let max_d = 1_000_000i64;
    let mut n0: i64 = 1;
    let mut n1: i64 = abs.floor() as i64;
    let mut d0: i64 = 0;
    let mut d1: i64 = 1;
    let mut a = abs;
    let mut best_n = n1;
    let mut best_d = d1;
    for _ in 0..50 {
        if (abs - best_n as f64 / best_d as f64).abs() <= tol {
            break;
        }
        let f = a.fract();
        if f < 1e-10 {
            break;
        }
        a = 1.0 / f;
        let q = a.floor() as i64;
        let Some(n2) = q.checked_mul(n1).and_then(|v| v.checked_add(n0)) else {
            break;
        };
        let Some(d2) = q.checked_mul(d1).and_then(|v| v.checked_add(d0)) else {
            break;
        };
        if d2 > max_d {
            break;
        }
        best_n = n2;
        best_d = d2;
        n0 = n1;
        n1 = n2;
        d0 = d1;
        d1 = d2;
    }
    let sign = if negative { "-" } else { "" };
    if best_d == 1 {
        format!("{sign}{best_n}")
    } else {
        format!("{sign}{best_n}/{best_d}")
    }
}

fn fmt_hex(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

// -------- Exception type --------
#[derive(Debug, Clone, PartialEq)]
pub struct MException {
    pub identifier: String,
    pub message: String,
    pub stack: Vec<String>,
}

impl MException {
    pub fn new(identifier: String, message: String) -> Self {
        Self {
            identifier,
            message,
            stack: Vec::new(),
        }
    }
}

/// Reference to a GC-allocated object providing language handle semantics
#[derive(Debug, Clone)]
pub struct HandleRef {
    pub class_name: String,
    pub target: GcHandle,
    pub valid: bool,
}

impl PartialEq for HandleRef {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target
    }
}

/// Event listener handle for events
#[derive(Debug, Clone, PartialEq)]
pub struct Listener {
    pub id: u64,
    pub target: GcHandle,
    pub target_class_name: String,
    pub event_name: String,
    pub callback: GcHandle,
    pub enabled: bool,
    pub valid: bool,
}

impl Listener {
    pub fn class_name(&self) -> String {
        self.target_class_name.clone()
    }
}

impl Trace for CellArray {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in &self.data {
            value.trace(tracer);
        }
    }
}

impl Trace for StructValue {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in self.fields.values() {
            value.trace(tracer);
        }
    }
}

impl Trace for Closure {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in &self.captures {
            value.trace(tracer);
        }
    }
}

impl Trace for ObjectInstance {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in self.properties.values() {
            value.trace(tracer);
        }
        if let Some(dynamic_properties) = &self.dynamic_properties {
            for property in dynamic_properties.values() {
                if let Some(metadata_handle) = property.metadata_handle {
                    tracer.mark(metadata_handle);
                }
            }
        }
    }
}

impl Trace for HandleRef {
    fn trace(&self, tracer: &mut dyn Tracer) {
        tracer.mark(self.target);
    }
}

impl Trace for Listener {
    fn trace(&self, tracer: &mut dyn Tracer) {
        tracer.mark(self.target);
        tracer.mark(self.callback);
    }
}

impl Trace for Value {
    fn trace(&self, tracer: &mut dyn Tracer) {
        match self {
            Value::Cell(cells) => cells.trace(tracer),
            Value::Struct(struct_value) => struct_value.trace(tracer),
            Value::HandleObject(handle) => handle.trace(tracer),
            Value::Listener(listener) => listener.trace(tracer),
            Value::Closure(closure) => closure.trace(tracer),
            Value::Object(object) => object.trace(tracer),
            Value::OutputList(values) => {
                for value in values {
                    value.trace(tracer);
                }
            }
            Value::Int(_)
            | Value::Num(_)
            | Value::Complex(_, _)
            | Value::Bool(_)
            | Value::LogicalArray(_)
            | Value::String(_)
            | Value::StringArray(_)
            | Value::CharArray(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::ComplexTensor(_)
            | Value::Symbolic(_)
            | Value::GpuTensor(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::ClassRef(_)
            | Value::MException(_) => {}
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{}", i.decimal_string()),
            Value::Num(n) => write!(f, "{}", format_number(*n)),
            Value::Complex(re, im) => {
                if *im == 0.0 {
                    write!(f, "{}", format_number(*re))
                } else if *re == 0.0 {
                    write!(f, "{}i", format_number(*im))
                } else if *im < 0.0 {
                    write!(f, "{}-{}i", format_number(*re), format_number(im.abs()))
                } else {
                    write!(f, "{}+{}i", format_number(*re), format_number(*im))
                }
            }
            Value::Bool(b) => write!(f, "{}", if *b { 1 } else { 0 }),
            Value::LogicalArray(la) => write!(f, "{la}"),
            Value::String(s) => write!(f, "'{s}'"),
            Value::StringArray(sa) => write!(f, "{sa}"),
            Value::CharArray(ca) => write!(f, "{ca}"),
            Value::Tensor(m) => write!(f, "{m}"),
            Value::SparseTensor(m) => write!(f, "{m}"),
            Value::ComplexTensor(m) => write!(f, "{m}"),
            Value::Symbolic(expr) => write!(f, "{expr}"),
            Value::Cell(ca) => ca.fmt(f),

            Value::GpuTensor(h) => write!(
                f,
                "GpuTensor(shape={:?}, device={}, buffer={})",
                h.shape, h.device_id, h.buffer_id
            ),
            Value::Object(obj) => write!(f, "{}(props={})", obj.class_name, obj.properties.len()),
            Value::HandleObject(h) => {
                write!(
                    f,
                    "<handle {} @0x{:x} valid={}>",
                    h.class_name,
                    h.target.addr(),
                    h.valid
                )
            }
            Value::Listener(l) => {
                write!(
                    f,
                    "<listener id={} {}@0x{:x} '{}' enabled={} valid={}>",
                    l.id,
                    l.class_name(),
                    l.target.addr(),
                    l.event_name,
                    l.enabled,
                    l.valid
                )
            }
            Value::Struct(st) => {
                write!(f, "struct {{")?;
                for (i, (key, val)) in st.fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, val)?;
                }
                write!(f, "}}")
            }
            Value::OutputList(values) => {
                write!(f, "[")?;
                for (i, value) in values.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", value)?;
                }
                write!(f, "]")
            }
            Value::FunctionHandle(name)
            | Value::ExternalFunctionHandle(name)
            | Value::MethodFunctionHandle(name) => {
                write!(f, "@{name}")
            }
            Value::BoundFunctionHandle { name, .. } => write!(f, "@{name}"),
            Value::Closure(c) => write!(
                f,
                "<closure {} captures={}>",
                c.function_name,
                c.captures.len()
            ),
            Value::ClassRef(name) => write!(f, "<class {name}>"),
            Value::MException(e) => write!(
                f,
                "MException(identifier='{}', message='{}')",
                e.identifier, e.message
            ),
        }
    }
}

impl fmt::Display for ComplexTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.shape.len() {
            0 | 1 => {
                write!(f, "[")?;
                let len = self
                    .integer_data
                    .as_ref()
                    .map_or(self.data.len(), IntegerComplexStorage::len);
                for i in 0..len {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    let s = self.format_element(i);
                    write!(f, "{s}")?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.rows;
                let cols = self.cols;
                write!(f, "[")?;
                for r in 0..rows {
                    for c in 0..cols {
                        if c > 0 {
                            write!(f, " ")?;
                        }
                        let s = self.format_element(r + c * rows);
                        write!(f, "{s}")?;
                    }
                    if r + 1 < rows {
                        write!(f, "; ")?;
                    }
                }
                write!(f, "]")
            }
            _ => {
                if should_expand_nd_display(&self.shape) {
                    write_nd_pages(f, &self.shape, |f, idx| {
                        write!(f, "{}", self.format_element(idx))
                    })
                } else {
                    write!(f, "ComplexTensor(shape={:?})", self.shape)
                }
            }
        }
    }
}

#[cfg(test)]
mod display_tests {
    use super::{
        fmt_rational, format_number, set_display_format, ComplexTensor, FormatMode,
        IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor,
    };

    #[test]
    fn fmt_rational_large_value_with_tiny_fract_does_not_overflow() {
        // abs ~1e15 with a small fractional part: q*n1 would overflow i64 without
        // checked arithmetic.
        let result = std::panic::catch_unwind(|| fmt_rational(1_000_000_000_000_000.000_1));
        assert!(
            result.is_ok(),
            "fmt_rational panicked on large value with tiny fract"
        );

        // Negative counterpart.
        let result = std::panic::catch_unwind(|| fmt_rational(-1_000_000_000_000_000.000_1));
        assert!(
            result.is_ok(),
            "fmt_rational panicked on negative large value with tiny fract"
        );
    }

    #[test]
    fn tensor_nd_display_uses_page_headers() {
        let tensor = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![2, 3, 2],
        )
        .expect("tensor");
        let rendered = tensor.to_string();
        assert!(rendered.contains("(:, :, 1) ="));
        assert!(rendered.contains("(:, :, 2) ="));
        assert!(rendered.contains("  1  0  0"));
    }

    #[test]
    fn dense_integer_tensor_display_uses_exact_storage_values() {
        let vector = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![2],
        )
        .expect("uint64 vector");
        assert_eq!(
            vector.to_string(),
            "[18446744073709551615 9007199254740993]"
        );

        let matrix = Tensor::new_integer(
            IntegerStorage::I64(vec![i64::MIN, -1, 1, i64::MAX]),
            vec![2, 2],
        )
        .expect("int64 matrix");
        let rendered = matrix.to_string();
        assert!(rendered.contains("-9223372036854775808"));
        assert!(rendered.contains("9223372036854775807"));
    }

    #[test]
    fn dense_integer_nd_display_uses_exact_storage_values() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 7, 8]),
            vec![1, 2, 2],
        )
        .expect("uint64 nd tensor");
        let rendered = tensor.to_string();
        assert!(rendered.contains("(:, :, 1) ="));
        assert!(rendered.contains("(:, :, 2) ="));
        assert!(rendered.contains("18446744073709551615"));
        assert!(rendered.contains("9007199254740993"));
    }

    #[test]
    fn tensor_nd_display_falls_back_for_large_arrays() {
        let tensor = Tensor::new(vec![0.0; 4097], vec![1, 1, 4097]).expect("tensor");
        assert_eq!(tensor.to_string(), "Tensor(shape=[1, 1, 4097])");
    }

    #[test]
    fn logical_nd_display_uses_headers_and_fallback_summary() {
        let logical =
            LogicalArray::new(vec![1, 0, 0, 1, 1, 0, 0, 1], vec![2, 2, 2]).expect("logical");
        let rendered = logical.to_string();
        assert!(rendered.contains("(:, :, 1) ="));
        assert!(rendered.contains("(:, :, 2) ="));

        let large = LogicalArray::new(vec![1; 4097], vec![1, 1, 4097]).expect("large logical");
        assert_eq!(large.to_string(), "1x1x4097 logical array");
    }

    #[test]
    fn complex_nd_display_uses_page_headers() {
        let complex = ComplexTensor::new(
            vec![(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)],
            vec![2, 1, 2],
        )
        .expect("complex");
        let rendered = complex.to_string();
        assert!(rendered.contains("(:, :, 1) ="));
        assert!(rendered.contains("(:, :, 2) ="));
    }

    #[test]
    fn typed_complex_integer_display_uses_exact_components() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
            IntegerStorage::U64(vec![7, 0]),
        )
        .expect("matching components");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).expect("typed complex");
        assert_eq!(
            tensor.to_string(),
            format!("[{}+7i {}]", u64::MAX, 1_u64 << 63)
        );

        let negative_imaginary = IntegerComplexStorage::new(
            IntegerStorage::I64(vec![1]),
            IntegerStorage::I64(vec![i64::MIN]),
        )
        .expect("matching components");
        let tensor =
            ComplexTensor::new_integer(negative_imaginary, vec![1, 1]).expect("typed complex");
        assert_eq!(
            tensor.to_string(),
            format!("[1-{}i]", i64::MIN.unsigned_abs())
        );
    }

    #[test]
    fn format_hex_preserves_negative_zero_sign_bit() {
        set_display_format(FormatMode::Hex);
        assert_eq!(format_number(-0.0), "8000000000000000");
        assert_eq!(format_number(0.0), "0000000000000000");
        set_display_format(FormatMode::Short);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellArray {
    pub data: Vec<Value>,
    /// Full MATLAB-visible shape vector (column-major semantics).
    pub shape: Vec<usize>,
    /// Cached row count for 2-D interop; equals `shape[0]` when present.
    pub rows: usize,
    /// Cached column count for 2-D interop; equals `shape[1]` when present, otherwise 1 (or 0 for empty).
    pub cols: usize,
}

impl CellArray {
    pub fn new(data: Vec<Value>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new_with_shape(data, vec![rows, cols])
    }

    pub fn new_with_shape(data: Vec<Value>, shape: Vec<usize>) -> Result<Self, String> {
        let expected = total_len(&shape)
            .ok_or_else(|| "Cell data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Cell data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = shape_rows_cols(&shape);
        Ok(CellArray {
            data,
            shape,
            rows,
            cols,
        })
    }

    pub fn get(&self, row: usize, col: usize) -> Result<Value, String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!(
                "Cell index ({row}, {col}) out of bounds for {}x{} cell array",
                self.rows, self.cols
            ));
        }
        Ok(self.data[row * self.cols + col].clone())
    }
}

fn total_len(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    if shape.is_empty() {
        return (0, 0);
    }
    if shape.len() == 1 {
        return (1, shape[0]);
    }
    (shape[0], shape[1])
}

impl fmt::Display for CellArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        if self.shape.len() > 2 {
            return write!(f, "{} cell array", dims.join("x"));
        }
        write!(f, "{}x{} cell array", self.rows, self.cols)?;
        if self.rows == 0 || self.cols == 0 {
            return Ok(());
        }
        for r in 0..self.rows {
            writeln!(f)?;
            write!(f, "  ")?;
            for c in 0..self.cols {
                if c > 0 {
                    write!(f, "  ")?;
                }
                let value = self.get(r, c).unwrap_or_else(|_| Value::Num(f64::NAN));
                write!(f, "{{{value}}}")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectInstance {
    pub class_name: String,
    pub properties: HashMap<String, Value>,
    pub dynamic_properties: Option<Box<HashMap<String, DynamicPropertyDef>>>,
}

impl ObjectInstance {
    pub fn new(class_name: String) -> Self {
        Self {
            class_name,
            properties: HashMap::new(),
            dynamic_properties: None,
        }
    }

    pub fn is_class(&self, name: &str) -> bool {
        self.class_name == name
    }

    pub fn dynamic_property(&self, name: &str) -> Option<&DynamicPropertyDef> {
        self.dynamic_properties
            .as_ref()
            .and_then(|properties| properties.get(name))
    }

    pub fn dynamic_property_mut(&mut self, name: &str) -> Option<&mut DynamicPropertyDef> {
        self.dynamic_properties
            .as_mut()
            .and_then(|properties| properties.get_mut(name))
    }

    pub fn has_dynamic_property(&self, name: &str) -> bool {
        self.dynamic_property(name).is_some()
    }

    pub fn insert_dynamic_property(
        &mut self,
        name: String,
        property: DynamicPropertyDef,
    ) -> Option<DynamicPropertyDef> {
        self.dynamic_properties
            .get_or_insert_with(|| Box::new(HashMap::new()))
            .insert(name, property)
    }

    pub fn remove_dynamic_property(&mut self, name: &str) -> Option<DynamicPropertyDef> {
        let properties = self.dynamic_properties.as_mut()?;
        let removed = properties.remove(name);
        if properties.is_empty() {
            self.dynamic_properties = None;
        }
        removed
    }

    pub fn dynamic_property_names(&self) -> Vec<String> {
        self.dynamic_properties
            .as_ref()
            .map(|properties| properties.keys().cloned().collect())
            .unwrap_or_default()
    }
}

// -------- Class registry (scaffolding) --------
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    Public,
    Private,
    Protected,
}

#[derive(Debug, Clone)]
pub struct PropertyDef {
    pub name: String,
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub get_access: Access,
    pub set_access: Access,
    pub default_value: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicPropertyDef {
    pub name: String,
    pub defining_class: String,
    pub metadata_handle: Option<GcHandle>,
    pub get_access: Access,
    pub set_access: Access,
    pub dependent: bool,
    pub hidden: bool,
    pub transient: bool,
    pub non_copyable: bool,
    pub abort_set: bool,
    pub set_observable: bool,
    pub get_observable: bool,
    pub description: String,
}

impl DynamicPropertyDef {
    pub fn new(name: String, defining_class: String) -> Self {
        Self {
            name,
            defining_class,
            metadata_handle: None,
            get_access: Access::Public,
            set_access: Access::Public,
            dependent: false,
            hidden: false,
            transient: false,
            non_copyable: false,
            abort_set: false,
            set_observable: false,
            get_observable: false,
            description: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MethodDef {
    pub name: String,
    pub is_static: bool,
    pub is_abstract: bool,
    pub is_sealed: bool,
    pub access: Access,
    pub function_name: String, // bound runtime builtin/user func name
    pub implicit_class_argument: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ClassDef {
    pub name: String, // namespaced e.g. pkg.Point
    pub parent: Option<String>,
    pub properties: HashMap<String, PropertyDef>,
    pub methods: HashMap<String, MethodDef>,
}

thread_local! {
    static CLASS_REGISTRY: RefCell<HashMap<String, ClassDef>> =
        RefCell::new(primitive_class_registry());
    static SEALED_CLASS_REGISTRY: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
    static ABSTRACT_CLASS_REGISTRY: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
    static STATIC_VALUES: RefCell<HashMap<(String, String), Value>> = RefCell::new(HashMap::new());
    static STATIC_VALUE_THREAD_REGISTRATION: StaticValueThreadRegistration =
        const { StaticValueThreadRegistration };
    static ENUMERATION_REGISTRY: RefCell<HashMap<String, HashSet<String>>> =
        RefCell::new(HashMap::new());
}

static STATIC_VALUE_THREADS: once_cell::sync::Lazy<Mutex<HashSet<ThreadId>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashSet::new()));

struct StaticValueThreadRegistration;

impl Drop for StaticValueThreadRegistration {
    fn drop(&mut self) {
        if let Ok(mut threads) = STATIC_VALUE_THREADS.lock() {
            threads.remove(&std::thread::current().id());
        }
    }
}

fn mark_static_values_thread_active() {
    STATIC_VALUE_THREAD_REGISTRATION.with(|_| {});
    if let Ok(mut threads) = STATIC_VALUE_THREADS.lock() {
        threads.insert(std::thread::current().id());
    }
}

pub fn static_property_values_exist_on_other_threads() -> bool {
    let current = std::thread::current().id();
    STATIC_VALUE_THREADS
        .lock()
        .map(|threads| threads.iter().any(|thread_id| *thread_id != current))
        .unwrap_or(false)
}

pub fn static_property_gc_roots() -> Vec<GcHandle> {
    struct RootCollector {
        roots: Vec<GcHandle>,
    }

    impl Tracer for RootCollector {
        fn mark(&mut self, handle: GcHandle) {
            self.roots.push(handle);
        }
    }

    STATIC_VALUES.with(|values| {
        let values = values.borrow();
        let mut collector = RootCollector { roots: Vec::new() };
        for value in values.values() {
            value.trace(&mut collector);
        }
        collector.roots
    })
}

fn primitive_class_registry() -> HashMap<String, ClassDef> {
    let mut registry: HashMap<String, ClassDef> = [
        "double", "single", "logical", "int8", "int16", "int32", "int64", "uint8", "uint16",
        "uint32", "uint64",
    ]
    .into_iter()
    .map(|class_name| {
        let mut methods = HashMap::new();
        methods.insert(
            "zeros".to_string(),
            MethodDef {
                name: "zeros".to_string(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: "zeros".to_string(),
                implicit_class_argument: Some(class_name.to_string()),
            },
        );
        (
            class_name.to_string(),
            ClassDef {
                name: class_name.to_string(),
                parent: None,
                properties: HashMap::new(),
                methods,
            },
        )
    })
    .collect();

    registry.insert(
        "handle".to_string(),
        ClassDef {
            name: "handle".to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        },
    );
    registry.insert(
        "dynamicprops".to_string(),
        ClassDef {
            name: "dynamicprops".to_string(),
            parent: Some("handle".to_string()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        },
    );
    registry.insert(
        "matlab.metadata.Property".to_string(),
        ClassDef {
            name: "matlab.metadata.Property".to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        },
    );
    let mut dynamic_property_methods = HashMap::new();
    dynamic_property_methods.insert(
        "delete".to_string(),
        MethodDef {
            name: "delete".to_string(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: Access::Public,
            function_name: "matlab.metadata.DynamicProperty.delete".to_string(),
            implicit_class_argument: None,
        },
    );
    registry.insert(
        "matlab.metadata.DynamicProperty".to_string(),
        ClassDef {
            name: "matlab.metadata.DynamicProperty".to_string(),
            parent: Some("handle".to_string()),
            properties: HashMap::new(),
            methods: dynamic_property_methods,
        },
    );

    registry
}

pub fn register_class(def: ClassDef) {
    register_class_with_modifiers(def, false, false);
}

pub fn register_class_with_sealed(def: ClassDef, is_sealed: bool) {
    register_class_with_modifiers(def, is_sealed, false);
}

pub fn register_class_with_modifiers(def: ClassDef, is_sealed: bool, is_abstract: bool) {
    let class_name = def.name.clone();
    CLASS_REGISTRY.with(|registry| {
        registry.borrow_mut().insert(class_name.clone(), def);
    });
    SEALED_CLASS_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        if is_sealed {
            registry.insert(class_name.clone());
        } else {
            registry.remove(&class_name);
        }
    });
    ABSTRACT_CLASS_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        if is_abstract {
            registry.insert(class_name.clone());
        } else {
            registry.remove(&class_name);
        }
    });
    ENUMERATION_REGISTRY.with(|registry| {
        registry.borrow_mut().entry(class_name).or_default();
    });
}

pub fn register_class_enumerations(class_name: &str, members: impl IntoIterator<Item = String>) {
    ENUMERATION_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let entry = registry.entry(class_name.to_string()).or_default();
        entry.clear();
        entry.extend(members);
    });
}

pub fn class_has_enumeration_member(class_name: &str, member: &str) -> bool {
    ENUMERATION_REGISTRY.with(|registry| {
        registry
            .borrow()
            .get(class_name)
            .is_some_and(|members| members.contains(member))
    })
}

pub fn get_class(name: &str) -> Option<ClassDef> {
    CLASS_REGISTRY.with(|registry| registry.borrow().get(name).cloned())
}

pub fn class_names() -> Vec<String> {
    CLASS_REGISTRY.with(|registry| registry.borrow().keys().cloned().collect())
}

pub fn is_class_sealed(name: &str) -> bool {
    SEALED_CLASS_REGISTRY.with(|registry| registry.borrow().contains(name))
}

pub fn is_class_abstract(name: &str) -> bool {
    ABSTRACT_CLASS_REGISTRY.with(|registry| registry.borrow().contains(name))
}

pub fn is_class_or_subclass(class_name: &str, ancestor_name: &str) -> bool {
    if class_name == ancestor_name {
        return true;
    }
    CLASS_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let mut current = Some(class_name.to_string());
        let mut visited = std::collections::HashSet::new();
        while let Some(name) = current {
            if !visited.insert(name.clone()) {
                break;
            }
            if name == ancestor_name {
                return true;
            }
            current = registry
                .get(&name)
                .and_then(|class_def| class_def.parent.clone());
        }
        false
    })
}

pub fn superclass_chain(class_name: &str) -> Option<Vec<String>> {
    CLASS_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let mut current = registry
            .get(class_name)
            .and_then(|class_def| class_def.parent.clone());
        let mut visited = std::collections::HashSet::new();
        visited.insert(class_name.to_string());
        let mut supers = Vec::new();
        while let Some(name) = current {
            if !visited.insert(name.clone()) {
                break;
            }
            supers.push(name.clone());
            current = registry
                .get(&name)
                .and_then(|class_def| class_def.parent.clone());
        }
        if registry.contains_key(class_name) {
            Some(supers)
        } else {
            None
        }
    })
}

/// Resolve a property through the inheritance chain, returning the property definition and
/// the name of the class where it was defined.
pub fn lookup_property(class_name: &str, prop: &str) -> Option<(PropertyDef, String)> {
    CLASS_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let mut current = Some(class_name.to_string());
        let mut visited = std::collections::HashSet::new();
        while let Some(name) = current {
            if !visited.insert(name.clone()) {
                break;
            }
            if let Some(cls) = registry.get(&name) {
                if let Some(p) = cls.properties.get(prop) {
                    return Some((p.clone(), name));
                }
                current = cls.parent.clone();
            } else {
                break;
            }
        }
        None
    })
}

/// Resolve a method through the inheritance chain, returning the method definition and
/// the name of the class where it was defined.
pub fn lookup_method(class_name: &str, method: &str) -> Option<(MethodDef, String)> {
    CLASS_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let mut current = Some(class_name.to_string());
        let mut visited = std::collections::HashSet::new();
        while let Some(name) = current {
            if !visited.insert(name.clone()) {
                break;
            }
            if let Some(cls) = registry.get(&name) {
                if let Some(m) = cls.methods.get(method) {
                    return Some((m.clone(), name));
                }
                current = cls.parent.clone();
            } else {
                break;
            }
        }
        None
    })
}

pub fn get_static_property_value(class_name: &str, prop: &str) -> Option<Value> {
    STATIC_VALUES.with(|values| {
        values
            .borrow()
            .get(&(class_name.to_string(), prop.to_string()))
            .cloned()
    })
}

pub fn set_static_property_value(class_name: &str, prop: &str, value: Value) {
    mark_static_values_thread_active();
    STATIC_VALUES.with(|values| {
        values
            .borrow_mut()
            .insert((class_name.to_string(), prop.to_string()), value);
    });
}

/// Set a static property, resolving the defining ancestor class for storage.
pub fn set_static_property_value_in_owner(
    class_name: &str,
    prop: &str,
    value: Value,
) -> Result<(), String> {
    if let Some((_p, owner)) = lookup_property(class_name, prop) {
        set_static_property_value(&owner, prop, value);
        Ok(())
    } else {
        Err(format!("Unknown static property '{class_name}.{prop}'"))
    }
}

#[cfg(test)]
mod class_registry_tests {
    use super::{
        get_class, lookup_method, lookup_property, register_class, superclass_chain, Access,
        ClassDef, MethodDef, PropertyDef,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    #[test]
    fn primitive_classes_expose_static_zeros_method_metadata() {
        for class_name in [
            "double", "single", "logical", "int8", "int16", "int32", "int64", "uint8", "uint16",
            "uint32", "uint64",
        ] {
            let class_def = get_class(class_name).expect("primitive class should be registered");
            let method = class_def
                .methods
                .get("zeros")
                .expect("primitive class should expose zeros static method");
            assert!(method.is_static, "zeros should be static on {class_name}");
            assert_eq!(method.function_name, "zeros");
            assert_eq!(method.implicit_class_argument.as_deref(), Some(class_name));

            let (resolved, owner) =
                lookup_method(class_name, "zeros").expect("lookup should find primitive zeros");
            assert_eq!(owner, class_name);
            assert_eq!(resolved.function_name, "zeros");
            assert_eq!(
                resolved.implicit_class_argument.as_deref(),
                Some(class_name)
            );
        }
    }

    #[test]
    fn superclass_chain_reports_nearest_to_root_order() {
        let grand = unique_class_name("super_chain_grand");
        let parent = unique_class_name("super_chain_parent");
        let child = unique_class_name("super_chain_child");

        register_class(ClassDef {
            name: grand.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        register_class(ClassDef {
            name: parent.clone(),
            parent: Some(grand.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        register_class(ClassDef {
            name: child.clone(),
            parent: Some(parent.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        assert_eq!(
            superclass_chain(&child),
            Some(vec![parent.clone(), grand.clone()])
        );
        assert_eq!(superclass_chain(&grand), Some(Vec::new()));
        assert_eq!(superclass_chain("MissingSuperclassChainClass"), None);
    }

    #[test]
    fn superclass_chain_reports_recorded_parent_when_parent_metadata_is_missing() {
        let child = unique_class_name("super_chain_missing_parent_child");
        let parent = unique_class_name("super_chain_missing_parent_parent");

        register_class(ClassDef {
            name: child.clone(),
            parent: Some(parent.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        assert_eq!(superclass_chain(&child), Some(vec![parent]));
    }

    #[test]
    fn superclass_chain_stops_before_repeating_cycle_start() {
        let first = unique_class_name("super_chain_cycle_first");
        let second = unique_class_name("super_chain_cycle_second");

        register_class(ClassDef {
            name: first.clone(),
            parent: Some(second.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        register_class(ClassDef {
            name: second.clone(),
            parent: Some(first.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        assert_eq!(superclass_chain(&first), Some(vec![second]));
    }

    #[test]
    fn method_lookup_uses_parent_class_metadata_chain() {
        let parent_name = unique_class_name("plan6_parent");
        let child_name = unique_class_name("plan6_child");

        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            "parentOnly".to_string(),
            MethodDef {
                name: "parentOnly".to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: "parentOnly_impl".to_string(),
                implicit_class_argument: None,
            },
        );
        register_class(ClassDef {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        register_class(ClassDef {
            name: child_name.clone(),
            parent: Some(parent_name.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let (method, owner) = lookup_method(&child_name, "parentOnly")
            .expect("child lookup should resolve inherited method through parent metadata");
        assert_eq!(owner, parent_name);
        assert_eq!(method.function_name, "parentOnly_impl");
    }

    #[test]
    fn method_lookup_handles_parent_cycle() {
        let class_a = unique_class_name("plan6_cycle_method_a");
        let class_b = unique_class_name("plan6_cycle_method_b");

        register_class(ClassDef {
            name: class_a.clone(),
            parent: Some(class_b.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        register_class(ClassDef {
            name: class_b.clone(),
            parent: Some(class_a.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        assert!(
            lookup_method(&class_a, "missing").is_none(),
            "cyclic parent metadata should terminate missing method lookup"
        );
    }

    #[test]
    fn property_lookup_uses_parent_class_metadata_chain() {
        let parent_name = unique_class_name("plan6_property_parent");
        let child_name = unique_class_name("plan6_property_child");

        let mut parent_properties = HashMap::new();
        parent_properties.insert(
            "parentFlag".to_string(),
            PropertyDef {
                name: "parentFlag".to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
        register_class(ClassDef {
            name: parent_name.clone(),
            parent: None,
            properties: parent_properties,
            methods: HashMap::new(),
        });
        register_class(ClassDef {
            name: child_name.clone(),
            parent: Some(parent_name.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let (property, owner) = lookup_property(&child_name, "parentFlag")
            .expect("child property lookup should resolve inherited property through parent");
        assert_eq!(owner, parent_name);
        assert_eq!(property.name, "parentFlag");
        assert!(!property.is_static);
    }

    #[test]
    fn property_lookup_handles_parent_cycle() {
        let class_a = unique_class_name("plan6_cycle_property_a");
        let class_b = unique_class_name("plan6_cycle_property_b");

        register_class(ClassDef {
            name: class_a.clone(),
            parent: Some(class_b.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        register_class(ClassDef {
            name: class_b.clone(),
            parent: Some(class_a.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        assert!(
            lookup_property(&class_a, "missing").is_none(),
            "cyclic parent metadata should terminate missing property lookup"
        );
    }
}
