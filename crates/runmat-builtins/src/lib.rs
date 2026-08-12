pub use inventory;
mod catalog_fingerprint;
pub mod symbolic;
pub use catalog_fingerprint::{builtin_catalog_fingerprint, BUILTIN_CATALOG_SCHEMA};
use runmat_gc_api::{GcHandle, Trace, Tracer};
use runmat_value::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::thread::ThreadId;

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
    /// Symbolic array type with optional shape information.
    SymbolicArray {
        /// Optional full shape; None means unknown/dynamic; individual dims can be omitted by using None
        shape: Option<Vec<Option<usize>>>,
    },
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
    /// Scalar or array object type with compatible class and shape facts.
    Object {
        /// Fully qualified class name when statically known.
        class_name: Option<String>,
        /// MATLAB shape; None means dynamic and individual dimensions may be unknown.
        shape: Option<Vec<Option<usize>>>,
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
            (
                Type::Object {
                    class_name: a_class,
                    ..
                },
                Type::Object {
                    class_name: b_class,
                    ..
                },
            ) => {
                a_class.is_none()
                    || b_class.is_none()
                    || a_class
                        .as_ref()
                        .zip(b_class.as_ref())
                        .is_some_and(|(a, b)| {
                            a == b || is_class_or_subclass(a, b) || is_class_or_subclass(b, a)
                        })
            }
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
            (
                Type::Object {
                    class_name: a_class,
                    shape: a_shape,
                },
                Type::Object {
                    class_name: b_class,
                    shape: b_shape,
                },
            ) => {
                let class_name = match (a_class, b_class) {
                    (Some(a), Some(b)) if a == b => Some(a.clone()),
                    (Some(a), Some(b)) if is_class_or_subclass(a, b) => Some(b.clone()),
                    (Some(a), Some(b)) if is_class_or_subclass(b, a) => Some(a.clone()),
                    _ => None,
                };
                Type::Object {
                    class_name,
                    shape: if a_shape == b_shape {
                        a_shape.clone()
                    } else {
                        None
                    },
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
            Value::SymbolicArray(array) => Type::SymbolicArray {
                shape: Some(array.shape.iter().map(|&d| Some(d)).collect()),
            },
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
            Value::Object(object) => Type::Object {
                class_name: Some(object.class_name.clone()),
                shape: Some(vec![Some(1), Some(1)]),
            },
            Value::ObjectArray(array) => Type::Object {
                class_name: Some(array.class_name().to_owned()),
                shape: Some(array.shape().iter().copied().map(Some).collect()),
            },
            Value::HandleObject(handle) => Type::Object {
                class_name: Some(handle.class_name.clone()),
                shape: Some(vec![Some(1), Some(1)]),
            },
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
            Value::Future(_) | Value::Task(_) | Value::Pool(_) | Value::Job(_) => Type::Unknown,
            Value::CharArray(ca) => {
                // Treat as cell of char for type purposes; or a 2-D char matrix conceptually
                Type::Cell {
                    element_type: Some(Box::new(Type::String)),
                    length: Some(ca.data.len()),
                }
            }
            Value::OutputList(values) => {
                Type::OutputList(values.iter().map(Type::from_value).collect())
            }
        }
    }
}

#[cfg(test)]
mod type_tests {
    use super::Type;
    use runmat_value::{ObjectArray, ObjectInstance, Value};

    #[test]
    fn object_array_type_preserves_class_and_shape() {
        let values = vec![
            Value::Object(ObjectInstance::new("pkg.Result".into())),
            Value::Object(ObjectInstance::new("pkg.Result".into())),
        ];
        let array = ObjectArray::new("pkg.Result", values, vec![1, 2]).expect("object array");
        assert!(matches!(
            Type::from_value(&Value::ObjectArray(array)),
            Type::Object {
                class_name: Some(name),
                shape: Some(shape)
            } if name == "pkg.Result" && shape == vec![Some(1), Some(2)]
        ));
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerClass {
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerScalarDoubleRule {
    NotApplicable,
    Allowed,
    AllowedExceptWith64BitInteger,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerInputAvailability {
    Documented,
    RunMatOnly,
    Rejected,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinIntegerInputCapability {
    pub name: &'static str,
    pub classes: &'static [BuiltinIntegerClass],
    pub availability: BuiltinIntegerInputAvailability,
    pub scalar_double: BuiltinIntegerScalarDoubleRule,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerComputationDomain {
    ExactInteger,
    FloatingPoint,
    Predicate,
    Structural,
    FunctionSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerOutputClassRule {
    PreserveInput,
    PreserveNondoubleInput,
    Double,
    Logical,
    OptionDependent,
    NotApplicable,
    FunctionSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerOverflowRule {
    Saturate,
    Error,
    NotApplicable,
    EvidenceOpen,
    FunctionSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerBackendRule {
    HostOnly,
    HostAndGpu,
    GatherFallback,
    GpuRestricted,
    FunctionSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerOverloadKind {
    ScalarOnly,
    ElementwiseShapePreserving,
    SameSizeOrScalar,
    BroadcastCompatible,
    StructuralParameter,
    Multiple,
    FunctionSpecific,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinIntegerCapabilityDescriptor {
    pub form: &'static str,
    pub inputs: &'static [BuiltinIntegerInputCapability],
    pub computation_domain: BuiltinIntegerComputationDomain,
    pub output_class: BuiltinIntegerOutputClassRule,
    pub overflow: BuiltinIntegerOverflowRule,
    pub backend: BuiltinIntegerBackendRule,
    pub overload: BuiltinIntegerOverloadKind,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinIntegerAuditKind {
    AliasOf,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuiltinIntegerAuditDescriptor {
    pub kind: BuiltinIntegerAuditKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_builtin: Option<&'static str>,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinExtensionMode {
    RunMatOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinExtensionDescriptor {
    pub id: &'static str,
    pub mode: BuiltinExtensionMode,
    pub description: &'static str,
    pub error_identifier: Option<&'static str>,
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
    pub extensions: &'static [BuiltinExtensionDescriptor],
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
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
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
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

    pub fn with_extensions(mut self, extensions: &'static [BuiltinExtensionDescriptor]) -> Self {
        self.extensions = extensions;
        self
    }

    pub fn with_integer_capabilities(
        mut self,
        capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    ) -> Self {
        self.integer_capabilities = capabilities;
        self
    }

    pub fn with_integer_audit(
        mut self,
        audit: Option<&'static BuiltinIntegerAuditDescriptor>,
    ) -> Self {
        self.integer_audit = audit;
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
