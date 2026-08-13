pub use inventory;
mod class_declarations;
pub use class_declarations::{standard_class_declaration, standard_class_is_subclass};
pub mod catalog;
pub use catalog::*;
mod catalog_fingerprint;
pub use catalog_fingerprint::{builtin_catalog_fingerprint, BUILTIN_CATALOG_SCHEMA};
pub use runmat_types::{LiteralContext as ResolveContext, LiteralValue};
use runmat_value::*;
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

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
                            a == b
                                || standard_class_is_subclass(a, b)
                                || standard_class_is_subclass(b, a)
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
                    (Some(a), Some(b)) if standard_class_is_subclass(a, b) => Some(b.clone()),
                    (Some(a), Some(b)) if standard_class_is_subclass(b, a) => Some(a.clone()),
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
            Value::Future(_)
            | Value::Task(_)
            | Value::Pool(_)
            | Value::Job(_)
            | Value::Foreign(_) => Type::Unknown,
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

pub fn builtin_name_is_known(name: &str) -> bool {
    builtin_catalog_entry_by_name(name).is_some() || builtin_function_by_name(name).is_some()
}

#[cfg(target_arch = "wasm32")]
pub fn builtin_function_by_name(name: &str) -> Option<&'static BuiltinFunction> {
    wasm_registry::builtin_functions()
        .into_iter()
        .find(|f| f.name.eq_ignore_ascii_case(name))
}

pub fn suppresses_auto_output(name: &str) -> bool {
    builtin_catalog_entry_by_name(name)
        .map(|entry| entry.suppress_auto_output)
        .or_else(|| builtin_function_by_name(name).map(|function| function.suppress_auto_output))
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
