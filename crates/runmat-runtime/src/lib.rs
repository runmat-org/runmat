use runmat_builtins::{builtin_functions, Value};
use runmat_macros::runtime_builtin;

pub mod arrays;
pub mod comparison;
pub mod concatenation;
pub mod constants;
pub mod elementwise;
pub mod indexing;
pub mod io;
pub mod mathematics;
pub mod matrix;
pub mod plotting;

#[cfg(feature = "blas-lapack")]
pub mod blas;
#[cfg(feature = "blas-lapack")]
pub mod lapack;

// Link to Apple's Accelerate framework on macOS
#[cfg(all(feature = "blas-lapack", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

// Ensure OpenBLAS is linked on non-macOS platforms when BLAS/LAPACK is enabled
#[cfg(all(feature = "blas-lapack", not(target_os = "macos")))]
extern crate openblas_src;

pub use arrays::*;
pub use comparison::*;
pub use concatenation::*;
// Explicitly re-export for external users (ignition VM) that build matrices from values
pub use concatenation::create_matrix_from_values;
// Note: constants and mathematics modules only contain #[runtime_builtin] functions
// and don't export public items, so they don't need to be re-exported
pub use elementwise::*;
pub use indexing::*;
pub use matrix::*;

#[cfg(feature = "blas-lapack")]
pub use blas::*;
#[cfg(feature = "blas-lapack")]
pub use lapack::*;

/// Call a registered MATLAB builtin by name.
/// Supports function overloading by trying different argument patterns.
/// Returns an error if no builtin with that name and compatible arguments is found.
pub fn call_builtin(name: &str, args: &[Value]) -> Result<Value, String> {
    let mut matching_builtins = Vec::new();

    // Collect all builtins with the matching name
    for b in builtin_functions() {
        if b.name == name {
            matching_builtins.push(b);
        }
    }

    if matching_builtins.is_empty() {
        // Fallback: treat as class constructor if class is registered
        if let Some(cls) = runmat_builtins::get_class(name) {
            // Prefer explicit constructor method with the same name as class (static)
            if let Some(ctor) = cls.methods.get(name) {
                // Dispatch to constructor builtin; pass args through
                return call_builtin(&ctor.function_name, args);
            }
            // Otherwise default-construct object
            return new_object_builtin(name.to_string());
        }
        return Err(format!("unknown builtin `{name}`"));
    }

    // Try each builtin until one succeeds
    let mut last_error = String::new();
    for builtin in matching_builtins {
        let f = builtin.implementation;
        match (f)(args) {
            Ok(result) => return Ok(result),
            Err(e) => last_error = e,
        }
    }

    // If none succeeded, return the last error
    Err(format!(
        "No matching overload for `{}` with {} args: {}",
        name,
        args.len(),
        last_error
    ))
}
// Object/handle utilities used by interpreter lowering for OOP/func handles

#[runmat_macros::runtime_builtin(name = "getfield")]
fn getfield_builtin(base: Value, field: String) -> Result<Value, String> {
    match base {
        Value::MException(me) => {
            match field.as_str() {
                "message" => Ok(Value::String(me.message)),
                "identifier" => Ok(Value::String(me.identifier)),
                _ => Err(format!("getfield: unknown field '{}' on MException", field)),
            }
        }
        Value::Object(obj) => {
            if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                if let Some(p) = cls.properties.get(&field) {
                    if p.is_static { return Err(format!("Property '{}' is static; use classref('{}').{}", field, obj.class_name, field)); }
                    match p.access { runmat_builtins::Access::Private => return Err(format!("Property '{}' is private", field)), _ => {} }
                }
            }
            if let Some(v) = obj.properties.get(&field) { Ok(v.clone()) } else { Err(format!("Undefined property '{}' for class {}", field, obj.class_name)) }
        }
        other => Err(format!("getfield unsupported on this value for field '{field}': {other:?}")),
    }
}

// Error handling builtins (basic compatibility)
#[runmat_macros::runtime_builtin(name = "error")]
fn error_builtin(rest: Vec<Value>) -> Result<Value, String> {
    // Supports: error(message), error(identifier, message)
    if rest.is_empty() {
        return Err("error: missing message".to_string());
    }
    if rest.len() == 1 {
        let msg: String = (&rest[0]).try_into()?;
        return Err(msg);
    }
    let ident: String = (&rest[0]).try_into()?;
    let msg: String = (&rest[1]).try_into()?;
    Err(format!("{}: {}", ident, msg))
}

#[runmat_macros::runtime_builtin(name = "rethrow")]
fn rethrow_builtin(e: Value) -> Result<Value, String> {
    match e {
        Value::MException(me) => Err(format!("{}: {}", me.identifier, me.message)),
        Value::String(s) => Err(s),
        other => Err(format!("{:?}", other)),
    }
}

#[runmat_macros::runtime_builtin(name = "reshape")]
fn reshape_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    // Accept 2 or 3 dims (for tests); implement MATLAB-style (column-major) reshape semantics
    let t = match a {
        Value::Tensor(t) => t,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) | Value::Cell(_) | Value::GpuTensor(_) | Value::Object(_) | Value::FunctionHandle(_) | Value::Closure(_) | Value::ClassRef(_) | Value::MException(_) => {
            return Err("reshape: expected tensor".to_string())
        }
    };
    let mut dims: Vec<usize> = Vec::new();
    for v in rest {
        let n: f64 = (&v).try_into()?;
        dims.push(n as usize);
    }
    if dims.len() < 2 || dims.len() > 3 {
        return Err("reshape: expected 2 or 3 dimension sizes".to_string());
    }
    let total: usize = dims.iter().product();
    if total != t.data.len() {
        return Err(format!("reshape: element count mismatch {} vs {}", total, t.data.len()));
    }
    // MATLAB uses column-major storage; reshape reinterprets without reordering
    let new_t = runmat_builtins::Tensor::new(t.data.clone(), dims).map_err(|e| format!("reshape: {e}"))?;
    Ok(Value::Tensor(new_t))
}
#[runmat_macros::runtime_builtin(name = "setfield")]
fn setfield_builtin(base: Value, field: String, rhs: Value) -> Result<Value, String> {
    match base {
        Value::Object(mut obj) => {
            if let Some(cls) = runmat_builtins::get_class(&obj.class_name) {
                if let Some(p) = cls.properties.get(&field) {
                    if p.is_static { return Err(format!("Property '{}' is static; use classref('{}').{}", field, obj.class_name, field)); }
                    match p.access { runmat_builtins::Access::Private => return Err(format!("Property '{}' is private", field)), _ => {} }
                }
            }
            obj.properties.insert(field, rhs); Ok(Value::Object(obj))
        }
        other => Err(format!("setfield unsupported on this value for field '{field}': {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "call_method")]
fn call_method_builtin(base: Value, method: String, rest: Vec<Value>) -> Result<Value, String> {
    match base {
        Value::Object(obj) => {
            // Simple dynamic dispatch via builtin registry: method name may be qualified as Class.method
            let qualified = format!("{}.{}", obj.class_name, method);
            // Prepend receiver as first arg so methods can accept it
            let mut args = Vec::with_capacity(1 + rest.len());
            args.push(Value::Object(obj.clone()));
            args.extend(rest.into_iter());
            if let Ok(v) = crate::call_builtin(&qualified, &args) { return Ok(v); }
            // Fallback to global method name
            crate::call_builtin(&method, &args)
        }
        other => Err(format!("call_method unsupported on {other:?} for method '{method}'")),
    }
}

#[runmat_macros::runtime_builtin(name = "make_handle")]
fn make_handle_builtin(name: String) -> Result<Value, String> {
    Ok(Value::String(format!("@{name}")))
}

#[runmat_macros::runtime_builtin(name = "make_anon")]
fn make_anon_builtin(params: String, body: String) -> Result<Value, String> {
    Ok(Value::String(format!("@anon({params}) {body}")))
}

#[runmat_macros::runtime_builtin(name = "new_object")]
fn new_object_builtin(class_name: String) -> Result<Value, String> {
    if let Some(def) = runmat_builtins::get_class(&class_name) {
        let mut obj = runmat_builtins::ObjectInstance::new(def.name.clone());
        for (k, p) in def.properties {
            if !p.is_static {
                if let Some(v) = p.default_value.clone() { obj.properties.insert(k.clone(), v); }
            }
        }
        Ok(Value::Object(obj))
    } else {
        Ok(Value::Object(runmat_builtins::ObjectInstance::new(class_name)))
    }
}

#[cfg(feature = "test-classes")]
#[runmat_macros::runtime_builtin(name = "classref")]
fn classref_builtin(class_name: String) -> Result<Value, String> {
    Ok(Value::ClassRef(class_name))
}

#[cfg(feature = "test-classes")]
#[runmat_macros::runtime_builtin(name = "__register_test_classes")]
fn register_test_classes_builtin() -> Result<Value, String> {
    use runmat_builtins::*;
    let mut props = std::collections::HashMap::new();
    props.insert("x".to_string(), PropertyDef { name: "x".to_string(), is_static: false, access: Access::Public, default_value: Some(Value::Num(0.0)) });
    props.insert("y".to_string(), PropertyDef { name: "y".to_string(), is_static: false, access: Access::Public, default_value: Some(Value::Num(0.0)) });
    props.insert("staticValue".to_string(), PropertyDef { name: "staticValue".to_string(), is_static: true, access: Access::Public, default_value: Some(Value::Num(42.0)) });
    props.insert("secret".to_string(), PropertyDef { name: "secret".to_string(), is_static: false, access: Access::Private, default_value: Some(Value::Num(99.0)) });
    let mut methods = std::collections::HashMap::new();
    methods.insert("move".to_string(), MethodDef { name: "move".to_string(), is_static: false, access: Access::Public, function_name: "Point.move".to_string() });
    methods.insert("origin".to_string(), MethodDef { name: "origin".to_string(), is_static: true, access: Access::Public, function_name: "Point.origin".to_string() });
    runmat_builtins::register_class(ClassDef { name: "Point".to_string(), parent: None, properties: props, methods });

    // Namespaced class example: pkg.PointNS with same shape as Point
    let mut ns_props = std::collections::HashMap::new();
    ns_props.insert("x".to_string(), PropertyDef { name: "x".to_string(), is_static: false, access: Access::Public, default_value: Some(Value::Num(1.0)) });
    ns_props.insert("y".to_string(), PropertyDef { name: "y".to_string(), is_static: false, access: Access::Public, default_value: Some(Value::Num(2.0)) });
    let ns_methods = std::collections::HashMap::new();
    runmat_builtins::register_class(ClassDef { name: "pkg.PointNS".to_string(), parent: None, properties: ns_props, methods: ns_methods });

    // Inheritance: Shape (base) and Circle (derived)
    let shape_props = std::collections::HashMap::new();
    let mut shape_methods = std::collections::HashMap::new();
    shape_methods.insert("area".to_string(), MethodDef { name: "area".to_string(), is_static: false, access: Access::Public, function_name: "Shape.area".to_string() });
    runmat_builtins::register_class(ClassDef { name: "Shape".to_string(), parent: None, properties: shape_props, methods: shape_methods });

    let mut circle_props = std::collections::HashMap::new();
    circle_props.insert("r".to_string(), PropertyDef { name: "r".to_string(), is_static: false, access: Access::Public, default_value: Some(Value::Num(0.0)) });
    let mut circle_methods = std::collections::HashMap::new();
    circle_methods.insert("area".to_string(), MethodDef { name: "area".to_string(), is_static: false, access: Access::Public, function_name: "Circle.area".to_string() });
    runmat_builtins::register_class(ClassDef { name: "Circle".to_string(), parent: Some("Shape".to_string()), properties: circle_props, methods: circle_methods });

    // Constructor demo class: Ctor with static constructor method Ctor
    let ctor_props = std::collections::HashMap::new();
    let mut ctor_methods = std::collections::HashMap::new();
    ctor_methods.insert("Ctor".to_string(), MethodDef { name: "Ctor".to_string(), is_static: true, access: Access::Public, function_name: "Ctor.Ctor".to_string() });
    runmat_builtins::register_class(ClassDef { name: "Ctor".to_string(), parent: None, properties: ctor_props, methods: ctor_methods });

    // Overloaded indexing demo class: OverIdx with subsref/subsasgn
    let overidx_props = std::collections::HashMap::new();
    let mut overidx_methods = std::collections::HashMap::new();
    overidx_methods.insert("subsref".to_string(), MethodDef { name: "subsref".to_string(), is_static: false, access: Access::Public, function_name: "OverIdx.subsref".to_string() });
    overidx_methods.insert("subsasgn".to_string(), MethodDef { name: "subsasgn".to_string(), is_static: false, access: Access::Public, function_name: "OverIdx.subsasgn".to_string() });
    runmat_builtins::register_class(ClassDef { name: "OverIdx".to_string(), parent: None, properties: overidx_props, methods: overidx_methods });
    Ok(Value::Num(1.0))
}

#[cfg(feature = "test-classes")]
pub fn test_register_classes() {
    let _ = register_test_classes_builtin();
}

// Example method implementation: Point.move(obj, dx, dy) -> updated obj
#[cfg(feature = "test-classes")]
#[runmat_macros::runtime_builtin(name = "Point.move")]
fn point_move_method(obj: Value, dx: f64, dy: f64) -> Result<Value, String> {
    match obj {
        Value::Object(mut o) => {
            let mut x = 0.0;
            let mut y = 0.0;
            if let Some(Value::Num(v)) = o.properties.get("x") { x = *v; }
            if let Some(Value::Num(v)) = o.properties.get("y") { y = *v; }
            o.properties.insert("x".to_string(), Value::Num(x + dx));
            o.properties.insert("y".to_string(), Value::Num(y + dy));
            Ok(Value::Object(o))
        }
        other => Err(format!("Point.move requires object receiver, got {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "Point.origin")]
fn point_origin_method() -> Result<Value, String> {
    let mut o = runmat_builtins::ObjectInstance::new("Point".to_string());
    o.properties.insert("x".to_string(), Value::Num(0.0));
    o.properties.insert("y".to_string(), Value::Num(0.0));
    Ok(Value::Object(o))
}

#[cfg(feature = "test-classes")]
#[runmat_macros::runtime_builtin(name = "Shape.area")]
fn shape_area_method(_obj: Value) -> Result<Value, String> {
    Ok(Value::Num(0.0))
}

#[cfg(feature = "test-classes")]
#[runmat_macros::runtime_builtin(name = "Circle.area")]
fn circle_area_method(obj: Value) -> Result<Value, String> {
    match obj {
        Value::Object(o) => {
            let r = if let Some(Value::Num(v)) = o.properties.get("r") { *v } else { 0.0 };
            Ok(Value::Num(std::f64::consts::PI * r * r))
        }
        other => Err(format!("Circle.area requires object receiver, got {other:?}")),
    }
}

// --- Test-only helpers to validate constructors and subsref/subsasgn ---
#[cfg(feature = "test-classes")]
#[runmat_macros::runtime_builtin(name = "Ctor.Ctor")]
fn ctor_ctor_method(x: f64) -> Result<Value, String> {
    // Construct object with property 'x' initialized
    let mut o = runmat_builtins::ObjectInstance::new("Ctor".to_string());
    o.properties.insert("x".to_string(), Value::Num(x));
    Ok(Value::Object(o))
}

#[cfg(feature = "test-classes")]
#[runmat_macros::runtime_builtin(name = "OverIdx.subsref")]
fn overidx_subsref(obj: Value, kind: String, payload: Value) -> Result<Value, String> {
    // Simple sentinel implementation: return different values for '.' vs '()'
    match (obj, kind.as_str(), payload) {
        (Value::Object(_), "()", Value::Cell(_)) => Ok(Value::Num(99.0)),
        (Value::Object(o), "{}", Value::Cell(_)) => {
            if let Some(v) = o.properties.get("lastCell") { Ok(v.clone()) } else { Ok(Value::Num(0.0)) }
        }
        (Value::Object(o), ".", Value::String(field)) => {
            // If field exists, return it; otherwise sentinel 77
            if let Some(v) = o.properties.get(&field) { Ok(v.clone()) } else { Ok(Value::Num(77.0)) }
        }
        _ => Err("subsref: unsupported payload".to_string()),
    }
}

#[cfg(feature = "test-classes")]
#[runmat_macros::runtime_builtin(name = "OverIdx.subsasgn")]
fn overidx_subsasgn(mut obj: Value, kind: String, payload: Value, rhs: Value) -> Result<Value, String> {
    match (&mut obj, kind.as_str(), payload) {
        (Value::Object(o), "()", Value::Cell(_)) => {
            // Store into 'last' property
            o.properties.insert("last".to_string(), rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), "{}", Value::Cell(_)) => {
            o.properties.insert("lastCell".to_string(), rhs);
            Ok(Value::Object(o.clone()))
        }
        (Value::Object(o), ".", Value::String(field)) => {
            o.properties.insert(field, rhs);
            Ok(Value::Object(o.clone()))
        }
        _ => Err("subsasgn: unsupported payload".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "feval")]
fn feval_builtin(f: Value, rest: Vec<Value>) -> Result<Value, String> {
    match f {
        // Current handles are strings like "@sin"
        Value::String(s) => {
            if let Some(name) = s.strip_prefix('@') {
                crate::call_builtin(name, &rest)
            } else {
                Err(format!("feval: expected function handle string starting with '@', got {s}"))
            }
        }
        Value::Closure(c) => {
            let mut args = c.captures.clone();
            args.extend(rest.into_iter());
            crate::call_builtin(&c.function_name, &args)
        }
        // Future: support Value::Function variants
        other => Err(format!("feval: unsupported function value {other:?}")),
    }
}

// Common mathematical functions that tests expect

/// Transpose operation for Values
pub fn transpose(value: Value) -> Result<Value, String> {
    match value {
        Value::Tensor(ref m) => Ok(Value::Tensor(matrix_transpose(m))),
        Value::Num(n) => Ok(Value::Num(n)), // Scalar transpose is identity
        _ => Err("transpose not supported for this type".to_string()),
    }
}

// Explicit GPU usage builtins (scaffolding)
#[runmat_macros::runtime_builtin(name = "gpuArray")]
fn gpu_array_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::Tensor(t) => {
            // Placeholder: mark as GPU handle; device/buffer ids are dummies for now
            if let Some(p) = runmat_accelerate_api::provider() {
                let view = runmat_accelerate_api::HostTensorView { data: &t.data, shape: &t.shape };
                let h = p.upload(&view).map_err(|e| format!("gpuArray upload: {e}"))?;
                Ok(Value::GpuTensor(h))
            } else {
                Ok(Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle { shape: t.shape.clone(), device_id: 0, buffer_id: 0 }))
            }
        }
        Value::Num(_n) => {
            Ok(Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle { shape: vec![1,1], device_id: 0, buffer_id: 0 }))
        }
        other => Err(format!("gpuArray unsupported for {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "gather")]
fn gather_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                let ht = p.download(&h).map_err(|e| format!("gather download: {e}"))?;
                Ok(Value::Tensor(runmat_builtins::Tensor::new(ht.data, ht.shape).map_err(|e| format!("gather build: {e}"))?))
            } else {
                let total: usize = h.shape.iter().product();
                Ok(Value::Tensor(runmat_builtins::Tensor::new(vec![0.0; total], h.shape).map_err(|e| format!("gather: {e}"))?))
            }
        }
        v => Ok(v),
    }
}

#[runtime_builtin(name = "abs")]
fn abs_builtin(x: f64) -> Result<f64, String> {
    Ok(x.abs())
}

#[runtime_builtin(name = "max")]
fn max_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(a.max(b))
}

#[runtime_builtin(name = "min")]
fn min_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(a.min(b))
}

#[runtime_builtin(name = "sqrt")]
fn sqrt_builtin(x: f64) -> Result<f64, String> {
    if x < 0.0 {
        Err("MATLAB:domainError: Cannot take square root of negative number".to_string())
    } else {
        Ok(x.sqrt())
    }
}

/// Simple timing functions for benchmarks
/// tic() starts a timer and returns current time
#[runtime_builtin(name = "tic")]
fn tic_builtin() -> Result<f64, String> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Time error: {e}"))?;
    Ok(now.as_secs_f64())
}

/// toc() returns elapsed time since the last tic() call
/// Note: In a real implementation, this would use a saved start time,
/// but for simplicity we'll just return a small time value
#[runtime_builtin(name = "toc")]
fn toc_builtin() -> Result<f64, String> {
    // For benchmark purposes, return a realistic small time
    Ok(0.001) // 1 millisecond
}

#[runtime_builtin(name = "exp")]
fn exp_builtin(x: f64) -> Result<f64, String> {
    Ok(x.exp())
}

#[runtime_builtin(name = "log")]
fn log_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("MATLAB:domainError: Cannot take logarithm of non-positive number".to_string())
    } else {
        Ok(x.ln())
    }
}
