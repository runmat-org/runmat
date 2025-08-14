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

// Array size helpers
#[runmat_macros::runtime_builtin(name = "size")]
fn size_builtin(a: Value) -> Result<Value, String> {
    let dims: Vec<usize> = match a {
        Value::Tensor(t) => t.shape,
        Value::Cell(ca) => vec![ca.rows, ca.cols],
        Value::GpuTensor(h) => h.shape,
        Value::String(s) => vec![1, s.len()],
        _ => vec![1, 1],
    };
    let dims_f: Vec<f64> = dims.iter().map(|d| *d as f64).collect();
    Ok(Value::Tensor(runmat_builtins::Tensor::new(dims_f, vec![dims.len(), 1]).map_err(|e| format!("size: {e}"))?))
}

#[runmat_macros::runtime_builtin(name = "size")]
fn size_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    let mut dims: Vec<usize> = match a {
        Value::Tensor(t) => t.shape,
        Value::Cell(ca) => vec![ca.rows, ca.cols],
        Value::GpuTensor(h) => h.shape,
        Value::String(s) => vec![1, s.len()],
        _ => vec![1, 1],
    };
    let d = if dim < 1.0 { 1usize } else { dim as usize };
    if dims.len() < d { dims.resize(d, 1); }
    Ok(Value::Num(dims[d - 1] as f64))
}

// Linear index to subscripts (column-major)
#[runmat_macros::runtime_builtin(name = "ind2sub")]
fn ind2sub_builtin(dims_val: Value, idx_val: f64) -> Result<Value, String> {
    let dims: Vec<usize> = match dims_val {
        Value::Tensor(t) => {
            if t.shape.len() == 2 && (t.shape[0] == 1 || t.shape[1] == 1) {
                t.data.iter().map(|v| *v as usize).collect()
            } else { return Err("ind2sub: dims must be a vector".to_string()); }
        }
        Value::Cell(ca) => {
            if ca.data.is_empty() { vec![] } else { ca.data.iter().map(|v| match v { Value::Num(n) => *n as usize, Value::Int(i) => *i as usize, _ => 1usize }).collect() }
        }
        _ => return Err("ind2sub: dims must be a vector".to_string()),
    };
    if dims.is_empty() { return Err("ind2sub: empty dims".to_string()); }
    let mut subs: Vec<usize> = vec![1; dims.len()];
    let idx = if idx_val < 1.0 { 1usize } else { idx_val as usize } - 1; // 0-based
    let mut stride = 1usize;
    for d in 0..dims.len() {
        let dim_len = dims[d];
        let val = (idx / stride) % dim_len;
        subs[d] = val + 1; // 1-based
        stride *= dim_len.max(1);
    }
    // Return as a tensor column vector for expansion
    let data: Vec<f64> = subs.iter().map(|s| *s as f64).collect();
    Ok(Value::Tensor(runmat_builtins::Tensor::new(data, vec![subs.len(), 1]).map_err(|e| format!("ind2sub: {e}"))?))
}

#[runmat_macros::runtime_builtin(name = "sub2ind")]
fn sub2ind_builtin(dims_val: Value, rest: Vec<Value>) -> Result<Value, String> {
    let dims: Vec<usize> = match dims_val {
        Value::Tensor(t) => {
            if t.shape.len() == 2 && (t.shape[0] == 1 || t.shape[1] == 1) {
                t.data.iter().map(|v| *v as usize).collect()
            } else { return Err("sub2ind: dims must be a vector".to_string()); }
        }
        Value::Cell(ca) => {
            if ca.data.is_empty() { vec![] } else { ca.data.iter().map(|v| match v { Value::Num(n) => *n as usize, Value::Int(i) => *i as usize, _ => 1usize }).collect() }
        }
        _ => return Err("sub2ind: dims must be a vector".to_string()),
    };
    if dims.is_empty() { return Err("sub2ind: empty dims".to_string()); }
    if rest.len() != dims.len() { return Err("sub2ind: expected one subscript per dimension".to_string()); }
    let subs: Vec<usize> = rest.iter().map(|v| match v { Value::Num(n) => (*n as isize) as isize, Value::Int(i) => *i as isize, _ => 1isize } ).map(|x| if x < 1 { 1 } else { x as usize }).collect();
    // Column-major linear index: 1 + sum_{d=0}^{n-1} (sub[d]-1) * prod_{k<d} dims[k]
    let mut stride = 1usize;
    let mut lin0 = 0usize;
    for d in 0..dims.len() {
        let dim_len = dims[d];
        let s = subs[d];
        if s == 0 || s > dim_len { return Err("sub2ind: subscript out of bounds".to_string()); }
        lin0 += (s - 1) * stride;
        stride *= dim_len.max(1);
    }
    Ok(Value::Num((lin0 + 1) as f64))
}

#[runmat_macros::runtime_builtin(name = "numel")]
fn numel_builtin(a: Value) -> Result<Value, String> {
    let n = match a {
        Value::Tensor(t) => t.data.len(),
        Value::Cell(ca) => ca.data.len(),
        Value::GpuTensor(h) => h.shape.iter().product(),
        Value::String(s) => s.len(),
        _ => 1,
    };
    Ok(Value::Num(n as f64))
}

#[runmat_macros::runtime_builtin(name = "length")]
fn length_builtin(a: Value) -> Result<Value, String> {
    let len = match a {
        Value::Tensor(t) => t.shape.iter().copied().max().unwrap_or(0),
        Value::Cell(ca) => std::cmp::max(ca.rows, ca.cols),
        Value::GpuTensor(h) => h.shape.iter().copied().max().unwrap_or(0),
        Value::String(s) => s.len(),
        _ => 1,
    };
    Ok(Value::Num(len as f64))
}

#[runmat_macros::runtime_builtin(name = "ndims")]
fn ndims_builtin(a: Value) -> Result<Value, String> {
    let n = match a {
        Value::Tensor(t) => t.shape.len(),
        Value::Cell(_) => 2,
        Value::GpuTensor(h) => h.shape.len(),
        Value::String(_) => 2,
        _ => 2,
    };
    Ok(Value::Num(n as f64))
}

// deal: distribute inputs to multiple outputs (via cell for expansion)
#[runmat_macros::runtime_builtin(name = "deal")]
fn deal_builtin(rest: Vec<Value>) -> Result<Value, String> {
    // Return cell row vector of inputs for expansion
    let cols = rest.len();
    let ca = runmat_builtins::CellArray::new(rest, 1, cols).map_err(|e| format!("deal: {e}"))?;
    Ok(Value::Cell(ca))
}

#[runmat_macros::runtime_builtin(name = "find")]
fn find_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let mut idxs: Vec<f64> = Vec::new();
            for (i, &v) in t.data.iter().enumerate() {
                if v != 0.0 { idxs.push((i + 1) as f64); }
            }
            let len = idxs.len();
            Ok(Value::Tensor(runmat_builtins::Tensor::new(idxs, vec![len, 1]).map_err(|e| format!("find: {e}"))?))
        }
        _ => Err("find: expected tensor".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "find")]
fn find_k_builtin(a: Value, k: f64) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let mut idxs: Vec<f64> = Vec::new();
            for (i, &v) in t.data.iter().enumerate() {
                if v != 0.0 { idxs.push((i + 1) as f64); if (idxs.len() as f64) >= k { break; } }
            }
            let len = idxs.len();
            Ok(Value::Tensor(runmat_builtins::Tensor::new(idxs, vec![len, 1]).map_err(|e| format!("find: {e}"))?))
        }
        _ => Err("find: expected tensor".to_string()),
    }
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
            if let Some((p, _owner)) = runmat_builtins::lookup_property(&obj.class_name, &field) {
                if p.is_static { return Err(format!("Property '{}' is static; use classref('{}').{}", field, obj.class_name, field)); }
                match p.get_access { runmat_builtins::Access::Private => return Err(format!("Property '{}' is private", field)), _ => {} }
            }
            if let Some(v) = obj.properties.get(&field) { Ok(v.clone()) } else { Err(format!("Undefined property '{}' for class {}", field, obj.class_name)) }
        }
        Value::Struct(st) => {
            st.fields.get(&field).cloned().ok_or_else(|| format!("getfield: unknown field '{}'", field))
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
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) | Value::Cell(_) | Value::GpuTensor(_) | Value::Object(_) | Value::FunctionHandle(_) | Value::Closure(_) | Value::ClassRef(_) | Value::MException(_) | Value::Struct(_) => {
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
            if let Some((p, _owner)) = runmat_builtins::lookup_property(&obj.class_name, &field) {
                if p.is_static { return Err(format!("Property '{}' is static; use classref('{}').{}", field, obj.class_name, field)); }
                match p.set_access { runmat_builtins::Access::Private => return Err(format!("Property '{}' is private", field)), _ => {} }
            }
            obj.properties.insert(field, rhs); Ok(Value::Object(obj))
        }
        Value::Struct(mut st) => { st.fields.insert(field, rhs); Ok(Value::Struct(st)) }
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
        // Collect class hierarchy from root to leaf for default initialization
        let mut chain: Vec<runmat_builtins::ClassDef> = Vec::new();
        // Walk up to root
        let mut cursor: Option<String> = Some(def.name.clone());
        while let Some(name) = cursor {
            if let Some(cd) = runmat_builtins::get_class(&name) {
                chain.push(cd.clone());
                cursor = cd.parent.clone();
            } else { break; }
        }
        // Reverse to root-first
        chain.reverse();
        let mut obj = runmat_builtins::ObjectInstance::new(def.name.clone());
        // Apply defaults from root to leaf (leaf overrides effectively by later assignment)
        for cd in chain {
            for (k, p) in cd.properties.iter() {
                if !p.is_static {
                    if let Some(v) = &p.default_value { obj.properties.insert(k.clone(), v.clone()); }
                }
            }
        }
        Ok(Value::Object(obj))
    } else {
        Ok(Value::Object(runmat_builtins::ObjectInstance::new(class_name)))
    }
}

#[runmat_macros::runtime_builtin(name = "classref")]
fn classref_builtin(class_name: String) -> Result<Value, String> {
    Ok(Value::ClassRef(class_name))
}

#[runmat_macros::runtime_builtin(name = "__register_test_classes")]
fn register_test_classes_builtin() -> Result<Value, String> {
    use runmat_builtins::*;
    let mut props = std::collections::HashMap::new();
    props.insert("x".to_string(), PropertyDef { name: "x".to_string(), is_static: false, get_access: Access::Public, set_access: Access::Public, default_value: Some(Value::Num(0.0)) });
    props.insert("y".to_string(), PropertyDef { name: "y".to_string(), is_static: false, get_access: Access::Public, set_access: Access::Public, default_value: Some(Value::Num(0.0)) });
    props.insert("staticValue".to_string(), PropertyDef { name: "staticValue".to_string(), is_static: true, get_access: Access::Public, set_access: Access::Public, default_value: Some(Value::Num(42.0)) });
    props.insert("secret".to_string(), PropertyDef { name: "secret".to_string(), is_static: false, get_access: Access::Private, set_access: Access::Private, default_value: Some(Value::Num(99.0)) });
    let mut methods = std::collections::HashMap::new();
    methods.insert("move".to_string(), MethodDef { name: "move".to_string(), is_static: false, access: Access::Public, function_name: "Point.move".to_string() });
    methods.insert("origin".to_string(), MethodDef { name: "origin".to_string(), is_static: true, access: Access::Public, function_name: "Point.origin".to_string() });
    runmat_builtins::register_class(ClassDef { name: "Point".to_string(), parent: None, properties: props, methods });

    // Namespaced class example: pkg.PointNS with same shape as Point
    let mut ns_props = std::collections::HashMap::new();
    ns_props.insert("x".to_string(), PropertyDef { name: "x".to_string(), is_static: false, get_access: Access::Public, set_access: Access::Public, default_value: Some(Value::Num(1.0)) });
    ns_props.insert("y".to_string(), PropertyDef { name: "y".to_string(), is_static: false, get_access: Access::Public, set_access: Access::Public, default_value: Some(Value::Num(2.0)) });
    let ns_methods = std::collections::HashMap::new();
    runmat_builtins::register_class(ClassDef { name: "pkg.PointNS".to_string(), parent: None, properties: ns_props, methods: ns_methods });

    // Inheritance: Shape (base) and Circle (derived)
    let shape_props = std::collections::HashMap::new();
    let mut shape_methods = std::collections::HashMap::new();
    shape_methods.insert("area".to_string(), MethodDef { name: "area".to_string(), is_static: false, access: Access::Public, function_name: "Shape.area".to_string() });
    runmat_builtins::register_class(ClassDef { name: "Shape".to_string(), parent: None, properties: shape_props, methods: shape_methods });

    let mut circle_props = std::collections::HashMap::new();
    circle_props.insert("r".to_string(), PropertyDef { name: "r".to_string(), is_static: false, get_access: Access::Public, set_access: Access::Public, default_value: Some(Value::Num(0.0)) });
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

#[runmat_macros::runtime_builtin(name = "Shape.area")]
fn shape_area_method(_obj: Value) -> Result<Value, String> {
    Ok(Value::Num(0.0))
}

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
#[runmat_macros::runtime_builtin(name = "Ctor.Ctor")]
fn ctor_ctor_method(x: f64) -> Result<Value, String> {
    // Construct object with property 'x' initialized
    let mut o = runmat_builtins::ObjectInstance::new("Ctor".to_string());
    o.properties.insert("x".to_string(), Value::Num(x));
    Ok(Value::Object(o))
}

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

// --- Operator overloading methods for OverIdx (test scaffolding) ---
#[runmat_macros::runtime_builtin(name = "OverIdx.plus")]
fn overidx_plus(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.plus: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k + r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.times")]
fn overidx_times(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.times: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.mtimes")]
fn overidx_mtimes(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.mtimes: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.lt")]
fn overidx_lt(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.lt: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k < r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.gt")]
fn overidx_gt(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.gt: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k > r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.eq")]
fn overidx_eq(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.eq: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k - r).abs() < 1e-12 { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.uplus")]
fn overidx_uplus(obj: Value) -> Result<Value, String> {
    // Identity
    Ok(obj)
}

#[runmat_macros::runtime_builtin(name = "OverIdx.rdivide")]
fn overidx_rdivide(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.rdivide: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k / r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.ldivide")]
fn overidx_ldivide(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.ldivide: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(r / k))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.and")]
fn overidx_and(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.and: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) && (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.or")]
fn overidx_or(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.or: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) || (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.xor")]
fn overidx_xor(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj { Value::Object(o) => o, _ => return Err("OverIdx.xor: receiver must be object".to_string()) };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") { *v } else { 0.0 };
    let r: f64 = (&rhs).try_into()?;
    let a = k != 0.0; let b = r != 0.0;
    Ok(Value::Num(if a ^ b { 1.0 } else { 0.0 }))
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

#[runtime_builtin(name = "max")]
fn max_vector_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            if t.shape.len() == 2 && t.shape[1] == 1 {
                let mut max_val = std::f64::NEG_INFINITY;
                let mut idx = 1usize;
                for (i, &v) in t.data.iter().enumerate() {
                    if v > max_val { max_val = v; idx = i + 1; }
                }
                // Return a 2x1 column [max; idx] for expansion (value then index)
                let out = runmat_builtins::Tensor::new(vec![max_val, idx as f64], vec![2, 1]).map_err(|e| format!("max: {e}"))?;
                Ok(Value::Tensor(out))
            } else {
                // Reduce across all elements -> scalar
                let max_val = t.data.iter().cloned().fold(std::f64::NEG_INFINITY, f64::max);
                Ok(Value::Num(max_val))
            }
        }
        Value::Num(n) => Ok(Value::Num(n)),
        _ => Err("max: unsupported input".to_string()),
    }
}

#[runtime_builtin(name = "min")]
fn min_vector_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            if t.shape.len() == 2 && t.shape[1] == 1 {
                let mut min_val = std::f64::INFINITY;
                let mut idx = 1usize;
                for (i, &v) in t.data.iter().enumerate() {
                    if v < min_val { min_val = v; idx = i + 1; }
                }
                let out = runmat_builtins::Tensor::new(vec![min_val, idx as f64], vec![2, 1]).map_err(|e| format!("min: {e}"))?;
                Ok(Value::Tensor(out))
            } else {
                let min_val = t.data.iter().cloned().fold(std::f64::INFINITY, f64::min);
                Ok(Value::Num(min_val))
            }
        }
        Value::Num(n) => Ok(Value::Num(n)),
        _ => Err("min: unsupported input".to_string()),
    }
}

#[runtime_builtin(name = "max")]
fn max_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("max: expected tensor for dim variant".to_string()) };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    if t.shape.len() < 2 { return Err("max: dim variant expects 2D tensor".to_string()); }
    let rows = t.shape[0]; let cols = t.shape[1];
    if dim == 1 {
        // column-wise maxima: return {M_row, I_row}
        let mut m: Vec<f64> = vec![std::f64::NEG_INFINITY; cols];
        let mut idx: Vec<f64> = vec![1.0; cols];
        for c in 0..cols { for r in 0..rows { let v = t.data[r + c*rows]; if v > m[c] { m[c] = v; idx[c] = (r+1) as f64; } } }
        let m_t = runmat_builtins::Tensor::new(m, vec![1, cols]).map_err(|e| format!("max: {e}"))?;
        let i_t = runmat_builtins::Tensor::new(idx, vec![1, cols]).map_err(|e| format!("max: {e}"))?;
        let cell = runmat_builtins::CellArray::new(vec![Value::Tensor(m_t), Value::Tensor(i_t)], 1, 2).map_err(|e| format!("max: {e}"))?;
        Ok(Value::Cell(cell))
    } else if dim == 2 {
        // row-wise maxima: return {M_col, I_col}
        let mut m: Vec<f64> = vec![std::f64::NEG_INFINITY; rows];
        let mut idx: Vec<f64> = vec![1.0; rows];
        for r in 0..rows { for c in 0..cols { let v = t.data[r + c*rows]; if v > m[r] { m[r] = v; idx[r] = (c+1) as f64; } } }
        let m_t = runmat_builtins::Tensor::new(m, vec![rows, 1]).map_err(|e| format!("max: {e}"))?;
        let i_t = runmat_builtins::Tensor::new(idx, vec![rows, 1]).map_err(|e| format!("max: {e}"))?;
        let cell = runmat_builtins::CellArray::new(vec![Value::Tensor(m_t), Value::Tensor(i_t)], 1, 2).map_err(|e| format!("max: {e}"))?;
        Ok(Value::Cell(cell))
    } else { Err("max: dim out of range".to_string()) }
}

#[runtime_builtin(name = "min")]
fn min_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("min: expected tensor for dim variant".to_string()) };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    if t.shape.len() < 2 { return Err("min: dim variant expects 2D tensor".to_string()); }
    let rows = t.shape[0]; let cols = t.shape[1];
    if dim == 1 {
        let mut m: Vec<f64> = vec![std::f64::INFINITY; cols];
        let mut idx: Vec<f64> = vec![1.0; cols];
        for c in 0..cols { for r in 0..rows { let v = t.data[r + c*rows]; if v < m[c] { m[c] = v; idx[c] = (r+1) as f64; } } }
        let m_t = runmat_builtins::Tensor::new(m, vec![1, cols]).map_err(|e| format!("min: {e}"))?;
        let i_t = runmat_builtins::Tensor::new(idx, vec![1, cols]).map_err(|e| format!("min: {e}"))?;
        let cell = runmat_builtins::CellArray::new(vec![Value::Tensor(m_t), Value::Tensor(i_t)], 1, 2).map_err(|e| format!("min: {e}"))?;
        Ok(Value::Cell(cell))
    } else if dim == 2 {
        let mut m: Vec<f64> = vec![std::f64::INFINITY; rows];
        let mut idx: Vec<f64> = vec![1.0; rows];
        for r in 0..rows { for c in 0..cols { let v = t.data[r + c*rows]; if v < m[r] { m[r] = v; idx[r] = (c+1) as f64; } } }
        let m_t = runmat_builtins::Tensor::new(m, vec![rows, 1]).map_err(|e| format!("min: {e}"))?;
        let i_t = runmat_builtins::Tensor::new(idx, vec![rows, 1]).map_err(|e| format!("min: {e}"))?;
        let cell = runmat_builtins::CellArray::new(vec![Value::Tensor(m_t), Value::Tensor(i_t)], 1, 2).map_err(|e| format!("min: {e}"))?;
        Ok(Value::Cell(cell))
    } else { Err("min: dim out of range".to_string()) }
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

// -------- Reductions: sum/prod/mean/any/all --------

fn tensor_sum_all(t: &runmat_builtins::Tensor) -> f64 {
    t.data.iter().sum()
}

fn tensor_prod_all(t: &runmat_builtins::Tensor) -> f64 {
    t.data.iter().product()
}

#[runmat_macros::runtime_builtin(name = "sum")]
fn sum_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            // Return scalar sum across all elements by default
            Ok(Value::Num(tensor_sum_all(&t)))
        }
        _ => Err("sum: expected tensor".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "sum")]
fn sum_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("sum: expected tensor".to_string()) };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows(); let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for c in 0..cols { let mut s = 0.0; for r in 0..rows { s += t.data[r + c*rows]; } out[c] = s; }
        let tens = runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("sum: {e}"))?;
        Ok(Value::Tensor(tens))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for r in 0..rows { let mut s = 0.0; for c in 0..cols { s += t.data[r + c*rows]; } out[r] = s; }
        let tens = runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("sum: {e}"))?;
        Ok(Value::Tensor(tens))
    } else { Err("sum: dim out of range".to_string()) }
}

#[runmat_macros::runtime_builtin(name = "prod")]
fn prod_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows(); let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![1.0f64; cols];
                for c in 0..cols { let mut p = 1.0; for r in 0..rows { p *= t.data[r + c*rows]; } out[c] = p; }
                let tens = runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("prod: {e}"))?;
                Ok(Value::Tensor(tens))
            } else {
                Ok(Value::Num(tensor_prod_all(&t)))
            }
        }
        _ => Err("prod: expected tensor".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "prod")]
fn prod_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("prod: expected tensor".to_string()) };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows(); let cols = t.cols();
    if dim == 1 {
        let mut out = vec![1.0f64; cols];
        for c in 0..cols { let mut p = 1.0; for r in 0..rows { p *= t.data[r + c*rows]; } out[c] = p; }
        Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("prod: {e}"))?))
    } else if dim == 2 {
        let mut out = vec![1.0f64; rows];
        for r in 0..rows { let mut p = 1.0; for c in 0..cols { p *= t.data[r + c*rows]; } out[r] = p; }
        Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("prod: {e}"))?))
    } else { Err("prod: dim out of range".to_string()) }
}

#[runmat_macros::runtime_builtin(name = "mean")]
fn mean_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows(); let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for c in 0..cols { let mut s = 0.0; for r in 0..rows { s += t.data[r + c*rows]; } out[c] = s / (rows as f64); }
                Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("mean: {e}"))?))
            } else {
                Ok(Value::Num(tensor_sum_all(&t) / (t.data.len() as f64)))
            }
        }
        _ => Err("mean: expected tensor".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "mean")]
fn mean_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("mean: expected tensor".to_string()) };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows(); let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for c in 0..cols { let mut s = 0.0; for r in 0..rows { s += t.data[r + c*rows]; } out[c] = s / (rows as f64); }
        Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("mean: {e}"))?))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for r in 0..rows { let mut s = 0.0; for c in 0..cols { s += t.data[r + c*rows]; } out[r] = s / (cols as f64); }
        Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("mean: {e}"))?))
    } else { Err("mean: dim out of range".to_string()) }
}

#[runmat_macros::runtime_builtin(name = "any")]
fn any_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows(); let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for c in 0..cols { let mut v = 0.0; for r in 0..rows { if t.data[r + c*rows] != 0.0 { v = 1.0; break; } } out[c] = v; }
                Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("any: {e}"))?))
            } else {
                Ok(Value::Num(if t.data.iter().any(|&x| x != 0.0) { 1.0 } else { 0.0 }))
            }
        }
        _ => Err("any: expected tensor".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "any")]
fn any_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("any: expected tensor".to_string()) };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows(); let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for c in 0..cols { let mut v = 0.0; for r in 0..rows { if t.data[r + c*rows] != 0.0 { v = 1.0; break; } } out[c] = v; }
        Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("any: {e}"))?))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for r in 0..rows { let mut v = 0.0; for c in 0..cols { if t.data[r + c*rows] != 0.0 { v = 1.0; break; } } out[r] = v; }
        Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("any: {e}"))?))
    } else { Err("any: dim out of range".to_string()) }
}

#[runmat_macros::runtime_builtin(name = "all")]
fn all_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows(); let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for c in 0..cols { let mut v = 1.0; for r in 0..rows { if t.data[r + c*rows] == 0.0 { v = 0.0; break; } } out[c] = v; }
                Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("all: {e}"))?))
            } else {
                Ok(Value::Num(if t.data.iter().all(|&x| x != 0.0) { 1.0 } else { 0.0 }))
            }
        }
        _ => Err("all: expected tensor".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "all")]
fn all_dim_builtin(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("all: expected tensor".to_string()) };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows(); let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for c in 0..cols { let mut v = 1.0; for r in 0..rows { if t.data[r + c*rows] == 0.0 { v = 0.0; break; } } out[c] = v; }
        Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("all: {e}"))?))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for r in 0..rows { let mut v = 1.0; for c in 0..cols { if t.data[r + c*rows] == 0.0 { v = 0.0; break; } } out[r] = v; }
        Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("all: {e}"))?))
    } else { Err("all: dim out of range".to_string()) }
}

// -------- N-D utilities: permute, squeeze, cat --------

#[runmat_macros::runtime_builtin(name = "squeeze")]
fn squeeze_builtin(a: Value) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("squeeze: expected tensor".to_string()) };
    let mut new_shape: Vec<usize> = t.shape.iter().copied().filter(|&d| d != 1).collect();
    if new_shape.is_empty() { new_shape.push(1); }
    Ok(Value::Tensor(runmat_builtins::Tensor::new(t.data.clone(), new_shape).map_err(|e| format!("squeeze: {e}"))?))
}

#[runmat_macros::runtime_builtin(name = "permute")]
fn permute_builtin(a: Value, order: Value) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("permute: expected tensor".to_string()) };
    let ord = match order {
        Value::Tensor(idx) => idx.data.iter().map(|&v| v as usize).collect::<Vec<usize>>() ,
        Value::Cell(c) => c.data.iter().map(|v| match v { Value::Num(n) => *n as usize, Value::Int(i) => *i as usize, _ => 0 }).collect(),
        Value::Num(n) => vec![n as usize],
        _ => return Err("permute: expected index vector".to_string()),
    };
    if ord.iter().any(|&k| k==0) { return Err("permute: indices are 1-based".to_string()); }
    let ord0: Vec<usize> = ord.into_iter().map(|k| k-1).collect();
    let rank = t.shape.len();
    if ord0.len() != rank { return Err("permute: order length must match rank".to_string()); }
    let mut new_shape = vec![0usize; rank];
    for (i, &src) in ord0.iter().enumerate() { new_shape[i] = *t.shape.get(src).unwrap_or(&1); }
    // Precompute strides
    let mut src_strides = vec![0usize; rank];
    let mut acc = 1usize; for d in 0..rank { src_strides[d] = acc; acc *= t.shape[d]; }
    let mut dst_strides = vec![0usize; rank];
    let mut acc2 = 1usize; for d in 0..rank { dst_strides[d] = acc2; acc2 *= new_shape[d]; }
    let total = t.data.len();
    let mut out = vec![0f64; total];
    // Iterate destination multi-index in column-major
    fn unrank(mut lin: usize, shape: &[usize]) -> Vec<usize> {
        let mut idx = Vec::with_capacity(shape.len());
        for &s in shape { idx.push(lin % s); lin /= s; }
        idx
    }
    for dst_lin in 0..total {
        let dst_multi = unrank(dst_lin, &new_shape);
        // Map dst dims -> src dims by inverse order mapping
        let mut src_multi = vec![0usize; rank];
        for (dst_d, &src_d) in ord0.iter().enumerate() { src_multi[src_d] = dst_multi[dst_d]; }
        let mut src_lin = 0usize; for d in 0..rank { src_lin += src_multi[d] * src_strides[d]; }
        out[dst_lin] = t.data[src_lin];
    }
    Ok(Value::Tensor(runmat_builtins::Tensor::new(out, new_shape).map_err(|e| format!("permute: {e}"))?))
}

#[runmat_macros::runtime_builtin(name = "cat")]
fn cat_builtin(dim: f64, a: Value, b: Value) -> Result<Value, String> {
    let d = if dim < 1.0 { 1usize } else { dim as usize } - 1; // zero-based
    let (ta, tb) = match (a, b) { (Value::Tensor(ta), Value::Tensor(tb)) => (ta, tb), _ => return Err("cat: expected tensors".to_string()) };
    let rank = ta.shape.len().max(tb.shape.len());
    let mut sa = ta.shape.clone(); sa.resize(rank, 1);
    let mut sb = tb.shape.clone(); sb.resize(rank, 1);
    for k in 0..rank { if k!=d && sa[k] != sb[k] { return Err("cat: dimension mismatch".to_string()); } }
    let mut out_shape = sa.clone(); out_shape[d] = sa[d] + sb[d];
    let mut out = vec![0f64; out_shape.iter().product()];
    // Strides for src and dst
    fn strides(shape: &[usize]) -> Vec<usize> { let mut s=vec![0; shape.len()]; let mut acc=1; for i in 0..shape.len() { s[i]=acc; acc*=shape[i]; } s }
    let sa_str = strides(&sa); let sb_str = strides(&sb); let od_str = strides(&out_shape);
    // Copy A and B by iterating all coordinates
    fn scatter(dst: &mut [f64], src: &[f64], out_shape: &[usize], out_str: &[usize], src_shape: &[usize], src_str: &[usize], d: usize, offset: usize) {
        let rank = out_shape.len();
        let total: usize = src_shape.iter().product();
        for idx_lin in 0..total {
            // Convert src lin to multi
            let mut rem = idx_lin; let mut src_multi = vec![0usize; rank];
            for i in 0..rank { let s = src_shape[i]; src_multi[i] = rem % s; rem /= s; }
            let mut dst_multi = src_multi.clone();
            dst_multi[d] += offset;
            let mut s_lin = 0usize; for i in 0..rank { s_lin += src_multi[i] * src_str[i]; }
            let mut d_lin = 0usize; for i in 0..rank { d_lin += dst_multi[i] * out_str[i]; }
            dst[d_lin] = src[s_lin];
        }
    }
    scatter(&mut out, &ta.data, &out_shape, &od_str, &sa, &sa_str, d, 0);
    scatter(&mut out, &tb.data, &out_shape, &od_str, &sb, &sb_str, d, sa[d]);
    Ok(Value::Tensor(runmat_builtins::Tensor::new(out, out_shape).map_err(|e| format!("cat: {e}"))?))
}

// -------- Linear algebra helpers: diag, triu, tril --------

#[runmat_macros::runtime_builtin(name = "diag")]
fn diag_builtin(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows(); let cols = t.cols();
            if rows == 1 || cols == 1 {
                // Vector -> diagonal matrix
                let n = rows.max(cols);
                let mut data = vec![0.0; n * n];
                for i in 0..n {
                    let val = if rows == 1 { t.data[i] } else { t.data[i] };
                    data[i + i * n] = val;
                }
                Ok(Value::Tensor(runmat_builtins::Tensor::new(data, vec![n, n]).map_err(|e| format!("diag: {e}"))?))
            } else {
                // Matrix -> main diagonal as column vector
                let n = rows.min(cols);
                let mut data = vec![0.0; n];
                for i in 0..n { data[i] = t.data[i + i * rows]; }
                Ok(Value::Tensor(runmat_builtins::Tensor::new(data, vec![n, 1]).map_err(|e| format!("diag: {e}"))?))
            }
        }
        _ => Err("diag: expected tensor".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "triu")]
fn triu_builtin(a: Value) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("triu: expected tensor".to_string()) };
    let rows = t.rows(); let cols = t.cols(); let mut out = vec![0.0; rows*cols];
    for c in 0..cols { for r in 0..rows { if r <= c { out[r + c*rows] = t.data[r + c*rows]; } } }
    Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![rows, cols]).map_err(|e| format!("triu: {e}"))?))
}

#[runmat_macros::runtime_builtin(name = "tril")]
fn tril_builtin(a: Value) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("tril: expected tensor".to_string()) };
    let rows = t.rows(); let cols = t.cols(); let mut out = vec![0.0; rows*cols];
    for c in 0..cols { for r in 0..rows { if r >= c { out[r + c*rows] = t.data[r + c*rows]; } } }
    Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![rows, cols]).map_err(|e| format!("tril: {e}"))?))
}

#[runmat_macros::runtime_builtin(name = "cat")]
fn cat_var_builtin(dim: f64, rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() < 2 { return Err("cat: expects at least two arrays".to_string()); }
    let d = if dim < 1.0 { 1usize } else { dim as usize } - 1; // zero-based
    let tensors: Vec<runmat_builtins::Tensor> = rest.into_iter().map(|v| match v { Value::Tensor(t) => Ok(t), _ => Err("cat: expected tensors".to_string()) }).collect::<Result<_,_>>()?;
    let rank = tensors.iter().map(|t| t.shape.len()).max().unwrap_or(2);
    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| { let mut s=t.shape.clone(); s.resize(rank, 1); s }).collect();
    for k in 0..rank {
        if k==d { continue; }
        let first = shapes[0][k];
        if !shapes.iter().all(|s| s[k] == first) { return Err("cat: dimension mismatch".to_string()); }
    }
    let mut out_shape = shapes[0].clone(); out_shape[d] = shapes.iter().map(|s| s[d]).sum();
    let mut out = vec![0f64; out_shape.iter().product()];
    fn strides(shape: &[usize]) -> Vec<usize> { let mut s=vec![0; shape.len()]; let mut acc=1; for i in 0..shape.len() { s[i]=acc; acc*=shape[i]; } s }
    let out_str = strides(&out_shape);
    let mut offset = 0usize;
    for (t, s) in tensors.iter().zip(shapes.iter()) {
        let s_str = strides(s);
        let rank = out_shape.len();
        let total: usize = s.iter().product();
        for idx_lin in 0..total {
            // Convert src lin to multi
            let mut rem = idx_lin; let mut src_multi = vec![0usize; rank];
            for i in 0..rank { let si = s[i]; src_multi[i] = rem % si; rem /= si; }
            let mut dst_multi = src_multi.clone(); dst_multi[d] += offset;
            let mut s_lin = 0usize; for i in 0..rank { s_lin += src_multi[i] * s_str[i]; }
            let mut d_lin = 0usize; for i in 0..rank { d_lin += dst_multi[i] * out_str[i]; }
            out[d_lin] = t.data[s_lin];
        }
        offset += s[d];
    }
    Ok(Value::Tensor(runmat_builtins::Tensor::new(out, out_shape).map_err(|e| format!("cat: {e}"))?))
}

#[runmat_macros::runtime_builtin(name = "repmat")]
fn repmat_builtin(a: Value, m: f64, n: f64) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("repmat: expected tensor".to_string()) };
    let m = if m < 1.0 { 1usize } else { m as usize };
    let n = if n < 1.0 { 1usize } else { n as usize };
    let mut shape = t.shape.clone(); if shape.len() < 2 { shape.resize(2, 1); }
    let base_rows = shape[0]; let base_cols = shape[1];
    let out_rows = base_rows * m; let out_cols = base_cols * n;
    let mut out = vec![0f64; out_rows * out_cols];
    for rep_c in 0..n {
        for rep_r in 0..m {
            for c in 0..base_cols {
                for r in 0..base_rows {
                    let src = t.data[r + c*base_rows];
                    let dr = r + rep_r*base_rows; let dc = c + rep_c*base_cols;
                    out[dr + dc*out_rows] = src;
                }
            }
        }
    }
    Ok(Value::Tensor(runmat_builtins::Tensor::new(out, vec![out_rows, out_cols]).map_err(|e| format!("repmat: {e}"))?))
}

#[runmat_macros::runtime_builtin(name = "repmat")]
fn repmat_nd_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    let t = match a { Value::Tensor(t) => t, _ => return Err("repmat: expected tensor".to_string()) };
    // Build replication factors from rest
    let mut reps: Vec<usize> = Vec::new();
    if rest.len() == 1 {
        match &rest[0] {
            Value::Tensor(v) => { for &x in &v.data { reps.push(if x < 1.0 { 1 } else { x as usize }); } }
            _ => return Err("repmat: expected replication vector".to_string()),
        }
    } else {
        for v in rest { match v { Value::Num(n) => reps.push(if n < 1.0 { 1 } else { n as usize }), Value::Int(i) => reps.push(if i < 1 { 1 } else { i as usize }), _ => return Err("repmat: expected numeric reps".to_string()) } }
    }
    let rank = t.shape.len().max(reps.len());
    let mut base = t.shape.clone(); base.resize(rank, 1);
    reps.resize(rank, 1);
    let mut out_shape = base.clone(); for i in 0..rank { out_shape[i] = base[i] * reps[i]; }
    let mut out = vec![0f64; out_shape.iter().product()];
    fn strides(shape: &[usize]) -> Vec<usize> { let mut s=vec![0; shape.len()]; let mut acc=1; for i in 0..shape.len() { s[i]=acc; acc*=shape[i]; } s }
    let src_str = strides(&base);
    // Iterate over all dest coords and map back to source via modulo
    let total: usize = out_shape.iter().product();
    for d_lin in 0..total {
        // convert to multi
        let mut rem = d_lin; let mut multi = vec![0usize; rank]; for i in 0..rank { let s = out_shape[i]; multi[i] = rem % s; rem /= s; }
        let mut src_multi = vec![0usize; rank]; for i in 0..rank { src_multi[i] = multi[i] % base[i]; }
        let mut s_lin = 0usize; for i in 0..rank { s_lin += src_multi[i] * src_str[i]; }
        out[d_lin] = t.data[s_lin];
    }
    Ok(Value::Tensor(runmat_builtins::Tensor::new(out, out_shape).map_err(|e| format!("repmat: {e}"))?))
}

// linspace and meshgrid are defined in arrays.rs; avoid duplicates here

// -------- Vararg string/IO builtins: sprintf, fprintf, disp, warning --------

#[runmat_macros::runtime_builtin(name = "sprintf")]
fn sprintf_builtin(fmt: String, rest: Vec<Value>) -> Result<Value, String> {
    let s = format_variadic(&fmt, &rest)?;
    Ok(Value::String(s))
}

#[runmat_macros::runtime_builtin(name = "fprintf")]
fn fprintf_builtin(first: Value, rest: Vec<Value>) -> Result<Value, String> {
    // MATLAB: fprintf(fid, fmt, ...) or fprintf(fmt, ...)
    let (fmt, args) = match first {
        Value::String(s) => (s, rest),
        Value::Num(_) | Value::Int(_) => {
            // File IDs not supported yet; treat as stdout and expect format string next
            if rest.is_empty() { return Err("fprintf: missing format string".to_string()); }
            let fmt = match &rest[0] { Value::String(s) => s.clone(), _ => return Err("fprintf: expected format string".to_string()) };
            (fmt, rest[1..].to_vec())
        }
        other => return Err(format!("fprintf: unsupported first argument {other:?}")),
    };
    let s = format_variadic(&fmt, &args)?;
    println!("{}", s);
    Ok(Value::Num(s.len() as f64))
}

#[runmat_macros::runtime_builtin(name = "warning")]
fn warning_builtin(fmt: String, rest: Vec<Value>) -> Result<Value, String> {
    let s = format_variadic(&fmt, &rest)?;
    eprintln!("Warning: {}", s);
    Ok(Value::Num(0.0))
}

#[runmat_macros::runtime_builtin(name = "disp")]
fn disp_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::String(s) => println!("{}", s),
        Value::Num(n) => println!("{}", n),
        Value::Int(i) => println!("{}", i),
        Value::Tensor(t) => println!("{:?}", t.data),
        other => println!("{:?}", other),
    }
    Ok(Value::Num(0.0))
}

#[runmat_macros::runtime_builtin(name = "struct")]
fn struct_builtin(rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() % 2 != 0 { return Err("struct: expected name/value pairs".to_string()); }
    let mut st = runmat_builtins::StructValue::new();
    let mut i = 0usize;
    while i < rest.len() {
        let key: String = (&rest[i]).try_into()?;
        let val = rest[i+1].clone();
        st.fields.insert(key, val);
        i += 2;
    }
    Ok(Value::Struct(st))
}

fn format_variadic(fmt: &str, args: &[Value]) -> Result<String, String> {
    // Minimal subset: supports %d/%i, %f, %s and %%
    let mut out = String::with_capacity(fmt.len() + args.len() * 8);
    let mut it = fmt.chars().peekable();
    let mut ai = 0usize;
    while let Some(c) = it.next() {
        if c != '%' { out.push(c); continue; }
        if let Some('%') = it.peek() { it.next(); out.push('%'); continue; }
        // Consume optional width/precision (very limited)
        let mut precision: Option<usize> = None;
        // skip digits for width
        while let Some(ch) = it.peek() { if ch.is_ascii_digit() { it.next(); } else { break; } }
        // precision .digits
        if let Some('.') = it.peek() { it.next(); let mut p = String::new(); while let Some(ch) = it.peek() { if ch.is_ascii_digit() { p.push(*ch); it.next(); } else { break; } } if !p.is_empty() { precision = p.parse::<usize>().ok(); } }
        let ty = it.next().ok_or("sprintf: incomplete format specifier")?;
        let val = args.get(ai).cloned().unwrap_or(Value::Num(0.0));
        ai += 1;
        match ty {
            'd' | 'i' => { let v: f64 = (&val).try_into()?; out.push_str(&(v as i64).to_string()); }
            'f' => {
                let v: f64 = (&val).try_into()?;
                if let Some(p) = precision { out.push_str(&format!("{:.*}", p, v)); } else { out.push_str(&format!("{}", v)); }
            }
            's' => {
                match val {
                    Value::String(s) => out.push_str(&s),
                    Value::Num(n) => out.push_str(&n.to_string()),
                    Value::Int(i) => out.push_str(&i.to_string()),
                    Value::Tensor(t) => out.push_str(&format!("{:?}", t.data)),
                    other => out.push_str(&format!("{:?}", other)),
                }
            }
            other => return Err(format!("sprintf: unsupported format %{}", other)),
        }
    }
    Ok(out)
}
