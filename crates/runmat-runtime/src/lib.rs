use crate::builtins::common::format::format_variadic;
use runmat_builtins::Value;
use runmat_gc_api::GcPtr;

pub mod dispatcher;

pub mod arrays;
pub mod builtins;
pub mod comparison;
pub mod concatenation;
pub mod constants;
pub mod elementwise;
pub mod indexing;
pub mod introspection;
pub mod matrix;
pub mod plotting;
pub mod workspace;

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

pub use dispatcher::{call_builtin, gather_if_needed, is_gpu_value, value_contains_gpu};

// Transitional public shim for tests using matrix_transpose
pub use crate::matrix::matrix_transpose;

// Pruned legacy re-exports; prefer builtins::* and explicit shims only
// Transitional root-level shims for widely used helpers
pub use arrays::create_range;
pub use concatenation::create_matrix_from_values;
pub use elementwise::{elementwise_div, elementwise_mul, elementwise_neg, elementwise_pow, power};
pub use indexing::perform_indexing;
// Explicitly re-export for external users (ignition VM) that build matrices from values
// (kept above)
// Note: constants and mathematics modules only contain #[runtime_builtin] functions
// and don't export public items, so they don't need to be re-exported

#[cfg(feature = "blas-lapack")]
pub use blas::*;
#[cfg(feature = "blas-lapack")]
pub use lapack::*;

pub(crate) fn make_cell_with_shape(values: Vec<Value>, shape: Vec<usize>) -> Result<Value, String> {
    let handles: Vec<GcPtr<Value>> = values
        .into_iter()
        .map(|v| runmat_gc::gc_allocate(v).expect("gc alloc"))
        .collect();
    let ca = runmat_builtins::CellArray::new_handles_with_shape(handles, shape)
        .map_err(|e| format!("Cell creation error: {e}"))?;
    Ok(Value::Cell(ca))
}

pub(crate) fn make_cell(values: Vec<Value>, rows: usize, cols: usize) -> Result<Value, String> {
    make_cell_with_shape(values, vec![rows, cols])
}

// Internal builtin to construct a cell from a vector of values (used by ignition)
#[runmat_macros::runtime_builtin(name = "__make_cell")]
fn make_cell_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let rows = 1usize;
    let cols = rest.len();
    make_cell(rest, rows, cols)
}

fn to_string_scalar(v: &Value) -> Result<String, String> {
    let s: String = v.try_into()?;
    Ok(s)
}

fn to_string_array(v: &Value) -> Result<runmat_builtins::StringArray, String> {
    match v {
        Value::String(s) => runmat_builtins::StringArray::new(vec![s.clone()], vec![1, 1])
            .map_err(|e| e.to_string()),
        Value::StringArray(sa) => Ok(sa.clone()),
        Value::CharArray(ca) => {
            // Convert each row to a string; treat as column vector
            let mut out: Vec<String> = Vec::with_capacity(ca.rows);
            for r in 0..ca.rows {
                let mut s = String::with_capacity(ca.cols);
                for c in 0..ca.cols {
                    s.push(ca.data[r * ca.cols + c]);
                }
                out.push(s);
            }
            runmat_builtins::StringArray::new(out, vec![ca.rows, 1]).map_err(|e| e.to_string())
        }
        other => Err(format!("cannot convert to string array: {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "strtrim")]
fn strtrim_builtin(a: Value) -> Result<Value, String> {
    let s = to_string_scalar(&a)?;
    Ok(Value::String(s.trim().to_string()))
}

// Adjust strjoin semantics: join rows (row-wise)
#[runmat_macros::runtime_builtin(name = "strjoin")]
fn strjoin_rowwise(a: Value, delim: Value) -> Result<Value, String> {
    let d = to_string_scalar(&delim)?;
    let sa = to_string_array(&a)?;
    let rows = *sa.shape.first().unwrap_or(&sa.data.len());
    let cols = *sa.shape.get(1).unwrap_or(&1);
    if rows == 0 || cols == 0 {
        return Ok(Value::StringArray(
            runmat_builtins::StringArray::new(Vec::new(), vec![0, 0]).unwrap(),
        ));
    }
    let mut out: Vec<String> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut s = String::new();
        for c in 0..cols {
            if c > 0 {
                s.push_str(&d);
            }
            s.push_str(&sa.data[r + c * rows]);
        }
        out.push(s);
    }
    Ok(Value::StringArray(
        runmat_builtins::StringArray::new(out, vec![rows, 1])
            .map_err(|e| format!("strjoin: {e}"))?,
    ))
}

// deal: distribute inputs to multiple outputs (via cell for expansion)
#[runmat_macros::runtime_builtin(name = "deal")]
fn deal_builtin(rest: Vec<Value>) -> Result<Value, String> {
    // Return cell row vector of inputs for expansion
    let cols = rest.len();
    make_cell(rest, 1, cols)
}

// Object/handle utilities used by interpreter lowering for OOP/func handles

#[runmat_macros::runtime_builtin(name = "rethrow")]
fn rethrow_builtin(e: Value) -> Result<Value, String> {
    match e {
        Value::MException(me) => Err(format!("{}: {}", me.identifier, me.message)),
        Value::String(s) => Err(s),
        other => Err(format!("MATLAB:error: {other:?}")),
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
            args.extend(rest);
            if let Ok(v) = crate::call_builtin(&qualified, &args) {
                return Ok(v);
            }
            // Fallback to global method name
            crate::call_builtin(&method, &args)
        }
        Value::HandleObject(h) => {
            // Methods on handle classes dispatch to the underlying target's class namespace
            let target = unsafe { &*h.target.as_raw() };
            let class_name = match target {
                Value::Object(o) => o.class_name.clone(),
                Value::Struct(_) => h.class_name.clone(),
                _ => h.class_name.clone(),
            };
            let qualified = format!("{class_name}.{method}");
            let mut args = Vec::with_capacity(1 + rest.len());
            args.push(Value::HandleObject(h.clone()));
            args.extend(rest);
            if let Ok(v) = crate::call_builtin(&qualified, &args) {
                return Ok(v);
            }
            crate::call_builtin(&method, &args)
        }
        other => Err(format!(
            "call_method unsupported on {other:?} for method '{method}'"
        )),
    }
}

// Global dispatch helpers for overloaded indexing (subsref/subsasgn) to support fallback resolution paths
#[runmat_macros::runtime_builtin(name = "subsasgn")]
fn subsasgn_dispatch(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> Result<Value, String> {
    match &obj {
        Value::Object(o) => {
            let qualified = format!("{}.subsasgn", o.class_name);
            crate::call_builtin(&qualified, &[obj, Value::String(kind), payload, rhs])
        }
        Value::HandleObject(h) => {
            let target = unsafe { &*h.target.as_raw() };
            let class_name = match target {
                Value::Object(o) => o.class_name.clone(),
                _ => h.class_name.clone(),
            };
            let qualified = format!("{class_name}.subsasgn");
            crate::call_builtin(&qualified, &[obj, Value::String(kind), payload, rhs])
        }
        other => Err(format!("subsasgn: receiver must be object, got {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "subsref")]
fn subsref_dispatch(obj: Value, kind: String, payload: Value) -> Result<Value, String> {
    match &obj {
        Value::Object(o) => {
            let qualified = format!("{}.subsref", o.class_name);
            crate::call_builtin(&qualified, &[obj, Value::String(kind), payload])
        }
        Value::HandleObject(h) => {
            let target = unsafe { &*h.target.as_raw() };
            let class_name = match target {
                Value::Object(o) => o.class_name.clone(),
                _ => h.class_name.clone(),
            };
            let qualified = format!("{class_name}.subsref");
            crate::call_builtin(&qualified, &[obj, Value::String(kind), payload])
        }
        other => Err(format!("subsref: receiver must be object, got {other:?}")),
    }
}

// -------- Handle classes & events --------

#[runmat_macros::runtime_builtin(name = "new_handle_object")]
fn new_handle_object_builtin(class_name: String) -> Result<Value, String> {
    // Create an underlying object instance and wrap it in a handle
    let obj = new_object_builtin(class_name.clone())?;
    let gc = runmat_gc::gc_allocate(obj).map_err(|e| format!("gc: {e}"))?;
    Ok(Value::HandleObject(runmat_builtins::HandleRef {
        class_name,
        target: gc,
        valid: true,
    }))
}

#[runmat_macros::runtime_builtin(name = "isvalid")]
fn isvalid_builtin(v: Value) -> Result<Value, String> {
    match v {
        Value::HandleObject(h) => Ok(Value::Bool(h.valid)),
        Value::Listener(l) => Ok(Value::Bool(l.valid && l.enabled)),
        _ => Ok(Value::Bool(false)),
    }
}

use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct EventRegistry {
    next_id: u64,
    listeners: std::collections::HashMap<(usize, String), Vec<runmat_builtins::Listener>>,
}

static EVENT_REGISTRY: OnceLock<Mutex<EventRegistry>> = OnceLock::new();

fn events() -> &'static Mutex<EventRegistry> {
    EVENT_REGISTRY.get_or_init(|| Mutex::new(EventRegistry::default()))
}

#[runmat_macros::runtime_builtin(name = "addlistener")]
fn addlistener_builtin(
    target: Value,
    event_name: String,
    callback: Value,
) -> Result<Value, String> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => (unsafe { h.target.as_raw() }) as usize,
        Value::Object(o) => o as *const _ as usize,
        _ => return Err("addlistener: target must be handle or object".to_string()),
    };
    let mut reg = events().lock().unwrap();
    let id = {
        reg.next_id += 1;
        reg.next_id
    };
    let tgt_gc = match target {
        Value::HandleObject(h) => h.target,
        Value::Object(o) => {
            runmat_gc::gc_allocate(Value::Object(o)).map_err(|e| format!("gc: {e}"))?
        }
        _ => unreachable!(),
    };
    let cb_gc = runmat_gc::gc_allocate(callback).map_err(|e| format!("gc: {e}"))?;
    let listener = runmat_builtins::Listener {
        id,
        target: tgt_gc,
        event_name: event_name.clone(),
        callback: cb_gc,
        enabled: true,
        valid: true,
    };
    reg.listeners
        .entry((key_ptr, event_name))
        .or_default()
        .push(listener.clone());
    Ok(Value::Listener(listener))
}

#[runmat_macros::runtime_builtin(name = "notify")]
fn notify_builtin(target: Value, event_name: String, rest: Vec<Value>) -> Result<Value, String> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => (unsafe { h.target.as_raw() }) as usize,
        Value::Object(o) => o as *const _ as usize,
        _ => return Err("notify: target must be handle or object".to_string()),
    };
    let mut to_call: Vec<runmat_builtins::Listener> = Vec::new();
    {
        let reg = events().lock().unwrap();
        if let Some(list) = reg.listeners.get(&(key_ptr, event_name.clone())) {
            for l in list {
                if l.valid && l.enabled {
                    to_call.push(l.clone());
                }
            }
        }
    }
    for l in to_call {
        // Call callback via feval-like protocol
        let mut args = Vec::new();
        args.push(target.clone());
        args.extend(rest.iter().cloned());
        let cbv: Value = (*l.callback).clone();
        match &cbv {
            Value::String(s) if s.starts_with('@') => {
                let mut a = vec![Value::String(s.clone())];
                a.extend(args.into_iter());
                let _ = crate::call_builtin("feval", &a)?;
            }
            Value::FunctionHandle(name) => {
                let mut a = vec![Value::FunctionHandle(name.clone())];
                a.extend(args.into_iter());
                let _ = crate::call_builtin("feval", &a)?;
            }
            Value::Closure(_) => {
                let mut a = vec![cbv.clone()];
                a.extend(args.into_iter());
                let _ = crate::call_builtin("feval", &a)?;
            }
            _ => {}
        }
    }
    Ok(Value::Num(0.0))
}

// Test-oriented dependent property handlers (global). If a class defines a Dependent
// property named 'p', the VM will try to call get.p / set.p. We provide generic
// implementations that read/write a conventional backing field 'p_backing'.
#[runmat_macros::runtime_builtin(name = "get.p")]
fn get_p_builtin(obj: Value) -> Result<Value, String> {
    match obj {
        Value::Object(o) => {
            if let Some(v) = o.properties.get("p_backing") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        other => Err(format!("get.p requires object, got {other:?}")),
    }
}

#[runmat_macros::runtime_builtin(name = "set.p")]
fn set_p_builtin(obj: Value, val: Value) -> Result<Value, String> {
    match obj {
        Value::Object(mut o) => {
            o.properties.insert("p_backing".to_string(), val);
            Ok(Value::Object(o))
        }
        other => Err(format!("set.p requires object, got {other:?}")),
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
pub(crate) fn new_object_builtin(class_name: String) -> Result<Value, String> {
    if let Some(def) = runmat_builtins::get_class(&class_name) {
        // Collect class hierarchy from root to leaf for default initialization
        let mut chain: Vec<runmat_builtins::ClassDef> = Vec::new();
        // Walk up to root
        let mut cursor: Option<String> = Some(def.name.clone());
        while let Some(name) = cursor {
            if let Some(cd) = runmat_builtins::get_class(&name) {
                chain.push(cd.clone());
                cursor = cd.parent.clone();
            } else {
                break;
            }
        }
        // Reverse to root-first
        chain.reverse();
        let mut obj = runmat_builtins::ObjectInstance::new(def.name.clone());
        // Apply defaults from root to leaf (leaf overrides effectively by later assignment)
        for cd in chain {
            for (k, p) in cd.properties.iter() {
                if !p.is_static {
                    if let Some(v) = &p.default_value {
                        obj.properties.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        Ok(Value::Object(obj))
    } else {
        Ok(Value::Object(runmat_builtins::ObjectInstance::new(
            class_name,
        )))
    }
}

// handle-object builtins removed for now

#[runmat_macros::runtime_builtin(name = "classref")]
fn classref_builtin(class_name: String) -> Result<Value, String> {
    Ok(Value::ClassRef(class_name))
}

#[runmat_macros::runtime_builtin(name = "__register_test_classes")]
fn register_test_classes_builtin() -> Result<Value, String> {
    use runmat_builtins::*;
    let mut props = std::collections::HashMap::new();
    props.insert(
        "x".to_string(),
        PropertyDef {
            name: "x".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    props.insert(
        "y".to_string(),
        PropertyDef {
            name: "y".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    props.insert(
        "staticValue".to_string(),
        PropertyDef {
            name: "staticValue".to_string(),
            is_static: true,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(42.0)),
        },
    );
    props.insert(
        "secret".to_string(),
        PropertyDef {
            name: "secret".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Private,
            set_access: Access::Private,
            default_value: Some(Value::Num(99.0)),
        },
    );
    let mut methods = std::collections::HashMap::new();
    methods.insert(
        "move".to_string(),
        MethodDef {
            name: "move".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "Point.move".to_string(),
        },
    );
    methods.insert(
        "origin".to_string(),
        MethodDef {
            name: "origin".to_string(),
            is_static: true,
            access: Access::Public,
            function_name: "Point.origin".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Point".to_string(),
        parent: None,
        properties: props,
        methods,
    });

    // Namespaced class example: pkg.PointNS with same shape as Point
    let mut ns_props = std::collections::HashMap::new();
    ns_props.insert(
        "x".to_string(),
        PropertyDef {
            name: "x".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(1.0)),
        },
    );
    ns_props.insert(
        "y".to_string(),
        PropertyDef {
            name: "y".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(2.0)),
        },
    );
    let ns_methods = std::collections::HashMap::new();
    runmat_builtins::register_class(ClassDef {
        name: "pkg.PointNS".to_string(),
        parent: None,
        properties: ns_props,
        methods: ns_methods,
    });

    // Inheritance: Shape (base) and Circle (derived)
    let shape_props = std::collections::HashMap::new();
    let mut shape_methods = std::collections::HashMap::new();
    shape_methods.insert(
        "area".to_string(),
        MethodDef {
            name: "area".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "Shape.area".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Shape".to_string(),
        parent: None,
        properties: shape_props,
        methods: shape_methods,
    });

    let mut circle_props = std::collections::HashMap::new();
    circle_props.insert(
        "r".to_string(),
        PropertyDef {
            name: "r".to_string(),
            is_static: false,
            is_dependent: false,
            get_access: Access::Public,
            set_access: Access::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    let mut circle_methods = std::collections::HashMap::new();
    circle_methods.insert(
        "area".to_string(),
        MethodDef {
            name: "area".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "Circle.area".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Circle".to_string(),
        parent: Some("Shape".to_string()),
        properties: circle_props,
        methods: circle_methods,
    });

    // Constructor demo class: Ctor with static constructor method Ctor
    let ctor_props = std::collections::HashMap::new();
    let mut ctor_methods = std::collections::HashMap::new();
    ctor_methods.insert(
        "Ctor".to_string(),
        MethodDef {
            name: "Ctor".to_string(),
            is_static: true,
            access: Access::Public,
            function_name: "Ctor.Ctor".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "Ctor".to_string(),
        parent: None,
        properties: ctor_props,
        methods: ctor_methods,
    });

    // Overloaded indexing demo class: OverIdx with subsref/subsasgn
    let overidx_props = std::collections::HashMap::new();
    let mut overidx_methods = std::collections::HashMap::new();
    overidx_methods.insert(
        "subsref".to_string(),
        MethodDef {
            name: "subsref".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "OverIdx.subsref".to_string(),
        },
    );
    overidx_methods.insert(
        "subsasgn".to_string(),
        MethodDef {
            name: "subsasgn".to_string(),
            is_static: false,
            access: Access::Public,
            function_name: "OverIdx.subsasgn".to_string(),
        },
    );
    runmat_builtins::register_class(ClassDef {
        name: "OverIdx".to_string(),
        parent: None,
        properties: overidx_props,
        methods: overidx_methods,
    });
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
            if let Some(Value::Num(v)) = o.properties.get("x") {
                x = *v;
            }
            if let Some(Value::Num(v)) = o.properties.get("y") {
                y = *v;
            }
            o.properties.insert("x".to_string(), Value::Num(x + dx));
            o.properties.insert("y".to_string(), Value::Num(y + dy));
            Ok(Value::Object(o))
        }
        other => Err(format!(
            "Point.move requires object receiver, got {other:?}"
        )),
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
            let r = if let Some(Value::Num(v)) = o.properties.get("r") {
                *v
            } else {
                0.0
            };
            Ok(Value::Num(std::f64::consts::PI * r * r))
        }
        other => Err(format!(
            "Circle.area requires object receiver, got {other:?}"
        )),
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

// --- Test-only package functions to exercise import precedence ---
#[runmat_macros::runtime_builtin(name = "PkgF.foo")]
fn pkgf_foo() -> Result<Value, String> {
    Ok(Value::Num(10.0))
}

#[runmat_macros::runtime_builtin(name = "PkgG.foo")]
fn pkgg_foo() -> Result<Value, String> {
    Ok(Value::Num(20.0))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.subsref")]
fn overidx_subsref(obj: Value, kind: String, payload: Value) -> Result<Value, String> {
    // Simple sentinel implementation: return different values for '.' vs '()'
    match (obj, kind.as_str(), payload) {
        (Value::Object(_), "()", Value::Cell(_)) => Ok(Value::Num(99.0)),
        (Value::Object(o), "{}", Value::Cell(_)) => {
            if let Some(v) = o.properties.get("lastCell") {
                Ok(v.clone())
            } else {
                Ok(Value::Num(0.0))
            }
        }
        (Value::Object(o), ".", Value::String(field)) => {
            // If field exists, return it; otherwise sentinel 77
            if let Some(v) = o.properties.get(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        (Value::Object(o), ".", Value::CharArray(ca)) => {
            let field: String = ca.data.iter().collect();
            if let Some(v) = o.properties.get(&field) {
                Ok(v.clone())
            } else {
                Ok(Value::Num(77.0))
            }
        }
        _ => Err("subsref: unsupported payload".to_string()),
    }
}

#[runmat_macros::runtime_builtin(name = "OverIdx.subsasgn")]
fn overidx_subsasgn(
    mut obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> Result<Value, String> {
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
        (Value::Object(o), ".", Value::CharArray(ca)) => {
            let field: String = ca.data.iter().collect();
            o.properties.insert(field, rhs);
            Ok(Value::Object(o.clone()))
        }
        _ => Err("subsasgn: unsupported payload".to_string()),
    }
}

// --- Operator overloading methods for OverIdx (test scaffolding) ---
#[runmat_macros::runtime_builtin(name = "OverIdx.plus")]
fn overidx_plus(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.plus: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k + r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.times")]
fn overidx_times(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.times: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.mtimes")]
fn overidx_mtimes(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.mtimes: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k * r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.lt")]
fn overidx_lt(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.lt: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k < r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.gt")]
fn overidx_gt(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.gt: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if k > r { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.eq")]
fn overidx_eq(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.eq: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
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
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.rdivide: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(k / r))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.ldivide")]
fn overidx_ldivide(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.ldivide: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(r / k))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.and")]
fn overidx_and(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.and: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) && (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.or")]
fn overidx_or(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.or: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    Ok(Value::Num(if (k != 0.0) || (r != 0.0) { 1.0 } else { 0.0 }))
}

#[runmat_macros::runtime_builtin(name = "OverIdx.xor")]
fn overidx_xor(obj: Value, rhs: Value) -> Result<Value, String> {
    let o = match obj {
        Value::Object(o) => o,
        _ => return Err("OverIdx.xor: receiver must be object".to_string()),
    };
    let k = if let Some(Value::Num(v)) = o.properties.get("k") {
        *v
    } else {
        0.0
    };
    let r: f64 = (&rhs).try_into()?;
    let a = k != 0.0;
    let b = r != 0.0;
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
                Err(format!(
                    "feval: expected function handle string starting with '@', got {s}"
                ))
            }
        }
        // Also accept character row vector handles like '@max'
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                let s: String = ca.data.iter().collect();
                if let Some(name) = s.strip_prefix('@') {
                    crate::call_builtin(name, &rest)
                } else {
                    Err(format!(
                        "feval: expected function handle string starting with '@', got {s}"
                    ))
                }
            } else {
                Err("feval: function handle char array must be a row vector".to_string())
            }
        }
        Value::Closure(c) => {
            let mut args = c.captures.clone();
            args.extend(rest);
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
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.transpose(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("transpose: unsupported for gpuArray".to_string())
        }
        Value::Tensor(ref m) => Ok(Value::Tensor(crate::matrix::matrix_transpose(m))),
        // For complex scalars, transpose is conjugate transpose => scalar transpose is conjugate
        Value::Complex(re, im) => Ok(Value::Complex(re, -im)),
        Value::ComplexTensor(ref ct) => {
            let mut data: Vec<(f64, f64)> = vec![(0.0, 0.0); ct.rows * ct.cols];
            // Conjugate transpose for complex matrices
            for i in 0..ct.rows {
                for j in 0..ct.cols {
                    let (re, im) = ct.data[i + j * ct.rows];
                    data[j + i * ct.cols] = (re, -im);
                }
            }
            Ok(Value::ComplexTensor(
                runmat_builtins::ComplexTensor::new_2d(data, ct.cols, ct.rows).unwrap(),
            ))
        }
        Value::Num(n) => Ok(Value::Num(n)), // Scalar transpose is identity
        _ => Err("transpose not supported for this type".to_string()),
    }
}

// -------- Reductions: sum/prod/mean/any/all --------

#[allow(dead_code)]
fn tensor_sum_all(t: &runmat_builtins::Tensor) -> f64 {
    t.data.iter().sum()
}

fn tensor_prod_all(t: &runmat_builtins::Tensor) -> f64 {
    t.data.iter().product()
}

fn prod_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![1.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut p = 1.0;
                    for r in 0..rows {
                        p *= t.data[r + c * rows];
                    }
                    *oc = p;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("prod: {e}"))?,
                ))
            } else {
                Ok(Value::Num(tensor_prod_all(&t)))
            }
        }
        _ => Err("prod: expected tensor".to_string()),
    }
}

fn prod_dim(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("prod: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![1.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut p = 1.0;
            for r in 0..rows {
                p *= t.data[r + c * rows];
            }
            *oc = p;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("prod: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![1.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut p = 1.0;
            for c in 0..cols {
                p *= t.data[r + c * rows];
            }
            *orow = p;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("prod: {e}"))?,
        ))
    } else {
        Err("prod: dim out of range".to_string())
    }
}

#[runmat_macros::runtime_builtin(name = "prod")]
fn prod_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return prod_all_or_cols(a);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return prod_dim(a, *d),
            Value::Int(i) => return prod_dim(a, i.to_i64() as f64),
            _ => {}
        }
    }
    Err("prod: unsupported arguments".to_string())
}

fn any_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut v = 0.0;
                    for r in 0..rows {
                        if t.data[r + c * rows] != 0.0 {
                            v = 1.0;
                            break;
                        }
                    }
                    *oc = v;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("any: {e}"))?,
                ))
            } else {
                Ok(Value::Num(if t.data.iter().any(|&x| x != 0.0) {
                    1.0
                } else {
                    0.0
                }))
            }
        }
        _ => Err("any: expected tensor".to_string()),
    }
}

fn any_dim(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("any: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut v = 0.0;
            for r in 0..rows {
                if t.data[r + c * rows] != 0.0 {
                    v = 1.0;
                    break;
                }
            }
            *oc = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("any: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut v = 0.0;
            for c in 0..cols {
                if t.data[r + c * rows] != 0.0 {
                    v = 1.0;
                    break;
                }
            }
            *orow = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("any: {e}"))?,
        ))
    } else {
        Err("any: dim out of range".to_string())
    }
}

#[runmat_macros::runtime_builtin(name = "any")]
fn any_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return any_all_or_cols(a);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return any_dim(a, *d),
            Value::Int(i) => return any_dim(a, i.to_i64() as f64),
            _ => {}
        }
    }
    Err("any: unsupported arguments".to_string())
}

fn all_all_or_cols(a: Value) -> Result<Value, String> {
    match a {
        Value::Tensor(t) => {
            let rows = t.rows();
            let cols = t.cols();
            if rows > 1 && cols > 1 {
                let mut out = vec![0.0f64; cols];
                for (c, oc) in out.iter_mut().enumerate().take(cols) {
                    let mut v = 1.0;
                    for r in 0..rows {
                        if t.data[r + c * rows] == 0.0 {
                            v = 0.0;
                            break;
                        }
                    }
                    *oc = v;
                }
                Ok(Value::Tensor(
                    runmat_builtins::Tensor::new(out, vec![1, cols])
                        .map_err(|e| format!("all: {e}"))?,
                ))
            } else {
                Ok(Value::Num(if t.data.iter().all(|&x| x != 0.0) {
                    1.0
                } else {
                    0.0
                }))
            }
        }
        _ => Err("all: expected tensor".to_string()),
    }
}

fn all_dim(a: Value, dim: f64) -> Result<Value, String> {
    let t = match a {
        Value::Tensor(t) => t,
        _ => return Err("all: expected tensor".to_string()),
    };
    let dim = if dim < 1.0 { 1usize } else { dim as usize };
    let rows = t.rows();
    let cols = t.cols();
    if dim == 1 {
        let mut out = vec![0.0f64; cols];
        for (c, oc) in out.iter_mut().enumerate().take(cols) {
            let mut v = 1.0;
            for r in 0..rows {
                if t.data[r + c * rows] == 0.0 {
                    v = 0.0;
                    break;
                }
            }
            *oc = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![1, cols]).map_err(|e| format!("all: {e}"))?,
        ))
    } else if dim == 2 {
        let mut out = vec![0.0f64; rows];
        for (r, orow) in out.iter_mut().enumerate().take(rows) {
            let mut v = 1.0;
            for c in 0..cols {
                if t.data[r + c * rows] == 0.0 {
                    v = 0.0;
                    break;
                }
            }
            *orow = v;
        }
        Ok(Value::Tensor(
            runmat_builtins::Tensor::new(out, vec![rows, 1]).map_err(|e| format!("all: {e}"))?,
        ))
    } else {
        Err("all: dim out of range".to_string())
    }
}

#[runmat_macros::runtime_builtin(name = "all")]
fn all_var_builtin(a: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return all_all_or_cols(a);
    }
    if rest.len() == 1 {
        match &rest[0] {
            Value::Num(d) => return all_dim(a, *d),
            Value::Int(i) => return all_dim(a, i.to_i64() as f64),
            _ => {}
        }
    }
    Err("all: unsupported arguments".to_string())
}

#[runmat_macros::runtime_builtin(name = "warning", sink = true)]
fn warning_builtin(fmt: String, rest: Vec<Value>) -> Result<Value, String> {
    let s = format_variadic(&fmt, &rest)?;
    eprintln!("Warning: {s}");
    Ok(Value::Num(0.0))
}

#[runmat_macros::runtime_builtin(name = "getmethod")]
fn getmethod_builtin(obj: Value, name: String) -> Result<Value, String> {
    match obj {
        Value::Object(o) => {
            // Return a closure capturing the receiver; feval will call runtime builtin call_method
            Ok(Value::Closure(runmat_builtins::Closure {
                function_name: "call_method".to_string(),
                captures: vec![Value::Object(o), Value::String(name)],
            }))
        }
        Value::ClassRef(cls) => Ok(Value::String(format!("@{cls}.{name}"))),
        other => Err(format!("getmethod unsupported on {other:?}")),
    }
}
