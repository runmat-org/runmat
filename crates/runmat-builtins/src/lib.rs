pub use inventory;
use std::convert::TryFrom;
use std::fmt;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i32),
    Num(f64),
    Bool(bool),
    String(String),
    // MATLAB string array (R2016b+): N-D array of string scalars
    StringArray(StringArray),
    // MATLAB char array (single-quoted): 2-D character array (rows x cols)
    CharArray(CharArray),
    Tensor(Tensor),
    Cell(CellArray),
    // MATLAB struct (scalar or nested). Struct arrays are represented in higher layers;
    // this variant holds a single struct's fields.
    Struct(StructValue),
    // GPU-resident tensor handle (opaque; buffer managed by backend)
    GpuTensor(runmat_accelerate_api::GpuTensorHandle),
    // Simple object instance until full class system lands
    Object(ObjectInstance),
    // Function handle pointing to a named function (builtin or user)
    FunctionHandle(String),
    Closure(Closure),
    ClassRef(String),
    MException(MException),
}
#[derive(Debug, Clone, PartialEq)]
pub struct StructValue {
    pub fields: HashMap<String, Value>,
}

impl StructValue {
    pub fn new() -> Self { Self { fields: HashMap::new() } }
}


#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>, // Column-major layout
    pub rows: usize,        // Compatibility for 2D usage
    pub cols: usize,        // Compatibility for 2D usage
}

#[derive(Debug, Clone, PartialEq)]
pub struct StringArray {
    pub data: Vec<String>,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CharArray {
    pub data: Vec<char>,
    pub rows: usize,
    pub cols: usize,
}

impl CharArray {
    pub fn new_row(s: &str) -> Self { CharArray { data: s.chars().collect(), rows: 1, cols: s.chars().count() } }
    pub fn new(data: Vec<char>, rows: usize, cols: usize) -> Result<Self, String> {
        if rows * cols != data.len() { return Err(format!("Char data length {} doesn't match dimensions {}x{}", data.len(), rows, cols)); }
        Ok(CharArray { data, rows, cols })
    }
}

impl StringArray {
    pub fn new(data: Vec<String>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "StringArray data length {} doesn't match shape {:?} ({} elements)",
                data.len(), shape, expected
            ));
        }
        let (rows, cols) = if shape.len() >= 2 { (shape[0], shape[1]) } else if shape.len() == 1 { (1, shape[0]) } else { (0, 0) };
        Ok(StringArray { data, shape, rows, cols })
    }
    pub fn new_2d(data: Vec<String>, rows: usize, cols: usize) -> Result<Self, String> { Self::new(data, vec![rows, cols]) }
    pub fn rows(&self) -> usize { self.shape.get(0).copied().unwrap_or(1) }
    pub fn cols(&self) -> usize { self.shape.get(1).copied().unwrap_or(1) }
}

// GpuTensorHandle now lives in runmat-accel-api


impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "Tensor data length {} doesn't match shape {:?} ({} elements)",
                data.len(), shape, expected
            ));
        }
        let (rows, cols) = if shape.len() >= 2 { (shape[0], shape[1]) } else if shape.len() == 1 { (1, shape[0]) } else { (0, 0) };
        Ok(Tensor { data, shape, rows, cols })
    }

    pub fn new_2d(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let (rows, cols) = if shape.len() >= 2 { (shape[0], shape[1]) } else if shape.len() == 1 { (1, shape[0]) } else { (0, 0) };
        Tensor { data: vec![0.0; size], shape, rows, cols }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let (rows, cols) = if shape.len() >= 2 { (shape[0], shape[1]) } else if shape.len() == 1 { (1, shape[0]) } else { (0, 0) };
        Tensor { data: vec![1.0; size], shape, rows, cols }
    }

    // 2D helpers for transitional call sites
    pub fn zeros2(rows: usize, cols: usize) -> Self { Self::zeros(vec![rows, cols]) }
    pub fn ones2(rows: usize, cols: usize) -> Self { Self::ones(vec![rows, cols]) }

    pub fn rows(&self) -> usize { self.shape.get(0).copied().unwrap_or(1) }
    pub fn cols(&self) -> usize { self.shape.get(1).copied().unwrap_or(1) }

    pub fn get2(&self, row: usize, col: usize) -> Result<f64, String> {
        let rows = self.rows(); let cols = self.cols();
        if row >= rows || col >= cols {
            return Err(format!("Index ({}, {}) out of bounds for {}x{} tensor", row, col, rows, cols));
        }
        // Column-major linearization: lin = row + col*rows
        Ok(self.data[row + col * rows])
    }

    pub fn set2(&mut self, row: usize, col: usize, value: f64) -> Result<(), String> {
        let rows = self.rows(); let cols = self.cols();
        if row >= rows || col >= cols {
            return Err(format!("Index ({}, {}) out of bounds for {}x{} tensor", row, col, rows, cols));
        }
        // Column-major linearization
        self.data[row + col * rows] = value;
        Ok(())
    }

    pub fn scalar_to_tensor2(scalar: f64, rows: usize, cols: usize) -> Tensor {
        Tensor { data: vec![scalar; rows * cols], shape: vec![rows, cols], rows, cols }
    }
    // No-compat constructors: prefer new/new_2d/zeros/zeros2/ones/ones2
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.shape.len() {
            0 | 1 => {
                // Treat as row vector for display
                write!(f, "[")?;
                for (i, v) in self.data.iter().enumerate() {
                    if i > 0 { write!(f, " ")?; }
                    write!(f, "{}", format_number_short_g(*v))?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.rows(); let cols = self.cols();
                write!(f, "[")?;
                for r in 0..rows {
                    for c in 0..cols {
                        if c > 0 { write!(f, " ")?; }
                        let v = self.data[r + c * rows];
                        write!(f, "{}", format_number_short_g(v))?;
                    }
                    if r + 1 < rows { write!(f, "; ")?; }
                }
                write!(f, "]")
            }
            _ => write!(f, "Tensor(shape={:?})", self.shape),
        }
    }
}

impl fmt::Display for StringArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.shape.len() {
            0 | 1 => {
                write!(f, "[")?;
                for (i, v) in self.data.iter().enumerate() {
                    if i > 0 { write!(f, " ")?; }
                    write!(f, "\"{}\"", v.replace('"', "\""))?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.rows(); let cols = self.cols();
                write!(f, "[")?;
                for r in 0..rows {
                    for c in 0..cols {
                        if c > 0 { write!(f, " ")?; }
                        let v = &self.data[r + c * rows];
                        write!(f, "\"{}\"", v.replace('"', "\""))?;
                    }
                    if r + 1 < rows { write!(f, "; ")?; }
                }
                write!(f, "]")
            }
            _ => write!(f, "StringArray(shape={:?})", self.shape),
        }
    }
}

impl fmt::Display for CharArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display as single-quoted rows separated by ;
        write!(f, "[")?;
        for r in 0..self.rows {
            if r > 0 { write!(f, "; ")?; }
            write!(f, "'")?;
            for c in 0..self.cols { let ch = self.data[r * self.cols + c]; if ch == '\'' { write!(f, "''")?; } else { write!(f, "{}", ch)?; } }
            write!(f, "'")?;
        }
        write!(f, "]")
    }
}

// From implementations for Value
impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Int(i)
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
            Value::Int(i) => Ok(*i),
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
            Value::Int(i) => Ok(*i as f64),
            _ => Err(format!("cannot convert {v:?} to f64")),
        }
    }
}

impl TryFrom<&Value> for bool {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Bool(b) => Ok(*b),
            Value::Int(i) => Ok(*i != 0),
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
                if sa.data.len() == 1 { Ok(sa.data[0].clone()) } else { Err("cannot convert string array to scalar string".to_string()) }
            }
            Value::CharArray(ca) => {
                // Convert full char array to one string if it is a single row; else error
                if ca.rows == 1 { Ok(ca.data.iter().collect()) } else { Err("cannot convert multi-row char array to scalar string".to_string()) }
            }
            Value::Int(i) => Ok(i.to_string()),
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
            _ => Err(format!("cannot convert {v:?} to Matrix")),
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
    /// String type
    String,
    /// Tensor type with optional shape information (column-major semantics in runtime)
    Tensor {
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
}

impl Type {
    /// Create a tensor type with unknown shape
    pub fn tensor() -> Self { Type::Tensor { shape: None } }

    /// Create a tensor type with known shape
    pub fn tensor_with_shape(shape: Vec<usize>) -> Self {
        Type::Tensor { shape: Some(shape.into_iter().map(|d| Some(d)).collect()) }
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
            (a, b) => a == b,
        }
    }

    /// Get the most specific common type between two types
    pub fn unify(&self, other: &Type) -> Type {
        match (self, other) {
            (Type::Unknown, t) | (t, Type::Unknown) => t.clone(),
            (Type::Int, Type::Num) | (Type::Num, Type::Int) => Type::Num,
            (Type::Tensor { .. }, Type::Tensor { .. }) => Type::tensor(), // Lose shape info for now
            (Type::Struct { known_fields: a }, Type::Struct { known_fields: b }) => {
                match (a, b) {
                    (None, None) => Type::Struct { known_fields: None },
                    (Some(ka), None) | (None, Some(ka)) => Type::Struct { known_fields: Some(ka.clone()) },
                    (Some(ka), Some(kb)) => {
                        let mut set: std::collections::BTreeSet<String> = ka.iter().cloned().collect();
                        set.extend(kb.iter().cloned());
                        Type::Struct { known_fields: Some(set.into_iter().collect()) }
                    }
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
            Value::Bool(_) => Type::Bool,
            Value::String(_) => Type::String,
            Value::StringArray(sa) => {
                // Model as Cell of String for type system for now
                if sa.data.is_empty() { Type::cell_of(Type::String) } else { Type::cell_of(Type::String) }
            }
            Value::Tensor(t) => {
                Type::Tensor { shape: Some(t.shape.iter().map(|&d| Some(d)).collect()) }
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
            Value::GpuTensor(h) => { Type::Tensor { shape: Some(h.shape.iter().map(|&d| Some(d)).collect()) } }
            Value::Object(_) => Type::Unknown,
            Value::Struct(_) => Type::Struct { known_fields: None },
            Value::FunctionHandle(_) => Type::Function { params: vec![Type::Unknown], returns: Box::new(Type::Unknown) },
            Value::Closure(_) => Type::Function { params: vec![Type::Unknown], returns: Box::new(Type::Unknown) },
            Value::ClassRef(_) => Type::Unknown,
            Value::MException(_) => Type::Unknown,
            Value::CharArray(ca) => {
                // Treat as cell of char for type purposes; or a 2-D char matrix conceptually
                Type::Cell { element_type: Some(Box::new(Type::String)), length: Some(ca.rows * ca.cols) }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    pub function_name: String,
    pub captures: Vec<Value>,
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
    pub implementation: fn(&[Value]) -> Result<Value, String>,
}

impl BuiltinFunction {
    pub fn new(
        name: &'static str,
        description: &'static str,
        category: &'static str,
        doc: &'static str,
        examples: &'static str,
        param_types: Vec<Type>,
        return_type: Type,
        implementation: fn(&[Value]) -> Result<Value, String>,
    ) -> Self {
        Self { name, description, category, doc, examples, param_types, return_type, implementation }
    }
}

/// A constant value that can be accessed as a variable
#[derive(Clone)]
pub struct Constant {
    pub name: &'static str,
    pub value: Value,
}

impl std::fmt::Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Constant {{ name: {:?}, value: {:?} }}",
            self.name, self.value
        )
    }
}

inventory::collect!(BuiltinFunction);
inventory::collect!(Constant);

pub fn builtin_functions() -> Vec<&'static BuiltinFunction> {
    inventory::iter::<BuiltinFunction>().collect()
}

pub fn constants() -> Vec<&'static Constant> {
    inventory::iter::<Constant>().collect()
}

// ----------------------
// Display implementations
// ----------------------

fn format_number_short_g(value: f64) -> String {
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
    // Normalize -0.0 to 0
    let mut v = value;
    if v == 0.0 {
        v = 0.0;
    }

    let abs = v.abs();
    if abs == 0.0 {
        return "0".to_string();
    }

    // Decide between fixed and scientific notation roughly like MATLAB short g
    let use_scientific = !(1e-4..1e6).contains(&abs);

    if use_scientific {
        // 5 significant digits in scientific notation for short g style
        let s = format!("{v:.5e}");
        // Trim trailing zeros in fraction part
        if let Some(idx) = s.find('e') {
            let (mut mantissa, exp) = s.split_at(idx);
            // mantissa like "-1.23450"
            if let Some(dot_idx) = mantissa.find('.') {
                // Trim trailing zeros
                let mut end = mantissa.len();
                while end > dot_idx + 1 && mantissa.as_bytes()[end - 1] == b'0' {
                    end -= 1;
                }
                if end > 0 && mantissa.as_bytes()[end - 1] == b'.' {
                    end -= 1;
                }
                mantissa = &mantissa[..end];
            }
            return format!("{mantissa}{exp}");
        }
        return s;
    }

    // Fixed notation with up to 12 significant digits, trim trailing zeros
    // Compute number of decimals to retain to reach ~12 significant digits
    let exp10 = abs.log10().floor() as i32; // position of most significant digit
    let sig_digits: i32 = 12;
    let decimals = (sig_digits - 1 - exp10).clamp(0, 12) as usize;
    // Round to that many decimals
    let pow = 10f64.powi(decimals as i32);
    let rounded = (v * pow).round() / pow;
    let mut s = format!("{rounded:.decimals$}");
    if let Some(dot) = s.find('.') {
        // Trim trailing zeros
        let mut end = s.len();
        while end > dot + 1 && s.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if end > 0 && s.as_bytes()[end - 1] == b'.' {
            end -= 1;
        }
        s.truncate(end);
    }
    if s.is_empty() || s == "-0" {
        s = "0".to_string();
    }
    s
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
        Self { identifier, message, stack: Vec::new() }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{i}"),
            Value::Num(n) => write!(f, "{}", format_number_short_g(*n)),
            Value::Bool(b) => write!(f, "{}", if *b { 1 } else { 0 }),
            Value::String(s) => write!(f, "'{s}'"),
            Value::StringArray(sa) => write!(f, "{sa}"),
            Value::CharArray(ca) => write!(f, "{ca}"),
            Value::Tensor(m) => write!(f, "{m}"),
            Value::Cell(ca) => ca.fmt(f),
            Value::GpuTensor(h) => write!(f, "GpuTensor(shape={:?}, device={}, buffer={})", h.shape, h.device_id, h.buffer_id),
            Value::Object(obj) => write!(f, "{}(props={})", obj.class_name, obj.properties.len()),
            Value::Struct(st) => write!(f, "struct(fields={})", st.fields.len()),
            Value::FunctionHandle(name) => write!(f, "@{}", name),
            Value::Closure(c) => write!(f, "<closure {} captures={}>", c.function_name, c.captures.len()),
            Value::ClassRef(name) => write!(f, "<class {}>", name),
            Value::MException(e) => write!(f, "MException(identifier='{}', message='{}')", e.identifier, e.message),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellArray {
    pub data: Vec<Value>,
    pub rows: usize,
    pub cols: usize,
}

impl CellArray {
    pub fn new(data: Vec<Value>, rows: usize, cols: usize) -> Result<Self, String> {
        if rows * cols != data.len() {
            return Err(format!("Cell data length {} doesn't match dimensions {}x{}", data.len(), rows, cols));
        }
        Ok(CellArray { data, rows, cols })
    }

    pub fn get(&self, row: usize, col: usize) -> Result<&Value, String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!("Cell index ({}, {}) out of bounds for {}x{} cell array", row, col, self.rows, self.cols));
        }
        Ok(&self.data[row * self.cols + col])
    }
}

impl fmt::Display for CellArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for r in 0..self.rows {
            for c in 0..self.cols {
                if c > 0 { write!(f, ", ")?; }
                let v = &self.data[r * self.cols + c];
                write!(f, "{v}")?;
            }
            if r + 1 < self.rows { write!(f, "; ")?; }
        }
        write!(f, "}}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectInstance {
    pub class_name: String,
    pub properties: HashMap<String, Value>,
}

impl ObjectInstance {
    pub fn new(class_name: String) -> Self { Self { class_name, properties: HashMap::new() } }
}

// -------- Class registry (scaffolding) --------
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access { Public, Private }

#[derive(Debug, Clone)]
pub struct PropertyDef {
    pub name: String,
    pub is_static: bool,
    pub is_dependent: bool,
    pub get_access: Access,
    pub set_access: Access,
    pub default_value: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct MethodDef {
    pub name: String,
    pub is_static: bool,
    pub access: Access,
    pub function_name: String, // bound runtime builtin/user func name
}

#[derive(Debug, Clone)]
pub struct ClassDef {
    pub name: String,                // namespaced e.g. pkg.Point
    pub parent: Option<String>,
    pub properties: HashMap<String, PropertyDef>,
    pub methods: HashMap<String, MethodDef>,
}

use std::sync::{Mutex, OnceLock};

static CLASS_REGISTRY: OnceLock<Mutex<HashMap<String, ClassDef>>> = OnceLock::new();
static STATIC_VALUES: OnceLock<Mutex<HashMap<(String, String), Value>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<String, ClassDef>> {
    CLASS_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_class(def: ClassDef) {
    let mut m = registry().lock().unwrap();
    m.insert(def.name.clone(), def);
}

pub fn get_class(name: &str) -> Option<ClassDef> {
    registry().lock().unwrap().get(name).cloned()
}

/// Resolve a property through the inheritance chain, returning the property definition and
/// the name of the class where it was defined.
pub fn lookup_property(class_name: &str, prop: &str) -> Option<(PropertyDef, String)> {
    let reg = registry().lock().unwrap();
    let mut current = Some(class_name.to_string());
    let guard: Option<std::sync::MutexGuard<'_, std::collections::HashMap<String, ClassDef>>> = None;
    drop(guard);
    while let Some(name) = current {
        if let Some(cls) = reg.get(&name) {
            if let Some(p) = cls.properties.get(prop) {
                return Some((p.clone(), name));
            }
            current = cls.parent.clone();
        } else {
            break;
        }
    }
    None
}

/// Resolve a method through the inheritance chain, returning the method definition and
/// the name of the class where it was defined.
pub fn lookup_method(class_name: &str, method: &str) -> Option<(MethodDef, String)> {
    let reg = registry().lock().unwrap();
    let mut current = Some(class_name.to_string());
    while let Some(name) = current {
        if let Some(cls) = reg.get(&name) {
            if let Some(m) = cls.methods.get(method) {
                return Some((m.clone(), name));
            }
            current = cls.parent.clone();
        } else {
            break;
        }
    }
    None
}

fn static_values() -> &'static Mutex<HashMap<(String, String), Value>> {
    STATIC_VALUES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn get_static_property_value(class_name: &str, prop: &str) -> Option<Value> {
    static_values().lock().unwrap().get(&(class_name.to_string(), prop.to_string())).cloned()
}

pub fn set_static_property_value(class_name: &str, prop: &str, value: Value) {
    static_values().lock().unwrap().insert((class_name.to_string(), prop.to_string()), value);
}

/// Set a static property, resolving the defining ancestor class for storage.
pub fn set_static_property_value_in_owner(class_name: &str, prop: &str, value: Value) -> Result<(), String> {
    if let Some((_p, owner)) = lookup_property(class_name, prop) {
        set_static_property_value(&owner, prop, value);
        Ok(())
    } else {
        Err(format!("Unknown static property '{}.{}'", class_name, prop))
    }
}
