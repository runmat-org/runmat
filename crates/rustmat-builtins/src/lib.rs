pub use inventory;
use std::convert::TryFrom;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i32),
    Num(f64),
    Bool(bool),
    String(String),
    Matrix(Matrix),
    Cell(Vec<Value>), // Cell arrays
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "Matrix data length {} doesn't match dimensions {}x{}",
                data.len(),
                rows,
                cols
            ));
        }
        Ok(Matrix { data, rows, cols })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![1.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Result<f64, String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            ));
        }
        Ok(self.data[row * self.cols + col])
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                row, col, self.rows, self.cols
            ));
        }
        self.data[row * self.cols + col] = value;
        Ok(())
    }

    pub fn scalar_to_matrix(scalar: f64, rows: usize, cols: usize) -> Matrix {
        Matrix {
            data: vec![scalar; rows * cols],
            rows,
            cols,
        }
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

impl From<Matrix> for Value {
    fn from(m: Matrix) -> Self {
        Value::Matrix(m)
    }
}

impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::Cell(v)
    }
}

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
            Value::Int(i) => Ok(i.to_string()),
            Value::Num(n) => Ok(n.to_string()),
            Value::Bool(b) => Ok(b.to_string()),
            _ => Err(format!("cannot convert {v:?} to String")),
        }
    }
}

impl TryFrom<&Value> for Matrix {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Matrix(m) => Ok(m.clone()),
            _ => Err(format!("cannot convert {v:?} to Matrix")),
        }
    }
}

impl TryFrom<&Value> for Vec<Value> {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Cell(c) => Ok(c.clone()),
            _ => Err(format!("cannot convert {v:?} to Vec<Value>")),
        }
    }
}

use serde::{Deserialize, Serialize};

/// Enhanced type system used throughout RustMat for HIR and builtin functions
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
    /// Matrix type with optional dimension information
    Matrix {
        /// Optional number of rows (None means unknown/dynamic)
        rows: Option<usize>,
        /// Optional number of columns (None means unknown/dynamic)
        cols: Option<usize>,
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
}

impl Type {
    /// Create a matrix type with unknown dimensions
    pub fn matrix() -> Self {
        Type::Matrix { rows: None, cols: None }
    }
    
    /// Create a matrix type with known dimensions
    pub fn matrix_with_dims(rows: usize, cols: usize) -> Self {
        Type::Matrix { rows: Some(rows), cols: Some(cols) }
    }
    
    /// Create a cell array type with unknown element type
    pub fn cell() -> Self {
        Type::Cell { element_type: None, length: None }
    }
    
    /// Create a cell array type with known element type
    pub fn cell_of(element_type: Type) -> Self {
        Type::Cell { element_type: Some(Box::new(element_type)), length: None }
    }
    
    /// Check if this type is compatible with another type
    pub fn is_compatible_with(&self, other: &Type) -> bool {
        match (self, other) {
            (Type::Unknown, _) | (_, Type::Unknown) => true,
            (Type::Int, Type::Num) | (Type::Num, Type::Int) => true, // Number compatibility
            (Type::Matrix { .. }, Type::Matrix { .. }) => true, // Matrix compatibility regardless of dims
            (a, b) => a == b,
        }
    }
    
    /// Get the most specific common type between two types
    pub fn unify(&self, other: &Type) -> Type {
        match (self, other) {
            (Type::Unknown, t) | (t, Type::Unknown) => t.clone(),
            (Type::Int, Type::Num) | (Type::Num, Type::Int) => Type::Num,
            (Type::Matrix { .. }, Type::Matrix { .. }) => Type::matrix(), // Lose dimension info
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
            Value::Matrix(m) => Type::Matrix { 
                rows: Some(m.rows), 
                cols: Some(m.cols) 
            },
            Value::Cell(cells) => {
                if cells.is_empty() {
                    Type::cell()
                } else {
                    // Infer element type from first element
                    let element_type = Type::from_value(&cells[0]);
                    Type::Cell { 
                        element_type: Some(Box::new(element_type)), 
                        length: Some(cells.len()) 
                    }
                }
            }
        }
    }
}

/// Simple builtin function definition using the unified type system
#[derive(Debug, Clone)]
pub struct BuiltinFunction {
    pub name: &'static str,
    pub description: &'static str,
    pub param_types: Vec<Type>,
    pub return_type: Type,
    pub implementation: fn(&[Value]) -> Result<Value, String>,
}

impl BuiltinFunction {
    pub fn new(
        name: &'static str,
        description: &'static str,
        param_types: Vec<Type>,
        return_type: Type,
        implementation: fn(&[Value]) -> Result<Value, String>,
    ) -> Self {
        Self {
            name,
            description,
            param_types,
            return_type,
            implementation,
        }
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
        write!(f, "Constant {{ name: {:?}, value: {:?} }}", self.name, self.value)
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