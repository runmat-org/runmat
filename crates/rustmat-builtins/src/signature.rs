//! Function signature system for robust builtin definitions

use crate::Value;
use std::collections::HashMap;

/// Represents a function parameter with type constraints and default values
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub param_type: ParameterType,
    pub required: bool,
    pub default_value: Option<Value>,
    pub description: String,
}

/// Type constraints for function parameters
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterType {
    /// Single numeric value (int or float)
    Scalar,
    /// Matrix of any size
    Matrix,
    /// Matrix with specific constraints
    MatrixConstrained { min_rows: Option<usize>, min_cols: Option<usize> },
    /// Vector (1xN or Nx1 matrix)
    Vector,
    /// String value
    String,
    /// Boolean value
    Bool,
    /// Cell array
    Cell,
    /// Any type
    Any,
    /// One of multiple types
    Union(Vec<ParameterType>),
    /// Optional parameter
    Optional(Box<ParameterType>),
}

/// Function signature with parameter definitions
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub description: String,
    pub parameters: Vec<Parameter>,
    pub return_type: ParameterType,
    pub examples: Vec<String>,
    pub category: FunctionCategory,
}

/// Categories for organizing functions (like MATLAB toolboxes)
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionCategory {
    /// Core mathematical functions
    Mathematics,
    /// Linear algebra operations
    LinearAlgebra,
    /// Statistical functions
    Statistics,
    /// Signal processing
    SignalProcessing,
    /// Image processing
    ImageProcessing,
    /// Control systems
    ControlSystems,
    /// Optimization
    Optimization,
    /// Plotting and visualization
    Plotting,
    /// File I/O
    FileIO,
    /// String manipulation
    Strings,
    /// Data analysis
    DataAnalysis,
    /// Numerical methods
    Numerical,
    /// System utilities
    System,
    /// User-defined
    User,
}

/// Arguments passed to a builtin function with type validation
#[derive(Debug)]
pub struct Arguments {
    pub positional: Vec<Value>,
    pub named: HashMap<String, Value>,
}

impl Arguments {
    pub fn new(positional: Vec<Value>) -> Self {
        Self {
            positional,
            named: HashMap::new(),
        }
    }
    
    pub fn with_named(positional: Vec<Value>, named: HashMap<String, Value>) -> Self {
        Self { positional, named }
    }
    
    /// Get positional argument by index, with type validation
    pub fn get(&self, index: usize, expected_type: &ParameterType) -> Result<&Value, String> {
        let value = self.positional.get(index)
            .ok_or_else(|| format!("Missing argument at position {}", index))?;
        
        if self.validate_type(value, expected_type) {
            Ok(value)
        } else {
            Err(format!("Argument {} has wrong type", index))
        }
    }
    
    /// Get named argument with type validation
    pub fn get_named(&self, name: &str, expected_type: &ParameterType) -> Result<&Value, String> {
        let value = self.named.get(name)
            .ok_or_else(|| format!("Missing named argument '{}'", name))?;
        
        if self.validate_type(value, expected_type) {
            Ok(value)
        } else {
            Err(format!("Named argument '{}' has wrong type", name))
        }
    }
    
    /// Get optional argument with default
    pub fn get_optional(&self, index: usize, default: Value) -> Value {
        self.positional.get(index).cloned().unwrap_or(default)
    }
    
    fn validate_type(&self, value: &Value, param_type: &ParameterType) -> bool {
        match param_type {
            ParameterType::Scalar => matches!(value, Value::Num(_) | Value::Int(_)),
            ParameterType::Matrix => matches!(value, Value::Matrix(_)),
            ParameterType::Vector => {
                if let Value::Matrix(m) = value {
                    m.rows == 1 || m.cols == 1
                } else {
                    false
                }
            }
            ParameterType::String => matches!(value, Value::String(_)),
            ParameterType::Bool => matches!(value, Value::Bool(_)),
            ParameterType::Cell => matches!(value, Value::Cell(_)),
            ParameterType::Any => true,
            ParameterType::Union(types) => {
                types.iter().any(|t| self.validate_type(value, t))
            }
            ParameterType::Optional(inner) => self.validate_type(value, inner),
            ParameterType::MatrixConstrained { min_rows, min_cols } => {
                if let Value::Matrix(m) = value {
                    let rows_ok = min_rows.map_or(true, |min| m.rows >= min);
                    let cols_ok = min_cols.map_or(true, |min| m.cols >= min);
                    rows_ok && cols_ok
                } else {
                    false
                }
            }
        }
    }
}

/// Result of a builtin function call
pub type BuiltinResult = Result<Value, String>;

/// Enhanced builtin function type that takes Arguments
pub type BuiltinFunctionNew = fn(&Arguments) -> BuiltinResult;

/// Builtin function metadata
#[derive(Debug, Clone)]
pub struct BuiltinFunction {
    pub signature: FunctionSignature,
    pub implementation: BuiltinFunctionNew,
}

impl BuiltinFunction {
    pub fn new(
        name: &str,
        description: &str,
        category: FunctionCategory,
        parameters: Vec<Parameter>,
        return_type: ParameterType,
        implementation: BuiltinFunctionNew,
    ) -> Self {
        Self {
            signature: FunctionSignature {
                name: name.to_string(),
                description: description.to_string(),
                parameters,
                return_type,
                examples: Vec::new(),
                category,
            },
            implementation,
        }
    }
    
    /// Call the function with argument validation
    pub fn call(&self, args: &Arguments) -> BuiltinResult {
        // Validate argument count
        let expected_params = self.signature.parameters.len();
        let provided_args = args.len();
        
        // Allow variable argument functions (empty parameter list means any number)
        if !self.signature.parameters.is_empty() && provided_args != expected_params {
            return Err(format!(
                "Function '{}' expects {} arguments, got {}",
                self.name, expected_params, provided_args
            ));
        }
        
        // Validate parameter types if specified
        for (i, param) in self.signature.parameters.iter().enumerate() {
            if i >= args.len() {
                if param.required {
                    return Err(format!(
                        "Missing required parameter '{}' for function '{}'",
                        param.name, self.name
                    ));
                }
                break;
            }
            
            // Type validation could be added here
            // For now, we rely on runtime type checking in the function implementations
        }
        
        // Call the function
        (self.implementation)(args)
    }
}