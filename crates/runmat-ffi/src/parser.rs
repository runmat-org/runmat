//! Parser for `.ffi` signature files.
//!
//! # File Format
//!
//! ```text
//! # Comment lines start with #
//!
//! # Function signature: name: (arg_types) -> return_type
//! add: (f64, f64) -> f64
//! square: (f64) -> f64
//! get_pi: () -> f64
//!
//! # Pointer types for array functions
//! scale_array: (ptr<f64>, usize, usize, f64, ptr_mut<f64>) -> i32
//! ```
//!
//! # Supported Types
//!
//! - `f64`, `double` - 64-bit float
//! - `f32`, `float` - 32-bit float
//! - `i32`, `int` - 32-bit signed integer
//! - `i64` - 64-bit signed integer
//! - `u32` - 32-bit unsigned integer
//! - `usize`, `size_t` - pointer-sized unsigned integer
//! - `ptr<T>` - immutable pointer to T
//! - `ptr_mut<T>` - mutable pointer to T
//! - `void` - no return value

use crate::types::{FfiSignature, FfiType};
use std::collections::HashMap;
use std::path::Path;

/// A collection of parsed function signatures for a library.
#[derive(Debug, Clone, Default)]
pub struct SignatureFile {
    /// Function signatures indexed by name
    pub signatures: HashMap<String, FfiSignature>,
}

impl SignatureFile {
    /// Create an empty signature file.
    pub fn new() -> Self {
        Self {
            signatures: HashMap::new(),
        }
    }

    /// Parse a `.ffi` file from a path.
    pub fn parse_file(path: impl AsRef<Path>) -> Result<Self, ParseError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| ParseError {
            line: 0,
            message: format!("Failed to read file: {}", e),
        })?;
        Self::parse(&content)
    }

    /// Parse `.ffi` content from a string.
    pub fn parse(content: &str) -> Result<Self, ParseError> {
        let mut signatures = HashMap::new();

        for (line_num, line) in content.lines().enumerate() {
            let line_num = line_num + 1; // 1-indexed for error messages
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let sig = parse_signature_line(line, line_num)?;
            signatures.insert(sig.name.clone(), sig);
        }

        Ok(Self { signatures })
    }

    /// Get a signature by function name.
    pub fn get(&self, name: &str) -> Option<&FfiSignature> {
        self.signatures.get(name)
    }

    /// Check if a signature exists.
    pub fn contains(&self, name: &str) -> bool {
        self.signatures.contains_key(name)
    }

    /// Iterate over all signatures.
    pub fn iter(&self) -> impl Iterator<Item = &FfiSignature> {
        self.signatures.values()
    }
}

/// Error during signature file parsing.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line > 0 {
            write!(f, "line {}: {}", self.line, self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse a single signature line: `name: (arg_types) -> return_type`
fn parse_signature_line(line: &str, line_num: usize) -> Result<FfiSignature, ParseError> {
    // Split on ':' to get name and signature
    let colon_pos = line.find(':').ok_or_else(|| ParseError {
        line: line_num,
        message: "Expected ':' after function name".to_string(),
    })?;

    let name = line[..colon_pos].trim().to_string();
    if name.is_empty() {
        return Err(ParseError {
            line: line_num,
            message: "Function name cannot be empty".to_string(),
        });
    }

    let rest = line[colon_pos + 1..].trim();

    // Parse (arg_types) -> return_type
    let (args, ret) = parse_type_signature(rest, line_num)?;

    Ok(FfiSignature::new(name, args, ret))
}

/// Parse `(arg_types) -> return_type`
fn parse_type_signature(s: &str, line_num: usize) -> Result<(Vec<FfiType>, FfiType), ParseError> {
    // Find the opening paren
    if !s.starts_with('(') {
        return Err(ParseError {
            line: line_num,
            message: "Expected '(' at start of type signature".to_string(),
        });
    }

    // Find matching closing paren (handle nested generics)
    let close_paren = find_matching_paren(s, 0).ok_or_else(|| ParseError {
        line: line_num,
        message: "Unmatched '(' in type signature".to_string(),
    })?;

    let args_str = &s[1..close_paren];
    let rest = s[close_paren + 1..].trim();

    // Parse arrow and return type
    let rest = rest.strip_prefix("->").ok_or_else(|| ParseError {
        line: line_num,
        message: "Expected '->' after argument list".to_string(),
    })?;
    let ret_str = rest.trim();

    // Parse argument types
    let args = parse_arg_list(args_str, line_num)?;

    // Parse return type
    let ret = FfiType::parse(ret_str).ok_or_else(|| ParseError {
        line: line_num,
        message: format!("Unknown return type: '{}'", ret_str),
    })?;

    Ok((args, ret))
}

/// Parse comma-separated argument list.
fn parse_arg_list(s: &str, line_num: usize) -> Result<Vec<FfiType>, ParseError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(vec![]);
    }

    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in s.chars() {
        match ch {
            '<' => {
                depth += 1;
                current.push(ch);
            }
            '>' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                let arg_type = FfiType::parse(current.trim()).ok_or_else(|| ParseError {
                    line: line_num,
                    message: format!("Unknown argument type: '{}'", current.trim()),
                })?;
                args.push(arg_type);
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }

    // Don't forget the last argument
    if !current.trim().is_empty() {
        let arg_type = FfiType::parse(current.trim()).ok_or_else(|| ParseError {
            line: line_num,
            message: format!("Unknown argument type: '{}'", current.trim()),
        })?;
        args.push(arg_type);
    }

    Ok(args)
}

/// Find the position of the matching closing parenthesis.
fn find_matching_paren(s: &str, open_pos: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, ch) in s[open_pos..].char_indices() {
        match ch {
            '(' | '<' => depth += 1,
            ')' | '>' if depth > 1 => depth -= 1,
            ')' if depth == 1 => return Some(open_pos + i),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_signature() {
        let sig = parse_signature_line("add: (f64, f64) -> f64", 1).unwrap();
        assert_eq!(sig.name, "add");
        assert_eq!(sig.args, vec![FfiType::F64, FfiType::F64]);
        assert_eq!(sig.ret, FfiType::F64);
    }

    #[test]
    fn test_parse_nullary() {
        let sig = parse_signature_line("get_pi: () -> f64", 1).unwrap();
        assert_eq!(sig.name, "get_pi");
        assert!(sig.args.is_empty());
        assert_eq!(sig.ret, FfiType::F64);
    }

    #[test]
    fn test_parse_unary() {
        let sig = parse_signature_line("square: (f64) -> f64", 1).unwrap();
        assert_eq!(sig.name, "square");
        assert_eq!(sig.args, vec![FfiType::F64]);
        assert_eq!(sig.ret, FfiType::F64);
    }

    #[test]
    fn test_parse_pointer_types() {
        let sig =
            parse_signature_line("scale: (ptr<f64>, usize, f64, ptr_mut<f64>) -> i32", 1).unwrap();
        assert_eq!(sig.name, "scale");
        assert_eq!(
            sig.args,
            vec![
                FfiType::Ptr(Box::new(FfiType::F64)),
                FfiType::Usize,
                FfiType::F64,
                FfiType::PtrMut(Box::new(FfiType::F64)),
            ]
        );
        assert_eq!(sig.ret, FfiType::I32);
    }

    #[test]
    fn test_parse_file_content() {
        let content = r#"
# Math functions
add: (f64, f64) -> f64
square: (f64) -> f64
get_pi: () -> f64

# Array operations
sum5: (f64, f64, f64, f64, f64) -> f64
"#;
        let file = SignatureFile::parse(content).unwrap();
        assert_eq!(file.signatures.len(), 4);
        assert!(file.contains("add"));
        assert!(file.contains("square"));
        assert!(file.contains("get_pi"));
        assert!(file.contains("sum5"));
    }

    #[test]
    fn test_parse_void_return() {
        let sig = parse_signature_line("init: (i32) -> void", 1).unwrap();
        assert_eq!(sig.name, "init");
        assert_eq!(sig.args, vec![FfiType::I32]);
        assert_eq!(sig.ret, FfiType::Void);
    }
}
