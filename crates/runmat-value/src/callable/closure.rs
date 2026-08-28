use crate::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    pub function_name: String,
    pub bound_function: Option<usize>,
    pub captures: Vec<Value>,
}
