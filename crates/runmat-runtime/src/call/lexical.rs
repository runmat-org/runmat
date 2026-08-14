use runmat_types::BindingId;
use runmat_value::Value;

/// One binding-keyed value shared with a lexically nested function.
#[derive(Clone, Debug, PartialEq)]
pub struct LexicalCapture {
    pub binding: BindingId,
    pub value: Value,
}

/// Executor-neutral nested-function invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct LexicalCall {
    pub function: usize,
    pub captures: Vec<LexicalCapture>,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
}

/// Nested-function result plus the final value of every shared binding.
#[derive(Clone, Debug, PartialEq)]
pub struct LexicalCallResult {
    pub value: Value,
    pub captures: Vec<LexicalCapture>,
}
