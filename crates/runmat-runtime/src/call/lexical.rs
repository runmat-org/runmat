use runmat_gc::{Trace, Tracer};
use runmat_types::BindingId;
use runmat_value::{Value, ValueSequence};

#[path = "lexical/environment.rs"]
mod environment;
pub use environment::{
    allocate_environment, read_binding, remove_binding, remove_owned_binding, root_environment,
    write_binding, write_owned_binding,
};

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
    pub environment: Option<runmat_value::LexicalEnvironment>,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
}

impl Trace for LexicalCall {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for capture in &self.captures {
            capture.value.trace(tracer);
        }
        for argument in &self.arguments {
            argument.trace(tracer);
        }
        if let Some(environment) = &self.environment {
            environment.for_each_handle(|handle| tracer.mark(handle));
        }
    }
}

/// Nested-function result plus the final value of every shared binding.
#[derive(Clone, Debug, PartialEq)]
pub struct LexicalCallResult {
    pub outputs: ValueSequence,
    pub captures: Vec<LexicalCapture>,
}

pub type RootedLexicalCallResult = crate::execution::RootedResult<LexicalCallResult>;

impl Trace for LexicalCallResult {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.outputs.trace(tracer);
        for capture in &self.captures {
            capture.value.trace(tracer);
        }
    }
}
