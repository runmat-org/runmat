mod floating;
mod integer;
mod provider;
mod provider_rejection;
mod admission;
mod facts;

use runmat_value::Value;

use crate::BuiltinResult;

fn intmax(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::intmax::intmax_builtin(inputs(arguments)?)
}

fn intmin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::intmin::intmin_builtin(inputs(arguments)?)
}

fn realmax(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::realmax::realmax_builtin(inputs(arguments)?)
}

fn realmin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::realmin::realmin_builtin(inputs(arguments)?)
}

fn flintmax(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::flintmax::flintmax_builtin(inputs(arguments)?)
}

fn eps(arguments: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(super::eps::eps_builtin(None, inputs(arguments)?))
}

// These direct-body fixtures supply their already-created ordinary workspace
// inputs, independently of the output admission policy under test.
fn inputs(arguments: Vec<Value>) -> BuiltinResult<crate::call::BuiltinArguments> {
    crate::execution::RootedValues::new(arguments)
        .map(Into::into)
        .map_err(|error| error.into_runtime_error())
}
