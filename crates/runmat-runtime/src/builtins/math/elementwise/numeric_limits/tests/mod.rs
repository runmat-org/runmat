mod floating;
mod integer;
mod provider;
mod provider_rejection;
mod admission;

use runmat_value::Value;

use crate::BuiltinResult;

fn intmax(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::intmax::intmax_builtin(arguments)
}

fn intmin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::intmin::intmin_builtin(arguments)
}

fn realmax(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::realmax::realmax_builtin(arguments)
}

fn realmin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::realmin::realmin_builtin(arguments)
}

fn flintmax(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::flintmax::flintmax_builtin(arguments)
}
