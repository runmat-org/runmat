mod floating;
mod integer;
mod provider;

use runmat_value::Value;

use crate::BuiltinResult;

use super::operation::LimitOperation;

fn invoke(operation: LimitOperation, arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::execute::run(arguments, operation)
}

fn intmax(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke(LimitOperation::INTMAX, arguments)
}

fn intmin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke(LimitOperation::INTMIN, arguments)
}

fn realmax(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke(LimitOperation::REALMAX, arguments)
}

fn realmin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke(LimitOperation::REALMIN, arguments)
}

fn flintmax(arguments: Vec<Value>) -> BuiltinResult<Value> {
    invoke(LimitOperation::FLINTMAX, arguments)
}
