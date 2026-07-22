//! MATLAB-compatible `iscell` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input is a cell array.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to inspect.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = iscell(A)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISCELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn bool_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

#[runtime_builtin(
    name = "iscell",
    category = "cells/core",
    summary = "Return true when a value is a cell array.",
    keywords = "iscell,cell,type predicate",
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(crate::builtins::cells::core::iscell::ISCELL_DESCRIPTOR),
    builtin_path = "crate::builtins::cells::core::iscell"
)]
fn iscell_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(matches!(value, Value::Cell(_))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{CellArray, Tensor};

    #[test]
    fn detects_cell_arrays_only() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        assert_eq!(
            iscell_builtin(Value::Cell(cell)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            iscell_builtin(Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap())).unwrap(),
            Value::Bool(false)
        );
    }
}
