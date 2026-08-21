//! MATLAB-compatible `iscellstr` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_builtins::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input is a cell array of character arrays.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to inspect.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = iscellstr(C)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISCELLSTR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ISCELLSTR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "iscellstr is a universal container predicate; integer values or integer cell members return scalar false without numeric conversion." };

fn bool_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

#[runtime_builtin(
    name = "iscellstr",
    category = "cells/core",
    summary = "Return true for cell arrays of character vectors.",
    keywords = "iscellstr,cellstr,cell,char,type predicate",
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(crate::builtins::cells::core::iscellstr::ISCELLSTR_DESCRIPTOR),
    integer_audit(crate::builtins::cells::core::iscellstr::ISCELLSTR_INTEGER_AUDIT),
    builtin_path = "crate::builtins::cells::core::iscellstr"
)]
fn iscellstr_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let Value::Cell(cell) = value else {
        return Ok(Value::Bool(false));
    };
    Ok(Value::Bool(cell.data.iter().all(is_char_array)))
}

fn is_char_array(value: &Value) -> bool {
    matches!(value, Value::CharArray(CharArray { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::CellArray;

    #[test]
    fn accepts_empty_and_char_row_cells() {
        let empty = CellArray::new(Vec::new(), 0, 0).unwrap();
        assert_eq!(
            iscellstr_builtin(Value::Cell(empty)).unwrap(),
            Value::Bool(true)
        );

        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("a")),
                Value::CharArray(CharArray::new_row("bc")),
            ],
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            iscellstr_builtin(Value::Cell(cell)).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn rejects_non_cells_and_non_char_members() {
        assert_eq!(
            iscellstr_builtin(Value::Num(1.0)).unwrap(),
            Value::Bool(false)
        );
        let cell = CellArray::new(vec![Value::String("not char".into())], 1, 1).unwrap();
        assert_eq!(
            iscellstr_builtin(Value::Cell(cell)).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn accepts_character_matrices_inside_cells() {
        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let cell = CellArray::new(vec![Value::CharArray(chars)], 1, 1).unwrap();
        assert_eq!(
            iscellstr_builtin(Value::Cell(cell)).unwrap(),
            Value::Bool(true)
        );
    }
}
