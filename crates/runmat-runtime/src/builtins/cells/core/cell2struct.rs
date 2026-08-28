//! MATLAB-compatible `cell2struct` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, StructValue, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cell2struct";

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Struct or struct array result.",
}];

const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array containing field values.",
    },
    BuiltinParamDescriptor {
        name: "fields",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Field names as char, string, or cellstr.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Dimension whose entries correspond to field names.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "S = cell2struct(C, fields, dim)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2STRUCT.INVALID_INPUT",
    identifier: Some("RunMat:cell2struct:InvalidInput"),
    when: "Arguments are not a cell array, field-name list, and valid dimension.",
    message: "cell2struct: invalid input",
};

const ERROR_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2STRUCT.SHAPE",
    identifier: Some("RunMat:cell2struct:ShapeMismatch"),
    when: "The number of field names does not match the selected cell dimension.",
    message: "cell2struct: field count does not match selected dimension",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_SHAPE];

pub const CELL2STRUCT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const CELL2STRUCT_PAYLOAD_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "C payload",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer payload values are structural data and pass through without conversion, including resident handles nested in C.",
    }];
const CELL2STRUCT_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "dim",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "All eight integer scalar classes and an exact positive integral double are accepted; dim defaults to 1.",
}];

pub const CELL2STRUCT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "S = cell2struct(C, fields, dim) with integer payload",
        inputs: &CELL2STRUCT_PAYLOAD_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Payloads are cloned exactly into fields without eager gather. RunMat currently represents nonscalar struct arrays as cells of scalar structs; that representation mismatch is separate from payload preservation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = cell2struct(C, fields, integer_dim)",
        inputs: &CELL2STRUCT_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "dim is decoded exactly from authoritative integer storage and must be positive and representable as usize.",
    },
];

#[runtime_builtin(
    name = "cell2struct",
    category = "cells/core",
    summary = "Convert a cell array into a scalar struct or struct array.",
    keywords = "cell2struct,cell,struct,conversion",
    accel = "gather",
    descriptor(crate::builtins::cells::core::cell2struct::CELL2STRUCT_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::cells::core::cell2struct::CELL2STRUCT_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::cells::core::cell2struct"
)]
fn cell2struct_builtin(cells: Value, fields: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "cell2struct: expected C, fields, and optional dim",
        ));
    }
    let dim = rest.first().cloned().unwrap_or(Value::Num(1.0));
    let Value::Cell(cells) = cells else {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "cell2struct: C must be a cell array",
        ));
    };
    let fields = field_names(&fields)?;
    let dim = parse_dim(&dim)?;
    build_structs(cells, fields, dim)
}

fn build_structs(cells: CellArray, fields: Vec<String>, dim: usize) -> BuiltinResult<Value> {
    if fields.is_empty() {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "cell2struct: fields must not be empty",
        ));
    }
    let rank = cells.shape.len().max(dim);
    let mut shape = cells.shape.clone();
    shape
        .try_reserve_exact(rank.saturating_sub(shape.len()))
        .map_err(|_| {
            error(
                &ERROR_INVALID_INPUT,
                "cell2struct: dim exceeds platform limits",
            )
        })?;
    shape.resize(rank, 1);
    let field_dim = dim - 1;
    if shape[field_dim] != fields.len() {
        return Err(error(
            &ERROR_SHAPE,
            format!(
                "cell2struct: selected dimension has extent {}, but {} field names were supplied",
                shape[field_dim],
                fields.len()
            ),
        ));
    }

    let mut out_shape = shape.clone();
    out_shape[field_dim] = 1;
    let output_count = out_shape.iter().try_fold(1usize, |count, extent| {
        count.checked_mul(*extent).ok_or_else(|| {
            error(
                &ERROR_INVALID_INPUT,
                "cell2struct: output shape exceeds platform limits",
            )
        })
    })?;
    if output_count == 1 {
        let mut st = StructValue::new();
        for (field_idx, field) in fields.iter().enumerate() {
            let mut coords = vec![0usize; rank];
            coords[field_dim] = field_idx;
            st.insert(
                field.clone(),
                cells.data[linear_col_major(&coords, &shape)].clone(),
            );
        }
        return Ok(Value::Struct(st));
    }

    let mut structs = Vec::new();
    structs.try_reserve_exact(output_count).map_err(|_| {
        error(
            &ERROR_INVALID_INPUT,
            "cell2struct: output shape exceeds platform limits",
        )
    })?;
    for out_linear in 0..output_count {
        let out_coords = coords_col_major(out_linear, &out_shape);
        let mut st = StructValue::new();
        for (field_idx, field) in fields.iter().enumerate() {
            let mut coords = out_coords.clone();
            coords[field_dim] = field_idx;
            st.insert(
                field.clone(),
                cells.data[linear_col_major(&coords, &shape)].clone(),
            );
        }
        structs.push(Value::Struct(st));
    }
    if structs.len() == 1 {
        return Ok(structs.pop().expect("one struct"));
    }
    CellArray::new_with_shape(structs, out_shape)
        .map(Value::Cell)
        .map_err(|err| error(&ERROR_INVALID_INPUT, format!("cell2struct: {err}")))
}

fn field_names(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(vec![chars.data.iter().collect()]),
        Value::CharArray(chars) => {
            let mut names = Vec::with_capacity(chars.rows);
            for row in 0..chars.rows {
                let start = row * chars.cols;
                let text = chars.data[start..start + chars.cols]
                    .iter()
                    .collect::<String>()
                    .trim_end()
                    .to_string();
                names.push(text);
            }
            Ok(names)
        }
        Value::Cell(cell) => cell.data.iter().map(field_name_scalar).collect(),
        _ => Err(error(
            &ERROR_INVALID_INPUT,
            "cell2struct: fields must be text or cellstr",
        )),
    }
    .and_then(|names| {
        if names.iter().any(|name| name.is_empty()) {
            Err(error(
                &ERROR_INVALID_INPUT,
                "cell2struct: field names must not be empty",
            ))
        } else {
            Ok(names)
        }
    })
}

fn field_name_scalar(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(StringArray { data, .. }) if data.len() == 1 => Ok(data[0].clone()),
        Value::CharArray(CharArray { rows: 1, data, .. }) => Ok(data.iter().collect()),
        _ => Err(error(
            &ERROR_INVALID_INPUT,
            "cell2struct: cell field names must be text scalars",
        )),
    }
}

fn parse_dim(value: &Value) -> BuiltinResult<usize> {
    if let Some(value) = tensor::scalar_integer_value(value) {
        return value
            .try_to_usize()
            .filter(|value| *value >= 1)
            .ok_or_else(|| {
                error(
                    &ERROR_INVALID_INPUT,
                    "cell2struct: dim must be a positive integer",
                )
            });
    }
    let raw = match value {
        Value::Num(value) if value.is_finite() => *value,
        _ => {
            return Err(error(
                &ERROR_INVALID_INPUT,
                "cell2struct: dim must be a positive integer",
            ))
        }
    };
    if raw < 1.0 || raw.fract() != 0.0 {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "cell2struct: dim must be a positive integer",
        ));
    }
    if raw > usize::MAX as f64 || (usize::BITS == 64 && raw == usize::MAX as f64) {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "cell2struct: dim exceeds platform limits",
        ));
    }
    Ok(raw as usize)
}

fn coords_col_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for &extent in shape {
        coords.push(if extent == 0 { 0 } else { linear % extent });
        if extent != 0 {
            linear /= extent;
        }
    }
    coords
}

fn linear_col_major(coords: &[usize], shape: &[usize]) -> usize {
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (&coord, &extent) in coords.iter().zip(shape.iter()) {
        linear += coord * stride;
        stride *= extent;
    }
    linear
}

fn error(desc: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = desc.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_value::{IntValue, Tensor};

    fn cell2struct_builtin(cells: Value, fields: Value, dim: Value) -> BuiltinResult<Value> {
        super::cell2struct_builtin(cells, fields, vec![dim])
    }

    fn cell2struct_default(cells: Value, fields: Value) -> BuiltinResult<Value> {
        super::cell2struct_builtin(cells, fields, Vec::new())
    }

    #[test]
    fn scalar_struct_from_row_cell() {
        let cells = CellArray::new(vec![Value::Num(1.0), Value::from("Ada")], 1, 2).unwrap();
        let fields = CellArray::new(vec![Value::from("id"), Value::from("name")], 1, 2).unwrap();
        let out =
            cell2struct_builtin(Value::Cell(cells), Value::Cell(fields), Value::Num(2.0)).unwrap();
        let Value::Struct(st) = out else {
            panic!("expected scalar struct");
        };
        assert_eq!(st.fields.get("id"), Some(&Value::Num(1.0)));
        assert_eq!(st.fields.get("name"), Some(&Value::from("Ada")));
    }

    #[test]
    fn default_dimension_is_one_and_payload_is_exact() {
        let cells = CellArray::new(
            vec![
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::I64(i64::MIN)),
            ],
            2,
            1,
        )
        .unwrap();
        let fields = CellArray::new(vec![Value::from("hi"), Value::from("lo")], 2, 1).unwrap();
        let Value::Struct(st) =
            cell2struct_default(Value::Cell(cells), Value::Cell(fields)).unwrap()
        else {
            panic!("expected scalar struct");
        };
        assert_eq!(
            st.fields.get("hi"),
            Some(&Value::Int(IntValue::U64(u64::MAX)))
        );
        assert_eq!(
            st.fields.get("lo"),
            Some(&Value::Int(IntValue::I64(i64::MIN)))
        );
    }

    #[test]
    fn every_integer_dim_class_is_accepted_exactly() {
        let dims = [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ];
        for dim in dims {
            let cells = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
            let fields = CellArray::new(vec![Value::from("id")], 1, 1).unwrap();
            assert!(matches!(
                cell2struct_builtin(Value::Cell(cells), Value::Cell(fields), Value::Int(dim))
                    .unwrap(),
                Value::Struct(_)
            ));
        }
    }

    #[test]
    fn every_integer_payload_class_passes_through_exactly() {
        for payload in [
            IntValue::I8(-7),
            IntValue::I16(-7),
            IntValue::I32(-7),
            IntValue::I64(-7),
            IntValue::U8(7),
            IntValue::U16(7),
            IntValue::U32(7),
            IntValue::U64(7),
        ] {
            let cells = CellArray::new(vec![Value::Int(payload.clone())], 1, 1).unwrap();
            let fields = CellArray::new(vec![Value::from("value")], 1, 1).unwrap();
            let Value::Struct(st) =
                cell2struct_default(Value::Cell(cells), Value::Cell(fields)).unwrap()
            else {
                panic!("expected scalar struct");
            };
            assert_eq!(st.fields.get("value"), Some(&Value::Int(payload)));
        }
    }

    #[test]
    fn resident_payload_handle_passes_through_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![7.0], vec![1, 1]).unwrap();
            let data = tensor.materialize_f64();
            let view = runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).unwrap();
            let cells = CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).unwrap();
            let fields = CellArray::new(vec![Value::from("resident")], 1, 1).unwrap();
            let Value::Struct(st) =
                cell2struct_default(Value::Cell(cells), Value::Cell(fields)).unwrap()
            else {
                panic!("expected struct");
            };
            assert!(
                matches!(st.fields.get("resident"), Some(Value::GpuTensor(actual)) if actual == &handle)
            );
        });
    }

    #[test]
    fn struct_array_from_field_dimension() {
        let cells = CellArray::new(
            vec![
                Value::Num(1.0),
                Value::from("Ada"),
                Value::Num(2.0),
                Value::from("Grace"),
            ],
            2,
            2,
        )
        .unwrap();
        let fields = CellArray::new(vec![Value::from("id"), Value::from("name")], 2, 1).unwrap();
        let out =
            cell2struct_builtin(Value::Cell(cells), Value::Cell(fields), Value::Num(1.0)).unwrap();
        let Value::Cell(out) = out else {
            panic!("expected struct array cell");
        };
        assert_eq!(out.shape, vec![1, 2]);
        assert!(
            matches!(&out.data[0], Value::Struct(st) if st.fields.get("id") == Some(&Value::Num(1.0)))
        );
        assert!(
            matches!(&out.data[1], Value::Struct(st) if st.fields.get("name") == Some(&Value::from("Grace")))
        );
    }

    #[test]
    fn typed_integer_dimensions_are_exactly_validated() {
        let cells = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let fields = CellArray::new(vec![Value::from("id")], 1, 1).unwrap();
        let out = cell2struct_builtin(
            Value::Cell(cells.clone()),
            Value::Cell(fields.clone()),
            Value::Int(runmat_value::IntValue::U8(1)),
        )
        .unwrap();
        assert!(matches!(out, Value::Struct(_)));

        let err = cell2struct_builtin(
            Value::Cell(cells),
            Value::Cell(fields),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("platform limits"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn typed_tensor_dimensions_are_exactly_validated() {
        let cells = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let fields = CellArray::new(vec![Value::from("id")], 1, 1).unwrap();
        let dim = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U16(vec![1]),
            vec![1, 1],
        )
        .expect("typed dim");

        let out = cell2struct_builtin(
            Value::Cell(cells.clone()),
            Value::Cell(fields.clone()),
            Value::Tensor(dim),
        )
        .unwrap();
        assert!(matches!(out, Value::Struct(_)));

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        let err = cell2struct_builtin(
            Value::Cell(cells),
            Value::Cell(fields),
            Value::Num(boundary),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("platform limits"),
            "unexpected error message: {err}"
        );
    }
}
