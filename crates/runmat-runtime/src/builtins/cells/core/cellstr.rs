//! MATLAB-compatible `cellstr` builtin implemented for the modern RunMat runtime.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, StringArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::cells::type_resolvers::cellstr_type;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::dispatcher::gather_if_needed_async;
use crate::{build_runtime_error, make_cell, make_cell_with_shape, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cellstr";

const CELLSTR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of character vectors.",
}];

const CELLSTR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character array, string array, string scalar, or text cell array.",
}];

const CELLSTR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = cellstr(str)",
    inputs: &CELLSTR_INPUTS,
    outputs: &CELLSTR_OUTPUT,
}];

const CELLSTR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLSTR.INVALID_INPUT",
    identifier: Some("RunMat:cellstr:InvalidInput"),
    when: "The input value is not a supported text container.",
    message: "cellstr: input must be text-compatible",
};

const CELLSTR_ERROR_INVALID_CONTENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLSTR.INVALID_CONTENTS",
    identifier: Some("RunMat:cellstr:InvalidContents"),
    when: "Cell elements are not valid character vectors or string scalars.",
    message: "cellstr: cell array elements must be text scalars",
};

const CELLSTR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLSTR.INTERNAL",
    identifier: None,
    when: "Internal conversion or allocation failed.",
    message: "cellstr: internal error",
};

const CELLSTR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CELLSTR_ERROR_INVALID_INPUT,
    CELLSTR_ERROR_INVALID_CONTENTS,
    CELLSTR_ERROR_INTERNAL,
];

pub const CELLSTR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CELLSTR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CELLSTR_ERRORS,
};

const CELLSTR_INPUT_NOT_TEXT_TEXT: &str =
    "cellstr: input must be a character array, string array, or cell array of character vectors";
const CELLSTR_CELL_CONTENT_NOT_TEXT_TEXT: &str =
    "cellstr: cell array elements must be character vectors or string scalars";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::cells::core::cellstr")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cellstr",
    op_kind: GpuOpKind::Custom("text-convert"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only text conversion. Inputs originating on the GPU are gathered before processing, and the output is always a host cell array.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::cells::core::cellstr")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cellstr",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Terminates fusion because the result is a host-resident cell array of character vectors.",
};

fn cellstr_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "cellstr",
    category = "cells/core",
    summary = "Convert text values to a cell array of character vectors.",
    keywords = "cellstr,text,character,string,conversion",
    accel = "gather",
    type_resolver(cellstr_type),
    descriptor(crate::builtins::cells::core::cellstr::CELLSTR_DESCRIPTOR),
    builtin_path = "crate::builtins::cells::core::cellstr"
)]
async fn cellstr_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let host = gather_if_needed_async(&value).await?;
    match host {
        Value::CharArray(ca) => cellstr_from_char_array(ca),
        Value::StringArray(sa) => cellstr_from_string_array(sa),
        Value::String(text) => cellstr_from_string(text),
        Value::Cell(cell) => cellstr_from_cell(cell).await,
        Value::LogicalArray(_)
        | Value::Bool(_)
        | Value::Int(_)
        | Value::Num(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(cellstr_error_with_message(
            CELLSTR_INPUT_NOT_TEXT_TEXT,
            &CELLSTR_ERROR_INVALID_INPUT,
        )),
        Value::GpuTensor(_) => Err(cellstr_error_with_message(
            "cellstr: input must be gathered to the host before conversion",
            &CELLSTR_ERROR_INVALID_INPUT,
        )),
    }
}

fn cellstr_from_string(text: String) -> BuiltinResult<Value> {
    let row = Value::CharArray(CharArray::new_row(&text));
    make_cell(vec![row], 1, 1)
        .map_err(|e| cellstr_error_with_message(format!("cellstr: {e}"), &CELLSTR_ERROR_INTERNAL))
}

fn cellstr_from_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let rows = ca.rows;
    let cols = ca.cols;
    if rows == 0 {
        return make_cell(Vec::new(), 0, 1).map_err(|e| {
            cellstr_error_with_message(format!("cellstr: {e}"), &CELLSTR_ERROR_INTERNAL)
        });
    }
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let slice = &ca.data[start..end];
        let trimmed = trim_trailing_spaces(slice);
        values.push(Value::CharArray(CharArray::new_row(&trimmed)));
    }
    make_cell(values, rows, 1)
        .map_err(|e| cellstr_error_with_message(format!("cellstr: {e}"), &CELLSTR_ERROR_INTERNAL))
}

fn cellstr_from_string_array(sa: StringArray) -> BuiltinResult<Value> {
    let shape = if sa.shape.is_empty() {
        vec![sa.rows.max(1), sa.cols.max(1)]
    } else {
        sa.shape.clone()
    };
    let total = shape.iter().product::<usize>();
    if total == 0 {
        return make_cell_with_shape(Vec::new(), shape).map_err(|e| {
            cellstr_error_with_message(format!("cellstr: {e}"), &CELLSTR_ERROR_INTERNAL)
        });
    }
    if total != sa.data.len() {
        return Err(cellstr_error_with_message(
            "cellstr: internal string array shape mismatch",
            &CELLSTR_ERROR_INVALID_INPUT,
        ));
    }
    let mut values = Vec::with_capacity(total);
    for row_major in 0..total {
        let coords = linear_to_multi_row_major(row_major, &shape);
        let column_major = multi_to_linear_column_major(&coords, &shape);
        let text = sa.data[column_major].clone();
        values.push(Value::CharArray(CharArray::new_row(&text)));
    }
    make_cell_with_shape(values, shape)
        .map_err(|e| cellstr_error_with_message(format!("cellstr: {e}"), &CELLSTR_ERROR_INTERNAL))
}

async fn cellstr_from_cell(cell: CellArray) -> BuiltinResult<Value> {
    let mut values = Vec::with_capacity(cell.data.len());
    for ptr in &cell.data {
        let element = unsafe { &*ptr.as_raw() };
        let gathered = gather_if_needed_async(element).await?;
        values.push(coerce_to_char_vector(gathered)?);
    }
    make_cell_with_shape(values, cell.shape.clone())
        .map_err(|e| cellstr_error_with_message(format!("cellstr: {e}"), &CELLSTR_ERROR_INTERNAL))
}

fn coerce_to_char_vector(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::CharArray(ca) => {
            if ca.rows == 1 || (ca.rows == 0 && ca.cols == 0) {
                Ok(Value::CharArray(ca))
            } else {
                Err(cellstr_error_with_message(
                    CELLSTR_CELL_CONTENT_NOT_TEXT_TEXT,
                    &CELLSTR_ERROR_INVALID_CONTENTS,
                ))
            }
        }
        Value::String(text) => Ok(Value::CharArray(CharArray::new_row(&text))),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(Value::CharArray(CharArray::new_row(&sa.data[0])))
            } else {
                Err(cellstr_error_with_message(
                    CELLSTR_CELL_CONTENT_NOT_TEXT_TEXT,
                    &CELLSTR_ERROR_INVALID_CONTENTS,
                ))
            }
        }
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::GpuTensor(_) => Err(cellstr_error_with_message(
            CELLSTR_CELL_CONTENT_NOT_TEXT_TEXT,
            &CELLSTR_ERROR_INVALID_CONTENTS,
        )),
        Value::Cell(_) | Value::Struct(_) | Value::Object(_) | Value::HandleObject(_) => {
            Err(cellstr_error_with_message(
                CELLSTR_CELL_CONTENT_NOT_TEXT_TEXT,
                &CELLSTR_ERROR_INVALID_CONTENTS,
            ))
        }
        other => Err(cellstr_error_with_message(
            format!("cellstr: unsupported cell element {other:?}"),
            &CELLSTR_ERROR_INVALID_CONTENTS,
        )),
    }
}

fn trim_trailing_spaces(chars: &[char]) -> String {
    let mut end = chars.len();
    while end > 0 && chars[end - 1] == ' ' {
        end -= 1;
    }
    chars[..end].iter().collect()
}

fn linear_to_multi_row_major(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut coords = vec![0usize; shape.len()];
    for (dim, &extent) in shape.iter().enumerate().rev() {
        if extent == 0 {
            coords[dim] = 0;
        } else {
            coords[dim] = index % extent;
            index /= extent;
        }
    }
    coords
}

fn multi_to_linear_column_major(coords: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for (dim, &coord) in coords.iter().enumerate() {
        let extent = shape[dim];
        if extent == 0 {
            return 0;
        }
        index += coord * stride;
        stride *= extent;
    }
    index
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn cellstr_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::cellstr_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn descriptor_signatures_cover_cellstr_forms() {
        let labels: Vec<&str> = CELLSTR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert_eq!(labels, vec!["C = cellstr(str)"]);
    }

    fn cell_to_strings(cell: &CellArray) -> Vec<String> {
        cell.data
            .iter()
            .map(|ptr| match unsafe { &*ptr.as_raw() } {
                Value::CharArray(ca) => ca.data.iter().collect(),
                other => panic!("expected CharArray in cell, found {other:?}"),
            })
            .collect()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn converts_char_matrix_and_trims() {
        let data: Vec<char> = vec!['c', 'a', 't', ' ', 'd', 'o', 'g', ' ', 'f', 'o', 'x', ' '];
        let ca = CharArray::new(data, 3, 4).expect("char array");
        let result = cellstr_builtin(Value::CharArray(ca)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 1);
                let rows = cell_to_strings(&cell);
                assert_eq!(
                    rows,
                    vec!["cat".to_string(), "dog".to_string(), "fox".to_string()]
                );
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn converts_string_array_with_shape() {
        let data = vec![
            "north".to_string(),
            "south".to_string(),
            "east".to_string(),
            "west".to_string(),
        ];
        let sa = StringArray::new(data, vec![2, 2]).expect("string array");
        let result = cellstr_builtin(Value::StringArray(sa)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 2);
                let rows = cell_to_strings(&cell);
                assert_eq!(
                    rows,
                    vec![
                        "north".to_string(),
                        "east".to_string(),
                        "south".to_string(),
                        "west".to_string(),
                    ]
                );
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn converts_string_scalar() {
        let result = cellstr_builtin(Value::String("RunMat".to_string())).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                let rows = cell_to_strings(&cell);
                assert_eq!(rows, vec!["RunMat".to_string()]);
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn normalises_cell_elements() {
        let alpha = Value::CharArray(CharArray::new_row("alpha"));
        let beta = Value::String("beta".to_string());
        let cell = crate::make_cell(vec![alpha, beta], 1, 2).expect("cell");
        let result = cellstr_builtin(cell).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 2);
                let rows = cell_to_strings(&cell);
                assert_eq!(rows, vec!["alpha".to_string(), "beta".to_string()]);
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_text_cell_element() {
        let cell = crate::make_cell(vec![Value::Num(1.0)], 1, 1).expect("cell");
        let err = cellstr_builtin(cell)
            .expect_err("expected error")
            .to_string();
        assert!(err.contains("cell array elements must be"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_multirow_char_element() {
        let ca = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("char array");
        let cell = crate::make_cell(vec![Value::CharArray(ca)], 1, 1).expect("cell");
        let err = cellstr_builtin(cell)
            .expect_err("expected error")
            .to_string();
        assert!(err.contains("cell array elements must be"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_text_input() {
        let err = cellstr_builtin(Value::Num(std::f64::consts::PI))
            .expect_err("expected error")
            .to_string();
        assert!(err.contains("input must be"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handles_empty_char_array() {
        let ca = CharArray::new(Vec::new(), 0, 5).expect("empty char");
        let result = cellstr_builtin(Value::CharArray(ca)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 0);
                assert_eq!(cell.cols, 1);
                assert!(cell.data.is_empty());
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_row_of_spaces_becomes_empty_vector() {
        let ca = CharArray::new(vec![' '; 3], 1, 3).expect("char array");
        let result = cellstr_builtin(Value::CharArray(ca)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match unsafe { &*cell.data[0].as_raw() } {
                    Value::CharArray(row) => {
                        assert_eq!(row.rows, 1);
                        assert_eq!(row.cols, 0);
                        assert!(row.data.is_empty());
                    }
                    other => panic!("expected CharArray, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_elements_preserve_trailing_spaces() {
        let ca = CharArray::new(vec!['a', ' ', ' '], 1, 3).expect("char array");
        let cell = crate::make_cell(vec![Value::CharArray(ca.clone())], 1, 1).expect("cell");
        let result = cellstr_builtin(cell).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match unsafe { &*cell.data[0].as_raw() } {
                    Value::CharArray(row) => {
                        assert_eq!(row.rows, ca.rows);
                        assert_eq!(row.cols, ca.cols);
                        assert_eq!(row.data, ca.data);
                    }
                    other => panic!("expected CharArray, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_array_missing_value_converts() {
        let sa = StringArray::new(vec!["<missing>".to_string()], vec![1, 1]).expect("string array");
        let result = cellstr_builtin(Value::StringArray(sa)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                let rows = cell_to_strings(&cell);
                assert_eq!(rows, vec!["<missing>".to_string()]);
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_string_array_produces_empty_cell_shape() {
        let sa = StringArray::new(Vec::new(), vec![0, 2]).expect("string array");
        let result = cellstr_builtin(Value::StringArray(sa)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 0);
                assert_eq!(cell.cols, 2);
                assert!(cell.data.is_empty());
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }
}
