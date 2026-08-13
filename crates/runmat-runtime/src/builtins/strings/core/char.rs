//! MATLAB-compatible `char` builtin with GPU-aware conversion semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, IntValue, LogicalArray, NumericScalar, SparseTensor, StringArray,
    SymbolicArray, Tensor, Value,
};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::type_resolvers::string_array_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::char")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "char",
    op_kind: GpuOpKind::Custom("conversion"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Conversion always runs on the CPU. Interactive resident numeric input is undocumented and therefore mode-gated before the gather fallback; output is host character data.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::char")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "char",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Character materialisation runs outside of fusion; results always live on the host.",
};

const BUILTIN_NAME: &str = "char";
const CHAR_SPARSE_DENSE_ELEMENT_LIMIT: usize = 10_000_000;

pub(crate) const CHAR_RESIDENT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "char-resident-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "char with interactive resident numeric input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CharResidentNumericInputExtension"),
    };
pub(crate) const CHAR_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "char-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "char with logical input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CharLogicalInputExtension"),
    };

pub const CHAR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CHAR_LOGICAL_INPUT_EXTENSION,
    CHAR_RESIDENT_NUMERIC_EXTENSION,
];

const CHAR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character array result.",
}];

const CHAR_INPUT_SINGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to convert into character data.",
}];

const CHAR_INPUT_VARIADIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X...",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Multiple inputs converted row-wise and padded.",
}];

const CHAR_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "C = char()",
        inputs: &[],
        outputs: &CHAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = char(X)",
        inputs: &CHAR_INPUT_SINGLE,
        outputs: &CHAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = char(X...)",
        inputs: &CHAR_INPUT_VARIADIC,
        outputs: &CHAR_OUTPUT,
    },
];

const CHAR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INVALID_INPUT",
    identifier: Some("RunMat:char:InvalidInput"),
    when: "Input type cannot be converted to character data.",
    message: "char: invalid input",
};

const CHAR_ERROR_INVALID_CODEPOINT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INVALID_CODEPOINT",
    identifier: Some("RunMat:char:InvalidCodePoint"),
    when: "Numeric input cannot be represented by RunMat's scalar-value character storage.",
    message: "char: numeric input cannot be represented as a RunMat character",
};

const CHAR_ERROR_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INVALID_DIMENSION",
    identifier: Some("RunMat:char:InvalidDimension"),
    when: "Array inputs are not 2-D (or trailing singleton dimensions).",
    message: "char: inputs must be 2-D",
};

const CHAR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INTERNAL",
    identifier: Some("RunMat:char:InternalError"),
    when: "Internal character array construction failed.",
    message: "char: internal error",
};

const CHAR_ERRORS: [BuiltinErrorDescriptor; 4] = [
    CHAR_ERROR_INVALID_INPUT,
    CHAR_ERROR_INVALID_CODEPOINT,
    CHAR_ERROR_DIMENSION,
    CHAR_ERROR_INTERNAL,
];

pub const CHAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CHAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CHAR_ERRORS,
};

const CHAR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "All eight integer classes are documented numeric-code inputs. Floating values truncate toward zero and all numeric codes clamp to the UTF-16 code-unit interval 0..65535.",
}];

pub const CHAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = char(integer_X) or char(X1, ..., XN)",
        inputs: &CHAR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host integer values are decoded exactly, then clamped to 0..65535. RunMat CharArray stores Rust Unicode scalar values, so isolated UTF-16 surrogate code units U+D800..U+DFFF cannot yet be represented and produce an explicit error. Interactive resident numeric input is a mode-gated RunMat extension before gather.",
    }];

fn char_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    char_error_with_message(error.message, error)
}

fn char_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn char_flow(message: impl Into<String>) -> RuntimeError {
    char_error_with_message(message, &CHAR_ERROR_INTERNAL)
}

fn remap_char_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "char",
    category = "strings/core",
    summary = "Convert numeric codes and text values into character arrays.",
    keywords = "char,character,string,gpu",
    accel = "conversion",
    type_resolver(string_array_type),
    descriptor(crate::builtins::strings::core::char::CHAR_DESCRIPTOR),
    extensions(crate::builtins::strings::core::char::CHAR_EXTENSIONS),
    integer_capabilities(crate::builtins::strings::core::char::CHAR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::strings::core::char"
)]
async fn char_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        let empty =
            CharArray::new(Vec::new(), 0, 0).map_err(|_| char_error(&CHAR_ERROR_INTERNAL))?;
        return Ok(Value::CharArray(empty));
    }

    let mut rows: Vec<Vec<char>> = Vec::new();
    let mut max_width = 0usize;

    for arg in rest {
        if matches!(&arg, Value::Bool(_) | Value::LogicalArray(_))
            || matches!(&arg, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CHAR_LOGICAL_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(&arg, Value::GpuTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CHAR_RESIDENT_NUMERIC_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        let gathered = gather_if_needed_async(&arg)
            .await
            .map_err(remap_char_flow)?;
        let mut produced = value_to_char_rows(&gathered)?;
        for row in &produced {
            if row.len() > max_width {
                max_width = row.len();
            }
        }
        rows.append(&mut produced);
    }

    if rows.is_empty() {
        let empty =
            CharArray::new(Vec::new(), 0, 0).map_err(|_| char_error(&CHAR_ERROR_INTERNAL))?;
        return Ok(Value::CharArray(empty));
    }

    let cols = max_width;
    let total_rows = rows.len();
    let mut data = vec![' '; total_rows * cols];
    for (row_idx, row) in rows.into_iter().enumerate() {
        for (col_idx, ch) in row.into_iter().enumerate() {
            if col_idx < cols {
                data[row_idx * cols + col_idx] = ch;
            }
        }
    }

    let array =
        CharArray::new(data, total_rows, cols).map_err(|_| char_error(&CHAR_ERROR_INTERNAL))?;
    Ok(Value::CharArray(array))
}

fn value_to_char_rows(value: &Value) -> BuiltinResult<Vec<Vec<char>>> {
    if let Some(array) = crate::builtins::datetime::datetime_char_array(value)
        .map_err(|err| char_flow(err.message().to_string()))?
    {
        return Ok(char_array_rows(&array));
    }
    if let Some(array) = crate::builtins::duration::duration_char_array(value)
        .map_err(|err| char_flow(err.message().to_string()))?
    {
        return Ok(char_array_rows(&array));
    }
    match value {
        Value::CharArray(ca) => Ok(char_array_rows(ca)),
        Value::String(s) => Ok(vec![s.chars().collect()]),
        Value::Symbolic(expr) => Ok(vec![expr.to_string().chars().collect()]),
        Value::SymbolicArray(array) => symbolic_array_rows(array),
        Value::StringArray(sa) => string_array_rows(sa),
        Value::Num(n) => Ok(vec![vec![number_to_char(*n)?]]),
        Value::Int(i) => Ok(vec![vec![integer_value_to_char(i)?]]),
        Value::Bool(b) => {
            let code = if *b { 1.0 } else { 0.0 };
            Ok(vec![vec![number_to_char(code)?]])
        }
        Value::Tensor(t) => tensor_rows(t),
        Value::SparseTensor(s) => {
            ensure_sparse_dense_conversion(s)?;
            let dense = s.to_dense().map_err(char_flow)?;
            tensor_rows(&dense)
        }
        Value::LogicalArray(la) => logical_rows(la),
        Value::Cell(ca) => cell_rows(ca),
        Value::GpuTensor(_) => Err(char_error(&CHAR_ERROR_INVALID_INPUT)),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(char_error_with_message(
            "char: complex inputs are not supported",
            &CHAR_ERROR_INVALID_INPUT,
        )),
        Value::Struct(_)
        | Value::ObjectArray(_)
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
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Foreign(_)
        | Value::OutputList(_) => Err(char_error_with_message(
            format!("char: unsupported input type {:?}", value),
            &CHAR_ERROR_INVALID_INPUT,
        )),
    }
}

fn char_array_rows(ca: &CharArray) -> Vec<Vec<char>> {
    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let mut row = Vec::with_capacity(ca.cols);
        for c in 0..ca.cols {
            row.push(ca.data[r * ca.cols + c]);
        }
        rows.push(row);
    }
    rows
}

fn string_array_rows(sa: &StringArray) -> BuiltinResult<Vec<Vec<char>>> {
    ensure_two_dimensional(&sa.shape, "char")?;
    if sa.data.is_empty() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::with_capacity(sa.data.len());
    let rows_count = sa.rows();
    let cols_count = sa.cols();
    if rows_count == 0 || cols_count == 0 {
        return Ok(Vec::new());
    }
    for c in 0..cols_count {
        for r in 0..rows_count {
            let idx = r + c * rows_count;
            rows.push(sa.data[idx].chars().collect());
        }
    }
    Ok(rows)
}

fn symbolic_array_rows(array: &SymbolicArray) -> BuiltinResult<Vec<Vec<char>>> {
    ensure_two_dimensional(&array.shape, "char")?;
    let (rows, cols) = infer_rows_cols(&array.shape, array.data.len());
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::new();
        for c in 0..cols {
            if cols == 0 {
                continue;
            }
            let idx = r + c * rows;
            row.extend(array.data[idx].to_string().chars());
        }
        out.push(row);
    }
    Ok(out)
}

fn tensor_rows(t: &Tensor) -> BuiltinResult<Vec<Vec<char>>> {
    ensure_two_dimensional(&t.shape, "char")?;
    let element_len = t.len();
    let (rows, cols) = infer_rows_cols(&t.shape, element_len);
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            if cols == 0 {
                continue;
            }
            let idx = r + c * rows;
            let value = t.numeric_value_at(idx).ok_or_else(|| {
                char_error_with_message(
                    "char: numeric storage length does not match tensor shape",
                    &CHAR_ERROR_INTERNAL,
                )
            })?;
            let ch = match value {
                NumericScalar::F64(value) => number_to_char(value)?,
                NumericScalar::F32(value) => number_to_char(f64::from(value))?,
                value => integer_value_to_char(
                    &value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer"),
                )?,
            };
            row.push(ch);
        }
        out.push(row);
    }
    Ok(out)
}

fn logical_rows(la: &LogicalArray) -> BuiltinResult<Vec<Vec<char>>> {
    ensure_two_dimensional(&la.shape, "char")?;
    let (rows, cols) = infer_rows_cols(&la.shape, la.data.len());
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            if cols == 0 {
                continue;
            }
            let idx = r + c * rows;
            let code = if la.data[idx] != 0 { 1.0 } else { 0.0 };
            row.push(number_to_char(code)?);
        }
        out.push(row);
    }
    Ok(out)
}

fn cell_rows(ca: &CellArray) -> BuiltinResult<Vec<Vec<char>>> {
    let mut rows = Vec::with_capacity(ca.data.len());
    for ptr in &ca.data {
        let element = (ptr).clone();
        let mut converted = value_to_char_rows(&element)?;
        match converted.len() {
            0 => rows.push(Vec::new()),
            1 => rows.push(converted.remove(0)),
            _ => {
                return Err(char_error_with_message(
                    "char: cell elements must be character vectors or string scalars",
                    &CHAR_ERROR_INVALID_INPUT,
                ))
            }
        }
    }
    Ok(rows)
}

fn ensure_sparse_dense_conversion(sparse: &SparseTensor) -> BuiltinResult<()> {
    let total_elements = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
        char_error_with_message(
            "char: sparse matrix dimensions overflow",
            &CHAR_ERROR_INVALID_INPUT,
        )
    })?;
    if total_elements > CHAR_SPARSE_DENSE_ELEMENT_LIMIT {
        return Err(char_error_with_message(
            format!(
                "char: cannot convert sparse tensor {}x{} with {} stored entries to dense character array ({} elements exceeds safe threshold)",
                sparse.rows,
                sparse.cols,
                sparse.nnz(),
                total_elements
            ),
            &CHAR_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

fn number_to_char(value: f64) -> BuiltinResult<char> {
    if !value.is_finite() {
        return Err(char_error_with_message(
            "char: numeric inputs must be finite",
            &CHAR_ERROR_INVALID_CODEPOINT,
        ));
    }
    let code_unit = value.trunc().clamp(0.0, u16::MAX as f64) as u16;
    utf16_code_unit_to_char(code_unit)
}

fn integer_value_to_char(value: &IntValue) -> BuiltinResult<char> {
    let code_unit = match value {
        IntValue::I8(value) => signed_integer_to_code_unit(*value as i128),
        IntValue::I16(value) => signed_integer_to_code_unit(*value as i128),
        IntValue::I32(value) => signed_integer_to_code_unit(*value as i128),
        IntValue::I64(value) => signed_integer_to_code_unit(*value as i128),
        IntValue::U8(value) => unsigned_integer_to_code_unit(*value as u128),
        IntValue::U16(value) => unsigned_integer_to_code_unit(*value as u128),
        IntValue::U32(value) => unsigned_integer_to_code_unit(*value as u128),
        IntValue::U64(value) => unsigned_integer_to_code_unit(*value as u128),
    };
    utf16_code_unit_to_char(code_unit)
}

fn utf16_code_unit_to_char(code_unit: u16) -> BuiltinResult<char> {
    char::from_u32(u32::from(code_unit)).ok_or_else(|| {
        char_error_with_message(
            format!(
                "char: UTF-16 surrogate code unit U+{code_unit:04X} cannot be represented by the current RunMat CharArray scalar-value storage"
            ),
            &CHAR_ERROR_INVALID_CODEPOINT,
        )
    })
}

fn signed_integer_to_code_unit(value: i128) -> u16 {
    value.clamp(0, u16::MAX as i128) as u16
}

fn unsigned_integer_to_code_unit(value: u128) -> u16 {
    value.min(u16::MAX as u128) as u16
}

fn ensure_two_dimensional(shape: &[usize], context: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape.iter().skip(2).all(|&d| d == 1) {
        return Ok(());
    }
    Err(char_error_with_message(
        format!("{context}: inputs must be 2-D"),
        &CHAR_ERROR_DIMENSION,
    ))
}

fn infer_rows_cols(shape: &[usize], len: usize) -> (usize, usize) {
    match shape.len() {
        0 => {
            if len == 0 {
                (0, 0)
            } else {
                (1, 1)
            }
        }
        1 => (1, shape[0]),
        2 => (shape[0], shape[1]),
        _ => {
            let rows = shape[0];
            let cols = if shape.len() > 1 { shape[1] } else { 1 };
            (rows, cols)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntegerStorage, NumericStorage, SymbolicArray, SymbolicExpr};

    fn char_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::char_builtin(rest))
    }
    use runmat_value::StringArray;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_no_arguments_returns_empty() {
        let result = char_builtin(Vec::new()).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 0);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_string_scalar() {
        let value = Value::String("RunMat".to_string());
        let result = char_builtin(vec![value]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                assert_eq!(ca.data, "RunMat".chars().collect::<Vec<_>>());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_numeric_tensor() {
        let tensor =
            Tensor::new(vec![82.0, 85.0, 78.0, 77.0, 65.0, 84.0], vec![1, 6]).expect("tensor");
        let result = char_builtin(vec![Value::Tensor(tensor)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                assert_eq!(ca.data, "RUNMAT".chars().collect::<Vec<_>>());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn char_from_native_single_tensor_reads_authoritative_storage() {
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![82.0, 77.0]), vec![1, 2])
                .expect("single tensor");
        let result = char_builtin(vec![Value::Tensor(tensor)]).expect("char");
        match result {
            Value::CharArray(array) => assert_eq!(array.data, vec!['R', 'M']),
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_typed_integer_tensor_reads_exact_storage_without_mirror() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![82, u64::MAX]), vec![1, 2])
            .expect("typed tensor");

        let result = char_builtin(vec![Value::Tensor(tensor)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 2);
                assert_eq!(ca.data, vec!['R', '\u{FFFF}']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn char_reads_every_integer_tensor_storage_class() {
        for storage in [
            IntegerStorage::I8(vec![65]),
            IntegerStorage::I16(vec![65]),
            IntegerStorage::I32(vec![65]),
            IntegerStorage::I64(vec![65]),
            IntegerStorage::U8(vec![65]),
            IntegerStorage::U16(vec![65]),
            IntegerStorage::U32(vec![65]),
            IntegerStorage::U64(vec![65]),
        ] {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed tensor");
            let Value::CharArray(array) = char_builtin(vec![Value::Tensor(tensor)]).expect("char")
            else {
                panic!("expected char array");
            };
            assert_eq!(array.data, vec!['A']);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_clamps_negative_typed_integer_storage_without_mirror() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("typed tensor");

        let Value::CharArray(array) = char_builtin(vec![Value::Tensor(tensor)]).expect("char")
        else {
            panic!("expected char array");
        };
        assert_eq!(array.data, vec!['\0']);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_clamps_out_of_range_uint64_storage_without_mirror() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![0x110000]), vec![1, 1])
            .expect("typed tensor");

        let Value::CharArray(array) = char_builtin(vec![Value::Tensor(tensor)]).expect("char")
        else {
            panic!("expected char array");
        };
        assert_eq!(array.data, vec!['\u{FFFF}']);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_string_array_with_padding() {
        let data = vec!["cat".to_string(), "giraffe".to_string()];
        let sa = StringArray::new(data, vec![2, 1]).expect("string array");
        let result = char_builtin(vec![Value::StringArray(sa)]).expect("char from string array");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 7);
                assert_eq!(
                    ca.data,
                    vec!['c', 'a', 't', ' ', ' ', ' ', ' ', 'g', 'i', 'r', 'a', 'f', 'f', 'e']
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_cell_array_of_strings() {
        let cell = CellArray::new(
            vec![
                Value::from("north"),
                Value::from("east"),
                Value::from("west"),
            ],
            3,
            1,
        )
        .expect("cell array");
        let result = char_builtin(vec![Value::Cell(cell)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 3);
                assert_eq!(ca.cols, 5);
                assert_eq!(
                    ca.data,
                    vec!['n', 'o', 'r', 't', 'h', 'e', 'a', 's', 't', ' ', 'w', 'e', 's', 't', ' ']
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_numeric_and_text_arguments_concatenate() {
        let text = Value::String("hi".to_string());
        let codes = Tensor::new(vec![65.0, 66.0], vec![1, 2]).expect("tensor");
        let result = char_builtin(vec![text, Value::Tensor(codes)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 2);
                assert_eq!(ca.data, vec!['h', 'i', 'A', 'B']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_gpu_tensor_round_trip() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![82.0, 85.0, 78.0], vec![1, 3]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = char_builtin(vec![Value::GpuTensor(handle)]).expect("char");
            match result {
                Value::CharArray(ca) => {
                    assert_eq!(ca.rows, 1);
                    assert_eq!(ca.cols, 3);
                    assert_eq!(ca.data, vec!['R', 'U', 'N']);
                }
                other => panic!("expected char array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_truncates_floating_numeric_toward_zero() {
        let Value::CharArray(array) = char_builtin(vec![Value::Num(65.9)]).expect("char") else {
            panic!("expected char array");
        };
        assert_eq!(array.data, vec!['A']);
    }

    #[test]
    fn char_supports_all_integer_classes_and_reports_surrogate_gap() {
        let values = [
            IntValue::I8(65),
            IntValue::I16(65),
            IntValue::I32(65),
            IntValue::I64(65),
            IntValue::U8(65),
            IntValue::U16(65),
            IntValue::U32(65),
            IntValue::U64(65),
        ];
        for value in values {
            let Value::CharArray(array) = char_builtin(vec![Value::Int(value)]).expect("char")
            else {
                panic!("expected char array");
            };
            assert_eq!(array.data, vec!['A']);
        }

        let err = char_builtin(vec![Value::Int(IntValue::U16(0xD800))])
            .expect_err("surrogate is not a Rust char");
        assert_eq!(err.identifier(), Some("RunMat:char:InvalidCodePoint"));
        assert!(err.message().contains("surrogate code unit"));
    }

    #[test]
    fn char_logical_extension_is_ordered_and_mode_gated_before_conversion() {
        assert_eq!(CHAR_EXTENSIONS[0].id, "char-logical-input");
        assert_eq!(CHAR_EXTENSIONS[1].id, "char-resident-numeric-input");

        let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = char_builtin(vec![Value::Bool(true)]).expect_err("logical extension gate");
        assert_eq!(
            err.identifier(),
            CHAR_LOGICAL_INPUT_EXTENSION.error_identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_high_dimension_tensor() {
        let tensor =
            Tensor::new(vec![65.0, 66.0], vec![1, 1, 2]).expect("tensor construction failed");
        let err = error_message(
            char_builtin(vec![Value::Tensor(tensor)]).expect_err("should reject >2D tensor"),
        );
        assert!(err.contains("2-D"), "expected dimension error, got {err}");
    }

    #[test]
    fn char_rejects_oversized_sparse_tensor_before_densifying() {
        let sparse = SparseTensor::zeros(CHAR_SPARSE_DENSE_ELEMENT_LIMIT + 1, 1);
        let err = char_builtin(vec![Value::SparseTensor(sparse)]).unwrap_err();

        assert_eq!(err.identifier(), Some("RunMat:char:InvalidInput"));
        assert!(err.message().contains("exceeds safe threshold"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_string_array_column_major_order() {
        let data = vec![
            "c0r0".to_string(),
            "c0r1".to_string(),
            "c1r0".to_string(),
            "c1r1".to_string(),
        ];
        let sa = StringArray::new(data, vec![2, 2]).expect("string array");
        let result = char_builtin(vec![Value::StringArray(sa)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 4);
                assert_eq!(ca.cols, 4);
                assert_eq!(ca.data, "c0r0c0r1c1r0c1r1".chars().collect::<Vec<char>>());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_symbolic_array_preserves_matrix_rows() {
        let array = SymbolicArray::new(
            vec![
                SymbolicExpr::variable("x"),
                SymbolicExpr::variable("z"),
                SymbolicExpr::variable("y"),
                SymbolicExpr::variable("w"),
            ],
            vec![2, 2],
        )
        .expect("symbolic array");

        let result = char_builtin(vec![Value::SymbolicArray(array)]).expect("char");

        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 2);
                assert_eq!(ca.data, vec!['x', 'y', 'z', 'w']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_symbolic_array_pads_multi_character_rows() {
        let array = SymbolicArray::new(
            vec![
                SymbolicExpr::variable("x1"),
                SymbolicExpr::variable("y"),
                SymbolicExpr::variable("theta"),
                SymbolicExpr::variable("z"),
            ],
            vec![2, 2],
        )
        .expect("symbolic array");

        let result = char_builtin(vec![Value::SymbolicArray(array)]).expect("char");

        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 7);
                assert_eq!(ca.data.iter().collect::<String>(), "x1thetayz     ");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_symbolic_one_dimensional_array_is_row_vector() {
        let array = SymbolicArray::new(
            vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
            vec![2],
        )
        .expect("symbolic array");

        let result = char_builtin(vec![Value::SymbolicArray(array)]).expect("char");

        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 2);
                assert_eq!(ca.data, vec!['x', 'y']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_high_dimension_string_array() {
        let sa = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 1, 2])
            .expect("string array");
        let err = error_message(
            char_builtin(vec![Value::StringArray(sa)]).expect_err("should reject >2D string array"),
        );
        assert!(err.contains("2-D"), "expected dimension error, got {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_complex_input() {
        let err =
            error_message(char_builtin(vec![Value::Complex(1.0, 2.0)]).expect_err("complex input"));
        assert!(
            err.contains("complex"),
            "expected complex error message, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn char_wgpu_numeric_codes_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _ = register_wgpu_provider(WgpuProviderOptions::default());

        let tensor = Tensor::new(vec![82.0, 85.0, 78.0], vec![1, 3]).unwrap();
        let cpu = char_builtin(vec![Value::Tensor(tensor.clone())]).expect("char cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let gpu = char_builtin(vec![Value::GpuTensor(handle)]).expect("char gpu");

        match (cpu, gpu) {
            (Value::CharArray(expected), Value::CharArray(actual)) => {
                assert_eq!(actual, expected);
            }
            other => panic!("unexpected results {other:?}"),
        }
    }

    #[test]
    fn char_type_is_string_array() {
        assert_eq!(
            string_array_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::cell_of(Type::String)
        );
    }
}
