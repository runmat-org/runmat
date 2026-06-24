//! MATLAB-compatible `importdata` builtin for legacy text imports.

use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, StructValue, Tensor, Value,
};
use runmat_filesystem as fs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "importdata";

const IMPORTDATA_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Imported numeric matrix or import structure.",
}];
const IMPORTDATA_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File to import.",
}];
const IMPORTDATA_INPUTS_DELIMITER: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File to import.",
    },
    BuiltinParamDescriptor {
        name: "delimiterIn",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Delimiter to use for text files.",
    },
];
const IMPORTDATA_INPUTS_DELIMITER_HEADER: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File to import.",
    },
    BuiltinParamDescriptor {
        name: "delimiterIn",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Delimiter to use for text files.",
    },
    BuiltinParamDescriptor {
        name: "headerlinesIn",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Number of header lines to skip.",
    },
];
const IMPORTDATA_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "A = importdata(filename)",
        inputs: &IMPORTDATA_INPUTS_FILENAME,
        outputs: &IMPORTDATA_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "A = importdata(filename, delimiterIn)",
        inputs: &IMPORTDATA_INPUTS_DELIMITER,
        outputs: &IMPORTDATA_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "A = importdata(filename, delimiterIn, headerlinesIn)",
        inputs: &IMPORTDATA_INPUTS_DELIMITER_HEADER,
        outputs: &IMPORTDATA_OUTPUTS,
    },
];

const IMPORTDATA_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPORTDATA.ARGUMENT",
    identifier: Some("RunMat:importdata:InvalidArgument"),
    when: "Filename, delimiter, or header line arguments are malformed.",
    message: "importdata: invalid argument",
};
const IMPORTDATA_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPORTDATA.IO",
    identifier: Some("RunMat:importdata:Io"),
    when: "The input file cannot be read.",
    message: "importdata: unable to read file",
};
const IMPORTDATA_ERROR_PARSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPORTDATA.PARSE",
    identifier: Some("RunMat:importdata:Parse"),
    when: "Text content cannot be imported as supported numeric/header data.",
    message: "importdata: unable to parse text data",
};
const IMPORTDATA_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IMPORTDATA_ERROR_ARGUMENT,
    IMPORTDATA_ERROR_IO,
    IMPORTDATA_ERROR_PARSE,
];

pub const IMPORTDATA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMPORTDATA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMPORTDATA_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::importdata")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "importdata",
    op_kind: GpuOpKind::Custom("io-importdata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs on the host; file import is not an acceleration operation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::importdata")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "importdata",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side file I/O.",
};

fn importdata_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    importdata_error_with(error, error.message)
}

fn importdata_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn importdata_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|value| value.to_string());
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "importdata",
    category = "io/import",
    summary = "Import numeric text data with optional headers.",
    keywords = "importdata,text,csv,delimited,header,numeric import",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::importdata_type),
    descriptor(crate::builtins::io::importdata::IMPORTDATA_DESCRIPTOR),
    builtin_path = "crate::builtins::io::importdata"
)]
async fn importdata_builtin(filename: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(importdata_error(&IMPORTDATA_ERROR_ARGUMENT));
    }
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(map_control_flow)?;
    let path = resolve_path(&filename)?;

    let delimiter = if let Some(value) = rest.first() {
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(map_control_flow)?;
        Some(parse_delimiter_arg(&gathered)?)
    } else {
        None
    };
    let header_lines = if let Some(value) = rest.get(1) {
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(map_control_flow)?;
        Some(parse_header_lines(&gathered)?)
    } else {
        None
    };

    let text = fs::read_to_string_async(&path).await.map_err(|err| {
        importdata_error_with_source(
            &IMPORTDATA_ERROR_IO,
            format!("importdata: unable to read \"{}\" ({err})", path.display()),
            err,
        )
    })?;
    import_text_data(&text, delimiter.as_deref(), header_lines)
}

#[derive(Debug, Clone)]
struct ImportedText {
    data: Vec<Vec<f64>>,
    textdata: Vec<Vec<String>>,
    colheaders: Vec<String>,
    rowheaders: Vec<String>,
}

fn import_text_data(
    text: &str,
    delimiter: Option<&str>,
    header_lines: Option<usize>,
) -> BuiltinResult<Value> {
    let lines: Vec<&str> = text.lines().collect();
    let nonempty: Vec<(usize, &str)> = lines
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .collect();
    if nonempty.is_empty() {
        return Ok(Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).map_err(
            |err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")),
        )?));
    }

    let delimiter = delimiter
        .map(Delimiter::Explicit)
        .unwrap_or_else(|| detect_delimiter(nonempty.iter().map(|(_, line)| *line)));
    let records: Vec<(usize, Vec<String>)> = nonempty
        .iter()
        .map(|(idx, line)| (*idx, split_record(line, &delimiter)))
        .collect();

    let data_start = header_lines.unwrap_or_else(|| infer_header_lines(&records));
    if data_start > records.len() {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: headerlinesIn exceeds number of non-empty lines",
        ));
    }

    let header_records: Vec<Vec<String>> = records[..data_start]
        .iter()
        .map(|(_, record)| record.clone())
        .collect();
    let data_records = &records[data_start..];

    let imported = parse_numeric_records(data_records, &header_records)?;
    let tensor = rows_to_tensor(&imported.data)?;
    if imported.textdata.is_empty()
        && imported.colheaders.is_empty()
        && imported.rowheaders.is_empty()
    {
        return Ok(Value::Tensor(tensor));
    }

    let mut out = StructValue::new();
    out.insert("data", Value::Tensor(tensor));
    if !imported.textdata.is_empty() {
        out.insert("textdata", cell_from_rows(&imported.textdata)?);
    }
    if !imported.colheaders.is_empty() {
        out.insert("colheaders", cell_from_row(&imported.colheaders)?);
    }
    if !imported.rowheaders.is_empty() {
        out.insert("rowheaders", cell_from_col(&imported.rowheaders)?);
    }
    Ok(Value::Struct(out))
}

fn parse_numeric_records(
    data_records: &[(usize, Vec<String>)],
    header_records: &[Vec<String>],
) -> BuiltinResult<ImportedText> {
    if data_records.is_empty() {
        return Ok(ImportedText {
            data: Vec::new(),
            textdata: header_records.to_vec(),
            colheaders: header_records.last().cloned().unwrap_or_default(),
            rowheaders: Vec::new(),
        });
    }

    let first = &data_records[0].1;
    let row_header_cols = infer_row_header_cols(data_records);
    let numeric_cols = first.len().saturating_sub(row_header_cols);
    if numeric_cols == 0 {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_PARSE,
            "importdata: no numeric columns found",
        ));
    }

    let mut rows = Vec::with_capacity(data_records.len());
    let mut rowheaders = Vec::new();
    for (line_idx, record) in data_records {
        let expected_cols = row_header_cols + numeric_cols;
        if record.len() != expected_cols {
            return Err(importdata_error_with(
                &IMPORTDATA_ERROR_PARSE,
                format!(
                    "importdata: row {} has {} columns, expected {}",
                    line_idx + 1,
                    record.len(),
                    expected_cols
                ),
            ));
        }
        if row_header_cols > 0 {
            rowheaders.push(record[..row_header_cols].join(" "));
        }
        let mut row = Vec::with_capacity(numeric_cols);
        for (col, token) in record[row_header_cols..row_header_cols + numeric_cols]
            .iter()
            .enumerate()
        {
            row.push(parse_numeric_token(token).ok_or_else(|| {
                importdata_error_with(
                    &IMPORTDATA_ERROR_PARSE,
                    format!(
                        "importdata: nonnumeric token '{}' at row {}, column {}",
                        token,
                        line_idx + 1,
                        row_header_cols + col + 1
                    ),
                )
            })?);
        }
        rows.push(row);
    }

    let mut colheaders = Vec::new();
    if let Some(last_header) = header_records.last() {
        if last_header.len() >= row_header_cols + numeric_cols {
            colheaders = last_header[row_header_cols..row_header_cols + numeric_cols].to_vec();
        } else if last_header.len() == numeric_cols {
            colheaders = last_header.clone();
        }
    }

    Ok(ImportedText {
        data: rows,
        textdata: header_records.to_vec(),
        colheaders,
        rowheaders,
    })
}

fn infer_row_header_cols(records: &[(usize, Vec<String>)]) -> usize {
    let Some(first) = records.first() else {
        return 0;
    };
    if first.1.len() < 2 || parse_numeric_token(&first.1[0]).is_some() {
        return 0;
    }
    if records.iter().all(|(_, row)| {
        row.len() == first.1.len()
            && parse_numeric_token(&row[0]).is_none()
            && row[1..]
                .iter()
                .all(|token| parse_numeric_token(token).is_some())
    }) {
        1
    } else {
        0
    }
}

fn infer_header_lines(records: &[(usize, Vec<String>)]) -> usize {
    records
        .iter()
        .position(|(_, row)| is_numeric_data_row(row))
        .unwrap_or(records.len())
}

fn is_numeric_data_row(row: &[String]) -> bool {
    if row.is_empty() {
        return false;
    }
    if row.iter().all(|token| parse_numeric_token(token).is_some()) {
        return true;
    }
    row.len() > 1
        && parse_numeric_token(&row[0]).is_none()
        && row[1..]
            .iter()
            .all(|token| parse_numeric_token(token).is_some())
}

fn rows_to_tensor(rows: &[Vec<f64>]) -> BuiltinResult<Tensor> {
    let row_count = rows.len();
    let col_count = rows.first().map(|row| row.len()).unwrap_or(0);
    if rows.iter().any(|row| row.len() != col_count) {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_PARSE,
            "importdata: numeric rows have inconsistent column counts",
        ));
    }
    let mut data = Vec::with_capacity(row_count * col_count);
    for col in 0..col_count {
        for row in rows {
            data.push(row[col]);
        }
    }
    Tensor::new(data, vec![row_count, col_count])
        .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Delimiter<'a> {
    Whitespace,
    Explicit(&'a str),
}

fn detect_delimiter<'a>(lines: impl Iterator<Item = &'a str>) -> Delimiter<'static> {
    let candidates = [",", "\t", ";", "|"];
    let sample: Vec<&str> = lines.take(12).collect();
    let mut best: Option<(&str, usize, usize)> = None;
    for candidate in candidates {
        let counts: Vec<usize> = sample
            .iter()
            .map(|line| split_record(line, &Delimiter::Explicit(candidate)).len())
            .filter(|count| *count > 1)
            .collect();
        if counts.is_empty() {
            continue;
        }
        let consistent = counts.iter().filter(|count| **count == counts[0]).count();
        let score = (consistent, counts[0]);
        if best
            .map(|(_, best_consistent, best_cols)| score > (best_consistent, best_cols))
            .unwrap_or(true)
        {
            best = Some((candidate, consistent, counts[0]));
        }
    }
    best.map(|(candidate, _, _)| Delimiter::Explicit(candidate))
        .unwrap_or(Delimiter::Whitespace)
}

fn split_record(line: &str, delimiter: &Delimiter<'_>) -> Vec<String> {
    match delimiter {
        Delimiter::Whitespace => line
            .split_whitespace()
            .map(|token| unquote(token.trim()))
            .filter(|token| !token.is_empty())
            .collect(),
        Delimiter::Explicit(delimiter) => split_explicit(line, delimiter),
    }
}

fn split_explicit(line: &str, delimiter: &str) -> Vec<String> {
    if delimiter.is_empty() {
        return vec![line.trim().to_string()];
    }
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut idx = 0usize;
    while idx < line.len() {
        let Some(ch) = line[idx..].chars().next() else {
            break;
        };
        if ch == '"' {
            if in_quotes && line[idx + ch.len_utf8()..].starts_with('"') {
                current.push('"');
                idx += ch.len_utf8() * 2;
                continue;
            }
            in_quotes = !in_quotes;
            idx += ch.len_utf8();
            continue;
        }
        if !in_quotes && line[idx..].starts_with(delimiter) {
            fields.push(unquote(current.trim()));
            current.clear();
            idx += delimiter.len();
            continue;
        }
        current.push(ch);
        idx += ch.len_utf8();
    }
    fields.push(unquote(current.trim()));
    fields
}

fn unquote(token: &str) -> String {
    let trimmed = token.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        trimmed[1..trimmed.len() - 1].replace("\"\"", "\"")
    } else {
        trimmed.to_string()
    }
}

fn parse_numeric_token(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Some(f64::NAN);
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "nan" => Some(f64::NAN),
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => trimmed.parse::<f64>().ok(),
    }
}

fn parse_delimiter_arg(value: &Value) -> BuiltinResult<String> {
    let text = string_scalar(value, "delimiterIn")?;
    match text.as_str() {
        "\\t" => Ok("\t".to_string()),
        "\\n" => Ok("\n".to_string()),
        "\\r" => Ok("\r".to_string()),
        _ => Ok(text),
    }
}

fn parse_header_lines(value: &Value) -> BuiltinResult<usize> {
    let raw = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_i64() as f64,
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        _ => {
            return Err(importdata_error_with(
                &IMPORTDATA_ERROR_ARGUMENT,
                "importdata: headerlinesIn must be a nonnegative integer scalar",
            ));
        }
    };
    if !raw.is_finite() || raw < 0.0 || raw.fract() != 0.0 {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: headerlinesIn must be a nonnegative integer scalar",
        ));
    }
    Ok(raw as usize)
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_path(&text)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_path(&sa.data[0]),
        _ => Err(importdata_error(&IMPORTDATA_ERROR_ARGUMENT)),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME)
        .map_err(|msg| importdata_error_with(&IMPORTDATA_ERROR_ARGUMENT, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            format!("importdata: expected {context} as a string scalar or character vector"),
        )),
    }
}

fn cell_from_rows(rows: &[Vec<String>]) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let col_count = rows.iter().map(|row| row.len()).max().unwrap_or(0);
    let mut values = Vec::with_capacity(row_count * col_count);
    for row in rows {
        for col in 0..col_count {
            values.push(Value::String(row.get(col).cloned().unwrap_or_default()));
        }
    }
    CellArray::new(values, row_count, col_count)
        .map(Value::Cell)
        .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

fn cell_from_row(values: &[String]) -> BuiltinResult<Value> {
    CellArray::new(
        values.iter().cloned().map(Value::String).collect(),
        1,
        values.len(),
    )
    .map(Value::Cell)
    .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

fn cell_from_col(values: &[String]) -> BuiltinResult<Value> {
    CellArray::new(
        values.iter().cloned().map(Value::String).collect(),
        values.len(),
        1,
    )
    .map(Value::Cell)
    .map_err(|err| importdata_error_with(&IMPORTDATA_ERROR_PARSE, format!("importdata: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_importdata_{}_{}_{}.{}",
            std::process::id(),
            millis,
            unique,
            ext
        ));
        path
    }

    fn write_fixture(ext: &str, contents: &str) -> PathBuf {
        let path = temp_path(ext);
        fs::write(&path, contents).expect("write fixture");
        path
    }

    fn struct_field<'a>(value: &'a Value, name: &str) -> &'a Value {
        let Value::Struct(st) = value else {
            panic!("expected struct");
        };
        st.fields
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"))
    }

    fn tensor_data(value: &Value) -> (&[f64], &[usize]) {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        (&tensor.data, &tensor.shape)
    }

    fn cell_text(value: &Value, row: usize, col: usize) -> String {
        let Value::Cell(cell) = value else {
            panic!("expected cell");
        };
        let Value::String(text) = cell.get(row, col).expect("cell value") else {
            panic!("expected string cell");
        };
        text
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn importdata_descriptor_covers_core_forms() {
        let labels: Vec<&str> = IMPORTDATA_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"A = importdata(filename)"));
        assert!(labels.contains(&"A = importdata(filename, delimiterIn)"));
        assert!(labels.contains(&"A = importdata(filename, delimiterIn, headerlinesIn)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn importdata_reads_plain_numeric_matrix() {
        let path = write_fixture("txt", "1 2 3\n4 5 6\n");
        let out = block_on(importdata_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            Vec::new(),
        ))
        .expect("importdata");
        let (data, shape) = tensor_data(&out);
        assert_eq!(shape, &[2, 3]);
        assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn importdata_headerlines_zero_numeric_input_returns_tensor() {
        let path = write_fixture("txt", "1 2\n3 4\n");
        let out = block_on(importdata_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            vec![Value::from(" "), Value::Num(0.0)],
        ))
        .expect("importdata");
        let (data, shape) = tensor_data(&out);
        assert_eq!(shape, &[2, 2]);
        assert_eq!(data, &[1.0, 3.0, 2.0, 4.0]);
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn importdata_detects_csv_header_and_colheaders() {
        let path = write_fixture("csv", "time,value\n0,1.5\n1,2.5\n");
        let out = block_on(importdata_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            Vec::new(),
        ))
        .expect("importdata");
        let data = struct_field(&out, "data");
        let (values, shape) = tensor_data(data);
        assert_eq!(shape, &[2, 2]);
        assert_eq!(values, &[0.0, 1.0, 1.5, 2.5]);
        assert_eq!(cell_text(struct_field(&out, "colheaders"), 0, 0), "time");
        assert_eq!(cell_text(struct_field(&out, "colheaders"), 0, 1), "value");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn importdata_honors_explicit_delimiter_and_header_lines() {
        let path = write_fixture("dat", "# instrument log\nA|B\n10|20\n30|40\n");
        let out = block_on(importdata_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            vec![Value::from("|"), Value::Num(2.0)],
        ))
        .expect("importdata");
        let data = struct_field(&out, "data");
        let (values, shape) = tensor_data(data);
        assert_eq!(shape, &[2, 2]);
        assert_eq!(values, &[10.0, 30.0, 20.0, 40.0]);
        assert_eq!(
            cell_text(struct_field(&out, "textdata"), 0, 0),
            "# instrument log"
        );
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn importdata_preserves_rowheaders() {
        let path = write_fixture("txt", "label x y\nr1 1 2\nr2 3 4\n");
        let out = block_on(importdata_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            Vec::new(),
        ))
        .expect("importdata");
        assert_eq!(cell_text(struct_field(&out, "rowheaders"), 0, 0), "r1");
        assert_eq!(cell_text(struct_field(&out, "rowheaders"), 1, 0), "r2");
        assert_eq!(cell_text(struct_field(&out, "colheaders"), 0, 0), "x");
        assert_eq!(cell_text(struct_field(&out, "colheaders"), 0, 1), "y");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn importdata_reports_mixed_unsupported_data() {
        let path = write_fixture("txt", "1 2\n3 nope\n");
        let err = block_on(importdata_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            Vec::new(),
        ))
        .expect_err("parse error");
        assert!(err.message().contains("nonnumeric token"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn importdata_rejects_rows_with_extra_numeric_columns() {
        let path = write_fixture("txt", "1 2\n3 4 5\n");
        let err = block_on(importdata_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            Vec::new(),
        ))
        .expect_err("width mismatch");
        assert!(err.message().contains("expected 2"));
        let _ = fs::remove_file(path);
    }
}
