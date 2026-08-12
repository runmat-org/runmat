//! File-backed text extraction helpers for Text Analytics workflows.

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use encoding_rs::Encoding;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericScalar, Value};

use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::html::{extract_html_text_value, ExtractionMethod};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

const NAME: &str = "extractFileText";
const MAX_FILE_BYTES: usize = 512 * 1024 * 1024;
const MAX_DOCX_XML_BYTES: usize = 128 * 1024 * 1024;

const RESIDENT_PAGES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "extractfiletext-resident-pages",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "extractFileText with a resident Pages control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExtractFileTextResidentPagesExtension"),
};
const BROAD_FILE_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "extractfiletext-broad-file-cell",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "extractFileText with a string-valued or nested filename cell is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExtractFileTextBroadFileCellExtension"),
};
const EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [RESIDENT_PAGES_EXTENSION, BROAD_FILE_CELL_EXTENSION];
const INTEGER_PAGES_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Pages",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Pages is a nonempty vector of positive page numbers and accepts every built-in integer class.",
    }];
const INTEGER_FILENAME_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "filename",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "The filename or URL is text; integer data rejects before file IO or provider access.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "str = extractFileText(filename, 'Pages', integer_pages)",
        inputs: &INTEGER_PAGES_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Pages values are parsed exactly without floating materialization. PDF extraction itself remains an explicitly unsupported format in RunMat.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "str = extractFileText(integer_filename)",
        inputs: &INTEGER_FILENAME_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer filenames are outside the public text domain and reject without conversion.",
    },
];

const OUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Extracted text.",
}];

const IN_FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to a text, HTML, or DOCX file.",
}];

const IN_FILE_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to a text, HTML, or DOCX file.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: Encoding, ExtractionMethod, Password, and Pages.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_FILE_TEXT.INVALID_INPUT",
    identifier: Some("RunMat:extractFileText:InvalidInput"),
    when: "Inputs do not match a supported extractFileText form.",
    message: "extractFileText: invalid input",
};

const ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_FILE_TEXT.IO",
    identifier: Some("RunMat:extractFileText:IOError"),
    when: "The requested file cannot be read.",
    message: "extractFileText: file read failed",
};

const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_FILE_TEXT.UNSUPPORTED",
    identifier: Some("RunMat:extractFileText:UnsupportedFormat"),
    when: "The requested file type or option requires unsupported extraction infrastructure.",
    message: "extractFileText: unsupported file type or option",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INVALID_INPUT, ERROR_IO, ERROR_UNSUPPORTED];

pub const EXTRACT_FILE_TEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "str = extractFileText(filename)",
            inputs: &IN_FILE,
            outputs: &OUT_TEXT,
        },
        BuiltinSignatureDescriptor {
            label: "str = extractFileText(filename,Name,Value)",
            inputs: &IN_FILE_REST,
            outputs: &OUT_TEXT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn string_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::String
}

fn extract_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "extractFileText",
    category = "strings/text_analytics",
    summary = "Read text from text, HTML, and DOCX files.",
    keywords = "extractFileText,text analytics,file,text,HTML,DOCX",
    accel = "sink",
    extensions(EXTENSIONS),
    integer_capabilities(INTEGER_CAPABILITIES),
    type_resolver(string_type),
    descriptor(crate::builtins::strings::text_analytics::file_text::EXTRACT_FILE_TEXT_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::file_text"
)]
async fn extract_file_text_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    preflight_args(&args)?;
    let args = gather_args(args).await?;
    let (source, options) = parse_args(args)?;
    if is_url(&source) {
        return Err(extract_error(
            &ERROR_UNSUPPORTED,
            "extractFileText: URL extraction requires website fetching support and remains tracked",
        ));
    }
    let path = PathBuf::from(&source);
    let kind = FileKind::from_path(&path);
    validate_options_for_kind(kind, &options)?;
    let bytes = read_file(&path).await?;
    let text = match kind {
        FileKind::Html => {
            let html = decode_bytes(&bytes, options.encoding.as_deref())?;
            let method = options.extraction_method.unwrap_or(ExtractionMethod::Tree);
            match extract_html_text_value(Value::String(html), method)? {
                Value::String(text) => text,
                other => {
                    return Err(extract_error(
                        &ERROR_INVALID_INPUT,
                        format!("extractFileText: unexpected HTML extraction output {other:?}"),
                    ))
                }
            }
        }
        FileKind::Docx => extract_docx_text(&bytes)?,
        FileKind::Pdf => unreachable!("PDF files are rejected before file IO"),
        FileKind::PlainText => decode_bytes(&bytes, options.encoding.as_deref())?,
    };
    Ok(Value::String(text))
}

fn preflight_args(args: &[Value]) -> BuiltinResult<()> {
    if let Some(source) = args.first() {
        if is_numeric_or_resident(source) || contains_numeric_or_resident(source) {
            return Err(extract_error(
                &ERROR_INVALID_INPUT,
                "extractFileText: expected a text filename or URL",
            ));
        }
        if filename_cell_is_broad(source) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BROAD_FILE_CELL_EXTENSION,
                NAME,
            )?;
        }
    }
    for (idx, value) in args.iter().enumerate().skip(1) {
        if contains_resident(value) {
            let is_pages_value = idx >= 2
                && idx % 2 == 0
                && scalar_text(&args[idx - 1], NAME)
                    .is_ok_and(|name| name.eq_ignore_ascii_case("pages"));
            if is_pages_value {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &RESIDENT_PAGES_EXTENSION,
                    NAME,
                )?;
            } else {
                return Err(extract_error(
                    &ERROR_INVALID_INPUT,
                    "extractFileText: resident values are supported only for the Pages extension",
                ));
            }
        }
    }
    Ok(())
}

fn is_numeric_or_resident(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::GpuTensor(_)
    )
}

fn contains_numeric_or_resident(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .any(|value| is_numeric_or_resident(value) || contains_numeric_or_resident(value)),
        _ => false,
    }
}

fn contains_resident(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(cell) => cell.data.iter().any(contains_resident),
        _ => false,
    }
}

fn filename_cell_is_broad(value: &Value) -> bool {
    match value {
        Value::Cell(cell) if cell.data.len() == 1 => !matches!(
            cell.data.first(),
            Some(Value::CharArray(array)) if array.rows <= 1
        ),
        _ => false,
    }
}

async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            extract_error(
                &ERROR_INVALID_INPUT,
                format!("extractFileText: failed to gather input: {err}"),
            )
        })?);
    }
    Ok(out)
}

#[derive(Clone, Debug, Default)]
struct ExtractFileTextOptions {
    encoding: Option<String>,
    extraction_method: Option<ExtractionMethod>,
    password: Option<String>,
    pages: Option<Vec<usize>>,
}

fn parse_args(args: Vec<Value>) -> BuiltinResult<(String, ExtractFileTextOptions)> {
    if args.is_empty() {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: expected filename input",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: name-value options must appear in pairs",
        ));
    }
    let source = scalar_file_input(&args[0])?;
    let mut options = ExtractFileTextOptions::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], NAME)
            .map_err(|err| extract_error(&ERROR_INVALID_INPUT, err.to_string()))?;
        match name.to_ascii_lowercase().as_str() {
            "encoding" => {
                options.encoding = Some(
                    scalar_text(&args[idx + 1], NAME)
                        .map_err(|err| extract_error(&ERROR_INVALID_INPUT, err.to_string()))?,
                );
            }
            "extractionmethod" => {
                let method = scalar_text(&args[idx + 1], NAME)
                    .map_err(|err| extract_error(&ERROR_INVALID_INPUT, err.to_string()))?;
                options.extraction_method = Some(
                    ExtractionMethod::parse(&method)
                        .map_err(|err| extract_error(&ERROR_INVALID_INPUT, err.to_string()))?,
                );
            }
            "password" => {
                options.password = Some(
                    scalar_text(&args[idx + 1], NAME)
                        .map_err(|err| extract_error(&ERROR_INVALID_INPUT, err.to_string()))?,
                );
            }
            "pages" => {
                options.pages = Some(parse_pages(&args[idx + 1])?);
            }
            other => {
                return Err(extract_error(
                    &ERROR_INVALID_INPUT,
                    format!("extractFileText: unsupported option '{other}'"),
                ))
            }
        }
        idx += 2;
    }
    Ok((source, options))
}

fn scalar_file_input(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(value) => nonempty_source(value),
        Value::StringArray(array) if array.data.len() == 1 => nonempty_source(&array.data[0]),
        Value::CharArray(array) if array.rows <= 1 => {
            nonempty_source(&array.data.iter().collect::<String>())
        }
        Value::Cell(cell) if cell.data.len() == 1 => scalar_file_input(&cell.data[0]),
        other => Err(extract_error(
            &ERROR_INVALID_INPUT,
            format!("extractFileText: expected scalar filename or URL, got {other:?}"),
        )),
    }
}

fn nonempty_source(value: &str) -> BuiltinResult<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: filename must not be empty",
        ))
    } else {
        Ok(trimmed.to_string())
    }
}

fn parse_pages(value: &Value) -> BuiltinResult<Vec<usize>> {
    let values = match value {
        Value::Num(value) => vec![NumericScalar::F64(*value)],
        Value::Int(value) => vec![int_numeric_scalar(value.clone())],
        Value::Tensor(tensor) if tensor_shape_is_vector(&tensor.shape) => (0..tensor.len())
            .map(|idx| {
                tensor.numeric_value_at(idx).ok_or_else(|| {
                    extract_error(
                        &ERROR_INVALID_INPUT,
                        "extractFileText: Pages must contain real numeric values",
                    )
                })
            })
            .collect::<BuiltinResult<Vec<_>>>()?,
        Value::Tensor(_) => {
            return Err(extract_error(
                &ERROR_INVALID_INPUT,
                "extractFileText: Pages must be a vector of positive integers",
            ))
        }
        other => {
            return Err(extract_error(
                &ERROR_INVALID_INPUT,
                format!("extractFileText: Pages must be a positive integer vector, got {other:?}"),
            ))
        }
    };
    if values.is_empty() {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: Pages must not be empty",
        ));
    }
    values.into_iter().map(parse_page_number).collect()
}

fn tensor_shape_is_vector(shape: &[usize]) -> bool {
    shape.iter().filter(|dim| **dim > 1).count() <= 1
}

fn int_numeric_scalar(value: IntValue) -> NumericScalar {
    match value {
        IntValue::I8(value) => NumericScalar::I8(value),
        IntValue::I16(value) => NumericScalar::I16(value),
        IntValue::I32(value) => NumericScalar::I32(value),
        IntValue::I64(value) => NumericScalar::I64(value),
        IntValue::U8(value) => NumericScalar::U8(value),
        IntValue::U16(value) => NumericScalar::U16(value),
        IntValue::U32(value) => NumericScalar::U32(value),
        IntValue::U64(value) => NumericScalar::U64(value),
    }
}

fn parse_page_number(value: NumericScalar) -> BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => parse_float_page(value),
        NumericScalar::F32(value) => parse_float_page(f64::from(value)),
        integer => integer
            .into_int_value()
            .expect("non-floating numeric scalar is integer")
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                extract_error(
                    &ERROR_INVALID_INPUT,
                    "extractFileText: Pages values must be positive host-representable integers",
                )
            }),
    }
}

fn parse_float_page(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            format!("extractFileText: Pages values must be positive integers, got {value}"),
        ));
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: Pages value exceeds the host index range",
        ));
    }
    let page = value as usize;
    if page as f64 != value {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: Pages value is not exactly representable as a host index",
        ));
    }
    Ok(page)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FileKind {
    PlainText,
    Html,
    Docx,
    Pdf,
}

impl FileKind {
    fn from_path(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref()
        {
            Some("html" | "htm" | "xhtml") => Self::Html,
            Some("docx") => Self::Docx,
            Some("pdf") => Self::Pdf,
            _ => Self::PlainText,
        }
    }
}

fn validate_options_for_kind(
    kind: FileKind,
    options: &ExtractFileTextOptions,
) -> BuiltinResult<()> {
    if kind == FileKind::Pdf {
        return Err(extract_error(
            &ERROR_UNSUPPORTED,
            "extractFileText: PDF extraction requires PDF text extraction support and remains tracked",
        ));
    }
    if options.password.is_some() || options.pages.is_some() {
        return Err(extract_error(
            &ERROR_UNSUPPORTED,
            "extractFileText: Password and Pages options require PDF extraction support and remain tracked",
        ));
    }
    if options.encoding.is_some() && kind == FileKind::Docx {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: Encoding is supported for text and HTML files only",
        ));
    }
    if options.extraction_method.is_some() && kind != FileKind::Html {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: ExtractionMethod is supported for HTML files only",
        ));
    }
    Ok(())
}

fn is_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

async fn read_file(path: &Path) -> BuiltinResult<Vec<u8>> {
    let bytes = runmat_filesystem::read_async(path).await.map_err(|err| {
        extract_error(
            &ERROR_IO,
            format!(
                "extractFileText: unable to read '{}': {err}",
                path.display()
            ),
        )
    })?;
    if bytes.len() > MAX_FILE_BYTES {
        return Err(extract_error(
            &ERROR_UNSUPPORTED,
            format!(
                "extractFileText: file '{}' exceeds the {} byte extraction limit",
                path.display(),
                MAX_FILE_BYTES
            ),
        ));
    }
    Ok(bytes)
}

fn decode_bytes(bytes: &[u8], encoding: Option<&str>) -> BuiltinResult<String> {
    let encoding = encoding.unwrap_or("utf-8");
    let Some(encoding) = Encoding::for_label(encoding.trim().as_bytes()) else {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            format!("extractFileText: unsupported Encoding '{encoding}'"),
        ));
    };
    let (text, _, had_errors) = encoding.decode(bytes);
    if had_errors && encoding.name().eq_ignore_ascii_case("UTF-8") {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: unable to decode file as UTF-8",
        ));
    }
    Ok(text.into_owned())
}

fn extract_docx_text(bytes: &[u8]) -> BuiltinResult<String> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|err| {
        extract_error(
            &ERROR_UNSUPPORTED,
            format!("extractFileText: unable to open DOCX archive: {err}"),
        )
    })?;
    let mut document = archive.by_name("word/document.xml").map_err(|err| {
        extract_error(
            &ERROR_UNSUPPORTED,
            format!("extractFileText: DOCX archive is missing word/document.xml: {err}"),
        )
    })?;
    if document.size() > MAX_DOCX_XML_BYTES as u64 {
        return Err(extract_error(
            &ERROR_UNSUPPORTED,
            "extractFileText: DOCX document.xml is too large for this slice",
        ));
    }
    let mut xml = String::new();
    document.read_to_string(&mut xml).map_err(|err| {
        extract_error(
            &ERROR_INVALID_INPUT,
            format!("extractFileText: unable to decode DOCX XML as UTF-8: {err}"),
        )
    })?;
    Ok(text_from_docx_document_xml(&xml))
}

fn text_from_docx_document_xml(xml: &str) -> String {
    let mut out = String::new();
    let mut cursor = 0usize;
    while cursor < xml.len() {
        let next_text = xml[cursor..].find("<w:t").map(|idx| cursor + idx);
        let next_paragraph = xml[cursor..].find("</w:p>").map(|idx| cursor + idx);
        let next_break = xml[cursor..].find("<w:br").map(|idx| cursor + idx);
        let next_tab = xml[cursor..].find("<w:tab").map(|idx| cursor + idx);
        let next = [next_text, next_paragraph, next_break, next_tab]
            .into_iter()
            .flatten()
            .min();
        let Some(pos) = next else {
            break;
        };
        if Some(pos) == next_text {
            let Some(start_rel) = xml[pos..].find('>') else {
                break;
            };
            let text_start = pos + start_rel + 1;
            let Some(end_rel) = xml[text_start..].find("</w:t>") else {
                break;
            };
            let text_end = text_start + end_rel;
            out.push_str(&xml_unescape(&xml[text_start..text_end]));
            cursor = text_end + "</w:t>".len();
        } else if Some(pos) == next_break {
            if !out.ends_with('\n') {
                out.push('\n');
            }
            cursor = xml[pos..]
                .find('>')
                .map(|idx| pos + idx + 1)
                .unwrap_or(xml.len());
        } else if Some(pos) == next_tab {
            out.push('\t');
            cursor = xml[pos..]
                .find('>')
                .map(|idx| pos + idx + 1)
                .unwrap_or(xml.len());
        } else {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            cursor = pos + "</w:p>".len();
        }
    }
    out.trim_end_matches('\n').to_string()
}

fn xml_unescape(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_time::unix_timestamp_ms;
    use runmat_value::{CellArray, IntegerStorage, Tensor};
    use std::io::Write;

    fn run(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(extract_file_text_builtin(args))
    }

    fn unique_path(name: &str, ext: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_extract_file_text_{}_{}_{}.{}",
            name,
            std::process::id(),
            unix_timestamp_ms(),
            ext
        ));
        path
    }

    fn string_value(value: Value) -> String {
        match value {
            Value::String(text) => text,
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[test]
    fn parse_pages_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2, 5]), vec![1, 2])
            .expect("integer tensor");

        assert_eq!(
            parse_pages(&Value::Tensor(tensor)).expect("pages"),
            vec![2, 5]
        );
    }

    #[test]
    fn parse_pages_accepts_every_integer_scalar_class() {
        for page in [
            IntValue::I8(2),
            IntValue::I16(2),
            IntValue::I32(2),
            IntValue::I64(2),
            IntValue::U8(2),
            IntValue::U16(2),
            IntValue::U32(2),
            IntValue::U64(2),
        ] {
            assert_eq!(parse_pages(&Value::Int(page)).unwrap(), vec![2]);
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn parse_pages_preserves_wide_u64_values_without_f64_materialization() {
        let page = (1_u64 << 53) + 1;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![page]), vec![1, 1]).unwrap();
        assert_eq!(
            parse_pages(&Value::Tensor(tensor)).unwrap(),
            vec![page as usize]
        );
    }

    #[test]
    fn parse_pages_rejects_nonvector_and_invalid_numeric_controls() {
        let matrix = Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();
        assert!(parse_pages(&Value::Tensor(matrix)).is_err());
        for value in [Value::Num(0.0), Value::Num(-1.0), Value::Num(1.5)] {
            assert!(parse_pages(&value).is_err());
        }
    }

    #[test]
    fn extract_file_text_strict_mode_rejects_resident_pages_before_gather() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let error = run(vec![
            Value::String("document.pdf".into()),
            Value::String("Pages".into()),
            resident,
        ])
        .expect_err("strict resident Pages gate");
        assert_eq!(
            error.identifier(),
            RESIDENT_PAGES_EXTENSION.error_identifier
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn reads_plain_text_file_with_encoding_option() {
        let path = unique_path("plain", "txt");
        test_support::fs::write(&path, b"caf\xe9").expect("write sample");
        let value = run(vec![
            Value::String(path.to_string_lossy().to_string()),
            Value::String("Encoding".to_string()),
            Value::String("windows-1252".to_string()),
        ])
        .expect("extract");
        assert_eq!(string_value(value), "café");
        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn extracts_visible_text_from_html_file() {
        let path = unique_path("html", "html");
        test_support::fs::write(
            &path,
            "<html><body><h1>Title</h1><script>hidden()</script><p>Body</p></body></html>",
        )
        .expect("write sample");
        let value = run(vec![
            Value::String(path.to_string_lossy().to_string()),
            Value::String("ExtractionMethod".to_string()),
            Value::String("all-text".to_string()),
        ])
        .expect("extract");
        let text = string_value(value);
        assert!(text.contains("Title"));
        assert!(text.contains("Body"));
        assert!(!text.contains("hidden"));
        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn extracts_text_from_docx_document_xml() {
        let path = unique_path("docx", "docx");
        let file = std::fs::File::create(&path).expect("create docx");
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file("word/document.xml", options)
            .expect("start document");
        zip.write_all(
            br#"<w:document><w:body><w:p><w:r><w:t>Hello &amp; welcome</w:t></w:r></w:p><w:p><w:r><w:t>Second</w:t></w:r></w:p></w:body></w:document>"#,
        )
        .expect("write document");
        zip.finish().expect("finish docx");

        let value = run(vec![Value::String(path.to_string_lossy().to_string())]).expect("extract");
        assert_eq!(string_value(value), "Hello & welcome\nSecond");
        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn rejects_pdf_options_and_urls_as_tracked_gaps() {
        let err = run(vec![
            Value::String("https://example.com".to_string()),
            Value::String("ExtractionMethod".to_string()),
            Value::String("tree".to_string()),
        ])
        .expect_err("expected URL rejection");
        assert!(err.to_string().contains("URL extraction"));

        let err = run(vec![
            Value::String("paper.pdf".to_string()),
            Value::String("Pages".to_string()),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        ])
        .expect_err("expected PDF option rejection");
        assert!(err.to_string().contains("PDF extraction"));

        let err =
            run(vec![Value::String("paper.pdf".to_string())]).expect_err("expected PDF rejection");
        assert!(err.to_string().contains("PDF extraction"));

        let err = run(vec![
            Value::String("report.docx".to_string()),
            Value::String("Encoding".to_string()),
            Value::String("UTF-8".to_string()),
        ])
        .expect_err("expected DOCX encoding rejection");
        assert!(err.to_string().contains("Encoding is supported"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn accepts_single_cell_filename_input() {
        let path = unique_path("cell", "txt");
        test_support::fs::write(&path, "cell path").expect("write sample");
        let cell = CellArray::new(
            vec![Value::CharArray(runmat_value::CharArray::new_row(
                &path.to_string_lossy(),
            ))],
            1,
            1,
        )
        .unwrap();
        let value = run(vec![Value::Cell(cell)]).expect("extract");
        assert_eq!(string_value(value), "cell path");
        let _ = test_support::fs::remove_file(&path);
    }
}
