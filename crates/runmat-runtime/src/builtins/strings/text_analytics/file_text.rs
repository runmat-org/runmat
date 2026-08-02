//! File-backed text extraction helpers for Text Analytics workflows.

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use encoding_rs::Encoding;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::html::{extract_html_text_value, ExtractionMethod};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

const NAME: &str = "extractFileText";
const MAX_FILE_BYTES: usize = 512 * 1024 * 1024;
const MAX_DOCX_XML_BYTES: usize = 128 * 1024 * 1024;

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
        description: "Name-value options: Encoding and ExtractionMethod in this slice.",
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
    type_resolver(string_type),
    descriptor(crate::builtins::strings::text_analytics::file_text::EXTRACT_FILE_TEXT_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::file_text"
)]
async fn extract_file_text_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    let raw = match value {
        Value::Num(value) => vec![*value],
        Value::Tensor(tensor) => tensor_utils::tensor_values_f64(tensor),
        other => {
            return Err(extract_error(
                &ERROR_INVALID_INPUT,
                format!("extractFileText: Pages must be a positive integer vector, got {other:?}"),
            ))
        }
    };
    if raw.is_empty() {
        return Err(extract_error(
            &ERROR_INVALID_INPUT,
            "extractFileText: Pages must not be empty",
        ));
    }
    raw.into_iter()
        .map(|value| {
            if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
                return Err(extract_error(
                    &ERROR_INVALID_INPUT,
                    format!("extractFileText: Pages values must be positive integers, got {value}"),
                ));
            }
            Ok(value as usize)
        })
        .collect()
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
    use runmat_builtins::{CellArray, IntegerStorage, Tensor};
    use runmat_time::unix_timestamp_ms;
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
            vec![Value::String(path.to_string_lossy().to_string())],
            1,
            1,
        )
        .unwrap();
        let value = run(vec![Value::Cell(cell)]).expect("extract");
        assert_eq!(string_value(value), "cell path");
        let _ = test_support::fs::remove_file(&path);
    }
}
