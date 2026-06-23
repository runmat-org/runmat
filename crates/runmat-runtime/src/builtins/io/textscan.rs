//! MATLAB-compatible `textscan` builtin for formatted text imports.

use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::{helpers::decode_bytes, registry};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "textscan";

const TEXTSCAN_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array containing one output per conversion, or collected groups.",
}];
const TEXTSCAN_INPUTS_TEXT_FORMAT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "textOrFileID",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text or file identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format specification such as '%f %s'.",
    },
];
const TEXTSCAN_INPUTS_TEXT_FORMAT_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "textOrFileID",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text or file identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format specification such as '%f %s'.",
    },
    BuiltinParamDescriptor {
        name: "args...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional repeat count followed by name-value pairs.",
    },
];
const TEXTSCAN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = textscan(textOrFileID, formatSpec)",
        inputs: &TEXTSCAN_INPUTS_TEXT_FORMAT,
        outputs: &TEXTSCAN_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = textscan(textOrFileID, formatSpec, args...)",
        inputs: &TEXTSCAN_INPUTS_TEXT_FORMAT_OPTIONS,
        outputs: &TEXTSCAN_OUTPUTS,
    },
];

const TEXTSCAN_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCAN.ARGUMENT",
    identifier: Some("RunMat:textscan:InvalidArgument"),
    when: "Input, format specification, repeat count, or name-value options are malformed.",
    message: "textscan: invalid argument",
};
const TEXTSCAN_ERROR_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCAN.FORMAT",
    identifier: Some("RunMat:textscan:InvalidFormat"),
    when: "Format specification cannot be parsed or contains unsupported conversions.",
    message: "textscan: invalid format specification",
};
const TEXTSCAN_ERROR_FILE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCAN.FILE",
    identifier: Some("RunMat:textscan:File"),
    when: "A file identifier is invalid or cannot be read.",
    message: "textscan: file read failed",
};
const TEXTSCAN_ERROR_PARSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCAN.PARSE",
    identifier: Some("RunMat:textscan:Parse"),
    when: "Input text cannot be parsed according to the format specification.",
    message: "textscan: parse failed",
};
const TEXTSCAN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    TEXTSCAN_ERROR_ARGUMENT,
    TEXTSCAN_ERROR_FORMAT,
    TEXTSCAN_ERROR_FILE,
    TEXTSCAN_ERROR_PARSE,
];

pub const TEXTSCAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEXTSCAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TEXTSCAN_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::textscan")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "textscan",
    op_kind: GpuOpKind::Custom("io-textscan"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs on the host; formatted text import is not an acceleration operation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::textscan")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "textscan",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side formatted text parsing.",
};

fn textscan_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn textscan_error_with_source<E>(
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
    name = "textscan",
    category = "io/import",
    summary = "Parse formatted text from a string or file identifier.",
    keywords = "textscan,formatted text,delimiter,header,format specifier,csv,log import",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::textscan_type),
    descriptor(crate::builtins::io::textscan::TEXTSCAN_DESCRIPTOR),
    builtin_path = "crate::builtins::io::textscan"
)]
async fn textscan_builtin(
    input: Value,
    format_spec: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let input = gather_if_needed_async(&input)
        .await
        .map_err(map_control_flow)?;
    let format_spec = gather_if_needed_async(&format_spec)
        .await
        .map_err(map_control_flow)?;
    let format_spec = string_scalar(&format_spec, "formatSpec")?;
    let gathered_rest = gather_rest(rest).await?;
    let (repeat, options) = parse_args(&gathered_rest)?;
    let parsed = parse_input(&input, &format_spec, repeat, &options)?;
    build_output(parsed, &options)
}

async fn gather_rest(rest: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(rest.len());
    for value in rest {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
}

fn parse_input(
    value: &Value,
    format_spec: &str,
    repeat: Option<usize>,
    options: &TextscanOptions,
) -> BuiltinResult<Vec<ColumnData>> {
    if let Some(fid) = numeric_fid(value) {
        return parse_registered_file(fid, format_spec, repeat, options)
            .map(|parsed| parsed.columns);
    }
    let text = string_scalar(value, "textOrFileID")?;
    parse_textscan(&text, format_spec, repeat, options).map(|parsed| parsed.columns)
}

fn parse_registered_file(
    fid: i32,
    format_spec: &str,
    repeat: Option<usize>,
    options: &TextscanOptions,
) -> BuiltinResult<ParsedTextscan> {
    validate_fid(fid)?;
    let info = registry::info_for(fid).ok_or_else(|| {
        textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: invalid file identifier {fid}"),
        )
    })?;
    if !permission_allows_read(&info.permission) {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: file identifier {fid} is not open for reading"),
        ));
    }
    let handle = registry::shared_handle(fid).ok_or_else(|| {
        textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: invalid file identifier {fid}"),
        )
    })?;
    let mut guard = handle
        .lock()
        .map_err(|_| textscan_error_with(&TEXTSCAN_ERROR_FILE, "textscan: file handle poisoned"))?;
    let file = guard.as_mut().ok_or_else(|| {
        textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: file identifier {fid} is closed"),
        )
    })?;
    let start = file.stream_position().map_err(|err| {
        textscan_error_with_source(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: unable to seek file identifier {fid} ({err})"),
            err,
        )
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|err| {
        textscan_error_with_source(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: unable to read from file identifier {fid} ({err})"),
            err,
        )
    })?;
    let encoding = if info.encoding.trim().is_empty() {
        "UTF-8"
    } else {
        info.encoding.as_str()
    };
    let decoded = DecodedFileText::decode(&bytes, encoding)?;
    let parsed = parse_textscan(&decoded.text, format_spec, repeat, options)?;
    let consumed_bytes = decoded.byte_offset_for_text_pos(parsed.consumed_text_pos)?;
    let target = start.saturating_add(consumed_bytes as u64);
    file.seek(SeekFrom::Start(target)).map_err(|err| {
        textscan_error_with_source(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: unable to restore file position for identifier {fid} ({err})"),
            err,
        )
    })?;
    Ok(parsed)
}

fn validate_fid(fid: i32) -> BuiltinResult<()> {
    if fid < 0 {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            "textscan: file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            "textscan: standard input/output identifiers are not supported yet",
        ));
    }
    Ok(())
}

fn permission_allows_read(permission: &str) -> bool {
    let lower = permission.to_ascii_lowercase();
    lower.starts_with('r') || lower.contains('+')
}

struct DecodedFileText {
    text: String,
    text_offsets: Vec<usize>,
    byte_offsets: Vec<usize>,
    byte_len: usize,
}

impl DecodedFileText {
    fn decode(bytes: &[u8], encoding: &str) -> BuiltinResult<Self> {
        let chars = decode_bytes(bytes, encoding, BUILTIN_NAME)
            .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_FILE, err.message()))?;
        let byte_width = byte_preserving_encoding_width(encoding)?;
        let mut text = String::new();
        let mut text_offsets = Vec::with_capacity(chars.len());
        let mut byte_offsets = Vec::with_capacity(chars.len());
        let mut byte_offset = 0usize;
        for ch in chars {
            text_offsets.push(text.len());
            byte_offsets.push(byte_offset);
            text.push(ch);
            byte_offset += byte_width.unwrap_or_else(|| ch.len_utf8());
        }
        Ok(Self {
            text,
            text_offsets,
            byte_offsets,
            byte_len: bytes.len(),
        })
    }

    fn byte_offset_for_text_pos(&self, text_pos: usize) -> BuiltinResult<usize> {
        if text_pos == self.text.len() {
            return Ok(self.byte_len);
        }
        match self.text_offsets.binary_search(&text_pos) {
            Ok(idx) => Ok(self.byte_offsets[idx]),
            Err(idx) if idx == self.text_offsets.len() => Ok(self.byte_len),
            Err(_) => Err(textscan_error_with(
                &TEXTSCAN_ERROR_FILE,
                "textscan: parsed position did not fall on a decoded character boundary",
            )),
        }
    }
}

fn byte_preserving_encoding_width(encoding: &str) -> BuiltinResult<Option<usize>> {
    let label = encoding.trim();
    if label.is_empty() || label.eq_ignore_ascii_case("utf-8") || label.eq_ignore_ascii_case("utf8")
    {
        return Ok(None);
    }
    if label.eq_ignore_ascii_case("shift_jis")
        || label.eq_ignore_ascii_case("shift-jis")
        || label.eq_ignore_ascii_case("sjis")
    {
        return Ok(None);
    }
    if label.eq_ignore_ascii_case("binary")
        || label.eq_ignore_ascii_case("latin1")
        || label.eq_ignore_ascii_case("latin-1")
        || label.eq_ignore_ascii_case("iso-8859-1")
        || label.eq_ignore_ascii_case("windows-1252")
        || label.eq_ignore_ascii_case("cp1252")
        || label.eq_ignore_ascii_case("us-ascii")
        || label.eq_ignore_ascii_case("ascii")
        || label.eq_ignore_ascii_case("us_ascii")
        || label.eq_ignore_ascii_case("usascii")
    {
        return Ok(Some(1));
    }
    Err(textscan_error_with(
        &TEXTSCAN_ERROR_FILE,
        format!(
            "textscan: file-position preserving reads do not yet support encoding '{encoding}'"
        ),
    ))
}

#[derive(Debug, Clone)]
struct TextscanOptions {
    delimiters: Vec<String>,
    whitespace: String,
    multiple_delims_as_one: bool,
    header_lines: usize,
    treat_as_empty: Vec<String>,
    comment_style: CommentStyle,
    collect_output: bool,
    return_on_error: bool,
}

impl Default for TextscanOptions {
    fn default() -> Self {
        Self {
            delimiters: Vec::new(),
            whitespace: " \u{0008}\t".to_string(),
            multiple_delims_as_one: false,
            header_lines: 0,
            treat_as_empty: Vec::new(),
            comment_style: CommentStyle::None,
            collect_output: false,
            return_on_error: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CommentStyle {
    None,
    Line(Vec<String>),
    Block { start: String, end: String },
}

fn parse_args(args: &[Value]) -> BuiltinResult<(Option<usize>, TextscanOptions)> {
    let mut idx = 0usize;
    let repeat = if let Some(value) = args.first() {
        if is_numeric_scalar(value) {
            idx = 1;
            Some(nonnegative_usize(value, "repeat count")?)
        } else {
            None
        }
    } else {
        None
    };
    if !(args.len() - idx).is_multiple_of(2) {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            "textscan: options must be provided as name-value pairs",
        ));
    }
    let mut options = TextscanOptions::default();
    while idx < args.len() {
        let name = string_scalar(&args[idx], "option name")?;
        let value = &args[idx + 1];
        apply_option(&mut options, &name, value)?;
        idx += 2;
    }
    Ok((repeat, options))
}

fn apply_option(options: &mut TextscanOptions, name: &str, value: &Value) -> BuiltinResult<()> {
    match normalize_name(name).as_str() {
        "delimiter" => options.delimiters = delimiter_list(value)?,
        "multipledelimsasone" => options.multiple_delims_as_one = bool_like(value, name)?,
        "headerlines" => options.header_lines = nonnegative_usize(value, name)?,
        "treatasempty" => options.treat_as_empty = string_list(value, name)?,
        "commentstyle" => options.comment_style = parse_comment_style(value)?,
        "collectoutput" => options.collect_output = bool_like(value, name)?,
        "returnonerror" => options.return_on_error = bool_like(value, name)?,
        "whitespace" => options.whitespace = string_scalar(value, name)?,
        "emptyvalue" | "endofline" | "bufsize" | "expchars" | "texttype" | "datelocale" => {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                format!("textscan: option '{name}' is not implemented yet"),
            ));
        }
        other => {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                format!("textscan: unsupported option '{other}'"),
            ));
        }
    }
    Ok(())
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|ch| *ch != '_' && *ch != ' ')
        .flat_map(char::to_lowercase)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FormatItem {
    kind: FormatKind,
    skip: bool,
    width: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FormatKind {
    Float,
    SignedInt,
    UnsignedInt,
    String,
    QuotedString,
    Char,
    CharSet { chars: HashSet<char>, negated: bool },
}

impl FormatKind {
    fn output_kind(&self) -> OutputKind {
        match self {
            FormatKind::Float | FormatKind::SignedInt | FormatKind::UnsignedInt => {
                OutputKind::Numeric
            }
            FormatKind::String
            | FormatKind::QuotedString
            | FormatKind::Char
            | FormatKind::CharSet { .. } => OutputKind::Text,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputKind {
    Numeric,
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FormatElement {
    Conversion(FormatItem),
    Literal(String),
    Whitespace,
}

fn parse_format_spec(format: &str) -> BuiltinResult<Vec<FormatElement>> {
    let mut elements = Vec::new();
    let mut conversion_count = 0usize;
    let chars: Vec<char> = format.chars().collect();
    let mut idx = 0usize;
    while idx < chars.len() {
        if chars[idx] != '%' {
            if chars[idx].is_whitespace() {
                while idx < chars.len() && chars[idx].is_whitespace() {
                    idx += 1;
                }
                if !matches!(elements.last(), Some(FormatElement::Whitespace)) {
                    elements.push(FormatElement::Whitespace);
                }
                continue;
            }
            let start = idx;
            while idx < chars.len() && chars[idx] != '%' && !chars[idx].is_whitespace() {
                idx += 1;
            }
            elements.push(FormatElement::Literal(chars[start..idx].iter().collect()));
            continue;
        }
        idx += 1;
        if idx < chars.len() && chars[idx] == '%' {
            elements.push(FormatElement::Literal("%".to_string()));
            idx += 1;
            continue;
        }
        let mut skip = false;
        if idx < chars.len() && chars[idx] == '*' {
            skip = true;
            idx += 1;
        }
        let width_start = idx;
        while idx < chars.len() && chars[idx].is_ascii_digit() {
            idx += 1;
        }
        let width = if idx > width_start {
            Some(
                chars[width_start..idx]
                    .iter()
                    .collect::<String>()
                    .parse::<usize>()
                    .map_err(|_| {
                        textscan_error_with(&TEXTSCAN_ERROR_FORMAT, "textscan: invalid field width")
                    })?,
            )
        } else {
            None
        };
        if idx >= chars.len() {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_FORMAT,
                "textscan: incomplete conversion specifier",
            ));
        }
        let kind = match chars[idx] {
            'f' | 'e' | 'E' | 'g' | 'G' | 'n' => {
                idx += 1;
                FormatKind::Float
            }
            'd' | 'i' => {
                idx += 1;
                FormatKind::SignedInt
            }
            'u' => {
                idx += 1;
                FormatKind::UnsignedInt
            }
            's' => {
                idx += 1;
                FormatKind::String
            }
            'q' => {
                idx += 1;
                FormatKind::QuotedString
            }
            'c' => {
                idx += 1;
                FormatKind::Char
            }
            '[' => {
                idx += 1;
                let mut negated = false;
                if idx < chars.len() && chars[idx] == '^' {
                    negated = true;
                    idx += 1;
                }
                let mut set = HashSet::new();
                while idx < chars.len() && chars[idx] != ']' {
                    set.insert(chars[idx]);
                    idx += 1;
                }
                if idx >= chars.len() {
                    return Err(textscan_error_with(
                        &TEXTSCAN_ERROR_FORMAT,
                        "textscan: unterminated character set conversion",
                    ));
                }
                idx += 1;
                FormatKind::CharSet {
                    chars: set,
                    negated,
                }
            }
            other => {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_FORMAT,
                    format!("textscan: unsupported conversion '%{other}'"),
                ));
            }
        };
        elements.push(FormatElement::Conversion(FormatItem { kind, skip, width }));
        conversion_count += 1;
    }
    if conversion_count == 0 {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_FORMAT,
            "textscan: formatSpec must contain at least one conversion",
        ));
    }
    Ok(elements)
}

#[derive(Debug, Clone)]
enum ColumnData {
    Numeric(Vec<f64>),
    Text(Vec<String>),
}

impl ColumnData {
    fn new(kind: OutputKind) -> Self {
        match kind {
            OutputKind::Numeric => ColumnData::Numeric(Vec::new()),
            OutputKind::Text => ColumnData::Text(Vec::new()),
        }
    }

    fn kind(&self) -> OutputKind {
        match self {
            ColumnData::Numeric(_) => OutputKind::Numeric,
            ColumnData::Text(_) => OutputKind::Text,
        }
    }

    fn len(&self) -> usize {
        match self {
            ColumnData::Numeric(values) => values.len(),
            ColumnData::Text(values) => values.len(),
        }
    }

    fn truncate(&mut self, len: usize) {
        match self {
            ColumnData::Numeric(values) => values.truncate(len),
            ColumnData::Text(values) => values.truncate(len),
        }
    }

    fn push_numeric(&mut self, value: f64) {
        let ColumnData::Numeric(values) = self else {
            unreachable!("numeric pushed into text column");
        };
        values.push(value);
    }

    fn push_text(&mut self, value: String) {
        let ColumnData::Text(values) = self else {
            unreachable!("text pushed into numeric column");
        };
        values.push(value);
    }
}

fn parse_textscan(
    text: &str,
    format: &str,
    repeat: Option<usize>,
    options: &TextscanOptions,
) -> BuiltinResult<ParsedTextscan> {
    let elements = parse_format_spec(format)?;
    let output_kinds: Vec<OutputKind> = elements
        .iter()
        .filter_map(|element| match element {
            FormatElement::Conversion(item) if !item.skip => Some(item.kind.output_kind()),
            _ => None,
        })
        .collect();
    let mut columns: Vec<ColumnData> = output_kinds.into_iter().map(ColumnData::new).collect();
    let mut scanner = TextScanner::new(text, options);
    scanner.skip_header_lines();
    let mut records = 0usize;
    while !scanner.is_eof() && repeat.map(|limit| records < limit).unwrap_or(true) {
        scanner.skip_separators();
        if scanner.is_eof() {
            break;
        }
        let row_len = columns.first().map(ColumnData::len).unwrap_or(0);
        let mut output_idx = 0usize;
        let row_start = scanner.pos;
        let mut failed = false;
        for idx in 0..elements.len() {
            match &elements[idx] {
                FormatElement::Whitespace => scanner.skip_format_whitespace(),
                FormatElement::Literal(literal) => {
                    if let Err(err) = scanner.consume_literal(literal) {
                        if options.return_on_error {
                            failed = true;
                            break;
                        }
                        return Err(err);
                    }
                }
                FormatElement::Conversion(item) => {
                    let next_literal = next_literal(&elements[idx + 1..]);
                    let parsed = match scanner.parse_conversion(item, next_literal) {
                        Ok(parsed) => parsed,
                        Err(_) if options.return_on_error => {
                            failed = true;
                            break;
                        }
                        Err(err) => return Err(err),
                    };
                    let Some(parsed) = parsed else {
                        continue;
                    };
                    match parsed {
                        ParsedValue::Number(value) => {
                            if !item.skip {
                                columns[output_idx].push_numeric(value);
                                output_idx += 1;
                            }
                        }
                        ParsedValue::Text(value) => {
                            if !item.skip {
                                columns[output_idx].push_text(value);
                                output_idx += 1;
                            }
                        }
                    }
                }
            }
        }
        if failed || output_idx < columns.len() {
            for column in &mut columns {
                column.truncate(row_len);
            }
            break;
        }
        if scanner.pos == row_start {
            break;
        }
        records += 1;
        scanner.skip_separators();
    }
    Ok(ParsedTextscan {
        columns,
        consumed_text_pos: scanner.pos,
    })
}

#[derive(Debug, Clone)]
struct ParsedTextscan {
    columns: Vec<ColumnData>,
    consumed_text_pos: usize,
}

fn next_literal(elements: &[FormatElement]) -> Option<&str> {
    if let Some(element) = elements.first() {
        return match element {
            FormatElement::Literal(literal) => Some(literal),
            FormatElement::Whitespace | FormatElement::Conversion(_) => None,
        };
    }
    None
}

struct TextScanner<'a> {
    text: &'a str,
    pos: usize,
    options: &'a TextscanOptions,
    whitespace: HashSet<char>,
    delimiters: Vec<String>,
}

impl<'a> TextScanner<'a> {
    fn new(text: &'a str, options: &'a TextscanOptions) -> Self {
        let mut delimiters = options.delimiters.clone();
        delimiters.push("\r\n".to_string());
        delimiters.push("\n".to_string());
        delimiters.push("\r".to_string());
        delimiters.sort_by_key(|delimiter| std::cmp::Reverse(delimiter.len()));
        Self {
            text,
            pos: 0,
            options,
            whitespace: options.whitespace.chars().collect(),
            delimiters,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.text.len()
    }

    fn current_char(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }

    fn skip_header_lines(&mut self) {
        for _ in 0..self.options.header_lines {
            while let Some(ch) = self.current_char() {
                self.pos += ch.len_utf8();
                if ch == '\n' {
                    break;
                }
            }
        }
    }

    fn skip_format_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if !ch.is_whitespace() {
                break;
            }
            self.pos += ch.len_utf8();
        }
    }

    fn skip_separators(&mut self) {
        loop {
            if self.skip_comment() {
                continue;
            }
            if let Some(delimiter) = self.match_delimiter() {
                self.pos += delimiter.len();
                continue;
            }
            let Some(ch) = self.current_char() else {
                break;
            };
            if ch.is_whitespace() || self.whitespace.contains(&ch) {
                self.pos += ch.len_utf8();
                continue;
            }
            break;
        }
    }

    fn consume_literal(&mut self, literal: &str) -> BuiltinResult<()> {
        if self.text[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            return Ok(());
        }
        Err(textscan_error_with(
            &TEXTSCAN_ERROR_PARSE,
            format!("textscan: expected literal '{literal}'"),
        ))
    }

    fn parse_conversion(
        &mut self,
        item: &FormatItem,
        next_literal: Option<&str>,
    ) -> BuiltinResult<Option<ParsedValue>> {
        if !matches!(item.kind, FormatKind::Char | FormatKind::CharSet { .. }) {
            self.skip_separators();
        }
        let parsed = match &item.kind {
            FormatKind::Float => ParsedValue::Number(self.parse_numeric_field(
                item.width,
                next_literal,
                parse_float,
            )?),
            FormatKind::SignedInt => ParsedValue::Number(self.parse_numeric_field(
                item.width,
                next_literal,
                |field| parse_signed_int(field).map(|value| value as f64),
            )?),
            FormatKind::UnsignedInt => ParsedValue::Number(self.parse_numeric_field(
                item.width,
                next_literal,
                |field| parse_unsigned_int(field).map(|value| value as f64),
            )?),
            FormatKind::String => {
                ParsedValue::Text(self.read_field(item.width, next_literal, true)?)
            }
            FormatKind::QuotedString => {
                ParsedValue::Text(self.read_quoted_or_field(item.width, next_literal)?)
            }
            FormatKind::Char => ParsedValue::Text(self.read_chars(item.width.unwrap_or(1))?),
            FormatKind::CharSet { chars, negated } => {
                ParsedValue::Text(self.read_charset(chars, *negated, item.width)?)
            }
        };
        if item.skip {
            Ok(None)
        } else {
            Ok(Some(parsed))
        }
    }

    fn parse_numeric_field(
        &mut self,
        width: Option<usize>,
        next_literal: Option<&str>,
        parse: impl FnOnce(&str) -> BuiltinResult<f64>,
    ) -> BuiltinResult<f64> {
        let field = self.read_field(width, next_literal, false)?;
        if self
            .options
            .treat_as_empty
            .iter()
            .any(|empty| empty == &field)
        {
            return Ok(f64::NAN);
        }
        parse(&field)
    }

    fn read_field(
        &mut self,
        width: Option<usize>,
        next_literal: Option<&str>,
        allow_treat_empty: bool,
    ) -> BuiltinResult<String> {
        let start = self.pos;
        let mut chars = 0usize;
        while !self.is_eof() {
            if width.map(|limit| chars >= limit).unwrap_or(false) {
                break;
            }
            if next_literal
                .filter(|literal| self.text[self.pos..].starts_with(*literal))
                .is_some()
            {
                break;
            }
            if self.is_at_separator() || self.is_at_comment() {
                break;
            }
            let Some(ch) = self.current_char() else {
                break;
            };
            self.pos += ch.len_utf8();
            chars += 1;
        }
        let field = self.text[start..self.pos].trim().to_string();
        if allow_treat_empty
            && self
                .options
                .treat_as_empty
                .iter()
                .any(|empty| empty == &field)
        {
            return Ok(String::new());
        }
        if field.is_empty() {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                "textscan: empty field",
            ));
        }
        Ok(field)
    }

    fn read_quoted_or_field(
        &mut self,
        width: Option<usize>,
        next_literal: Option<&str>,
    ) -> BuiltinResult<String> {
        if self.current_char() != Some('"') {
            return self.read_field(width, next_literal, true);
        }
        self.pos += 1;
        let mut out = String::new();
        while let Some(ch) = self.current_char() {
            self.pos += ch.len_utf8();
            if ch == '"' {
                if self.current_char() == Some('"') {
                    self.pos += 1;
                    out.push('"');
                    continue;
                }
                return Ok(out);
            }
            if width
                .map(|limit| out.chars().count() >= limit)
                .unwrap_or(false)
            {
                return Ok(out);
            }
            out.push(ch);
        }
        Err(textscan_error_with(
            &TEXTSCAN_ERROR_PARSE,
            "textscan: unterminated quoted field",
        ))
    }

    fn read_chars(&mut self, count: usize) -> BuiltinResult<String> {
        let mut out = String::new();
        for _ in 0..count {
            let Some(ch) = self.current_char() else {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_PARSE,
                    "textscan: not enough characters for %c conversion",
                ));
            };
            self.pos += ch.len_utf8();
            out.push(ch);
        }
        Ok(out)
    }

    fn read_charset(
        &mut self,
        chars: &HashSet<char>,
        negated: bool,
        width: Option<usize>,
    ) -> BuiltinResult<String> {
        let mut out = String::new();
        while let Some(ch) = self.current_char() {
            if width
                .map(|limit| out.chars().count() >= limit)
                .unwrap_or(false)
            {
                break;
            }
            if !(chars.contains(&ch) ^ negated) {
                break;
            }
            self.pos += ch.len_utf8();
            out.push(ch);
        }
        if out.is_empty() {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                "textscan: character set conversion matched no characters",
            ));
        }
        Ok(out)
    }

    fn is_at_separator(&self) -> bool {
        self.match_delimiter().is_some()
            || self
                .current_char()
                .map(|ch| ch.is_whitespace() || self.whitespace.contains(&ch))
                .unwrap_or(false)
    }

    fn match_delimiter(&self) -> Option<&str> {
        self.delimiters
            .iter()
            .find(|delimiter| self.text[self.pos..].starts_with(delimiter.as_str()))
            .map(String::as_str)
    }

    fn skip_comment(&mut self) -> bool {
        match &self.options.comment_style {
            CommentStyle::None => false,
            CommentStyle::Line(markers) => {
                if markers
                    .iter()
                    .any(|marker| !marker.is_empty() && self.text[self.pos..].starts_with(marker))
                {
                    while let Some(ch) = self.current_char() {
                        self.pos += ch.len_utf8();
                        if ch == '\n' {
                            break;
                        }
                    }
                    true
                } else {
                    false
                }
            }
            CommentStyle::Block { start, end } => {
                if start.is_empty() || !self.text[self.pos..].starts_with(start) {
                    return false;
                }
                let after_start = self.pos + start.len();
                if let Some(end_idx) = self.text[after_start..].find(end) {
                    self.pos = after_start + end_idx + end.len();
                } else {
                    self.pos = self.text.len();
                }
                true
            }
        }
    }

    fn is_at_comment(&self) -> bool {
        match &self.options.comment_style {
            CommentStyle::None => false,
            CommentStyle::Line(markers) => markers
                .iter()
                .any(|marker| !marker.is_empty() && self.text[self.pos..].starts_with(marker)),
            CommentStyle::Block { start, .. } => {
                !start.is_empty() && self.text[self.pos..].starts_with(start)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum ParsedValue {
    Number(f64),
    Text(String),
}

fn parse_float(token: &str) -> BuiltinResult<f64> {
    match token.trim().to_ascii_lowercase().as_str() {
        "" => Ok(f64::NAN),
        "nan" => Ok(f64::NAN),
        "inf" | "+inf" | "infinity" | "+infinity" => Ok(f64::INFINITY),
        "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
        _ => token.trim().parse::<f64>().map_err(|_| {
            textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                format!("textscan: cannot parse '{token}' as a floating-point value"),
            )
        }),
    }
}

fn parse_signed_int(token: &str) -> BuiltinResult<i64> {
    token.trim().parse::<i64>().map_err(|_| {
        textscan_error_with(
            &TEXTSCAN_ERROR_PARSE,
            format!("textscan: cannot parse '{token}' as an integer value"),
        )
    })
}

fn parse_unsigned_int(token: &str) -> BuiltinResult<u64> {
    token.trim().parse::<u64>().map_err(|_| {
        textscan_error_with(
            &TEXTSCAN_ERROR_PARSE,
            format!("textscan: cannot parse '{token}' as an unsigned integer value"),
        )
    })
}

fn build_output(columns: Vec<ColumnData>, options: &TextscanOptions) -> BuiltinResult<Value> {
    let values = if options.collect_output {
        collect_output(columns)?
    } else {
        columns
            .into_iter()
            .map(column_to_value)
            .collect::<BuiltinResult<Vec<_>>>()?
    };
    let len = values.len();
    CellArray::new(values, 1, len)
        .map(Value::Cell)
        .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}")))
}

fn collect_output(columns: Vec<ColumnData>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < columns.len() {
        if columns[idx].kind() == OutputKind::Numeric {
            let start = idx;
            while idx < columns.len() && columns[idx].kind() == OutputKind::Numeric {
                idx += 1;
            }
            out.push(numeric_group_to_value(&columns[start..idx])?);
        } else {
            out.push(column_to_value(columns[idx].clone())?);
            idx += 1;
        }
    }
    Ok(out)
}

fn column_to_value(column: ColumnData) -> BuiltinResult<Value> {
    match column {
        ColumnData::Numeric(values) => Tensor::new(values.clone(), vec![values.len(), 1])
            .map(Value::Tensor)
            .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}"))),
        ColumnData::Text(values) => cell_string_column(&values),
    }
}

fn numeric_group_to_value(columns: &[ColumnData]) -> BuiltinResult<Value> {
    let rows = columns.first().map(ColumnData::len).unwrap_or(0);
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        let ColumnData::Numeric(values) = column else {
            unreachable!("numeric group contains text column");
        };
        if values.len() != rows {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                "textscan: collected numeric columns have inconsistent lengths",
            ));
        }
        data.extend_from_slice(values);
    }
    Tensor::new(data, vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}")))
}

fn cell_string_column(values: &[String]) -> BuiltinResult<Value> {
    CellArray::new(
        values.iter().cloned().map(Value::String).collect(),
        values.len(),
        1,
    )
    .map(Value::Cell)
    .map_err(|err| textscan_error_with(&TEXTSCAN_ERROR_PARSE, format!("textscan: {err}")))
}

fn string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            format!("textscan: expected {context} as a string scalar or character vector"),
        )),
    }
}

fn string_list(value: &Value, context: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for row in 0..cell.rows {
                for col in 0..cell.cols {
                    out.push(string_scalar(
                        &cell.get(row, col).map_err(|err| {
                            textscan_error_with(
                                &TEXTSCAN_ERROR_ARGUMENT,
                                format!("textscan: {err}"),
                            )
                        })?,
                        context,
                    )?);
                }
            }
            Ok(out)
        }
        Value::StringArray(sa) => Ok(sa.data.clone()),
        _ => Ok(vec![string_scalar(value, context)?]),
    }
}

fn delimiter_list(value: &Value) -> BuiltinResult<Vec<String>> {
    let mut delimiters = string_list(value, "Delimiter")?;
    for delimiter in &mut delimiters {
        *delimiter = match delimiter.as_str() {
            "\\t" => "\t".to_string(),
            "\\n" => "\n".to_string(),
            "\\r" => "\r".to_string(),
            other => other.to_string(),
        };
        if delimiter.is_empty() {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                "textscan: Delimiter entries must not be empty",
            ));
        }
    }
    Ok(delimiters)
}

fn parse_comment_style(value: &Value) -> BuiltinResult<CommentStyle> {
    match value {
        Value::String(s) if s.eq_ignore_ascii_case("none") => Ok(CommentStyle::None),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            if text.eq_ignore_ascii_case("none") {
                Ok(CommentStyle::None)
            } else {
                Ok(CommentStyle::Line(vec![text]))
            }
        }
        Value::Cell(cell) if cell.data.len() == 2 => {
            let first = string_scalar(
                &cell.get(0, 0).map_err(|err| {
                    textscan_error_with(&TEXTSCAN_ERROR_ARGUMENT, format!("textscan: {err}"))
                })?,
                "CommentStyle",
            )?;
            let second = if cell.rows == 1 {
                string_scalar(
                    &cell.get(0, 1).map_err(|err| {
                        textscan_error_with(&TEXTSCAN_ERROR_ARGUMENT, format!("textscan: {err}"))
                    })?,
                    "CommentStyle",
                )?
            } else {
                string_scalar(
                    &cell.get(1, 0).map_err(|err| {
                        textscan_error_with(&TEXTSCAN_ERROR_ARGUMENT, format!("textscan: {err}"))
                    })?,
                    "CommentStyle",
                )?
            };
            Ok(CommentStyle::Block {
                start: first,
                end: second,
            })
        }
        Value::Cell(cell) => {
            let mut markers = Vec::new();
            for row in 0..cell.rows {
                for col in 0..cell.cols {
                    markers.push(string_scalar(
                        &cell.get(row, col).map_err(|err| {
                            textscan_error_with(
                                &TEXTSCAN_ERROR_ARGUMENT,
                                format!("textscan: {err}"),
                            )
                        })?,
                        "CommentStyle",
                    )?);
                }
            }
            Ok(CommentStyle::Line(markers))
        }
        _ => {
            let text = string_scalar(value, "CommentStyle")?;
            if text.eq_ignore_ascii_case("none") {
                Ok(CommentStyle::None)
            } else {
                Ok(CommentStyle::Line(vec![text]))
            }
        }
    }
}

fn bool_like(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if (*value - 0.0).abs() < f64::EPSILON => Ok(false),
        Value::Num(value) if (*value - 1.0).abs() < f64::EPSILON => Ok(true),
        Value::Int(value) if value.to_i64() == 0 => Ok(false),
        Value::Int(value) if value.to_i64() == 1 => Ok(true),
        _ => match string_scalar(value, context)?
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "true" | "on" | "yes" | "1" => Ok(true),
            "false" | "off" | "no" | "0" => Ok(false),
            _ => Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                format!("textscan: {context} must be logical"),
            )),
        },
    }
}

fn nonnegative_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    let raw = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_i64() as f64,
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        _ => {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_ARGUMENT,
                format!("textscan: {context} must be a nonnegative integer scalar"),
            ));
        }
    };
    if !raw.is_finite() || raw < 0.0 || raw.fract() != 0.0 {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_ARGUMENT,
            format!("textscan: {context} must be a nonnegative integer scalar"),
        ));
    }
    Ok(raw as usize)
}

fn numeric_fid(value: &Value) -> Option<i32> {
    let raw = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_i64() as f64,
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        _ => return None,
    };
    if raw.is_finite() && raw.fract() == 0.0 && raw >= i32::MIN as f64 && raw <= i32::MAX as f64 {
        Some(raw as i32)
    } else {
        None
    }
}

fn is_numeric_scalar(value: &Value) -> bool {
    numeric_fid(value).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_filesystem::OpenOptions;
    use std::sync::{Arc, Mutex as StdMutex};

    use crate::builtins::io::filetext::registry::RegisteredFile;

    fn output_cell(value: &Value) -> &CellArray {
        let Value::Cell(cell) = value else {
            panic!("expected cell array output");
        };
        cell
    }

    fn output_value(value: &Value, col: usize) -> Value {
        output_cell(value).get(0, col).expect("output cell")
    }

    fn numeric_column(value: &Value, col: usize) -> Vec<f64> {
        let Value::Tensor(tensor) = output_value(value, col) else {
            panic!("expected tensor");
        };
        tensor.data
    }

    fn text_column(value: &Value, col: usize) -> Vec<String> {
        let Value::Cell(cell) = output_value(value, col) else {
            panic!("expected text cell column");
        };
        let mut out = Vec::new();
        for row in 0..cell.rows {
            let Value::String(text) = cell.get(row, 0).expect("text cell") else {
                panic!("expected string");
            };
            out.push(text);
        }
        out
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_descriptor_covers_core_forms() {
        let labels: Vec<&str> = TEXTSCAN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"C = textscan(textOrFileID, formatSpec)"));
        assert!(labels.contains(&"C = textscan(textOrFileID, formatSpec, args...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_reads_mixed_columns_from_text() {
        let out = block_on(textscan_builtin(
            Value::from("1 alpha\n2 beta\n"),
            Value::from("%f %s"),
            Vec::new(),
        ))
        .expect("textscan");
        assert_eq!(numeric_column(&out, 0), vec![1.0, 2.0]);
        assert_eq!(
            text_column(&out, 1),
            vec!["alpha".to_string(), "beta".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_honors_delimiter_header_comments_and_treat_empty() {
        let text = "# header\nA,1.5\nB,NA\n% ignored\nC,3.5\n";
        let out = block_on(textscan_builtin(
            Value::from(text),
            Value::from("%s %f"),
            vec![
                Value::from("Delimiter"),
                Value::from(","),
                Value::from("HeaderLines"),
                Value::Num(1.0),
                Value::from("CommentStyle"),
                Value::from("%"),
                Value::from("TreatAsEmpty"),
                Value::from("NA"),
            ],
        ))
        .expect("textscan");
        assert_eq!(
            text_column(&out, 0),
            vec!["A".to_string(), "B".to_string(), "C".to_string()]
        );
        let nums = numeric_column(&out, 1);
        assert_eq!(nums[0], 1.5);
        assert!(nums[1].is_nan());
        assert_eq!(nums[2], 3.5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_supports_repeat_skip_collect_and_quotes() {
        let out = block_on(textscan_builtin(
            Value::from("1,drop,2,\"hello, world\"\n3,drop,4,\"tail\"\n"),
            Value::from("%f %*s %f %q"),
            vec![
                Value::Num(1.0),
                Value::from("Delimiter"),
                Value::from(","),
                Value::from("CollectOutput"),
                Value::Bool(true),
            ],
        ))
        .expect("textscan");
        let Value::Tensor(group) = output_value(&out, 0) else {
            panic!("expected collected numeric group");
        };
        assert_eq!(group.shape, vec![1, 2]);
        assert_eq!(group.data, vec![1.0, 2.0]);
        assert_eq!(text_column(&out, 1), vec!["hello, world".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_return_on_error_false_reports_parse_failure() {
        let err = block_on(textscan_builtin(
            Value::from("1\nbad\n"),
            Value::from("%f"),
            vec![Value::from("ReturnOnError"), Value::Bool(false)],
        ))
        .expect_err("parse failure");
        assert!(err.message().contains("cannot parse"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_honors_literals_in_format_spec() {
        let out = block_on(textscan_builtin(
            Value::from("1,2\n3,4\n"),
            Value::from("%f,%f"),
            Vec::new(),
        ))
        .expect("textscan");
        assert_eq!(numeric_column(&out, 0), vec![1.0, 3.0]);
        assert_eq!(numeric_column(&out, 1), vec![2.0, 4.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_width_and_char_conversions_leave_remainder() {
        let strings = block_on(textscan_builtin(
            Value::from("abcdef"),
            Value::from("%2s%s"),
            Vec::new(),
        ))
        .expect("textscan strings");
        assert_eq!(text_column(&strings, 0), vec!["ab".to_string()]);
        assert_eq!(text_column(&strings, 1), vec!["cdef".to_string()]);

        let chars = block_on(textscan_builtin(
            Value::from("abc"),
            Value::from("%c%c%c"),
            Vec::new(),
        ))
        .expect("textscan chars");
        assert_eq!(text_column(&chars, 0), vec!["a".to_string()]);
        assert_eq!(text_column(&chars, 1), vec!["b".to_string()]);
        assert_eq!(text_column(&chars, 2), vec!["c".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_collect_output_preserves_column_major_numeric_group() {
        let out = block_on(textscan_builtin(
            Value::from("1 2\n3 4\n"),
            Value::from("%f %f"),
            vec![Value::from("CollectOutput"), Value::Bool(true)],
        ))
        .expect("textscan");
        let Value::Tensor(group) = output_value(&out, 0) else {
            panic!("expected collected numeric group");
        };
        assert_eq!(group.shape, vec![2, 2]);
        assert_eq!(group.data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_repeated_file_read_preserves_next_record_position() {
        let _guard = registry::test_guard();
        registry::reset_for_tests();
        let mut path = std::env::temp_dir();
        path.push("runmat_textscan_file_position.txt");
        std::fs::write(&path, "10 ten\n20 twenty\n").expect("write fixture");

        let mut options = OpenOptions::new();
        options.read(true);
        let file = block_on(options.open_async(&path)).expect("open file");
        let handle = Arc::new(StdMutex::new(Some(file)));
        let fid = registry::register_file(RegisteredFile {
            path: path.clone(),
            permission: "r".to_string(),
            machinefmt: "native".to_string(),
            encoding: "UTF-8".to_string(),
            handle: handle.clone(),
        });

        let out = block_on(textscan_builtin(
            Value::Num(fid as f64),
            Value::from("%f %s"),
            vec![Value::Num(1.0)],
        ))
        .expect("textscan");
        assert_eq!(numeric_column(&out, 0), vec![10.0]);
        assert_eq!(text_column(&out, 1), vec!["ten".to_string()]);

        let mut remaining = String::new();
        let mut guard = handle.lock().expect("lock");
        let file = guard.as_mut().expect("file");
        std::io::Read::read_to_string(file, &mut remaining).expect("read remaining");
        assert_eq!(remaining, "20 twenty\n");

        let _ = registry::close(fid);
        let _ = std::fs::remove_file(path);
        registry::reset_for_tests();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_rejects_standard_stream_identifiers() {
        let err = block_on(textscan_builtin(
            Value::Num(0.0),
            Value::from("%f"),
            Vec::new(),
        ))
        .expect_err("standard stream rejected");
        assert!(err.message().contains("standard input/output"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn textscan_reads_from_registered_file_identifier() {
        let _guard = registry::test_guard();
        registry::reset_for_tests();
        let mut path = std::env::temp_dir();
        path.push("runmat_textscan_registered_file.txt");
        std::fs::write(&path, "skip\n10 ten\n20 twenty\n").expect("write fixture");

        let mut options = OpenOptions::new();
        options.read(true);
        let file = block_on(options.open_async(&path)).expect("open file");
        let handle = Arc::new(StdMutex::new(Some(file)));
        let fid = registry::register_file(RegisteredFile {
            path: path.clone(),
            permission: "r".to_string(),
            machinefmt: "native".to_string(),
            encoding: "UTF-8".to_string(),
            handle,
        });

        let out = block_on(textscan_builtin(
            Value::Num(fid as f64),
            Value::from("%f %s"),
            vec![Value::from("HeaderLines"), Value::Num(1.0)],
        ))
        .expect("textscan");
        assert_eq!(numeric_column(&out, 0), vec![10.0, 20.0]);
        assert_eq!(
            text_column(&out, 1),
            vec!["ten".to_string(), "twenty".to_string()]
        );

        let _ = registry::close(fid);
        let _ = std::fs::remove_file(path);
        registry::reset_for_tests();
    }
}
