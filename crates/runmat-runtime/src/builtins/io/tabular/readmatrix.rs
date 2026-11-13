//! MATLAB-compatible `readmatrix` builtin for RunMat.

use runmat_value::NumericScalar;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::convert::TryFrom;
use std::io::Read;
use std::path::{Path, PathBuf};

use encoding_rs::Encoding;
#[cfg(test)]
use runmat_builtins::READMATRIX_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, ReadMatrixControl, ReadMatrixOutputClass, READMATRIX_ERROR_ARG_CONFIG,
    READMATRIX_ERROR_FILENAME, READMATRIX_ERROR_FILENAME_EMPTY, READMATRIX_ERROR_IO_OPEN,
    READMATRIX_ERROR_IO_READ, READMATRIX_ERROR_NON_NUMERIC_TOKEN, READMATRIX_ERROR_OPTION_NAME,
    READMATRIX_ERROR_OPTION_VALUE, READMATRIX_ERROR_OUTPUT_TYPE, READMATRIX_ERROR_RANGE,
    READMATRIX_ERROR_TENSOR_BUILD, READMATRIX_LIKE_EXTENSION,
    READMATRIX_TYPED_INTEGER_CONTROL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntegerStorage, LogicalArray, NumericDType, NumericStorage, Tensor, Value};

use super::cooperation::{await_io, TabularCooperation};
use super::import_options::{ImportOptions, VariableImportOptions};
use super::path::{BorrowedTabularReadPath, TabularFsOwner, TabularReadPath, TabularReadSource};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::io::filetext::helpers::decode_bytes;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "readmatrix";
const MAX_READMATRIX_TEXT_BYTES: u64 = 512 * 1024 * 1024;

fn readmatrix_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    readmatrix_error_with(error, error.message)
}

fn readmatrix_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn readmatrix_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
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
    name = "readmatrix",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::tabular::readmatrix"
)]
pub(crate) async fn readmatrix_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut cooperation = TabularCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    if raw_args_contain_option(rest.view(), ReadMatrixControl::Like) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &READMATRIX_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest.iter().any(value_contains_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &READMATRIX_TYPED_INTEGER_CONTROL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    preflight_host_options(runtime.as_ref(), rest.view())?;
    if admission::finite_budget(runtime.as_ref()) {
        if matches!(&path, Value::String(runmat_value::StringElement::Text(raw)) if raw.is_empty())
        {
            return Err(readmatrix_error(&READMATRIX_ERROR_FILENAME_EMPTY));
        }
        if let (Some(runtime), Value::String(runmat_value::StringElement::Text(raw))) =
            (runtime.as_ref(), &path)
        {
            let provider = runmat_filesystem::current_provider();
            let borrowed = Path::new(raw.as_str());
            if rest.is_empty()
                && provider.path_namespace().is_absolute(borrowed)
                && crate::output_count::current_output_count().is_none_or(|count| count == 1)
            {
                let stage = admission::admit_default_absolute(runtime, borrowed, &provider)?;
                let selected = BorrowedTabularReadPath::capture_absolute_on(provider, borrowed)
                    .map_err(|source| {
                        readmatrix_error_with_source(
                            &READMATRIX_ERROR_IO_OPEN,
                            "readmatrix: unable to capture admitted filesystem source",
                            source,
                        )
                    })?;
                return reader::read_default_staged(
                    runtime,
                    &mut cooperation,
                    &selected,
                    selected.provider(),
                    stage,
                )
                .await;
            }
        }
        return Err(build_runtime_error(
            "readmatrix: finite admission for these path, control, or output forms is unavailable",
        )
        .with_builtin(BUILTIN_NAME)
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }
    let known_path = if matches!(path, Value::GpuTensor(_)) {
        None
    } else {
        Some(resolve_path(&path)?)
    };
    let owner = TabularFsOwner::capture(known_path.as_deref()).map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_OPEN,
            "readmatrix: unable to capture filesystem authority",
            source,
        )
    })?;
    let path_value = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&path),
    )
    .await?
    .map_err(map_control_flow)?;
    cooperation.completed(0).await?;
    let options = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        parse_options(runtime.as_ref(), rest.view()),
    )
    .await??;
    cooperation.completed(0).await?;
    options.validate()?;
    let resolved = resolve_path(&path_value)?;
    let selected = owner.path(resolved).map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_OPEN,
            "readmatrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    let tensor =
        read_numeric_matrix(runtime.as_ref(), &mut cooperation, &selected, &options).await?;
    cooperation.completed(0).await?;
    selected.check().map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    let output = finalize_output(tensor, &options)?;
    selected.check().map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    Ok(output)
}

fn value_contains_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
        || matches!(value, Value::Struct(structure) if structure.fields.values().any(value_contains_typed_integer))
        || matches!(value, Value::Object(object) if object.properties().values().any(value_contains_typed_integer))
        || matches!(value, Value::Cell(cell) if cell.data().iter().any(value_contains_typed_integer))
}

fn raw_args_contain_option(
    args: crate::call::FunctionArgumentView<'_>,
    sought: ReadMatrixControl,
) -> bool {
    let first_pair = match args.first() {
        Some(Value::Struct(structure)) => {
            if structure
                .fields
                .keys()
                .any(|name| ReadMatrixControl::parse(name) == sought)
            {
                return true;
            }
            1
        }
        Some(Value::Object(object)) => {
            if object
                .properties()
                .keys()
                .any(|name| ReadMatrixControl::parse(name) == sought)
            {
                return true;
            }
            1
        }
        _ => 0,
    };
    (first_pair..args.len()).step_by(2).any(|index| {
        raw_scalar_text(&args[index]).is_some_and(|name| ReadMatrixControl::parse(&name) == sought)
    })
}

fn raw_scalar_text(value: &Value) -> Option<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
}

async fn parse_options(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ReadMatrixOptions> {
    let mut options = ReadMatrixOptions::default();
    let mut index = 0usize;
    if let Some(source) = args
        .get(index)
        .filter(|value| matches!(value, Value::Struct(_) | Value::Object(_)))
    {
        match source {
            Value::Struct(structure) => parse_struct_options(structure, &mut options).await?,
            _ => {
                let import = ImportOptions::decode(runtime, source)?.ok_or_else(|| {
                    readmatrix_error_with(
                        &READMATRIX_ERROR_ARG_CONFIG,
                        "readmatrix: expected tabular import options",
                    )
                })?;
                options.apply_import_options(&import).await?;
            }
        }
        index += 1;
    }
    while index < args.len() {
        if index + 1 >= args.len() {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_ARG_CONFIG,
                "readmatrix: name/value inputs must appear in pairs",
            ));
        }
        let name_value = gather_if_needed_async(&args[index])
            .await
            .map_err(map_control_flow)?;
        let name = option_name_from_value(&name_value)?;
        let value = &args[index + 1];
        apply_option(&mut options, &name, value).await?;
        index += 2;
    }
    Ok(options)
}

async fn parse_struct_options(
    struct_value: &runmat_value::StructValue,
    options: &mut ReadMatrixOptions,
) -> BuiltinResult<()> {
    for (name, value) in &struct_value.fields {
        apply_option(options, name, value).await?;
    }
    Ok(())
}

async fn apply_option(
    options: &mut ReadMatrixOptions,
    name: &str,
    value: &Value,
) -> BuiltinResult<()> {
    let control = ReadMatrixControl::parse(name);
    let effective = if control == ReadMatrixControl::Like {
        value.clone()
    } else {
        gather_if_needed_async(value)
            .await
            .map_err(map_control_flow)?
    };
    if control == ReadMatrixControl::TreatAsMissing {
        for token in parse_treat_as_missing(&effective).await? {
            options.add_missing_token(&token);
        }
        return Ok(());
    }
    apply_decoded_control(options, control, effective)
}

fn apply_decoded_control(
    options: &mut ReadMatrixOptions,
    control: ReadMatrixControl,
    value: Value,
) -> BuiltinResult<()> {
    match control {
        ReadMatrixControl::Delimiter => options.delimiter = Some(parse_delimiter(&value)?),
        ReadMatrixControl::NumHeaderLines => {
            options.num_header_lines = value_to_usize(&value, "NumHeaderLines")?;
        }
        ReadMatrixControl::DecimalSeparator => {
            options.decimal_separator = parse_separator_char(&value, "DecimalSeparator")?;
        }
        ReadMatrixControl::ThousandsSeparator => {
            options.thousands_separator = Some(parse_separator_char(&value, "ThousandsSeparator")?);
        }
        ReadMatrixControl::EmptyValue => {
            options.empty_value = Some(value_to_numeric_scalar(&value, "EmptyValue")?);
        }
        ReadMatrixControl::OutputType => {
            options.set_output_type(&value_to_string_scalar(&value, "OutputType")?)?;
        }
        ReadMatrixControl::Range => options.range = Some(parse_range(&value)?),
        ReadMatrixControl::Encoding => {
            let encoding = value_to_string_scalar(&value, "Encoding")?;
            validate_encoding_label(&encoding)?;
            options.encoding = Cow::Owned(encoding);
        }
        ReadMatrixControl::Like => options.set_like(value)?,
        ReadMatrixControl::TreatAsMissing | ReadMatrixControl::Unknown => {}
    }
    Ok(())
}

fn preflight_host_options(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let mut options = ReadMatrixOptions::default();
    let mut index = 0;
    if let Some(source) = args
        .first()
        .filter(|value| matches!(value, Value::Struct(_) | Value::Object(_)))
    {
        match source {
            Value::Struct(structure) => {
                for (name, value) in &structure.fields {
                    preflight_host_option(&mut options, ReadMatrixControl::parse(name), value)?;
                }
            }
            _ => {
                let import = ImportOptions::decode(runtime, source)?.ok_or_else(|| {
                    readmatrix_error_with(
                        &READMATRIX_ERROR_ARG_CONFIG,
                        "readmatrix: expected tabular import options",
                    )
                })?;
                options.preflight_import_options(&import)?;
            }
        }
        index = 1;
    }
    while index < args.len() {
        if index + 1 >= args.len() {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_ARG_CONFIG,
                "readmatrix: name/value inputs must appear in pairs",
            ));
        }
        if !matches!(args[index], Value::GpuTensor(_)) {
            let name = option_name_from_value(&args[index])?;
            preflight_host_option(
                &mut options,
                ReadMatrixControl::parse(&name),
                &args[index + 1],
            )?;
        }
        index += 2;
    }
    options.validate()
}

fn preflight_host_option(
    options: &mut ReadMatrixOptions,
    control: ReadMatrixControl,
    value: &Value,
) -> BuiltinResult<()> {
    if control == ReadMatrixControl::TreatAsMissing {
        return validate_host_missing_tokens(value);
    }
    if !matches!(value, Value::GpuTensor(_)) {
        apply_decoded_control(options, control, value.clone())?;
    }
    Ok(())
}

fn validate_host_missing_tokens(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::GpuTensor(_) | Value::String(_) | Value::Num(_) | Value::Int(_) => Ok(()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(()),
        Value::StringArray(_) => Ok(()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => Ok(()),
        Value::Cell(cell) => {
            for item in cell.data() {
                validate_host_missing_tokens(item)?;
            }
            Ok(())
        }
        _ => Err(readmatrix_error_with(
            &READMATRIX_ERROR_OPTION_VALUE,
            "readmatrix: TreatAsMissing values must be strings or numeric scalars",
        )),
    }
}

fn option_name_from_value(value: &Value) -> BuiltinResult<String> {
    value_to_string_scalar(value, "option name")
        .map_err(|_| readmatrix_error(&READMATRIX_ERROR_OPTION_NAME))
}

fn parse_delimiter(value: &Value) -> BuiltinResult<Delimiter> {
    let text = value_to_string_scalar(value, "Delimiter")?;
    if text.is_empty() {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_OPTION_VALUE,
            "readmatrix: Delimiter cannot be empty",
        ));
    }
    let trimmed_lower = text.trim().to_ascii_lowercase();
    match trimmed_lower.as_str() {
        "tab" => Ok(Delimiter::Char('\t')),
        "space" | "whitespace" => Ok(Delimiter::Whitespace),
        "comma" => Ok(Delimiter::Char(',')),
        "semicolon" => Ok(Delimiter::Char(';')),
        "pipe" => Ok(Delimiter::Char('|')),
        _ => {
            if text.chars().count() == 1 {
                Ok(Delimiter::Char(text.chars().next().unwrap()))
            } else {
                Ok(Delimiter::String(text))
            }
        }
    }
}

fn parse_separator_char(value: &Value, option_name: &str) -> BuiltinResult<char> {
    let text = value_to_string_scalar(value, option_name)?;
    if text.is_empty() {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_OPTION_VALUE,
            format!("readmatrix: {option_name} must be a single character"),
        ));
    }
    let mut chars = text.chars();
    let ch = chars.next().unwrap();
    if chars.next().is_some() {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_OPTION_VALUE,
            format!("readmatrix: {option_name} must be a single character"),
        ));
    }
    if ch == '\n' || ch == '\r' {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_OPTION_VALUE,
            format!("readmatrix: {option_name} cannot be a newline character"),
        ));
    }
    Ok(ch)
}

fn value_to_string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    if let Some(text) = crate::builtins::common::validation::value_text_scalar_string(value) {
        return Ok(text);
    }
    if matches!(value, Value::StringArray(_)) {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_OPTION_VALUE,
            format!("readmatrix: {context} must be a scalar string array"),
        ));
    }
    Err(readmatrix_error_with(
        &READMATRIX_ERROR_OPTION_VALUE,
        format!("readmatrix: expected {context} as a string scalar or character vector"),
    ))
}

fn value_to_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            readmatrix_error_with(
                &READMATRIX_ERROR_OPTION_VALUE,
                format!("readmatrix: {context} must be a non-negative integer"),
            )
        });
    }

    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OPTION_VALUE,
                    format!("readmatrix: {context} must be a finite non-negative integer"),
                ));
            }
            if *n < 0.0 {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OPTION_VALUE,
                    format!("readmatrix: {context} must be a non-negative integer"),
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OPTION_VALUE,
                    format!("readmatrix: {context} must be an integer value"),
                ));
            }
            if rounded > usize::MAX.saturating_sub(1) as f64 {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OPTION_VALUE,
                    format!("readmatrix: {context} is too large"),
                ));
            }
            let parsed = rounded as usize;
            if parsed as f64 != rounded || parsed == usize::MAX {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OPTION_VALUE,
                    format!("readmatrix: {context} is too large"),
                ));
            }
            Ok(parsed)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            value_to_usize(&Value::Num(tensor::tensor_value_f64(t, 0)), context)
        }
        _ => Err(readmatrix_error_with(
            &READMATRIX_ERROR_OPTION_VALUE,
            format!("readmatrix: {context} must be provided as a numeric scalar"),
        )),
    }
}

fn value_to_numeric_scalar(value: &Value, context: &str) -> BuiltinResult<NumericScalar> {
    Ok(match value {
        Value::Num(value) => NumericScalar::F64(*value),
        Value::Int(value) => NumericScalar::from(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .numeric_value_at(0)
            .expect("scalar tensor has authoritative numeric storage"),
        _ => {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_OPTION_VALUE,
                format!("readmatrix: {context} must be a numeric scalar"),
            ))
        }
    })
}

async fn parse_treat_as_missing(value: &Value) -> BuiltinResult<Vec<String>> {
    let mut out = Vec::new();
    let mut stack = vec![value.clone()];
    while let Some(value) = stack.pop() {
        match value {
            Value::String(runmat_value::StringElement::Text(s)) => {
                out.push(s.into_string_unbounded())
            }
            Value::CharArray(ca) if ca.rows == 1 => out.push(
                crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                    readmatrix_error_with(
                        &READMATRIX_ERROR_OPTION_VALUE,
                        "readmatrix: TreatAsMissing contains malformed UTF-16 text",
                    )
                })?,
            ),
            Value::StringArray(sa) if sa.data().iter().all(|text| !text.is_missing()) => out
                .extend(
                    sa.data()
                        .iter()
                        .map(|text| text.as_str().expect("checked nonmissing string").to_owned()),
                ),
            Value::Num(n) => out.push(format_numeric_token(n)),
            Value::Int(i) => out.push(i.decimal_string()),
            Value::Tensor(t) => {
                if tensor::is_scalar_tensor(&t) {
                    if let Some(storage) = t.integer_storage() {
                        let value = storage.value_at(0).expect("one-element integer storage");
                        out.push(value.decimal_string());
                    } else {
                        out.push(format_numeric_token(tensor::tensor_value_f64(&t, 0)));
                    }
                } else {
                    return Err(readmatrix_error_with(
                        &READMATRIX_ERROR_OPTION_VALUE,
                        "readmatrix: TreatAsMissing entries must be scalar values",
                    ));
                }
            }
            Value::Cell(cell) => {
                for handle in cell.data() {
                    let inner = handle;
                    let gathered = gather_if_needed_async(inner)
                        .await
                        .map_err(map_control_flow)?;
                    stack.push(gathered);
                }
            }
            _ => {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OPTION_VALUE,
                    "readmatrix: TreatAsMissing values must be strings or numeric scalars",
                ))
            }
        }
    }
    Ok(out)
}

fn format_numeric_token(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{}", value)
    }
}

#[derive(Clone)]
struct ReadMatrixOptions {
    variable_options: Vec<VariableImportOptions>,
    variable_source_origin: Option<usize>,
    delimiter: Option<Delimiter>,
    num_header_lines: usize,
    decimal_separator: char,
    thousands_separator: Option<char>,
    treat_as_missing: BTreeSet<String>,
    empty_value: Option<NumericScalar>,
    range: Option<RangeSpec>,
    output_template: OutputTemplate,
    output_type_seen: bool,
    encoding: Cow<'static, str>,
}

impl Default for ReadMatrixOptions {
    fn default() -> Self {
        Self {
            variable_options: Vec::new(),
            variable_source_origin: None,
            delimiter: None,
            num_header_lines: 0,
            decimal_separator: '.',
            thousands_separator: None,
            treat_as_missing: BTreeSet::new(),
            empty_value: None,
            range: None,
            output_template: OutputTemplate::Numeric(NumericDType::F64),
            output_type_seen: false,
            encoding: Cow::Borrowed("utf-8"),
        }
    }
}

impl ReadMatrixOptions {
    fn preflight_import_options(&mut self, import: &ImportOptions) -> BuiltinResult<()> {
        if import.source() != super::import_options::ImportSource::DelimitedText {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_ARG_CONFIG,
                "readmatrix: spreadsheet import options require a spreadsheet reader",
            ));
        }
        for variable in import.variables() {
            if !variable.type_name.eq_ignore_ascii_case("auto")
                && ReadMatrixOutputClass::parse(&variable.type_name).is_none()
            {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OUTPUT_TYPE,
                    format!("readmatrix: variable '{}' is not numeric", variable.name),
                ));
            }
            if variable.input_format.is_some() {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OPTION_VALUE,
                    "readmatrix: InputFormat requires a datetime table variable",
                ));
            }
            variable.numeric_fill()?;
        }
        self.variable_options = import.variables().to_vec();
        self.variable_source_origin = import.source_origin();
        for (name, value) in &import.base().fields {
            let control = ReadMatrixControl::parse(name);
            if control != ReadMatrixControl::Unknown {
                preflight_host_option(self, control, value)?;
            }
        }
        self.validate()
    }

    async fn apply_import_options(&mut self, import: &ImportOptions) -> BuiltinResult<()> {
        self.preflight_import_options(import)?;
        for (name, value) in &import.base().fields {
            let control = ReadMatrixControl::parse(name);
            if control != ReadMatrixControl::Unknown {
                apply_option(self, name, value).await?;
            }
        }
        Ok(())
    }

    fn variable(&self, column: MatrixFieldColumn) -> Option<&VariableImportOptions> {
        column.selected_index()?;
        let selected_width = self
            .range
            .as_ref()
            .and_then(|range| range.end_col?.checked_sub(range.start_col)?.checked_add(1));
        let projection = super::import_options::ImportColumnProjection::new(
            self.variable_source_origin,
            self.range.as_ref().map_or(0, |range| range.start_col),
        )
        .with_selected_width(self.variable_options.len(), selected_width);
        self.variable_options
            .get(projection.metadata_index_for_source(column.source_index())?)
    }

    fn add_missing_token(&mut self, token: &str) {
        let normalized = normalize_missing_token(token);
        self.treat_as_missing.insert(normalized);
    }

    fn is_missing_token(&self, token: &str) -> bool {
        if self.treat_as_missing.is_empty() {
            return false;
        }
        let norm = normalize_missing_token(token);
        self.treat_as_missing.contains(&norm)
    }

    fn is_missing_token_for(&self, token: &str, column: MatrixFieldColumn) -> bool {
        self.is_missing_token(token)
            || self
                .variable(column)
                .is_some_and(|variable| variable.is_missing(token))
    }

    fn empty_value(&self) -> NumericScalar {
        self.empty_value.unwrap_or(NumericScalar::F64(f64::NAN))
    }

    fn empty_value_f64(&self) -> f64 {
        self.empty_value().materialize_f64()
    }

    fn empty_value_for(&self, column: MatrixFieldColumn) -> BuiltinResult<NumericScalar> {
        match self.variable(column) {
            Some(variable) => {
                if variable.empty_field_rule == super::import_options::EmptyFieldRule::Error {
                    return Err(readmatrix_error_with(
                        &READMATRIX_ERROR_OPTION_VALUE,
                        format!("readmatrix: empty field in variable '{}'", variable.name),
                    ));
                }
                Ok(variable
                    .numeric_fill()?
                    .unwrap_or_else(|| self.empty_value()))
            }
            None => Ok(self.empty_value()),
        }
    }

    fn missing_value_for(&self, column: MatrixFieldColumn) -> BuiltinResult<NumericScalar> {
        match self.variable(column) {
            Some(variable) => Ok(variable
                .numeric_fill()?
                .unwrap_or(NumericScalar::F64(f64::NAN))),
            None => Ok(NumericScalar::F64(f64::NAN)),
        }
    }

    fn validate(&self) -> BuiltinResult<()> {
        if let Some(range) = &self.range {
            range.validate()?;
        }
        if let Some(sep) = self.thousands_separator {
            if sep == self.decimal_separator {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OPTION_VALUE,
                    "readmatrix: DecimalSeparator and ThousandsSeparator must differ",
                ));
            }
        }
        for variable in &self.variable_options {
            let decimal = variable.decimal_separator.unwrap_or(self.decimal_separator);
            let thousands = variable
                .thousands_separator
                .unwrap_or(self.thousands_separator);
            if thousands == Some(decimal) {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OPTION_VALUE,
                    format!(
                        "readmatrix: numeric separators coincide for variable '{}'",
                        variable.name
                    ),
                ));
            }
        }
        Ok(())
    }

    fn set_output_type(&mut self, spec: &str) -> BuiltinResult<()> {
        if matches!(self.output_template, OutputTemplate::Like(_)) {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_OUTPUT_TYPE,
                "readmatrix: cannot combine 'Like' with OutputType",
            ));
        }
        let kind = ReadMatrixOutputClass::parse(spec).ok_or_else(|| {
            readmatrix_error_with(
                &READMATRIX_ERROR_OUTPUT_TYPE,
                format!("readmatrix: unsupported OutputType '{}'", spec),
            )
        })?;
        self.output_template = match kind {
            ReadMatrixOutputClass::Logical => OutputTemplate::Logical,
            ReadMatrixOutputClass::Numeric(class) => OutputTemplate::Numeric(match class {
                runmat_types::NumericClass::Double => NumericDType::F64,
                runmat_types::NumericClass::Single => NumericDType::F32,
                runmat_types::NumericClass::Int8 => NumericDType::I8,
                runmat_types::NumericClass::Int16 => NumericDType::I16,
                runmat_types::NumericClass::Int32 => NumericDType::I32,
                runmat_types::NumericClass::Int64 => NumericDType::I64,
                runmat_types::NumericClass::UInt8 => NumericDType::U8,
                runmat_types::NumericClass::UInt16 => NumericDType::U16,
                runmat_types::NumericClass::UInt32 => NumericDType::U32,
                runmat_types::NumericClass::UInt64 => NumericDType::U64,
            }),
        };
        self.output_type_seen = true;
        Ok(())
    }

    fn set_like(&mut self, proto: Value) -> BuiltinResult<()> {
        if matches!(self.output_template, OutputTemplate::Like(_)) {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_OUTPUT_TYPE,
                "readmatrix: multiple 'Like' specifications are not supported",
            ));
        }
        if self.output_type_seen
            || !matches!(
                self.output_template,
                OutputTemplate::Numeric(NumericDType::F64)
            )
        {
            return Err(readmatrix_error_with(
                &READMATRIX_ERROR_OUTPUT_TYPE,
                "readmatrix: cannot combine 'Like' with OutputType overrides",
            ));
        }
        self.output_template = OutputTemplate::Like(proto);
        Ok(())
    }
}

#[derive(Clone, serde::Serialize)]
enum Delimiter {
    Char(char),
    String(String),
    Whitespace,
}

#[derive(Clone)]
enum OutputTemplate {
    Numeric(NumericDType),
    Logical,
    Like(Value),
}

mod admission;
mod column;
mod range;
use column::MatrixFieldColumn;
#[cfg(test)]
use range::parse_range_numeric;
use range::{apply_range, parse_range, RangeSpec};

mod output;
use output::finalize_output;
#[cfg(test)]
use output::{finalize_like, tensor_to_gpu};

mod reader;
use range::normalize_missing_token;
#[cfg(test)]
use reader::parse_integer_numeric_token;
use reader::read_numeric_matrix;
use reader::validate_encoding_label;

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => normalize_path(s),
        Value::CharArray(_) => {
            match crate::builtins::common::validation::value_text_scalar_string(value) {
                Some(text) => normalize_path(&text),
                None => Err(readmatrix_error_with(
                    &READMATRIX_ERROR_FILENAME,
                    "readmatrix: expected a 1-by-N character vector for the file name",
                )),
            }
        }
        Value::StringArray(_) => {
            match crate::builtins::common::validation::value_text_scalar_string(value) {
                Some(text) => normalize_path(&text),
                None => Err(readmatrix_error_with(
                    &READMATRIX_ERROR_FILENAME,
                    "readmatrix: string array inputs must be scalar",
                )),
            }
        }
        other => Err(readmatrix_error_with(
            &READMATRIX_ERROR_FILENAME,
            format!(
                "readmatrix: expected filename as string scalar or character vector, got {other:?}"
            ),
        )),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.is_empty() {
        return Err(readmatrix_error(&READMATRIX_ERROR_FILENAME_EMPTY));
    }
    Ok(Path::new(raw).to_path_buf())
}

#[cfg(test)]
#[path = "readmatrix/tests.rs"]
pub(crate) mod tests;
