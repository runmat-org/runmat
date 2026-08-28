use super::*;
use crate::builtins::common::tensor;
use runmat_value::NumericScalar;
#[derive(Clone)]
pub(in crate::builtins::table) struct ReadTableOptions {
    pub(super) file_type: ImportFileType,
    pub(super) delimiter: Option<Delimiter>,
    pub(super) read_variable_names: Option<bool>,
    pub(super) read_row_names: bool,
    pub(super) num_variables: Option<usize>,
    pub(super) variable_names: Option<Vec<String>>,
    pub(super) variable_types: Option<Vec<ImportVariableType>>,
    pub(super) row_names: Option<Vec<String>>,
    pub(super) num_header_lines: usize,
    pub(super) range: Option<RangeSpec>,
    pub(super) sheet: Option<SheetSelector>,
    pub(super) preserve_variable_names: bool,
    pub(super) treat_as_missing: HashSet<String>,
    pub(super) empty_line_rule: EmptyLineRule,
    pub(super) text_type: TextImportType,
    pub(super) encoding: String,
    pub(super) datetime_type: DatetimeImportType,
}

impl Default for ReadTableOptions {
    fn default() -> Self {
        Self {
            file_type: ImportFileType::Auto,
            delimiter: None,
            read_variable_names: None,
            read_row_names: false,
            num_variables: None,
            variable_names: None,
            variable_types: None,
            row_names: None,
            num_header_lines: 0,
            range: None,
            sheet: None,
            preserve_variable_names: false,
            treat_as_missing: HashSet::new(),
            empty_line_rule: EmptyLineRule::Skip,
            text_type: TextImportType::String,
            encoding: "utf-8".to_string(),
            datetime_type: DatetimeImportType::Datetime,
        }
    }
}

impl ReadTableOptions {
    pub(in crate::builtins::table) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut options = Self::default();
        let mut idx = 0usize;
        if let Some(Value::Struct(st)) = args.first() {
            for (name, value) in &st.fields {
                // Import-options objects use zero to mean that the variable count
                // should be inferred. A direct readtable name-value argument still
                // requires a positive count.
                if name.eq_ignore_ascii_case("NumVariables")
                    && matches!(value, Value::Num(value) if *value == 0.0)
                {
                    continue;
                }
                options.apply(name, value)?;
            }
            idx = 1;
        }
        while idx < args.len() {
            if idx + 1 >= args.len() {
                return Err(invalid_argument(
                    "readtable: name-value options must be provided in pairs",
                ));
            }
            let name = scalar_text(&args[idx], "readtable option")?;
            options.apply(&name, &args[idx + 1])?;
            idx += 2;
        }
        Ok(options)
    }

    fn apply(&mut self, name: &str, value: &Value) -> BuiltinResult<()> {
        if name.eq_ignore_ascii_case("FileType") {
            self.file_type = ImportFileType::parse(value)?;
        } else if name.eq_ignore_ascii_case("Delimiter") {
            self.delimiter = Some(Delimiter::parse(value)?);
        } else if name.eq_ignore_ascii_case("ReadVariableNames") {
            self.read_variable_names = Some(zero_one_bool_scalar(value, "ReadVariableNames")?);
        } else if name.eq_ignore_ascii_case("ReadRowNames") {
            self.read_row_names = zero_one_bool_scalar(value, "ReadRowNames")?;
        } else if name.eq_ignore_ascii_case("ExpectedNumVariables")
            || name.eq_ignore_ascii_case("NumVariables")
        {
            self.num_variables = Some(positive_usize(value, "ExpectedNumVariables")?);
        } else if name.eq_ignore_ascii_case("VariableNames") {
            self.variable_names = optional_raw_variable_name_list(value)?;
        } else if name.eq_ignore_ascii_case("VariableTypes") {
            self.variable_types = optional_variable_type_list(value)?;
        } else if name.eq_ignore_ascii_case("RowNames") {
            self.row_names = Some(string_list(value)?);
        } else if name.eq_ignore_ascii_case("NumHeaderLines") {
            self.num_header_lines = nonnegative_usize(value, "NumHeaderLines")?;
        } else if name.eq_ignore_ascii_case("VariableNamesLine") {
            let line = nonnegative_usize(value, "VariableNamesLine")?;
            self.read_variable_names = Some(line != 0);
            self.num_header_lines = line.saturating_sub(1);
        } else if name.eq_ignore_ascii_case("Range") {
            self.range = Some(RangeSpec::parse(value)?);
        } else if name.eq_ignore_ascii_case("DataRange") {
            self.range = if option_value_is_empty(value) {
                None
            } else {
                Some(RangeSpec::parse_data_range(value)?)
            };
        } else if name.eq_ignore_ascii_case("Sheet") {
            self.sheet = optional_sheet_selector(value)?;
        } else if name.eq_ignore_ascii_case("TreatAsMissing") {
            for token in string_list(value)? {
                self.treat_as_missing
                    .insert(token.trim().to_ascii_lowercase());
            }
        } else if name.eq_ignore_ascii_case("PreserveVariableNames") {
            self.preserve_variable_names = zero_one_bool_scalar(value, "PreserveVariableNames")?;
        } else if name.eq_ignore_ascii_case("VariableNamingRule") {
            let rule = scalar_text(value, "VariableNamingRule")?;
            if rule.eq_ignore_ascii_case("preserve") {
                self.preserve_variable_names = true;
            } else if rule.eq_ignore_ascii_case("modify") {
                self.preserve_variable_names = false;
            } else {
                return Err(invalid_argument(format!(
                    "readtable: unsupported VariableNamingRule '{rule}'"
                )));
            }
        } else if name.eq_ignore_ascii_case("EmptyLineRule") {
            let rule = scalar_text(value, "EmptyLineRule")?;
            self.empty_line_rule = if rule.eq_ignore_ascii_case("read") {
                EmptyLineRule::Read
            } else if rule.eq_ignore_ascii_case("skip") {
                EmptyLineRule::Skip
            } else {
                return Err(invalid_argument(format!(
                    "readtable: unsupported EmptyLineRule '{rule}'"
                )));
            };
        } else if name.eq_ignore_ascii_case("Encoding") {
            let encoding = scalar_text(value, "Encoding")?;
            validate_encoding_label(&encoding)?;
            self.encoding = encoding;
        } else if name.eq_ignore_ascii_case("TextType") {
            self.text_type = TextImportType::parse(value, "readtable")?;
        } else if name.eq_ignore_ascii_case("DatetimeType") {
            self.datetime_type = DatetimeImportType::parse(value)?;
        } else {
            return Err(invalid_argument(format!(
                "readtable: unsupported option '{name}'"
            )));
        }
        Ok(())
    }

    pub(super) fn is_missing(&self, token: &str) -> bool {
        let trimmed = token.trim();
        trimmed.is_empty()
            || self
                .treat_as_missing
                .contains(&trimmed.to_ascii_lowercase())
    }
}

pub(in crate::builtins::table) fn spreadsheet_import_options(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "spreadsheetImportOptions: name-value options must be provided in pairs",
        ));
    }
    let mut options = SpreadsheetImportOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "spreadsheetImportOptions option")?;
        options.apply(&name, &args[idx + 1])?;
        idx += 2;
    }
    Ok(Value::Struct(options.into_struct()?))
}

pub(in crate::builtins::table) async fn detect_import_options_from_file(
    path: &Path,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    match options.file_type {
        ImportFileType::Spreadsheet => detect_spreadsheet_import_options(path, options).await,
        ImportFileType::Text => detect_text_import_options(path, options).await,
        ImportFileType::Auto if is_spreadsheet_path(path) => {
            detect_spreadsheet_import_options(path, options).await
        }
        ImportFileType::Auto => detect_text_import_options(path, options).await,
    }
}

pub(in crate::builtins::table) async fn detect_text_import_options(
    path: &Path,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    if options.sheet.is_some() {
        return Err(invalid_argument(
            "detectImportOptions: Sheet is only valid for spreadsheet files",
        ));
    }
    let bytes = read_file_bytes(path).await?;
    let text = strip_utf8_bom(decode_text_bytes(&bytes, &options.encoding)?);
    let mut raw_lines = text.lines().map(ToString::to_string).collect::<Vec<_>>();
    if let Some(first) = raw_lines.first_mut() {
        if first.starts_with('\u{FEFF}') {
            *first = first.trim_start_matches('\u{FEFF}').to_string();
        }
    }
    let delimiter = options
        .delimiter
        .clone()
        .or_else(|| detect_delimiter(&raw_lines))
        .unwrap_or(Delimiter::Whitespace);
    let mut rows = parse_text_records(&text, &delimiter, options.empty_line_rule);
    if options.num_header_lines > 0 {
        rows = rows.into_iter().skip(options.num_header_lines).collect();
    }
    if let Some(range) = options.range {
        rows = apply_import_range(rows, range);
    }
    detected_options_from_rows(
        ImportFileType::Text,
        rows,
        options,
        Some(delimiter),
        options.sheet.as_ref(),
    )
}

pub(in crate::builtins::table) async fn detect_spreadsheet_import_options(
    path: &Path,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    if options.delimiter.is_some() {
        return Err(invalid_argument(
            "detectImportOptions: Delimiter is only valid for text files",
        ));
    }
    let bytes = read_file_bytes(path).await?;
    let cursor = Cursor::new(bytes);
    let mut workbook = open_workbook_auto_from_rs(cursor).map_err(|err| {
        table_error(
            &TABLE_ERROR_UNSUPPORTED_FILE,
            format!(
                "detectImportOptions: unable to open spreadsheet '{}': {err}",
                path.display()
            ),
        )
    })?;
    let range = match &options.sheet {
        Some(SheetSelector::Name(name)) => workbook.worksheet_range(name).map_err(|err| {
            invalid_argument(format!(
                "detectImportOptions: unable to read sheet '{name}': {err:?}"
            ))
        })?,
        Some(SheetSelector::Index(index)) => workbook
            .worksheet_range_at(*index)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "detectImportOptions: sheet index {} exceeds bounds",
                    index + 1
                ))
            })?
            .map_err(|err| {
                invalid_argument(format!(
                    "detectImportOptions: unable to read sheet {}: {err:?}",
                    index + 1
                ))
            })?,
        None => workbook
            .worksheet_range_at(0)
            .ok_or_else(|| {
                invalid_argument("detectImportOptions: spreadsheet contains no worksheets")
            })?
            .map_err(|err| {
                invalid_argument(format!(
                    "detectImportOptions: unable to read first sheet: {err:?}"
                ))
            })?,
    };
    let rows = spreadsheet_range_to_rows(&range, options)?;
    detected_options_from_rows(
        ImportFileType::Spreadsheet,
        rows,
        options,
        None,
        options.sheet.as_ref(),
    )
}

pub(in crate::builtins::table) fn detected_options_from_rows(
    file_type: ImportFileType,
    mut rows: Vec<Vec<ImportCell>>,
    options: &ReadTableOptions,
    delimiter: Option<Delimiter>,
    sheet: Option<&SheetSelector>,
) -> BuiltinResult<Value> {
    let mut variable_names = options.variable_names.clone();
    let read_variable_names = options
        .read_variable_names
        .unwrap_or_else(|| variable_names.is_none() && should_read_variable_names(&rows, options));
    let header_rows_consumed = usize::from(read_variable_names && variable_names.is_none());
    if header_rows_consumed > 0 && !rows.is_empty() {
        variable_names = Some(
            rows.remove(0)
                .into_iter()
                .map(|cell| cell.display_text())
                .collect(),
        );
    }

    let mut data_rows = rows;
    let mut data_variable_names = variable_names.clone();
    let row_name_header = if options.read_row_names {
        for row in &mut data_rows {
            if !row.is_empty() {
                row.remove(0);
            }
        }
        let mut header = None;
        if let Some(names) = data_variable_names.as_mut() {
            if !names.is_empty() {
                header = Some(names.remove(0));
            }
        }
        Some(
            header
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| "Row".to_string()),
        )
    } else {
        None
    };

    let column_count = import_column_count(&data_rows, &data_variable_names, options)?;
    let data_names = import_variable_names(data_variable_names, column_count, options);
    let names = if let Some(row_name_header) = row_name_header {
        let mut names = Vec::with_capacity(data_names.len() + 1);
        names.push(row_name_header);
        names.extend(data_names);
        names
    } else {
        data_names
    };
    let types = detected_variable_type_labels(&data_rows, options, column_count)?;
    let output_num_header_lines = detected_output_header_lines(options, header_rows_consumed);
    let output_range = detected_output_range(options.range, header_rows_consumed);

    let mut out = StructValue::new();
    out.insert("FileType", Value::String(import_file_type_label(file_type)));
    if let Some(delimiter) = delimiter {
        out.insert("Delimiter", Value::String(delimiter_label(&delimiter)));
    }
    out.insert("NumHeaderLines", Value::Num(output_num_header_lines as f64));
    out.insert("ReadVariableNames", Value::Bool(false));
    out.insert("ReadRowNames", Value::Bool(options.read_row_names));
    out.insert("NumVariables", Value::Num(column_count as f64));
    out.insert(
        "VariableNames",
        string_array_value(names, "detectImportOptions")?,
    );
    out.insert(
        "VariableTypes",
        string_array_value(types, "detectImportOptions")?,
    );
    if let Some(range) = output_range {
        out.insert("Range", range_spec_value(range)?);
        out.insert("DataRange", range_spec_value(range)?);
    }
    if let Some(sheet) = sheet {
        out.insert("Sheet", sheet_value(sheet));
    }
    let mut treat_as_missing = options.treat_as_missing.iter().cloned().collect::<Vec<_>>();
    treat_as_missing.sort();
    out.insert(
        "TreatAsMissing",
        string_array_value(treat_as_missing, "detectImportOptions")?,
    );
    out.insert(
        "PreserveVariableNames",
        Value::Bool(options.preserve_variable_names),
    );
    out.insert(
        "VariableNamingRule",
        Value::String(if options.preserve_variable_names {
            "preserve".to_string()
        } else {
            "modify".to_string()
        }),
    );
    out.insert(
        "EmptyLineRule",
        Value::String(
            match options.empty_line_rule {
                EmptyLineRule::Skip => "skip",
                EmptyLineRule::Read => "read",
            }
            .to_string(),
        ),
    );
    out.insert(
        "TextType",
        Value::String(
            match options.text_type {
                TextImportType::String => "string",
                TextImportType::Char => "char",
            }
            .to_string(),
        ),
    );
    out.insert(
        "DatetimeType",
        Value::String(
            match options.datetime_type {
                DatetimeImportType::Datetime => "datetime",
                DatetimeImportType::Text => "text",
                DatetimeImportType::ExcelDatenum => "exceldatenum",
            }
            .to_string(),
        ),
    );
    out.insert("Encoding", Value::String(options.encoding.clone()));
    Ok(Value::Struct(out))
}

pub(in crate::builtins::table) fn detected_variable_type_labels(
    rows: &[Vec<ImportCell>],
    options: &ReadTableOptions,
    column_count: usize,
) -> BuiltinResult<Vec<String>> {
    if let Some(requested) = &options.variable_types {
        let mut labels = requested
            .iter()
            .map(import_variable_type_label)
            .collect::<Vec<_>>();
        while labels.len() < column_count {
            labels.push("auto".to_string());
        }
        labels.truncate(column_count);
        return Ok(labels);
    }
    Ok((0..column_count)
        .map(|col| {
            let values = rows
                .iter()
                .map(|row| row.get(col).cloned().unwrap_or(ImportCell::Empty))
                .collect::<Vec<_>>();
            infer_import_type_label(&values, options)
        })
        .collect())
}

pub(in crate::builtins::table) fn infer_import_type_label(
    values: &[ImportCell],
    options: &ReadTableOptions,
) -> String {
    if values
        .iter()
        .all(|value| is_detected_numeric(value, options))
    {
        return "double".to_string();
    }
    if values
        .iter()
        .all(|value| is_detected_logical(value, options))
    {
        return "logical".to_string();
    }
    if !matches!(options.datetime_type, DatetimeImportType::Text)
        && values
            .iter()
            .all(|value| is_detected_datetime(value, options))
    {
        return "datetime".to_string();
    }
    match options.text_type {
        TextImportType::String => "string".to_string(),
        TextImportType::Char => "char".to_string(),
    }
}

pub(in crate::builtins::table) fn is_detected_numeric(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> bool {
    match value {
        ImportCell::Empty | ImportCell::Number(_) => true,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            options.is_missing(token) || parse_numeric(token).is_some()
        }
        _ => false,
    }
}

pub(in crate::builtins::table) fn is_detected_logical(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> bool {
    match value {
        ImportCell::Empty | ImportCell::Logical(_) => true,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            options.is_missing(token) || parse_logical(token).is_some()
        }
        _ => false,
    }
}

pub(in crate::builtins::table) fn is_detected_datetime(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> bool {
    match value {
        ImportCell::Empty | ImportCell::DateTime(_) => true,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            options.is_missing(token) || parse_iso_datetime_to_datenum(token).is_some()
        }
        _ => false,
    }
}

pub(in crate::builtins::table) fn import_variable_type_label(kind: &ImportVariableType) -> String {
    match kind {
        ImportVariableType::Auto => "auto",
        ImportVariableType::Numeric(NumericDType::F64) => "double",
        ImportVariableType::Numeric(NumericDType::F32) => "single",
        ImportVariableType::Numeric(NumericDType::I8) => "int8",
        ImportVariableType::Numeric(NumericDType::I16) => "int16",
        ImportVariableType::Numeric(NumericDType::I32) => "int32",
        ImportVariableType::Numeric(NumericDType::I64) => "int64",
        ImportVariableType::Numeric(NumericDType::U8) => "uint8",
        ImportVariableType::Numeric(NumericDType::U16) => "uint16",
        ImportVariableType::Numeric(NumericDType::U32) => "uint32",
        ImportVariableType::Numeric(NumericDType::U64) => "uint64",
        ImportVariableType::Integer(target) => target.class_name(),
        ImportVariableType::Logical => "logical",
        ImportVariableType::Text(TextImportType::String) => "string",
        ImportVariableType::Text(TextImportType::Char) => "char",
        ImportVariableType::CellStr => "cellstr",
        ImportVariableType::Categorical => "categorical",
        ImportVariableType::Datetime => "datetime",
        ImportVariableType::Duration => "duration",
    }
    .to_string()
}

pub(in crate::builtins::table) fn detected_output_header_lines(
    options: &ReadTableOptions,
    header_rows_consumed: usize,
) -> usize {
    if options.range.is_some() {
        options.num_header_lines
    } else {
        options.num_header_lines + header_rows_consumed
    }
}

pub(in crate::builtins::table) fn detected_output_range(
    range: Option<RangeSpec>,
    header_rows_consumed: usize,
) -> Option<RangeSpec> {
    range.map(|mut range| {
        range.start_row = range.start_row.saturating_add(header_rows_consumed);
        range
    })
}

pub(in crate::builtins::table) fn import_file_type_label(file_type: ImportFileType) -> String {
    match file_type {
        ImportFileType::Text | ImportFileType::Auto => "text",
        ImportFileType::Spreadsheet => "spreadsheet",
    }
    .to_string()
}

pub(in crate::builtins::table) fn delimiter_label(delimiter: &Delimiter) -> String {
    match delimiter {
        Delimiter::Char('\t') => "\t".to_string(),
        Delimiter::Char(ch) => ch.to_string(),
        Delimiter::String(text) => text.clone(),
        Delimiter::Whitespace => "whitespace".to_string(),
    }
}

pub(in crate::builtins::table) fn sheet_value(sheet: &SheetSelector) -> Value {
    match sheet {
        SheetSelector::Name(name) => Value::String(name.clone()),
        SheetSelector::Index(index) => Value::Num((*index + 1) as f64),
    }
}

pub(in crate::builtins::table) fn range_spec_value(range: RangeSpec) -> BuiltinResult<Value> {
    Ok(Value::String(range_spec_text(range)))
}

pub(in crate::builtins::table) fn range_spec_text(range: RangeSpec) -> String {
    let has_end = range.end_row.is_some() || range.end_col.is_some();
    let include_start_col = range.start_col > 0 || range.end_col.is_some() || !has_end;
    let include_start_row = range.start_row > 0 || range.end_row.is_some() || !has_end;
    let start = range_ref_text(
        range.start_row,
        range.start_col,
        include_start_row,
        include_start_col,
    );
    if !has_end {
        return start;
    }

    let end = range_ref_text(
        range.end_row.unwrap_or(0),
        range.end_col.unwrap_or(0),
        range.end_row.is_some(),
        range.end_col.is_some(),
    );
    format!("{start}:{end}")
}

pub(in crate::builtins::table) fn range_ref_text(
    row: usize,
    col: usize,
    include_row: bool,
    include_col: bool,
) -> String {
    let mut out = String::new();
    if include_col {
        out.push_str(&spreadsheet_column_label(col));
    }
    if include_row {
        out.push_str(&(row + 1).to_string());
    }
    out
}

pub(in crate::builtins::table) fn spreadsheet_column_label(mut col: usize) -> String {
    let mut chars = Vec::new();
    loop {
        let rem = col % 26;
        chars.push((b'A' + rem as u8) as char);
        if col < 26 {
            break;
        }
        col = col / 26 - 1;
    }
    chars.iter().rev().collect()
}

pub(in crate::builtins::table) fn string_array_value(
    values: Vec<String>,
    context: &str,
) -> BuiltinResult<Value> {
    let len = values.len();
    StringArray::new(values, vec![1, len])
        .map(Value::StringArray)
        .map_err(|err| invalid_variable(format!("{context}: {err}")))
}

#[derive(Clone)]
pub(in crate::builtins::table) struct SpreadsheetImportOptions {
    num_variables: usize,
    read_variable_names: Option<bool>,
    read_row_names: bool,
    variable_names: Vec<String>,
    variable_types: Vec<String>,
    data_range: Option<Value>,
    sheet: Option<Value>,
    treat_as_missing: Vec<String>,
    preserve_variable_names: bool,
    empty_line_rule: String,
    text_type: String,
    datetime_type: String,
}

impl Default for SpreadsheetImportOptions {
    fn default() -> Self {
        let num_variables = 0;
        Self {
            num_variables,
            read_variable_names: None,
            read_row_names: false,
            variable_names: Vec::new(),
            variable_types: Vec::new(),
            data_range: None,
            sheet: None,
            treat_as_missing: Vec::new(),
            preserve_variable_names: false,
            empty_line_rule: "skip".to_string(),
            text_type: "string".to_string(),
            datetime_type: "datetime".to_string(),
        }
    }
}

impl SpreadsheetImportOptions {
    fn apply(&mut self, name: &str, value: &Value) -> BuiltinResult<()> {
        if name.eq_ignore_ascii_case("NumVariables") {
            self.resize_variables(positive_usize(value, "NumVariables")?);
        } else if name.eq_ignore_ascii_case("VariableNames") {
            self.variable_names = raw_variable_name_list(value)?;
            self.align_variable_metadata_count(self.variable_names.len(), "VariableNames")?;
            self.ensure_variable_metadata_len();
        } else if name.eq_ignore_ascii_case("VariableTypes") {
            let types = variable_type_names(value)?;
            self.variable_types = types;
            self.align_variable_metadata_count(self.variable_types.len(), "VariableTypes")?;
            self.ensure_variable_metadata_len();
        } else if name.eq_ignore_ascii_case("DataRange") || name.eq_ignore_ascii_case("Range") {
            self.data_range = if option_value_is_empty(value) {
                None
            } else {
                RangeSpec::parse(value)?;
                Some(value.clone())
            };
        } else if name.eq_ignore_ascii_case("Sheet") {
            self.sheet = if option_value_is_empty(value) {
                None
            } else {
                SheetSelector::parse(value)?;
                Some(value.clone())
            };
        } else if name.eq_ignore_ascii_case("ReadVariableNames") {
            self.read_variable_names = Some(bool_scalar(value, "ReadVariableNames")?);
        } else if name.eq_ignore_ascii_case("ReadRowNames") {
            self.read_row_names = bool_scalar(value, "ReadRowNames")?;
        } else if name.eq_ignore_ascii_case("TreatAsMissing") {
            self.treat_as_missing = string_list(value)?;
        } else if name.eq_ignore_ascii_case("PreserveVariableNames") {
            self.preserve_variable_names = bool_scalar(value, "PreserveVariableNames")?;
        } else if name.eq_ignore_ascii_case("VariableNamingRule") {
            let rule = scalar_text(value, "VariableNamingRule")?;
            if rule.eq_ignore_ascii_case("preserve") {
                self.preserve_variable_names = true;
            } else if rule.eq_ignore_ascii_case("modify") {
                self.preserve_variable_names = false;
            } else {
                return Err(invalid_argument(format!(
                    "spreadsheetImportOptions: unsupported VariableNamingRule '{rule}'"
                )));
            }
        } else if name.eq_ignore_ascii_case("EmptyLineRule") {
            let rule = scalar_text(value, "EmptyLineRule")?;
            if !(rule.eq_ignore_ascii_case("read") || rule.eq_ignore_ascii_case("skip")) {
                return Err(invalid_argument(format!(
                    "spreadsheetImportOptions: unsupported EmptyLineRule '{rule}'"
                )));
            }
            self.empty_line_rule = rule.to_ascii_lowercase();
        } else if name.eq_ignore_ascii_case("TextType") {
            let text_type = scalar_text(value, "TextType")?;
            if !(text_type.eq_ignore_ascii_case("string") || text_type.eq_ignore_ascii_case("char"))
            {
                return Err(invalid_argument(format!(
                    "spreadsheetImportOptions: unsupported TextType '{text_type}'"
                )));
            }
            self.text_type = text_type.to_ascii_lowercase();
        } else if name.eq_ignore_ascii_case("DatetimeType") {
            let datetime_type = scalar_text(value, "DatetimeType")?;
            if !(datetime_type.eq_ignore_ascii_case("datetime")
                || datetime_type.eq_ignore_ascii_case("text")
                || datetime_type.eq_ignore_ascii_case("exceldatenum"))
            {
                return Err(invalid_argument(format!(
                    "spreadsheetImportOptions: unsupported DatetimeType '{datetime_type}'"
                )));
            }
            self.datetime_type = datetime_type.to_ascii_lowercase();
        } else {
            return Err(invalid_argument(format!(
                "spreadsheetImportOptions: unsupported option '{name}'"
            )));
        }
        Ok(())
    }

    fn resize_variables(&mut self, num_variables: usize) {
        self.num_variables = num_variables;
        if self.variable_names.len() > num_variables {
            self.variable_names.truncate(num_variables);
        }
        if self.variable_types.len() > num_variables {
            self.variable_types.truncate(num_variables);
        }
        self.ensure_variable_metadata_len();
    }

    fn align_variable_metadata_count(&mut self, len: usize, field: &str) -> BuiltinResult<()> {
        if self.num_variables == 0 {
            self.num_variables = len;
            return Ok(());
        }
        if len > self.num_variables {
            return Err(invalid_argument(format!(
                "spreadsheetImportOptions: {field} length exceeds NumVariables"
            )));
        }
        Ok(())
    }

    fn ensure_variable_metadata_len(&mut self) {
        if self.num_variables == 0 {
            return;
        }
        while self.variable_names.len() < self.num_variables {
            self.variable_names
                .push(format!("Var{}", self.variable_names.len() + 1));
        }
        self.variable_names.truncate(self.num_variables);
        while self.variable_types.len() < self.num_variables {
            self.variable_types.push("auto".to_string());
        }
        self.variable_types.truncate(self.num_variables);
    }

    fn into_struct(mut self) -> BuiltinResult<StructValue> {
        self.ensure_variable_metadata_len();
        let mut out = StructValue::new();
        out.insert("FileType", Value::String("spreadsheet".to_string()));
        out.insert("NumVariables", Value::Num(self.num_variables as f64));
        if let Some(read_variable_names) = self.read_variable_names {
            out.insert("ReadVariableNames", Value::Bool(read_variable_names));
        }
        out.insert("ReadRowNames", Value::Bool(self.read_row_names));
        out.insert(
            "VariableNames",
            Value::StringArray(
                StringArray::new(
                    self.variable_names.clone(),
                    vec![1, self.variable_names.len()],
                )
                .map_err(|err| invalid_variable(format!("spreadsheetImportOptions: {err}")))?,
            ),
        );
        out.insert(
            "VariableTypes",
            Value::StringArray(
                StringArray::new(
                    self.variable_types.clone(),
                    vec![1, self.variable_types.len()],
                )
                .map_err(|err| invalid_variable(format!("spreadsheetImportOptions: {err}")))?,
            ),
        );
        out.insert(
            "DataRange",
            self.data_range
                .unwrap_or_else(|| Value::String(String::new())),
        );
        out.insert(
            "Sheet",
            self.sheet.unwrap_or_else(|| Value::String(String::new())),
        );
        out.insert(
            "TreatAsMissing",
            Value::StringArray(
                StringArray::new(
                    self.treat_as_missing.clone(),
                    vec![1, self.treat_as_missing.len()],
                )
                .map_err(|err| invalid_variable(format!("spreadsheetImportOptions: {err}")))?,
            ),
        );
        out.insert(
            "PreserveVariableNames",
            Value::Bool(self.preserve_variable_names),
        );
        out.insert(
            "VariableNamingRule",
            Value::String(if self.preserve_variable_names {
                "preserve".to_string()
            } else {
                "modify".to_string()
            }),
        );
        out.insert("EmptyLineRule", Value::String(self.empty_line_rule));
        out.insert("TextType", Value::String(self.text_type));
        out.insert("DatetimeType", Value::String(self.datetime_type));
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::table) enum ImportVariableType {
    Auto,
    Numeric(NumericDType),
    Integer(crate::builtins::math::elementwise::integer_cast::IntegerTarget),
    Logical,
    Text(TextImportType),
    CellStr,
    Categorical,
    Datetime,
    Duration,
}

impl ImportVariableType {
    pub(in crate::builtins::table) fn parse(raw: &str) -> BuiltinResult<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "double" => Ok(Self::Numeric(NumericDType::F64)),
            "single" => Ok(Self::Numeric(NumericDType::F32)),
            "int8" => Ok(Self::Integer(
                crate::builtins::math::elementwise::integer_cast::IntegerTarget::I8,
            )),
            "int16" => Ok(Self::Integer(
                crate::builtins::math::elementwise::integer_cast::IntegerTarget::I16,
            )),
            "int32" => Ok(Self::Integer(
                crate::builtins::math::elementwise::integer_cast::IntegerTarget::I32,
            )),
            "int64" => Ok(Self::Integer(
                crate::builtins::math::elementwise::integer_cast::IntegerTarget::I64,
            )),
            "uint8" => Ok(Self::Integer(
                crate::builtins::math::elementwise::integer_cast::IntegerTarget::U8,
            )),
            "uint16" => Ok(Self::Integer(
                crate::builtins::math::elementwise::integer_cast::IntegerTarget::U16,
            )),
            "uint32" => Ok(Self::Integer(
                crate::builtins::math::elementwise::integer_cast::IntegerTarget::U32,
            )),
            "uint64" => Ok(Self::Integer(
                crate::builtins::math::elementwise::integer_cast::IntegerTarget::U64,
            )),
            "logical" | "bool" | "boolean" => Ok(Self::Logical),
            "string" => Ok(Self::Text(TextImportType::String)),
            "char" => Ok(Self::Text(TextImportType::Char)),
            "cellstr" => Ok(Self::CellStr),
            "categorical" => Ok(Self::Categorical),
            "datetime" => Ok(Self::Datetime),
            "duration" => Ok(Self::Duration),
            other => Err(invalid_argument(format!(
                "readtable: unsupported VariableTypes entry '{other}'"
            ))),
        }
    }

    pub(in crate::builtins::table) fn canonical_label(raw: &str) -> BuiltinResult<String> {
        Self::parse(raw)?;
        let label = raw.trim().to_ascii_lowercase();
        Ok(if label.is_empty() {
            "auto".to_string()
        } else {
            label
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::table) enum TextImportType {
    String,
    Char,
}

impl TextImportType {
    pub(in crate::builtins::table) fn parse(value: &Value, context: &str) -> BuiltinResult<Self> {
        let text_type = scalar_text(value, "TextType")?;
        match text_type.trim().to_ascii_lowercase().as_str() {
            "string" => Ok(Self::String),
            "char" => Ok(Self::Char),
            other => Err(invalid_argument(format!(
                "{context}: unsupported TextType '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::builtins::table) enum EmptyLineRule {
    Skip,
    Read,
}

#[derive(Clone, Copy)]
pub(in crate::builtins::table) enum DatetimeImportType {
    Datetime,
    Text,
    ExcelDatenum,
}

impl DatetimeImportType {
    pub(in crate::builtins::table) fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "DatetimeType")?;
        match text.trim().to_ascii_lowercase().as_str() {
            "datetime" => Ok(Self::Datetime),
            "text" => Ok(Self::Text),
            "exceldatenum" => Ok(Self::ExcelDatenum),
            other => Err(invalid_argument(format!(
                "readtable: unsupported DatetimeType '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::table) enum ImportFileType {
    Auto,
    Text,
    Spreadsheet,
}

impl ImportFileType {
    pub(in crate::builtins::table) fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "FileType")?;
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "text" | "delimitedtext" | "delimited" => Ok(Self::Text),
            "spreadsheet" | "excel" => Ok(Self::Spreadsheet),
            other => Err(invalid_argument(format!(
                "readtable: unsupported FileType '{other}'"
            ))),
        }
    }
}

#[derive(Clone)]
pub(in crate::builtins::table) enum SheetSelector {
    Name(String),
    Index(usize),
}

impl SheetSelector {
    pub(in crate::builtins::table) fn parse(value: &Value) -> BuiltinResult<Self> {
        if let Some(integer) = tensor::scalar_integer_value(value) {
            return integer
                .try_to_usize()
                .and_then(|index| index.checked_sub(1))
                .map(Self::Index)
                .ok_or_else(|| invalid_argument("readtable: Sheet must be one-based"));
        }

        match value {
            Value::Num(n)
                if n.is_finite() && *n >= 1.0 && (n.round() - n).abs() <= f64::EPSILON =>
            {
                let rounded = n.round();
                if rounded > usize::MAX.saturating_sub(1) as f64 {
                    return Err(invalid_argument("readtable: Sheet index is too large"));
                }
                let parsed = rounded as usize;
                if parsed as f64 != rounded || parsed == usize::MAX {
                    return Err(invalid_argument("readtable: Sheet index is too large"));
                }
                parsed
                    .checked_sub(1)
                    .map(Self::Index)
                    .ok_or_else(|| invalid_argument("readtable: Sheet must be one-based"))
            }
            _ => {
                let text = scalar_text(value, "Sheet")?;
                if text.trim().is_empty() {
                    return Err(invalid_argument("readtable: Sheet must not be empty"));
                }
                Ok(Self::Name(text))
            }
        }
    }
}

#[derive(Clone)]
pub(in crate::builtins::table) enum Delimiter {
    Char(char),
    String(String),
    Whitespace,
}

impl Delimiter {
    pub(in crate::builtins::table) fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "Delimiter")?;
        if text.is_empty() {
            return Err(invalid_argument("readtable: Delimiter must not be empty"));
        }
        match text.trim().to_ascii_lowercase().as_str() {
            "tab" => Ok(Self::Char('\t')),
            "space" | "whitespace" => Ok(Self::Whitespace),
            "comma" => Ok(Self::Char(',')),
            "semicolon" => Ok(Self::Char(';')),
            "bar" | "pipe" => Ok(Self::Char('|')),
            _ if text.chars().count() == 1 => Ok(Self::Char(text.chars().next().unwrap())),
            _ => Ok(Self::String(text)),
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::builtins::table) struct RangeSpec {
    pub(super) start_row: usize,
    pub(super) start_col: usize,
    pub(super) end_row: Option<usize>,
    pub(super) end_col: Option<usize>,
}

impl RangeSpec {
    pub(in crate::builtins::table) fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Self::parse_text(text),
            Value::CharArray(ca) if ca.rows == 1 => {
                let text: String = ca.data.iter().collect();
                Self::parse_text(&text)
            }
            Value::StringArray(sa) if sa.data.len() == 1 => Self::parse_text(&sa.data[0]),
            Value::Tensor(t) if tensor_len(t) == 2 || tensor_len(t) == 4 => {
                let len = tensor_len(t);
                let mut indices = Vec::with_capacity(len);
                for idx in 0..len {
                    let value = t
                        .numeric_value_at(idx)
                        .ok_or_else(|| invalid_index("table: Range index out of bounds"))?;
                    indices.push(match value {
                        NumericScalar::F64(value) => one_based_to_zero(value, usize::MAX, "Range")?,
                        NumericScalar::F32(value) => {
                            one_based_to_zero(f64::from(value), usize::MAX, "Range")?
                        }
                        value => one_based_integer_to_zero(
                            &value
                                .into_int_value()
                                .expect("non-floating numeric scalar is integer"),
                            "Range",
                        )?,
                    });
                }
                Ok(Self {
                    start_row: indices[0],
                    start_col: indices[1],
                    end_row: indices.get(2).copied(),
                    end_col: indices.get(3).copied(),
                })
            }
            _ => Err(invalid_argument(
                "readtable: Range must be a cell reference string or numeric vector",
            )),
        }
    }

    pub(in crate::builtins::table) fn parse_text(text: &str) -> BuiltinResult<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(invalid_argument("readtable: Range must not be empty"));
        }
        let parts: Vec<&str> = trimmed.split(':').collect();
        if parts.len() > 2 {
            return Err(invalid_argument(format!(
                "readtable: invalid Range specification '{trimmed}'"
            )));
        }
        let start = parse_cell_ref(parts[0])?;
        let end = if parts.len() == 2 {
            Some(parse_cell_ref(parts[1])?)
        } else {
            None
        };
        Ok(Self {
            start_row: start.0.unwrap_or(0),
            start_col: start.1.unwrap_or(0),
            end_row: end.and_then(|item| item.0),
            end_col: end.and_then(|item| item.1),
        })
    }

    pub(in crate::builtins::table) fn parse_data_range(value: &Value) -> BuiltinResult<Self> {
        if matches!(
            value,
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_)
        ) {
            return Self::parse(value);
        }
        if let Some(integer) = tensor::scalar_integer_value(value) {
            return Ok(Self {
                start_row: one_based_integer_to_zero(&integer, "DataRange")?,
                start_col: 0,
                end_row: None,
                end_col: None,
            });
        }
        if let Value::Num(value) = value {
            return Ok(Self {
                start_row: one_based_to_zero(*value, usize::MAX, "DataRange")?,
                start_col: 0,
                end_row: None,
                end_col: None,
            });
        }
        let Value::Tensor(values) = value else {
            return Err(invalid_argument(
                "readtable: DataRange must be a positive row or an Nx2 row-interval matrix",
            ));
        };
        if crate::builtins::common::tensor::is_scalar_tensor(values) {
            let value = values
                .numeric_value_at(0)
                .ok_or_else(|| invalid_argument("readtable: invalid DataRange row"))?;
            let start_row = match value {
                NumericScalar::F64(value) => one_based_to_zero(value, usize::MAX, "DataRange")?,
                NumericScalar::F32(value) => {
                    one_based_to_zero(f64::from(value), usize::MAX, "DataRange")?
                }
                value => one_based_integer_to_zero(
                    &value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer"),
                    "DataRange",
                )?,
            };
            return Ok(Self {
                start_row,
                start_col: 0,
                end_row: None,
                end_col: None,
            });
        }
        if values.shape.as_slice() != [1, 2] {
            return Err(invalid_argument(
                "readtable: DataRange supports one 1-by-2 row interval; multiple Nx2 intervals are not yet supported",
            ));
        }
        let mut rows = Vec::with_capacity(2);
        for index in 0..2 {
            let value = values
                .numeric_value_at(index)
                .ok_or_else(|| invalid_argument("readtable: invalid DataRange row"))?;
            rows.push(match value {
                NumericScalar::F64(value) => one_based_to_zero(value, usize::MAX, "DataRange")?,
                NumericScalar::F32(value) => {
                    one_based_to_zero(f64::from(value), usize::MAX, "DataRange")?
                }
                value => one_based_integer_to_zero(
                    &value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer"),
                    "DataRange",
                )?,
            });
        }
        if rows[1] < rows[0] {
            return Err(invalid_argument(
                "readtable: DataRange row interval must be increasing",
            ));
        }
        Ok(Self {
            start_row: rows[0],
            start_col: 0,
            end_row: Some(rows[1]),
            end_col: None,
        })
    }
}

fn tensor_len(tensor: &runmat_value::Tensor) -> usize {
    tensor.len()
}

fn one_based_integer_to_zero(
    value: &runmat_value::IntValue,
    context: &str,
) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            invalid_index(format!(
                "table: {context} indices must be positive finite integers"
            ))
        })
}

pub(in crate::builtins::table) fn parse_cell_ref(
    token: &str,
) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    let mut letters = String::new();
    let mut digits = String::new();
    for ch in token.trim().chars() {
        if ch == '$' {
            continue;
        }
        if ch.is_ascii_alphabetic() {
            letters.push(ch.to_ascii_uppercase());
        } else if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            return Err(invalid_argument(format!(
                "readtable: invalid Range component '{token}'"
            )));
        }
    }
    let col = if letters.is_empty() {
        None
    } else {
        let mut value = 0usize;
        for ch in letters.chars() {
            value = value
                .checked_mul(26)
                .and_then(|v| v.checked_add((ch as u8 - b'A' + 1) as usize))
                .ok_or_else(|| invalid_argument("readtable: Range column overflow"))?;
        }
        Some(value - 1)
    };
    let row = if digits.is_empty() {
        None
    } else {
        let parsed = digits
            .parse::<usize>()
            .map_err(|_| invalid_argument("readtable: invalid Range row"))?;
        if parsed == 0 {
            return Err(invalid_argument("readtable: Range rows are one-based"));
        }
        Some(parsed - 1)
    };
    Ok((row, col))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

    fn integer_storages(values: &[u64]) -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(values.iter().map(|&value| value as i8).collect()),
            IntegerStorage::I16(values.iter().map(|&value| value as i16).collect()),
            IntegerStorage::I32(values.iter().map(|&value| value as i32).collect()),
            IntegerStorage::I64(values.iter().map(|&value| value as i64).collect()),
            IntegerStorage::U8(values.iter().map(|&value| value as u8).collect()),
            IntegerStorage::U16(values.iter().map(|&value| value as u16).collect()),
            IntegerStorage::U32(values.iter().map(|&value| value as u32).collect()),
            IntegerStorage::U64(values.to_vec()),
        ]
    }

    #[test]
    fn range_spec_reads_typed_integer_storage_exactly() {
        let cases = [
            IntegerStorage::I8(vec![2, 3, 4, 5]),
            IntegerStorage::I16(vec![2, 3, 4, 5]),
            IntegerStorage::I32(vec![2, 3, 4, 5]),
            IntegerStorage::I64(vec![2, 3, 4, 5]),
            IntegerStorage::U8(vec![2, 3, 4, 5]),
            IntegerStorage::U16(vec![2, 3, 4, 5]),
            IntegerStorage::U32(vec![2, 3, 4, 5]),
            IntegerStorage::U64(vec![2, 3, 4, 5]),
        ];

        for storage in cases {
            let range = Tensor::new_integer(storage, vec![1, 4]).expect("range");
            let parsed = RangeSpec::parse(&Value::Tensor(range)).expect("typed range");
            assert_eq!(parsed.start_row, 1);
            assert_eq!(parsed.start_col, 2);
            assert_eq!(parsed.end_row, Some(3));
            assert_eq!(parsed.end_col, Some(4));
        }
    }

    #[test]
    fn range_and_sheet_parsers_ignore_poisoned_mirrors_for_every_integer_class() {
        for storage in integer_storages(&[2, 3]) {
            let range = Tensor::new_integer(storage, vec![1, 2]).unwrap();
            let parsed = RangeSpec::parse(&Value::Tensor(range)).unwrap();
            assert_eq!((parsed.start_row, parsed.start_col), (1, 2));
        }
        for storage in integer_storages(&[2]) {
            let sheet = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            assert!(matches!(
                SheetSelector::parse(&Value::Tensor(sheet)).unwrap(),
                SheetSelector::Index(1)
            ));
        }
    }

    #[test]
    fn range_spec_rejects_nonpositive_typed_integer_indices() {
        let range = Tensor::new_integer(IntegerStorage::I16(vec![1, 0]), vec![1, 2]).unwrap();

        assert!(RangeSpec::parse(&Value::Tensor(range)).is_err());
    }

    #[test]
    fn sheet_selector_reads_typed_integer_storage_exactly() {
        let sheet = Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).unwrap();

        match SheetSelector::parse(&Value::Tensor(sheet)).unwrap() {
            SheetSelector::Index(index) => assert_eq!(index, 2),
            SheetSelector::Name(name) => panic!("expected sheet index, got {name}"),
        }
    }

    #[test]
    fn sheet_selector_rejects_invalid_integer_and_double_values() {
        let zero = Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).unwrap();
        assert!(SheetSelector::parse(&Value::Tensor(zero)).is_err());
        assert!(SheetSelector::parse(&Value::Num(1.0e300)).is_err());
    }

    #[test]
    fn data_range_uses_documented_row_grammar_and_exact_integer_storage() {
        for storage in integer_storages(&[2, 5]) {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
            let range = RangeSpec::parse_data_range(&value).unwrap();
            assert_eq!(range.start_row, 1);
            assert_eq!(range.start_col, 0);
            assert_eq!(range.end_row, Some(4));
            assert_eq!(range.end_col, None);
        }
        let scalar = Value::Int(runmat_value::IntValue::U64(3));
        let range = RangeSpec::parse_data_range(&scalar).unwrap();
        assert_eq!(range.start_row, 2);
        assert_eq!(range.end_row, None);
        let disjoint = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 4, 5]), vec![2, 2]).unwrap(),
        );
        assert!(RangeSpec::parse_data_range(&disjoint).is_err());
        let column =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap());
        assert!(RangeSpec::parse_data_range(&column).is_err());
        let floating_scalar = Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap());
        let range = RangeSpec::parse_data_range(&floating_scalar).unwrap();
        assert_eq!(range.start_row, 2);
        assert_eq!(range.end_row, None);
        let textual = RangeSpec::parse_data_range(&Value::from("A2:B5")).unwrap();
        assert_eq!(textual.start_row, 1);
        assert_eq!(textual.start_col, 0);
        assert_eq!(textual.end_row, Some(4));
        assert_eq!(textual.end_col, Some(1));
    }

    #[test]
    fn expected_num_variables_is_positive_for_every_integer_class() {
        for storage in integer_storages(&[2]) {
            let mut options = ReadTableOptions::default();
            options
                .apply(
                    "ExpectedNumVariables",
                    &Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap()),
                )
                .unwrap();
            assert_eq!(options.num_variables, Some(2));
        }
        let mut options = ReadTableOptions::default();
        assert!(options
            .apply(
                "ExpectedNumVariables",
                &Value::Int(runmat_value::IntValue::U8(0)),
            )
            .is_err());
    }

    #[test]
    fn detect_boolean_controls_require_exact_zero_or_one() {
        for storage in integer_storages(&[1]) {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
            assert!(zero_one_bool_scalar(&value, "ReadVariableNames").unwrap());
        }
        assert!(zero_one_bool_scalar(&Value::Int(runmat_value::IntValue::I8(-1)), "flag").is_err());
        assert!(zero_one_bool_scalar(&Value::Num(2.0), "flag").is_err());
        assert!(zero_one_bool_scalar(&Value::from("on"), "flag").is_err());
    }
}
