use super::*;

pub(super) async fn write_tabular_file(
    value: Value,
    rest: Vec<Value>,
    convert_row_times: bool,
) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(invalid_argument("writetable: filename is required"));
    }
    let path = resolve_path(&rest[0])?;
    let delimiter = parse_named_option(&rest[1..], "Delimiter")
        .map(|value| scalar_text(value, "Delimiter"))
        .transpose()?
        .unwrap_or_else(|| ",".to_string());
    let write_variable_names = parse_bool_option(
        &strip_known_text_option(&rest[1..], "Delimiter")?,
        "WriteVariableNames",
        true,
        "writetable",
    )?;
    let object = into_table_object(value, "writetable")?;
    let text = table_delimited_text(&object, &delimiter, write_variable_names, convert_row_times)?;
    let bytes = text.into_bytes();
    runmat_filesystem::write_async(path, &bytes)
        .await
        .map_err(|err| {
            table_error_with_source(&TABLE_ERROR_IO, "writetable: file write failed", err)
        })?;
    Ok(Value::Num(bytes.len() as f64))
}

pub(super) fn strip_known_text_option(args: &[Value], name: &str) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "writetable: name-value options must be provided in pairs",
            ));
        }
        let option_name = scalar_text(&args[idx], "option name")?;
        if !option_name.eq_ignore_ascii_case(name) {
            out.push(args[idx].clone());
            out.push(args[idx + 1].clone());
        }
        idx += 2;
    }
    Ok(out)
}

pub(super) fn table_delimited_text(
    object: &ObjectInstance,
    delimiter: &str,
    write_variable_names: bool,
    convert_row_times: bool,
) -> BuiltinResult<String> {
    let mut names = table_variable_names_from_object(object)?;
    let variables = table_variables(object)?;
    let mut columns = Vec::with_capacity(names.len() + usize::from(convert_row_times));
    if convert_row_times {
        if let Some(row_times) = timetable_row_times(object)? {
            names.insert(0, "Time".to_string());
            columns.push(row_times);
        }
    }
    for name in &table_variable_names_from_object(object)? {
        columns.push(
            variables.fields.get(name).cloned().ok_or_else(|| {
                invalid_variable(format!("writetable: missing variable '{name}'"))
            })?,
        );
    }
    let height = table_height(object)?;
    let mut lines = Vec::new();
    if write_variable_names {
        lines.push(
            names
                .iter()
                .map(|name| escape_delimited_field(name, delimiter))
                .collect::<Vec<_>>()
                .join(delimiter),
        );
    }
    for row in 0..height {
        lines.push(
            columns
                .iter()
                .map(|value| {
                    row_value(value, row)
                        .map(|cell| escape_delimited_field(&cell_to_text(&cell), delimiter))
                })
                .collect::<BuiltinResult<Vec<_>>>()?
                .join(delimiter),
        );
    }
    lines.push(String::new());
    Ok(lines.join("\n"))
}

pub(super) fn cell_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::CharArray(array) if array.rows == 1 => array.data.iter().collect(),
        Value::Num(value) => format_key_number(*value),
        Value::Bool(value) => {
            if *value {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => format_key_number(tensor.data[0]),
        Value::StringArray(array) if array.data.len() == 1 => array.data[0].clone(),
        other => other.to_string(),
    }
}

pub(super) fn escape_delimited_field(text: &str, delimiter: &str) -> String {
    if text.contains(delimiter) || text.contains('"') || text.contains('\n') || text.contains('\r')
    {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

pub(super) fn char_rows(array: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        rows.push(array.data[start..start + array.cols].iter().collect());
    }
    rows
}

#[derive(Clone)]
pub(super) struct ReadTableOptions {
    file_type: ImportFileType,
    delimiter: Option<Delimiter>,
    read_variable_names: Option<bool>,
    read_row_names: bool,
    num_variables: Option<usize>,
    variable_names: Option<Vec<String>>,
    variable_types: Option<Vec<ImportVariableType>>,
    row_names: Option<Vec<String>>,
    num_header_lines: usize,
    range: Option<RangeSpec>,
    sheet: Option<SheetSelector>,
    preserve_variable_names: bool,
    treat_as_missing: HashSet<String>,
    empty_line_rule: EmptyLineRule,
    text_type: TextImportType,
    encoding: String,
    datetime_type: DatetimeImportType,
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
    pub(super) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut options = Self::default();
        let mut idx = 0usize;
        if let Some(Value::Struct(st)) = args.first() {
            for (name, value) in &st.fields {
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
            self.read_variable_names = Some(bool_scalar(value, "ReadVariableNames")?);
        } else if name.eq_ignore_ascii_case("ReadRowNames") {
            self.read_row_names = bool_scalar(value, "ReadRowNames")?;
        } else if name.eq_ignore_ascii_case("NumVariables") {
            let count = nonnegative_usize(value, "NumVariables")?;
            self.num_variables = (count > 0).then_some(count);
        } else if name.eq_ignore_ascii_case("VariableNames") {
            self.variable_names = optional_raw_variable_name_list(value)?;
        } else if name.eq_ignore_ascii_case("VariableTypes") {
            self.variable_types = optional_variable_type_list(value)?;
        } else if name.eq_ignore_ascii_case("RowNames") {
            self.row_names = Some(string_list(value)?);
        } else if name.eq_ignore_ascii_case("NumHeaderLines") {
            self.num_header_lines = nonnegative_usize(value, "NumHeaderLines")?;
        } else if name.eq_ignore_ascii_case("Range") {
            self.range = Some(RangeSpec::parse(value)?);
        } else if name.eq_ignore_ascii_case("DataRange") {
            self.range = optional_range_spec(value)?;
        } else if name.eq_ignore_ascii_case("Sheet") {
            self.sheet = optional_sheet_selector(value)?;
        } else if name.eq_ignore_ascii_case("TreatAsMissing") {
            for token in string_list(value)? {
                self.treat_as_missing
                    .insert(token.trim().to_ascii_lowercase());
            }
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

    fn is_missing(&self, token: &str) -> bool {
        let trimmed = token.trim();
        trimmed.is_empty()
            || self
                .treat_as_missing
                .contains(&trimmed.to_ascii_lowercase())
    }
}

pub(super) fn spreadsheet_import_options(args: Vec<Value>) -> BuiltinResult<Value> {
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

pub(super) async fn detect_import_options_from_file(
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

pub(super) async fn detect_text_import_options(
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

pub(super) async fn detect_spreadsheet_import_options(
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

pub(super) fn detected_options_from_rows(
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

pub(super) fn detected_variable_type_labels(
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

pub(super) fn infer_import_type_label(values: &[ImportCell], options: &ReadTableOptions) -> String {
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

pub(super) fn is_detected_numeric(value: &ImportCell, options: &ReadTableOptions) -> bool {
    match value {
        ImportCell::Empty | ImportCell::Number(_) => true,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            options.is_missing(token) || parse_numeric(token).is_some()
        }
        _ => false,
    }
}

pub(super) fn is_detected_logical(value: &ImportCell, options: &ReadTableOptions) -> bool {
    match value {
        ImportCell::Empty | ImportCell::Logical(_) => true,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            options.is_missing(token) || parse_logical(token).is_some()
        }
        _ => false,
    }
}

pub(super) fn is_detected_datetime(value: &ImportCell, options: &ReadTableOptions) -> bool {
    match value {
        ImportCell::Empty | ImportCell::DateTime(_) => true,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            options.is_missing(token) || parse_iso_datetime_to_datenum(token).is_some()
        }
        _ => false,
    }
}

pub(super) fn import_variable_type_label(kind: &ImportVariableType) -> String {
    match kind {
        ImportVariableType::Auto => "auto",
        ImportVariableType::Numeric(NumericDType::F64) => "double",
        ImportVariableType::Numeric(NumericDType::F32) => "single",
        ImportVariableType::Numeric(NumericDType::U8) => "uint8",
        ImportVariableType::Numeric(NumericDType::U16) => "uint16",
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

pub(super) fn detected_output_header_lines(
    options: &ReadTableOptions,
    header_rows_consumed: usize,
) -> usize {
    if options.range.is_some() {
        options.num_header_lines
    } else {
        options.num_header_lines + header_rows_consumed
    }
}

pub(super) fn detected_output_range(
    range: Option<RangeSpec>,
    header_rows_consumed: usize,
) -> Option<RangeSpec> {
    range.map(|mut range| {
        range.start_row = range.start_row.saturating_add(header_rows_consumed);
        range
    })
}

pub(super) fn import_file_type_label(file_type: ImportFileType) -> String {
    match file_type {
        ImportFileType::Text | ImportFileType::Auto => "text",
        ImportFileType::Spreadsheet => "spreadsheet",
    }
    .to_string()
}

pub(super) fn delimiter_label(delimiter: &Delimiter) -> String {
    match delimiter {
        Delimiter::Char('\t') => "\t".to_string(),
        Delimiter::Char(ch) => ch.to_string(),
        Delimiter::String(text) => text.clone(),
        Delimiter::Whitespace => "whitespace".to_string(),
    }
}

pub(super) fn sheet_value(sheet: &SheetSelector) -> Value {
    match sheet {
        SheetSelector::Name(name) => Value::String(name.clone()),
        SheetSelector::Index(index) => Value::Num((*index + 1) as f64),
    }
}

pub(super) fn range_spec_value(range: RangeSpec) -> BuiltinResult<Value> {
    Ok(Value::String(range_spec_text(range)))
}

pub(super) fn range_spec_text(range: RangeSpec) -> String {
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

pub(super) fn range_ref_text(
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

pub(super) fn spreadsheet_column_label(mut col: usize) -> String {
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

pub(super) fn string_array_value(values: Vec<String>, context: &str) -> BuiltinResult<Value> {
    let len = values.len();
    StringArray::new(values, vec![1, len])
        .map(Value::StringArray)
        .map_err(|err| invalid_variable(format!("{context}: {err}")))
}

#[derive(Clone)]
pub(super) struct SpreadsheetImportOptions {
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
pub(super) enum ImportVariableType {
    Auto,
    Numeric(NumericDType),
    Logical,
    Text(TextImportType),
    CellStr,
    Categorical,
    Datetime,
    Duration,
}

impl ImportVariableType {
    pub(super) fn parse(raw: &str) -> BuiltinResult<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "double" => Ok(Self::Numeric(NumericDType::F64)),
            "single" => Ok(Self::Numeric(NumericDType::F32)),
            "uint8" => Ok(Self::Numeric(NumericDType::U8)),
            "uint16" => Ok(Self::Numeric(NumericDType::U16)),
            "logical" | "bool" | "boolean" => Ok(Self::Logical),
            "string" => Ok(Self::Text(TextImportType::String)),
            "char" => Ok(Self::Text(TextImportType::Char)),
            "cellstr" => Ok(Self::CellStr),
            "categorical" => Ok(Self::Categorical),
            "int8" | "int16" | "int32" | "int64" | "uint32" | "uint64" => {
                Err(invalid_argument(format!(
                    "readtable: unsupported VariableTypes entry '{}'; RunMat table imports currently support double, single, uint8, and uint16 numeric arrays",
                    raw.trim()
                )))
            }
            "datetime" => Ok(Self::Datetime),
            "duration" => Ok(Self::Duration),
            other => Err(invalid_argument(format!(
                "readtable: unsupported VariableTypes entry '{other}'"
            ))),
        }
    }

    pub(super) fn canonical_label(raw: &str) -> BuiltinResult<String> {
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
pub(super) enum TextImportType {
    String,
    Char,
}

impl TextImportType {
    pub(super) fn parse(value: &Value, context: &str) -> BuiltinResult<Self> {
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
pub(super) enum EmptyLineRule {
    Skip,
    Read,
}

#[derive(Clone, Copy)]
pub(super) enum DatetimeImportType {
    Datetime,
    Text,
    ExcelDatenum,
}

impl DatetimeImportType {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
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
pub(super) enum ImportFileType {
    Auto,
    Text,
    Spreadsheet,
}

impl ImportFileType {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
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
pub(super) enum SheetSelector {
    Name(String),
    Index(usize),
}

impl SheetSelector {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::Int(i) if i.to_i64() >= 1 => Ok(Self::Index(i.to_i64() as usize - 1)),
            Value::Num(n)
                if n.is_finite() && *n >= 1.0 && (n.round() - n).abs() <= f64::EPSILON =>
            {
                Ok(Self::Index(n.round() as usize - 1))
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
pub(super) enum Delimiter {
    Char(char),
    String(String),
    Whitespace,
}

impl Delimiter {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
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
pub(super) struct RangeSpec {
    start_row: usize,
    start_col: usize,
    end_row: Option<usize>,
    end_col: Option<usize>,
}

impl RangeSpec {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Self::parse_text(text),
            Value::CharArray(ca) if ca.rows == 1 => {
                let text: String = ca.data.iter().collect();
                Self::parse_text(&text)
            }
            Value::StringArray(sa) if sa.data.len() == 1 => Self::parse_text(&sa.data[0]),
            Value::Tensor(t) if t.data.len() == 2 || t.data.len() == 4 => {
                let mut indices = Vec::with_capacity(t.data.len());
                for value in &t.data {
                    indices.push(one_based_to_zero(*value, usize::MAX, "Range")?);
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

    pub(super) fn parse_text(text: &str) -> BuiltinResult<Self> {
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
}

pub(super) fn parse_cell_ref(token: &str) -> BuiltinResult<(Option<usize>, Option<usize>)> {
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

pub(super) fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let text = scalar_text(value, "filename").map_err(|_| {
        table_error(
            &TABLE_ERROR_INVALID_ARGUMENT,
            "readtable: filename must be a string scalar or character vector",
        )
    })?;
    if text.trim().is_empty() {
        return Err(invalid_argument("readtable: filename must not be empty"));
    }
    let expanded =
        expand_user_path(&text, "readtable").map_err(|msg| invalid_argument(msg.to_string()))?;
    Ok(Path::new(&expanded).to_path_buf())
}

pub(super) async fn read_table_from_file(
    path: &Path,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    match options.file_type {
        ImportFileType::Spreadsheet => read_spreadsheet_table(path, options).await,
        ImportFileType::Text => read_text_table(path, options).await,
        ImportFileType::Auto if is_spreadsheet_path(path) => {
            read_spreadsheet_table(path, options).await
        }
        ImportFileType::Auto => read_text_table(path, options).await,
    }
}

pub(super) async fn read_cell_from_file(
    path: &Path,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    match options.file_type {
        ImportFileType::Spreadsheet => read_spreadsheet_cells(path, options).await,
        ImportFileType::Text => read_text_cells(path, options).await,
        ImportFileType::Auto if is_spreadsheet_path(path) => {
            read_spreadsheet_cells(path, options).await
        }
        ImportFileType::Auto => read_text_cells(path, options).await,
    }
}

pub(super) async fn read_text_table(
    path: &Path,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    if options.sheet.is_some() {
        return Err(invalid_argument(
            "readtable: Sheet is only valid for spreadsheet files",
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
    import_rows_to_table(rows, options)
}

pub(super) async fn read_text_cells(
    path: &Path,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    if options.sheet.is_some() {
        return Err(invalid_argument(
            "readcell: Sheet is only valid for spreadsheet files",
        ));
    }
    let bytes = read_file_bytes(path).await?;
    let text = strip_utf8_bom(decode_text_bytes(&bytes, &options.encoding)?);
    let raw_lines = text.lines().map(ToString::to_string).collect::<Vec<_>>();
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
    import_rows_to_cell(rows, options)
}

pub(super) async fn read_spreadsheet_table(
    path: &Path,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    if options.delimiter.is_some() {
        return Err(invalid_argument(
            "readtable: Delimiter is only valid for text files",
        ));
    }
    let bytes = read_file_bytes(path).await?;
    let cursor = Cursor::new(bytes);
    let mut workbook = open_workbook_auto_from_rs(cursor).map_err(|err| {
        table_error(
            &TABLE_ERROR_UNSUPPORTED_FILE,
            format!(
                "readtable: unable to open spreadsheet '{}': {err}",
                path.display()
            ),
        )
    })?;
    let range = match &options.sheet {
        Some(SheetSelector::Name(name)) => workbook.worksheet_range(name).map_err(|err| {
            invalid_argument(format!("readtable: unable to read sheet '{name}': {err:?}"))
        })?,
        Some(SheetSelector::Index(index)) => workbook
            .worksheet_range_at(*index)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "readtable: sheet index {} exceeds bounds",
                    index + 1
                ))
            })?
            .map_err(|err| {
                invalid_argument(format!(
                    "readtable: unable to read sheet {}: {err:?}",
                    index + 1
                ))
            })?,
        None => workbook
            .worksheet_range_at(0)
            .ok_or_else(|| invalid_argument("readtable: spreadsheet contains no worksheets"))?
            .map_err(|err| {
                invalid_argument(format!("readtable: unable to read first sheet: {err:?}"))
            })?,
    };
    let rows = spreadsheet_range_to_rows(&range, options)?;
    import_rows_to_table(rows, options)
}

pub(super) async fn read_spreadsheet_cells(
    path: &Path,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    if options.delimiter.is_some() {
        return Err(invalid_argument(
            "readcell: Delimiter is only valid for text files",
        ));
    }
    let bytes = read_file_bytes(path).await?;
    let cursor = Cursor::new(bytes);
    let mut workbook = open_workbook_auto_from_rs(cursor).map_err(|err| {
        table_error(
            &TABLE_ERROR_UNSUPPORTED_FILE,
            format!(
                "readcell: unable to open spreadsheet '{}': {err}",
                path.display()
            ),
        )
    })?;
    let range = match &options.sheet {
        Some(SheetSelector::Name(name)) => workbook.worksheet_range(name).map_err(|err| {
            invalid_argument(format!("readcell: unable to read sheet '{name}': {err:?}"))
        })?,
        Some(SheetSelector::Index(index)) => workbook
            .worksheet_range_at(*index)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "readcell: sheet index {} exceeds bounds",
                    index + 1
                ))
            })?
            .map_err(|err| {
                invalid_argument(format!(
                    "readcell: unable to read sheet {}: {err:?}",
                    index + 1
                ))
            })?,
        None => workbook
            .worksheet_range_at(0)
            .ok_or_else(|| invalid_argument("readcell: spreadsheet contains no worksheets"))?
            .map_err(|err| {
                invalid_argument(format!("readcell: unable to read first sheet: {err:?}"))
            })?,
    };
    let rows = spreadsheet_range_to_rows(&range, options)?;
    import_rows_to_cell(rows, options)
}

pub(super) async fn read_file_bytes(path: &Path) -> BuiltinResult<Vec<u8>> {
    let mut file = File::open_async(path).await.map_err(|err| {
        table_error_with_source(
            &TABLE_ERROR_IO,
            format!("readtable: unable to open '{}': {err}", path.display()),
            err,
        )
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|err| {
        table_error_with_source(
            &TABLE_ERROR_IO,
            format!("readtable: unable to read '{}': {err}", path.display()),
            err,
        )
    })?;
    Ok(bytes)
}

pub(super) fn is_spreadsheet_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref(),
        Some("xls") | Some("xlsx") | Some("xlsm") | Some("xlsb") | Some("ods")
    )
}

pub(super) fn validate_encoding_label(label: &str) -> BuiltinResult<()> {
    encoding_for_label(label)
        .map(|_| ())
        .ok_or_else(|| invalid_argument(format!("readtable: unsupported Encoding '{label}'")))
}

pub(super) fn encoding_for_label(label: &str) -> Option<&'static Encoding> {
    let label = label.trim();
    if label.is_empty()
        || label.eq_ignore_ascii_case("auto")
        || label.eq_ignore_ascii_case("default")
        || label.eq_ignore_ascii_case("system")
        || label.eq_ignore_ascii_case("native")
        || label.eq_ignore_ascii_case("utf-8")
        || label.eq_ignore_ascii_case("utf8")
        || label.eq_ignore_ascii_case("unicode")
    {
        return Some(UTF_8);
    }
    Encoding::for_label(label.as_bytes())
}

pub(super) fn decode_text_bytes(bytes: &[u8], encoding: &str) -> BuiltinResult<String> {
    let (encoding, offset) = if encoding.trim().eq_ignore_ascii_case("auto") {
        Encoding::for_bom(bytes).unwrap_or((UTF_8, 0))
    } else {
        (
            encoding_for_label(encoding).ok_or_else(|| {
                invalid_argument(format!("readtable: unsupported Encoding '{encoding}'"))
            })?,
            0,
        )
    };
    let (decoded, _, had_errors) = encoding.decode(&bytes[offset..]);
    if had_errors {
        return Err(table_error(
            &TABLE_ERROR_IO,
            format!(
                "readtable: unable to decode file contents using encoding '{}'",
                encoding.name()
            ),
        ));
    }
    Ok(decoded.into_owned())
}

pub(super) fn strip_utf8_bom(text: String) -> String {
    text.strip_prefix('\u{FEFF}')
        .map(ToString::to_string)
        .unwrap_or(text)
}

#[derive(Clone, Debug)]
pub(super) enum ImportCell {
    Empty,
    Text(String),
    Number(f64),
    Logical(bool),
    DateTime(f64),
    Error(String),
}

impl ImportCell {
    fn from_text(text: String) -> Self {
        if text.trim().is_empty() {
            Self::Empty
        } else {
            Self::Text(text)
        }
    }

    fn display_text(&self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Text(text) => text.clone(),
            Self::Number(value) => format_key_number(*value),
            Self::Logical(value) => value.to_string(),
            Self::DateTime(serial) => format_key_number(*serial),
            Self::Error(text) => text.clone(),
        }
    }

    fn is_missing(&self, options: &ReadTableOptions) -> bool {
        match self {
            Self::Empty => true,
            Self::Text(text) => options.is_missing(text),
            _ => false,
        }
    }

    fn is_likely_data_token(&self, options: &ReadTableOptions) -> bool {
        match self {
            Self::Number(_) | Self::Logical(_) | Self::DateTime(_) => true,
            Self::Empty => false,
            Self::Text(text) => {
                let token = unquote(text.trim()).trim();
                options.is_missing(token)
                    || parse_numeric(token).is_some()
                    || parse_logical(token).is_some()
                    || parse_iso_datetime_to_datenum(token).is_some()
            }
            Self::Error(_) => true,
        }
    }
}

pub(super) fn spreadsheet_cell_to_import(cell: &SpreadsheetData) -> ImportCell {
    match cell {
        SpreadsheetData::Empty => ImportCell::Empty,
        SpreadsheetData::Int(value) => ImportCell::Number(*value as f64),
        SpreadsheetData::Float(value) => ImportCell::Number(*value),
        SpreadsheetData::String(text) => ImportCell::Text(text.clone()),
        SpreadsheetData::Bool(value) => ImportCell::Logical(*value),
        SpreadsheetData::DateTime(value) => value
            .as_datetime()
            .map(crate::builtins::datetime::datenum_from_naive)
            .map(ImportCell::DateTime)
            .unwrap_or_else(|| ImportCell::Number(value.as_f64())),
        SpreadsheetData::DateTimeIso(text) => parse_iso_datetime_to_datenum(text)
            .map(ImportCell::DateTime)
            .unwrap_or_else(|| ImportCell::Text(text.clone())),
        SpreadsheetData::DurationIso(text) => ImportCell::Text(text.clone()),
        SpreadsheetData::Error(err) => ImportCell::Error(err.to_string()),
    }
}

pub(super) fn spreadsheet_range_to_rows(
    range: &calamine::Range<SpreadsheetData>,
    options: &ReadTableOptions,
) -> BuiltinResult<Vec<Vec<ImportCell>>> {
    if range.is_empty() {
        return Ok(Vec::new());
    }
    let Some((range_start_row, range_start_col)) = range.start() else {
        return Ok(Vec::new());
    };
    let Some((range_end_row, range_end_col)) = range.end() else {
        return Ok(Vec::new());
    };
    let start_row = options
        .range
        .map(|spec| checked_u32(spec.start_row, "Range row"))
        .transpose()?
        .unwrap_or(range_start_row);
    let start_col = options
        .range
        .map(|spec| checked_u32(spec.start_col, "Range column"))
        .transpose()?
        .unwrap_or(range_start_col);
    let end_row = options
        .range
        .and_then(|spec| spec.end_row)
        .map(|row| checked_u32(row, "Range row"))
        .transpose()?
        .unwrap_or(range_end_row);
    let end_col = options
        .range
        .and_then(|spec| spec.end_col)
        .map(|col| checked_u32(col, "Range column"))
        .transpose()?
        .unwrap_or(range_end_col);
    if start_row > end_row || start_col > end_col {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for row_idx in start_row..=end_row {
        let mut row = Vec::new();
        for col_idx in start_col..=end_col {
            row.push(
                range
                    .get_value((row_idx, col_idx))
                    .map(spreadsheet_cell_to_import)
                    .unwrap_or(ImportCell::Empty),
            );
        }
        if matches!(options.empty_line_rule, EmptyLineRule::Skip)
            && row.iter().all(|cell| cell.is_missing(options))
        {
            continue;
        }
        rows.push(row);
    }
    if options.num_header_lines > 0 {
        Ok(rows.into_iter().skip(options.num_header_lines).collect())
    } else {
        Ok(rows)
    }
}

pub(super) fn checked_u32(value: usize, context: &str) -> BuiltinResult<u32> {
    u32::try_from(value).map_err(|_| invalid_argument(format!("readtable: {context} overflow")))
}

pub(super) fn detect_delimiter(lines: &[String]) -> Option<Delimiter> {
    let candidates = [',', '\t', ';', '|'];
    let mut best: Option<(f64, Delimiter)> = None;
    for candidate in candidates {
        let counts = lines
            .iter()
            .take(32)
            .filter(|line| line.contains(candidate))
            .map(|line| split_with_char_delim(line, candidate).len())
            .filter(|count| *count >= 2)
            .collect::<Vec<_>>();
        if counts.is_empty() {
            continue;
        }
        let avg = counts.iter().copied().sum::<usize>() as f64 / counts.len() as f64;
        if avg >= 2.0
            && best
                .as_ref()
                .map(|(best_avg, _)| avg > *best_avg)
                .unwrap_or(true)
        {
            best = Some((avg, Delimiter::Char(candidate)));
        }
    }
    best.map(|(_, delimiter)| delimiter).or_else(|| {
        lines
            .iter()
            .take(32)
            .any(|line| line.split_whitespace().count() > 1)
            .then_some(Delimiter::Whitespace)
    })
}

pub(super) fn split_with_char_delim(line: &str, delimiter: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                current.push('"');
                chars.next();
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if ch == delimiter && !in_quotes {
            out.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    out.push(current);
    out
}

pub(super) fn parse_text_records(
    text: &str,
    delimiter: &Delimiter,
    empty_line_rule: EmptyLineRule,
) -> Vec<Vec<ImportCell>> {
    match delimiter {
        Delimiter::Whitespace => parse_whitespace_records(text, empty_line_rule),
        Delimiter::Char(ch) => parse_delimited_records(text, &ch.to_string(), empty_line_rule),
        Delimiter::String(pattern) => parse_delimited_records(text, pattern, empty_line_rule),
    }
}

pub(super) fn parse_delimited_records(
    text: &str,
    delimiter: &str,
    empty_line_rule: EmptyLineRule,
) -> Vec<Vec<ImportCell>> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut idx = 0usize;
    while idx < text.len() {
        let ch = text[idx..].chars().next().expect("valid char boundary");
        if ch == '"' {
            if in_quotes && text[idx + ch.len_utf8()..].starts_with('"') {
                current.push('"');
                idx += ch.len_utf8() + 1;
                continue;
            }
            in_quotes = !in_quotes;
            idx += ch.len_utf8();
            continue;
        }
        if !in_quotes && !delimiter.is_empty() && text[idx..].starts_with(delimiter) {
            row.push(ImportCell::from_text(std::mem::take(&mut current)));
            idx += delimiter.len();
            continue;
        }
        if !in_quotes && (ch == '\n' || ch == '\r') {
            row.push(ImportCell::from_text(std::mem::take(&mut current)));
            push_import_record(&mut records, std::mem::take(&mut row), empty_line_rule);
            idx += ch.len_utf8();
            if ch == '\r' && text[idx..].starts_with('\n') {
                idx += 1;
            }
            continue;
        }
        current.push(ch);
        idx += ch.len_utf8();
    }
    if !current.is_empty() || !row.is_empty() || text.ends_with(delimiter) {
        row.push(ImportCell::from_text(current));
        push_import_record(&mut records, row, empty_line_rule);
    }
    records
}

pub(super) fn parse_whitespace_records(
    text: &str,
    empty_line_rule: EmptyLineRule,
) -> Vec<Vec<ImportCell>> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut field_open = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                current.push('"');
                chars.next();
            } else {
                in_quotes = !in_quotes;
            }
            field_open = true;
            continue;
        }
        if !in_quotes && (ch == '\n' || ch == '\r') {
            if field_open || !current.is_empty() {
                row.push(ImportCell::from_text(std::mem::take(&mut current)));
            }
            field_open = false;
            push_import_record(&mut records, std::mem::take(&mut row), empty_line_rule);
            if ch == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
            }
            continue;
        }
        if !in_quotes && ch.is_whitespace() {
            if field_open || !current.is_empty() {
                row.push(ImportCell::from_text(std::mem::take(&mut current)));
                field_open = false;
            }
            continue;
        }
        current.push(ch);
        field_open = true;
    }
    if field_open || !current.is_empty() {
        row.push(ImportCell::from_text(current));
    }
    if !row.is_empty() {
        push_import_record(&mut records, row, empty_line_rule);
    }
    records
}

pub(super) fn push_import_record(
    records: &mut Vec<Vec<ImportCell>>,
    row: Vec<ImportCell>,
    empty_line_rule: EmptyLineRule,
) {
    if matches!(empty_line_rule, EmptyLineRule::Skip)
        && row.iter().all(|cell| matches!(cell, ImportCell::Empty))
    {
        return;
    }
    records.push(row);
}

pub(super) fn apply_import_range(
    rows: Vec<Vec<ImportCell>>,
    range: RangeSpec,
) -> Vec<Vec<ImportCell>> {
    if rows.is_empty() {
        return rows;
    }
    let end_row = range
        .end_row
        .unwrap_or_else(|| rows.len().saturating_sub(1));
    let max_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let end_col = range.end_col.unwrap_or_else(|| max_cols.saturating_sub(1));
    rows.into_iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            if idx < range.start_row || idx > end_row {
                return None;
            }
            let selected = (range.start_col..=end_col)
                .map(|col| row.get(col).cloned().unwrap_or(ImportCell::Empty))
                .collect::<Vec<_>>();
            Some(selected)
        })
        .collect()
}

pub(super) fn import_rows_to_cell(
    rows: Vec<Vec<ImportCell>>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let col_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut data = Vec::with_capacity(row_count.saturating_mul(col_count));
    for row in rows {
        for col in 0..col_count {
            let cell = row.get(col).cloned().unwrap_or(ImportCell::Empty);
            data.push(import_cell_value(cell, options));
        }
    }
    CellArray::new(data, row_count, col_count)
        .map(Value::Cell)
        .map_err(|err| invalid_variable(format!("readcell: {err}")))
}

pub(super) fn import_cell_value(cell: ImportCell, options: &ReadTableOptions) -> Value {
    match cell {
        ImportCell::Empty => Value::String(String::new()),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Value::String(String::new())
            } else if let Some(value) = parse_logical(token) {
                Value::Bool(value)
            } else if let Some(value) = parse_numeric(token) {
                Value::Num(value)
            } else {
                Value::String(token.to_string())
            }
        }
        ImportCell::Number(value) => Value::Num(value),
        ImportCell::Logical(value) => Value::Bool(value),
        ImportCell::DateTime(serial) => Value::Num(serial),
        ImportCell::Error(text) => Value::String(text),
    }
}

pub(super) fn import_rows_to_table(
    mut rows: Vec<Vec<ImportCell>>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut variable_names = options.variable_names.clone();
    let read_variable_names = options
        .read_variable_names
        .unwrap_or_else(|| variable_names.is_none() && should_read_variable_names(&rows, options));
    if variable_names.is_none() && read_variable_names && !rows.is_empty() {
        variable_names = Some(
            rows.remove(0)
                .into_iter()
                .map(|cell| cell.display_text())
                .collect(),
        );
    }

    let mut row_names = options.row_names.clone();
    if options.read_row_names && !rows.is_empty() {
        row_names = Some(
            rows.iter_mut()
                .map(|row| {
                    if row.is_empty() {
                        String::new()
                    } else {
                        row.remove(0).display_text()
                    }
                })
                .collect(),
        );
        if let Some(names) = variable_names.as_mut() {
            if !names.is_empty() {
                names.remove(0);
            }
        }
    }

    let column_count = import_column_count(&rows, &variable_names, options)?;
    let names = import_variable_names(variable_names, column_count, options);

    let mut columns = Vec::with_capacity(names.len());
    for col in 0..names.len() {
        let values = rows
            .iter()
            .map(|row| row.get(col).cloned().unwrap_or(ImportCell::Empty))
            .collect::<Vec<_>>();
        let requested_type = options
            .variable_types
            .as_ref()
            .and_then(|types| types.get(col))
            .copied();
        columns.push(import_column(values, options, requested_type)?);
    }
    table_from_columns_with_properties(names, columns, row_names)
}

pub(super) fn import_column_count(
    rows: &[Vec<ImportCell>],
    variable_names: &Option<Vec<String>>,
    options: &ReadTableOptions,
) -> BuiltinResult<usize> {
    let data_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let name_cols = variable_names.as_ref().map(Vec::len).unwrap_or(0);
    let type_cols = options.variable_types.as_ref().map(Vec::len).unwrap_or(0);
    if let Some(count) = options.num_variables {
        if name_cols > count {
            return Err(invalid_argument(
                "readtable: VariableNames length exceeds NumVariables",
            ));
        }
        if type_cols > count {
            return Err(invalid_argument(
                "readtable: VariableTypes length exceeds NumVariables",
            ));
        }
        return Ok(count);
    }
    Ok(data_cols.max(name_cols).max(type_cols))
}

pub(super) fn import_variable_names(
    variable_names: Option<Vec<String>>,
    column_count: usize,
    options: &ReadTableOptions,
) -> Vec<String> {
    match variable_names {
        Some(mut names) => {
            while names.len() < column_count {
                names.push(format!("Var{}", names.len() + 1));
            }
            names.truncate(column_count);
            if options.preserve_variable_names {
                make_unique_names(names)
            } else {
                make_unique_variable_names(names)
            }
        }
        None => generated_variable_names(column_count),
    }
}

pub(super) fn should_read_variable_names(
    rows: &[Vec<ImportCell>],
    options: &ReadTableOptions,
) -> bool {
    let Some(first) = rows.first() else {
        return false;
    };
    if first.is_empty() {
        return false;
    }
    let names = first
        .iter()
        .map(ImportCell::display_text)
        .map(|text| text.trim().to_string())
        .collect::<Vec<_>>();
    if names.iter().any(|name| name.is_empty()) {
        return false;
    }
    if first.iter().all(|cell| cell.is_likely_data_token(options)) {
        return false;
    }
    true
}

pub(super) fn import_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    requested_type: Option<ImportVariableType>,
) -> BuiltinResult<Value> {
    match requested_type.unwrap_or(ImportVariableType::Auto) {
        ImportVariableType::Auto => infer_import_column(values, options),
        ImportVariableType::Numeric(dtype) => import_numeric_column(values, options, dtype),
        ImportVariableType::Logical => import_logical_column(values, options),
        ImportVariableType::Text(kind) => import_text_column(values, options, kind),
        ImportVariableType::CellStr => import_cellstr_column(values, options),
        ImportVariableType::Categorical => import_categorical_column(values, options),
        ImportVariableType::Datetime => import_datetime_column(values, options),
        ImportVariableType::Duration => import_duration_column(values, options),
    }
}

pub(super) fn import_numeric_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    let mut numeric = Vec::with_capacity(values.len());
    for value in &values {
        let parsed = numeric_from_import_cell(value, options, dtype.class_name())?;
        numeric.push(cast_import_numeric(parsed, dtype));
    }
    Tensor::new_with_dtype(numeric, vec![values.len(), 1], dtype)
        .map(Value::Tensor)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

pub(super) fn numeric_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
    context: &str,
) -> BuiltinResult<f64> {
    match value {
        ImportCell::Empty => Ok(f64::NAN),
        ImportCell::Number(value) => Ok(*value),
        ImportCell::Logical(value) => Ok(if *value { 1.0 } else { 0.0 }),
        ImportCell::DateTime(serial) => Ok(*serial),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Ok(f64::NAN)
            } else {
                parse_numeric(token).ok_or_else(|| {
                    invalid_variable(format!("readtable: cannot import '{token}' as {context}"))
                })
            }
        }
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as {context}"
        ))),
    }
}

pub(super) fn cast_import_numeric(value: f64, dtype: NumericDType) -> f64 {
    match dtype {
        NumericDType::F64 => value,
        NumericDType::F32 => (value as f32) as f64,
        NumericDType::U8 => {
            if value.is_finite() {
                value.round().clamp(0.0, u8::MAX as f64)
            } else {
                0.0
            }
        }
        NumericDType::U16 => {
            if value.is_finite() {
                value.round().clamp(0.0, u16::MAX as f64)
            } else {
                0.0
            }
        }
    }
}

pub(super) fn import_logical_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut logical = Vec::with_capacity(values.len());
    for value in &values {
        logical.push(logical_from_import_cell(value, options)?);
    }
    LogicalArray::new(logical, vec![values.len(), 1])
        .map(Value::LogicalArray)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

pub(super) fn logical_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> BuiltinResult<u8> {
    let flag = match value {
        ImportCell::Empty => false,
        ImportCell::Logical(value) => *value,
        ImportCell::Number(value) => *value != 0.0,
        ImportCell::DateTime(serial) => *serial != 0.0,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                false
            } else if let Some(value) = parse_logical(token) {
                value
            } else if let Some(value) = parse_numeric(token) {
                value != 0.0
            } else {
                return Err(invalid_variable(format!(
                    "readtable: cannot import '{token}' as logical"
                )));
            }
        }
        ImportCell::Error(text) => {
            return Err(invalid_variable(format!(
                "readtable: cannot import spreadsheet error '{text}' as logical"
            )));
        }
    };
    Ok(u8::from(flag))
}

pub(super) fn import_text_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    kind: TextImportType,
) -> BuiltinResult<Value> {
    let strings = import_text_values(values, options);
    match kind {
        TextImportType::String => StringArray::new(strings.clone(), vec![strings.len(), 1])
            .map(Value::StringArray)
            .map_err(|err| invalid_variable(format!("readtable: {err}"))),
        TextImportType::Char => import_char_column(strings),
    }
}

pub(super) fn import_text_values(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> Vec<String> {
    values
        .into_iter()
        .map(|value| {
            if value.is_missing(options) {
                String::new()
            } else {
                unquote(value.display_text().trim()).to_string()
            }
        })
        .collect()
}

pub(super) fn import_char_column(strings: Vec<String>) -> BuiltinResult<Value> {
    let rows = strings.len();
    let cols = strings
        .iter()
        .map(|text| text.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = vec![' '; rows * cols];
    for (row, text) in strings.iter().enumerate() {
        for (col, ch) in text.chars().enumerate() {
            data[row * cols + col] = ch;
        }
    }
    CharArray::new(data, rows, cols)
        .map(Value::CharArray)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

pub(super) fn import_cellstr_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let strings = import_text_values(values, options);
    let rows = strings.len();
    let cells = strings
        .into_iter()
        .map(|text| Value::CharArray(CharArray::new_row(&text)))
        .collect::<Vec<_>>();
    CellArray::new(cells, rows, 1)
        .map(Value::Cell)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

pub(super) fn import_categorical_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let strings = import_text_values(values, options);
    categorical_from_args(vec![Value::StringArray(
        StringArray::new(strings.clone(), vec![strings.len(), 1])
            .map_err(|err| invalid_variable(format!("readtable: {err}")))?,
    )])
}

pub(super) fn import_datetime_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    if matches!(options.datetime_type, DatetimeImportType::Text) {
        return import_text_column(values, options, options.text_type);
    }

    let mut serials = Vec::with_capacity(values.len());
    for value in &values {
        serials.push(datetime_serial_from_import_cell(value, options)?);
    }
    let tensor = Tensor::new(serials, vec![values.len(), 1])
        .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
    if matches!(options.datetime_type, DatetimeImportType::ExcelDatenum) {
        Ok(Value::Tensor(tensor))
    } else {
        crate::builtins::datetime::datetime_object_from_serial_tensor(tensor, "yyyy-MM-dd HH:mm:ss")
    }
}

pub(super) fn datetime_serial_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> BuiltinResult<f64> {
    match value {
        ImportCell::Empty => Ok(f64::NAN),
        ImportCell::DateTime(serial) => Ok(*serial),
        ImportCell::Number(value) => Ok(*value),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Ok(f64::NAN)
            } else if let Some(serial) = parse_iso_datetime_to_datenum(token) {
                Ok(serial)
            } else if let Some(serial) = parse_numeric(token) {
                Ok(serial)
            } else {
                Err(invalid_variable(format!(
                    "readtable: cannot import '{token}' as datetime"
                )))
            }
        }
        ImportCell::Logical(_) => Err(invalid_variable(
            "readtable: cannot import logical value as datetime",
        )),
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as datetime"
        ))),
    }
}

pub(super) fn import_duration_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut days = Vec::with_capacity(values.len());
    for value in &values {
        days.push(duration_days_from_import_cell(value, options)?);
    }
    let tensor = Tensor::new(days, vec![values.len(), 1])
        .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
    crate::builtins::duration::duration_object_from_days_tensor(
        tensor,
        crate::builtins::duration::DEFAULT_DURATION_FORMAT,
    )
}

pub(super) fn duration_days_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> BuiltinResult<f64> {
    match value {
        ImportCell::Empty => Ok(f64::NAN),
        ImportCell::Number(value) => Ok(*value),
        ImportCell::Logical(value) => Ok(if *value { 1.0 } else { 0.0 }),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Ok(f64::NAN)
            } else {
                parse_duration_to_days(token).ok_or_else(|| {
                    invalid_variable(format!("readtable: cannot import '{token}' as duration"))
                })
            }
        }
        ImportCell::DateTime(_) => Err(invalid_variable(
            "readtable: cannot import datetime value as duration",
        )),
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as duration"
        ))),
    }
}

pub(super) fn infer_import_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut numeric = Vec::with_capacity(values.len());
    let mut all_numeric = true;
    for value in &values {
        match value {
            ImportCell::Empty => numeric.push(f64::NAN),
            ImportCell::Number(value) => numeric.push(*value),
            ImportCell::Text(text) => {
                let token = unquote(text.trim()).trim();
                if options.is_missing(token) {
                    numeric.push(f64::NAN);
                } else if let Some(value) = parse_numeric(token) {
                    numeric.push(value);
                } else {
                    all_numeric = false;
                    break;
                }
            }
            _ => {
                all_numeric = false;
                break;
            }
        }
    }
    if all_numeric {
        return Tensor::new(numeric, vec![values.len(), 1])
            .map(Value::Tensor)
            .map_err(|err| invalid_variable(format!("readtable: {err}")));
    }

    let mut logical = Vec::with_capacity(values.len());
    let mut all_logical = true;
    for value in &values {
        match value {
            ImportCell::Empty => logical.push(0),
            ImportCell::Logical(value) => logical.push(i32::from(*value) as u8),
            ImportCell::Text(text) => {
                let token = unquote(text.trim()).trim();
                if options.is_missing(token) {
                    logical.push(0);
                } else if let Some(value) = parse_logical(token) {
                    logical.push(i32::from(value) as u8);
                } else {
                    all_logical = false;
                    break;
                }
            }
            _ => {
                all_logical = false;
                break;
            }
        }
    }
    if all_logical {
        return LogicalArray::new(logical, vec![values.len(), 1])
            .map(Value::LogicalArray)
            .map_err(|err| invalid_variable(format!("readtable: {err}")));
    }

    if !matches!(options.datetime_type, DatetimeImportType::Text) {
        let mut serials = Vec::with_capacity(values.len());
        let mut all_datetime = true;
        for value in &values {
            match value {
                ImportCell::Empty => serials.push(f64::NAN),
                ImportCell::DateTime(serial) => serials.push(*serial),
                ImportCell::Text(text) => {
                    let token = unquote(text.trim()).trim();
                    if options.is_missing(token) {
                        serials.push(f64::NAN);
                    } else if let Some(serial) = parse_iso_datetime_to_datenum(token) {
                        serials.push(serial);
                    } else {
                        all_datetime = false;
                        break;
                    }
                }
                _ => {
                    all_datetime = false;
                    break;
                }
            }
        }
        if all_datetime {
            let tensor = Tensor::new(serials, vec![values.len(), 1])
                .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
            if matches!(options.datetime_type, DatetimeImportType::ExcelDatenum) {
                return Ok(Value::Tensor(tensor));
            }
            return crate::builtins::datetime::datetime_object_from_serial_tensor(
                tensor,
                "yyyy-MM-dd HH:mm:ss",
            );
        }
    }

    import_text_column(values, options, options.text_type)
}

pub(super) fn parse_numeric(token: &str) -> Option<f64> {
    match token.to_ascii_lowercase().as_str() {
        "nan" => Some(f64::NAN),
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => token.parse::<f64>().ok(),
    }
}

pub(super) fn parse_logical(token: &str) -> Option<bool> {
    match token.to_ascii_lowercase().as_str() {
        "true" | "t" | "yes" | "on" => Some(true),
        "false" | "f" | "no" | "off" => Some(false),
        _ => None,
    }
}

pub(super) fn parse_duration_to_days(token: &str) -> Option<f64> {
    parse_numeric(token).or_else(|| parse_clock_duration_to_days(token))
}

pub(super) fn parse_clock_duration_to_days(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (sign, body) = if let Some(rest) = trimmed.strip_prefix('-') {
        (-1.0, rest)
    } else if let Some(rest) = trimmed.strip_prefix('+') {
        (1.0, rest)
    } else {
        (1.0, trimmed)
    };
    let parts = body.split(':').collect::<Vec<_>>();
    let (hours, minutes, seconds) = match parts.as_slice() {
        [hours, minutes] => (
            hours.parse::<f64>().ok()?,
            minutes.parse::<f64>().ok()?,
            0.0,
        ),
        [hours, minutes, seconds] => (
            hours.parse::<f64>().ok()?,
            minutes.parse::<f64>().ok()?,
            seconds.parse::<f64>().ok()?,
        ),
        _ => return None,
    };
    if !hours.is_finite()
        || !minutes.is_finite()
        || !seconds.is_finite()
        || !(0.0..60.0).contains(&minutes)
        || !(0.0..60.0).contains(&seconds)
    {
        return None;
    }
    Some(sign * (hours * 3600.0 + minutes * 60.0 + seconds) / 86_400.0)
}

pub(super) fn parse_iso_datetime_to_datenum(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S%.f",
        "%m/%d/%Y %H:%M:%S%.f",
    ] {
        if let Ok(value) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Some(crate::builtins::datetime::datenum_from_naive(value));
        }
    }
    for format in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"] {
        if let Ok(date) = NaiveDate::parse_from_str(trimmed, format) {
            return Some(crate::builtins::datetime::datenum_from_naive(
                date.and_time(NaiveTime::MIN),
            ));
        }
    }
    None
}

pub(super) fn unquote(token: &str) -> &str {
    if token.len() >= 2 {
        let bytes = token.as_bytes();
        if (bytes[0] == b'"' && bytes[token.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[token.len() - 1] == b'\'')
        {
            return &token[1..token.len() - 1];
        }
    }
    token
}
