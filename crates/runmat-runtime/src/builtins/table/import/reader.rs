use super::*;

pub(in crate::builtins::table) fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
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

pub(in crate::builtins::table) async fn read_table_from_file(
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

pub(in crate::builtins::table) async fn read_cell_from_file(
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

pub(in crate::builtins::table) async fn read_text_table(
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

pub(in crate::builtins::table) async fn read_text_cells(
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

pub(in crate::builtins::table) async fn read_spreadsheet_table(
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

pub(in crate::builtins::table) async fn read_spreadsheet_cells(
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

pub(in crate::builtins::table) async fn read_file_bytes(path: &Path) -> BuiltinResult<Vec<u8>> {
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

pub(in crate::builtins::table) fn is_spreadsheet_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref(),
        Some("xls") | Some("xlsx") | Some("xlsm") | Some("xlsb") | Some("ods")
    )
}

pub(in crate::builtins::table) fn validate_encoding_label(label: &str) -> BuiltinResult<()> {
    encoding_for_label(label)
        .map(|_| ())
        .ok_or_else(|| invalid_argument(format!("readtable: unsupported Encoding '{label}'")))
}

pub(in crate::builtins::table) fn encoding_for_label(label: &str) -> Option<&'static Encoding> {
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

pub(in crate::builtins::table) fn decode_text_bytes(
    bytes: &[u8],
    encoding: &str,
) -> BuiltinResult<String> {
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

pub(in crate::builtins::table) fn strip_utf8_bom(text: String) -> String {
    text.strip_prefix('\u{FEFF}')
        .map(ToString::to_string)
        .unwrap_or(text)
}
