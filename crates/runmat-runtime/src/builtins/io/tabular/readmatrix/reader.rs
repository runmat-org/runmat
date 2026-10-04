use super::*;

pub(super) async fn read_numeric_matrix(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
    options: &ReadMatrixOptions,
) -> BuiltinResult<Tensor> {
    let bytes = read_file_bytes(runtime, cooperation, path).await?;
    let text = strip_utf8_bom(decode_text_bytes(&bytes, &options.encoding)?);
    let mut data_lines: Vec<(usize, String)> = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line_number = idx + 1;
        let cleaned = line.trim_end_matches('\r');
        if line_number <= options.num_header_lines {
            continue;
        }
        if cleaned.trim().is_empty() {
            continue;
        }
        data_lines.push((line_number, cleaned.to_string()));
        cooperation.completed(cleaned.len()).await?;
        path.check().map_err(|source| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_READ,
                "readmatrix: captured filesystem authority was revoked",
                source,
            )
        })?;
    }

    if data_lines.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| {
            readmatrix_error_with(&READMATRIX_ERROR_TENSOR_BUILD, format!("readmatrix: {e}"))
        });
    }

    if let Some((_, first_line)) = data_lines.first_mut() {
        if first_line.starts_with('\u{FEFF}') {
            let stripped = first_line.trim_start_matches('\u{FEFF}').to_string();
            *first_line = stripped;
        }
    }

    let delimiter = options
        .delimiter
        .clone()
        .or_else(|| detect_delimiter(&data_lines))
        .unwrap_or(Delimiter::Whitespace);
    let integer_dtype = integer_output_dtype(&options.output_template);

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut max_cols = 0usize;

    for (line_number, text) in &data_lines {
        let fields = split_fields(text, &delimiter);
        if fields.is_empty() {
            continue;
        }
        let mut row = Vec::with_capacity(fields.len());
        for (index, field) in fields.iter().enumerate() {
            if let Some(dtype) = integer_dtype {
                parse_integer_numeric_token(field, options, dtype, *line_number, index + 1)?;
            } else {
                validate_numeric_token(field, options, *line_number, index + 1)?;
            }
            row.push(field.clone());
        }
        if row.len() > max_cols {
            max_cols = row.len();
        }
        rows.push(row);
        cooperation.completed(fields.len()).await?;
    }

    if rows.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| {
            readmatrix_error_with(&READMATRIX_ERROR_TENSOR_BUILD, format!("readmatrix: {e}"))
        });
    }

    if let Some(range) = &options.range {
        let (subset_rows, subset_cols) = apply_range(&rows, max_cols, range, String::new());
        rows = subset_rows;
        max_cols = subset_cols;
    }

    if rows.is_empty() || max_cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| {
            readmatrix_error_with(&READMATRIX_ERROR_TENSOR_BUILD, format!("readmatrix: {e}"))
        });
    }
    let row_count = rows.len();
    let element_count = row_count.checked_mul(max_cols).ok_or_else(|| {
        readmatrix_error_with(
            &READMATRIX_ERROR_TENSOR_BUILD,
            "readmatrix: matrix dimensions exceed addressable storage",
        )
    })?;
    if let Some(dtype) = integer_dtype {
        let mut storage = NumericStorage::zeros(dtype, element_count);
        for (row_index, row) in rows.iter().enumerate() {
            for col_index in 0..max_cols {
                let token = row.get(col_index).map(String::as_str).unwrap_or("");
                let value = parse_integer_numeric_token(
                    token,
                    options,
                    dtype,
                    row_index + 1,
                    col_index + 1,
                )?;
                storage
                    .set_value(col_index * row_count + row_index, value)
                    .map_err(|error| {
                        readmatrix_error_with(
                            &READMATRIX_ERROR_TENSOR_BUILD,
                            format!("readmatrix: {error}"),
                        )
                    })?;
            }
            cooperation.completed(max_cols).await?;
        }
        return Tensor::from_numeric_storage(storage, vec![row_count, max_cols]).map_err(|error| {
            readmatrix_error_with(
                &READMATRIX_ERROR_TENSOR_BUILD,
                format!("readmatrix: {error}"),
            )
        });
    }

    let default_fill = options.empty_value_f64();
    let mut data = vec![default_fill; element_count];
    for (row_index, row) in rows.iter().enumerate() {
        for col_index in 0..max_cols {
            let token = row.get(col_index).map(String::as_str).unwrap_or("");
            data[col_index * row_count + row_index] =
                parse_numeric_token(token, options, row_index + 1, col_index + 1)?;
        }
        cooperation.completed(max_cols).await?;
    }
    Tensor::new(data, vec![row_count, max_cols]).map_err(|error| {
        readmatrix_error_with(
            &READMATRIX_ERROR_TENSOR_BUILD,
            format!("readmatrix: {error}"),
        )
    })
}

async fn read_file_bytes(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
) -> BuiltinResult<Vec<u8>> {
    let file = await_io(runtime, BUILTIN_NAME, path.open_read())
        .await?
        .map_err(|err| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_OPEN,
                format!(
                    "readmatrix: unable to read '{}': {err}",
                    path.path().display()
                ),
                err,
            )
        })?;
    cooperation.completed(0).await?;
    path.check().map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    let mut limited = file.take(MAX_READMATRIX_TEXT_BYTES + 1);
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        let count = limited.read(&mut chunk).map_err(|err| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_READ,
                format!(
                    "readmatrix: error reading '{}': {err}",
                    path.path().display()
                ),
                err,
            )
        })?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
        cooperation.completed(count).await?;
        path.check().map_err(|source| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_READ,
                "readmatrix: captured filesystem authority was revoked",
                source,
            )
        })?;
    }
    if bytes.len() as u64 > MAX_READMATRIX_TEXT_BYTES {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_IO_READ,
            format!(
                "readmatrix: file exceeds maximum supported text size of {MAX_READMATRIX_TEXT_BYTES} bytes"
            ),
        ));
    }
    Ok(bytes)
}

fn validate_encoding_label(label: &str) -> BuiltinResult<()> {
    let label = label.trim();
    if label.eq_ignore_ascii_case("auto") || label.eq_ignore_ascii_case("unicode") {
        return Ok(());
    }
    decode_bytes(&[], normalized_encoding_label(label), BUILTIN_NAME)
        .map(|_| ())
        .map_err(|_| {
            readmatrix_error_with(
                &READMATRIX_ERROR_OPTION_VALUE,
                format!("readmatrix: unsupported Encoding '{label}'"),
            )
        })
}

fn decode_text_bytes(bytes: &[u8], encoding: &str) -> BuiltinResult<String> {
    let label = encoding.trim();
    let normalized = normalized_encoding_label(label);
    let chars = if normalized.eq_ignore_ascii_case("auto") {
        if let Some((encoding, bom_len)) = Encoding::for_bom(bytes) {
            decode_bom_bytes(bytes, encoding, bom_len)
        } else {
            decode_bytes(bytes, "utf-8", BUILTIN_NAME)
        }
    } else {
        decode_bytes(bytes, normalized, BUILTIN_NAME)
    }
    .map_err(|err| {
        let message = err.message().to_string();
        readmatrix_error_with_source(&READMATRIX_ERROR_IO_READ, message, err)
    })?;
    Ok(chars.into_iter().collect())
}

fn decode_bom_bytes(
    bytes: &[u8],
    encoding: &'static Encoding,
    bom_len: usize,
) -> BuiltinResult<Vec<char>> {
    let (decoded, had_errors) = encoding.decode_without_bom_handling(&bytes[bom_len..]);
    if had_errors {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: file contains bytes that are invalid for the detected Encoding",
        ));
    }
    Ok(decoded.chars().collect())
}

fn normalized_encoding_label(label: &str) -> &str {
    if label.eq_ignore_ascii_case("unicode") {
        "utf-8"
    } else if label.eq_ignore_ascii_case("default") || label.eq_ignore_ascii_case("native") {
        "system"
    } else {
        label
    }
}

fn strip_utf8_bom(text: String) -> String {
    text.strip_prefix('\u{FEFF}')
        .map(ToString::to_string)
        .unwrap_or(text)
}

fn detect_delimiter(lines: &[(usize, String)]) -> Option<Delimiter> {
    if lines.is_empty() {
        return None;
    }
    let sample: Vec<&str> = lines
        .iter()
        .take(32)
        .map(|(_, line)| line.as_str())
        .collect();
    let candidates = [',', '\t', ';', '|'];
    let mut best: Option<(f64, Delimiter)> = None;

    for candidate in candidates {
        let mut counts = Vec::new();
        for line in &sample {
            if !line.contains(candidate) {
                continue;
            }
            let fields = line.split(candidate).count();
            if fields >= 2 {
                counts.push(fields);
            }
        }
        if counts.is_empty() {
            continue;
        }
        let average = counts.iter().copied().sum::<usize>() as f64 / counts.len() as f64;
        if average < 2.0 {
            continue;
        }
        if best
            .as_ref()
            .map(|(best_avg, _)| average > *best_avg)
            .unwrap_or(true)
        {
            best = Some((average, Delimiter::Char(candidate)));
        }
    }

    if let Some((_, delimiter)) = best {
        return Some(delimiter);
    }

    let mut whitespace_hits = 0usize;
    for line in &sample {
        if line.split_whitespace().count() > 1 {
            whitespace_hits += 1;
        }
    }
    if whitespace_hits > 0 {
        Some(Delimiter::Whitespace)
    } else {
        None
    }
}

fn split_fields(line: &str, delimiter: &Delimiter) -> Vec<String> {
    match delimiter {
        Delimiter::Char(ch) => split_with_char_delim(line, *ch),
        Delimiter::String(pattern) => line.split(pattern).map(|s| s.to_string()).collect(),
        Delimiter::Whitespace => line.split_whitespace().map(|s| s.to_string()).collect(),
    }
}

fn split_with_char_delim(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
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
            fields.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

fn parse_numeric_token(
    token: &str,
    options: &ReadMatrixOptions,
    line_number: usize,
    column_number: usize,
) -> BuiltinResult<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Ok(options.empty_value_f64());
    }
    let unquoted = unquote(trimmed);
    let inner = unquoted.trim();
    if inner.is_empty() {
        return Ok(options.empty_value_f64());
    }
    if options.is_missing_token(inner) {
        return Ok(f64::NAN);
    }
    let normalized = normalize_numeric_token(inner, options);
    if normalized.is_empty() {
        return Ok(options.empty_value_f64());
    }
    let lower = normalized.to_ascii_lowercase();
    if lower == "nan" {
        return Ok(f64::NAN);
    }
    if matches!(lower.as_str(), "inf" | "+inf" | "infinity" | "+infinity") {
        return Ok(f64::INFINITY);
    }
    if matches!(lower.as_str(), "-inf" | "-infinity") {
        return Ok(f64::NEG_INFINITY);
    }
    normalized.parse::<f64>().map_err(|_| {
        readmatrix_error_with(
            &READMATRIX_ERROR_NON_NUMERIC_TOKEN,
            format!(
                "readmatrix: unable to parse numeric value '{}' on line {} column {}",
                inner, line_number, column_number
            ),
        )
    })
}

fn validate_numeric_token(
    token: &str,
    options: &ReadMatrixOptions,
    line_number: usize,
    column_number: usize,
) -> BuiltinResult<()> {
    parse_numeric_token(token, options, line_number, column_number).map(|_| ())
}

fn integer_output_dtype(template: &OutputTemplate) -> Option<NumericDType> {
    let dtype = match template {
        OutputTemplate::Numeric(dtype) => *dtype,
        OutputTemplate::Like(Value::Tensor(tensor)) => tensor.numeric_dtype(),
        OutputTemplate::Like(Value::Int(value)) => {
            IntegerStorage::from_scalar(value.clone()).numeric_dtype()
        }
        OutputTemplate::Like(Value::GpuTensor(handle)) => {
            integer_element_dtype(runmat_accelerate_api::handle_integer_type(handle)?)
        }
        OutputTemplate::Logical | OutputTemplate::Like(_) => return None,
    };
    matches!(
        dtype,
        NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::I64
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32
            | NumericDType::U64
    )
    .then_some(dtype)
}

pub(super) fn parse_integer_numeric_token(
    token: &str,
    options: &ReadMatrixOptions,
    dtype: NumericDType,
    line_number: usize,
    column_number: usize,
) -> BuiltinResult<NumericScalar> {
    let trimmed = token.trim();
    let unquoted = unquote(trimmed);
    let inner = unquoted.trim();
    if inner.is_empty() {
        return Ok(integer_scalar_from_numeric(dtype, options.empty_value()));
    }
    if options.is_missing_token(inner) {
        return Ok(integer_scalar_from_f64(dtype, f64::NAN));
    }
    let normalized = normalize_numeric_token(inner, options);
    if normalized.is_empty() {
        return Ok(integer_scalar_from_numeric(dtype, options.empty_value()));
    }
    let lower = normalized.to_ascii_lowercase();
    if lower == "nan" {
        return Ok(integer_scalar_from_f64(dtype, f64::NAN));
    }
    if matches!(lower.as_str(), "inf" | "+inf" | "infinity" | "+infinity") {
        return Ok(integer_scalar_from_f64(dtype, f64::INFINITY));
    }
    if matches!(lower.as_str(), "-inf" | "-infinity") {
        return Ok(integer_scalar_from_f64(dtype, f64::NEG_INFINITY));
    }
    let invalid = || {
        readmatrix_error_with(
            &READMATRIX_ERROR_NON_NUMERIC_TOKEN,
            format!(
                "readmatrix: unable to parse numeric value '{}' on line {} column {}",
                inner, line_number, column_number
            ),
        )
    };
    if normalized.chars().any(|ch| matches!(ch, '.' | 'e' | 'E')) {
        return normalized
            .parse::<f64>()
            .map(|value| integer_scalar_from_f64(dtype, value))
            .map_err(|_| invalid());
    }
    parse_decimal_integer_saturating(&normalized, dtype).ok_or_else(invalid)
}

fn parse_decimal_integer_saturating(text: &str, dtype: NumericDType) -> Option<NumericScalar> {
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        Some(_) => (false, text),
        None => return None,
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let magnitude = digits.parse::<u128>().ok();
    Some(if negative {
        match magnitude {
            Some(value) if value <= i128::MAX as u128 => {
                integer_scalar_from_signed(dtype, -(value as i128))
            }
            _ => integer_scalar_from_signed(dtype, i128::MIN),
        }
    } else {
        integer_scalar_from_unsigned(dtype, magnitude.unwrap_or(u128::MAX))
    })
}

fn integer_scalar_from_numeric(dtype: NumericDType, value: NumericScalar) -> NumericScalar {
    match value {
        NumericScalar::F64(value) => integer_scalar_from_f64(dtype, value),
        NumericScalar::F32(value) => integer_scalar_from_f64(dtype, f64::from(value)),
        NumericScalar::I8(value) => integer_scalar_from_signed(dtype, i128::from(value)),
        NumericScalar::I16(value) => integer_scalar_from_signed(dtype, i128::from(value)),
        NumericScalar::I32(value) => integer_scalar_from_signed(dtype, i128::from(value)),
        NumericScalar::I64(value) => integer_scalar_from_signed(dtype, i128::from(value)),
        NumericScalar::U8(value) => integer_scalar_from_unsigned(dtype, u128::from(value)),
        NumericScalar::U16(value) => integer_scalar_from_unsigned(dtype, u128::from(value)),
        NumericScalar::U32(value) => integer_scalar_from_unsigned(dtype, u128::from(value)),
        NumericScalar::U64(value) => integer_scalar_from_unsigned(dtype, u128::from(value)),
    }
}

fn integer_scalar_from_signed(dtype: NumericDType, value: i128) -> NumericScalar {
    match dtype {
        NumericDType::I8 | NumericDType::I16 | NumericDType::I32 | NumericDType::I64 => {
            integer_scalar_from_i128(dtype, value)
        }
        NumericDType::U8 | NumericDType::U16 | NumericDType::U32 | NumericDType::U64 => {
            integer_scalar_from_u128(dtype, value.max(0) as u128)
        }
        NumericDType::F64 | NumericDType::F32 => {
            unreachable!("integer conversion requested floating dtype")
        }
    }
}

fn integer_scalar_from_unsigned(dtype: NumericDType, value: u128) -> NumericScalar {
    match dtype {
        NumericDType::I8 | NumericDType::I16 | NumericDType::I32 | NumericDType::I64 => {
            integer_scalar_from_i128(dtype, value.min(i128::MAX as u128) as i128)
        }
        NumericDType::U8 | NumericDType::U16 | NumericDType::U32 | NumericDType::U64 => {
            integer_scalar_from_u128(dtype, value)
        }
        NumericDType::F64 | NumericDType::F32 => {
            unreachable!("integer conversion requested floating dtype")
        }
    }
}

fn integer_scalar_from_i128(dtype: NumericDType, value: i128) -> NumericScalar {
    match dtype {
        NumericDType::I8 => NumericScalar::I8(value.clamp(i8::MIN as i128, i8::MAX as i128) as i8),
        NumericDType::I16 => {
            NumericScalar::I16(value.clamp(i16::MIN as i128, i16::MAX as i128) as i16)
        }
        NumericDType::I32 => {
            NumericScalar::I32(value.clamp(i32::MIN as i128, i32::MAX as i128) as i32)
        }
        NumericDType::I64 => {
            NumericScalar::I64(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
        }
        _ => unreachable!("signed conversion requested non-signed dtype"),
    }
}

fn integer_scalar_from_u128(dtype: NumericDType, value: u128) -> NumericScalar {
    match dtype {
        NumericDType::U8 => NumericScalar::U8(value.min(u8::MAX as u128) as u8),
        NumericDType::U16 => NumericScalar::U16(value.min(u16::MAX as u128) as u16),
        NumericDType::U32 => NumericScalar::U32(value.min(u32::MAX as u128) as u32),
        NumericDType::U64 => NumericScalar::U64(value.min(u64::MAX as u128) as u64),
        _ => unreachable!("unsigned conversion requested non-unsigned dtype"),
    }
}

fn integer_scalar_from_f64(dtype: NumericDType, value: f64) -> NumericScalar {
    if value.is_nan() {
        return match dtype {
            NumericDType::I8 => NumericScalar::I8(0),
            NumericDType::I16 => NumericScalar::I16(0),
            NumericDType::I32 => NumericScalar::I32(0),
            NumericDType::I64 => NumericScalar::I64(0),
            NumericDType::U8 => NumericScalar::U8(0),
            NumericDType::U16 => NumericScalar::U16(0),
            NumericDType::U32 => NumericScalar::U32(0),
            NumericDType::U64 => NumericScalar::U64(0),
            _ => unreachable!("integer conversion requested floating dtype"),
        };
    }
    let rounded = value.round();
    match dtype {
        NumericDType::I8 => NumericScalar::I8(rounded.clamp(i8::MIN as f64, i8::MAX as f64) as i8),
        NumericDType::I16 => {
            NumericScalar::I16(rounded.clamp(i16::MIN as f64, i16::MAX as f64) as i16)
        }
        NumericDType::I32 => {
            NumericScalar::I32(rounded.clamp(i32::MIN as f64, i32::MAX as f64) as i32)
        }
        NumericDType::I64 => {
            NumericScalar::I64(rounded.clamp(i64::MIN as f64, i64::MAX as f64) as i64)
        }
        NumericDType::U8 => NumericScalar::U8(rounded.clamp(0.0, u8::MAX as f64) as u8),
        NumericDType::U16 => NumericScalar::U16(rounded.clamp(0.0, u16::MAX as f64) as u16),
        NumericDType::U32 => NumericScalar::U32(rounded.clamp(0.0, u32::MAX as f64) as u32),
        NumericDType::U64 => NumericScalar::U64(rounded.clamp(0.0, u64::MAX as f64) as u64),
        NumericDType::F64 | NumericDType::F32 => {
            unreachable!("integer conversion requested floating dtype")
        }
    }
}

fn normalize_numeric_token(token: &str, options: &ReadMatrixOptions) -> String {
    let mut text = token.to_string();
    if let Some(thousands) = options.thousands_separator {
        if thousands != options.decimal_separator {
            text = text.chars().filter(|ch| *ch != thousands).collect();
        }
    }
    if options.decimal_separator != '.' {
        text = text.replace(options.decimal_separator, ".");
    }
    text
}

fn unquote(token: &str) -> &str {
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
