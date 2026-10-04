use super::*;

pub(super) async fn write_csv(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    matrix: &CsvMatrix,
    row_offset: usize,
    col_offset: usize,
) -> BuiltinResult<usize> {
    let mut options = OpenOptions::new();
    options.create(true).write(true).truncate(true);
    let path = selected.path();
    let mut file = await_io(runtime, BUILTIN_NAME, selected.open_with(&options))
        .await?
        .map_err(|err| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_OPEN,
                format!(
                    "csvwrite: unable to open \"{}\" for writing ({err})",
                    path.display()
                ),
                err,
            )
        })?;

    let line_ending = "\n";
    let rows = matrix.rows();
    let cols = matrix.cols();

    let mut bytes_written = 0usize;

    for _ in 0..row_offset {
        selected.check().map_err(|source| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_WRITE,
                "csvwrite: captured filesystem authority was revoked",
                source,
            )
        })?;
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_WRITE,
                format!("csvwrite: failed to write line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
        cooperation.completed(line_ending.len()).await?;
    }

    if rows == 0 || cols == 0 {
        await_io(runtime, BUILTIN_NAME, file.flush_async())
            .await?
            .map_err(|err| {
                csvwrite_error_with_source(
                    &CSVWRITE_ERROR_IO_WRITE,
                    format!("csvwrite: failed to flush output ({err})"),
                    err,
                )
            })?;
        selected.check().map_err(|source| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_WRITE,
                "csvwrite: captured filesystem authority was revoked",
                source,
            )
        })?;
        return Ok(bytes_written);
    }

    for row in 0..rows {
        selected.check().map_err(|source| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_WRITE,
                "csvwrite: captured filesystem authority was revoked",
                source,
            )
        })?;
        let mut fields = Vec::with_capacity(col_offset + cols);
        for _ in 0..col_offset {
            fields.push(String::new());
        }
        for col in 0..cols {
            let idx = row + col * rows;
            fields.push(matrix.format_at(idx));
        }
        let line = fields.join(",");
        if !line.is_empty() {
            file.write_all(line.as_bytes()).map_err(|err| {
                csvwrite_error_with_source(
                    &CSVWRITE_ERROR_IO_WRITE,
                    format!("csvwrite: failed to write value ({err})"),
                    err,
                )
            })?;
            bytes_written += line.len();
        }
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_WRITE,
                format!("csvwrite: failed to write line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
        cooperation
            .completed(line.len() + line_ending.len())
            .await?;
    }

    await_io(runtime, BUILTIN_NAME, file.flush_async())
        .await?
        .map_err(|err| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_WRITE,
                format!("csvwrite: failed to flush output ({err})"),
                err,
            )
        })?;

    selected.check().map_err(|source| {
        csvwrite_error_with_source(
            &CSVWRITE_ERROR_IO_WRITE,
            "csvwrite: captured filesystem authority was revoked",
            source,
        )
    })?;
    Ok(bytes_written)
}

fn format_complex(re: f64, im: f64) -> String {
    let real = format_numeric(re);
    let imag = format_numeric(im.abs());
    if im.is_sign_negative() {
        format!("{real}-{imag}i")
    } else {
        format!("{real}+{imag}i")
    }
}

pub(super) fn format_numeric(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }
    if value == 0.0 {
        return "0".to_string();
    }

    let precision: i32 = 5;
    let abs = value.abs();
    let exp10 = abs.log10().floor() as i32;
    let use_scientific = exp10 < -4 || exp10 >= precision;

    let raw = if use_scientific {
        let digits_after = (precision - 1).max(0) as usize;
        format!("{:.*e}", digits_after, value)
    } else {
        let decimals = (precision - 1 - exp10).max(0) as usize;
        format!("{:.*}", decimals, value)
    };

    let mut trimmed = trim_trailing_zeros(raw);
    if trimmed == "-0" {
        trimmed = "0".to_string();
    }
    trimmed
}

fn format_tensor_value(tensor: &Tensor, idx: usize) -> String {
    let value = tensor
        .numeric_value_at(idx)
        .expect("index within authoritative numeric storage");
    if let Some(value) = value.into_int_value() {
        return value.decimal_string();
    }
    format_numeric(value.materialize_f64())
}

fn trim_trailing_zeros(mut value: String) -> String {
    if let Some(exp_pos) = value.find(['e', 'E']) {
        let exponent = value.split_off(exp_pos);
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
        value.push_str(&normalize_exponent(&exponent));
        value
    } else {
        if value.contains('.') {
            while value.ends_with('0') {
                value.pop();
            }
            if value.ends_with('.') {
                value.pop();
            }
        }
        if value.is_empty() {
            "0".to_string()
        } else {
            value
        }
    }
}

pub(super) fn normalize_exponent(exponent: &str) -> String {
    if exponent.len() <= 1 {
        return exponent.to_string();
    }
    let mut chars = exponent.chars();
    let marker = chars.next().unwrap();
    let rest: String = chars.collect();
    match rest.parse::<i32>() {
        Ok(parsed) => format!("{}{:+03}", marker, parsed),
        Err(_) => exponent.to_string(),
    }
}
