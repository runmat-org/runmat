use super::*;

#[derive(Debug)]
pub(super) enum ReadFailure {
    Io {
        operation: &'static str,
        source: std::io::Error,
    },
    Semantic(String),
    Execution(RuntimeError),
}

impl From<String> for ReadFailure {
    fn from(message: String) -> Self {
        Self::Semantic(message)
    }
}

impl std::fmt::Display for ReadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { operation, source } => write!(f, "fread: {operation} ({source})"),
            Self::Semantic(message) => f.write_str(message),
            Self::Execution(error) => std::fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for ReadFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Semantic(_) => None,
            Self::Execution(error) => Some(error),
        }
    }
}

pub(super) async fn read_from_handle<R: Read + Seek>(
    file: &mut R,
    size_spec: &SizeSpec,
    precision: &PrecisionSpec,
    skip: usize,
    machine: MachineFormat,
    cooperation: &mut super::super::cooperation::FileCooperation,
) -> Result<(FreadEval, bool), ReadFailure> {
    let endianness = machine.to_endianness();
    match precision.output {
        OutputKind::Numeric(output_dtype) => {
            let limit = size_spec.element_limit();
            let (values, count, encountered_eof) = read_numeric_values(
                file,
                precision.input,
                precision.repeat,
                limit,
                skip,
                endianness,
                cooperation,
            )
            .await?;
            let (data, rows, cols) =
                finalize_numeric(size_spec, count, values, cooperation).await?;
            let tensor = Tensor::from_numeric_storage(data, vec![rows, cols])
                .map_err(|e| format!("fread: {e}"))?;
            let tensor = if tensor.numeric_dtype() == output_dtype {
                tensor
            } else {
                tensor::coerce_tensor_dtype(tensor, output_dtype)
            };
            Ok((
                FreadEval::new(Value::Tensor(tensor), count),
                encountered_eof,
            ))
        }
        OutputKind::Char => {
            let limit = size_spec.element_limit();
            let (values, count, encountered_eof) = read_char_values(
                file,
                precision.input,
                precision.repeat,
                limit,
                skip,
                endianness,
                cooperation,
            )
            .await?;
            let (row_major, rows, cols) =
                finalize_char(size_spec, count, values, cooperation).await?;
            let char_array =
                CharArray::new(row_major, rows, cols).map_err(|e| format!("fread: {e}"))?;
            Ok((
                FreadEval::new(Value::CharArray(char_array), count),
                encountered_eof,
            ))
        }
    }
}

pub(super) async fn read_numeric_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
    cooperation: &mut super::super::cooperation::FileCooperation,
) -> Result<(NumericStorage, usize, bool), ReadFailure> {
    if let Some(0) = limit {
        return Ok((
            NumericStorage::zeros(NumericDType::from(input.numeric_class()), 0),
            0,
            false,
        ));
    }
    let element_size = input.byte_len();
    let mut buffer = vec![0u8; element_size];
    let mut values = Vec::new();
    let mut count = 0usize;
    let target = limit.unwrap_or(usize::MAX);
    let mut encountered_eof = false;

    'outer: loop {
        if count >= target {
            break;
        }
        let mut remaining = element_size;
        while remaining > 0 {
            match reader.read(&mut buffer[element_size - remaining..element_size]) {
                Ok(0) => {
                    encountered_eof = true;
                    break 'outer;
                }
                Ok(n) => remaining -= n,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(source) => {
                    return Err(ReadFailure::Io {
                        operation: "failed to read from file",
                        source,
                    })
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let value = decode_numeric_scalar(&buffer, input, endianness)?;
        values.push(value);
        count += 1;
        let applied_skip = skip > 0 && count.is_multiple_of(repeat);
        if applied_skip {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|source| ReadFailure::Io {
                    operation: "failed to skip bytes",
                    source,
                })?;
        }
        cooperation
            .completed(element_size.saturating_add(if applied_skip { skip } else { 0 }))
            .await
            .map_err(ReadFailure::Execution)?;
    }
    Ok((
        numeric_storage_from_scalars(input, values, cooperation).await?,
        count,
        encountered_eof,
    ))
}

pub(super) async fn read_char_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
    cooperation: &mut super::super::cooperation::FileCooperation,
) -> Result<(Vec<char>, usize, bool), ReadFailure> {
    if let Some(0) = limit {
        return Ok((Vec::new(), 0, false));
    }
    let element_size = input.byte_len();
    let mut buffer = vec![0u8; element_size];
    let mut values = Vec::new();
    let mut count = 0usize;
    let target = limit.unwrap_or(usize::MAX);
    let mut encountered_eof = false;

    'outer: loop {
        if count >= target {
            break;
        }
        let mut remaining = element_size;
        while remaining > 0 {
            match reader.read(&mut buffer[element_size - remaining..element_size]) {
                Ok(0) => {
                    encountered_eof = true;
                    break 'outer;
                }
                Ok(n) => remaining -= n,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(source) => {
                    return Err(ReadFailure::Io {
                        operation: "failed to read from file",
                        source,
                    })
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let ch = decode_to_char(&buffer, input, endianness)?;
        values.push(ch);
        count += 1;
        let applied_skip = skip > 0 && count.is_multiple_of(repeat);
        if applied_skip {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|source| ReadFailure::Io {
                    operation: "failed to skip bytes",
                    source,
                })?;
        }
        cooperation
            .completed(element_size.saturating_add(if applied_skip { skip } else { 0 }))
            .await
            .map_err(ReadFailure::Execution)?;
    }

    Ok((values, count, encountered_eof))
}
