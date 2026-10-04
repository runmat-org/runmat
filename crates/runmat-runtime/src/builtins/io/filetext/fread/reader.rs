use super::*;

pub(super) fn read_from_handle(
    file: &mut File,
    size_spec: &SizeSpec,
    precision: &PrecisionSpec,
    skip: usize,
    machine: MachineFormat,
) -> Result<(FreadEval, bool), String> {
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
            )?;
            let (data, rows, cols) = finalize_numeric(size_spec, count, values);
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
            )?;
            let (row_major, rows, cols) = finalize_char(size_spec, count, values);
            let char_array =
                CharArray::new(row_major, rows, cols).map_err(|e| format!("fread: {e}"))?;
            Ok((
                FreadEval::new(Value::CharArray(char_array), count),
                encountered_eof,
            ))
        }
    }
}

pub(super) fn read_numeric_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
) -> Result<(NumericStorage, usize, bool), String> {
    if let Some(0) = limit {
        return Ok((NumericStorage::zeros(input.numeric_dtype(), 0), 0, false));
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
                Err(err) => {
                    return Err(format!("fread: failed to read from file ({err})"));
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let value = decode_numeric_scalar(&buffer, input, endianness)?;
        values.push(value);
        count += 1;
        if skip > 0 && count.is_multiple_of(repeat) {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|err| format!("fread: failed to skip bytes ({err})"))?;
        }
    }
    Ok((
        numeric_storage_from_scalars(input, values)?,
        count,
        encountered_eof,
    ))
}

pub(super) fn read_char_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
) -> Result<(Vec<char>, usize, bool), String> {
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
                Err(err) => {
                    return Err(format!("fread: failed to read from file ({err})"));
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let ch = decode_to_char(&buffer, input, endianness)?;
        values.push(ch);
        count += 1;
        if skip > 0 && count.is_multiple_of(repeat) {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|err| format!("fread: failed to skip bytes ({err})"))?;
        }
    }

    Ok((values, count, encountered_eof))
}
