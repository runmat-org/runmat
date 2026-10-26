use super::*;

mod encoded;
mod fixed;
mod output;
pub(super) mod presentation;
mod scalar;
mod workspace;

use encoded::read_encoded_source;
use fixed::read_fixed_source;
use presentation::Projection;

#[derive(Debug)]
pub(super) enum ReadFailure {
    Io {
        operation: &'static str,
        source: std::io::Error,
    },
    Semantic(String),
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
        }
    }
}

impl std::error::Error for ReadFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Semantic(_) => None,
        }
    }
}

impl From<ReadFailure> for RuntimeError {
    fn from(error: ReadFailure) -> Self {
        let detail = error.to_string();
        let detail = detail.strip_prefix("fread: ").unwrap_or(&detail);
        let mut builder = build_runtime_error(format!("{}: {}", FREAD_ERROR_IO.message, detail))
            .with_builtin(BUILTIN_NAME)
            .with_source(error);
        if let Some(identifier) = FREAD_ERROR_IO.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

pub(super) async fn read_from_handle(
    file: &mut registry::FileOperation,
    size_spec: &SizeSpec,
    precision: &PrecisionSpec,
    skip: usize,
    machine: MachineFormat,
    cooperation: &mut super::super::cooperation::FileCooperation,
    runtime: Option<&crate::context::RuntimeContext>,
    projection: Projection<'_>,
) -> BuiltinResult<(FreadEval, bool)> {
    let endianness = machine.to_endianness();
    if matches!(precision.input, InputSource::EncodedCharacter) {
        return read_encoded_source(
            file,
            size_spec,
            precision,
            skip,
            cooperation,
            runtime,
            projection,
        )
        .await;
    }
    let InputSource::Numeric(input) = precision.input else {
        unreachable!("encoded source selected above")
    };
    macro_rules! read {
        ($native:ty) => {
            read_fixed_source::<$native>(
                file,
                size_spec,
                precision,
                skip,
                endianness,
                cooperation,
                runtime,
                projection,
            )
            .await
        };
    }
    match input {
        InputType::UInt8 => read!(u8),
        InputType::Int8 => read!(i8),
        InputType::UInt16 => read!(u16),
        InputType::Int16 => read!(i16),
        InputType::UInt32 => read!(u32),
        InputType::Int32 => read!(i32),
        InputType::UInt64 => read!(u64),
        InputType::Int64 => read!(i64),
        InputType::Float32 => read!(f32),
        InputType::Float64 => read!(f64),
    }
}

#[cfg(test)]
pub(super) async fn read_numeric_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
    cooperation: &mut super::super::cooperation::FileCooperation,
) -> BuiltinResult<(NumericStorage, usize, bool)> {
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
                    }
                    .into())
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let value =
            decode_numeric_scalar(&buffer, input, endianness).map_err(ReadFailure::Semantic)?;
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
            .await?;
    }
    Ok((
        numeric_storage_from_scalars(input, values, cooperation).await?,
        count,
        encountered_eof,
    ))
}
