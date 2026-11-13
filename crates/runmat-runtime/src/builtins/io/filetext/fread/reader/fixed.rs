//! Fixed-width native input uses the same claimed streaming/output owners as
//! encoded input, without a NumericScalar payload or conversion tensor.

use super::*;
use scalar::FixedValue;
use workspace::StreamValues;

pub(super) async fn read_fixed_values<'a, T: FixedValue, R: Read + Seek>(
    reader: &mut R,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
    cooperation: &mut super::super::super::cooperation::FileCooperation,
    runtime: Option<&'a crate::context::RuntimeContext>,
) -> BuiltinResult<(StreamValues<'a, T>, usize, bool)> {
    let mut values = StreamValues::new(runtime)?;
    let mut count = 0usize;
    let mut eof = false;
    let width = T::INPUT.byte_len();
    let mut buffer = [0u8; 8];
    'values: while count < limit.unwrap_or(usize::MAX) {
        let mut filled = 0;
        while filled < width {
            match reader.read(&mut buffer[filled..width]) {
                Ok(0) => {
                    eof = true;
                    break 'values;
                }
                Ok(read) => {
                    filled += read;
                    cooperation.completed(read).await?;
                }
                Err(source) if source.kind() == ErrorKind::Interrupted => {
                    cooperation.completed(1).await?;
                }
                Err(source) => {
                    return Err(ReadFailure::Io {
                        operation: "failed to read from file",
                        source,
                    }
                    .into())
                }
            }
        }
        values.push(T::decode(&buffer[..width], endianness).map_err(ReadFailure::Semantic)?)?;
        count += 1;
        if skip != 0 && count.is_multiple_of(repeat) {
            reader
                .seek(SeekFrom::Current(
                    i64::try_from(skip)
                        .map_err(|_| "fread: skip exceeds the file cursor range".to_string())
                        .map_err(ReadFailure::Semantic)?,
                ))
                .map_err(|source| ReadFailure::Io {
                    operation: "failed to skip bytes",
                    source,
                })?;
            cooperation.completed(skip).await?;
        }
    }
    Ok((values, count, eof))
}

pub(super) async fn read_fixed_source<T: FixedValue>(
    file: &mut registry::FileOperation,
    size: &SizeSpec,
    precision: &PrecisionSpec,
    skip: usize,
    endianness: Endianness,
    cooperation: &mut super::super::super::cooperation::FileCooperation,
    runtime: Option<&crate::context::RuntimeContext>,
    projection: presentation::Projection<'_>,
) -> BuiltinResult<(FreadEval, bool)> {
    let (values, count, eof) = read_fixed_values::<T, _>(
        file,
        precision.repeat,
        size.element_limit(),
        skip,
        endianness,
        cooperation,
        runtime,
    )
    .await?;
    let output = output::finish(&values, size, projection, cooperation).await?;
    Ok((FreadEval::new(output, count), eof))
}
