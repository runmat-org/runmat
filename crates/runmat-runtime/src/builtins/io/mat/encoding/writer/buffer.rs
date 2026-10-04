//! Bounded byte/tag primitives for the single MAT output builder.

use super::super::super::cooperation::MatCodecCooperation;
use super::super::super::format::{
    MI_DOUBLE, MI_INT16, MI_INT32, MI_INT64, MI_INT8, MI_SINGLE, MI_UINT16, MI_UINT32, MI_UINT64,
    MI_UINT8,
};
use super::super::{MatEncodingFailure, MatEncodingOwner};
use super::ENCODE_WORK_CHUNK;
use crate::BuiltinResult;
use runmat_value::NumericStorage;

pub(super) async fn completed(
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
    work: usize,
) -> BuiltinResult<()> {
    if let Some(cooperation) = cooperation.as_deref_mut() {
        cooperation.completed(work).await?;
    }
    Ok(())
}

pub(super) fn reserve(
    bytes: &mut Vec<u8>,
    additional: usize,
    owner: MatEncodingOwner,
) -> BuiltinResult<()> {
    bytes.try_reserve(additional).map_err(|error| {
        owner.with_source(
            MatEncodingFailure::Io,
            "MAT bytes cannot be allocated",
            error,
        )
    })
}

pub(super) fn begin_tag(
    bytes: &mut Vec<u8>,
    data_type: u32,
    owner: MatEncodingOwner,
) -> BuiltinResult<usize> {
    reserve(bytes, 8, owner)?;
    let position = bytes.len();
    bytes.extend_from_slice(&data_type.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    Ok(position)
}

pub(super) fn end_tag(
    bytes: &mut Vec<u8>,
    position: usize,
    owner: MatEncodingOwner,
) -> BuiltinResult<()> {
    let length = bytes
        .len()
        .checked_sub(position + 8)
        .ok_or_else(|| owner.error(MatEncodingFailure::Io, "invalid MAT tag boundary"))?;
    let length = u32::try_from(length).map_err(|_| {
        owner.error(
            MatEncodingFailure::Io,
            "MAT subelement exceeds its uint32 length",
        )
    })?;
    bytes[position + 4..position + 8].copy_from_slice(&length.to_le_bytes());
    let padding = (8 - (length as usize % 8)) % 8;
    reserve(bytes, padding, owner)?;
    bytes.resize(bytes.len() + padding, 0);
    Ok(())
}

pub(super) async fn append_bytes(
    bytes: &mut Vec<u8>,
    source: &[u8],
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    for chunk in source.chunks(ENCODE_WORK_CHUNK) {
        reserve(bytes, chunk.len(), owner)?;
        bytes.extend_from_slice(chunk);
        completed(cooperation, chunk.len()).await?;
    }
    Ok(())
}

pub(super) async fn append_zeroes(
    bytes: &mut Vec<u8>,
    count: usize,
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    let mut remaining = count;
    while remaining != 0 {
        let chunk = remaining.min(ENCODE_WORK_CHUNK);
        reserve(bytes, chunk, owner)?;
        bytes.resize(bytes.len() + chunk, 0);
        completed(cooperation, chunk).await?;
        remaining -= chunk;
    }
    Ok(())
}

pub(super) async fn append_tagged_bytes(
    bytes: &mut Vec<u8>,
    data_type: u32,
    payload: &[u8],
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    u32::try_from(payload.len()).map_err(|_| {
        owner.error(
            MatEncodingFailure::Io,
            "MAT subelement exceeds its uint32 length",
        )
    })?;
    let position = begin_tag(bytes, data_type, owner)?;
    append_bytes(bytes, payload, owner, cooperation).await?;
    end_tag(bytes, position, owner)
}

pub(super) async fn append_scalars<T: Copy, const WIDTH: usize>(
    bytes: &mut Vec<u8>,
    data_type: u32,
    values: &[T],
    encode: impl Fn(T) -> [u8; WIDTH],
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    values
        .len()
        .checked_mul(WIDTH)
        .and_then(|length| u32::try_from(length).ok())
        .ok_or_else(|| {
            owner.error(
                MatEncodingFailure::Io,
                "MAT scalar payload exceeds uint32 length",
            )
        })?;
    let position = begin_tag(bytes, data_type, owner)?;
    let count = (ENCODE_WORK_CHUNK / WIDTH).max(1);
    for chunk in values.chunks(count) {
        let length = chunk.len().checked_mul(WIDTH).ok_or_else(|| {
            owner.error(
                MatEncodingFailure::Io,
                "MAT scalar bytes exceed platform limits",
            )
        })?;
        reserve(bytes, length, owner)?;
        for value in chunk {
            bytes.extend_from_slice(&encode(*value));
        }
        completed(cooperation, length).await?;
    }
    end_tag(bytes, position, owner)
}

pub(super) async fn append_numeric(
    bytes: &mut Vec<u8>,
    storage: &NumericStorage,
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    match storage {
        NumericStorage::F64(values) => {
            append_scalars(
                bytes,
                MI_DOUBLE,
                values,
                f64::to_le_bytes,
                owner,
                cooperation,
            )
            .await
        }
        NumericStorage::F32(values) => {
            append_scalars(
                bytes,
                MI_SINGLE,
                values,
                f32::to_le_bytes,
                owner,
                cooperation,
            )
            .await
        }
        NumericStorage::I8(values) => {
            append_scalars(
                bytes,
                MI_INT8,
                values,
                |value| [value as u8],
                owner,
                cooperation,
            )
            .await
        }
        NumericStorage::U8(values) => {
            append_tagged_bytes(bytes, MI_UINT8, values, owner, cooperation).await
        }
        NumericStorage::I16(values) => {
            append_scalars(
                bytes,
                MI_INT16,
                values,
                i16::to_le_bytes,
                owner,
                cooperation,
            )
            .await
        }
        NumericStorage::U16(values) => {
            append_scalars(
                bytes,
                MI_UINT16,
                values,
                u16::to_le_bytes,
                owner,
                cooperation,
            )
            .await
        }
        NumericStorage::I32(values) => {
            append_scalars(
                bytes,
                MI_INT32,
                values,
                i32::to_le_bytes,
                owner,
                cooperation,
            )
            .await
        }
        NumericStorage::U32(values) => {
            append_scalars(
                bytes,
                MI_UINT32,
                values,
                u32::to_le_bytes,
                owner,
                cooperation,
            )
            .await
        }
        NumericStorage::I64(values) => {
            append_scalars(
                bytes,
                MI_INT64,
                values,
                i64::to_le_bytes,
                owner,
                cooperation,
            )
            .await
        }
        NumericStorage::U64(values) => {
            append_scalars(
                bytes,
                MI_UINT64,
                values,
                u64::to_le_bytes,
                owner,
                cooperation,
            )
            .await
        }
    }
}

pub(super) async fn append_indices(
    bytes: &mut Vec<u8>,
    values: &[usize],
    label: &str,
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    values
        .len()
        .checked_mul(4)
        .and_then(|length| u32::try_from(length).ok())
        .ok_or_else(|| {
            owner.error(
                MatEncodingFailure::Io,
                "MAT index payload exceeds uint32 length",
            )
        })?;
    let position = begin_tag(bytes, MI_INT32, owner)?;
    for chunk in values.chunks(ENCODE_WORK_CHUNK / 4) {
        reserve(bytes, chunk.len() * 4, owner)?;
        for &value in chunk {
            let index = i32::try_from(value).map_err(|_| {
                owner.error(
                    MatEncodingFailure::Io,
                    format!("{label} exceeds MAT-file int32 range"),
                )
            })?;
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        completed(cooperation, chunk.len() * 4).await?;
    }
    end_tag(bytes, position, owner)
}
