//! Checked Level-5 MAT encoding into one unpublished byte owner. Nested tags
//! backpatch their lengths in that owner instead of retaining matrix copies.

use super::super::super::authority::MatFsPath;
use super::super::super::cooperation::MatCodecCooperation;
use super::super::super::format::{
    MatArray, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MAT_HEADER_LEN, MI_DOUBLE, MI_INT16, MI_INT32,
    MI_INT64, MI_INT8, MI_MATRIX, MI_SINGLE, MI_UINT16, MI_UINT32, MI_UINT64, MI_UINT8,
};
use super::super::{save_error_with, save_error_with_source, SAVE_ERROR_IO};
use super::{tensor_dtype_mat_class, MatVar};
use crate::BuiltinResult;
use futures::future::LocalBoxFuture;
use runmat_value::NumericStorage;

const ENCODE_WORK_CHUNK: usize = 64 * 1024;

pub(in super::super) async fn write_mat_file(
    path: &MatFsPath,
    vars: &[MatVar],
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<()> {
    let bytes = write_mat_bytes_cooperative(vars, Some(cooperation)).await?;
    cooperation
        .await_io(path.write(&bytes))
        .await?
        .map_err(|error| {
            save_error_with_source(
                &SAVE_ERROR_IO,
                format!("save: failed to write '{}': {error}", path.path().display()),
                error,
            )
        })
}

pub(super) async fn write_mat_bytes_cooperative(
    vars: &[MatVar],
    mut cooperation: Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<u8>> {
    completed(&mut cooperation, 0).await?;
    let mut bytes = Vec::new();
    reserve(&mut bytes, MAT_HEADER_LEN)?;
    bytes.resize(MAT_HEADER_LEN, 0);
    let desc = b"MATLAB 5.0 MAT-file, RunMat save";
    bytes[..desc.len()].copy_from_slice(desc);
    bytes[124] = 0x00;
    bytes[125] = 0x01;
    bytes[126] = b'I';
    bytes[127] = b'M';
    for var in vars {
        append_matrix(&mut bytes, &var.array, Some(&var.name), &mut cooperation).await?;
    }
    completed(&mut cooperation, 0).await?;
    Ok(bytes)
}

#[cfg(test)]
fn write_mat_bytes(vars: &[MatVar]) -> BuiltinResult<Vec<u8>> {
    futures::executor::block_on(write_mat_bytes_cooperative(vars, None))
}

async fn completed(
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
    work: usize,
) -> BuiltinResult<()> {
    if let Some(cooperation) = cooperation.as_deref_mut() {
        cooperation.completed(work).await?;
    }
    Ok(())
}

fn reserve(bytes: &mut Vec<u8>, additional: usize) -> BuiltinResult<()> {
    bytes.try_reserve(additional).map_err(|error| {
        save_error_with_source(&SAVE_ERROR_IO, "save: MAT bytes cannot be allocated", error)
    })
}

fn begin_tag(bytes: &mut Vec<u8>, data_type: u32) -> BuiltinResult<usize> {
    reserve(bytes, 8)?;
    let position = bytes.len();
    bytes.extend_from_slice(&data_type.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    Ok(position)
}

fn end_tag(bytes: &mut Vec<u8>, position: usize) -> BuiltinResult<()> {
    let length = bytes
        .len()
        .checked_sub(position + 8)
        .ok_or_else(|| save_error_with(&SAVE_ERROR_IO, "save: invalid MAT tag boundary"))?;
    let length = u32::try_from(length).map_err(|_| {
        save_error_with(
            &SAVE_ERROR_IO,
            "save: MAT subelement exceeds its uint32 length",
        )
    })?;
    bytes[position + 4..position + 8].copy_from_slice(&length.to_le_bytes());
    let padding = (8 - (length as usize % 8)) % 8;
    reserve(bytes, padding)?;
    bytes.resize(bytes.len() + padding, 0);
    Ok(())
}

async fn append_bytes(
    bytes: &mut Vec<u8>,
    source: &[u8],
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    for chunk in source.chunks(ENCODE_WORK_CHUNK) {
        reserve(bytes, chunk.len())?;
        bytes.extend_from_slice(chunk);
        completed(cooperation, chunk.len()).await?;
    }
    Ok(())
}

async fn append_zeroes(
    bytes: &mut Vec<u8>,
    count: usize,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    let mut remaining = count;
    while remaining != 0 {
        let chunk = remaining.min(ENCODE_WORK_CHUNK);
        reserve(bytes, chunk)?;
        bytes.resize(bytes.len() + chunk, 0);
        completed(cooperation, chunk).await?;
        remaining -= chunk;
    }
    Ok(())
}

async fn append_tagged_bytes(
    bytes: &mut Vec<u8>,
    data_type: u32,
    payload: &[u8],
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    let position = begin_tag(bytes, data_type)?;
    append_bytes(bytes, payload, cooperation).await?;
    end_tag(bytes, position)
}

async fn append_scalars<T: Copy, const WIDTH: usize>(
    bytes: &mut Vec<u8>,
    data_type: u32,
    values: &[T],
    encode: impl Fn(T) -> [u8; WIDTH],
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    let position = begin_tag(bytes, data_type)?;
    let count = (ENCODE_WORK_CHUNK / WIDTH).max(1);
    for chunk in values.chunks(count) {
        let length = chunk.len().checked_mul(WIDTH).ok_or_else(|| {
            save_error_with(
                &SAVE_ERROR_IO,
                "save: MAT scalar bytes exceed platform limits",
            )
        })?;
        reserve(bytes, length)?;
        for value in chunk {
            bytes.extend_from_slice(&encode(*value));
        }
        completed(cooperation, length).await?;
    }
    end_tag(bytes, position)
}

async fn append_numeric(
    bytes: &mut Vec<u8>,
    storage: &NumericStorage,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    match storage {
        NumericStorage::F64(values) => {
            append_scalars(bytes, MI_DOUBLE, values, f64::to_le_bytes, cooperation).await
        }
        NumericStorage::F32(values) => {
            append_scalars(bytes, MI_SINGLE, values, f32::to_le_bytes, cooperation).await
        }
        NumericStorage::I8(values) => {
            append_scalars(bytes, MI_INT8, values, |value| [value as u8], cooperation).await
        }
        NumericStorage::U8(values) => {
            append_tagged_bytes(bytes, MI_UINT8, values, cooperation).await
        }
        NumericStorage::I16(values) => {
            append_scalars(bytes, MI_INT16, values, i16::to_le_bytes, cooperation).await
        }
        NumericStorage::U16(values) => {
            append_scalars(bytes, MI_UINT16, values, u16::to_le_bytes, cooperation).await
        }
        NumericStorage::I32(values) => {
            append_scalars(bytes, MI_INT32, values, i32::to_le_bytes, cooperation).await
        }
        NumericStorage::U32(values) => {
            append_scalars(bytes, MI_UINT32, values, u32::to_le_bytes, cooperation).await
        }
        NumericStorage::I64(values) => {
            append_scalars(bytes, MI_INT64, values, i64::to_le_bytes, cooperation).await
        }
        NumericStorage::U64(values) => {
            append_scalars(bytes, MI_UINT64, values, u64::to_le_bytes, cooperation).await
        }
    }
}

async fn append_indices(
    bytes: &mut Vec<u8>,
    values: &[usize],
    label: &str,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    let position = begin_tag(bytes, MI_INT32)?;
    for chunk in values.chunks(ENCODE_WORK_CHUNK / 4) {
        reserve(bytes, chunk.len() * 4)?;
        for &value in chunk {
            let index = i32::try_from(value).map_err(|_| {
                save_error_with(
                    &SAVE_ERROR_IO,
                    format!("save: {label} exceeds MAT-file int32 range"),
                )
            })?;
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        completed(cooperation, chunk.len() * 4).await?;
    }
    end_tag(bytes, position)
}

fn append_matrix<'a, 'b>(
    bytes: &'a mut Vec<u8>,
    array: &'a MatArray,
    name: Option<&'a str>,
    cooperation: &'a mut Option<&mut MatCodecCooperation<'b>>,
) -> LocalBoxFuture<'a, BuiltinResult<()>> {
    Box::pin(async move {
        completed(cooperation, 0).await?;
        let matrix = begin_tag(bytes, MI_MATRIX)?;
        let (flags0, flags1) = match &array.data {
            MatData::Numeric { imag, .. } => (
                array.class.class_code() | if imag.is_some() { FLAG_COMPLEX } else { 0 },
                0,
            ),
            MatData::Logical { .. } => (array.class.class_code() | FLAG_LOGICAL, 0),
            MatData::Sparse {
                values, logical, ..
            } => (
                array.class.class_code() | if *logical { FLAG_LOGICAL } else { 0 },
                u32::try_from(values.len()).map_err(|_| {
                    save_error_with(
                        &SAVE_ERROR_IO,
                        "save: sparse value count exceeds MAT-file limits",
                    )
                })?,
            ),
            _ => (array.class.class_code(), 0),
        };
        let mut flags = [0u8; 8];
        flags[..4].copy_from_slice(&flags0.to_le_bytes());
        flags[4..].copy_from_slice(&flags1.to_le_bytes());
        append_tagged_bytes(bytes, MI_UINT32, &flags, cooperation).await?;

        let dimensions = begin_tag(bytes, MI_INT32)?;
        for &dimension in &array.dims {
            let dimension = i32::try_from(dimension).map_err(|_| {
                save_error_with(
                    &SAVE_ERROR_IO,
                    "save: dimension exceeds MAT-file int32 range",
                )
            })?;
            append_bytes(bytes, &dimension.to_le_bytes(), cooperation).await?;
        }
        end_tag(bytes, dimensions)?;
        append_tagged_bytes(bytes, MI_INT8, name.unwrap_or("").as_bytes(), cooperation).await?;

        match &array.data {
            MatData::Numeric { real, imag } => {
                if super::tensor_dtype_mat_class(real.numeric_dtype()) != array.class {
                    return Err(save_error_with(
                        &SAVE_ERROR_IO,
                        "save: numeric MAT class does not match authoritative storage",
                    ));
                }
                append_numeric(bytes, real, cooperation).await?;
                if let Some(imag) = imag {
                    if imag.numeric_dtype() != real.numeric_dtype() || imag.len() != real.len() {
                        return Err(save_error_with(
                            &SAVE_ERROR_IO,
                            "save: complex numeric components must have matching class and length",
                        ));
                    }
                    append_numeric(bytes, imag, cooperation).await?;
                }
            }
            MatData::Logical { data } => {
                append_tagged_bytes(bytes, MI_UINT8, data, cooperation).await?;
            }
            MatData::Char { data } => {
                append_scalars(bytes, MI_UINT16, data, u16::to_le_bytes, cooperation).await?;
            }
            MatData::Cell { elements } => {
                for element in elements {
                    append_matrix(bytes, element, None, cooperation).await?;
                }
            }
            MatData::Struct {
                field_names,
                field_values,
            } => {
                let width = field_names
                    .iter()
                    .map(String::len)
                    .max()
                    .unwrap_or(0)
                    .max(1);
                let width = i32::try_from(width).map_err(|_| {
                    save_error_with(
                        &SAVE_ERROR_IO,
                        "save: MAT field name width exceeds int32 range",
                    )
                })?;
                append_tagged_bytes(bytes, MI_INT32, &width.to_le_bytes(), cooperation).await?;
                let names = begin_tag(bytes, MI_INT8)?;
                for name in field_names {
                    append_bytes(bytes, name.as_bytes(), cooperation).await?;
                    append_zeroes(bytes, width as usize - name.len(), cooperation).await?;
                }
                end_tag(bytes, names)?;
                for value in field_values {
                    append_matrix(bytes, value, None, cooperation).await?;
                }
            }
            MatData::Sparse {
                col_ptrs,
                row_indices,
                values,
                ..
            } => {
                append_indices(bytes, row_indices, "sparse row index", cooperation).await?;
                append_indices(bytes, col_ptrs, "sparse column pointer", cooperation).await?;
                append_numeric(bytes, values, cooperation).await?;
            }
        }
        end_tag(bytes, matrix)
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::super::format::MatClass;
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_filesystem::FsProvider;
    use std::num::NonZeroU64;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn a_large_dimension_rejects_instead_of_wrapping_its_mat_tag() {
        let vars = [MatVar {
            name: "x".into(),
            array: MatArray {
                class: MatClass::Double,
                dims: vec![i32::MAX as usize + 1, 1],
                data: MatData::Numeric {
                    real: NumericStorage::F64(Vec::new()),
                    imag: None,
                },
            },
        }];
        let error = write_mat_bytes(&vars).expect_err("MAT dimensions are signed 32-bit");
        assert_eq!(error.identifier(), Some("RunMat:save:Io"));
        assert!(error.message().contains("int32 range"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cancelled_encoding_never_publishes_to_captured_owner() {
        let _lock = runmat_filesystem::provider_override_lock();
        let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
            "/mat",
        ));
        let _selected = runmat_filesystem::replace_provider(provider.clone());
        let path = MatFsPath::capture(PathBuf::from("cancelled.mat")).expect("capture owner");
        let cancellation = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            cancellation.token(),
            || Box::pin(async { Err(ExecutionServiceError::Cancelled) }),
        );
        let mut cooperation =
            MatCodecCooperation::with_budget(budget, "save", &runmat_builtins::SAVE_ERROR_IO);
        let vars = [MatVar {
            name: "x".into(),
            array: MatArray {
                class: MatClass::Double,
                dims: vec![1, 1],
                data: MatData::Numeric {
                    real: NumericStorage::F64(vec![42.0]),
                    imag: None,
                },
            },
        }];
        let error = block_on(write_mat_file(&path, &vars, &mut cooperation))
            .expect_err("cancel before publication");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(
            error
                .source
                .as_deref()
                .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
            Some(&ExecutionServiceError::Cancelled)
        );
        let error = block_on(provider.read(path.path())).expect_err("no published MAT file");
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn large_numeric_encoding_cancels_inside_payload_before_write() {
        let _lock = runmat_filesystem::provider_override_lock();
        let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
            "/mat",
        ));
        let _selected = runmat_filesystem::replace_provider(provider.clone());
        let path = MatFsPath::capture(PathBuf::from("large-cancelled.mat"))
            .expect("capture original owner");
        let cancellation = CancellationSource::new();
        let turns = Arc::new(AtomicUsize::new(0));
        let observed = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(ENCODE_WORK_CHUNK as u64).unwrap()),
            cancellation.token(),
            move || {
                observed.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Err(ExecutionServiceError::Cancelled) })
            },
        );
        let mut cooperation =
            MatCodecCooperation::with_budget(budget, "save", &runmat_builtins::SAVE_ERROR_IO);
        let vars = [MatVar {
            name: "large".into(),
            array: MatArray {
                class: MatClass::UInt64,
                dims: vec![1, 20_000],
                data: MatData::Numeric {
                    real: NumericStorage::U64(vec![u64::MAX; 20_000]),
                    imag: None,
                },
            },
        }];
        let error = block_on(write_mat_file(&path, &vars, &mut cooperation))
            .expect_err("service cancels during numeric payload encoding");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(turns.load(Ordering::SeqCst), 1);
        assert_eq!(
            block_on(provider.read(path.path()))
                .expect_err("unpublished MAT bytes")
                .kind(),
            std::io::ErrorKind::NotFound
        );
    }
}
