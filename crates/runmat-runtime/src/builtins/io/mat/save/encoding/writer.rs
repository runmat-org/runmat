//! Serialized Level-5 subelements and filesystem publication.

use super::super::super::authority::MatFsPath;
use super::super::super::cooperation::MatCodecCooperation;
use super::super::super::format::{
    MatArray, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MAT_HEADER_LEN, MI_DOUBLE, MI_INT16, MI_INT32,
    MI_INT64, MI_INT8, MI_MATRIX, MI_SINGLE, MI_UINT16, MI_UINT32, MI_UINT64, MI_UINT8,
};
use super::super::{save_error_with, save_error_with_source, SAVE_ERROR_IO};
use super::{tensor_dtype_mat_class, MatVar};
use crate::BuiltinResult;
use runmat_value::NumericStorage;
use std::io::{BufWriter, Cursor, Write};

pub(in super::super) async fn write_mat_file(
    path: &MatFsPath,
    vars: &[MatVar],
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<()> {
    cooperation.completed(0).await?;
    let bytes = write_mat_bytes(vars)?;
    cooperation.completed(bytes.len()).await?;
    cooperation
        .await_io(path.write(&bytes))
        .await?
        .map_err(|e| {
            save_error_with_source(
                &SAVE_ERROR_IO,
                format!("save: failed to write '{}': {e}", path.path().display()),
                e,
            )
        })
}

pub(super) fn write_mat_bytes(vars: &[MatVar]) -> BuiltinResult<Vec<u8>> {
    let mut writer = BufWriter::new(Cursor::new(Vec::<u8>::new()));

    let mut header = [0u8; MAT_HEADER_LEN];
    let desc = b"MATLAB 5.0 MAT-file, RunMat save";
    for (i, byte) in desc.iter().enumerate() {
        header[i] = *byte;
    }
    header[124] = 0x00;
    header[125] = 0x01;
    header[126] = b'I';
    header[127] = b'M';
    writer.write_all(&header).map_err(|e| {
        save_error_with_source(
            &SAVE_ERROR_IO,
            format!("save: failed to write header: {e}"),
            e,
        )
    })?;

    for var in vars {
        let matrix_bytes = build_matrix_bytes(&var.array, Some(&var.name))?;
        write_tagged(&mut writer, MI_MATRIX, &matrix_bytes)?;
    }

    writer.flush().map_err(|e| {
        save_error_with_source(&SAVE_ERROR_IO, format!("save: flush failed: {e}"), e)
    })?;
    Ok(writer
        .into_inner()
        .map_err(|e| {
            save_error_with_source(&SAVE_ERROR_IO, "save: failed to finalize MAT bytes", e)
        })?
        .into_inner())
}

fn build_matrix_bytes(array: &MatArray, name: Option<&str>) -> BuiltinResult<Vec<u8>> {
    let mut buf = Vec::new();

    let (flags0, flags1) = match &array.data {
        MatData::Numeric { imag, .. } => {
            let mut f0 = array.class.class_code();
            if imag.is_some() {
                f0 |= FLAG_COMPLEX;
            }
            (f0, 0u32)
        }
        MatData::Logical { .. } => ((array.class.class_code()) | FLAG_LOGICAL, 0u32),
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
        _ => (array.class.class_code(), 0u32),
    };

    let mut flags = Vec::with_capacity(8);
    flags.extend_from_slice(&flags0.to_le_bytes());
    flags.extend_from_slice(&flags1.to_le_bytes());
    write_subelement(&mut buf, MI_UINT32, &flags)?;

    let dims_len = array.dims.len().checked_mul(4).ok_or_else(|| {
        save_error_with(
            &SAVE_ERROR_IO,
            "save: MAT dimensions exceed platform limits",
        )
    })?;
    let mut dims_bytes = Vec::new();
    dims_bytes.try_reserve_exact(dims_len).map_err(|error| {
        save_error_with_source(
            &SAVE_ERROR_IO,
            "save: MAT dimensions cannot be allocated",
            error,
        )
    })?;
    for &dim in &array.dims {
        let dim = i32::try_from(dim).map_err(|_| {
            save_error_with(
                &SAVE_ERROR_IO,
                "save: dimension exceeds MAT-file int32 range",
            )
        })?;
        dims_bytes.extend_from_slice(&dim.to_le_bytes());
    }
    write_subelement(&mut buf, MI_INT32, &dims_bytes)?;

    let name_bytes = name.unwrap_or("").as_bytes();
    write_subelement(&mut buf, MI_INT8, name_bytes)?;

    match &array.data {
        MatData::Numeric { real, imag } => {
            if tensor_dtype_mat_class(real.numeric_dtype()) != array.class {
                return Err(save_error_with(
                    &SAVE_ERROR_IO,
                    "save: numeric MAT class does not match authoritative storage",
                ));
            }
            let (data_type, real_bytes) = encode_numeric_storage(real)?;
            write_subelement(&mut buf, data_type, &real_bytes)?;
            if let Some(imag) = imag {
                if imag.numeric_dtype() != real.numeric_dtype() || imag.len() != real.len() {
                    return Err(save_error_with(
                        &SAVE_ERROR_IO,
                        "save: complex numeric components must have matching class and length",
                    ));
                }
                let (imag_type, imag_bytes) = encode_numeric_storage(imag)?;
                write_subelement(&mut buf, imag_type, &imag_bytes)?;
            }
        }
        MatData::Logical { data } => {
            write_subelement(&mut buf, MI_UINT8, data)?;
        }
        MatData::Char { data } => {
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for code in data {
                bytes.extend_from_slice(&code.to_le_bytes());
            }
            write_subelement(&mut buf, MI_UINT16, &bytes)?;
        }
        MatData::Cell { elements } => {
            for elem in elements {
                let elem_bytes = build_matrix_bytes(elem, None)?;
                write_subelement(&mut buf, MI_MATRIX, &elem_bytes)?;
            }
        }
        MatData::Struct {
            field_names,
            field_values,
        } => {
            let max_len = field_names
                .iter()
                .map(|n| n.len())
                .max()
                .unwrap_or(0)
                .max(1);
            let field_width = i32::try_from(max_len).map_err(|_| {
                save_error_with(
                    &SAVE_ERROR_IO,
                    "save: MAT field name width exceeds int32 range",
                )
            })?;
            let len_bytes = field_width.to_le_bytes();
            write_subelement(&mut buf, MI_INT32, &len_bytes)?;
            let names_len = max_len.checked_mul(field_names.len()).ok_or_else(|| {
                save_error_with(
                    &SAVE_ERROR_IO,
                    "save: MAT field name table exceeds platform limits",
                )
            })?;
            let mut names_bytes = Vec::new();
            names_bytes.try_reserve_exact(names_len).map_err(|error| {
                save_error_with_source(
                    &SAVE_ERROR_IO,
                    "save: MAT field names cannot be allocated",
                    error,
                )
            })?;
            for name in field_names {
                let bytes = name.as_bytes();
                for i in 0..max_len {
                    let b = if i < bytes.len() { bytes[i] } else { 0 };
                    names_bytes.push(b);
                }
            }
            write_subelement(&mut buf, MI_INT8, &names_bytes)?;
            for value in field_values {
                let value_bytes = build_matrix_bytes(value, None)?;
                write_subelement(&mut buf, MI_MATRIX, &value_bytes)?;
            }
        }
        MatData::Sparse {
            col_ptrs,
            row_indices,
            values,
            ..
        } => {
            let ir_bytes = encode_usize_i32_payload(row_indices, "sparse row index")?;
            write_subelement(&mut buf, MI_INT32, &ir_bytes)?;
            let jc_bytes = encode_usize_i32_payload(col_ptrs, "sparse column pointer")?;
            write_subelement(&mut buf, MI_INT32, &jc_bytes)?;
            let (value_type, value_bytes) = encode_numeric_storage(values)?;
            write_subelement(&mut buf, value_type, &value_bytes)?;
        }
    }

    Ok(buf)
}

fn encode_numeric_storage(storage: &NumericStorage) -> BuiltinResult<(u32, Vec<u8>)> {
    macro_rules! encode {
        ($values:expr, $data_type:expr) => {{
            let mut bytes = Vec::with_capacity(std::mem::size_of_val($values.as_slice()));
            for value in $values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok(($data_type, bytes))
        }};
    }

    match storage {
        NumericStorage::F64(values) => {
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok((MI_DOUBLE, bytes))
        }
        NumericStorage::F32(values) => {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok((MI_SINGLE, bytes))
        }
        NumericStorage::I8(values) => {
            Ok((MI_INT8, values.iter().map(|value| *value as u8).collect()))
        }
        NumericStorage::U8(values) => Ok((MI_UINT8, values.clone())),
        NumericStorage::I16(values) => encode!(values, MI_INT16),
        NumericStorage::U16(values) => encode!(values, MI_UINT16),
        NumericStorage::I32(values) => encode!(values, MI_INT32),
        NumericStorage::U32(values) => encode!(values, MI_UINT32),
        NumericStorage::I64(values) => encode!(values, MI_INT64),
        NumericStorage::U64(values) => encode!(values, MI_UINT64),
    }
}

fn encode_usize_i32_payload(values: &[usize], label: &str) -> BuiltinResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        let converted = i32::try_from(*value).map_err(|_| {
            save_error_with(
                &SAVE_ERROR_IO,
                format!("save: {label} exceeds MAT-file int32 range"),
            )
        })?;
        bytes.extend_from_slice(&converted.to_le_bytes());
    }
    Ok(bytes)
}

fn write_tagged<W: Write>(writer: &mut W, data_type: u32, data: &[u8]) -> BuiltinResult<()> {
    if data.len() > u32::MAX as usize {
        return Err(save_error_with(
            &SAVE_ERROR_IO,
            "save: data too large for MAT-file",
        ));
    }
    writer.write_all(&data_type.to_le_bytes()).map_err(|e| {
        save_error_with_source(&SAVE_ERROR_IO, format!("save: write failed: {e}"), e)
    })?;
    writer
        .write_all(&(data.len() as u32).to_le_bytes())
        .map_err(|e| {
            save_error_with_source(&SAVE_ERROR_IO, format!("save: write failed: {e}"), e)
        })?;
    writer.write_all(data).map_err(|e| {
        save_error_with_source(&SAVE_ERROR_IO, format!("save: write failed: {e}"), e)
    })?;
    let padding = (8 - (data.len() % 8)) % 8;
    if padding != 0 {
        let pad = [0u8; 8];
        writer.write_all(&pad[..padding]).map_err(|e| {
            save_error_with_source(&SAVE_ERROR_IO, format!("save: write failed: {e}"), e)
        })?;
    }
    Ok(())
}

fn write_subelement(buf: &mut Vec<u8>, data_type: u32, data: &[u8]) -> BuiltinResult<()> {
    let length = u32::try_from(data.len()).map_err(|_| {
        save_error_with(
            &SAVE_ERROR_IO,
            "save: MAT subelement exceeds its uint32 length",
        )
    })?;
    let padding = (8 - (data.len() % 8)) % 8;
    let additional = 8usize
        .checked_add(data.len())
        .and_then(|size| size.checked_add(padding))
        .ok_or_else(|| save_error_with(&SAVE_ERROR_IO, "save: MAT bytes exceed platform limits"))?;
    buf.try_reserve(additional).map_err(|error| {
        save_error_with_source(&SAVE_ERROR_IO, "save: MAT bytes cannot be allocated", error)
    })?;
    buf.extend_from_slice(&data_type.to_le_bytes());
    buf.extend_from_slice(&length.to_le_bytes());
    buf.extend_from_slice(data);
    if padding != 0 {
        buf.extend(std::iter::repeat_n(0u8, padding));
    }
    Ok(())
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
    use std::sync::Arc;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn a_large_dimension_rejects_instead_of_wrapping_its_mat_tag() {
        let array = MatArray {
            class: MatClass::Double,
            dims: vec![i32::MAX as usize + 1, 1],
            data: MatData::Numeric {
                real: NumericStorage::F64(Vec::new()),
                imag: None,
            },
        };
        let error =
            build_matrix_bytes(&array, Some("x")).expect_err("MAT dimensions are signed 32-bit");
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
}
