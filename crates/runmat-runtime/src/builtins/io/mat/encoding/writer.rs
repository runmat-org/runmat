//! Checked Level-5 MAT encoding into one unpublished byte owner. Nested tags
//! backpatch their lengths in that owner instead of retaining matrix copies.

use super::super::authority::MatFsPath;
use super::super::cooperation::MatCodecCooperation;
use super::super::format::{
    MatArray, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MAT_HEADER_LEN, MI_INT32, MI_INT8, MI_MATRIX,
    MI_UINT16, MI_UINT32, MI_UINT8,
};
use super::{MatEncodingFailure, MatEncodingOwner, MatVar};
use crate::BuiltinResult;
use futures::future::LocalBoxFuture;

const ENCODE_WORK_CHUNK: usize = 64 * 1024;

pub(in crate::builtins::io::mat) async fn write_mat_file(
    path: &MatFsPath,
    vars: &[MatVar],
    owner: MatEncodingOwner,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<()> {
    let bytes = write_mat_bytes_cooperative(vars, owner, Some(cooperation)).await?;
    cooperation
        .await_io(path.write(&bytes))
        .await?
        .map_err(|error| {
            owner.with_source(
                MatEncodingFailure::Io,
                format!("failed to write '{}': {error}", path.path().display()),
                error,
            )
        })
}

pub(in crate::builtins::io::mat) async fn write_mat_bytes_cooperative(
    vars: &[MatVar],
    owner: MatEncodingOwner,
    mut cooperation: Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<u8>> {
    completed(&mut cooperation, 0).await?;
    let plan = plan::plan_workspace(vars, owner, &mut cooperation).await?;
    let mut bytes = Vec::new();
    reserve(&mut bytes, plan.total_bytes, owner)?;
    bytes.resize(MAT_HEADER_LEN, 0);
    let desc = b"MATLAB 5.0 MAT-file, RunMat save";
    bytes[..desc.len()].copy_from_slice(desc);
    bytes[124] = 0x00;
    bytes[125] = 0x01;
    bytes[126] = b'I';
    bytes[127] = b'M';
    for (var, matrix_plan) in vars.iter().zip(&plan.matrices) {
        append_matrix(
            &mut bytes,
            &var.array,
            Some(&var.name),
            matrix_plan,
            owner,
            &mut cooperation,
        )
        .await?;
    }
    if bytes.len() != plan.total_bytes {
        return Err(owner.error(
            MatEncodingFailure::Io,
            "MAT encoded length changed after preflight",
        ));
    }
    completed(&mut cooperation, 0).await?;
    Ok(bytes)
}

#[cfg(test)]
fn write_mat_bytes(vars: &[MatVar]) -> BuiltinResult<Vec<u8>> {
    futures::executor::block_on(write_mat_bytes_cooperative(
        vars,
        MatEncodingOwner::Save,
        None,
    ))
}

mod buffer;
mod plan;
mod validation;
use buffer::{
    append_bytes, append_indices, append_numeric, append_scalars, append_tagged_bytes,
    append_zeroes, begin_tag, completed, end_tag, reserve,
};
use plan::MatrixPlan;

fn append_matrix<'a, 'b>(
    bytes: &'a mut Vec<u8>,
    array: &'a MatArray,
    name: Option<&'a str>,
    plan: &'a MatrixPlan,
    owner: MatEncodingOwner,
    cooperation: &'a mut Option<&mut MatCodecCooperation<'b>>,
) -> LocalBoxFuture<'a, BuiltinResult<()>> {
    Box::pin(async move {
        completed(cooperation, 0).await?;
        let matrix = begin_tag(bytes, MI_MATRIX, owner)?;
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
                    owner.error(
                        MatEncodingFailure::Io,
                        "sparse value count exceeds MAT-file limits",
                    )
                })?,
            ),
            _ => (array.class.class_code(), 0),
        };
        let mut flags = [0u8; 8];
        flags[..4].copy_from_slice(&flags0.to_le_bytes());
        flags[4..].copy_from_slice(&flags1.to_le_bytes());
        append_tagged_bytes(bytes, MI_UINT32, &flags, owner, cooperation).await?;

        let dimensions = begin_tag(bytes, MI_INT32, owner)?;
        for &dimension in &array.dims {
            let dimension = i32::try_from(dimension).map_err(|_| {
                owner.error(
                    MatEncodingFailure::Io,
                    "dimension exceeds MAT-file int32 range",
                )
            })?;
            append_bytes(bytes, &dimension.to_le_bytes(), owner, cooperation).await?;
        }
        end_tag(bytes, dimensions, owner)?;
        append_tagged_bytes(
            bytes,
            MI_INT8,
            name.unwrap_or("").as_bytes(),
            owner,
            cooperation,
        )
        .await?;

        match &array.data {
            MatData::Numeric { real, imag } => {
                if super::tensor_dtype_mat_class(real.numeric_dtype()) != array.class {
                    return Err(owner.error(
                        MatEncodingFailure::Io,
                        "numeric MAT class does not match authoritative storage",
                    ));
                }
                append_numeric(bytes, real, owner, cooperation).await?;
                if let Some(imag) = imag {
                    if imag.numeric_dtype() != real.numeric_dtype() || imag.len() != real.len() {
                        return Err(owner.error(
                            MatEncodingFailure::Io,
                            "complex numeric components must have matching class and length",
                        ));
                    }
                    append_numeric(bytes, imag, owner, cooperation).await?;
                }
            }
            MatData::Logical { data } => {
                append_tagged_bytes(bytes, MI_UINT8, data, owner, cooperation).await?;
            }
            MatData::Char { data } => {
                append_scalars(bytes, MI_UINT16, data, u16::to_le_bytes, owner, cooperation)
                    .await?;
            }
            MatData::Cell { elements } => {
                if elements.len() != plan.children.len() {
                    return Err(owner.error(
                        MatEncodingFailure::Io,
                        "MAT cell layout changed after preflight",
                    ));
                }
                for (element, child_plan) in elements.iter().zip(&plan.children) {
                    append_matrix(bytes, element, None, child_plan, owner, cooperation).await?;
                }
            }
            MatData::Struct {
                field_names,
                field_values,
            }
            | MatData::Object {
                field_names,
                field_values,
                ..
            } => {
                if let MatData::Object { class_name, .. } = &array.data {
                    append_tagged_bytes(bytes, MI_INT8, class_name.as_bytes(), owner, cooperation)
                        .await?;
                }
                if field_values.len() != plan.children.len() {
                    return Err(owner.error(
                        MatEncodingFailure::Io,
                        "MAT struct layout changed after preflight",
                    ));
                }
                let width = plan.field_name_width.ok_or_else(|| {
                    owner.error(
                        MatEncodingFailure::Io,
                        "MAT struct width missing from preflight",
                    )
                })?;
                append_tagged_bytes(bytes, MI_INT32, &width.to_le_bytes(), owner, cooperation)
                    .await?;
                let names = begin_tag(bytes, MI_INT8, owner)?;
                for name in field_names {
                    append_bytes(bytes, name.as_bytes(), owner, cooperation).await?;
                    append_zeroes(bytes, width as usize - name.len(), owner, cooperation).await?;
                }
                end_tag(bytes, names, owner)?;
                for (value, child_plan) in field_values.iter().zip(&plan.children) {
                    append_matrix(bytes, value, None, child_plan, owner, cooperation).await?;
                }
            }
            MatData::Sparse {
                col_ptrs,
                row_indices,
                values,
                ..
            } => {
                append_indices(bytes, row_indices, "sparse row index", owner, cooperation).await?;
                append_indices(bytes, col_ptrs, "sparse column pointer", owner, cooperation)
                    .await?;
                append_numeric(bytes, values, owner, cooperation).await?;
            }
        }
        if bytes.len().checked_sub(matrix + 8) != Some(plan.payload_bytes as usize) {
            return Err(owner.error(
                MatEncodingFailure::Io,
                "MAT encoded matrix disagrees with preflight",
            ));
        }
        end_tag(bytes, matrix, owner)
    })
}

#[cfg(test)]
#[path = "writer/tests.rs"]
mod tests;
