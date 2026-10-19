//! One checked post-order layout plan for the exact recursive MAT writer.
//! No encoded byte owner exists until every enclosing tag fits its u32 ABI.

use super::super::super::cooperation::MatCodecCooperation;
use super::super::super::format::{MatArray, MatData, MAT_HEADER_LEN};
use super::super::{MatEncodingFailure, MatEncodingOwner, MatVar};
use super::validation::validate_matrix;
use crate::BuiltinResult;
use futures::future::LocalBoxFuture;

pub(super) struct WorkspacePlan {
    pub(super) matrices: Vec<MatrixPlan>,
    pub(super) total_bytes: usize,
}

pub(super) struct MatrixPlan {
    pub(super) payload_bytes: u32,
    pub(super) children: Vec<MatrixPlan>,
    pub(super) field_name_width: Option<i32>,
    total_bytes: usize,
}

fn overflow(owner: MatEncodingOwner) -> crate::RuntimeError {
    owner.error(
        MatEncodingFailure::Io,
        "MAT subelement exceeds its uint32 length",
    )
}

fn padded(length: usize) -> usize {
    (8 - length % 8) % 8
}

fn tagged_length(payload: usize, owner: MatEncodingOwner) -> BuiltinResult<usize> {
    u32::try_from(payload).map_err(|_| overflow(owner))?;
    payload
        .checked_add(padded(payload))
        .and_then(|length| length.checked_add(8))
        .ok_or_else(|| overflow(owner))
}

struct MatrixPayload {
    bytes: usize,
}

impl MatrixPayload {
    fn new() -> Self {
        Self { bytes: 0 }
    }

    fn add(&mut self, part: usize, owner: MatEncodingOwner) -> BuiltinResult<()> {
        self.bytes = self
            .bytes
            .checked_add(part)
            .ok_or_else(|| overflow(owner))?;
        if self.bytes > u32::MAX as usize {
            return Err(overflow(owner));
        }
        Ok(())
    }

    fn finish(self, owner: MatEncodingOwner) -> BuiltinResult<(u32, usize)> {
        Ok((self.bytes as u32, tagged_length(self.bytes, owner)?))
    }
}

fn numeric_length(
    storage: &runmat_value::NumericStorage,
    owner: MatEncodingOwner,
) -> BuiltinResult<usize> {
    let payload = storage.checked_byte_len().ok_or_else(|| overflow(owner))?;
    tagged_length(payload, owner)
}

pub(super) async fn plan_workspace(
    vars: &[MatVar],
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<WorkspacePlan> {
    let mut matrices = Vec::new();
    matrices.try_reserve(vars.len()).map_err(|error| {
        owner.with_source(
            MatEncodingFailure::Io,
            "MAT layout plan cannot be allocated",
            error,
        )
    })?;
    let mut total_bytes = MAT_HEADER_LEN;
    for var in vars {
        super::completed(cooperation, 0).await?;
        let matrix = plan_matrix(&var.array, Some(&var.name), owner, cooperation).await?;
        total_bytes = total_bytes
            .checked_add(matrix.total_bytes)
            .ok_or_else(|| overflow(owner))?;
        matrices.push(matrix);
    }
    Ok(WorkspacePlan {
        matrices,
        total_bytes,
    })
}

fn plan_matrix<'a, 'b>(
    array: &'a MatArray,
    name: Option<&'a str>,
    owner: MatEncodingOwner,
    cooperation: &'a mut Option<&mut MatCodecCooperation<'b>>,
) -> LocalBoxFuture<'a, BuiltinResult<MatrixPlan>> {
    Box::pin(async move {
        super::completed(cooperation, 1).await?;
        validate_matrix(array, owner, cooperation).await?;
        let mut payload = MatrixPayload::new();
        payload.add(tagged_length(8, owner)?, owner)?; // array flags
        payload.add(
            tagged_length(
                array
                    .dims
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| overflow(owner))?,
                owner,
            )?,
            owner,
        )?;
        payload.add(tagged_length(name.unwrap_or("").len(), owner)?, owner)?;
        let mut children = Vec::new();
        let mut field_name_width = None;
        match &array.data {
            MatData::Numeric { real, imag } => {
                payload.add(numeric_length(real, owner)?, owner)?;
                if let Some(imag) = imag {
                    payload.add(numeric_length(imag, owner)?, owner)?;
                }
            }
            MatData::Logical { data } => payload.add(tagged_length(data.len(), owner)?, owner)?,
            MatData::Char { data } => {
                payload.add(
                    tagged_length(
                        data.len().checked_mul(2).ok_or_else(|| overflow(owner))?,
                        owner,
                    )?,
                    owner,
                )?;
            }
            MatData::Cell { elements } => {
                children.try_reserve(elements.len()).map_err(|error| {
                    owner.with_source(
                        MatEncodingFailure::Io,
                        "MAT layout plan cannot be allocated",
                        error,
                    )
                })?;
                for element in elements {
                    let child = plan_matrix(element, None, owner, cooperation).await?;
                    payload.add(child.total_bytes, owner)?;
                    children.push(child);
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
                    payload.add(tagged_length(class_name.len(), owner)?, owner)?;
                }
                let width = field_names
                    .iter()
                    .map(String::len)
                    .max()
                    .unwrap_or(0)
                    .checked_add(1)
                    .ok_or_else(|| overflow(owner))?;
                let width = i32::try_from(width).map_err(|_| {
                    owner.error(
                        MatEncodingFailure::Io,
                        "MAT field name width exceeds int32 range",
                    )
                })?;
                field_name_width = Some(width);
                payload.add(tagged_length(4, owner)?, owner)?;
                payload.add(
                    tagged_length(
                        (width as usize)
                            .checked_mul(field_names.len())
                            .ok_or_else(|| overflow(owner))?,
                        owner,
                    )?,
                    owner,
                )?;
                children.try_reserve(field_values.len()).map_err(|error| {
                    owner.with_source(
                        MatEncodingFailure::Io,
                        "MAT layout plan cannot be allocated",
                        error,
                    )
                })?;
                for value in field_values {
                    let child = plan_matrix(value, None, owner, cooperation).await?;
                    payload.add(child.total_bytes, owner)?;
                    children.push(child);
                }
            }
            MatData::Sparse {
                col_ptrs,
                row_indices,
                values,
                ..
            } => {
                payload.add(
                    tagged_length(
                        row_indices
                            .len()
                            .checked_mul(4)
                            .ok_or_else(|| overflow(owner))?,
                        owner,
                    )?,
                    owner,
                )?;
                payload.add(
                    tagged_length(
                        col_ptrs
                            .len()
                            .checked_mul(4)
                            .ok_or_else(|| overflow(owner))?,
                        owner,
                    )?,
                    owner,
                )?;
                payload.add(numeric_length(values, owner)?, owner)?;
            }
        }
        let (payload_bytes, total_bytes) = payload.finish(owner)?;
        Ok(MatrixPlan {
            payload_bytes,
            children,
            field_name_width,
            total_bytes,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn oversized_enclosing_matrix_rejects_without_a_payload_allocation() {
        let owner = MatEncodingOwner::Save;
        let mut payload = MatrixPayload::new();
        payload
            .add(u32::MAX as usize, owner)
            .expect("maximum tag length");
        let error = payload
            .add(16, owner)
            .expect_err("enclosing matrix length must be bounded before encoding");
        assert_eq!(error.identifier(), Some("RunMat:save:Io"));
        assert!(error.message().contains("uint32 length"));
        assert_eq!(tagged_length(8, owner).expect("aligned tag"), 16);
        assert_eq!(tagged_length(9, owner).expect("padded tag"), 24);
    }
}
