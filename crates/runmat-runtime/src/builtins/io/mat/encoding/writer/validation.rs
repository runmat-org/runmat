//! Structural MAT invariants checked by the layout plan before output allocation.

use super::super::super::cooperation::MatCodecCooperation;
use super::super::super::format::{MatArray, MatClass, MatData};
use super::super::{MatEncodingFailure, MatEncodingOwner};
use crate::BuiltinResult;
use runmat_value::NumericStorage;

fn invalid(owner: MatEncodingOwner, detail: &'static str) -> crate::RuntimeError {
    owner.error(MatEncodingFailure::Io, detail)
}

fn elements(dims: &[usize], owner: MatEncodingOwner) -> BuiltinResult<usize> {
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter().try_fold(1usize, |product, &extent| {
        product
            .checked_mul(extent)
            .ok_or_else(|| invalid(owner, "MAT shape element count exceeds platform limits"))
    })
}

fn numeric_payload(storage: &NumericStorage, owner: MatEncodingOwner) -> BuiltinResult<()> {
    storage
        .checked_byte_len()
        .and_then(|length| u32::try_from(length).ok())
        .ok_or_else(|| invalid(owner, "MAT numeric payload exceeds uint32 length"))?;
    Ok(())
}

pub(super) async fn validate_matrix(
    array: &MatArray,
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<()> {
    if array.dims.len() < 2 {
        return Err(invalid(owner, "MAT arrays require at least two dimensions"));
    }
    if array.dims.iter().any(|&extent| extent > i32::MAX as usize) {
        return Err(invalid(owner, "dimension exceeds MAT-file int32 range"));
    }
    array
        .dims
        .len()
        .checked_mul(4)
        .and_then(|length| u32::try_from(length).ok())
        .ok_or_else(|| invalid(owner, "MAT dimensions exceed uint32 length"))?;
    let count = if matches!(&array.data, MatData::Sparse { .. }) {
        0
    } else {
        elements(&array.dims, owner)?
    };
    match &array.data {
        MatData::Numeric { real, imag } => {
            if super::super::tensor_dtype_mat_class(real.numeric_dtype()) != array.class
                || real.len() != count
            {
                return Err(invalid(
                    owner,
                    "numeric MAT class or payload does not match its shape",
                ));
            }
            numeric_payload(real, owner)?;
            if let Some(imag) = imag {
                if imag.numeric_dtype() != real.numeric_dtype() || imag.len() != count {
                    return Err(invalid(
                        owner,
                        "complex numeric components must match class and shape",
                    ));
                }
                numeric_payload(imag, owner)?;
            }
        }
        MatData::Logical { data } => {
            if array.class != MatClass::Logical || data.len() != count {
                return Err(invalid(
                    owner,
                    "logical MAT payload does not match its shape",
                ));
            }
            u32::try_from(data.len())
                .map_err(|_| invalid(owner, "MAT logical payload exceeds uint32 length"))?;
        }
        MatData::Char { data } => {
            if array.class != MatClass::Char || data.len() != count {
                return Err(invalid(
                    owner,
                    "character MAT payload does not match its shape",
                ));
            }
            data.len()
                .checked_mul(2)
                .and_then(|length| u32::try_from(length).ok())
                .ok_or_else(|| invalid(owner, "MAT character payload exceeds uint32 length"))?;
        }
        MatData::Cell { elements } => {
            if array.class != MatClass::Cell || elements.len() != count {
                return Err(invalid(owner, "cell MAT payload does not match its shape"));
            }
        }
        MatData::Struct {
            field_names,
            field_values,
        } => {
            if array.class != MatClass::Struct
                || count.checked_mul(field_names.len()) != Some(field_values.len())
            {
                return Err(invalid(owner, "struct MAT fields do not match its shape"));
            }
        }
        MatData::Object {
            class_name,
            field_names,
            field_values,
        } => {
            if array.class != MatClass::Object
                || class_name.is_empty()
                || class_name.as_bytes().contains(&0)
                || count.checked_mul(field_names.len()) != Some(field_values.len())
            {
                return Err(invalid(
                    owner,
                    "object MAT fields do not match its class or shape",
                ));
            }
            u32::try_from(class_name.len())
                .map_err(|_| invalid(owner, "object MAT class name exceeds uint32 length"))?;
        }
        MatData::Sparse {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
            logical,
        } => {
            if array.class != MatClass::Sparse || array.dims.as_slice() != [*rows, *cols].as_slice()
            {
                return Err(invalid(owner, "sparse MAT class or shape is invalid"));
            }
            if cols.checked_add(1) != Some(col_ptrs.len())
                || col_ptrs.first() != Some(&0)
                || col_ptrs.last() != Some(&row_indices.len())
                || row_indices.len() != values.len()
            {
                return Err(invalid(
                    owner,
                    "sparse MAT pointer/value lengths are invalid",
                ));
            }
            numeric_payload(values, owner)?;
            if *logical {
                let NumericStorage::U8(data) = values else {
                    return Err(invalid(
                        owner,
                        "sparse logical MAT values must have uint8 storage",
                    ));
                };
                for chunk in data.chunks(4096) {
                    if chunk.iter().any(|&value| value != 1) {
                        return Err(invalid(
                            owner,
                            "sparse logical MAT values must be true at every stored index",
                        ));
                    }
                    super::completed(cooperation, chunk.len()).await?;
                }
            }
            let mut previous = 0;
            for chunk in col_ptrs.chunks(4096) {
                for &pointer in chunk {
                    if pointer < previous
                        || pointer > row_indices.len()
                        || pointer > i32::MAX as usize
                    {
                        return Err(invalid(owner, "sparse MAT column pointer is invalid"));
                    }
                    previous = pointer;
                }
                super::completed(cooperation, chunk.len()).await?;
            }
            for chunk in row_indices.chunks(4096) {
                if chunk
                    .iter()
                    .any(|&index| index >= *rows || index > i32::MAX as usize)
                {
                    return Err(invalid(owner, "sparse MAT row index exceeds row count"));
                }
                super::completed(cooperation, chunk.len()).await?;
            }
            for col in 0..*cols {
                super::completed(cooperation, 1).await?;
                let mut prior = None;
                for chunk in row_indices[col_ptrs[col]..col_ptrs[col + 1]].chunks(4096) {
                    for &row in chunk {
                        if prior.is_some_and(|previous| previous >= row) {
                            return Err(invalid(
                                owner,
                                "sparse MAT row indices must be sorted and unique in each column",
                            ));
                        }
                        prior = Some(row);
                    }
                    super::completed(cooperation, chunk.len()).await?;
                }
            }
        }
    }
    Ok(())
}
