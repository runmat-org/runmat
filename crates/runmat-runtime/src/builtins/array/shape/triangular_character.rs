//! Character triangular masking with unique reuse or an admitted replacement.

use crate::builtins::common::character_output::{build, CharacterBuildErrors, CHECKPOINT_CHUNK};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{CharArray, Utf16CodeUnit};

#[derive(Clone, Copy)]
pub(super) enum Triangle {
    Lower,
    Upper,
}

impl Triangle {
    fn name(self) -> &'static str {
        match self {
            Self::Lower => "tril",
            Self::Upper => "triu",
        }
    }

    fn keeps(self, row: usize, col: usize, offset: isize) -> bool {
        let diagonal = col as i128 - row as i128;
        match self {
            Self::Lower => diagonal <= offset as i128,
            Self::Upper => diagonal >= offset as i128,
        }
    }
}

pub(super) fn mask(
    context: Option<&RuntimeContext>,
    mut source: CharArray,
    offset: isize,
    triangle: Triangle,
) -> BuiltinResult<CharArray> {
    let policy = HostAllocationReceipt::policy(context)
        .map_err(|cause| allocation_error(context, triangle, cause))?;
    let (rows, cols) = geometry(context, triangle, &source)?;
    let page_len = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid(triangle, "Character page size overflow"))?;
    let retained = |index: usize| {
        let within_page = index % page_len;
        triangle.keeps(within_page / cols, within_page % cols, offset)
    };

    if let Some(data) = source.try_data_mut_unique() {
        for (index, value) in data.iter_mut().enumerate() {
            checkpoint(context, triangle, index)?;
            if !retained(index) {
                *value = Utf16CodeUnit::from_raw(0);
            }
        }
        return Ok(source);
    }

    if policy == HostAllocationPolicy::Managed {
        let context = context.ok_or_else(|| {
            allocation_error(None, triangle, HostAllocationError::MissingManagedOwner)
        })?;
        let errors = match triangle {
            Triangle::Lower => CharacterBuildErrors {
                admission: |context, cause| allocation_error(Some(context), Triangle::Lower, cause),
                invalid: |detail| invalid(Triangle::Lower, detail),
            },
            Triangle::Upper => CharacterBuildErrors {
                admission: |context, cause| allocation_error(Some(context), Triangle::Upper, cause),
                invalid: |detail| invalid(Triangle::Upper, detail),
            },
        };
        return build(
            context,
            source.shape.len(),
            source.data().len(),
            0,
            |axis| Ok(source.shape[axis]),
            |data, _, receipt| {
                for (index, (output, input)) in
                    data.iter_mut().zip(source.data().iter()).enumerate()
                {
                    if index.is_multiple_of(CHECKPOINT_CHUNK) {
                        receipt
                            .checkpoint(context)
                            .map_err(|cause| allocation_error(Some(context), triangle, cause))?;
                    }
                    *output = if retained(index) {
                        *input
                    } else {
                        Utf16CodeUnit::from_raw(0)
                    };
                }
                Ok(())
            },
            errors,
        );
    }

    // Ordinary shared input requires copy-on-write; never detach a finite
    // replacement through this branch.
    let mut data = Vec::with_capacity(source.data().len());
    for (index, value) in source.data().iter().copied().enumerate() {
        checkpoint(context, triangle, index)?;
        data.push(if retained(index) {
            value
        } else {
            Utf16CodeUnit::from_raw(0)
        });
    }
    CharArray::from_code_units_with_shape(data, source.shape.to_vec())
        .map_err(|detail| invalid(triangle, detail))
}

fn geometry(
    context: Option<&RuntimeContext>,
    triangle: Triangle,
    source: &CharArray,
) -> BuiltinResult<(usize, usize)> {
    let (rows, cols) = match source.shape.as_slice() {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    };
    let mut count = usize::from(!source.shape.is_empty());
    for (axis, extent) in source.shape.iter().copied().enumerate() {
        checkpoint(context, triangle, axis)?;
        count = count
            .checked_mul(extent)
            .ok_or_else(|| invalid(triangle, "Character shape exceeds platform limits"))?;
    }
    if count != source.data().len() || rows != source.rows || cols != source.cols {
        return Err(invalid(
            triangle,
            "Character payload or cached dimensions disagree with shape",
        ));
    }
    Ok((rows, cols))
}

fn checkpoint(
    context: Option<&RuntimeContext>,
    triangle: Triangle,
    index: usize,
) -> BuiltinResult<()> {
    if index.is_multiple_of(CHECKPOINT_CHUNK) {
        HostAllocationReceipt::policy(context)
            .map_err(|cause| allocation_error(context, triangle, cause))?;
    }
    Ok(())
}

fn allocation_error(
    context: Option<&RuntimeContext>,
    triangle: Triangle,
    cause: HostAllocationError,
) -> RuntimeError {
    build_runtime_error(format!(
        "{}: host character allocation: {cause}",
        triangle.name()
    ))
    .with_builtin(triangle.name())
    .with_identifier(format!(
        "{}:{}",
        context.map_or_else(|| "RunMat".to_owned(), RuntimeContext::error_namespace),
        cause.runtime_identifier_suffix()
    ))
    .with_source(cause)
    .build()
}

fn invalid(triangle: Triangle, detail: impl Into<String>) -> RuntimeError {
    build_runtime_error(detail)
        .with_builtin(triangle.name())
        .build()
}
