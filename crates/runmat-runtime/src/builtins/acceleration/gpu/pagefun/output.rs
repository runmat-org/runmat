use super::*;
pub(super) fn finalise_empty_output(
    operation: &PageOperation,
    inputs: &[PreparedInput],
    page_dims: &[usize],
    output_kind: OutputKind,
    resident_source: Option<&GpuTensorHandle>,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    let (rows, cols) = operation.output_matrix_shape(inputs, output_kind)?;
    let final_shape = assemble_shape(rows, cols, page_dims);
    let entries = checked_volume(&final_shape)?;
    match output_kind {
        OutputKind::Real => {
            let dtype = operation.output_numeric_dtype(inputs);
            let tensor =
                Tensor::from_numeric_storage(NumericStorage::zeros(dtype, entries), final_shape)
                    .map_err(|e| {
                        pagefun_error_with_detail(
                            &PAGEFUN_ERROR_INTERNAL,
                            format!("failed to build empty tensor ({e})"),
                        )
                    })?;
            FinalOutput::Real(tensor).into_value(resident_source, protected)
        }
        OutputKind::Complex => {
            let tensor = ComplexTensor::from_f64_values_with_dtype(
                vec![(0.0, 0.0); entries],
                final_shape,
                operation.output_numeric_dtype(inputs),
            )
            .map_err(|e| {
                pagefun_error_with_detail(
                    &PAGEFUN_ERROR_INTERNAL,
                    format!("failed to build empty complex tensor ({e})"),
                )
            })?;
            FinalOutput::Complex(tensor).into_value(resident_source, protected)
        }
    }
}

pub(super) fn assemble_shape(rows: usize, cols: usize, page_dims: &[usize]) -> Vec<usize> {
    let mut shape = vec![rows, cols];
    shape.extend_from_slice(page_dims);
    shape
}

pub(super) fn increment_multi_index(indices: &mut [usize], dims: &[usize]) -> BuiltinResult<()> {
    if dims.contains(&0) {
        return Ok(());
    }
    for (dim, &limit) in dims.iter().enumerate() {
        if limit == 0 {
            continue;
        }
        indices[dim] += 1;
        if indices[dim] < limit {
            return Ok(());
        }
        indices[dim] = 0;
        if dim + 1 == dims.len() {
            break;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OutputKind {
    Real,
    Complex,
}

pub(super) enum FinalOutput {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl FinalOutput {
    pub(super) fn into_value(
        self,
        resident_source: Option<&GpuTensorHandle>,
        protected: &[&GpuTensorHandle],
    ) -> BuiltinResult<Value> {
        let value = match self {
            FinalOutput::Real(tensor) => Value::Tensor(tensor),
            FinalOutput::Complex(tensor) => Value::ComplexTensor(tensor),
        };
        if crate::output_count::current_output_count() == Some(0) {
            return Ok(value);
        }
        match resident_source {
            Some(source) => gpu_helpers::restore_class_preserving_value_protecting(
                source,
                value,
                BUILTIN_NAME,
                protected,
            ),
            None => Ok(value),
        }
    }
}
