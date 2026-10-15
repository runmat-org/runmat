//! Admission for bit-preserving structural producer outputs, not dispatch policy.
use super::*;
use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, GpuTensorLayout,
};

fn like(input: &GpuTensorHandle, alias: GpuOutputAliasPolicy) -> UnaryGpuOutputContract {
    UnaryGpuOutputContract {
        storage: input.physical().storage(),
        precision: handle_precision(input),
        integer: handle_integer_type(input),
        logical: handle_is_logical(input),
        alias,
    }
}

/// Reshape preserves linear order. A lazy vector transpose has the same
/// physical order; a general lazy matrix view cannot be relabelled as dense.
pub(crate) fn reshape_output_matches(
    original: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    shape: &[usize],
) -> bool {
    let count = |shape: &[usize]| {
        if shape.contains(&0) {
            Some(0)
        } else {
            shape
                .iter()
                .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        }
    };
    let Some(original_count) = count(original.shape()) else {
        return false;
    };
    if Some(original_count) != count(shape) {
        return false;
    }
    if !unary_gpu_output_matches_shape(
        output,
        original,
        provider,
        shape,
        like(original, GpuOutputAliasPolicy::AllowInput),
    ) || output.physical().element_type() != original.physical().element_type()
        || output.view().validate(output.physical()).is_err()
    {
        return false;
    }
    if !original.shares_allocation_with(output) {
        return output.view().layout() == GpuTensorLayout::Contiguous;
    }
    (output.shape() == original.shape() && output.view().layout() == original.view().layout())
        || (output.view().layout() == GpuTensorLayout::Contiguous
            && match original.view().layout() {
                GpuTensorLayout::Contiguous => true,
                GpuTensorLayout::Transpose(info) => info.base_rows <= 1 || info.base_cols <= 1,
            })
}

pub(crate) fn replication_shape(shape: &[usize], repetitions: &[usize]) -> Option<Vec<usize>> {
    if repetitions.is_empty() {
        return None;
    }
    let rank = shape.len().max(repetitions.len()).max(2);
    let factors = if repetitions.len() == 1 {
        vec![runmat_types::DimensionFact::Known(repetitions[0]); rank]
    } else {
        repetitions
            .iter()
            .copied()
            .map(runmat_types::DimensionFact::Known)
            .collect()
    };
    runmat_types::repmat_shape(
        &runmat_types::ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        &factors,
    )
    .known_dims()?
    .into_iter()
    .collect()
}

pub(crate) fn replication_output_matches(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    shape: &[usize],
) -> bool {
    unary_gpu_output_matches_shape(
        output,
        input,
        provider,
        shape,
        like(input, GpuOutputAliasPolicy::RequireDistinctAllocation),
    ) && output.physical().element_type() == input.physical().element_type()
        && output.view().layout() == GpuTensorLayout::Contiguous
}

pub(crate) fn flip_output_matches(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    axes: &[usize],
) -> bool {
    if !unary_gpu_output_matches_shape(
        output,
        input,
        provider,
        input.shape(),
        like(input, GpuOutputAliasPolicy::AllowInput),
    ) || output.physical().element_type() != input.physical().element_type()
        || output.view().validate(output.physical()).is_err()
    {
        return false;
    }
    if !input.shares_allocation_with(output) {
        return output.view().layout() == GpuTensorLayout::Contiguous;
    }
    !runmat_types::geometry::FlipGeometry::new(input.shape(), axes).has_effect()
        && output.view() == input.view()
}

pub(crate) fn permutation_output_matches(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    order: &[usize],
) -> bool {
    let mut shape = input.shape().to_vec();
    if order.len() < shape.len() || order.is_empty() {
        return false;
    }
    shape.resize(order.len(), 1);
    let mut seen = vec![false; order.len()];
    let mut expected = Vec::with_capacity(order.len());
    for &axis in order {
        if axis >= order.len() || seen[axis] {
            return false;
        }
        seen[axis] = true;
        expected.push(shape[axis]);
    }
    if !unary_gpu_output_matches_shape(
        output,
        input,
        provider,
        &expected,
        like(input, GpuOutputAliasPolicy::AllowInput),
    ) || output.physical().element_type() != input.physical().element_type()
    {
        return false;
    }
    if !input.shares_allocation_with(output) {
        return output.view().layout() == GpuTensorLayout::Contiguous;
    }
    // A same-allocation result must be the exact coordinate transform, not
    // merely a plausible shape. Identity retains the original view layout.
    if order.iter().copied().eq(0..order.len()) {
        return output.view().layout() == input.view().layout();
    }
    if order == [1, 0] {
        match input.view().layout() {
            GpuTensorLayout::Transpose(info) => {
                return output.view().layout() == GpuTensorLayout::Contiguous
                    && output.shape() == [info.base_rows, info.base_cols];
            }
            GpuTensorLayout::Contiguous if input.shape().len() == 2 => {
                return output.view().layout()
                    == GpuTensorLayout::Transpose(runmat_accelerate_api::TransposeInfo {
                        base_rows: input.shape()[0],
                        base_cols: input.shape()[1],
                    });
            }
            _ => {}
        }
    }
    // Reordering only singleton axes preserves the physical linear sequence.
    input.view().layout() == GpuTensorLayout::Contiguous
        && output.view().layout() == GpuTensorLayout::Contiguous
        && order
            .iter()
            .copied()
            .filter(|axis| shape[*axis] != 1)
            .eq((0..shape.len()).filter(|axis| shape[*axis] != 1))
}
