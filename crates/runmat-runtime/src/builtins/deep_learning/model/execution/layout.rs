//! Pure result-axis planning and native-class host presentation.

use runmat_builtins::{DeepLearningAxisLabel, DeepLearningAxisLayout};
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor};

use super::{host::network::HostValue, request::NetworkOutputLayout};
use crate::builtins::deep_learning::{deep_learning_error, model::plan::axes::AxisPlan};
use crate::BuiltinResult;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

pub(super) fn target_axes(
    source: &AxisPlan,
    presentation: NetworkOutputLayout,
    function: &'static str,
) -> BuiltinResult<AxisPlan> {
    if presentation == NetworkOutputLayout::Preserve {
        return Ok(source.clone());
    }
    let axes = source.layout().axes();
    let Some(batch) = source.layout().index(DeepLearningAxisLabel::Batch) else {
        return Ok(source.clone());
    };
    if batch == 0 {
        return Ok(source.clone());
    }
    let mut ordered = Vec::with_capacity(axes.len());
    ordered.push(DeepLearningAxisLabel::Batch);
    ordered.extend(
        axes.iter()
            .enumerate()
            .filter_map(|(index, &label)| (index != batch).then_some(label)),
    );
    let layout = DeepLearningAxisLayout::from_axes(&ordered)
        .map_err(|error| fail(function, format!("invalid output axes: {error}")))?;
    let lowering = source
        .lower_to(&layout)
        .map_err(|error| fail(function, format!("cannot present output axes: {error:?}")))?;
    AxisPlan::new(lowering.target_layout().clone(), lowering.target_shape()).map_err(|error| {
        fail(
            function,
            format!("invalid presented output shape: {error:?}"),
        )
    })
}

pub(in crate::builtins::deep_learning) fn present_host(
    value: HostValue,
    source: &AxisPlan,
    target: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<HostValue> {
    if source == target {
        return Ok(value);
    }
    let lowering = source
        .lower_to(target.layout())
        .map_err(|error| fail(function, format!("cannot present output axes: {error:?}")))?;
    if lowering.target_shape() != target.shape() {
        return Err(fail(
            function,
            "presented output geometry changed after admission",
        ));
    }
    match value {
        HostValue::Real(tensor) => {
            if tensor.shape() != source.shape() {
                return Err(fail(
                    function,
                    "host output shape differs from admitted axes",
                ));
            }
            let storage = if let Some(values) = tensor.as_f32_slice() {
                NumericStorage::F32(reorder(values, &lowering, function)?)
            } else if let Some(values) = tensor.as_f64_slice() {
                NumericStorage::F64(reorder(values, &lowering, function)?)
            } else {
                return Err(fail(
                    function,
                    "host network output must be single or double",
                ));
            };
            Tensor::from_numeric_storage(storage, target.shape().to_vec())
                .map(HostValue::Real)
                .map_err(|error| fail(function, error))
        }
        HostValue::Complex(tensor) => {
            if tensor.shape() != source.shape() {
                return Err(fail(
                    function,
                    "complex output shape differs from admitted axes",
                ));
            }
            let storage = match tensor.complex_storage() {
                ComplexStorage::F32(values) => {
                    ComplexStorage::F32(reorder(values, &lowering, function)?.into())
                }
                ComplexStorage::F64(values) => {
                    ComplexStorage::F64(reorder(values, &lowering, function)?.into())
                }
                ComplexStorage::Integer(_) => {
                    return Err(fail(
                        function,
                        "complex integer output has no admitted network execution class",
                    ))
                }
            };
            ComplexTensor::from_complex_storage(storage, target.shape().to_vec())
                .map(HostValue::Complex)
                .map_err(|error| fail(function, error))
        }
    }
}

fn reorder<T: Copy>(
    values: &[T],
    lowering: &crate::builtins::deep_learning::model::plan::axes::AxisLowering,
    function: &'static str,
) -> BuiltinResult<Vec<T>> {
    let size = lowering
        .target_shape()
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| fail(function, "output shape overflows host indexing"))?;
    if values.len() != size {
        return Err(fail(
            function,
            "output payload length differs from admitted geometry",
        ));
    }
    let mut reordered = Vec::with_capacity(size);
    for target_index in 0..size {
        let source_index = lowering
            .source_offset_linear(target_index)
            .map_err(|error| fail(function, format!("invalid output coordinate: {error:?}")))?;
        reordered.push(values[source_index]);
    }
    Ok(reordered)
}

#[cfg(test)]
mod tests;
