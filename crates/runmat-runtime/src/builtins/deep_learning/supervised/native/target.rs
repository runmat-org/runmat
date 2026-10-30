//! Effect-free numerical target admission and selective host preparation.
//!
//! Unformatted targets follow the public observation-major convention. A
//! formatted dlarray instead supplies its own semantic axes. Both paths lower
//! through the same checked AxisPlan before loss evaluation.

#[cfg(test)]
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use runmat_accelerate_api::{handle_is_explicit, GpuTensorStorage, ProviderPrecision};
use runmat_builtins::{DeepLearningAxisLabel, DeepLearningAxisLayout};
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::deep_learning::{
    deep_learning_error,
    model::{
        execution::{host::network::HostValue, present_host},
        plan::axes::AxisPlan,
        precision::project_owned,
    },
    DLARRAY_CLASS,
};
use crate::BuiltinResult;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

pub(super) struct NumericTargetPlan<'a> {
    data: &'a Value,
    source: AxisPlan,
    output: AxisPlan,
    dtype: NumericDType,
    function: &'static str,
}

impl<'a> NumericTargetPlan<'a> {
    pub(super) fn admit(
        target: &'a Value,
        output: &AxisPlan,
        function: &'static str,
    ) -> BuiltinResult<Self> {
        let (data, format) = match target {
            Value::Object(object) if object.class_name().is(DLARRAY_CLASS) => {
                let data = object
                    .properties()
                    .get("Data")
                    .ok_or_else(|| fail(function, "target dlarray is missing Data"))?;
                let format = match object.properties().get("Format") {
                    Some(Value::String(runmat_value::StringElement::Text(text))) => text.as_str(),
                    _ => return Err(fail(function, "target dlarray Format must be text")),
                };
                match object.properties().get("Labels") {
                    Some(Value::String(runmat_value::StringElement::Text(labels)))
                        if labels == format => {}
                    _ => return Err(fail(function, "target dlarray Format and Labels conflict")),
                }
                (data, Some(format))
            }
            value => (value, None),
        };
        let (shape, dtype) = match data {
            Value::Tensor(tensor) => (tensor.shape(), tensor.numeric_dtype()),
            Value::GpuTensor(handle) => {
                if handle_is_explicit(handle) {
                    return Err(fail(
                        function,
                        "explicit gpuArray targets cannot be gathered for host training",
                    ));
                }
                if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
                    || runmat_accelerate_api::handle_is_logical(handle)
                {
                    return Err(fail(function, "training targets must be real numeric data"));
                }
                handle
                    .allocation_lease()
                    .validate_live()
                    .map_err(|error| fail(function, error))?;
                if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err() {
                    return Err(fail(function, "target has no live owning provider"));
                }
                let dtype = match runmat_accelerate_api::handle_precision(handle) {
                    Some(ProviderPrecision::F32) => NumericDType::F32,
                    Some(ProviderPrecision::F64) => NumericDType::F64,
                    None => {
                        return Err(fail(function, "target has unsupported resident precision"))
                    }
                };
                (handle.shape(), dtype)
            }
            _ => {
                return Err(fail(
                    function,
                    "numerical targets must be real arrays or dlarray data",
                ))
            }
        };
        let source = if let Some(format) = format.filter(|format| !format.is_empty()) {
            let layout = DeepLearningAxisLayout::parse(format, shape.len())
                .map_err(|error| fail(function, error))?;
            AxisPlan::new(layout, shape)
                .map_err(|error| fail(function, format!("invalid target axes: {error:?}")))?
        } else {
            let labels = output.layout().axes();
            let batch = output
                .layout()
                .index(DeepLearningAxisLabel::Batch)
                .ok_or_else(|| fail(function, "training output must have a batch axis"))?;
            let mut order = Vec::with_capacity(labels.len());
            order.push(batch);
            order.extend((0..labels.len()).filter(|&axis| axis != batch));
            let ordered_labels = order.iter().map(|&axis| labels[axis]).collect::<Vec<_>>();
            let ordered_shape = order
                .iter()
                .map(|&axis| output.shape()[axis])
                .collect::<Vec<_>>();
            if shape != ordered_shape {
                if shape.first() != ordered_shape.first() {
                    return Err(fail(
                        function,
                        format!(
                            "Y must have one row per observation; expected {}, got {}",
                            ordered_shape[0],
                            shape.first().copied().unwrap_or(0)
                        ),
                    ));
                }
                if let Some(channel_axis) = ordered_labels
                    .iter()
                    .position(|label| *label == DeepLearningAxisLabel::Channel)
                {
                    if let (Some(actual), Some(expected)) =
                        (shape.get(channel_axis), ordered_shape.get(channel_axis))
                    {
                        if actual != expected {
                            return Err(fail(
                                function,
                                format!(
                                    "Y has {actual} channels, but the network outputs {expected}"
                                ),
                            ));
                        }
                    }
                }
                return Err(fail(
                    function,
                    "unformatted Y must use observation-major output geometry",
                ));
            }
            let layout = DeepLearningAxisLayout::from_axes(&ordered_labels)
                .map_err(|error| fail(function, error))?;
            AxisPlan::new(layout, shape)
                .map_err(|error| fail(function, format!("invalid target geometry: {error:?}")))?
        };
        let lowering = source.lower_to(output.layout()).map_err(|error| {
            fail(
                function,
                format!("target and network output axes differ: {error:?}"),
            )
        })?;
        if lowering.target_shape() != output.shape() {
            return Err(fail(function, "target extent differs from network output"));
        }
        Ok(Self {
            data,
            source,
            output: output.clone(),
            dtype,
            function,
        })
    }

    pub(super) async fn prepare(self, output_dtype: NumericDType) -> BuiltinResult<Tensor> {
        let value = match self.data {
            Value::GpuTensor(_) => crate::gather_if_needed_async(self.data).await?,
            value => value.clone(),
        };
        let tensor = crate::builtins::common::tensor::value_into_tensor_for(self.function, value)
            .map_err(|error| fail(self.function, error))?;
        if tensor.shape() != self.source.shape() || tensor.numeric_dtype() != self.dtype {
            return Err(fail(
                self.function,
                "target payload changed after admission",
            ));
        }
        let tensor = project_owned(tensor, output_dtype, self.function)?;
        let HostValue::Real(tensor) = present_host(
            HostValue::Real(tensor),
            &self.source,
            &self.output,
            self.function,
        )?
        else {
            unreachable!("real target presentation")
        };
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes(format: &str, shape: &[usize]) -> AxisPlan {
        AxisPlan::new(
            DeepLearningAxisLayout::parse(format, shape.len()).unwrap(),
            shape,
        )
        .unwrap()
    }

    #[test]
    fn raw_observation_major_and_formatted_targets_share_one_output_geometry() {
        let output = axes("CB", &[2, 3]);
        let raw =
            Value::Tensor(Tensor::from_f32(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]).unwrap());
        let planned = NumericTargetPlan::admit(&raw, &output, "test").unwrap();
        let prepared = futures::executor::block_on(planned.prepare(NumericDType::F32)).unwrap();
        assert_eq!(prepared.as_f32_slice(), Some(&[1., 4., 2., 5., 3., 6.][..]));

        let mut object = runmat_value::ObjectInstance::new(DLARRAY_CLASS.owned());
        object
            .replace_dl_property(
                "Data",
                Value::Tensor(Tensor::new(vec![1., 4., 2., 5., 3., 6.], vec![2, 3]).unwrap()),
            )
            .expect("fixture property");
        object
            .replace_dl_property("Format", Value::String("CB".into()))
            .expect("fixture property");
        object
            .replace_dl_property("Labels", Value::String("CB".into()))
            .expect("fixture property");
        let formatted = Value::Object(object);
        let planned = NumericTargetPlan::admit(&formatted, &output, "test").unwrap();
        let prepared = futures::executor::block_on(planned.prepare(NumericDType::F64)).unwrap();
        assert_eq!(prepared.as_f64_slice(), Some(&[1., 4., 2., 5., 3., 6.][..]));
    }

    #[test]
    fn conflicting_labels_and_wrong_observation_extent_reject_before_preparation() {
        let output = axes("CBT", &[2, 3, 4]);
        let wrong = Value::Tensor(Tensor::new(vec![0.; 2 * 4 * 3], vec![2, 4, 3]).unwrap());
        assert!(NumericTargetPlan::admit(&wrong, &output, "test").is_err());
        let mut object = runmat_value::ObjectInstance::new(DLARRAY_CLASS.owned());
        object
            .replace_dl_property(
                "Data",
                Value::Tensor(Tensor::new(vec![0.; 24], vec![2, 3, 4]).unwrap()),
            )
            .expect("fixture property");
        object
            .replace_dl_property("Format", Value::String("CBT".into()))
            .expect("fixture property");
        object
            .replace_dl_property("Labels", Value::String("BCT".into()))
            .expect("fixture property");
        assert!(NumericTargetPlan::admit(&Value::Object(object), &output, "test").is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn singleton_observation_target_preserves_axes_and_rejects_equal_count_wrong_rows() {
        let output = axes("CB", &[2, 1]);
        let raw = Value::Tensor(Tensor::from_f32(vec![1., 2.], vec![1, 2]).unwrap());
        let plan = NumericTargetPlan::admit(&raw, &output, "test").unwrap();
        let prepared = futures::executor::block_on(plan.prepare(NumericDType::F32)).unwrap();
        assert_eq!(prepared.shape(), &[2, 1]);
        assert_eq!(prepared.as_f32_slice(), Some(&[1., 2.][..]));

        let wrong_rows = Value::Tensor(Tensor::from_f32(vec![1., 2.], vec![2, 1]).unwrap());
        let error = NumericTargetPlan::admit(&wrong_rows, &output, "test")
            .err()
            .unwrap();
        assert!(error
            .message()
            .contains("Y must have one row per observation"));
    }
}
