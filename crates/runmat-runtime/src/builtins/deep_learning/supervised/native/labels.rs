//! Exact external label identities lowered to native one-hot output storage.
//!
//! Label parsing remains in the existing classification owner. This plan
//! only binds admitted labels to the shared output axes and precision.

use runmat_builtins::DeepLearningAxisLabel;
use runmat_value::{NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::deep_learning::{
    deep_learning_error, model::plan::axes::AxisPlan, string_array,
};
use crate::BuiltinResult;

#[cfg(test)]
use super::super::targets::label_strings;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

pub(super) struct LabelTargetPlan {
    classes: Vec<String>,
    class_index: Vec<usize>,
    axes: AxisPlan,
    function: &'static str,
}

impl LabelTargetPlan {
    #[cfg(test)]
    pub(super) fn admit(
        labels: &Value,
        output: &AxisPlan,
        function: &'static str,
    ) -> BuiltinResult<Self> {
        let labels = label_strings(labels, function)?.ok_or_else(|| {
            fail(
                function,
                "expected categorical, text, or exact integer labels",
            )
        })?;
        Self::from_labels(labels, output, function)
    }

    pub(super) fn from_labels(
        labels: Vec<String>,
        output: &AxisPlan,
        function: &'static str,
    ) -> BuiltinResult<Self> {
        let batch = output
            .layout()
            .index(DeepLearningAxisLabel::Batch)
            .ok_or_else(|| fail(function, "classification output needs a batch axis"))?;
        let channel = output
            .layout()
            .index(DeepLearningAxisLabel::Channel)
            .ok_or_else(|| fail(function, "classification output needs a channel axis"))?;
        if labels.len() != output.shape()[batch]
            || output
                .shape()
                .iter()
                .enumerate()
                .any(|(axis, &size)| axis != batch && axis != channel && size != 1)
        {
            return Err(fail(
                function,
                "one-label-per-observation targets require singleton non-channel output axes",
            ));
        }
        let mut classes = labels.clone();
        classes.sort();
        classes.dedup();
        if classes.is_empty() || classes.len() != output.shape()[channel] {
            return Err(fail(
                function,
                "classification class count differs from network output channels",
            ));
        }
        let class_index = labels
            .iter()
            .map(|label| classes.binary_search(label).expect("class was collected"))
            .collect();
        Ok(Self {
            classes,
            class_index,
            axes: output.clone(),
            function,
        })
    }

    pub(super) fn classes_value(&self) -> BuiltinResult<Value> {
        string_array(
            self.classes.clone(),
            vec![1, self.classes.len()],
            self.function,
        )
    }

    pub(super) fn one_hot(&self, dtype: NumericDType) -> BuiltinResult<Tensor> {
        let batch = self
            .axes
            .layout()
            .index(DeepLearningAxisLabel::Batch)
            .expect("admitted batch axis");
        let channel = self
            .axes
            .layout()
            .index(DeepLearningAxisLabel::Channel)
            .expect("admitted channel axis");
        let mut indices = vec![0; self.axes.shape().len()];
        let storage = match dtype {
            NumericDType::F32 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(self.axes.element_count())
                    .map_err(|_| {
                        fail(
                            self.function,
                            "cannot reserve classification target storage",
                        )
                    })?;
                values.resize(self.axes.element_count(), 0.0_f32);
                for (observation, &class) in self.class_index.iter().enumerate() {
                    indices[batch] = observation;
                    indices[channel] = class;
                    let offset = self.axes.offset(&indices).map_err(|error| {
                        fail(
                            self.function,
                            format!("invalid target coordinate: {error:?}"),
                        )
                    })?;
                    values[offset] = 1.0;
                }
                NumericStorage::F32(values)
            }
            NumericDType::F64 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(self.axes.element_count())
                    .map_err(|_| {
                        fail(
                            self.function,
                            "cannot reserve classification target storage",
                        )
                    })?;
                values.resize(self.axes.element_count(), 0.0_f64);
                for (observation, &class) in self.class_index.iter().enumerate() {
                    indices[batch] = observation;
                    indices[channel] = class;
                    let offset = self.axes.offset(&indices).map_err(|error| {
                        fail(
                            self.function,
                            format!("invalid target coordinate: {error:?}"),
                        )
                    })?;
                    values[offset] = 1.0;
                }
                NumericStorage::F64(values)
            }
            _ => {
                return Err(fail(
                    self.function,
                    "training output must be single or double",
                ))
            }
        };
        Tensor::from_numeric_storage(storage, self.axes.shape().to_vec())
            .map_err(|error| fail(self.function, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::DeepLearningAxisLayout;
    use runmat_value::{IntegerStorage, NumericDType, StringArray};

    #[test]
    fn wide_integer_ids_remain_distinct_and_follow_physical_channel_axis() {
        let first = (1_u64 << 53) + 1;
        let labels = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![first, first + 1]), vec![2, 1]).unwrap(),
        );
        let axes = AxisPlan::new(DeepLearningAxisLayout::parse("CB", 2).unwrap(), &[2, 2]).unwrap();
        let plan = LabelTargetPlan::admit(&labels, &axes, "test").unwrap();
        let Value::StringArray(classes) = plan.classes_value().unwrap() else {
            panic!("class labels expected")
        };
        assert_eq!(
            classes.data(),
            vec![first.to_string(), (first + 1).to_string()]
        );
        let one_hot = plan.one_hot(NumericDType::F32).unwrap();
        assert_eq!(one_hot.as_f32_slice(), Some(&[1., 0., 0., 1.][..]));

        let text = Value::StringArray(
            StringArray::new(vec!["right".into(), "left".into()], vec![2, 1]).unwrap(),
        );
        let axes = AxisPlan::new(DeepLearningAxisLayout::parse("BC", 2).unwrap(), &[2, 2]).unwrap();
        let plan = LabelTargetPlan::admit(&text, &axes, "test").unwrap();
        assert_eq!(
            plan.one_hot(NumericDType::F64).unwrap().as_f64_slice(),
            Some(&[0., 1., 1., 0.][..])
        );
    }

    #[test]
    fn label_count_and_non_singleton_time_reject_before_allocation() {
        let labels =
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap());
        let axes =
            AxisPlan::new(DeepLearningAxisLayout::parse("CBT", 3).unwrap(), &[2, 2, 3]).unwrap();
        assert!(LabelTargetPlan::admit(&labels, &axes, "test").is_err());
        let axes = AxisPlan::new(DeepLearningAxisLayout::parse("CB", 2).unwrap(), &[2, 3]).unwrap();
        assert!(LabelTargetPlan::admit(&labels, &axes, "test").is_err());
    }
}
