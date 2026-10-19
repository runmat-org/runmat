//! Semantic-batch selection for admitted, native-precision host tensors.
//!
//! The batch coordinate may be the first, second, or a later physical axis.
//! Selection changes the B extent only; all other labels and extents remain
//! under the shared axis plan's authority.

use runmat_builtins::DeepLearningAxisLabel;
use runmat_value::{NumericDType, NumericStorage, Tensor};

use crate::builtins::deep_learning::{deep_learning_error, model::plan::axes::AxisPlan};
use crate::BuiltinResult;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

pub(super) struct BatchSelection<'a> {
    source: &'a AxisPlan,
    indices: &'a [usize],
    batch_axis: usize,
    inner: usize,
    outer: usize,
}

impl<'a> BatchSelection<'a> {
    pub(super) fn admit(
        source: &'a AxisPlan,
        indices: &'a [usize],
        function: &'static str,
    ) -> BuiltinResult<Self> {
        let batch_axis = source
            .layout()
            .index(DeepLearningAxisLabel::Batch)
            .ok_or_else(|| fail(function, "training data must have a batch axis"))?;
        let batch = source.shape()[batch_axis];
        if batch == 0 || indices.is_empty() || indices.iter().any(|&index| index >= batch) {
            return Err(fail(
                function,
                "training batch indices are empty or out of range",
            ));
        }
        let inner = source.shape()[..batch_axis]
            .iter()
            .try_fold(1usize, |size, &extent| size.checked_mul(extent))
            .ok_or_else(|| fail(function, "training batch geometry overflows"))?;
        let outer = source.shape()[batch_axis + 1..]
            .iter()
            .try_fold(1usize, |size, &extent| size.checked_mul(extent))
            .ok_or_else(|| fail(function, "training batch geometry overflows"))?;
        Ok(Self {
            source,
            indices,
            batch_axis,
            inner,
            outer,
        })
    }

    pub(super) fn axes(&self, function: &'static str) -> BuiltinResult<AxisPlan> {
        let mut shape = self.source.shape().to_vec();
        shape[self.batch_axis] = self.indices.len();
        AxisPlan::new(self.source.layout().clone(), &shape).map_err(|error| {
            fail(
                function,
                format!("invalid selected batch geometry: {error:?}"),
            )
        })
    }

    pub(super) fn select(&self, data: &Tensor, function: &'static str) -> BuiltinResult<Tensor> {
        if data.shape() != self.source.shape() || data.len() != self.source.element_count() {
            return Err(fail(
                function,
                "training tensor differs from admitted geometry",
            ));
        }
        if !matches!(data.numeric_dtype(), NumericDType::F32 | NumericDType::F64) {
            return Err(fail(function, "training tensor must be single or double"));
        }
        let batch = self.source.shape()[self.batch_axis];
        if self.indices.len() == batch && self.indices.iter().copied().eq(0..batch) {
            return Ok(data.clone());
        }
        let shape = self.axes(function)?.shape().to_vec();
        let storage = match data.numeric_dtype() {
            NumericDType::F32 => NumericStorage::F32(
                self.copy(data.as_f32_slice().expect("admitted F32 tensor"), batch),
            ),
            NumericDType::F64 => NumericStorage::F64(
                self.copy(data.as_f64_slice().expect("admitted F64 tensor"), batch),
            ),
            _ => unreachable!("admitted precision"),
        };
        Tensor::from_numeric_storage(storage, shape).map_err(|error| fail(function, error))
    }

    fn copy<T: Copy>(&self, values: &[T], batch: usize) -> Vec<T> {
        let mut selected = Vec::with_capacity(self.inner * self.indices.len() * self.outer);
        for outer in 0..self.outer {
            for &observation in self.indices {
                let begin = (outer * batch + observation) * self.inner;
                selected.extend_from_slice(&values[begin..begin + self.inner]);
            }
        }
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::DeepLearningAxisLayout;

    fn axes(format: &str, shape: &[usize]) -> AxisPlan {
        AxisPlan::new(
            DeepLearningAxisLayout::parse(format, shape.len()).unwrap(),
            shape,
        )
        .unwrap()
    }

    #[test]
    fn selects_semantic_batch_without_relabelling_or_widening() {
        let cb = axes("CB", &[2, 3]);
        let data = Tensor::from_f32(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
        let selected = BatchSelection::admit(&cb, &[2, 0], "test").unwrap();
        assert_eq!(selected.axes("test").unwrap().shape(), &[2, 2]);
        assert_eq!(
            selected.select(&data, "test").unwrap().as_f32_slice(),
            Some(&[5., 6., 1., 2.][..])
        );

        let bc = axes("BC", &[3, 2]);
        let data = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]).unwrap();
        let selected = BatchSelection::admit(&bc, &[2, 0], "test").unwrap();
        assert_eq!(
            selected.select(&data, "test").unwrap().as_f64_slice(),
            Some(&[3., 1., 6., 4.][..])
        );

        let cbt = axes("CBT", &[2, 2, 2]);
        let data =
            Tensor::from_f32((1..=8).map(|value| value as f32).collect(), vec![2, 2, 2]).unwrap();
        let selected = BatchSelection::admit(&cbt, &[1], "test").unwrap();
        assert_eq!(
            selected.select(&data, "test").unwrap().as_f32_slice(),
            Some(&[3., 4., 7., 8.][..])
        );
    }

    #[test]
    fn rejects_missing_or_invalid_batch_before_allocation() {
        let ct = axes("CT", &[2, 3]);
        assert!(BatchSelection::admit(&ct, &[0], "test").is_err());
        let cb = axes("CB", &[2, 3]);
        assert!(BatchSelection::admit(&cb, &[3], "test").is_err());
        assert!(BatchSelection::admit(&cb, &[], "test").is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn singleton_batch_selection_preserves_admitted_axes_and_rejects_equal_count_reshape() {
        let cb = axes("CB", &[2, 1]);
        let selected = BatchSelection::admit(&cb, &[0], "test").unwrap();
        let data = Tensor::from_f32(vec![1., 2.], vec![2, 1]).unwrap();
        let output = selected.select(&data, "test").unwrap();
        assert_eq!(output.shape(), &[2, 1]);
        assert_eq!(output.as_f32_slice(), Some(&[1., 2.][..]));

        let reshaped = Tensor::from_f32(vec![1., 2.], vec![1, 2]).unwrap();
        let error = selected.select(&reshaped, "test").err().unwrap();
        assert!(error
            .message()
            .contains("training tensor differs from admitted geometry"));
    }
}
