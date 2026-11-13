use runmat_value::Tensor;

use crate::builtins::common::tensor;

pub(super) fn input_min(tensor: &Tensor) -> f64 {
    reduce(tensor, |current, value| value < current)
}

pub(super) fn input_max(tensor: &Tensor) -> f64 {
    reduce(tensor, |current, value| value > current)
}

fn reduce(tensor: &Tensor, replaces: impl Fn(f64, f64) -> bool) -> f64 {
    let mut result = f64::NAN;
    let values = tensor::tensor_values_f64_cow(tensor);
    for &value in values.iter().filter(|value| !value.is_nan()) {
        if result.is_nan() || replaces(result, value) {
            result = value;
        }
    }
    result
}

pub(super) fn scalar_tensor(value: f64) -> Tensor {
    Tensor::new(vec![value], vec![1, 1]).expect("scalar tensor shape")
}
