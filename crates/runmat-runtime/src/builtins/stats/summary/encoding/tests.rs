use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage};

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn assert_f64_slice_eq_nan(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || (*a - *b).abs() < 1.0e-12,
            "mismatch at {idx}: {a:?} != {b:?}"
        );
    }
}

#[path = "tests/dummyvar.rs"]
mod dummyvar;
#[path = "tests/encode.rs"]
mod encode;
#[path = "tests/decode.rs"]
mod decode;
#[path = "tests/shared.rs"]
mod shared;
#[path = "tests/provider_contract.rs"]
mod provider_contract;
