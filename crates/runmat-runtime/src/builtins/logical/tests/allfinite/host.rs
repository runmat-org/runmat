//! Host-storage finite reduction; provider routing belongs to the boundary.

use runmat_value::{ComplexTensor, SparseTensor, Tensor};

pub(super) fn tensor_all_finite(value: &Tensor) -> bool {
    (0..value.len()).all(|index| {
        value
            .numeric_value_at(index)
            .expect("tensor storage is structurally valid")
            .is_finite()
    })
}

pub(super) fn complex_tensor_all_finite(value: &ComplexTensor) -> bool {
    (0..value.len()).all(|index| {
        let (real, imaginary) = value
            .numeric_value_at(index)
            .expect("complex tensor storage is structurally valid");
        real.is_finite() && imaginary.is_finite()
    })
}

pub(super) fn sparse_all_finite(value: &SparseTensor) -> bool {
    if value.integer_storage().is_some() || value.is_logical() {
        return true;
    }
    if value.is_complex() {
        return value
            .materialize_complex_f64()
            .expect("complex sparse storage is structurally valid")
            .iter()
            .all(|(real, imaginary)| real.is_finite() && imaginary.is_finite());
    }
    value
        .materialize_f64()
        .iter()
        .all(|value| value.is_finite())
}
