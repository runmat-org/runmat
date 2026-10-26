//! Native output class follows the existing floating matrix-product policy.
use super::*;
pub(super) trait Element: LinearOperatorScalar {
    fn convert(value: Complex64) -> Option<Self>;
    fn is_zero(value: Self) -> bool;
    fn full(values: Vec<Self>, m: usize, n: usize) -> Result<Value, String>;
    fn sparse(
        values: Vec<Self>,
        parts: runmat_value::ValidatedSparseCscParts,
    ) -> Result<Value, String>;
}
macro_rules! real {
    ($type:ty,$storage:ident) => {
        impl Element for $type {
            fn convert(value: Complex64) -> Option<Self> {
                (value.im == 0.).then_some(value.re as Self)
            }
            fn is_zero(value: Self) -> bool {
                value == 0.
            }
            fn full(values: Vec<Self>, m: usize, n: usize) -> Result<Value, String> {
                Tensor::from_numeric_storage(NumericStorage::$storage(values), vec![m, n])
                    .map(crate::builtins::common::tensor::tensor_into_value)
            }
            fn sparse(
                values: Vec<Self>,
                parts: runmat_value::ValidatedSparseCscParts,
            ) -> Result<Value, String> {
                SparseTensor::from_validated_numeric(
                    parts,
                    runmat_value::HostNumericBuffer::from_numeric_storage(
                        NumericStorage::$storage(values),
                    ),
                )
                .map(Value::SparseTensor)
            }
        }
    };
}
real!(f32, F32);
real!(f64, F64);
macro_rules! complex {
    ($type:ty,$storage:ident,$constructor:ident) => {
        impl Element for ComplexElement<$type> {
            fn convert(value: Complex64) -> Option<Self> {
                Some(Self(value.re as $type, value.im as $type))
            }
            fn is_zero(value: Self) -> bool {
                value.0 == 0. && value.1 == 0.
            }
            fn full(values: Vec<Self>, m: usize, n: usize) -> Result<Value, String> {
                ComplexTensor::from_complex_storage(
                    ComplexStorage::$storage(values.into()),
                    vec![m, n],
                )
                .map(crate::builtins::common::random_args::complex_tensor_into_value)
            }
            fn sparse(
                values: Vec<Self>,
                parts: runmat_value::ValidatedSparseCscParts,
            ) -> Result<Value, String> {
                SparseTensor::$constructor(parts, values.into()).map(Value::SparseTensor)
            }
        }
    };
}
complex!(f32, F32, from_validated_complex_f32);
complex!(f64, F64, from_validated_complex_f64);
