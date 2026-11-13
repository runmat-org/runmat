//! Exact native factor element layout and buffer transfer.
use super::super::{managed_cholesky::ManagedCholeskyScalar, managed_qr::SparseQrScalar};
use super::RetainedFactorError;
use runmat_value::{
    AdoptedHostAllocation, ComplexElement, NumericDType, NumericFactorBufferAdoption as Adoption,
    SparseTensor, ValidatedSparseCscParts,
};
/// The canonical complex representation is chosen before factorization.
/// Only heap-free native elements can enter a retained buffer transfer.
pub(in crate::builtins::math::linalg) trait RetainedFactorScalar:
    SparseQrScalar + ManagedCholeskyScalar + Send + Sync + 'static
{
    fn header_bytes() -> usize;
    fn storage_class() -> runmat_value::ExactHostSparseClass;
    fn borrowed(matrix: &SparseTensor) -> Option<&[Self]>;
    fn dense(value: &runmat_value::Value) -> Option<&[Self]>;
    fn adopt(
        parts: ValidatedSparseCscParts,
        allocation: AdoptedHostAllocation,
        len: usize,
    ) -> Result<SparseTensor, RetainedFactorError>;
}
macro_rules! real {
    ($ty:ty, $access:ident, $dtype:ident) => {
        impl RetainedFactorScalar for $ty {
            fn header_bytes() -> usize {
                Adoption::numeric_header_bytes()
            }
            fn storage_class() -> runmat_value::ExactHostSparseClass {
                runmat_value::ExactHostSparseClass::Numeric(NumericDType::$dtype)
            }
            fn borrowed(matrix: &SparseTensor) -> Option<&[Self]> {
                matrix.$access()
            }
            fn dense(value: &runmat_value::Value) -> Option<&[Self]> {
                if let runmat_value::Value::Tensor(matrix) = value {
                    matrix.$access()
                } else {
                    None
                }
            }
            fn adopt(
                parts: ValidatedSparseCscParts,
                allocation: AdoptedHostAllocation,
                len: usize,
            ) -> Result<SparseTensor, RetainedFactorError> {
                let values = Adoption::numeric(allocation, NumericDType::$dtype, len)
                    .map_err(RetainedFactorError::Storage)?;
                SparseTensor::from_validated_numeric(parts, values)
                    .map_err(RetainedFactorError::Invalid)
            }
        }
    };
}
real!(f32, as_f32_slice, F32);
real!(f64, as_f64_slice, F64);
macro_rules! complex {
    ($real:ty, $access:ident, $dense:ident, $dtype:ident, $header:ident, $adopt:ident, $finish:ident) => {
        impl RetainedFactorScalar for ComplexElement<$real> {
            fn header_bytes() -> usize {
                Adoption::$header()
            }
            fn storage_class() -> runmat_value::ExactHostSparseClass {
                runmat_value::ExactHostSparseClass::Complex(NumericDType::$dtype)
            }
            fn borrowed(matrix: &SparseTensor) -> Option<&[Self]> {
                matrix.$access()
            }
            fn dense(value: &runmat_value::Value) -> Option<&[Self]> {
                if let runmat_value::Value::ComplexTensor(matrix) = value {
                    matrix.$dense()
                } else {
                    None
                }
            }
            fn adopt(
                parts: ValidatedSparseCscParts,
                allocation: AdoptedHostAllocation,
                len: usize,
            ) -> Result<SparseTensor, RetainedFactorError> {
                let values =
                    Adoption::$adopt(allocation, len).map_err(RetainedFactorError::Storage)?;
                SparseTensor::$finish(parts, values).map_err(RetainedFactorError::Invalid)
            }
        }
    };
}
complex!(
    f32,
    as_complex_f32_slice,
    as_f32_slice,
    F32,
    complex_f32_header_bytes,
    complex_f32,
    from_validated_complex_f32
);
complex!(
    f64,
    as_complex_f64_slice,
    as_f64_slice,
    F64,
    complex_f64_header_bytes,
    complex_f64,
    from_validated_complex_f64
);
