//! Native-class admitted publication for sparse QR.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::{
    sparse_publication_bytes, SparsePublication, SparsePublicationError,
};
use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, ExactHostComplexTensorAllocation,
    ExactHostSparseClass, ExactHostTensorAllocation, Tensor,
};
pub(super) enum Column<T: SparseQrScalar> {
    Managed(HostTypedWorkspace<T>),
    Ordinary(Vec<T>),
}
impl<T: SparseQrScalar> Column<T> {
    pub(super) fn slice(&mut self) -> &mut [T] {
        match self {
            Self::Managed(v) => v.as_mut_slice(),
            Self::Ordinary(v) => v,
        }
    }
}
/// Ordinary storage is already in its publication class. Finishing moves its
/// backing; it never runs a hidden O(nnz) conversion or second payload copy.
pub(super) enum FloatingStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
    C32(Vec<ComplexElement<f32>>),
    C64(Vec<ComplexElement<f64>>),
}
impl FloatingStorage {
    pub(super) fn new(dtype: NumericDType, complex: bool, len: usize) -> BuiltinResult<Self> {
        fn capacity<T>(len: usize) -> BuiltinResult<Vec<T>> {
            let mut v = Vec::new();
            v.try_reserve_exact(len)
                .map_err(|_| invalid("qr: output allocation failed"))?;
            Ok(v)
        }
        Ok(match (dtype, complex) {
            (NumericDType::F32, false) => Self::F32(capacity(len)?),
            (_, false) => Self::F64(capacity(len)?),
            (NumericDType::F32, true) => Self::C32(capacity(len)?),
            (_, true) => Self::C64(capacity(len)?),
        })
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::C32(v) => v.len(),
            Self::C64(v) => v.len(),
        }
    }
    pub(super) fn push(&mut self, re: f64, im: f64) {
        match self {
            Self::F32(v) => v.push(re as f32),
            Self::F64(v) => v.push(re),
            Self::C32(v) => v.push(ComplexElement(re as f32, im as f32)),
            Self::C64(v) => v.push(ComplexElement(re, im)),
        }
    }
    pub(super) fn dense(self, rows: usize, cols: usize) -> BuiltinResult<Value> {
        match self {
            Self::F32(v) => Tensor::from_f32(v, vec![rows, cols]).map(Value::Tensor),
            Self::F64(v) => Tensor::new(v, vec![rows, cols]).map(Value::Tensor),
            Self::C32(v) => {
                ComplexTensor::from_complex_storage(ComplexStorage::F32(v.into()), vec![rows, cols])
                    .map(Value::ComplexTensor)
            }
            Self::C64(v) => {
                ComplexTensor::from_complex_storage(ComplexStorage::F64(v.into()), vec![rows, cols])
                    .map(Value::ComplexTensor)
            }
        }
        .map_err(invalid)
    }
}

pub(super) enum Dense {
    Real(ExactHostTensorAllocation),
    Complex(ExactHostComplexTensorAllocation),
    Ordinary {
        values: FloatingStorage,
        rows: usize,
        cols: usize,
    },
}
impl Dense {
    pub(super) fn new(
        runtime: &RuntimeContext,
        receipt: Option<&HostAllocationReceipt>,
        dtype: NumericDType,
        complex: bool,
        rows: usize,
        cols: usize,
    ) -> BuiltinResult<Self> {
        if let Some(receipt) = receipt {
            let permit = receipt
                .output_permit(runtime)
                .map_err(|e| allocation(runtime, e))?;
            return if complex {
                ExactHostComplexTensorAllocation::new(dtype, &[rows, cols], &permit)
                    .map(Self::Complex)
            } else {
                ExactHostTensorAllocation::new(dtype, &[rows, cols], &permit).map(Self::Real)
            }
            .map_err(|e| allocation(runtime, e.into()));
        }
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid("qr: output dimensions overflow"))?;
        let values = FloatingStorage::new(dtype, complex, len)?;
        Ok(Self::Ordinary { values, rows, cols })
    }
    pub(super) fn set(
        &mut self,
        index: usize,
        re: f64,
        im: f64,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<()> {
        match self {
            Self::Real(t) => {
                if let Some(v) = t.as_f32_slice_mut() {
                    v[index] = re as f32;
                } else {
                    t.as_f64_slice_mut().expect("selected double")[index] = re;
                }
            }
            Self::Complex(t) => t
                .set(index, re, im)
                .map_err(|e| allocation(runtime, e.into()))?,
            Self::Ordinary { values, .. } => {
                if index != values.len() {
                    return Err(invalid("qr: nonsequential output write"));
                }
                values.push(re, im);
            }
        };
        Ok(())
    }
    pub(super) fn finish(self, runtime: &RuntimeContext) -> BuiltinResult<Value> {
        match self {
            Self::Real(t) => t
                .into_tensor()
                .map(Value::Tensor)
                .map_err(|e| allocation(runtime, e.into())),
            Self::Complex(t) => t
                .into_tensor()
                .map(Value::ComplexTensor)
                .map_err(|e| allocation(runtime, e.into())),
            Self::Ordinary { values, rows, cols } => values.dense(rows, cols),
        }
    }
}

fn sparse_class(dtype: NumericDType, complex: bool) -> ExactHostSparseClass {
    if complex {
        ExactHostSparseClass::Complex(dtype)
    } else {
        ExactHostSparseClass::Numeric(dtype)
    }
}
pub(super) fn sparse_bytes(
    dtype: NumericDType,
    complex: bool,
    m: usize,
    n: usize,
    nnz: usize,
) -> Result<u64, HostAllocationError> {
    sparse_publication_bytes(sparse_class(dtype, complex), m, n, nnz)
}
pub(super) struct Sparse<'a>(SparsePublication<'a>);
impl<'a> Sparse<'a> {
    pub(super) fn new(
        runtime: &RuntimeContext,
        receipt: Option<&'a HostAllocationReceipt>,
        dtype: NumericDType,
        complex: bool,
        m: usize,
        n: usize,
        nnz: usize,
    ) -> BuiltinResult<Self> {
        SparsePublication::with_receipt(runtime, receipt, sparse_class(dtype, complex), m, n, nnz)
            .map(Self)
            .map_err(|error| publication_error(runtime, error))
    }
    pub(super) fn ptr(&mut self, column: usize, offset: usize) -> BuiltinResult<()> {
        self.0
            .pointer(column, offset)
            .map_err(|error| publication_error_without_runtime(error))
    }
    pub(super) fn entry(
        &mut self,
        index: usize,
        row: usize,
        value: (f64, f64),
    ) -> BuiltinResult<()> {
        self.0
            .entry(index, row, value)
            .map_err(publication_error_without_runtime)
    }
    pub(super) async fn finish(
        self,
        runtime: &RuntimeContext,
        check: impl Fn() -> BuiltinResult<()>,
    ) -> BuiltinResult<Value> {
        self.0
            .finish_with(runtime, check, |error| publication_error(runtime, error))
            .await
    }
}

fn publication_error(runtime: &RuntimeContext, error: SparsePublicationError) -> RuntimeError {
    match error {
        SparsePublicationError::Admission(source) => allocation(runtime, source),
        SparsePublicationError::Execution(source) => {
            source.into_runtime_error_in_context(Some(runtime))
        }
        other => invalid(other.to_string()),
    }
}

fn publication_error_without_runtime(error: SparsePublicationError) -> RuntimeError {
    invalid(error.to_string())
}
