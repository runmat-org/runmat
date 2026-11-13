//! LU result adapters; the common sparse publication owner retains every
//! sparse payload and its original output authority independently.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::{
    SparsePublication, SparsePublicationError,
};
use runmat_value::{ExactHostSparseClass, ExactHostTensorAllocation, Tensor};

fn publication_error(runtime: &RuntimeContext, error: SparsePublicationError) -> RuntimeError {
    match error {
        SparsePublicationError::Admission(error) => allocation(runtime, error),
        SparsePublicationError::Execution(error) => {
            error.into_runtime_error_in_context(Some(runtime))
        }
        other => crate::build_runtime_error(format!("lu: {other}"))
            .with_builtin("lu")
            .with_identifier("RunMat:lu:InvalidInput")
            .with_source(other)
            .build(),
    }
}
pub(super) struct Sparse<'a> {
    value: SparsePublication<'a>,
    runtime: &'a RuntimeContext,
}
impl<'a> Sparse<'a> {
    pub(super) fn new(
        runtime: &'a RuntimeContext,
        _receipt: Option<&HostAllocationReceipt>,
        dtype: NumericDType,
        complex: bool,
        m: usize,
        n: usize,
        nnz: usize,
    ) -> BuiltinResult<Self> {
        SparsePublication::new(
            runtime,
            if complex {
                ExactHostSparseClass::Complex(dtype)
            } else {
                ExactHostSparseClass::Numeric(dtype)
            },
            m,
            n,
            nnz,
        )
        .map(|value| Self { value, runtime })
        .map_err(|e| publication_error(runtime, e))
    }
    pub(super) fn ptr(&mut self, column: usize, offset: usize) -> BuiltinResult<()> {
        self.value
            .checkpoint(self.runtime)
            .map_err(|e| publication_error(self.runtime, e))?;
        self.value
            .pointer(column, offset)
            .map_err(|e| publication_error(self.runtime, e))
    }
    pub(super) fn entry(
        &mut self,
        index: usize,
        row: usize,
        value: (f64, f64),
        _dtype: NumericDType,
        _complex: bool,
    ) -> BuiltinResult<()> {
        self.value
            .checkpoint(self.runtime)
            .map_err(|e| publication_error(self.runtime, e))?;
        self.value
            .entry(index, row, value)
            .map_err(|e| publication_error(self.runtime, e))
    }
    pub(super) async fn finish(
        self,
        runtime: &RuntimeContext,
        check: impl Fn() -> BuiltinResult<()>,
    ) -> BuiltinResult<Value> {
        self.value
            .finish_with(runtime, check, |e| publication_error(runtime, e))
            .await
    }
}
pub(super) enum Dense {
    Managed(ExactHostTensorAllocation),
    Ordinary {
        values: Vec<f64>,
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
        if dtype != NumericDType::F64 || complex {
            return Err(invalid("lu: permutation vector must be real double"));
        }
        if let Some(receipt) = receipt {
            let permit = receipt
                .output_permit(runtime)
                .map_err(|e| allocation(runtime, e))?;
            return ExactHostTensorAllocation::new(dtype, &[rows, cols], &permit)
                .map(Self::Managed)
                .map_err(|e| allocation(runtime, e.into()));
        }
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid("lu: vector shape overflow"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| invalid("lu: vector allocation failed"))?;
        Ok(Self::Ordinary { values, rows, cols })
    }
    pub(super) fn set(
        &mut self,
        index: usize,
        re: f64,
        im: f64,
        _runtime: &RuntimeContext,
    ) -> BuiltinResult<()> {
        if im != 0.0 {
            return Err(invalid("lu: permutation vector must be real"));
        }
        match self {
            Self::Managed(t) => t.as_f64_slice_mut().expect("admitted native double")[index] = re,
            Self::Ordinary { values, .. } => {
                if index != values.len() {
                    return Err(invalid("lu: nonsequential vector write"));
                }
                values.push(re);
            }
        }
        Ok(())
    }
    pub(super) fn finish(self, runtime: &RuntimeContext) -> BuiltinResult<Value> {
        match self {
            Self::Managed(t) => t
                .into_tensor()
                .map(Value::Tensor)
                .map_err(|e| allocation(runtime, e.into())),
            Self::Ordinary { values, rows, cols } => Tensor::new(values, vec![rows, cols])
                .map(Value::Tensor)
                .map_err(invalid),
        }
    }
}
