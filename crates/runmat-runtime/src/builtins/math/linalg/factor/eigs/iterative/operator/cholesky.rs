//! A supplied R and retained permutation represent B(p,p)=R'*R.
//! Two borrowed triangular solves/products use caller-owned native scratch.
use super::{Scalar, Work};
use crate::{
    builtins::math::linalg::ops::{
        iterative_preconditioner::MatrixFactorError,
        linear_operator::{LinearOperator, ProductOrientation},
        triangular_operator::{Diagonal, Triangle, TriangularLinearOperator},
    },
    context::RuntimeContext,
    BuiltinResult,
};
use runmat_value::Value;

struct Permutation<'a> {
    source: &'a Value,
    order: usize,
}
fn index(value: &Value, position: usize, n: usize) -> BuiltinResult<usize> {
    let integer = match value {
        Value::Tensor(t) => t
            .numeric_value_at(position)
            .and_then(super::super::super::numeric_scalar_i128),
        Value::Int(v) => Some(super::super::super::int_value_i128(v)),
        _ => None,
    };
    if let Some(integer) = integer {
        if integer < 1 || integer > n as i128 {
            return Err(super::super::invalid(
                "CholeskyPermutation index is outside 1:n",
            ));
        }
        return Ok(integer as usize - 1);
    }
    let raw = super::super::basis::element(value, position)?;
    if raw.im != 0.0
        || !raw.re.is_finite()
        || raw.re.fract() != 0.0
        || raw.re < 1.0
        || raw.re >= usize::MAX as f64
        || raw.re > n as f64
    {
        return Err(super::super::invalid(
            "CholeskyPermutation requires 1-based integer indices",
        ));
    }
    Ok(raw.re as usize - 1)
}
impl<'a> Permutation<'a> {
    async fn validate(
        source: &'a Value,
        n: usize,
        marks: &mut [usize],
        work: &mut Work<'_>,
    ) -> BuiltinResult<Self> {
        if super::super::super::numeric_vector_len(source, "CholeskyPermutation")? != n
            || marks.len() != n
        {
            return Err(super::super::invalid(
                "CholeskyPermutation length must equal matrix order",
            ));
        }
        for mark in marks.iter_mut() {
            *mark = 0;
            work.tick().await?;
        }
        for position in 0..n {
            let row = index(source, position, n)?;
            if marks[row] != 0 {
                return Err(super::super::invalid(
                    "CholeskyPermutation indices must be unique",
                ));
            }
            marks[row] = 1;
            work.tick().await?;
        }
        work.flush().await?;
        Ok(Self { source, order: n })
    }
    fn get(&self, position: usize) -> BuiltinResult<usize> {
        index(self.source, position, self.order)
    }
}
pub(in crate::builtins::math::linalg::factor::eigs::iterative) struct Cholesky<'a> {
    upper: TriangularLinearOperator<'a>,
    product: LinearOperator<'a>,
    permutation: Option<Permutation<'a>>,
    n: usize,
}
impl<'a> Cholesky<'a> {
    pub async fn preflight(
        r: &'a Value,
        permutation: Option<&'a Value>,
        n: usize,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<Self> {
        use crate::context::host_allocation::{
            HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
        };
        let count = if permutation.is_some() { n } else { 0 };
        let quote = HostTypedWorkspace::<usize>::quoted_bytes(count)
            .map_err(|e| super::super::admission(runtime, e))?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(quote, 0).map_err(|e| super::super::admission(runtime, e))?,
        )
        .map_err(|e| super::super::admission(runtime, e))?;
        let mut work = Work::new(runtime, receipt.as_ref())?;
        let mut marks =
            super::super::storage::Buffer::<usize>::new(count, runtime, receipt.as_ref()).await?;
        let result = Self::prepare(r, permutation, n, marks.get_mut(), &mut work).await?;
        work.flush().await?;
        Ok(result)
    }
    pub fn order(&self) -> usize {
        self.n
    }
    async fn prepare(
        r: &'a Value,
        permutation: Option<&'a Value>,
        n: usize,
        marks: &mut [usize],
        work: &mut Work<'_>,
    ) -> BuiltinResult<Self> {
        let runtime = work.runtime.ok_or_else(|| {
            super::super::unsupported("supplied Cholesky action requires execution context")
        })?;
        work.check()?;
        let upper = TriangularLinearOperator::borrow_cooperatively(
            r,
            Triangle::Upper,
            Diagonal::Explicit,
            runtime,
        )
        .await
        .map_err(|e| super::super::factor_error(Some(runtime), MatrixFactorError::Triangular(e)))?;
        let product = LinearOperator::borrow_cooperatively(r, runtime)
            .await
            .map_err(|e| super::super::operator_error(Some(runtime), e))?;
        let permutation = match permutation {
            Some(value) => Some(Permutation::validate(value, n, marks, work).await?),
            None => None,
        };
        work.check()?;
        Ok(Self {
            upper,
            product,
            permutation,
            n,
        })
    }
    fn index(&self, index: usize) -> BuiltinResult<usize> {
        self.permutation
            .as_ref()
            .map_or(Ok(index), |p| p.get(index))
    }
    pub async fn apply<T: Scalar>(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        work: &mut Work<'_>,
    ) -> BuiltinResult<()> {
        self.execute(input, output, scratch, Action::Product, work)
            .await
    }
    pub async fn solve<T: Scalar>(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        work: &mut Work<'_>,
    ) -> BuiltinResult<()> {
        self.execute(input, output, scratch, Action::Inverse, work)
            .await
    }
    async fn execute<T: Scalar>(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        action: Action,
        work: &mut Work<'_>,
    ) -> BuiltinResult<()> {
        if input.len() != self.n || output.len() != self.n || scratch.len() < 2 * self.n {
            return Err(super::super::invalid(
                "Cholesky action workspace geometry differs",
            ));
        }
        let runtime = work
            .runtime
            .ok_or_else(|| super::super::unsupported("Cholesky action lost execution context"))?;
        let (first, tail) = scratch.split_at_mut(self.n);
        let second = &mut tail[..self.n];
        for (row, destination) in first.iter_mut().enumerate() {
            *destination = input[self.index(row)?];
            work.tick().await?;
        }
        match action {
            Action::Product => {
                self.product
                    .apply_converted(first, second, T::native, Some(runtime))
                    .await
                    .map_err(|e| super::super::operator_error(Some(runtime), e))?;
                self.product
                    .apply_oriented_converted(
                        second,
                        first,
                        T::native,
                        Some(runtime),
                        ProductOrientation::Adjoint,
                    )
                    .await
                    .map_err(|e| super::super::operator_error(Some(runtime), e))?;
            }
            Action::Inverse => {
                self.solve_factor(first, second, ProductOrientation::Adjoint, runtime)
                    .await?;
                self.solve_factor(second, first, ProductOrientation::Normal, runtime)
                    .await?;
            }
        }
        for (row, &value) in first.iter().enumerate() {
            output[self.index(row)?] = value;
            work.tick().await?;
        }
        work.flush().await
    }
    async fn solve_factor<T: Scalar>(
        &self,
        input: &[T],
        output: &mut [T],
        orientation: ProductOrientation,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<()> {
        self.upper
            .solve_converted(input, output, T::native, orientation, runtime)
            .await
            .map_err(|e| {
                super::super::factor_error(Some(runtime), MatrixFactorError::Triangular(e))
            })
    }
}
#[derive(Clone, Copy)]
enum Action {
    Product,
    Inverse,
}
