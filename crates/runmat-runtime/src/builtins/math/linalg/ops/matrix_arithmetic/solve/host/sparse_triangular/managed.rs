//! Borrowed CSC substitution directly in the shared exact final output.
use super::super::managed_output::ManagedSolveOutput;
use super::{admission, invalid, map_operator};
use crate::builtins::math::linalg::ops::linear_operator::OperatorWork;
use crate::builtins::math::linalg::ops::triangular_operator::{
    TriangularLinearOperator, TriangularOperator, TriangularOperatorError,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_value::{ComplexElement, Value};

macro_rules! solve {
    ($method:ident,$scalar:ty) => {
        async fn $method(
            operator: &TriangularOperator<'_, $scalar>,
            input: &[$scalar],
            columns: usize,
            context: &RuntimeContext,
        ) -> BuiltinResult<Value> {
            let n = operator.order();
            if n.checked_mul(columns) != Some(input.len()) {
                return Err(invalid("mldivide: triangular RHS cardinality differs"));
            }
            let mut output = ManagedSolveOutput::<$scalar>::new(&[n, columns], context)
                .map_err(|e| admission(e, context))?;
            let mut work =
                OperatorWork::new(context).map_err(|e| map_operator(e.into(), context))?;
            let target = output.as_mut_slice().map_err(|e| admission(e, context))?;
            // One work budget crosses all RHS columns; tiny columns do not
            // reset the execution quantum and starve cancellation/other tasks.
            for column in 0..columns {
                let range = column * n..(column + 1) * n;
                operator
                    .apply_with_work(&input[range.clone()], &mut target[range], &mut work)
                    .await
                    .map_err(|e| map_operator(e, context))?;
            }
            work.flush()
                .await
                .map_err(|e| map_operator(e.into(), context))?;
            output.finish(context).map_err(|e| admission(e, context))
        }
    };
}
solve!(real_f32, f32);
solve!(real_f64, f64);
solve!(complex_f32, ComplexElement<f32>);
solve!(complex_f64, ComplexElement<f64>);

pub(super) async fn solve(
    operator: TriangularLinearOperator<'_>,
    rhs: &Value,
    columns: usize,
    context: &RuntimeContext,
) -> BuiltinResult<Value> {
    match (operator, rhs) {
        (TriangularLinearOperator::RealF32(operator), Value::Tensor(rhs)) => {
            real_f32(
                &operator,
                rhs.as_f32_slice()
                    .ok_or_else(|| invalid("RHS class changed"))?,
                columns,
                context,
            )
            .await
        }
        (TriangularLinearOperator::RealF64(operator), Value::Tensor(rhs)) => {
            real_f64(
                &operator,
                rhs.as_f64_slice()
                    .ok_or_else(|| invalid("RHS class changed"))?,
                columns,
                context,
            )
            .await
        }
        (TriangularLinearOperator::ComplexF32(operator), Value::ComplexTensor(rhs)) => {
            complex_f32(
                &operator,
                rhs.as_f32_slice()
                    .ok_or_else(|| invalid("RHS class changed"))?,
                columns,
                context,
            )
            .await
        }
        (TriangularLinearOperator::ComplexF64(operator), Value::ComplexTensor(rhs)) => {
            complex_f64(
                &operator,
                rhs.as_f64_slice()
                    .ok_or_else(|| invalid("RHS class changed"))?,
                columns,
                context,
            )
            .await
        }
        (TriangularLinearOperator::RealF64(operator), Value::Num(input)) => {
            let mut output = [0.0];
            operator
                .apply_cooperatively(std::slice::from_ref(input), &mut output, context)
                .await
                .map_err(|e| map_operator(e, context))?;
            Ok(Value::Num(output[0]))
        }
        (TriangularLinearOperator::ComplexF64(operator), Value::Complex(real, imag)) => {
            let input = [ComplexElement(*real, *imag)];
            let mut output = [ComplexElement(0.0, 0.0)];
            operator
                .apply_cooperatively(&input, &mut output, context)
                .await
                .map_err(|e| map_operator(e, context))?;
            Ok(Value::Complex(output[0].0, output[0].1))
        }
        _ => Err(map_operator(
            TriangularOperatorError::UnsupportedClass,
            context,
        )),
    }
}

#[cfg(test)]
mod tests;
