use crate::accel::auto_promote::{
    accel_promote_binary, accel_promote_unary, AutoBinaryOp, AutoUnaryOp,
};
use crate::interpreter::dispatch::logical_truth_from_value;
use crate::ops::arithmetic as arithmetic_ops;
use crate::ops::comparison as comparison_ops;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

async fn call_operator_method(obj: Value, method: &str, arg: Value) -> Result<Value, RuntimeError> {
    runmat_runtime::object::dispatch::call_object_operator_method(obj, method, arg).await
}

async fn call_right_operator_method_ordered(
    lhs: Value,
    rhs: Value,
    method: &str,
) -> Result<Value, RuntimeError> {
    runmat_runtime::object::dispatch::call_rhs_object_operator_method_ordered(lhs, rhs, method)
        .await
}

mod operations;

use futures::{
    future::LocalBoxFuture,
    future::{ready, Either, Ready},
};

type ArithmeticDispatch<'a> =
    Either<Ready<Result<bool, RuntimeError>>, LocalBoxFuture<'a, Result<bool, RuntimeError>>>;

// Choose the operation before constructing its future. Unrelated instruction
// families need neither an arithmetic frame nor an allocation.
pub fn dispatch_arithmetic<'a>(
    instr: &crate::bytecode::Instr,
    stack: &'a mut Vec<Value>,
) -> ArithmeticDispatch<'a> {
    match instr {
        crate::bytecode::Instr::Add => Either::Right(Box::pin(operations::add(stack))),
        crate::bytecode::Instr::Sub => Either::Right(Box::pin(operations::sub(stack))),
        crate::bytecode::Instr::Mul => Either::Right(Box::pin(operations::mul(stack))),
        crate::bytecode::Instr::ElemMul => Either::Right(Box::pin(operations::elem_mul(stack))),
        crate::bytecode::Instr::ElemDiv => Either::Right(Box::pin(operations::elem_div(stack))),
        crate::bytecode::Instr::ElemPow => Either::Right(Box::pin(operations::elem_pow(stack))),
        crate::bytecode::Instr::ElemLeftDiv => {
            Either::Right(Box::pin(operations::elem_left_div(stack)))
        }
        crate::bytecode::Instr::RightDiv => Either::Right(Box::pin(operations::right_div(stack))),
        crate::bytecode::Instr::LeftDiv => Either::Right(Box::pin(operations::left_div(stack))),
        crate::bytecode::Instr::Neg => Either::Right(Box::pin(operations::neg(stack))),
        crate::bytecode::Instr::Pow => Either::Right(Box::pin(operations::pow(stack))),
        crate::bytecode::Instr::UPlus => Either::Right(Box::pin(operations::u_plus(stack))),
        crate::bytecode::Instr::Transpose => Either::Right(Box::pin(operations::transpose(stack))),
        crate::bytecode::Instr::ConjugateTranspose => {
            Either::Right(Box::pin(operations::conjugate_transpose(stack)))
        }
        crate::bytecode::Instr::LessEqual => Either::Right(Box::pin(operations::less_equal(stack))),
        crate::bytecode::Instr::Less => Either::Right(Box::pin(operations::less(stack))),
        crate::bytecode::Instr::Greater => Either::Right(Box::pin(operations::greater(stack))),
        crate::bytecode::Instr::GreaterEqual => {
            Either::Right(Box::pin(operations::greater_equal(stack)))
        }
        crate::bytecode::Instr::Equal => Either::Right(Box::pin(operations::equal(stack))),
        crate::bytecode::Instr::NotEqual => Either::Right(Box::pin(operations::not_equal(stack))),
        crate::bytecode::Instr::LogicalNot => {
            Either::Right(Box::pin(operations::logical_not(stack)))
        }
        crate::bytecode::Instr::LogicalAnd => {
            Either::Right(Box::pin(operations::logical_and(stack)))
        }
        crate::bytecode::Instr::LogicalOr => Either::Right(Box::pin(operations::logical_or(stack))),
        _ => Either::Left(ready(Ok(false))),
    }
}
