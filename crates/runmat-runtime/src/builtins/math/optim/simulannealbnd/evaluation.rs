//! One actual objective call per measurement; every result stays rooted.
use super::{common, inputs, invalid, ScalarCooperation};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_value::Value;
pub(super) struct Evaluator {
    pub function: RootedValue,
    pub count: usize,
}
impl Evaluator {
    pub async fn evaluate(&mut self, x: &Value, c: &mut ScalarCooperation) -> BuiltinResult<f64> {
        c.completed(0).await?;
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| invalid("function count overflow"))?;
        let result = common::call_function_in_context(
            c.runtime(),
            self.function.value(),
            std::slice::from_ref(x),
            1,
        )
        .await?;
        c.completed(0).await?;
        let value = inputs::scalar(result.value(), "objective value")?;
        if !value.is_finite() {
            return Err(invalid("objective must return a finite real double scalar"));
        }
        Ok(value)
    }
}
pub(super) async fn invoke(
    target: &Value,
    args: &[Value],
    outputs: usize,
    c: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    c.completed(0).await?;
    let result =
        common::call_function_sequence_in_context(c.runtime(), target, args, outputs).await?;
    c.completed(0).await?;
    Ok(result)
}
