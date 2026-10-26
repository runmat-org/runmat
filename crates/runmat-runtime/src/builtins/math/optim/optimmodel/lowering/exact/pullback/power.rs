//! Identity/zero powers retain the original affine path. Source-certified
//! squares use actual candidate operands and the same two-sided product rule.
use super::*;
use runmat_types::OperatorKind as Op;
pub(super) struct Power<'a> {
    pub source: GcHandle,
    pub operator: Op,
    pub exponents: &'a super::super::super::powers::Exponents,
    pub shape: &'a [usize],
}
pub(super) async fn propagate(
    state: &mut State<'_>,
    operands: &Evaluated,
    power: Power<'_>,
    weights: Weights,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let Power {
        source,
        operator,
        exponents,
        shape,
    } = power;
    use crate::builtins::common::broadcast::{broadcast_index, compute_strides};
    let input = state.shape(source)?.to_vec();
    if operator == Op::MatrixPower
        && evaluation::element_count(&input, state.policy)? != 1
        && exponents.at(0, state.policy)? == 2.
    {
        if !operands.retains_operands() {
            return Err(state
                .policy
                .invalid("original matrix square requires retained candidate operands"));
        }
        return binary::propagate(
            state,
            operands,
            source,
            source,
            Op::MatrixMultiply,
            shape,
            weights,
            c,
        )
        .await;
    }
    let strides = compute_strides(&input);
    for (row, value) in weights {
        let source_row = broadcast_index(row, shape, &input, &strides);
        if state.active_at(source, source_row)? {
            let exponent = exponents.broadcast(row, shape, state.policy)?;
            if exponent == 1. {
                state.add(source, source_row, value, c).await?;
            } else if exponent == 2. {
                if !operands.retains_operands() {
                    return Err(state
                        .policy
                        .invalid("original scalar square requires retained candidate operands"));
                }
                let operand = operands
                    .value(source, source_row, state.policy)?
                    .ok_or_else(|| {
                        state
                            .policy
                            .invalid("original square operand is unassigned")
                    })?;
                let slope = Rational::from_usize(2)
                    .mul(&operand, state.policy, c)
                    .await?;
                state
                    .add(
                        source,
                        source_row,
                        value.mul(&slope, state.policy, c).await?,
                        c,
                    )
                    .await?;
            } else if exponent != 0. {
                return Err(state
                    .policy
                    .invalid("unadmitted power reached original reverse"));
            }
        }
        c.completed(1).await?;
    }
    Ok(())
}
