use super::super::scalar_contract::*;
use runmat_types::*;
fn text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::String(s) | LiteralValue::Character(s) | LiteralValue::Keyword(s) => Some(s),
        _ => None,
    }
}
pub(super) fn valid(
    operation: SymbolicScalarOperation,
    index: usize,
    count: usize,
    literal: &LiteralValue,
) -> bool {
    let role = operation.operand(index, count);
    match literal {
        LiteralValue::Unknown => return true,
        LiteralValue::Empty | LiteralValue::Complex { .. } => return false,
        LiteralValue::Vector(values) => {
            return values.len() == 1 && valid(operation, index, count, &values[0])
        }
        LiteralValue::Matrix(rows) => {
            return rows.len() == 1
                && rows[0].len() == 1
                && valid(operation, index, count, &rows[0][0])
        }
        LiteralValue::Bool(_) => {
            if matches!(
                role,
                SymbolicScalarOperand::Variable
                    | SymbolicScalarOperand::Direction
                    | SymbolicScalarOperand::Digits
            ) {
                return false;
            }
        }
        LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. } => {
            if matches!(
                role,
                SymbolicScalarOperand::Variable | SymbolicScalarOperand::Direction
            ) {
                return false;
            }
        }
        LiteralValue::Symbolic(text) => {
            return match role {
                SymbolicScalarOperand::Variable => crate::is_valid_symbolic_identifier(text.trim()),
                SymbolicScalarOperand::Direction
                | SymbolicScalarOperand::Digits
                | SymbolicScalarOperand::Precision => false,
                _ => true,
            }
        }
        _ => {}
    }
    if let Some(text) = text(literal) {
        return match role {
            SymbolicScalarOperand::Variable | SymbolicScalarOperand::VariableOrPoint => {
                crate::is_valid_symbolic_identifier(text.trim())
            }
            SymbolicScalarOperand::Direction => {
                SymbolicLimitDirection::from_public_option(text).is_some()
            }
            SymbolicScalarOperand::Digits => {
                text.trim().eq_ignore_ascii_case("default")
                    || text
                        .trim()
                        .parse::<f64>()
                        .ok()
                        .and_then(|n| SymbolicPrecisionKind::Default.rounded(n))
                        .is_some()
            }
            SymbolicScalarOperand::Precision => false,
            _ => true,
        };
    }
    if matches!(
        role,
        SymbolicScalarOperand::Digits | SymbolicScalarOperand::Precision
    ) {
        let precision = if role == SymbolicScalarOperand::Digits {
            SymbolicPrecisionKind::Default
        } else {
            SymbolicPrecisionKind::Decimal
        };
        if let Some(n) = literal.integer_value_i128() {
            return usize::try_from(n)
                .ok()
                .is_some_and(|n| precision.accepts(n));
        }
        if let LiteralValue::Bool(value) = literal {
            return precision.rounded(if *value { 1.0 } else { 0.0 }).is_some();
        }
        return LiteralContext::numeric_from_literal(literal)
            .is_none_or(|n| precision.rounded(n).is_some());
    }
    if matches!(
        operation,
        SymbolicScalarOperation::Integrate | SymbolicScalarOperation::Limit
    ) && literal.integer_is_exact_binary64() == Some(false)
    {
        return false;
    }
    if matches!(
        role,
        SymbolicScalarOperand::Point | SymbolicScalarOperand::VariableOrPoint
    ) {
        return LiteralContext::numeric_from_literal(literal).is_none_or(f64::is_finite);
    }
    true
}
pub(super) fn unsupported_int_option(request: &CallRequest) -> bool {
    request.arguments.len() >= 3
        && request
            .literals
            .literal_args
            .get(request.arguments.len() - 2)
            .and_then(text)
            .and_then(SymbolicIntegrationOption::from_public_option)
            .is_some()
}
