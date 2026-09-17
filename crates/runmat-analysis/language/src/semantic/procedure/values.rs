use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaStatement, CheckedFeaStatementKind,
};

pub fn assignment<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a CheckedExpression> {
    for statement in statements {
        match &statement.kind {
            CheckedFeaStatementKind::Assignment { target, value, .. }
                if target.last().is_some_and(|segment| segment == name) =>
            {
                return Some(value);
            }
            CheckedFeaStatementKind::Block { statements, .. } => {
                if let Some(value) = assignment(statements, name) {
                    return Some(value);
                }
            }
            _ => {}
        }
    }
    None
}

pub fn positive_usize(value: &CheckedExpression) -> Option<usize> {
    let CheckedExpressionKind::Number(source) = &value.kind else {
        return None;
    };
    let value = source.parse::<usize>().ok()?;
    (value > 0).then_some(value)
}

pub fn canonical_scalar(value: &CheckedExpression) -> Option<f64> {
    match &value.kind {
        CheckedExpressionKind::Number(source) => source.parse().ok(),
        CheckedExpressionKind::Quantity {
            value,
            scale_bits,
            offset_bits,
        } => Some(
            canonical_scalar(value)? * f64::from_bits(*scale_bits) + f64::from_bits(*offset_bits),
        ),
        CheckedExpressionKind::Unary { operator, operand } => {
            let value = canonical_scalar(operand)?;
            Some(match operator {
                runmat_parser::fea::FeaUnaryOperator::Positive => value,
                runmat_parser::fea::FeaUnaryOperator::Negative => -value,
            })
        }
        _ => None,
    }
}

pub fn canonical_list(value: &CheckedExpression) -> Option<Vec<f64>> {
    let CheckedExpressionKind::List(values) = &value.kind else {
        return None;
    };
    values.iter().map(canonical_scalar).collect()
}

pub fn reference(value: &CheckedExpression) -> Option<&[String]> {
    match &value.kind {
        CheckedExpressionKind::Reference(path) => Some(path),
        _ => None,
    }
}

pub fn retained_name(statements: &[CheckedFeaStatement]) -> Option<String> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Retain(value) = &statement.kind else {
            return None;
        };
        match &value.kind {
            CheckedExpressionKind::Enum(name) => Some(name.clone()),
            CheckedExpressionKind::Reference(path) if path.len() == 1 => Some(path[0].clone()),
            _ => None,
        }
    })
}
