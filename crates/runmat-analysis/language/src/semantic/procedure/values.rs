use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaStatement, CheckedFeaStatementKind,
};

thread_local! {
    /// Member names a procedure lowering asked for; anything else in the block is unknown.
    static READ_MEMBERS: std::cell::RefCell<Option<std::collections::BTreeSet<String>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `lower` while recording every member name it reads with [`assignment`].
pub fn track_members<T>(lower: impl FnOnce() -> T) -> (T, std::collections::BTreeSet<String>) {
    let previous = READ_MEMBERS.with(|cell| cell.replace(Some(Default::default())));
    let value = lower();
    let read = READ_MEMBERS
        .with(|cell| cell.replace(previous))
        .unwrap_or_default();
    (value, read)
}

pub fn assignment<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a CheckedExpression> {
    READ_MEMBERS.with(|cell| {
        if let Some(read) = cell.borrow_mut().as_mut() {
            read.insert(name.to_string());
        }
    });
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
    let value = source
        .parse::<usize>()
        .ok()
        .and_then(runmat_types::wire_integer::bounded)?;
    (value > 0).then_some(value)
}

pub fn canonical_scalar(value: &CheckedExpression) -> Option<f64> {
    match &value.kind {
        CheckedExpressionKind::Number(source) => source.parse().ok(),
        CheckedExpressionKind::CadParameterValue(source) => Some(source.value_si()),
        CheckedExpressionKind::Quantity { value, unit, .. } => {
            unit.to_canonical(canonical_scalar(value)?).ok()
        }
        CheckedExpressionKind::Unary { operator, operand } => {
            let value = canonical_scalar(operand)?;
            Some(match operator {
                runmat_design_syntax::FeaUnaryOperator::Positive => value,
                runmat_design_syntax::FeaUnaryOperator::Negative => -value,
            })
        }
        _ => None,
    }
}

pub fn canonical_list(value: &CheckedExpression) -> Option<Vec<f64>> {
    match &value.kind {
        CheckedExpressionKind::List(values) => values.iter().map(canonical_scalar).collect(),
        CheckedExpressionKind::Quantity { value, unit, .. } => {
            let CheckedExpressionKind::List(values) = &value.kind else {
                return None;
            };
            let scale = unit.scale();
            let offset = unit.offset();
            values
                .iter()
                .map(|value| Some(canonical_scalar(value)? * scale + offset))
                .collect()
        }
        _ => None,
    }
}

pub fn canonical_vector3(value: &CheckedExpression) -> Option<[f64; 3]> {
    let values = match &value.kind {
        CheckedExpressionKind::Quantity { value, unit, .. } => {
            let CheckedExpressionKind::List(values) = &value.kind else {
                return None;
            };
            let scale = unit.scale();
            let offset = unit.offset();
            values
                .iter()
                .map(|component| Some(canonical_scalar(component)? * scale + offset))
                .collect::<Option<Vec<_>>>()?
        }
        CheckedExpressionKind::List(values) => values
            .iter()
            .map(canonical_scalar)
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    values.try_into().ok()
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
