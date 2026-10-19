use runmat_builtins::RelationalOperator;
use runmat_value::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeIdentity {
    Handle { address: usize, epoch: usize },
    Listener(u64),
}

pub(super) fn compare(lhs: &Value, rhs: &Value, operator: RelationalOperator) -> Option<Value> {
    if !operator.is_equality() {
        return None;
    }
    match (identity(lhs), identity(rhs)) {
        (Some(lhs), Some(rhs)) => Some(Value::Bool(apply(operator, lhs == rhs))),
        (Some(_), None) | (None, Some(_)) => Some(Value::Bool(apply(operator, false))),
        (None, None) => None,
    }
}

fn identity(value: &Value) -> Option<RuntimeIdentity> {
    match value {
        Value::HandleObject(handle) => Some(RuntimeIdentity::Handle {
            address: handle.target.addr(),
            epoch: handle.target.epoch(),
        }),
        Value::Listener(listener) => Some(RuntimeIdentity::Listener(listener.id)),
        _ => None,
    }
}

fn apply(operator: RelationalOperator, equal: bool) -> bool {
    match operator {
        RelationalOperator::Equal => equal,
        RelationalOperator::NotEqual => !equal,
        _ => unreachable!("identity comparison only supports equality operators"),
    }
}
