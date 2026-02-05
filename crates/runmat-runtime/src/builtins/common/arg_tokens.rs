use runmat_builtins::{LiteralValue, ResolveContext, Value};

#[derive(Clone, Debug, PartialEq)]
pub enum ArgToken {
    Number(f64),
    Bool(bool),
    String(String),
    Vector(Vec<ArgToken>),
    Unknown,
}

pub fn tokens_from_values(args: &[Value]) -> Vec<ArgToken> {
    args.iter().map(token_from_value).collect()
}

pub fn tokens_from_context(ctx: &ResolveContext) -> Vec<ArgToken> {
    ctx.literal_args.iter().map(token_from_literal).collect()
}

fn token_from_literal(value: &LiteralValue) -> ArgToken {
    match value {
        LiteralValue::Number(num) => ArgToken::Number(*num),
        LiteralValue::Bool(value) => ArgToken::Bool(*value),
        LiteralValue::String(text) => ArgToken::String(text.to_ascii_lowercase()),
        LiteralValue::Vector(values) => {
            ArgToken::Vector(values.iter().map(token_from_literal).collect())
        }
        LiteralValue::Unknown => ArgToken::Unknown,
    }
}

fn token_from_value(value: &Value) -> ArgToken {
    match value {
        Value::Num(num) => ArgToken::Number(*num),
        Value::Int(value) => ArgToken::Number(value.to_f64()),
        Value::Bool(value) => ArgToken::Bool(*value),
        Value::String(text) => ArgToken::String(text.to_ascii_lowercase()),
        Value::StringArray(arr) if arr.data.len() == 1 => {
            ArgToken::String(arr.data[0].to_ascii_lowercase())
        }
        Value::CharArray(arr) if arr.rows == 1 => {
            let text: String = arr.data.iter().collect();
            ArgToken::String(text.to_ascii_lowercase())
        }
        Value::Tensor(tensor) => token_from_tensor(&tensor.data, &tensor.shape),
        Value::LogicalArray(arr) => token_from_logical(&arr.data, &arr.shape),
        _ => ArgToken::Unknown,
    }
}

fn token_from_tensor(data: &[f64], shape: &[usize]) -> ArgToken {
    if data.len() == 1 {
        return ArgToken::Number(data[0]);
    }
    if is_vector_shape(shape) {
        return ArgToken::Vector(data.iter().copied().map(ArgToken::Number).collect());
    }
    ArgToken::Unknown
}

fn token_from_logical(data: &[u8], shape: &[usize]) -> ArgToken {
    if data.len() == 1 {
        return ArgToken::Bool(data[0] != 0);
    }
    if is_vector_shape(shape) {
        return ArgToken::Vector(data.iter().map(|b| ArgToken::Bool(*b != 0)).collect());
    }
    ArgToken::Unknown
}

fn is_vector_shape(shape: &[usize]) -> bool {
    if shape.is_empty() {
        return false;
    }
    if shape.len() == 1 {
        return true;
    }
    if shape.len() == 2 {
        return shape[0] == 1 || shape[1] == 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntValue, LiteralValue, ResolveContext};

    #[test]
    fn tokens_from_context_lowercases_strings() {
        let ctx = ResolveContext::new(vec![LiteralValue::String("OmItNaN".to_string())]);
        assert_eq!(
            tokens_from_context(&ctx),
            vec![ArgToken::String("omitnan".to_string())]
        );
    }

    #[test]
    fn tokens_from_context_handles_vectors() {
        let ctx = ResolveContext::new(vec![LiteralValue::Vector(vec![
            LiteralValue::Number(1.0),
            LiteralValue::Bool(true),
        ])]);
        assert_eq!(
            tokens_from_context(&ctx),
            vec![ArgToken::Vector(vec![
                ArgToken::Number(1.0),
                ArgToken::Bool(true)
            ])]
        );
    }

    #[test]
    fn tokens_from_values_handles_scalar_inputs() {
        let args = vec![
            Value::Num(2.0),
            Value::Int(IntValue::I32(3)),
            Value::Bool(true),
            Value::String("All".to_string()),
        ];
        assert_eq!(
            tokens_from_values(&args),
            vec![
                ArgToken::Number(2.0),
                ArgToken::Number(3.0),
                ArgToken::Bool(true),
                ArgToken::String("all".to_string()),
            ]
        );
    }

    #[test]
    fn tokens_from_values_handles_vector_tensor() {
        let tensor = runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        assert_eq!(
            tokens_from_values(&args),
            vec![ArgToken::Vector(vec![
                ArgToken::Number(1.0),
                ArgToken::Number(2.0)
            ])]
        );
    }
}
