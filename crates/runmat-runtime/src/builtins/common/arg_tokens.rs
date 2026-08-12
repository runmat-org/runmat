use runmat_builtins::{LiteralValue, ResolveContext};
use runmat_value::{IntValue, Value};

use crate::builtins::common::tensor;

#[derive(Clone, Debug, PartialEq)]
pub enum ArgToken {
    Number(f64),
    Integer(IntValue),
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
        Value::Int(value) => ArgToken::Integer(value.clone()),
        Value::Bool(value) => ArgToken::Bool(*value),
        Value::String(text) => ArgToken::String(text.to_ascii_lowercase()),
        Value::StringArray(arr) if arr.data.len() == 1 => {
            ArgToken::String(arr.data[0].to_ascii_lowercase())
        }
        Value::CharArray(arr) if arr.rows == 1 => {
            let text: String = arr.data.iter().collect();
            ArgToken::String(text.to_ascii_lowercase())
        }
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return token_from_integer_storage(storage, &tensor.shape);
            }
            let values = tensor::tensor_values_f64_cow(tensor);
            token_from_tensor(values.as_ref(), &tensor.shape)
        }
        Value::LogicalArray(arr) => token_from_logical(&arr.data, &arr.shape),
        _ => ArgToken::Unknown,
    }
}

fn token_from_integer_storage(storage: &runmat_value::IntegerStorage, shape: &[usize]) -> ArgToken {
    if storage.len() == 1 {
        return ArgToken::Integer(storage.value_at(0).expect("one-element integer storage"));
    }
    if is_vector_shape(shape) {
        return ArgToken::Vector(
            storage
                .exact_values()
                .into_iter()
                .map(ArgToken::Integer)
                .collect(),
        );
    }
    ArgToken::Unknown
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
    use runmat_builtins::{LiteralValue, ResolveContext};
    use runmat_value::{IntValue, IntegerStorage, NumericStorage, Tensor};

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
                ArgToken::Integer(IntValue::I32(3)),
                ArgToken::Bool(true),
                ArgToken::String("all".to_string()),
            ]
        );
    }

    #[test]
    fn tokens_from_values_handles_vector_tensor() {
        let tensor = runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        assert_eq!(
            tokens_from_values(&args),
            vec![ArgToken::Vector(vec![
                ArgToken::Number(1.0),
                ArgToken::Number(2.0)
            ])]
        );
    }

    #[test]
    fn tokens_from_values_reads_native_single_storage() {
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25, -2.5]), vec![1, 2])
                .unwrap();
        assert_eq!(
            tokens_from_values(&[Value::Tensor(tensor)]),
            vec![ArgToken::Vector(vec![
                ArgToken::Number(1.25),
                ArgToken::Number(-2.5),
            ])]
        );
    }

    #[test]
    fn tokens_from_values_preserves_exact_integer_tensors() {
        let scalar = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
        assert_eq!(
            tokens_from_values(&[Value::Tensor(scalar)]),
            vec![ArgToken::Integer(IntValue::U64(u64::MAX))]
        );

        let vector = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap();
        assert_eq!(
            tokens_from_values(&[Value::Tensor(vector)]),
            vec![ArgToken::Vector(vec![
                ArgToken::Integer(IntValue::U64(9_007_199_254_740_993)),
                ArgToken::Integer(IntValue::U64(u64::MAX)),
            ])]
        );
    }
}
