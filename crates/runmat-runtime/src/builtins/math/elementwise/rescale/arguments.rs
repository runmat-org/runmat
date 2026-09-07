use runmat_value::Value;

use crate::builtins::common::random_args::keyword_of;
use crate::BuiltinResult;

use super::error;

pub(super) struct ParsedArguments {
    pub lower: Value,
    pub upper: Value,
    pub input_min: Option<Value>,
    pub input_max: Option<Value>,
}

pub(super) fn parse(rest: &[Value]) -> BuiltinResult<ParsedArguments> {
    let mut parsed = ParsedArguments {
        lower: Value::Num(0.0),
        upper: Value::Num(1.0),
        input_min: None,
        input_max: None,
    };
    if rest.is_empty() {
        return Ok(parsed);
    }
    if keyword_of(&rest[0]).is_some() {
        parse_options(rest, 0, &mut parsed)?;
        return Ok(parsed);
    }
    if rest.len() < 2 {
        return Err(error::invalid_argument(
            "output interval requires both lower and upper bounds",
        ));
    }
    parsed.lower = rest[0].clone();
    parsed.upper = rest[1].clone();
    parse_options(rest, 2, &mut parsed)?;
    Ok(parsed)
}

fn parse_options(rest: &[Value], start: usize, parsed: &mut ParsedArguments) -> BuiltinResult<()> {
    if !rest.len().saturating_sub(start).is_multiple_of(2) {
        return Err(error::invalid_argument(
            "name-value arguments must appear in pairs",
        ));
    }
    for pair in rest[start..].chunks_exact(2) {
        let name = keyword_of(&pair[0])
            .ok_or_else(|| error::invalid_argument("expected name-value option name"))?;
        match name.as_str() {
            "inputmin" => parsed.input_min = Some(pair[1].clone()),
            "inputmax" => parsed.input_max = Some(pair[1].clone()),
            _ => return Err(error::invalid_argument(format!("unknown option '{name}'"))),
        }
    }
    Ok(())
}
