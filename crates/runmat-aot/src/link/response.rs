use crate::{AotError, AotResult};

pub(super) fn encode(tokens: &[String]) -> AotResult<String> {
    let mut response = String::new();
    for token in tokens {
        if token.bytes().any(|byte| matches!(byte, 0 | b'\r' | b'\n')) {
            return Err(AotError::contract(
                "aot.link.response",
                "response-file token contains a forbidden control character",
            ));
        }
        response.push('"');
        for character in token.chars() {
            if matches!(character, '\\' | '"') {
                response.push('\\');
            }
            response.push(character);
        }
        response.push_str("\"\n");
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_tokens_are_one_quoted_argument_per_line() {
        assert_eq!(
            encode(&["plain".into(), "with space".into(), "a\\b".into()]).unwrap(),
            "\"plain\"\n\"with space\"\n\"a\\\\b\"\n"
        );
        assert!(encode(&["bad\nargument".into()]).is_err());
    }
}
