use crate::{AotError, AotResult};

use super::LinkerFamily;

pub(super) fn encode(tokens: &[String], family: LinkerFamily) -> AotResult<String> {
    let mut response = String::new();
    for token in tokens {
        if token.bytes().any(|byte| matches!(byte, 0 | b'\r' | b'\n')) {
            return Err(AotError::contract(
                "aot.link.response",
                "response-file token contains a forbidden control character",
            ));
        }
        match family {
            LinkerFamily::UnixCc => encode_unix_token(&mut response, token),
            LinkerFamily::Msvc => encode_msvc_token(&mut response, token),
        }
        response.push('\n');
    }
    Ok(response)
}

fn encode_unix_token(output: &mut String, token: &str) {
    output.push('"');
    for character in token.chars() {
        if matches!(character, '\\' | '"') {
            output.push('\\');
        }
        output.push(character);
    }
    output.push('"');
}

fn encode_msvc_token(output: &mut String, token: &str) {
    output.push('"');
    let mut backslashes = 0;
    for character in token.chars() {
        if character == '\\' {
            backslashes += 1;
            continue;
        }
        if character == '"' {
            output.extend(std::iter::repeat_n('\\', (backslashes * 2) + 1));
        } else {
            output.extend(std::iter::repeat_n('\\', backslashes));
        }
        backslashes = 0;
        output.push(character);
    }
    output.extend(std::iter::repeat_n('\\', backslashes * 2));
    output.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_tokens_are_one_quoted_argument_per_line() {
        assert_eq!(
            encode(
                &["plain".into(), "with space".into(), "a\\b".into()],
                LinkerFamily::UnixCc,
            )
            .unwrap(),
            "\"plain\"\n\"with space\"\n\"a\\\\b\"\n"
        );
        assert!(encode(&["bad\nargument".into()], LinkerFamily::UnixCc).is_err());
    }

    #[test]
    fn msvc_response_preserves_path_separators_and_escapes_quotes() {
        assert_eq!(
            encode(
                &[
                    r"C:\Program Files\RunMat\program.obj".into(),
                    r#"define="quoted""#.into(),
                    r"C:\trailing\".into(),
                ],
                LinkerFamily::Msvc,
            )
            .unwrap(),
            "\"C:\\Program Files\\RunMat\\program.obj\"\n\"define=\\\"quoted\\\"\"\n\"C:\\trailing\\\\\"\n"
        );
    }
}
