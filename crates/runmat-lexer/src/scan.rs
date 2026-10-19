use crate::token::Token;

pub(crate) fn last_is_value_token(tok: &Token) -> bool {
    tok.can_be_identifier()
        || matches!(
            tok,
            Token::Integer
                | Token::RadixInteger
                | Token::Float
                | Token::True
                | Token::False
                | Token::RParen
                | Token::RBracket
                | Token::RBrace
                | Token::Str
        )
}

pub(crate) fn find_line_terminator(s: &str) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'\n' => return Some((i, 1)),
            b'\r' => {
                if bytes.get(i + 1) == Some(&b'\n') {
                    return Some((i, 2));
                } else {
                    return Some((i, 1));
                }
            }
            _ => continue,
        }
    }
    None
}

/// A bare trailing decimal point belongs to the number only when it cannot
/// begin an adjacent dot operator, nonconjugate transpose, or ellipsis.
/// Preserve the established `1..3` split as `1.` then `.` then `3`.
pub(crate) fn trailing_decimal_dot(s: &str) -> bool {
    let Some(rest) = s.strip_prefix('.') else {
        return false;
    };
    !matches!(
        rest.as_bytes(),
        [b'.', b'.', ..] | [b'*' | b'/' | b'\\' | b'^' | b'+' | b'-' | b'\'', ..]
    )
}

pub(crate) fn scan_leading_dot_float_suffix(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut saw_digit = false;

    while i < bytes.len() {
        match bytes[i] {
            b'0'..=b'9' => {
                saw_digit = true;
                i += 1;
            }
            b'_' if i > 0 && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() => {
                i += 1;
            }
            _ => break,
        }
    }

    if !saw_digit {
        return None;
    }

    if i < bytes.len() && matches!(bytes[i], b'e' | b'E' | b'd' | b'D') {
        let exp_start = i;
        i += 1;
        if i < bytes.len() && matches!(bytes[i], b'+' | b'-') {
            i += 1;
        }
        let mut exp_digits = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                b'0'..=b'9' => {
                    exp_digits += 1;
                    i += 1;
                }
                b'_' if exp_digits > 0 && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() => {
                    i += 1;
                }
                _ => break,
            }
        }
        if exp_digits == 0 {
            i = exp_start;
        }
    }

    Some(i)
}
