//! SPICE numbers: `4.7k`, `10meg`, `2.2nF`, `1e-3`, `5mil`. Letters after the scale suffix
//! are units and are ignored, as SPICE does.

/// Parses a SPICE number, or `None` when the text does not start with one.
pub fn parse_number(s: &str) -> Option<f64> {
    let s = s.trim();
    let b = s.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let digits_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i == digits_start || (i == digits_start + 1 && b[digits_start] == b'.') {
        return None;
    }
    // Exponent, only when digits follow.
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let k = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > k {
            i = j;
        }
    }
    let mantissa: f64 = s[..i].parse().ok()?;
    let rest = s[i..].to_ascii_lowercase();
    let scale = if rest.starts_with("meg") {
        1e6
    } else if rest.starts_with("mil") {
        25.4e-6
    } else {
        match rest.chars().next() {
            Some('t') => 1e12,
            Some('g') => 1e9,
            Some('k') => 1e3,
            Some('m') => 1e-3,
            Some('u') | Some('µ') => 1e-6,
            Some('n') => 1e-9,
            Some('p') => 1e-12,
            Some('f') => 1e-15,
            Some('a') => 1e-18,
            _ => 1.0,
        }
    };
    if !rest.is_empty() && !rest.chars().all(|c| c.is_alphabetic() || c == 'µ') {
        return None;
    }
    Some(mantissa * scale)
}

#[cfg(test)]
mod tests {
    use super::parse_number as p;
    #[test]
    fn suffixes() {
        assert_eq!(p("4.7k"), Some(4700.0));
        assert_eq!(p("10meg"), Some(1e7));
        assert_eq!(p("10MEGohm"), Some(1e7));
        assert!((p("2.2nF").unwrap() - 2.2e-9).abs() < 1e-24);
        assert_eq!(p("1e-3"), Some(1e-3));
        assert_eq!(p("1e"), Some(1.0));
        assert_eq!(p("-5"), Some(-5.0));
        assert_eq!(p(".5u"), Some(0.5e-6));
        assert_eq!(p("3V"), Some(3.0));
        assert_eq!(p("abc"), None);
        assert_eq!(p("1k2"), None);
    }
}
