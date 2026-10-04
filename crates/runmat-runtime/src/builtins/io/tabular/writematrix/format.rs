use super::*;

pub(super) fn default_delimiter_for_path(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("csv") => ",".to_string(),
        Some("tsv") | Some("tab") => "\t".to_string(),
        Some("txt") | Some("dat") | Some("dlm") => " ".to_string(),
        _ => ",".to_string(),
    }
}

pub(super) fn format_numeric(value: f64, decimal_separator: char) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }

    let abs = value.abs();
    let scientific = abs != 0.0 && !(1e-4..1e15).contains(&abs);
    let raw = if scientific {
        format!("{:.15e}", value)
    } else {
        format!("{:.15}", value)
    };

    let mut formatted = trim_trailing_zeros(raw);
    if formatted == "-0" {
        formatted = "0".to_string();
    }
    if decimal_separator != '.' && formatted.contains('.') {
        formatted = formatted.replace('.', &decimal_separator.to_string());
    }
    formatted
}

fn trim_trailing_zeros(mut value: String) -> String {
    if let Some(exp_pos) = value.find(['e', 'E']) {
        let exponent = value.split_off(exp_pos);
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
        value.push_str(&exponent);
        value
    } else {
        if value.contains('.') {
            while value.ends_with('0') {
                value.pop();
            }
            if value.ends_with('.') {
                value.pop();
            }
        }
        if value.is_empty() {
            "0".to_string()
        } else {
            value
        }
    }
}

pub(super) fn format_string(value: &str, quote: QuoteStrings, delimiter: &str) -> String {
    let quote = match quote {
        QuoteStrings::All => true,
        QuoteStrings::None => false,
        QuoteStrings::Minimal => {
            value.contains(delimiter) || value.contains('"') || value.contains(['\n', '\r'])
        }
    };
    if !quote {
        return value.to_string();
    }
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        if ch == '"' {
            escaped.push('"');
            escaped.push('"');
        } else {
            escaped.push(ch);
        }
    }
    escaped.push('"');
    escaped
}
