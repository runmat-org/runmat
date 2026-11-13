#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ParsedNumber {
    pub(super) re: f64,
    pub(super) im: f64,
    pub(super) complex: bool,
}

impl ParsedNumber {
    pub(super) const ZERO: Self = Self {
        re: 0.0,
        im: 0.0,
        complex: false,
    };
    pub(super) fn real(re: f64) -> Self {
        Self {
            re,
            im: 0.0,
            complex: false,
        }
    }
}

pub(super) fn parse_csv_number(text: &str) -> Option<ParsedNumber> {
    if let Ok(re) = text.parse::<f64>() {
        return Some(ParsedNumber::real(re));
    }
    let body = text.strip_suffix(['i', 'j'])?;
    let split = body.char_indices().rev().find_map(|(idx, ch)| {
        if idx == 0 {
            return None;
        }
        if ch != '+' && ch != '-' {
            return None;
        }
        let prior = body[..idx].chars().next_back();
        (!matches!(prior, Some('e' | 'E'))).then_some(idx)
    });
    let (re, imag) = if let Some(idx) = split {
        (
            body[..idx].parse::<f64>().ok()?,
            parse_imaginary(&body[idx..])?,
        )
    } else {
        (0.0, parse_imaginary(body)?)
    };
    Some(ParsedNumber {
        re,
        im: imag,
        complex: true,
    })
}

fn parse_imaginary(text: &str) -> Option<f64> {
    match text {
        "" | "+" => Some(1.0),
        "-" => Some(-1.0),
        _ => text.parse::<f64>().ok(),
    }
}
