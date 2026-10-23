//! Output-only legacy `datestr` grammar; datetime-object display tokens are distinct.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Token {
    Year4,
    Year2,
    MonthFull,
    MonthShort,
    MonthNumber,
    MonthInitial,
    WeekdayFull,
    WeekdayShort,
    DayNumber,
    WeekdayInitial,
    Hour,
    Minute,
    Second,
    Millisecond,
    Meridiem,
    Quarter,
    Literal(char),
}

pub(crate) enum LegacyDatestrFormat {
    Default,
    Explicit {
        tokens: Vec<Token>,
        twelve_hour: bool,
    },
}

pub(crate) fn parse_datestr_format(value: Option<&Value>) -> BuiltinResult<LegacyDatestrFormat> {
    let Some(value) = value else {
        return Ok(LegacyDatestrFormat::Default);
    };
    let code = match value {
        Value::Num(code) => Some(*code),
        Value::Tensor(tensor)
            if tensor.len() == 1 && tensor.numeric_dtype() == runmat_value::NumericDType::F64 =>
        {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    };
    if let Some(code) = code {
        if !code.is_finite() || code.fract() != 0.0 || !(-1.0..=31.0).contains(&code) {
            return Err(datetime_error(
                "datestr: format code must be an integer from -1 through 31",
            ));
        }
        if code == -1.0 {
            return Ok(LegacyDatestrFormat::Default);
        }
        return parse_pattern(numeric_pattern(code as usize));
    }
    parse_pattern(&scalar_text(value, "datestr format")?)
}

fn numeric_pattern(code: usize) -> &'static str {
    const PATTERNS: [&str; 32] = [
        "dd-mmm-yyyy HH:MM:SS",
        "dd-mmm-yyyy",
        "mm/dd/yy",
        "mmm",
        "m",
        "mm",
        "mm/dd",
        "dd",
        "ddd",
        "d",
        "yyyy",
        "yy",
        "mmmyy",
        "HH:MM:SS",
        "HH:MM:SS PM",
        "HH:MM",
        "HH:MM PM",
        "QQ-YY",
        "QQ",
        "dd/mm",
        "dd/mm/yy",
        "mmm.dd,yyyy HH:MM:SS",
        "mmm.dd,yyyy",
        "mm/dd/yyyy",
        "dd/mm/yyyy",
        "yy/mm/dd",
        "yyyy/mm/dd",
        "QQ-yyyy",
        "mmmyyyy",
        "yyyy-mm-dd",
        "yyyymmddTHHMMSS",
        "yyyy-mm-dd HH:MM:SS",
    ];
    PATTERNS[code]
}

fn parse_pattern(pattern: &str) -> BuiltinResult<LegacyDatestrFormat> {
    let mut tokens = Vec::new();
    let mut rest = pattern;
    while !rest.is_empty() {
        let field = [
            ("yyyy", Token::Year4),
            ("YYYY", Token::Year4),
            ("mmmm", Token::MonthFull),
            ("dddd", Token::WeekdayFull),
            ("mmm", Token::MonthShort),
            ("ddd", Token::WeekdayShort),
            ("FFF", Token::Millisecond),
            ("HH", Token::Hour),
            ("MM", Token::Minute),
            ("SS", Token::Second),
            ("AM", Token::Meridiem),
            ("PM", Token::Meridiem),
            ("QQ", Token::Quarter),
            ("YY", Token::Year2),
            ("yy", Token::Year2),
            ("mm", Token::MonthNumber),
            ("dd", Token::DayNumber),
            ("m", Token::MonthInitial),
            ("d", Token::WeekdayInitial),
        ]
        .into_iter()
        .find(|(spelling, _)| rest.starts_with(spelling));
        if let Some((spelling, token)) = field {
            tokens.push(token);
            rest = &rest[spelling.len()..];
        } else {
            let character = rest
                .chars()
                .next()
                .ok_or_else(|| datetime_error("datestr: invalid format"))?;
            tokens.push(Token::Literal(character));
            rest = &rest[character.len_utf8()..];
        }
    }
    let count =
        |predicate: fn(&Token) -> bool| tokens.iter().filter(|token| predicate(token)).count();
    let year = count(|token| matches!(token, Token::Year4 | Token::Year2));
    let month = count(|token| {
        matches!(
            token,
            Token::MonthFull | Token::MonthShort | Token::MonthNumber | Token::MonthInitial
        )
    });
    let named_day = count(|token| {
        matches!(
            token,
            Token::WeekdayFull | Token::WeekdayShort | Token::WeekdayInitial
        )
    });
    let day_number = count(|token| matches!(token, Token::DayNumber));
    let hour = count(|token| matches!(token, Token::Hour));
    let minute = count(|token| matches!(token, Token::Minute));
    let second = count(|token| matches!(token, Token::Second));
    let millis = count(|token| matches!(token, Token::Millisecond));
    let meridiem = count(|token| matches!(token, Token::Meridiem));
    let quarter = count(|token| matches!(token, Token::Quarter));
    if [
        year, month, named_day, day_number, hour, minute, second, millis, meridiem, quarter,
    ]
    .into_iter()
    .any(|count| count > 1)
        || (meridiem != 0 && hour == 0)
        || (quarter != 0
            && (month + named_day + day_number + hour + minute + second + millis + meridiem != 0))
    {
        return Err(datetime_error(
            "datestr: repeated or incompatible legacy format fields",
        ));
    }
    Ok(LegacyDatestrFormat::Explicit {
        tokens,
        twelve_hour: meridiem != 0,
    })
}

impl LegacyDatestrFormat {
    pub(crate) fn resolve_default(self, serials: &[f64]) -> BuiltinResult<Self> {
        if !matches!(self, Self::Default) {
            return Ok(self);
        }
        let mut has_time = false;
        for &serial in serials {
            if serial.is_finite() {
                let time = naive_from_datenum(serial)?.time();
                has_time = time.hour() != 0
                    || time.minute() != 0
                    || time.second() != 0
                    || time.nanosecond() != 0;
                if has_time {
                    break;
                }
            }
        }
        parse_pattern(if has_time {
            "dd-mmm-yyyy HH:MM:SS"
        } else {
            "dd-mmm-yyyy"
        })
    }

    pub(crate) fn render(&self, serial: f64) -> BuiltinResult<String> {
        if serial.is_nan() {
            return Ok("NaT".to_string());
        }
        if serial.is_infinite() {
            return Ok(if serial.is_sign_positive() {
                "Inf"
            } else {
                "-Inf"
            }
            .to_string());
        }
        let naive = naive_from_datenum(serial)?;
        let Self::Explicit {
            tokens,
            twelve_hour,
        } = self
        else {
            return Err(datetime_error("datestr: unresolved default format"));
        };
        let mut out = String::with_capacity(tokens.len().saturating_mul(4));
        for token in tokens {
            match token {
                Token::Year4 => out.push_str(&naive.format("%Y").to_string()),
                Token::Year2 => out.push_str(&naive.format("%y").to_string()),
                Token::MonthFull => out.push_str(&naive.format("%B").to_string()),
                Token::MonthShort => out.push_str(&naive.format("%b").to_string()),
                Token::MonthNumber => out.push_str(&naive.format("%m").to_string()),
                Token::MonthInitial => {
                    out.push(naive.format("%B").to_string().chars().next().unwrap_or(' '))
                }
                Token::WeekdayFull => out.push_str(&naive.format("%A").to_string()),
                Token::WeekdayShort => out.push_str(&naive.format("%a").to_string()),
                Token::DayNumber => out.push_str(&naive.format("%d").to_string()),
                Token::WeekdayInitial => {
                    out.push(naive.format("%A").to_string().chars().next().unwrap_or(' '))
                }
                Token::Hour if *twelve_hour => {
                    out.push_str(&format!("{:2}", (naive.hour() + 11) % 12 + 1))
                }
                Token::Hour => out.push_str(&naive.format("%H").to_string()),
                Token::Minute => out.push_str(&naive.format("%M").to_string()),
                Token::Second => out.push_str(&naive.format("%S").to_string()),
                Token::Millisecond => {
                    out.push_str(&format!("{:03}", naive.nanosecond() / 1_000_000))
                }
                Token::Meridiem => out.push_str(if naive.hour() < 12 { "AM" } else { "PM" }),
                Token::Quarter => out.push_str(&format!("Q{}", (naive.month() - 1) / 3 + 1)),
                Token::Literal(character) => out.push(*character),
            }
        }
        Ok(out)
    }
}
