use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MethodKeyword {
    Linear,
    Nearest,
    Next,
    Previous,
    Pchip,
    Spline,
    Cubic,
    Makima,
    None,
}

impl MethodKeyword {
    pub(super) fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "linear" => Some(Self::Linear),
            "nearest" => Some(Self::Nearest),
            "next" => Some(Self::Next),
            "previous" => Some(Self::Previous),
            "pchip" => Some(Self::Pchip),
            "spline" => Some(Self::Spline),
            "cubic" => Some(Self::Cubic),
            "makima" => Some(Self::Makima),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

pub(super) fn parse_trailing_methods(
    args: &mut crate::call::BuiltinArguments,
) -> BuiltinResult<(InterpMethod, ExtrapolationMethod)> {
    let mut methods = Vec::new();
    while methods.len() < 2 {
        let Some(name) = args.last().map(text_from_value).transpose()?.flatten() else {
            break;
        };
        let Some(keyword) = MethodKeyword::parse(&name) else {
            break;
        };
        methods.push(keyword);
        args.pop();
    }
    methods.reverse();
    let method = if let Some(keyword) = methods.first() {
        parse_interp_method(*keyword)?
    } else {
        InterpMethod::Linear
    };
    let extrapolation = if let Some(keyword) = methods.get(1) {
        parse_extrap_method(*keyword)?
    } else {
        match method {
            InterpMethod::Linear => ExtrapolationMethod::Linear,
            InterpMethod::Nearest => ExtrapolationMethod::Nearest,
            InterpMethod::Next => ExtrapolationMethod::Next,
            InterpMethod::Previous => ExtrapolationMethod::Previous,
            InterpMethod::Pchip => ExtrapolationMethod::Pchip,
            InterpMethod::Spline => ExtrapolationMethod::Spline,
            InterpMethod::Makima => ExtrapolationMethod::Makima,
        }
    };
    Ok((method, extrapolation))
}

pub(super) fn parse_interp_method(keyword: MethodKeyword) -> BuiltinResult<InterpMethod> {
    match keyword {
        MethodKeyword::Linear => Ok(InterpMethod::Linear),
        MethodKeyword::Nearest => Ok(InterpMethod::Nearest),
        MethodKeyword::Next => Ok(InterpMethod::Next),
        MethodKeyword::Previous => Ok(InterpMethod::Previous),
        MethodKeyword::Pchip => Ok(InterpMethod::Pchip),
        MethodKeyword::Cubic => Err(unsupported(
            "method 'cubic' is not yet implemented for griddedInterpolant",
        )),
        MethodKeyword::Spline => Ok(InterpMethod::Spline),
        MethodKeyword::Makima => Ok(InterpMethod::Makima),
        MethodKeyword::None => Err(invalid("'none' is only valid as an extrapolation method")),
    }
}

pub(super) fn parse_extrap_method(keyword: MethodKeyword) -> BuiltinResult<ExtrapolationMethod> {
    match keyword {
        MethodKeyword::Linear => Ok(ExtrapolationMethod::Linear),
        MethodKeyword::Nearest => Ok(ExtrapolationMethod::Nearest),
        MethodKeyword::Next => Ok(ExtrapolationMethod::Next),
        MethodKeyword::Previous => Ok(ExtrapolationMethod::Previous),
        MethodKeyword::Pchip => Ok(ExtrapolationMethod::Pchip),
        MethodKeyword::Cubic => Err(unsupported(
            "extrapolation method 'cubic' is not yet implemented for griddedInterpolant",
        )),
        MethodKeyword::Spline => Ok(ExtrapolationMethod::Spline),
        MethodKeyword::None => Ok(ExtrapolationMethod::None),
        MethodKeyword::Makima => Ok(ExtrapolationMethod::Makima),
    }
}

pub(super) fn text_from_value(value: &Value) -> crate::BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::CharArray(chars) if chars.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = invalid(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}
