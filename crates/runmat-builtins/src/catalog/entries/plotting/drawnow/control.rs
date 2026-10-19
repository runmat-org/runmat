//! One parser for command-style and function-call drawnow controls.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawNowControl {
    Full,
    LimitRate,
    NoCallbacks,
    LimitRateNoCallbacks,
    Update,
    Expose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawNowControlError {
    TooManyControls,
    InvalidControl,
    ConflictingControls,
}

impl DrawNowControl {
    pub fn decode(tokens: &[&str]) -> Result<Self, DrawNowControlError> {
        match tokens {
            [] => Ok(Self::Full),
            [one] if token_is(one, "limitrate") => Ok(Self::LimitRate),
            [one] if token_is(one, "nocallbacks") => Ok(Self::NoCallbacks),
            [one] if token_is(one, "update") => Ok(Self::Update),
            [one] if token_is(one, "expose") => Ok(Self::Expose),
            [first, second] if token_is(first, "limitrate") && token_is(second, "nocallbacks") => {
                Ok(Self::LimitRateNoCallbacks)
            }
            [_, _] => Err(Self::two_control_error(tokens)),
            [_] => Err(DrawNowControlError::InvalidControl),
            _ => Err(DrawNowControlError::TooManyControls),
        }
    }

    fn two_control_error(tokens: &[&str]) -> DrawNowControlError {
        let recognized = tokens.iter().all(|token| {
            ["limitrate", "nocallbacks", "update", "expose"]
                .iter()
                .any(|candidate| token_is(token, candidate))
        });
        if recognized {
            DrawNowControlError::ConflictingControls
        } else {
            DrawNowControlError::InvalidControl
        }
    }
}

fn token_is(actual: &str, expected: &str) -> bool {
    actual.trim().eq_ignore_ascii_case(expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_documented_modes_decode_with_distinct_legacy_update_semantics() {
        assert_eq!(DrawNowControl::decode(&[]), Ok(DrawNowControl::Full));
        assert_eq!(
            DrawNowControl::decode(&["LIMITRATE"]),
            Ok(DrawNowControl::LimitRate)
        );
        assert_eq!(
            DrawNowControl::decode(&["nocallbacks"]),
            Ok(DrawNowControl::NoCallbacks)
        );
        assert_eq!(
            DrawNowControl::decode(&[" limitrate ", "nocallbacks"]),
            Ok(DrawNowControl::LimitRateNoCallbacks)
        );
        assert_eq!(
            DrawNowControl::decode(&["update"]),
            Ok(DrawNowControl::Update)
        );
        assert_eq!(
            DrawNowControl::decode(&["expose"]),
            Ok(DrawNowControl::Expose)
        );
    }

    #[test]
    fn invalid_and_conflicting_controls_are_not_silently_ignored() {
        assert_eq!(
            DrawNowControl::decode(&["limitrate", "limitrate"]),
            Err(DrawNowControlError::ConflictingControls)
        );
        assert_eq!(
            DrawNowControl::decode(&["update", "nocallbacks"]),
            Err(DrawNowControlError::ConflictingControls)
        );
        assert_eq!(
            DrawNowControl::decode(&["limitrate", "other"]),
            Err(DrawNowControlError::InvalidControl)
        );
        assert_eq!(
            DrawNowControl::decode(&["bogus"]),
            Err(DrawNowControlError::InvalidControl)
        );
        assert_eq!(
            DrawNowControl::decode(&["a", "b", "c"]),
            Err(DrawNowControlError::TooManyControls)
        );
    }
}
