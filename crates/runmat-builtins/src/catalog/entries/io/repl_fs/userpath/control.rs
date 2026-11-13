//! One public-control boundary for userpath runtime admission and inference.

use std::path::Path;

#[cfg(windows)]
const PATH_LIST_SEPARATOR: char = ';';
#[cfg(not(windows))]
const PATH_LIST_SEPARATOR: char = ':';

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UserPathForm {
    Absolute,
    Partial,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UserPathControl<'a> {
    Reset,
    Clear,
    Custom { text: &'a str, form: UserPathForm },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UserPathControlError {
    Empty,
    NotOnePath,
}

impl<'a> UserPathControl<'a> {
    pub fn parse(text: &'a str) -> Result<Self, UserPathControlError> {
        if text.eq_ignore_ascii_case("reset") {
            return Ok(Self::Reset);
        }
        if text.eq_ignore_ascii_case("clear") {
            return Ok(Self::Clear);
        }
        if text.is_empty() {
            return Err(UserPathControlError::Empty);
        }
        if text.contains(PATH_LIST_SEPARATOR) || text.contains('\0') {
            return Err(UserPathControlError::NotOnePath);
        }
        let form = if Path::new(text).is_absolute() {
            UserPathForm::Absolute
        } else {
            UserPathForm::Partial
        };
        Ok(Self::Custom { text, form })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controls_and_path_forms_are_typed_once() {
        assert_eq!(UserPathControl::parse("ReSeT"), Ok(UserPathControl::Reset));
        assert_eq!(UserPathControl::parse("CLEAR"), Ok(UserPathControl::Clear));
        #[cfg(windows)]
        let absolute = "C:\\installed\\toolbox";
        #[cfg(not(windows))]
        let absolute = "/installed/toolbox";
        assert_eq!(
            UserPathControl::parse(absolute),
            Ok(UserPathControl::Custom {
                text: absolute,
                form: UserPathForm::Absolute,
            })
        );
        assert_eq!(
            UserPathControl::parse("relative/toolbox"),
            Ok(UserPathControl::Custom {
                text: "relative/toolbox",
                form: UserPathForm::Partial,
            })
        );
        assert_eq!(UserPathControl::parse(""), Err(UserPathControlError::Empty));
        assert_eq!(
            UserPathControl::parse("first\0second"),
            Err(UserPathControlError::NotOnePath)
        );
    }
}
