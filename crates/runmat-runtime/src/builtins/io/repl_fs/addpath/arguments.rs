#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Position {
    Begin,
    End,
}

pub(super) struct Request {
    pub(super) directories: Vec<String>,
    pub(super) position: Position,
}

enum OptionToken {
    Begin,
    End,
    Frozen,
}

pub(super) fn parse(tokens: Vec<String>) -> crate::BuiltinResult<Request> {
    let mut position = Position::Begin;
    let mut position_set = false;
    let mut directories = Vec::new();

    for token in tokens {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        match option(token) {
            Some(OptionToken::Begin) => {
                set_position(&mut position, &mut position_set, Position::Begin)?
            }
            Some(OptionToken::End) => {
                set_position(&mut position, &mut position_set, Position::End)?
            }
            Some(OptionToken::Frozen) => {}
            None => directories.extend(super::super::path_mutation::segments::split(token)),
        }
    }

    if directories.is_empty() {
        return Err(super::errors::descriptor(
            &runmat_builtins::ADDPATH_ERROR_TOO_FEW_ARGUMENTS,
        ));
    }
    Ok(Request {
        directories,
        position,
    })
}

fn option(token: &str) -> Option<OptionToken> {
    if token.eq_ignore_ascii_case("-begin") {
        Some(OptionToken::Begin)
    } else if token.eq_ignore_ascii_case("-end") {
        Some(OptionToken::End)
    } else if token.eq_ignore_ascii_case("-frozen") {
        Some(OptionToken::Frozen)
    } else {
        None
    }
}

fn set_position(
    position: &mut Position,
    position_set: &mut bool,
    requested: Position,
) -> crate::BuiltinResult<()> {
    if *position_set {
        return Err(super::errors::descriptor(
            &runmat_builtins::ADDPATH_ERROR_POSITION,
        ));
    }
    *position = requested;
    *position_set = true;
    Ok(())
}
