use runmat_value::Value;

use crate::BuiltinResult;

use super::{fopen_error_with_detail, scalar_string, FOPEN_ERROR_INVALID_PERMISSION};

#[derive(Clone)]
pub(super) struct Permission {
    pub(super) canonical: String,
    pub(super) read: bool,
    pub(super) write: bool,
    pub(super) append: bool,
    pub(super) create: bool,
    pub(super) truncate: bool,
    pub(super) binary: bool,
}

impl Permission {
    pub(super) fn access(&self) -> super::super::registry::FileAccess {
        super::super::registry::FileAccess {
            read: self.read,
            write: self.write,
        }
    }
    pub(super) fn parse(value: Option<&Value>) -> BuiltinResult<Self> {
        let raw = match value {
            Some(v) => {
                let text = scalar_string(
                    v,
                    "expected permission as a string scalar or character vector",
                    &FOPEN_ERROR_INVALID_PERMISSION,
                )?;
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    return Err(fopen_error_with_detail(
                        &FOPEN_ERROR_INVALID_PERMISSION,
                        "permission string must not be empty",
                    ));
                }
                trimmed.to_string()
            }
            None => "r".to_string(),
        };

        let mut chars = raw.chars();
        let base = chars
            .next()
            .ok_or_else(|| {
                fopen_error_with_detail(
                    &FOPEN_ERROR_INVALID_PERMISSION,
                    "permission string must not be empty",
                )
            })?
            .to_ascii_lowercase();

        let mut read = false;
        let mut write = false;
        let mut append = false;
        let mut create = false;
        let mut truncate = false;

        match base {
            'r' => {
                read = true;
            }
            'w' => {
                write = true;
                create = true;
                truncate = true;
            }
            'a' => {
                write = true;
                create = true;
                append = true;
            }
            _ => {
                return Err(fopen_error_with_detail(
                    &FOPEN_ERROR_INVALID_PERMISSION,
                    format!("unsupported permission prefix '{base}'"),
                ));
            }
        }

        let mut plus = false;
        let mut binary = false;
        let mut explicit_text = false;

        for c in chars {
            match c {
                '+' => {
                    if plus {
                        return Err(fopen_error_with_detail(
                            &FOPEN_ERROR_INVALID_PERMISSION,
                            "duplicate '+' modifier in permission string",
                        ));
                    }
                    plus = true;
                    read = true;
                    write = true;
                }
                'b' | 'B' => {
                    if binary {
                        return Err(fopen_error_with_detail(
                            &FOPEN_ERROR_INVALID_PERMISSION,
                            "duplicate 'b' modifier in permission string",
                        ));
                    }
                    binary = true;
                }
                't' | 'T' => {
                    if explicit_text {
                        return Err(fopen_error_with_detail(
                            &FOPEN_ERROR_INVALID_PERMISSION,
                            "duplicate 't' modifier in permission string",
                        ));
                    }
                    explicit_text = true;
                }
                other => {
                    return Err(fopen_error_with_detail(
                        &FOPEN_ERROR_INVALID_PERMISSION,
                        format!("unrecognised permission modifier '{other}'"),
                    ));
                }
            }
        }

        if binary && explicit_text {
            return Err(fopen_error_with_detail(
                &FOPEN_ERROR_INVALID_PERMISSION,
                "permission modifiers 'b' and 't' are mutually exclusive",
            ));
        }

        let mut canonical = String::new();
        canonical.push(base);
        if binary {
            canonical.push('b');
        } else if explicit_text {
            canonical.push('t');
        }
        if plus {
            canonical.push('+');
        }

        Ok(Self {
            canonical,
            read,
            write,
            append,
            create,
            truncate,
            binary,
        })
    }
}
