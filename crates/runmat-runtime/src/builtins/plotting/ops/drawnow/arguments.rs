use runmat_builtins::{DrawNowControl, DRAWNOW_ERROR_INVALID_CONTROL};
use runmat_value::Value;
use std::borrow::Cow;

use crate::BuiltinResult;

use super::errors::drawnow_error;

pub(super) fn parse_controls(
    arguments: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<DrawNowControl> {
    if arguments.len() > 2 {
        return Err(drawnow_error(
            &DRAWNOW_ERROR_INVALID_CONTROL,
            "at most two control tokens are accepted",
        ));
    }
    let mut text: [Cow<'_, str>; 2] = [Cow::Borrowed(""), Cow::Borrowed("")];
    for (index, argument) in arguments.iter().enumerate() {
        let value = match argument {
            Value::String(runmat_value::StringElement::Text(value)) => {
                Cow::Borrowed(value.as_str())
            }
            Value::CharArray(value) if value.rows == 1 => Cow::Owned(
                runmat_value::decode_utf16_code_units(value.data()).map_err(|source| {
                    let mut error =
                        drawnow_error(&DRAWNOW_ERROR_INVALID_CONTROL, source.to_string());
                    error.source = Some(Box::new(source));
                    error
                })?,
            ),
            Value::StringArray(value)
                if value.data().len() == 1 && !value.data()[0].is_missing() =>
            {
                Cow::Borrowed(
                    value.data()[0]
                        .text_ref()
                        .expect("checked nonmissing scalar string")
                        .as_str(),
                )
            }
            _ => {
                return Err(drawnow_error(
                    &DRAWNOW_ERROR_INVALID_CONTROL,
                    "controls must be scalar character or string text",
                ));
            }
        };
        text[index] = value;
    }
    let borrowed = [text[0].as_ref(), text[1].as_ref()];
    DrawNowControl::decode(&borrowed[..arguments.len()]).map_err(|error| {
        drawnow_error(
            &DRAWNOW_ERROR_INVALID_CONTROL,
            format!("unsupported control form: {error:?}"),
        )
    })
}
