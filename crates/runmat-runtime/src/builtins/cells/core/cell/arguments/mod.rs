mod size;

use runmat_builtins::{CELL_GPU_SIZE_EXTENSION, CELL_LIKE_EXTENSION};
use runmat_value::Value;

use super::error;
use crate::builtins::common::random_args::keyword_of;

pub(super) struct ParsedCell {
    pub(super) shape: Vec<usize>,
    pub(super) prototype: Option<Value>,
}

pub(super) async fn parse(args: Vec<Value>) -> crate::BuiltinResult<ParsedCell> {
    let mut dimensions = Vec::new();
    let mut prototype = None;
    let mut args = args.into_iter();
    while let Some(value) = args.next() {
        if is_like(&value) {
            crate::compatibility::ensure_builtin_extension_enabled(&CELL_LIKE_EXTENSION, "cell")?;
            if prototype.is_some() {
                return Err(error::invalid_input(
                    "multiple like specifications are not supported",
                ));
            }
            prototype = Some(
                args.next()
                    .ok_or_else(|| error::invalid_input("expected prototype after like"))?,
            );
            if args.len() != 0 {
                return Err(error::invalid_input(
                    "like and its prototype must be the final arguments",
                ));
            }
            continue;
        }
        if keyword_of(&value).is_some() {
            return Err(error::invalid_input("unrecognized option"));
        }
        if matches!(value, Value::GpuTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CELL_GPU_SIZE_EXTENSION,
                "cell",
            )?;
        }
        dimensions.push(value);
    }
    let shape = size::parse(&dimensions, prototype.as_ref()).await?;
    Ok(ParsedCell { shape, prototype })
}

fn is_like(value: &Value) -> bool {
    keyword_of(value).is_some_and(|text| text.eq_ignore_ascii_case("like"))
}
