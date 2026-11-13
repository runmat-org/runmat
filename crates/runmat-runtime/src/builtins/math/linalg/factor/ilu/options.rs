use super::{error, BuiltinResult};
use crate::builtins::common::tensor;
use crate::builtins::math::linalg::ops::sparse_factor::{IncompleteLuKind, ModifiedIncompleteLu};
use runmat_builtins::ILU_ERROR_INVALID_ARGUMENT;
use runmat_value::{StructValue, Value};

#[derive(Clone, Copy)]
pub(super) struct Options {
    pub kind: IncompleteLuKind,
    pub droptol: f64,
    pub modified: ModifiedIncompleteLu,
    pub udiag: bool,
    pub thresh: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            kind: IncompleteLuKind::NoFill,
            droptol: 0.0,
            modified: ModifiedIncompleteLu::Off,
            udiag: false,
            thresh: 1.0,
        }
    }
}
pub(super) async fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Options> {
    if args.len() > 1 {
        return Err(error(&ILU_ERROR_INVALID_ARGUMENT, None));
    }
    let mut result = Options::default();
    let Some(raw) = args.first() else {
        return Ok(result);
    };
    let Value::Struct(fields) = raw else {
        return Err(error(
            &ILU_ERROR_INVALID_ARGUMENT,
            Some("ilu: options must be a scalar struct"),
        ));
    };
    parse_struct(fields, &mut result).await?;
    Ok(result)
}

async fn parse_struct(fields: &StructValue, result: &mut Options) -> BuiltinResult<()> {
    for (name, value) in &fields.fields {
        match name.to_ascii_lowercase().as_str() {
            "type" => {
                result.kind = match text(value)?
                    .as_deref()
                    .map(str::to_ascii_lowercase)
                    .as_deref()
                {
                    Some("nofill") => IncompleteLuKind::NoFill,
                    Some("crout") => IncompleteLuKind::Crout,
                    Some("ilutp") => IncompleteLuKind::ThresholdPivoting,
                    _ => {
                        return Err(error(
                            &ILU_ERROR_INVALID_ARGUMENT,
                            Some("ilu: type must be 'nofill', 'crout', or 'ilutp'"),
                        ))
                    }
                }
            }
            "droptol" => {
                result.droptol = scalar(value).await?;
                if !result.droptol.is_finite() || result.droptol < 0.0 {
                    return Err(error(
                        &ILU_ERROR_INVALID_ARGUMENT,
                        Some("ilu: droptol must be a finite nonnegative scalar"),
                    ));
                }
            }
            "milu" => {
                result.modified = match text(value)?
                    .as_deref()
                    .map(str::to_ascii_lowercase)
                    .as_deref()
                {
                    Some("off") => ModifiedIncompleteLu::Off,
                    Some("row") => ModifiedIncompleteLu::Row,
                    Some("col") => ModifiedIncompleteLu::Column,
                    _ => {
                        return Err(error(
                            &ILU_ERROR_INVALID_ARGUMENT,
                            Some("ilu: milu must be 'off', 'row', or 'col'"),
                        ))
                    }
                }
            }
            "udiag" => {
                let v = scalar(value).await?;
                if v != 0.0 && v != 1.0 {
                    return Err(error(
                        &ILU_ERROR_INVALID_ARGUMENT,
                        Some("ilu: udiag must be 0 or 1"),
                    ));
                }
                result.udiag = v == 1.0;
            }
            "thresh" => {
                result.thresh = scalar(value).await?;
                if !result.thresh.is_finite() || !(0.0..=1.0).contains(&result.thresh) {
                    return Err(error(
                        &ILU_ERROR_INVALID_ARGUMENT,
                        Some("ilu: thresh must be in [0,1]"),
                    ));
                }
            }
            _ => {
                return Err(error(
                    &ILU_ERROR_INVALID_ARGUMENT,
                    Some("ilu: unknown option field"),
                ))
            }
        }
    }
    Ok(())
}

async fn scalar(value: &Value) -> BuiltinResult<f64> {
    tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|m| error(&ILU_ERROR_INVALID_ARGUMENT, Some(&m)))?
        .ok_or_else(|| {
            error(
                &ILU_ERROR_INVALID_ARGUMENT,
                Some("ilu: option must be a real scalar"),
            )
        })
}
fn text(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(v)) => Some(v.as_str().into()),
        Value::StringArray(v) if v.data().len() == 1 && !v.data()[0].is_missing() => Some(
            v.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .into(),
        ),
        Value::CharArray(v) if v.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(v.data()).map_err(|source| {
                let mut error = error(&ILU_ERROR_INVALID_ARGUMENT, Some(&source.to_string()));
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}
