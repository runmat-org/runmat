use super::*;
use runmat_builtins::{TtestOptionName, TtestTail};

pub(super) struct Options {
    pub alpha: f64,
    pub dim: Option<usize>,
    pub tail: TtestTail,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            dim: None,
            tail: TtestTail::Both,
        }
    }
}
impl Options {
    pub(super) async fn parse(
        args: crate::call::FunctionArgumentView<'_>,
        start: usize,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if !(args.len() - start).is_multiple_of(2) {
            return Err(invalid(NAME, "options require complete name/value pairs"));
        }
        let mut result = Self::default();
        for index in (start..args.len()).step_by(2) {
            let name = crate::builtins::common::validation::value_text_scalar_string(&args[index])
                .and_then(|name| TtestOptionName::from_public(&name))
                .ok_or_else(|| invalid(NAME, "supported options are Alpha, Dim, and Tail"))?;
            let value = &args[index + 1];
            match name {
                TtestOptionName::Tail => {
                    result.tail =
                        crate::builtins::common::validation::value_text_scalar_string(value)
                            .and_then(|value| TtestTail::from_public(&value))
                            .ok_or_else(|| invalid(NAME, "Tail must be both, right, or left"))?;
                }
                TtestOptionName::Alpha | TtestOptionName::Dim => {
                    let value = inferential::gather(NAME, value, runtime).await?;
                    if value.shape().iter().product::<usize>() != 1 {
                        return Err(invalid(NAME, "numeric options must be scalars"));
                    }
                    let value = crate::builtins::common::tensor::tensor_value_f64(&value, 0);
                    match name {
                        TtestOptionName::Alpha
                            if value.is_finite() && value > 0.0 && value < 1.0 =>
                        {
                            result.alpha = value
                        }
                        TtestOptionName::Dim => {
                            result.dim = Some(
                                runmat_types::decode_shape_dimension(value)
                                    .ok()
                                    .and_then(|dim| dim.checked_sub(1))
                                    .ok_or_else(|| {
                                        invalid(
                                            NAME,
                                            "Dim must be a positive representable integer",
                                        )
                                    })?,
                            );
                        }
                        _ => return Err(invalid(NAME, "Alpha must lie in (0,1)")),
                    }
                }
            }
        }
        Ok(result)
    }
}
