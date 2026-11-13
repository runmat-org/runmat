use super::*;
use runmat_builtins::{KsdensityFunction, KsdensityOptionName};
pub(super) struct Options {
    pub bandwidth: Option<f64>,
    pub points: usize,
    pub function: KsdensityFunction,
}
impl Options {
    pub(super) async fn parse(
        args: crate::call::FunctionArgumentView<'_>,
        start: usize,
        explicit_points: bool,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if !(args.len() - start).is_multiple_of(2) {
            return Err(invalid(NAME, "options require complete name/value pairs"));
        }
        let mut bandwidth = None;
        let mut point_count = None;
        let mut function = KsdensityFunction::Pdf;
        for index in (start..args.len()).step_by(2) {
            let name = text(&args[index])
                .and_then(|value| KsdensityOptionName::from_public(&value))
                .ok_or_else(|| {
                    invalid(
                        NAME,
                        "supported options are Bandwidth, NumPoints, Function, Kernel and Support",
                    )
                })?;
            let value = &args[index + 1];
            match name {
                KsdensityOptionName::Bandwidth => {
                    if let Some(name) = text(value) {
                        if !name.eq_ignore_ascii_case("normal-approx") {
                            return Err(invalid(NAME,"Bandwidth must be positive or normal-approx; plug-in estimation is unsupported"));
                        }
                        bandwidth = None;
                    } else {
                        inferential::real_sample(NAME, value)?;
                        bandwidth = Some(value);
                    }
                }
                KsdensityOptionName::NumPoints => {
                    inferential::real_sample(NAME, value)?;
                    point_count = Some(value);
                }
                KsdensityOptionName::Function => {
                    function = text(value)
                        .and_then(|value| KsdensityFunction::from_public(&value))
                        .ok_or_else(|| invalid(NAME, "Function must be pdf, cdf or survivor"))?
                }
                KsdensityOptionName::Kernel => {
                    if !text(value).is_some_and(|value| value.eq_ignore_ascii_case("normal")) {
                        return Err(invalid(NAME, "only Kernel='normal' is supported"));
                    }
                }
                KsdensityOptionName::Support => {
                    if !text(value).is_some_and(|value| value.eq_ignore_ascii_case("unbounded")) {
                        return Err(invalid(NAME,"only Support='unbounded' is supported; no bounded-support correction is applied"));
                    }
                }
            }
        }
        if explicit_points && point_count.is_some() {
            return Err(invalid(
                NAME,
                "NumPoints cannot be combined with explicit evaluation points",
            ));
        }
        let bandwidth = match bandwidth {
            Some(value) => {
                let value = scalar(value, runtime).await?;
                if !value.is_finite() || value <= 0.0 {
                    return Err(invalid(
                        NAME,
                        "Bandwidth must be finite and strictly positive",
                    ));
                }
                Some(value)
            }
            None => None,
        };
        let points = match point_count {
            Some(value) => {
                let value = scalar(value, runtime).await?;
                if !value.is_finite()
                    || !(1.0..2f64.powi(usize::BITS as i32)).contains(&value)
                    || value.fract() != 0.0
                {
                    return Err(invalid(
                        NAME,
                        "NumPoints must be a positive representable scalar integer",
                    ));
                }
                value as usize
            }
            None => 100,
        };
        Ok(Self {
            bandwidth,
            points,
            function,
        })
    }
}
fn text(value: &Value) -> Option<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
}
async fn scalar(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<f64> {
    let value = inferential::gather(NAME, value, runtime).await?;
    if value.len() != 1 {
        return Err(invalid(NAME, "numeric option controls must be scalars"));
    }
    Ok(crate::builtins::common::tensor::tensor_value_f64(&value, 0))
}
