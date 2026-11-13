use super::*;
use runmat_builtins::{PcaOptionName, PcaRows};
pub(super) struct Options {
    pub centered: bool,
    pub economy: bool,
    pub components: Option<usize>,
    pub rows: PcaRows,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            centered: true,
            economy: true,
            components: None,
            rows: PcaRows::Complete,
        }
    }
}
impl Options {
    pub(super) async fn parse(
        args: crate::call::FunctionArgumentView<'_>,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if !(args.len() - 1).is_multiple_of(2) {
            return Err(invalid(NAME, "options require complete name/value pairs"));
        }
        let mut result = Self::default();
        for index in (1..args.len()).step_by(2) {
            let name=text(&args[index]).and_then(|value|PcaOptionName::from_public(&value))
                .ok_or_else(||invalid(NAME,"supported options are Algorithm, Centered, Economy, NumComponents, and Rows"))?;
            let value = &args[index + 1];
            match name {
                PcaOptionName::Algorithm=> {
                    if !text(value).is_some_and(|value|value.eq_ignore_ascii_case("svd")) {
                        return Err(invalid(NAME,"only Algorithm='svd' is supported; eig and als are not aliases for SVD"));
                    }
                }
                PcaOptionName::Rows=>result.rows=text(value).and_then(|value|PcaRows::from_public(&value))
                    .ok_or_else(||invalid(NAME,"Rows must be complete or all; pairwise and ALS missing-data estimation are unsupported"))?,
                PcaOptionName::Centered=>result.centered=switch(&crate::dispatcher::gather_if_needed_in_context_async(value,runtime).await?)?,
                PcaOptionName::Economy=>result.economy=switch(&crate::dispatcher::gather_if_needed_in_context_async(value,runtime).await?)?,
                PcaOptionName::NumComponents=> {
                    let value=inferential::gather(NAME,value,runtime).await?;
                    if value.len()!=1 {return Err(invalid(NAME,"NumComponents must be a positive single/double scalar integer"));}
                    let value=crate::builtins::common::tensor::tensor_value_f64(&value,0);
                    if !value.is_finite() || value<1.0 || value.fract()!=0.0 || value>=2f64.powi(usize::BITS as i32) {
                        return Err(invalid(NAME,"NumComponents must be a positive representable scalar integer"));
                    }
                    result.components=Some(value as usize);
                }
            }
        }
        Ok(result)
    }
}
fn text(value: &Value) -> Option<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
}
fn switch(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(value) if value.data.len() == 1 => Ok(value.data[0] != 0),
        value => match text(value).as_deref() {
            Some(value) if value.eq_ignore_ascii_case("on") => Ok(true),
            Some(value) if value.eq_ignore_ascii_case("off") => Ok(false),
            _ => Err(invalid(
                NAME,
                "Centered and Economy require logical scalars or on/off text",
            )),
        },
    }
}
