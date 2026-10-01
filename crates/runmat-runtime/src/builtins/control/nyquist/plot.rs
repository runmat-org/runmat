//! Nyquist presentation; absent from numeric output forms.

use super::solver::NyquistResponse;
use super::*;
use crate::builtins::plotting::state::presentation_failure_kind;
use runmat_builtins::NYQUIST_ERROR_PLOT_FAILED;

pub(super) async fn render(response: &NyquistResponse) -> BuiltinResult<()> {
    let mut args = vec![response.re_value()?, response.im_value()?];
    if response.mirror_negative_frequency {
        args.push(response.re_value()?);
        args.push(column_tensor(
            response.im.iter().map(|value| -*value).collect(),
        )?);
    }
    if let Some((&re, &im)) = response.re.first().zip(response.im.first()) {
        args.push(column_tensor(vec![re])?);
        args.push(column_tensor(vec![im])?);
        args.push(Value::from("x"));
        if response.mirror_negative_frequency {
            args.push(column_tensor(vec![re])?);
            args.push(column_tensor(vec![-im])?);
            args.push(Value::from("o"));
        }
    }
    if let Err(error) = crate::call_builtin_async("plot", &args).await {
        if presentation_failure_kind(&error).is_some() {
            return Ok(());
        }
        return Err(plot_error(error));
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Nyquist Diagram")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Real Axis")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Imaginary Axis")]).await;
    let _ = crate::call_builtin_async("grid", &[Value::from("on")]).await;
    Ok(())
}

pub(super) fn plot_error(source: crate::RuntimeError) -> crate::RuntimeError {
    crate::build_runtime_error(format!(
        "{}: {}",
        NYQUIST_ERROR_PLOT_FAILED.message,
        source.message()
    ))
    .with_builtin(BUILTIN_NAME)
    .with_identifier(
        NYQUIST_ERROR_PLOT_FAILED
            .identifier
            .expect("plot failure has a stable identifier"),
    )
    .with_source(source)
    .build()
}
