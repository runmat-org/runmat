//! Nyquist presentation; absent from numeric output forms.

use super::*;
use crate::builtins::control::plot_setup_error::is_nonfatal_plot_setup_error;
use solver::NyquistResponse;

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
        if is_nonfatal_plot_setup_error(&error) {
            return Ok(());
        }
        return Err(error);
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Nyquist Diagram")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Real Axis")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Imaginary Axis")]).await;
    let _ = crate::call_builtin_async("grid", &[Value::from("on")]).await;
    Ok(())
}
