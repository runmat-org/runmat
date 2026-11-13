//! Nyquist presentation; absent from numeric output forms.

use super::solver::NyquistResponse;
use super::*;
use crate::builtins::plotting::state::presentation_failure_kind;
use runmat_builtins::NYQUIST_ERROR_PLOT_FAILED;

pub(super) async fn render(response: &NyquistResponse) -> BuiltinResult<()> {
    use crate::builtins::plotting::state;
    // MATLAB draws both frequency halves of one system in one color and marks
    // the critical point -1, around which encirclements are counted.
    let color = if state::current_hold_enabled() {
        state::next_line_color_for_axes(state::current_axes_state().active_index)
    } else {
        state::line_color_for_series_index(0)
    };
    let color = row_tensor(vec![color.x as f64, color.y as f64, color.z as f64])?;
    let colored = |x: Value, y: Value, marker: Option<&str>| {
        let mut call = vec![x, y];
        call.extend(marker.map(Value::from));
        call.extend([Value::from("Color"), color.clone()]);
        call
    };

    // The positive-frequency half is the system's single legend entry.
    if let Err(error) = crate::call_builtin_async(
        "plot",
        &colored(response.re_value()?, response.im_value()?, None),
    )
    .await
    {
        if presentation_failure_kind(&error).is_some() {
            return Ok(());
        }
        return Err(plot_error(error));
    }

    // The mirrored half, start markers and critical point stay out of the
    // legend, so legend('K = 1', 'K = 4') labels one entry per system.
    let mut hidden = Vec::new();
    if response.mirror_negative_frequency {
        hidden.push(colored(
            response.re_value()?,
            column_tensor(response.im.iter().map(|value| -*value).collect())?,
            None,
        ));
    }
    if let Some((&re, &im)) = response.re.first().zip(response.im.first()) {
        hidden.push(colored(
            column_tensor(vec![re])?,
            column_tensor(vec![im])?,
            Some("x"),
        ));
        if response.mirror_negative_frequency {
            hidden.push(colored(
                column_tensor(vec![re])?,
                column_tensor(vec![-im])?,
                Some("o"),
            ));
        }
    }
    hidden.push(vec![
        column_tensor(vec![-1.0])?,
        column_tensor(vec![0.0])?,
        Value::from("r+"),
    ]);
    let was_held = state::current_hold_enabled();
    let _ = crate::call_builtin_async("hold", &[Value::from("on")]).await;
    for mut call in hidden {
        call.extend([Value::from("HandleVisibility"), Value::from("off")]);
        let _ = crate::call_builtin_async("plot", &call).await;
    }
    if !was_held {
        let _ = crate::call_builtin_async("hold", &[Value::from("off")]).await;
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Nyquist Diagram")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Real Axis")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Imaginary Axis")]).await;
    let _ = crate::call_builtin_async("grid", &[Value::from("on")]).await;
    Ok(())
}

pub(super) fn plot_error(source: crate::RuntimeError) -> crate::RuntimeError {
    let mapped = crate::build_runtime_error(format!(
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
    .build();
    crate::builtins::plotting::state::with_render_cause(mapped, source)
}
