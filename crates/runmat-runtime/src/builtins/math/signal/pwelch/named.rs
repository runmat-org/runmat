//! Name-value admission and plot-target identity for Welch presentation.
use super::*;
#[cfg(feature = "plot-core")]
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
#[cfg(feature = "plot-core")]
use crate::builtins::plotting::state::FigureHandle;

#[cfg(feature = "plot-core")]
type AxesTarget = (FigureHandle, usize);
#[cfg(not(feature = "plot-core"))]
type AxesTarget = ();

pub(super) struct NamedControls<'a> {
    pub positional: Vec<&'a Value>,
    pub tokens: Vec<Option<PwelchKeyword>>,
    pub parent: Option<AxesTarget>,
}

impl NamedControls<'_> {
    pub fn wants_plot(&self) -> bool {
        self.parent.is_some() || statement_form_call()
    }
}

pub(super) fn statement_form_call() -> bool {
    matches!(crate::output_count::current_output_count(), Some(0))
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
}

pub(super) fn split_named_controls(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<NamedControls<'_>> {
    let mut positional = Vec::new();
    positional
        .try_reserve_exact(rest.len())
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(rest.len())
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    let mut parent = None;
    let mut index = 0;
    while index < rest.len() {
        let token = keyword(&rest[index])?;
        if token == Some(PwelchKeyword::Parent) {
            if parent.is_some() {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_PARENT));
            }
            let target = rest
                .get(index + 1)
                .ok_or_else(|| pwelch_error(&PWELCH_ERROR_INVALID_PARENT))?;
            parent = Some(resolve_parent(target)?);
            index += 2;
        } else {
            positional.push(rest.get(index).expect("Welch positional argument"));
            tokens.push(token);
            index += 1;
        }
    }
    Ok(NamedControls {
        positional,
        tokens,
        parent,
    })
}

pub(super) fn keyword(value: &Value) -> BuiltinResult<Option<PwelchKeyword>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => Some(PwelchKeyword::decode(text)),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Some(PwelchKeyword::decode(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string"),
            ))
        }
        Value::CharArray(chars) if chars.rows <= 1 => {
            let text = runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = pwelch_error(&PWELCH_ERROR_INVALID_RANGE);
                error.source = Some(Box::new(source));
                error
            })?;
            Some(PwelchKeyword::decode(&text))
        }
        _ => None,
    })
}

#[cfg(feature = "plot-core")]
fn resolve_parent(value: &Value) -> BuiltinResult<AxesTarget> {
    let handle = resolve_plot_handle(value, BUILTIN_NAME)
        .map_err(|source| pwelch_control_error(&PWELCH_ERROR_INVALID_PARENT, source))?;
    match handle {
        PlotHandle::Axes(figure, axes) => Ok((figure, axes)),
        // RunMat does not yet have UIAxes or Panel object identities. Do not
        // reinterpret another kind of handle as one of those containers.
        _ => Err(pwelch_error(&PWELCH_ERROR_INVALID_PARENT)),
    }
}

#[cfg(not(feature = "plot-core"))]
fn resolve_parent(value: &Value) -> BuiltinResult<AxesTarget> {
    let scalar = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            tensor_utils::tensor_values_f64(tensor)[0]
        }
        _ => return Err(pwelch_error(&PWELCH_ERROR_INVALID_PARENT)),
    };
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_PARENT));
    }
    // Preserve the Parent form through host-visible admission. The caller
    // reports the missing presentation capability before any resident input
    // is downloaded. Existence and axes identity need the plotting registry.
    Ok(())
}
