// Canonical adapters prepare targets without creating or changing a scene.
// Family parsers still own their control grammar and error precedence.
use super::*;
use crate::builtins::plotting::properties::select_graphics_property;
use crate::builtins::plotting::state::{
    resolve_axes_target, resolve_current_axes_for_figure, PropertyMutationTarget,
    ResolvedAxesTarget,
};
use crate::context::RuntimeContext;

/// Existing Figure/Axes roles paired with the exact authority that selected them.
pub(crate) struct PreparedGraphicsTarget {
    authority: PropertyMutationTarget,
    handle: PlotHandle,
}

impl PreparedGraphicsTarget {
    /// Preserve an already-admitted numeric Figure form (including the
    /// existing gca extension); this is routing, not a native receiver issuer.
    pub(crate) fn figure_alias(runtime: Option<&RuntimeContext>, handle: FigureHandle) -> Self {
        Self {
            authority: PropertyMutationTarget::alias().with_runtime(runtime),
            handle: PlotHandle::Figure(handle),
        }
    }
    pub(crate) fn handle(&self) -> &PlotHandle {
        &self.handle
    }

    pub(crate) fn authority(&self) -> &PropertyMutationTarget {
        &self.authority
    }

    pub(crate) fn into_axes(self, builtin: &'static str) -> BuiltinResult<PreparedAxesTarget> {
        match self.handle {
            PlotHandle::Axes(handle, index) => Ok(PreparedAxesTarget {
                authority: self.authority,
                explicit: Some((handle, index)),
            }),
            _ => Err(plotting_error(
                builtin,
                format!("{builtin}: expected an axes handle"),
            )),
        }
    }

    pub(crate) fn resolve_axes(self, builtin: &'static str) -> BuiltinResult<ResolvedAxesTarget> {
        match self.handle {
            PlotHandle::Axes(handle, index) => {
                resolve_axes_target(self.authority, Some((handle, index)))
            }
            PlotHandle::Figure(handle) => resolve_current_axes_for_figure(self.authority, handle),
            _ => unreachable!("only Figure and Axes roles are prepared"),
        }
        .map_err(|source| {
            plotting_error_with_source(builtin, format!("{builtin}: {source}"), source)
        })
    }
}

/// No numeric extraction from an exact object occurs without retaining its
/// original selection. Unsupported non-target values remain controls.
pub(crate) fn prepare_graphics_target(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Option<PreparedGraphicsTarget>> {
    let selected = select_graphics_property(value, builtin)?;
    let handle = match selected
        .as_ref()
        .map(|selection| selection.receiver().kind())
    {
        Some(runmat_value::GraphicsReceiverKind::Figure { figure }) => {
            PlotHandle::Figure(FigureHandle::from(figure))
        }
        Some(runmat_value::GraphicsReceiverKind::Axes { figure, index }) => {
            PlotHandle::Axes(FigureHandle::from(figure), index)
        }
        Some(_) => return Ok(None),
        None => match resolve_plot_handle(value, builtin) {
            Ok(handle @ (PlotHandle::Figure(_) | PlotHandle::Axes(_, _))) => handle,
            _ => return Ok(None),
        },
    };
    let authority = selected
        .map_or_else(
            PropertyMutationTarget::alias,
            PropertyMutationTarget::selected,
        )
        .with_runtime(runtime);
    Ok(Some(PreparedGraphicsTarget { authority, handle }))
}

pub(crate) struct PreparedAxesTarget {
    authority: PropertyMutationTarget,
    explicit: Option<(FigureHandle, usize)>,
}

impl PreparedAxesTarget {
    pub(crate) fn current(runtime: Option<&RuntimeContext>) -> Self {
        Self {
            authority: PropertyMutationTarget::alias().with_runtime(runtime),
            explicit: None,
        }
    }

    /// Call only after the family parser has checked its original numeric form.
    pub(crate) fn alias(
        runtime: Option<&RuntimeContext>,
        handle: FigureHandle,
        index: usize,
    ) -> Self {
        Self {
            authority: PropertyMutationTarget::alias().with_runtime(runtime),
            explicit: Some((handle, index)),
        }
    }

    pub(crate) fn is_explicit(&self) -> bool {
        self.explicit.is_some()
    }

    pub(crate) fn resolve(self, builtin: &'static str) -> BuiltinResult<ResolvedAxesTarget> {
        resolve_axes_target(self.authority, self.explicit).map_err(|source| {
            plotting_error_with_source(builtin, format!("{builtin}: {source}"), source)
        })
    }
}

pub(crate) fn prepare_axes_call<'a>(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<(PreparedAxesTarget, crate::call::FunctionArgumentView<'a>)> {
    if let Some(first) = args.first() {
        if let Some(prepared) = prepare_graphics_target(runtime, builtin, first)? {
            if matches!(prepared.handle(), PlotHandle::Axes(_, _)) {
                return Ok((prepared.into_axes(builtin)?, args.range(1..)?));
            }
        }
    }
    Ok((PreparedAxesTarget::current(runtime), args))
}
