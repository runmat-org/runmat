//! Lightweight wrappers that expose plotting-specific helpers without requiring
//! downstream crates to depend directly on the plotting feature flag.

/// The unavailable capability has a typed cause even when `runmat-plot` is
/// absent. Presentation callers must not turn this into a successful numeric
/// value or suppress it as a render failure.
#[cfg(not(feature = "plot-core"))]
#[derive(Debug)]
pub(crate) struct PlottingUnavailable;

#[cfg(not(feature = "plot-core"))]
impl std::fmt::Display for PlottingUnavailable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(PLOTTING_UNAVAILABLE_MESSAGE)
    }
}

#[cfg(not(feature = "plot-core"))]
impl std::error::Error for PlottingUnavailable {}

pub(crate) const PLOTTING_UNAVAILABLE_MESSAGE: &str =
    "Plotting is unavailable in this build (enable the `gui` or `plot-web` feature).";

#[cfg(not(feature = "plot-core"))]
pub(crate) fn unavailable_presentation() -> crate::RuntimeError {
    crate::build_runtime_error(PLOTTING_UNAVAILABLE_MESSAGE)
        .with_identifier("RunMat:plot:EngineError")
        .with_source(PlottingUnavailable)
        .build()
}

#[cfg(feature = "plot-core")]
pub(crate) fn presentation_failed(error: &crate::RuntimeError) -> bool {
    crate::builtins::plotting::state::presentation_failure_kind(error).is_some()
}

#[cfg(not(feature = "plot-core"))]
pub(crate) fn presentation_failed(_error: &crate::RuntimeError) -> bool {
    false
}

/// A graphics receiver is authoritative only while the plot registry owns its
/// exact lease. Without plot-core there is no registry that can own one.
#[cfg(feature = "plot-core")]
pub(crate) fn graphics_receiver_valid(receiver: &runmat_value::GraphicsReceiverRef) -> bool {
    crate::builtins::plotting::state::graphics_receiver_valid(receiver)
}

#[cfg(not(feature = "plot-core"))]
pub(crate) fn graphics_receiver_valid(_receiver: &runmat_value::GraphicsReceiverRef) -> bool {
    false
}

pub(crate) fn graphics_callback_payload_valid(
    receiver: &runmat_value::GraphicsReceiverRef,
    kind: runmat_value::GraphicsCallbackKind,
    payload: &runmat_value::GraphicsCallbackRolePayload,
) -> bool {
    #[cfg(feature = "plot-core")]
    {
        crate::builtins::plotting::state::callback_payload_valid(receiver, kind, payload)
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = (receiver, kind, payload);
        false
    }
}

pub(crate) fn graphics_callback_payload_valid_at_scene(
    receiver: &runmat_value::GraphicsReceiverRef,
    kind: runmat_value::GraphicsCallbackKind,
    payload: &runmat_value::GraphicsCallbackRolePayload,
    scene: &runmat_value::GraphicsSceneLease,
) -> bool {
    #[cfg(feature = "plot-core")]
    {
        crate::builtins::plotting::state::callback_payload_valid_at_scene(
            receiver, kind, payload, scene,
        )
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = (receiver, kind, payload, scene);
        false
    }
}

pub(crate) fn graphics_receiver_class(
    receiver: &runmat_value::GraphicsReceiverRef,
) -> Option<runmat_types::StaticClassIdentity> {
    #[cfg(feature = "plot-core")]
    {
        crate::builtins::plotting::state::graphics_receiver_class(receiver)
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = receiver;
        None
    }
}

/// Preserve the owning builtin's public error while retaining the original
/// cause, span, and retry disposition.
pub(crate) fn with_render_cause(
    mut mapped: crate::RuntimeError,
    cause: crate::RuntimeError,
) -> crate::RuntimeError {
    if mapped.span.is_none() {
        mapped.span = cause.span;
    }
    mapped.gpu_gather_retry = cause.gpu_gather_retry;
    mapped.source = Some(Box::new(cause));
    mapped
}

/// Reset the selected invocation's touched-figure receipt, or the thread-local
/// receipt when no runtime context is selected. No-op when plotting is disabled.
pub fn reset_recent_figures() {
    #[cfg(feature = "plot-core")]
    {
        crate::builtins::plotting::reset_recent_figures();
    }
}

/// Drain the selected invocation's touched-figure receipt, returning raw handles.
/// Without a selected runtime context, drain the thread-local receipt instead.
pub fn take_recent_figures() -> Vec<u32> {
    #[cfg(feature = "plot-core")]
    {
        crate::builtins::plotting::take_recent_figures()
            .into_iter()
            .map(|handle| handle.as_u32())
            .collect()
    }
    #[cfg(not(feature = "plot-core"))]
    {
        Vec::new()
    }
}

/// Record raw figure handles that were touched on another execution thread.
pub fn record_recent_figures(handles: impl IntoIterator<Item = u32>) {
    #[cfg(feature = "plot-core")]
    {
        for handle in handles {
            crate::builtins::plotting::record_recent_figure(handle.into());
        }
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = handles;
    }
}

#[cfg(all(test, not(feature = "plot-core")))]
mod unavailable_tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn disabled_presentation_retains_typed_unavailable_source() {
        let error = unavailable_presentation();
        assert_eq!(error.identifier(), Some("RunMat:plot:EngineError"));
        assert!(error
            .source()
            .and_then(|source| source.downcast_ref::<PlottingUnavailable>())
            .is_some());
        assert!(!presentation_failed(&error));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn issued_graphics_receivers_have_no_authority_without_plot_core() {
        use runmat_value::{GraphicsReceiverKind, GraphicsReceiverRef};

        for kind in [
            GraphicsReceiverKind::Figure { figure: 7 },
            GraphicsReceiverKind::Axes {
                figure: 7,
                index: 0,
            },
        ] {
            let receiver = GraphicsReceiverRef::issued(kind, 1);
            assert!(!graphics_receiver_valid(&receiver));
        }
    }
}
