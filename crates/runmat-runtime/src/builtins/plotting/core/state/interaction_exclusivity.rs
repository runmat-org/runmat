//! Legacy pan/zoom modes are exclusive at the Figure or selected Axes owner.
//!
//! Built-in drag/scroll interactions remain independent: the default mode
//! flags are both off, and enabling one legacy mode only disables the other
//! legacy mode's active flag. Motion/direction preferences are retained.

use super::{pan_mode_for_target, zoom_mode_for_target, FigureState};

pub(super) fn zoom_selected_for_figure(state: &mut FigureState, mode_command: bool) {
    // A Figure command applies to all current and future Axes. A directional
    // mode command replaces prior per-Axes motion; an Enable-property change
    // preserves it. Earlier per-Axes enabled flags never override the Figure.
    for mode in state.zoom_axes_modes.values_mut() {
        mode.enabled = state.zoom_mode.enabled;
        if mode_command && state.zoom_mode.enabled {
            mode.motion = state.zoom_mode.motion;
        }
    }
    if !state.zoom_mode.enabled {
        return;
    }
    state.pan_mode.enabled = false;
    for mode in state.pan_axes_modes.values_mut() {
        mode.enabled = false;
    }
}

pub(super) fn pan_selected_for_figure(state: &mut FigureState, mode_command: bool) {
    for mode in state.pan_axes_modes.values_mut() {
        mode.enabled = state.pan_mode.enabled;
        if mode_command && state.pan_mode.enabled {
            mode.motion = state.pan_mode.motion;
        }
    }
    if !state.pan_mode.enabled {
        return;
    }
    state.zoom_mode.enabled = false;
    for mode in state.zoom_axes_modes.values_mut() {
        mode.enabled = false;
    }
}

pub(super) fn zoom_activated_for_axes(state: &mut FigureState, index: usize) {
    let mut opposing = pan_mode_for_target(state, Some(index));
    opposing.enabled = false;
    state.pan_axes_modes.insert(index, opposing);
}

pub(super) fn pan_activated_for_axes(state: &mut FigureState, index: usize) {
    let mut opposing = zoom_mode_for_target(state, Some(index));
    opposing.enabled = false;
    state.zoom_axes_modes.insert(index, opposing);
}

#[cfg(test)]
mod tests {
    use crate::builtins::plotting::state::{
        create_axes_for_figure, pan_state_snapshot, reset_plot_state, select_figure,
        set_pan_enabled_for_figure, set_pan_mode_for_axes, set_pan_mode_for_figure,
        set_zoom_enabled_for_figure, set_zoom_mode_for_axes, set_zoom_mode_for_figure,
        zoom_state_snapshot, FigureHandle, PanModeCommand, ZoomModeCommand, ZoomMotion,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    #[test]
    fn enabling_either_legacy_mode_retires_only_the_opposing_mode() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        let figure = FigureHandle::from(71);
        select_figure(figure);
        assert!(!zoom_state_snapshot(figure, None).unwrap().mode.enabled);
        assert!(!pan_state_snapshot(figure, None).unwrap().mode.enabled);

        set_pan_mode_for_figure(figure, PanModeCommand::On).unwrap();
        set_zoom_mode_for_figure(figure, ZoomModeCommand::On).unwrap();
        assert!(zoom_state_snapshot(figure, None).unwrap().mode.enabled);
        assert!(!pan_state_snapshot(figure, None).unwrap().mode.enabled);

        set_pan_enabled_for_figure(figure, true).unwrap();
        assert!(!zoom_state_snapshot(figure, None).unwrap().mode.enabled);
        assert!(pan_state_snapshot(figure, None).unwrap().mode.enabled);
        set_zoom_enabled_for_figure(figure, true).unwrap();
        assert!(zoom_state_snapshot(figure, None).unwrap().mode.enabled);
        assert!(!pan_state_snapshot(figure, None).unwrap().mode.enabled);
    }

    #[test]
    fn axes_mode_enables_do_not_leave_both_modes_active_on_one_axes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        let figure = FigureHandle::from(72);
        select_figure(figure);
        set_pan_mode_for_figure(figure, PanModeCommand::On).unwrap();
        set_zoom_mode_for_axes(figure, 0, ZoomModeCommand::On).unwrap();
        assert!(zoom_state_snapshot(figure, Some(0)).unwrap().mode.enabled);
        assert!(!pan_state_snapshot(figure, Some(0)).unwrap().mode.enabled);
        set_pan_mode_for_axes(figure, 0, PanModeCommand::On).unwrap();
        assert!(!zoom_state_snapshot(figure, Some(0)).unwrap().mode.enabled);
        assert!(pan_state_snapshot(figure, Some(0)).unwrap().mode.enabled);
        set_zoom_mode_for_figure(figure, ZoomModeCommand::On).unwrap();
        assert!(!pan_state_snapshot(figure, Some(0)).unwrap().mode.enabled);
    }

    #[test]
    fn figure_commands_override_existing_axes_modes_and_future_axes_inherit() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        let figure = FigureHandle::from(73);
        select_figure(figure);

        set_zoom_mode_for_axes(figure, 0, ZoomModeCommand::Off).unwrap();
        set_zoom_mode_for_figure(figure, ZoomModeCommand::On).unwrap();
        assert!(zoom_state_snapshot(figure, Some(0)).unwrap().mode.enabled);
        set_zoom_mode_for_figure(figure, ZoomModeCommand::Off).unwrap();
        assert!(!zoom_state_snapshot(figure, Some(0)).unwrap().mode.enabled);

        set_pan_mode_for_axes(figure, 0, PanModeCommand::On).unwrap();
        set_pan_mode_for_figure(figure, PanModeCommand::Off).unwrap();
        assert!(!pan_state_snapshot(figure, Some(0)).unwrap().mode.enabled);
        set_pan_mode_for_figure(figure, PanModeCommand::On).unwrap();
        assert!(pan_state_snapshot(figure, Some(0)).unwrap().mode.enabled);

        set_pan_mode_for_axes(figure, 0, PanModeCommand::Off).unwrap();
        set_pan_enabled_for_figure(figure, true).unwrap();
        assert!(pan_state_snapshot(figure, Some(0)).unwrap().mode.enabled);
        set_zoom_mode_for_axes(figure, 0, ZoomModeCommand::Off).unwrap();
        set_zoom_enabled_for_figure(figure, true).unwrap();
        assert!(zoom_state_snapshot(figure, Some(0)).unwrap().mode.enabled);
        assert!(!pan_state_snapshot(figure, Some(0)).unwrap().mode.enabled);

        let (_, future_axes) = create_axes_for_figure(Some(figure)).unwrap();
        assert!(
            zoom_state_snapshot(figure, Some(future_axes))
                .unwrap()
                .mode
                .enabled
        );
        assert!(
            !pan_state_snapshot(figure, Some(future_axes))
                .unwrap()
                .mode
                .enabled
        );
    }

    #[test]
    fn figure_directional_commands_override_axes_motion_then_toggle_restores_selected_motion() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        let figure = FigureHandle::from(74);
        select_figure(figure);

        set_zoom_mode_for_axes(figure, 0, ZoomModeCommand::YOn).unwrap();
        set_zoom_mode_for_figure(figure, ZoomModeCommand::XOn).unwrap();
        assert_eq!(
            zoom_state_snapshot(figure, Some(0)).unwrap().mode.motion,
            ZoomMotion::Horizontal
        );
        set_zoom_mode_for_axes(figure, 0, ZoomModeCommand::XOn).unwrap();
        set_zoom_mode_for_figure(figure, ZoomModeCommand::YOn).unwrap();
        assert_eq!(
            zoom_state_snapshot(figure, Some(0)).unwrap().mode.motion,
            ZoomMotion::Vertical
        );
        set_zoom_mode_for_figure(figure, ZoomModeCommand::Off).unwrap();
        set_zoom_mode_for_figure(figure, ZoomModeCommand::Toggle).unwrap();
        assert_eq!(
            zoom_state_snapshot(figure, Some(0)).unwrap().mode.motion,
            ZoomMotion::Vertical
        );
        set_zoom_mode_for_figure(figure, ZoomModeCommand::On).unwrap();
        assert_eq!(
            zoom_state_snapshot(figure, Some(0)).unwrap().mode.motion,
            ZoomMotion::Both
        );

        set_pan_mode_for_axes(figure, 0, PanModeCommand::YOn).unwrap();
        set_pan_mode_for_figure(figure, PanModeCommand::XOn).unwrap();
        assert_eq!(
            pan_state_snapshot(figure, Some(0)).unwrap().mode.motion,
            ZoomMotion::Horizontal
        );
        set_pan_mode_for_axes(figure, 0, PanModeCommand::XOn).unwrap();
        set_pan_mode_for_figure(figure, PanModeCommand::YOn).unwrap();
        assert_eq!(
            pan_state_snapshot(figure, Some(0)).unwrap().mode.motion,
            ZoomMotion::Vertical
        );
        set_pan_mode_for_figure(figure, PanModeCommand::Off).unwrap();
        set_pan_mode_for_figure(figure, PanModeCommand::Toggle).unwrap();
        assert_eq!(
            pan_state_snapshot(figure, Some(0)).unwrap().mode.motion,
            ZoomMotion::Vertical
        );
    }

    #[test]
    fn figure_enable_property_preserves_selected_axes_motion_preferences() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        let figure = FigureHandle::from(75);
        select_figure(figure);

        set_zoom_mode_for_axes(figure, 0, ZoomModeCommand::YOn).unwrap();
        set_zoom_enabled_for_figure(figure, false).unwrap();
        set_zoom_enabled_for_figure(figure, true).unwrap();
        assert_eq!(
            zoom_state_snapshot(figure, Some(0)).unwrap().mode.motion,
            ZoomMotion::Vertical
        );

        set_pan_mode_for_axes(figure, 0, PanModeCommand::XOn).unwrap();
        set_pan_enabled_for_figure(figure, false).unwrap();
        set_pan_enabled_for_figure(figure, true).unwrap();
        assert_eq!(
            pan_state_snapshot(figure, Some(0)).unwrap().mode.motion,
            ZoomMotion::Horizontal
        );
    }
}
