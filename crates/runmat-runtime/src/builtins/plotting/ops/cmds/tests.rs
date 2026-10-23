use super::colorbar::colorbar_builtin_source;
use super::*;
use runmat_value::Value;

type Command = fn(Vec<Value>) -> crate::BuiltinResult<()>;

#[test]
fn graphics_mode_pages_reject_before_creating_or_mutating_a_figure() {
    let _guard = test_support::setup();
    let modes = ["manual", "on", "reset", "on", "jet", "on", "on", "flat"];
    for ((command, _), mode) in command_cases().into_iter().zip(modes) {
        for shape in [vec![mode.len(), 1], vec![1, 1, mode.len()]] {
            crate::builtins::plotting::state::reset_plot_state();
            let chars = runmat_value::CharArray::from_unicode_scalars_with_shape(
                mode.chars().collect(),
                shape,
            )
            .unwrap();
            assert!(command(vec![Value::CharArray(chars)]).is_err());
            assert!(crate::builtins::plotting::state::current_figure_handle_if_exists().is_none());
        }
        let chars = runmat_value::CharArray::from_unicode_scalars_with_shape(
            mode.chars().collect(),
            vec![1, mode.len(), 1],
        )
        .unwrap();
        command(vec![Value::CharArray(chars)]).expect("trailing singleton character row");
    }
}

fn command_cases() -> [(Command, Vec<Value>); 8] {
    [
        (
            |args| {
                axis_builtin(crate::builtins::common::test_support::builtin_arguments(
                    args,
                ))
                .map(|_| ())
            },
            vec![Value::String("manual".into())],
        ),
        (
            |args| {
                box_builtin(crate::builtins::common::test_support::builtin_arguments(
                    args,
                ))
                .map(|_| ())
            },
            vec![],
        ),
        (|args| cla_builtin(args).map(|_| ()), vec![]),
        (|args| colorbar_builtin_source(args).map(|_| ()), vec![]),
        (
            |args| {
                colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
                    args,
                ))
                .map(|_| ())
            },
            vec![],
        ),
        (
            |args| {
                grid_builtin(crate::builtins::common::test_support::builtin_arguments(
                    args,
                ))
                .map(|_| ())
            },
            vec![],
        ),
        (
            |args| {
                hidden_builtin(crate::builtins::common::test_support::builtin_arguments(
                    args,
                ))
                .map(|_| ())
            },
            vec![],
        ),
        (
            |args| {
                shading_builtin(crate::builtins::common::test_support::builtin_arguments(
                    args,
                ))
                .map(|_| ())
            },
            vec![Value::String("flat".into())],
        ),
    ]
}

#[test]
fn graphics_commands_reject_excess_outputs_before_creation() {
    let _guard = test_support::setup();
    for (command, arguments) in command_cases() {
        crate::builtins::plotting::state::reset_plot_state();
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        assert!(command(arguments).is_err());
        assert!(crate::builtins::plotting::state::current_figure_handle_if_exists().is_none());
    }
}

#[test]
fn graphics_commands_accept_zero_outputs_in_strict_mode() {
    let _guard = test_support::setup();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    for (command, arguments) in command_cases() {
        crate::builtins::plotting::state::reset_plot_state();
        command(arguments).unwrap();
        assert!(crate::builtins::plotting::state::current_figure_handle_if_exists().is_some());
    }
}

#[test]
fn graphics_commands_reject_resident_controls_without_creating_graphics() {
    let _guard = test_support::setup();
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    for (command, _) in command_cases() {
        crate::builtins::plotting::state::reset_plot_state();
        assert!(command(vec![resident.clone()]).is_err());
        assert!(crate::builtins::plotting::state::current_figure_handle_if_exists().is_none());
    }
}
#[test]
fn command_descriptors_cover_core_forms() {
    let grid_labels: Vec<&str> = GRID_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(grid_labels.contains(&"enabled = grid()"));
    assert!(grid_labels.contains(&"enabled = grid(mode)"));

    let box_labels: Vec<&str> = BOX_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(box_labels.contains(&"enabled = box()"));
    assert!(box_labels.contains(&"enabled = box(mode)"));

    let axis_labels: Vec<&str> = AXIS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(axis_labels.contains(&"lim = axis()"));
    assert!(axis_labels.contains(&"lim = axis(ax)"));
    assert!(axis_labels.contains(&"axis([xmin xmax ymin ymax | ... cmin cmax])"));
    assert!(axis_labels.contains(&"axis(mode)"));
    assert!(axis_labels.contains(&"axis(visibility)"));

    let cla_labels: Vec<&str> = CLA_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(cla_labels.contains(&"ok = cla()"));

    let colormap_labels: Vec<&str> = COLORMAP_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(colormap_labels.contains(&"cmap = colormap()"));
    assert!(colormap_labels.contains(&"cmap = colormap(target)"));
    assert!(colormap_labels.contains(&"colormap(name)"));
    assert!(colormap_labels.contains(&"colormap(map)"));
    assert!(colormap_labels.contains(&"colormap(target, map)"));

    let shading_labels: Vec<&str> = SHADING_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(shading_labels.contains(&"ok = shading(mode)"));

    let hidden_labels: Vec<&str> = HIDDEN_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(hidden_labels.contains(&"enabled = hidden()"));
    assert!(hidden_labels.contains(&"enabled = hidden(mode)"));
    assert!(hidden_labels.contains(&"enabled = hidden(ax)"));
    assert!(hidden_labels.contains(&"enabled = hidden(ax, mode)"));

    let colorbar_labels: Vec<&str> = COLORBAR_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(colorbar_labels.contains(&"enabled = colorbar()"));
    assert!(colorbar_labels.contains(&"enabled = colorbar(mode)"));
}
