use super::*;
fn setup() -> PlotTestLockGuard {
    let lock = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    lock
}
fn slot(handle: FigureHandle, axes: usize, kind: OptimPlotKind) -> GraphicsReceiverRef {
    registry().figures[&handle].optimization_plots[&(axes, kind.into())].clone()
}
#[test]
fn genetic_genealogy_replaces_same_step_edges_and_retires_with_its_exact_axes() {
    use runmat_builtins::{
        OptimGeneticObserverState as Phase, OptimGeneticPlotKind as Kind,
        OptimGeneticPlotSeries as Series,
    };
    let _lock = setup();
    let layout = OptimizationPlotLayout::prepare(1, None).unwrap();
    layout.select(0, None).unwrap();
    let handle = current_figure_handle();
    update_genetic_plot(
        Kind::Genealogy,
        Series::Primary,
        Phase::Init,
        GeneticPlotData::Edges {
            generation: 0.,
            x: vec![0., f64::NAN],
            y: vec![1., f64::NAN],
        },
        None,
    )
    .unwrap();
    let chart = OptimizationSeries::Genetic(Kind::Genealogy, Series::Primary);
    let receiver = registry().figures[&handle].optimization_plots[&(0, chart)].clone();
    for phase in [Phase::Iter, Phase::Done] {
        update_genetic_plot(
            Kind::Genealogy,
            Series::Primary,
            phase,
            GeneticPlotData::Edges {
                generation: 1.,
                x: vec![0., 1., f64::NAN],
                y: vec![1., 2., f64::NAN],
            },
            None,
        )
        .unwrap();
    }
    {
        let reg = registry();
        let GraphicsReceiverKind::PlotChild { id } = receiver.kind() else {
            panic!("line receiver")
        };
        let child = &reg.plot_children.entries[&id];
        let PlotElement::Line(line) = reg.figures[&handle]
            .figure
            .plots()
            .nth(child.state.plot_index().unwrap())
            .unwrap()
        else {
            panic!("genealogy line")
        };
        let (x, y) = line.host_xy_f64().unwrap().unwrap();
        assert_eq!(x.len(), 5);
        assert_eq!(y.len(), 5);
        assert_eq!([x[0], x[2], x[3]], [0., 0., 1.]);
        assert_eq!([y[0], y[2], y[3]], [1., 1., 2.]);
        assert!(line.marker.is_some());
    }
    clear_axes_target(handle, 0, false).unwrap();
    assert!(!graphics_receiver_valid(&receiver));
    close_figure(Some(handle)).unwrap();
    assert!(layout.select(0, None).is_err());
}
#[test]
fn milp_series_retain_two_real_bound_histories_and_clear_with_their_axes() {
    let _lock = setup();
    let layout = OptimizationPlotLayout::prepare(1, None).unwrap();
    layout.select(0, None).unwrap();
    let handle = current_figure_handle();
    update_milp_plot(OutputState::Init, 0.0, 10.0, 0.0, None).unwrap();
    update_milp_plot(OutputState::Iter, 1.0, 6.0, 4.0, None).unwrap();
    update_milp_plot(OutputState::Done, 1.0, 6.0, 4.0, None).unwrap();
    let mut receivers = Vec::new();
    for (kind, label, expected) in [
        (
            OptimizationSeries::MilpPrimal,
            "Primal Bound",
            vec![10.0, 6.0],
        ),
        (OptimizationSeries::MilpDual, "Dual Bound", vec![0.0, 4.0]),
    ] {
        let reg = registry();
        let state = &reg.figures[&handle];
        let receiver = state.optimization_plots[&(0, kind)].clone();
        let GraphicsReceiverKind::PlotChild { id } = receiver.kind() else {
            panic!("line child");
        };
        let child = &reg.plot_children.entries[&id];
        let PlotElement::Line(line) = state
            .figure
            .plots()
            .nth(child.state.plot_index().unwrap())
            .unwrap()
        else {
            panic!("bound line");
        };
        assert_eq!(line.label.as_deref(), Some(label));
        assert_eq!(
            line.host_xy_f64().unwrap().unwrap(),
            (vec![0.0, 1.0], expected)
        );
        receivers.push(receiver);
    }
    assert!(!receivers[0].shares_lease(&receivers[1]));
    clear_axes_target(handle, 0, false).unwrap();
    assert!(receivers
        .iter()
        .all(|receiver| !graphics_receiver_valid(receiver)));
    assert!(registry().figures[&handle].optimization_plots.is_empty());
}
#[test]
fn assigned_axes_store_real_bar_and_line_payloads_without_duplicate_done_points() {
    let _lock = setup();
    let layout = OptimizationPlotLayout::prepare(4, None).unwrap();
    let GraphicsReceiverKind::Figure { figure } = layout.figure.kind() else {
        panic!("figure receiver")
    };
    let handle = FigureHandle::from(figure);
    for (index, kind) in [
        OptimPlotKind::CurrentPoint,
        OptimPlotKind::FunctionCount,
        OptimPlotKind::FunctionValue,
        OptimPlotKind::ParticleBestFunctionValue,
    ]
    .into_iter()
    .enumerate()
    {
        layout.select(index, None).unwrap();
        update_optimization_plot(kind, OutputState::Init, vec![2.0, -1.0], 0.0, 4.0, None).unwrap();
        let receiver = slot(handle, index, kind);
        let before = registry().figures[&handle].scene_lease.clone();
        update_optimization_plot(kind, OutputState::Iter, vec![1.0, 0.0], 1.0, 2.0, None).unwrap();
        update_optimization_plot(kind, OutputState::Done, vec![1.0, 0.0], 1.0, 2.0, None).unwrap();
        assert!(receiver.shares_lease(&slot(handle, index, kind)));
        let reg = registry();
        let state = &reg.figures[&handle];
        assert!(!before.shares_lease(&state.scene_lease));
        let GraphicsReceiverKind::PlotChild { id } = receiver.kind() else {
            panic!("child")
        };
        let child = &reg.plot_children.entries[&id];
        let plot = state
            .figure
            .plots()
            .nth(child.state.plot_index().unwrap())
            .unwrap();
        match plot {
            PlotElement::Bar(bar) => assert_eq!(bar.values().unwrap(), [1.0, 0.0]),
            PlotElement::Line(line) => assert_eq!(
                line.host_xy_f64().unwrap().unwrap(),
                (vec![0.0, 1.0], vec![4.0, 2.0])
            ),
            _ => panic!("ordinary line/bar payload"),
        }
    }
}
#[test]
fn clear_retires_only_chart_payload_and_close_never_resurrects_layout_targets() {
    let _lock = setup();
    let layout = OptimizationPlotLayout::prepare(1, None).unwrap();
    layout.select(0, None).unwrap();
    let handle = current_figure_handle();
    update_optimization_plot(
        OptimPlotKind::CurrentPoint,
        OutputState::Init,
        vec![2.0],
        0.0,
        0.0,
        None,
    )
    .unwrap();
    let old_bar = slot(handle, 0, OptimPlotKind::CurrentPoint);
    update_optimization_plot(
        OptimPlotKind::FunctionValue,
        OutputState::Init,
        Vec::new(),
        0.0,
        4.0,
        None,
    )
    .unwrap();
    let old = slot(handle, 0, OptimPlotKind::FunctionValue);
    update_optimization_plot(
        OptimPlotKind::CurrentPoint,
        OutputState::Init,
        Vec::new(),
        0.0,
        None::<f64>,
        None,
    )
    .unwrap();
    assert!(!graphics_receiver_valid(&old_bar));
    assert!(graphics_receiver_valid(&old));
    {
        let reg = registry();
        let GraphicsReceiverKind::PlotChild { id } = old.kind() else {
            panic!("line receiver")
        };
        let index = reg.plot_children.entries[&id].state.plot_index().unwrap();
        assert_eq!(index, 0);
        assert_eq!(reg.figures[&handle].figure.plots().count(), 1);
    }
    clear_axes_target(handle, 0, false).unwrap();
    assert!(!graphics_receiver_valid(&old));
    assert!(registry().figures[&handle].optimization_plots.is_empty());
    layout.select(0, None).unwrap();
    update_optimization_plot(
        OptimPlotKind::FunctionValue,
        OutputState::Iter,
        Vec::new(),
        5.0,
        1.0,
        None,
    )
    .unwrap();
    assert!(!old.shares_lease(&slot(handle, 0, OptimPlotKind::FunctionValue)));
    close_figure(Some(handle)).unwrap();
    assert!(layout.select(0, None).is_err());
    assert!(!graphics_receiver_valid(&old));
}

#[test]
fn nonlinear_best_feasible_history_uses_scene_rows_and_retires_with_axes() {
    let _lock = setup();
    let layout = OptimizationPlotLayout::prepare(1, None).unwrap();
    layout.select(0, None).unwrap();
    let handle = current_figure_handle();
    update_constrained_optimization_plot(OutputState::Init, 0.0, 7.0, 0.1, None).unwrap();
    update_constrained_optimization_plot(OutputState::Iter, 1.0, 3.0, 0.0, None).unwrap();
    update_constrained_optimization_plot(OutputState::Iter, 2.0, 4.0, 0.0, None).unwrap();
    update_constrained_optimization_plot(OutputState::Done, 2.0, 4.0, 0.0, None).unwrap();
    let mut receivers = Vec::new();
    let mut colors = Vec::new();
    for (kind, label, expected) in [
        (
            OptimizationSeries::NonlinearFeasible,
            "Feasible",
            [f64::NAN, 3.0, 4.0],
        ),
        (
            OptimizationSeries::NonlinearInfeasible,
            "Infeasible",
            [7.0, f64::NAN, f64::NAN],
        ),
        (
            OptimizationSeries::NonlinearBest,
            "Best feasible",
            [f64::NAN, 3.0, 3.0],
        ),
    ] {
        let reg = registry();
        let state = &reg.figures[&handle];
        let receiver = state.optimization_plots[&(0, kind)].clone();
        let GraphicsReceiverKind::PlotChild { id } = receiver.kind() else {
            panic!("line child")
        };
        let child = &reg.plot_children.entries[&id];
        let PlotElement::Line(line) = state
            .figure
            .plots()
            .nth(child.state.plot_index().unwrap())
            .unwrap()
        else {
            panic!("line")
        };
        assert_eq!(line.label.as_deref(), Some(label));
        let (iterations, values) = line.host_xy_f64().unwrap().unwrap();
        assert_eq!(iterations, vec![0.0, 1.0, 2.0]);
        for (actual, expected) in values.iter().zip(expected) {
            assert!(if expected.is_nan() {
                actual.is_nan()
            } else {
                *actual == expected
            });
        }
        colors.push(line.color);
        receivers.push(receiver);
    }
    assert_ne!(colors[0], colors[1]);
    update_constrained_metric(
        OutputState::Init,
        0.0,
        ConstrainedPlotSample {
            objective: None,
            maximum: Some(0.01),
            tolerance: 0.02,
            feasible_count: Some(7),
        },
        None,
    )
    .unwrap();
    update_constrained_metric(
        OutputState::Iter,
        1.0,
        ConstrainedPlotSample {
            objective: None,
            maximum: Some(-0.2),
            tolerance: 0.02,
            feasible_count: Some(8),
        },
        None,
    )
    .unwrap();
    {
        let reg = registry();
        let receiver =
            &reg.figures[&handle].optimization_plots[&(0, OptimizationSeries::NonlinearFeasible)];
        assert_eq!(
            reg.figures[&handle].figure.axes_metadata[0]
                .title
                .as_deref(),
            Some("Feasible solutions: 8")
        );
        let GraphicsReceiverKind::PlotChild { id } = receiver.kind() else {
            panic!("line receiver")
        };
        let index = reg.plot_children.entries[&id].state.plot_index().unwrap();
        let PlotElement::Line(line) = reg.figures[&handle].figure.plots().nth(index).unwrap()
        else {
            panic!("line")
        };
        assert_eq!(
            line.host_xy_f64().unwrap().unwrap(),
            (vec![0.0, 1.0], vec![0.01, -0.2])
        );
    }
    let sample = |objective, maximum, origin| SurrogatePlotSample {
        objective,
        maximum,
        origin,
    };
    let mut step = SurrogatePlotStep {
        count: 1,
        best: sample(Some(2.0), Some(-0.1), None),
        incumbent: sample(Some(2.0), Some(-0.1), None),
        current: sample(
            Some(2.0),
            Some(-0.1),
            Some(runmat_builtins::SurrogateSampleOrigin::Random),
        ),
        reset: false,
        resume: false,
        reset_count: 0,
        tolerance: 1e-3,
    };
    update_surrogate_plot(OutputState::Init, step, None).unwrap();
    step = SurrogatePlotStep {
        count: 2,
        best: sample(Some(2.0), Some(-0.1), None),
        incumbent: sample(Some(2.0), Some(-0.1), None),
        current: sample(
            Some(3.0),
            Some(0.2),
            Some(runmat_builtins::SurrogateSampleOrigin::Adaptive),
        ),
        reset: true,
        resume: true,
        reset_count: 1,
        tolerance: 1e-3,
    };
    update_surrogate_plot(OutputState::Iter, step, None).unwrap();
    for (series, label) in [
        (surrogate::Series::Reset(1), "Surrogate Reset"),
        (surrogate::Series::CheckpointResume(2), "Checkpoint Resume"),
    ] {
        let reg = registry();
        let receiver =
            &reg.figures[&handle].optimization_plots[&(0, OptimizationSeries::Surrogate(series))];
        let GraphicsReceiverKind::PlotChild { id } = receiver.kind() else {
            panic!("reference receiver")
        };
        let index = reg.plot_children.entries[&id].state.plot_index().unwrap();
        let PlotElement::ReferenceLine(line) =
            reg.figures[&handle].figure.plots().nth(index).unwrap()
        else {
            panic!("real reference line")
        };
        assert_eq!(line.value, 2.0);
        assert_eq!(line.display_name.as_deref(), Some(label));
    }
    update_surrogate_plot(
        OutputState::Init,
        SurrogatePlotStep {
            count: 0,
            best: sample(None, None, None),
            incumbent: sample(None, None, None),
            current: sample(None, None, None),
            reset: false,
            resume: false,
            reset_count: 0,
            tolerance: 1e-3,
        },
        None,
    )
    .unwrap();
    assert!(receivers.iter().all(graphics_receiver_valid));
    assert!(!registry().figures[&handle]
        .optimization_plots
        .keys()
        .any(|(_, kind)| matches!(kind, OptimizationSeries::Surrogate(_))));
    clear_axes_target(handle, 0, false).unwrap();
    assert!(receivers.iter().all(|r| !graphics_receiver_valid(r)));
}
