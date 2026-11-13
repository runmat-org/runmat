//! Genetic charts use the same exact axes/child leases and scene payloads.
use super::*;
use runmat_builtins::OptimGeneticObserverState;

pub(crate) fn update_genetic_plot(
    kind: OptimGeneticPlotKind,
    series: OptimGeneticPlotSeries,
    phase: OptimGeneticObserverState,
    data: GeneticPlotData,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    if phase == OptimGeneticObserverState::Interrupt {
        return Ok(());
    }
    if series != OptimGeneticPlotSeries::Primary
        && !matches!(
            kind,
            OptimGeneticPlotKind::BestFitness | OptimGeneticPlotKind::Range
        )
    {
        return Err(failure("this genetic chart has only one series"));
    }
    let valid = match &data {
        GeneticPlotData::Empty | GeneticPlotData::Reference(_) => false,
        GeneticPlotData::Bars { labels, values } => kind.is_bar() && labels.len() == values.len(),
        GeneticPlotData::Edges { generation, x, y } => {
            kind == OptimGeneticPlotKind::Genealogy && x.len() == y.len() && generation.is_finite()
        }
        GeneticPlotData::Sample { iteration, .. } => {
            !kind.is_bar() && kind != OptimGeneticPlotKind::Genealogy && iteration.is_finite()
        }
    };
    if !valid {
        return Err(failure("genetic plot data does not match its typed chart"));
    }
    let phase = match phase {
        OptimGeneticObserverState::Init => OutputState::Init,
        OptimGeneticObserverState::Iter => OutputState::Iter,
        OptimGeneticObserverState::Done => OutputState::Done,
        OptimGeneticObserverState::Interrupt => {
            unreachable!("interrupt returned before scene work")
        }
    };
    update_series_payload(
        OptimizationSeries::Genetic(kind, series),
        phase,
        data,
        runtime,
        None,
        None,
    )
}

pub(super) fn edges(
    registry: &PlotRegistry,
    handle: FigureHandle,
    existing: Option<usize>,
    phase: OutputState,
    generation: f64,
    x: Vec<f64>,
    y: Vec<f64>,
) -> BuiltinResult<Update> {
    let previous = if phase == OutputState::Init {
        None
    } else {
        existing
            .and_then(
                |index| match registry.figures.get(&handle)?.figure.plots().nth(index)? {
                    PlotElement::Line(line) => Some(line.host_xy_f64()),
                    _ => None,
                },
            )
            .transpose()
            .map_err(failure)?
            .flatten()
    };
    let (mut old_x, mut old_y) = previous.unwrap_or_default();
    // Replace this generation's segments from the ordinary payload itself.
    // A changed observer population at the same generation must not retain
    // obsolete parentage, and Done must not append duplicate edges.
    let mut start = 0;
    let mut retained = 0;
    for end in 0..=old_x.len() {
        if end != old_x.len() && !old_x[end].is_nan() {
            continue;
        }
        let next = if end == old_x.len() { end } else { end + 1 };
        let belongs = old_x[start..end]
            .iter()
            .rev()
            .find(|value| value.is_finite())
            == Some(&generation);
        if !belongs {
            old_x.copy_within(start..next, retained);
            old_y.copy_within(start..next, retained);
            retained += next - start;
        }
        start = next;
    }
    old_x.truncate(retained);
    old_y.truncate(retained);
    old_x.try_reserve_exact(x.len()).map_err(graphics_failure)?;
    old_y.try_reserve_exact(y.len()).map_err(graphics_failure)?;
    old_x.extend(x);
    old_y.extend(y);
    Ok(Update::History(old_x, old_y))
}
