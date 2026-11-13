//! Staged chart data is temporary; only ordinary graphics payloads retain it.
use super::{
    super::{invalid, ScalarCooperation, POLICY},
    plot_metrics as metrics,
    plot_state::State,
};
use crate::builtins::plotting::state::GeneticPlotData as Data;
use crate::BuiltinResult;
use runmat_builtins::{
    OptimGeneticPlotKind as Kind, OptimGeneticPlotSeries as Series, OptimGeneticStateField as Field,
};
use runmat_value::Value;

pub(super) async fn prepare(
    kind: Kind,
    state: &State<'_>,
    controls: &Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<(Series, Data)>> {
    let mut generation = exact(state.count(Field::Generation, cooperation).await?)?;
    if kind == Kind::Genealogy {
        if let Some(value) = state.optional(Field::TotalInnerGenerations) {
            let steps =
                super::admission::count(value, "TotalInnerGenerations", cooperation).await?;
            if steps != 0 {
                generation = exact(steps)?;
            }
        }
    }
    let mut charts = POLICY.capacity(3, cooperation.runtime())?;
    match kind {
        Kind::BestFitness
        | Kind::BestIndividual
        | Kind::Range
        | Kind::ScoreDiversity
        | Kind::Scores => {
            let fitness = state.scores(cooperation).await?;
            let scores = fitness.scores(cooperation).await?;
            let summary = metrics::summarize(&scores, cooperation).await?;
            match kind {
                Kind::BestFitness => {
                    charts.push((Series::Primary, sample(generation, summary.minimum)));
                    charts.push((Series::Secondary, sample(generation, summary.mean)));
                }
                Kind::Range => {
                    charts.push((Series::Primary, sample(generation, summary.minimum)));
                    charts.push((Series::Secondary, sample(generation, summary.maximum)));
                    charts.push((Series::Tertiary, sample(generation, summary.mean)));
                }
                Kind::BestIndividual => {
                    let population = state.matrix(Field::Population, cooperation).await?;
                    if population.shape().len() != 2 || population.shape()[0] != scores.len() {
                        return Err(invalid("best-individual population rows must match Score"));
                    }
                    let columns = population.shape()[1];
                    let mut values = POLICY.capacity(columns, cooperation.runtime())?;
                    let data = population
                        .as_f64_slice()
                        .expect("prepared double population");
                    for column in 0..columns {
                        values.push(
                            summary
                                .best
                                .map_or(f64::NAN, |row| data[row + scores.len() * column]),
                        );
                        cooperation.completed(1).await?;
                    }
                    if !values.is_empty() {
                        charts.push((Series::Primary, bars(values, cooperation).await?));
                    }
                }
                Kind::Scores => {
                    let mut values = POLICY.capacity(scores.len(), cooperation.runtime())?;
                    for value in scores {
                        values.push(value.unwrap_or(f64::NAN));
                        cooperation.completed(1).await?;
                    }
                    if !values.is_empty() {
                        charts.push((Series::Primary, bars(values, cooperation).await?));
                    }
                }
                Kind::ScoreDiversity => {
                    let (labels, values) = metrics::histogram(&scores, cooperation).await?;
                    if !values.is_empty() {
                        charts.push((Series::Primary, Data::Bars { labels, values }));
                    } else {
                        charts.push((
                            Series::Primary,
                            availability("No finite scores", 0., cooperation)?,
                        ));
                    }
                }
                _ => unreachable!("score charts have closed typed cases"),
            }
        }
        Kind::Distance => {
            let population = state.matrix(Field::Population, cooperation).await?;
            let value = metrics::distance(
                population.shape(),
                population.as_f64_slice().expect("prepared population"),
                cooperation,
            )
            .await?;
            charts.push((Series::Primary, sample(generation, value)));
        }
        Kind::Expectation => {
            let values = state.vector(Field::Expectation, cooperation).await?;
            for value in &values {
                if !value.is_finite() || *value < 0. {
                    return Err(invalid(
                        "selection expectation must be finite and nonnegative",
                    ));
                }
                cooperation.completed(1).await?;
            }
            if !values.is_empty() {
                charts.push((Series::Primary, bars(values, cooperation).await?));
            } else {
                charts.push((
                    Series::Primary,
                    availability("Expectation unavailable", f64::NAN, cooperation)?,
                ));
            }
        }
        Kind::Selection => {
            // Only the score vector's immutable geometry supplies member
            // count; its payload and residency are not read or downloaded.
            let members = crate::value_fact::value_fact(state.field(Field::Score)?)
                .shape
                .element_count()
                .ok_or_else(|| invalid("selection histogram requires known member count"))?;
            let selected = state.vector(Field::Selection, cooperation).await?;
            let mut counts = POLICY.capacity(members, cooperation.runtime())?;
            counts.resize(members, 0.);
            for selected in selected {
                let row = index(selected, members)?;
                counts[row] += 1.;
                cooperation.completed(1).await?;
            }
            if !counts.is_empty() {
                charts.push((Series::Primary, bars(counts, cooperation).await?));
            }
        }
        Kind::MaxConstraint => {
            let mut maximum = 0_f64;
            for (field, equality) in [
                (Field::NonlinearInequality, false),
                (Field::NonlinearEquality, true),
            ] {
                let values = state.matrix(field, cooperation).await?;
                for value in values.as_f64_slice().expect("prepared constraints") {
                    if !value.is_finite() {
                        maximum = f64::NAN;
                        break;
                    }
                    maximum = maximum.max(if equality { value.abs() } else { *value });
                    cooperation.completed(1).await?;
                }
                if maximum.is_nan() {
                    break;
                }
            }
            charts.push((Series::Primary, sample(generation, maximum)));
        }
        Kind::Genealogy => {
            let parents = state.matrix(Field::GeneticParents, cooperation).await?;
            if parents.shape().len() != 2 || parents.shape()[1] != 2 {
                return Err(invalid("GeneticParents must have two columns"));
            }
            let members = parents.shape()[0];
            let count = members
                .checked_mul(6)
                .ok_or_else(|| invalid("genealogy edge extent overflow"))?;
            let mut x = POLICY.capacity(count, cooperation.runtime())?;
            let mut y = POLICY.capacity(count, cooperation.runtime())?;
            let values = parents.as_f64_slice().expect("prepared parent indices");
            for child in 0..members {
                let mut edge = false;
                for column in 0..2 {
                    let parent = values[child + members * column];
                    if !parent.is_nan() {
                        let parent = index(parent, members)?;
                        if generation == 0. {
                            return Err(invalid(
                                "initial genealogy cannot refer to a previous generation",
                            ));
                        }
                        x.extend([generation - 1., generation, f64::NAN]);
                        y.extend([
                            exact(
                                parent
                                    .checked_add(1)
                                    .ok_or_else(|| invalid("parent count overflow"))?,
                            )?,
                            exact(
                                child
                                    .checked_add(1)
                                    .ok_or_else(|| invalid("child count overflow"))?,
                            )?,
                            f64::NAN,
                        ]);
                        edge = true;
                    }
                    cooperation.completed(1).await?;
                }
                if !edge {
                    x.extend([generation, f64::NAN]);
                    y.extend([
                        exact(
                            child
                                .checked_add(1)
                                .ok_or_else(|| invalid("child count overflow"))?,
                        )?,
                        f64::NAN,
                    ]);
                }
            }
            charts.push((Series::Primary, Data::Edges { generation, x, y }));
        }
        Kind::Stopping => {
            let (labels, values) =
                super::plot_stopping::values(controls, state, cooperation).await?;
            charts.push((Series::Primary, Data::Bars { labels, values }));
        }
    }
    Ok(charts)
}
fn sample(iteration: f64, value: f64) -> Data {
    Data::Sample {
        point: Vec::new(),
        iteration,
        value,
    }
}
fn availability(label: &str, value: f64, cooperation: &ScalarCooperation) -> BuiltinResult<Data> {
    let mut labels = POLICY.capacity(1, cooperation.runtime())?;
    labels.push(label.to_owned());
    let mut values = POLICY.capacity(1, cooperation.runtime())?;
    values.push(value);
    Ok(Data::Bars { labels, values })
}
fn exact(count: usize) -> BuiltinResult<f64> {
    u64::try_from(count)
        .ok()
        .and_then(|n| runmat_value::IntValue::U64(n).to_exact_f64())
        .ok_or_else(|| invalid("genetic plot count is not exactly representable"))
}
fn index(value: f64, members: usize) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1. || value.fract() != 0. || value >= usize::MAX as f64 {
        return Err(invalid(
            "genetic plot parent index must be exact and positive",
        ));
    }
    let index = value as usize - 1;
    if index >= members {
        return Err(invalid("genetic plot parent index exceeds member count"));
    }
    Ok(index)
}
async fn bars(values: Vec<f64>, cooperation: &mut ScalarCooperation) -> BuiltinResult<Data> {
    let mut labels = POLICY.capacity(values.len(), cooperation.runtime())?;
    for index in 0..values.len() {
        labels.push((index + 1).to_string());
        cooperation.completed(1).await?;
    }
    Ok(Data::Bars { labels, values })
}
