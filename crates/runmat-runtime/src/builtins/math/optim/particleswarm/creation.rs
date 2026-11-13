//! Uniform/custom creation and honest reuse of supplied same-point fitness.
use super::{
    fitness::{Fitness, Measurement},
    invalid,
    population::{capacity, Population},
    random, ScalarCooperation,
};
use crate::BuiltinResult;
use runmat_value::{NumericDType, Tensor, Value};
pub(super) enum Source {
    Uniform,
    Custom(Value),
}
pub(super) struct Initial {
    pub positions: Tensor,
    pub fitness: Option<Fitness>,
}
pub(super) struct Created {
    pub population: Population,
    pub velocity: Vec<f64>,
    pub supplied: Vec<Option<Measurement>>,
    pub clipped: usize,
}
pub(super) struct Request<'a> {
    pub source: &'a Source,
    pub problem: &'a Value,
    pub initial: Option<&'a Initial>,
    pub members: usize,
    pub variables: usize,
    pub span: &'a [f64],
    pub lower: &'a [f64],
    pub upper: &'a [f64],
}
fn mix(a: f64, b: f64, u: f64) -> Option<f64> {
    let value = if (b - a).is_finite() {
        a + u * (b - a)
    } else {
        (1.0 - u) * a + u * b
    };
    value.is_finite().then_some(value)
}
fn interval(lower: f64, upper: f64, span: f64) -> BuiltinResult<(f64, f64)> {
    let pair = if lower.is_finite() && upper.is_finite() {
        // Both finite bounds define the uniform creation interval. Span still
        // limits initial velocities; one-sided/unbounded creation uses span.
        (lower, upper)
    } else if lower.is_finite() {
        (lower, lower + span)
    } else if upper.is_finite() {
        (upper - span, upper)
    } else {
        (-0.5 * span, 0.5 * span)
    };
    if !pair.0.is_finite() || !pair.1.is_finite() || pair.0 > pair.1 {
        return Err(invalid(
            "initial creation interval exceeds finite arithmetic",
        ));
    }
    Ok(pair)
}
pub(super) async fn create(
    request: Request<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Created> {
    let Request {
        source,
        problem,
        initial,
        members,
        variables,
        span,
        lower,
        upper,
    } = request;
    if members < 2
        || span.len() != variables
        || lower.len() != variables
        || upper.len() != variables
    {
        return Err(invalid("creation layout or bound extent mismatch"));
    }
    let length = members
        .checked_mul(variables)
        .ok_or_else(|| invalid("swarm extent overflow"))?;
    let positions = match source {
        Source::Uniform => {
            uniform_positions(initial, members, variables, span, lower, upper, cooperation).await?
        }
        Source::Custom(function) => {
            let returned = crate::builtins::math::optim::common::call_function_in_context(
                cooperation.runtime(),
                function,
                [problem.clone()],
                1,
            )
            .await?;
            returned
                .with_owned_async(|value| async move {
                    match value {
                        Value::Tensor(v)
                            if v.numeric_dtype() == NumericDType::F64
                                && v.shape() == [members, variables] =>
                        {
                            Ok(v)
                        }
                        _ => Err(invalid(
                            "CreationFcn must return a full double SwarmSize-by-nvars matrix",
                        )),
                    }
                })
                .await?
        }
    };
    let mut population = Population::new(super::POLICY, positions, members, variables)?;
    population.writable(cooperation).await?;
    let mut supplied = capacity(members, cooperation)?;
    let mut clipped = 0usize;
    for member in 0..members {
        let mut same = initial.is_some_and(|v| member < v.positions.shape()[0]);
        let mut changed = false;
        for column in 0..variables {
            let value = population.get(member, column);
            if !value.is_finite() {
                return Err(invalid("initial swarm coordinates must be finite"));
            }
            let bounded = if value < lower[column] {
                lower[column]
            } else if value > upper[column] {
                upper[column]
            } else {
                value
            };
            changed |= value.to_bits() != bounded.to_bits();
            if let Some(initial) = initial {
                if member < initial.positions.shape()[0] {
                    same &= bounded.to_bits()
                        == initial
                            .positions
                            .as_f64_slice()
                            .expect("double initial points")
                            [member + initial.positions.shape()[0] * column]
                            .to_bits();
                }
            }
            population
                .positions
                .try_as_f64_slice_mut_unique()
                .expect("writable initial generation")[member + members * column] = bounded;
            cooperation.completed(1).await?;
        }
        if changed {
            clipped = clipped
                .checked_add(1)
                .ok_or_else(|| invalid("clipped particle count overflow"))?;
        }
        supplied.push(if same {
            initial
                .and_then(|v| v.fitness.as_ref())
                .filter(|v| member < v.len())
                .map(|v| v.measurement(member))
        } else {
            None
        });
    }
    let draws = random::uniform(length, cooperation).await?;
    let mut velocity = capacity(length, cooperation)?;
    for column in 0..variables {
        let width = if (upper[column] - lower[column]).is_finite() {
            (upper[column] - lower[column]).min(span[column])
        } else {
            span[column]
        };
        if !width.is_finite() || width < 0.0 {
            return Err(invalid("initial velocity range is invalid"));
        }
        for member in 0..members {
            let value = (2.0 * draws[member + members * column] - 1.0) * width;
            if !value.is_finite() {
                return Err(invalid("initial velocity is not representable"));
            }
            velocity.push(value);
            cooperation.completed(1).await?;
        }
    }
    Ok(Created {
        population,
        velocity,
        supplied,
        clipped,
    })
}

pub(super) async fn uniform_positions(
    initial: Option<&Initial>,
    members: usize,
    variables: usize,
    span: &[f64],
    lower: &[f64],
    upper: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Tensor> {
    let length = members
        .checked_mul(variables)
        .ok_or_else(|| invalid("swarm extent overflow"))?;
    let initial_rows = initial.map_or(0, |v| v.positions.shape()[0].min(members));
    let missing = members - initial_rows;
    if missing > 0 {
        for column in 0..variables {
            interval(lower[column], upper[column], span[column])?;
            cooperation.completed(1).await?;
        }
    }
    let draws = if missing == 0 {
        Vec::new()
    } else {
        random::uniform(
            missing
                .checked_mul(variables)
                .ok_or_else(|| invalid("creation random count overflow"))?,
            cooperation,
        )
        .await?
    };
    let mut draw = draws.into_iter();
    let mut values = capacity(length, cooperation)?;
    for column in 0..variables {
        let (a, b) = if missing > 0 {
            interval(lower[column], upper[column], span[column])?
        } else {
            (0.0, 0.0)
        };
        for member in 0..members {
            let value = if member < initial_rows {
                initial
                    .expect("supplied prefix")
                    .positions
                    .as_f64_slice()
                    .expect("double initial points")
                    [member + initial.expect("prefix").positions.shape()[0] * column]
            } else {
                mix(a, b, draw.next().expect("admitted creation range"))
                    .ok_or_else(|| invalid("initial uniform point is not representable"))?
            };
            values.push(value);
            cooperation.completed(1).await?;
        }
    }
    Tensor::new(values, vec![members, variables]).map_err(invalid)
}
