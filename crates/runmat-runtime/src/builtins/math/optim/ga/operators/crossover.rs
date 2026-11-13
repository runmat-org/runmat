//! Real crossover proposals. Original bounds/linear/integer feasibility is
//! admitted separately before installing a population generation.
use super::super::{invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticCrossover;

pub(in super::super) struct Parameters<'a> {
    pub kind: OptimGeneticCrossover,
    pub ratio: &'a [f64],
    pub first_is_better: bool,
}

/// Preserve finite convex combinations even when subtracting the two parents
/// would overflow. Extrapolations remain undefined if the actual result is
/// outside finite arithmetic; no gene is silently clamped here.
pub(in super::super) use super::super::common::affine_coordinate::interpolate;

fn uniform(value: f64) -> bool {
    value.is_finite() && (0.0..1.0).contains(&value)
}

pub(in super::super) fn draw_count(parameters: &Parameters<'_>, n: usize) -> BuiltinResult<usize> {
    Ok(match parameters.kind {
        OptimGeneticCrossover::Scattered => n,
        OptimGeneticCrossover::SinglePoint | OptimGeneticCrossover::Arithmetic => 1,
        OptimGeneticCrossover::TwoPoint => 2,
        OptimGeneticCrossover::Intermediate => {
            if parameters.ratio.len() == 1 {
                1
            } else {
                n
            }
        }
        OptimGeneticCrossover::Heuristic => 0,
        OptimGeneticCrossover::Laplace => n
            .checked_add(1)
            .ok_or_else(|| invalid("Laplace random extent overflow"))?,
    })
}

pub(in super::super) async fn child(
    parameters: Parameters<'_>,
    first: &[f64],
    second: &[f64],
    draws: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let n = first.len();
    if n == 0
        || second.len() != n
        || draws.len() != draw_count(&parameters, n)?
        || first.iter().chain(second).any(|value| !value.is_finite())
        || draws.iter().any(|&value| !uniform(value))
        || (!parameters.ratio.is_empty()
            && parameters.ratio.len() != 1
            && parameters.ratio.len() != n)
        || parameters
            .ratio
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || (!matches!(
            parameters.kind,
            OptimGeneticCrossover::Intermediate | OptimGeneticCrossover::Heuristic
        ) && !parameters.ratio.is_empty())
        || (parameters.kind == OptimGeneticCrossover::Heuristic && parameters.ratio.len() > 1)
    {
        return Err(invalid(
            "invalid crossover population, parameters or random extent",
        ));
    }
    let mut values = POLICY.capacity(n, cooperation.runtime())?;
    let cut = |draw: f64| ((draw * n as f64) as usize).min(n - 1) + 1;
    let (low, high) = {
        if parameters.kind == OptimGeneticCrossover::TwoPoint {
            let a = cut(draws[0]);
            let b = cut(draws[1]);
            (a.min(b), a.max(b))
        } else {
            (0, 0)
        }
    };
    let ratio = |column: usize, default: f64| {
        if parameters.ratio.is_empty() {
            default
        } else {
            parameters.ratio[if parameters.ratio.len() == 1 {
                0
            } else {
                column
            }]
        }
    };
    for column in 0..n {
        let a = first[column];
        let b = second[column];
        let value = match parameters.kind {
            OptimGeneticCrossover::Scattered => Some(if draws[column] < 0.5 { a } else { b }),
            OptimGeneticCrossover::SinglePoint => Some(if column < cut(draws[0]) { a } else { b }),
            OptimGeneticCrossover::TwoPoint => {
                Some(if column >= low && column < high { b } else { a })
            }
            OptimGeneticCrossover::Intermediate => {
                let draw = if parameters.ratio.len() == 1 {
                    draws[0]
                } else {
                    draws[column]
                };
                interpolate(a, b, draw * ratio(column, 1.0))
            }
            OptimGeneticCrossover::Arithmetic => interpolate(a, b, draws[0]),
            OptimGeneticCrossover::Heuristic => {
                let (better, worse) = if parameters.first_is_better {
                    (a, b)
                } else {
                    (b, a)
                };
                interpolate(worse, better, ratio(column, 1.2))
            }
            OptimGeneticCrossover::Laplace => {
                // Inverse CDF of a centred unit Laplace variate. Zero is a
                // genuine endpoint with no finite inverse, not a substitute
                // draw or a reason to rewind the invocation's RNG stream.
                let u = draws[column];
                if a == b {
                    Some(a)
                } else if u == 0.0 {
                    None
                } else {
                    let laplace = if u < 0.5 {
                        (2.0 * u).ln()
                    } else {
                        -(2.0 * (1.0 - u)).ln()
                    };
                    let base = if draws[n] < 0.5 { a } else { b };
                    let difference = a - b;
                    let displacement = if difference.is_finite() {
                        laplace * difference.abs()
                    } else {
                        // Opposite-sign finite parents can have an infinite
                        // difference while this scaled displacement is finite.
                        let sign = if a >= b { 1.0 } else { -1.0 };
                        (laplace * sign * a) - (laplace * sign * b)
                    };
                    let value = base + displacement;
                    value.is_finite().then_some(value)
                }
            }
        };
        let Some(value) = value else {
            return Ok(None);
        };
        values.push(value);
        cooperation.completed(1).await?;
    }
    Ok(Some(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crossover_is_actual_gene_reproduction_and_finite_convex_mixing() {
        futures::executor::block_on(async {
            let mut cooperation =
                ScalarCooperation::for_builtin(super::super::super::IDENTITY, None);
            let first = [1.0, 2.0, 3.0, 4.0];
            let second = [10.0, 20.0, 30.0, 40.0];
            let result = child(
                Parameters {
                    kind: OptimGeneticCrossover::Scattered,
                    ratio: &[],
                    first_is_better: true,
                },
                &first,
                &second,
                &[0.1, 0.9, 0.1, 0.9],
                &mut cooperation,
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(result, [1.0, 20.0, 3.0, 40.0]);
            assert_eq!(interpolate(-1e308, 1e308, 0.5), Some(0.0));
            let result = child(
                Parameters {
                    kind: OptimGeneticCrossover::Arithmetic,
                    ratio: &[],
                    first_is_better: true,
                },
                &first,
                &second,
                &[0.5],
                &mut cooperation,
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(result, [5.5, 11.0, 16.5, 22.0]);
        });
    }
}
