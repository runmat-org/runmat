//! Nonnegative affine scaling with a target best/mean expectation ratio.
//! Equal best values remain equal; their attainable ratio bounds the target.
use super::super::{invalid, ScalarCooperation, POLICY};
use super::ranking::RankOrder;
use crate::BuiltinResult;

pub(in super::super) async fn expectation(
    raw: &[f64],
    order: &RankOrder,
    rate: f64,
    parents: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    if raw.len() != order.len() || !rate.is_finite() || rate < 1. || raw.is_empty() && parents != 0
    {
        return Err(invalid(
            "shift-linear scaling requires rate >= 1 and a matching nonempty rank order",
        ));
    }
    let mut scale = 0.0_f64;
    for &value in raw {
        if !value.is_finite() {
            return Err(invalid("shift-linear scaling requires finite real scores"));
        }
        scale = scale.max(value.abs());
        cooperation.completed(1).await?;
    }
    let mut values = POLICY.capacity(raw.len(), cooperation.runtime())?;
    if raw.is_empty() {
        return Ok(values);
    }
    let best = raw[order[0]];
    let scale = if scale == 0. { 1. } else { scale };
    for &value in raw {
        values.push(value / scale - best / scale);
        cooperation.completed(1).await?;
    }
    let mut previous = best;
    let mut ties = 0;
    for &member in order.as_slice() {
        if raw[member] < previous {
            return Err(invalid(
                "shift-linear rank order differs from comparison scores",
            ));
        }
        previous = raw[member];
        if values[member] == 0. {
            ties += 1;
        }
        cooperation.completed(1).await?;
    }
    let target = rate.min(raw.len() as f64 / ties as f64);
    let mut boundary = None;
    if target > 1. && target < raw.len() as f64 / ties as f64 {
        let mut sum = 0.;
        for (offset, &member) in order.iter().enumerate() {
            let active = offset + 1;
            sum += values[member];
            let denominator = target * active as f64 - raw.len() as f64;
            if denominator > 0. {
                let distance = target * sum / denominator;
                if distance > values[member]
                    && (active == raw.len() || distance <= values[order[active]])
                {
                    boundary = Some(distance);
                    break;
                }
            }
            cooperation.completed(1).await?;
        }
        if boundary.is_none() {
            return Err(invalid(
                "shift-linear expectation ratio cannot be represented",
            ));
        }
    }
    let mut sum = 0.;
    for value in &mut values {
        *value = if target == 1. {
            1.
        } else if let Some(boundary) = boundary {
            (boundary - *value).max(0.)
        } else {
            f64::from(u8::from(*value == 0.))
        };
        sum += *value;
        cooperation.completed(1).await?;
    }
    if !sum.is_finite() || sum <= 0. {
        return Err(invalid(
            "shift-linear expectations have no finite positive sum",
        ));
    }
    for value in &mut values {
        *value = (*value / sum) * parents as f64;
        cooperation.completed(1).await?;
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::super::ranking::{order, Score};
    use super::*;
    #[test]
    fn target_ratio_and_equal_score_degeneracy_preserve_nonnegative_parent_counts() {
        futures::executor::block_on(async {
            let runtime = crate::user_functions::test_support::exact_fixture_runtime();
            let mut cooperation =
                ScalarCooperation::for_builtin(super::super::super::IDENTITY, Some(&runtime));
            let ranked = order(
                &[Score::Defined(0.), Score::Defined(1.), Score::Defined(2.)],
                &mut cooperation,
            )
            .await
            .unwrap();
            let values = expectation(&[0., 1., 2.], &ranked, 2., 3, &mut cooperation)
                .await
                .unwrap();
            assert_eq!(values, [2., 1., 0.]);
            let tied = order(
                &[Score::Defined(-f64::MAX), Score::Defined(-f64::MAX)],
                &mut cooperation,
            )
            .await
            .unwrap();
            assert_eq!(
                expectation(&[-f64::MAX, -f64::MAX], &tied, 2., 4, &mut cooperation)
                    .await
                    .unwrap(),
                [2., 2.]
            );
        });
    }
}
