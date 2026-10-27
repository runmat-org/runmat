//! Bounded candidate proposals; only the caller's exact full-model check proves anything.
use super::super::ScalarCooperation;
use crate::{build_runtime_error, BuiltinResult};
use num_rational::{BigRational, Ratio};
use runmat_builtins::BuiltinCatalogIdentity;

#[derive(Clone, Copy)]
enum Recovery {
    Retained,
    Dyadic,
    I16,
    I32,
    I64,
    I128,
}

pub(crate) async fn certify<F>(
    values: &[f64],
    identity: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
    accept: F,
) -> BuiltinResult<bool>
where
    F: for<'a> std::ops::AsyncFnMut(
        &'a [BigRational],
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<bool>,
{
    Ok(checked_owned(values, identity, cooperation, None, accept)
        .await?
        .is_some())
}

/// The original bounded loop transfers an accepted candidate, without cloning.
pub(super) async fn checked_owned<F>(
    values: &[f64],
    identity: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
    initial: Option<Vec<BigRational>>,
    mut accept: F,
) -> BuiltinResult<Option<Vec<BigRational>>>
where
    F: for<'a> std::ops::AsyncFnMut(
        &'a [BigRational],
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<bool>,
{
    cooperation.completed(0).await?;
    let retained = initial
        .as_ref()
        .is_some_and(|candidate| candidate.len() == values.len());
    let mut candidate = match initial {
        Some(candidate) => candidate,
        None => runmat_value::exact_capacity_vec(values.len()).map_err(|source| {
            build_runtime_error(format!(
                "{}: exact certificate allocation failed",
                identity.name
            ))
            .with_builtin(identity.name)
            .with_source(source)
            .build()
        })?,
    };
    // One reusable candidate, never five simultaneous vectors. Each existing
    // Ratio approximator has at most 30 continued-fraction steps; small widths
    // propose simple relations despite backend residuals, larger widths extend
    // the available denominators. These are proposals, not proof tolerances.
    for recovery in [
        Recovery::Retained,
        Recovery::Dyadic,
        Recovery::I16,
        Recovery::I32,
        Recovery::I64,
        Recovery::I128,
    ]
    .into_iter()
    .skip(usize::from(!retained))
    {
        // Reuse the exact proposal's buffer for ordinary numerical recovery;
        // every proposal passes the same full-model acceptance check.
        if !matches!(recovery, Recovery::Retained) {
            candidate.clear();
            for chunk in values.chunks(1024) {
                for &value in chunk {
                    let Some(value) = recover(value, recovery) else {
                        return Ok(None);
                    };
                    candidate.push(value);
                }
                cooperation.completed(chunk.len() as u64).await?;
            }
        }
        let accepted = accept(&candidate, cooperation).await?;
        cooperation.completed(0).await?;
        if accepted {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn recover(value: f64, recovery: Recovery) -> Option<BigRational> {
    if !value.is_finite() {
        return None;
    }
    let proposed = match recovery {
        Recovery::Retained => return None,
        Recovery::Dyadic => None,
        Recovery::I16 => Ratio::<i16>::approximate_float(value)
            .map(|r| BigRational::new((*r.numer()).into(), (*r.denom()).into())),
        Recovery::I32 => Ratio::<i32>::approximate_float(value)
            .map(|r| BigRational::new((*r.numer()).into(), (*r.denom()).into())),
        Recovery::I64 => Ratio::<i64>::approximate_float(value)
            .map(|r| BigRational::new((*r.numer()).into(), (*r.denom()).into())),
        Recovery::I128 => Ratio::<i128>::approximate_float(value)
            .map(|r| BigRational::new((*r.numer()).into(), (*r.denom()).into())),
    };
    // A width's range is only a candidate-generation limitation. Keep the
    // exact original component when it cannot propose a rational replacement.
    proposed.or_else(|| BigRational::from_float(value))
}
