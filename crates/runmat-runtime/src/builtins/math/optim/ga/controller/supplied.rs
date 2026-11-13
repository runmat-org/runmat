//! Reuse preserves the source of each score and never turns a penalty
//! sentinel or a nearby genome into an objective measurement.
use super::super::{
    common::population::Measurement,
    evaluation::fitness::{Origin, RawFitness, Supplied},
    inputs::raw,
    invalid,
    options::{Field, Options},
    ScalarCooperation, POLICY,
};
use super::frame::Flags;
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) async fn initial(
    options: &Options,
    nonlinear: bool,
    invalidated: &[usize],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<Option<Supplied>>> {
    let members = options.layout.total();
    let mut supplied = POLICY.capacity(members, cooperation.runtime())?;
    for _ in 0..members {
        supplied.push(None);
        cooperation.completed(1).await?;
    }
    if nonlinear {
        return Ok(supplied);
    }
    let Some(value) = options
        .field(Field::InitialScoresMatrix)
        .filter(|value| !raw::unspecified(value))
    else {
        return Ok(supplied);
    };
    let count = match value {
        Value::Num(_) | Value::Int(_) => 1,
        Value::Tensor(value) => value.len(),
        _ => {
            return Err(invalid(
                "InitialScoresMatrix must be an admitted real vector",
            ))
        }
    };
    if count > members {
        return Err(invalid("InitialScoresMatrix exceeds PopulationSize"));
    }
    let fitness = super::score_value::decode(
        value.clone(),
        super::score_value::Role::Initial,
        count,
        options.check_values,
        cooperation,
    )
    .await?;
    for (row, value) in supplied.iter_mut().enumerate().take(count) {
        *value = Some(Supplied {
            value: fitness.measurement(row),
            origin: Origin::SuppliedInitial,
        });
        cooperation.completed(1).await?;
    }
    for &row in invalidated {
        let value = supplied
            .get_mut(row)
            .ok_or_else(|| invalid("changed initial score row is outside the population"))?;
        *value = None;
        cooperation.completed(1).await?;
    }
    Ok(supplied)
}
pub(super) async fn copied(
    copies: &[Option<usize>],
    old: &RawFitness,
    options: &Options,
    flags: &Flags,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<Option<Supplied>>> {
    if copies.len() != options.layout.total() {
        return Err(invalid("copied score geometry changed"));
    }
    let mut supplied = POLICY.capacity(copies.len(), cooperation.runtime())?;
    for (row, &source) in copies.iter().enumerate() {
        let group = options
            .layout
            .group_for(row)
            .ok_or_else(|| invalid("copied score row is outside the population"))?;
        let elite = row - options.layout.range(group).start < options.elites[group];
        let value = if !flags.eval_elites && (elite || flags.duplicates) {
            source
                .map(|source| {
                    if source >= old.values.len() {
                        return Err(invalid("copied score source is outside its generation"));
                    }
                    let value = old.supplied(source);
                    Ok((value.origin != Origin::ConstraintRejected).then_some(value))
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
        supplied.push(value);
        cooperation.completed(1).await?;
    }
    Ok(supplied)
}
pub(super) async fn repeated_elites_match(
    copies: &[Option<usize>],
    old: &RawFitness,
    new: &RawFitness,
    options: &Options,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if copies.len() != new.values.len() {
        return Err(invalid("elite score geometry changed"));
    }
    let mut observed = false;
    for group in 0..options.layout.groups() {
        for row in options.layout.range(group).take(options.elites[group]) {
            let source =
                copies[row].ok_or_else(|| invalid("an elite lost its original point identity"))?;
            if source >= old.values.len() {
                return Err(invalid("elite source is outside the prior generation"));
            }
            let old = old.supplied(source);
            let new = new.supplied(row);
            let equal = old.origin == Origin::Measured
                && new.origin == Origin::Measured
                && matches!((old.value, new.value), (Measurement::Real(a), Measurement::Real(b)) if a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits());
            if !equal {
                return Ok(false);
            }
            observed = true;
            cooperation.completed(1).await?;
        }
    }
    Ok(observed)
}
