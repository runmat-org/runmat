//! Population layout and controlled movement of actually owned generations.
pub(super) mod layout;
pub(super) mod migration;
pub(super) mod movement;
pub(super) mod custom;
pub(super) mod generation;
pub(super) use generation::Generation;

/// Numerical variation needs an owned coordinate row, rather than the public
/// Value returned by Population::row for semantic callback arguments.
pub(super) async fn coordinates(
    population: &super::common::population::Population,
    row: usize,
    cooperation: &mut super::ScalarCooperation,
) -> crate::BuiltinResult<Vec<f64>> {
    if row >= population.members {
        return Err(super::invalid("coordinate row exceeds population extent"));
    }
    let mut values = super::POLICY.capacity(population.variables, cooperation.runtime())?;
    for column in 0..population.variables {
        values.push(population.get(row, column));
        cooperation.completed(1).await?;
    }
    Ok(values)
}
