//! Real transpose remaps only active coefficients. Uniform offsets remain
//! uniform and actual sparse offsets use the existing numeric transpose owner.
use super::super::terms::Term;
use super::*;
pub(super) async fn transpose(
    input: Array,
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    if input.shape.iter().skip(2).any(|&n| n != 1) {
        return Err(policy.invalid("transpose requires a two-dimensional original expression"));
    }
    let m = input.shape.first().copied().unwrap_or(1);
    let n = input.shape.get(1).copied().unwrap_or(1);
    let count = input.count(policy)?;
    if count != evaluation::element_count(&shape, policy)?
        || shape.first().copied().unwrap_or(1) != n
        || shape.get(1).copied().unwrap_or(1) != m
    {
        return Err(policy.invalid("compiled transpose differs from original expression geometry"));
    }
    let offset = match &input.offset {
        Offset::Uniform(value) => Offset::Uniform(*value),
        _ => {
            let result =
                evaluation::invoke("transpose", &[input.offset.value()], policy, cooperation)
                    .await?;
            match Offset::from_value(result.value(), policy)? {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            }
        }
    };
    let assigned = match &input.assigned {
        Assigned::All => Assigned::All,
        Assigned::Missing => Assigned::Missing,
        Assigned::Partial(_) => {
            let mut values =
                runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
            for target in 0..count {
                let source = target / n + (target % n) * m;
                values.push(input.assigned.at(source, policy)?);
                cooperation.completed(1).await?;
            }
            Assigned::Partial(std::rc::Rc::new(values))
        }
    };
    let coefficients = if input.coefficients.is_zero() {
        Coefficients::Zero
    } else {
        let mut entries = runmat_value::exact_capacity_vec(input.coefficients.nonzeros())
            .map_err(|e| policy.storage(e))?;
        for term in input.coefficients.entries() {
            let row = term.row / m + (term.row % m) * n;
            entries.push(Term { row, ..term });
            cooperation.completed(1).await?;
        }
        match Coefficients::from_entries(entries, count, columns, policy, cooperation).await? {
            Ok(values) => values,
            Err(stop) => return Ok(Err(stop)),
        }
    };
    let degree = Degree::collect(
        count,
        |row| Ok(input.degree.at(row / n + (row % n) * m)),
        policy,
        cooperation,
    )
    .await?;
    Ok(Ok(Array {
        shape,
        offset,
        assigned,
        coefficients,
        degree,
    }))
}
