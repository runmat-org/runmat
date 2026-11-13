//! Products have at most one structurally dependent factor per output.
//! A zero/cancelled proposal coefficient never discharges that requirement.
use super::super::factors::Factors;
use super::*;
#[derive(Clone, Copy)]
struct Entry {
    offset: f64,
    constant: f64,
    source: Option<usize>,
}
impl Entry {
    fn one() -> Self {
        Self {
            offset: 1.,
            constant: 1.,
            source: None,
        }
    }
    fn consume(&mut self, input: &Array, row: usize, policy: Policy) -> Lowered<()> {
        if !input.assigned.at(row, policy)? {
            return Ok(Err(Stop::Unassigned));
        }
        let value = input.offset.at(row, policy)?;
        if input.degree.at(row) {
            if self.source.is_some() {
                return Ok(Err(Stop::NonAffine));
            }
            self.source = Some(row);
        } else {
            self.constant *= value;
        }
        self.offset *= value;
        Ok(if self.offset.is_finite() && self.constant.is_finite() {
            Ok(())
        } else {
            Err(Stop::NonFinite)
        })
    }
}
pub(super) async fn lower(
    input: Array,
    operation: Operation,
    arguments: &[Value],
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    let mut plan = Factors::prepare(operation, &input.shape, arguments, &shape, policy, c).await?;
    let count = evaluation::element_count(&shape, policy)?;
    let mut products = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    for _ in 0..count {
        products.push(Entry::one());
        c.completed(1).await?;
    }
    match &mut plan {
        Factors::Reduction(reduction) => {
            for row in 0..input.count(policy)? {
                let target = reduction
                    .project(row)
                    .ok_or_else(|| policy.invalid("factor source exceeds reduction"))?;
                if let Err(stop) = products[target].consume(&input, row, policy)? {
                    return Ok(Err(stop));
                }
                c.completed(1).await?;
            }
        }
        Factors::Scan(scan) => {
            let mut running = Entry::one();
            for step in scan.steps(false) {
                if step.first {
                    running = Entry::one();
                }
                if let Err(stop) = running.consume(&input, step.index, policy)? {
                    return Ok(Err(stop));
                }
                products[step.index] = running;
                c.completed(1).await?;
            }
        }
    }
    let mut entries = Vec::new();
    for term in input.coefficients.entries() {
        for row in plan.outputs(term.row, policy)? {
            if products[row].source != Some(term.row) {
                return Err(policy.invalid("coefficient lost original factor-row dependence"));
            }
            let value = term.value * products[row].constant;
            if !value.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            if value != 0. {
                entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                entries.push(Term {
                    row,
                    column: term.column,
                    value,
                });
            }
            c.completed(1).await?;
        }
    }
    let coefficients = match Coefficients::from_entries(entries, count, columns, policy, c).await? {
        Ok(values) => values,
        Err(stop) => return Ok(Err(stop)),
    };
    let degree =
        Degree::collect(count, |row| Ok(products[row].source.is_some()), policy, c).await?;
    let mut uniform = true;
    let first = products.first().map_or(1., |entry| entry.offset);
    for entry in &products {
        uniform &= entry.offset == first;
        c.completed(1).await?;
    }
    let offset = if uniform {
        Offset::Uniform(first)
    } else {
        let mut values = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        for entry in products {
            values.push(entry.offset);
            c.completed(1).await?;
        }
        Offset::Dense(
            runmat_value::Tensor::from_numeric_storage(
                runmat_value::NumericStorage::F64(values),
                shape.clone(),
            )
            .map_err(|e| policy.invalid(e))?,
        )
    };
    Ok(Ok(Array {
        shape,
        offset,
        coefficients,
        degree,
        assigned: Assigned::All,
    }))
}
