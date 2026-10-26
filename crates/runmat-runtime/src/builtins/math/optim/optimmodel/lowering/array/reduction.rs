//! Sum reuses the numerical owner's grammar and original row projection.
use super::super::terms::Term;
use super::*;
pub(super) async fn sum(
    input: Array,
    arguments: &[Value],
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    let mut geometry = crate::builtins::math::reduction::sum::reduction_geometry(
        &input.shape,
        crate::call::FunctionArgumentView::from_slice(arguments),
        cooperation.runtime(),
    )
    .await?;
    if !crate::builtins::common::shape::array_shapes_equal(&shape, geometry.output_shape()) {
        return Err(policy.invalid("compiled sum geometry differs from its original node"));
    }
    if geometry.is_identity() {
        return Ok(Ok(Array { shape, ..input }));
    }
    let count = input.count(policy)?;
    let out_count = evaluation::element_count(&shape, policy)?;
    let offset = match &input.offset {
        Offset::Uniform(value) if *value == 0. => Offset::Uniform(0.),
        Offset::Uniform(value) => {
            let value = *value * (geometry.factor() as f64);
            if !value.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            Offset::Uniform(value)
        }
        _ => {
            let mut args = runmat_value::exact_capacity_vec(
                arguments
                    .len()
                    .checked_add(1)
                    .ok_or_else(|| policy.invalid("sum argument count overflow"))?,
            )
            .map_err(|e| policy.storage(e))?;
            args.push(input.offset.value());
            for value in arguments {
                args.push(value.clone());
                cooperation.completed(1).await?;
            }
            let result = evaluation::invoke("sum", &args, policy, cooperation).await?;
            if !construct::finite_numeric(result.value(), policy, cooperation).await? {
                return Ok(Err(Stop::NonFinite));
            }
            match Offset::from_value(result.value(), policy)? {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            }
        }
    };
    let work = u64::try_from(geometry.rank().max(1))
        .map_err(|_| policy.invalid("reduction rank exceeds work geometry"))?;
    let assigned = if count == 0 || matches!(input.assigned, Assigned::All) {
        Assigned::All
    } else if matches!(input.assigned, Assigned::Missing) {
        Assigned::Missing
    } else {
        let mut mask =
            runmat_value::exact_capacity_vec(out_count).map_err(|e| policy.storage(e))?;
        for _ in 0..out_count {
            mask.push(true);
            cooperation.completed(1).await?;
        }
        let mut missing = false;
        for row in 0..count {
            let output = geometry
                .project(row)
                .ok_or_else(|| policy.invalid("sum source row exceeds original shape"))?;
            let valid = input.assigned.at(row, policy)?;
            mask[output] &= valid;
            missing |= !valid;
            cooperation.completed(work).await?;
        }
        if missing {
            Assigned::Partial(std::rc::Rc::new(mask))
        } else {
            Assigned::All
        }
    };
    let coefficients = if input.coefficients.is_zero() {
        Coefficients::Zero
    } else {
        let mut entries = runmat_value::exact_capacity_vec(input.coefficients.nonzeros())
            .map_err(|e| policy.storage(e))?;
        for term in input.coefficients.entries() {
            let row = geometry
                .project(term.row)
                .ok_or_else(|| policy.invalid("sum coefficient exceeds original source shape"))?;
            entries.push(Term { row, ..term });
            cooperation.completed(work).await?;
        }
        match Coefficients::from_entries(entries, out_count, columns, policy, cooperation).await? {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        }
    };
    let mut dependent = std::collections::HashSet::new();
    for source in input.degree.rows(count) {
        if input.degree.at(source) {
            let row = geometry
                .project(source)
                .ok_or_else(|| policy.invalid("sum dependence exceeds original geometry"))?;
            dependent.try_reserve(1).map_err(|e| policy.reserve(e))?;
            dependent.insert(row);
        }
        cooperation.completed(1).await?;
    }
    let degree = Degree::collect(
        out_count,
        |row| Ok(dependent.contains(&row)),
        policy,
        cooperation,
    )
    .await?;
    Ok(Ok(Array {
        shape,
        offset,
        coefficients,
        assigned,
        degree,
    }))
}
