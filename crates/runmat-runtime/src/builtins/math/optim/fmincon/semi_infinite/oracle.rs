//! Original-context seminfcon tuple ingress. Only real callback invocations
//! acquire Budget points; accepted objective data is not replayed on refresh.
use super::super::{
    evaluation::{self, Evaluation, EvaluationStop, Evaluator, Point},
    numeric::{self, Geometry, Numeric, Policy},
};
use super::{
    bernstein::invalid,
    generation::FiniteExtents,
    mesh::{Samples, Spacing},
    Payload,
};
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{NumericDType, Value};
use std::sync::Arc;
pub(in crate::builtins::math::optim::fmincon) fn admit_double(value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::Num(_) | Value::Complex(_, _))
        || matches!(value,Value::Tensor(t) if t.numeric_dtype()==NumericDType::F64)
        || matches!(value,Value::ComplexTensor(t) if t.numeric_dtype()==NumericDType::F64)
    {
        Ok(())
    } else {
        Err(invalid(
            "documented fseminf callback outputs must be full host double arrays",
        ))
    }
}
pub(super) struct Raw {
    pub inequality: Vec<f64>,
    pub equality: Vec<f64>,
    pub samples: Vec<Arc<Samples>>,
}
pub(super) struct Request<'a> {
    pub x: &'a [f64],
    pub shape: &'a [usize],
    pub scalar: bool,
    pub callback: &'a Value,
    pub intervals: Vec<f64>,
    pub families: usize,
    pub policy: &'a Policy,
    pub cooperation: &'a mut ScalarCooperation,
    pub budget: &'a mut evaluation::Budget,
    pub reserve_point: bool,
    pub check_values: bool,
}
pub(super) async fn call(
    request: Request<'_>,
) -> BuiltinResult<std::result::Result<Raw, EvaluationStop>> {
    let Request {
        x,
        shape,
        scalar,
        callback,
        intervals,
        families,
        policy,
        cooperation,
        budget,
        reserve_point,
        check_values,
        ..
    } = request;
    if reserve_point && !budget.begin_point() {
        return Ok(Err(EvaluationStop::BudgetExhausted));
    }
    cooperation.completed(0).await?;
    let argument = common::output_callback::vector(x, shape, scalar, cooperation).await?;
    let spacing =
        Value::Tensor(runmat_value::Tensor::new(intervals, vec![families, 2]).map_err(invalid)?);
    let width = families
        .checked_add(3)
        .ok_or_else(|| invalid("semi-infinite callback output count overflow"))?;
    let returned = common::call_function_sequence_in_context(
        cooperation.runtime(),
        callback,
        [argument, spacing],
        width,
    )
    .await?;
    budget.completed_constraints();
    let returned = evaluation::select_sequence(returned, width, cooperation.runtime())?;
    for value in returned.as_slice().iter() {
        admit_double(value)?;
    }
    let mut outputs = {
        let _entry = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned.try_into_values()?
    };
    let mut arrays = runmat_value::exact_capacity_vec(
        families
            .checked_add(2)
            .ok_or_else(|| invalid("callback tuple extent overflow"))?,
    )
    .map_err(|source| {
        crate::build_runtime_error("fseminf: callback tuple allocation failed")
            .with_source(source)
            .build()
    })?;
    for index in 0..families + 2 {
        let value = outputs.next().expect("selected seminfcon output tuple");
        let geometry = if index < 2 {
            Geometry::Array
        } else {
            Geometry::MatrixAny
        };
        let Numeric::Real { values, shape } = numeric::read(
            value,
            "semi-infinite callback output",
            geometry,
            policy,
            cooperation,
        )
        .await?
        else {
            if check_values {
                return Err(invalid(
                    "FunValCheck rejected an undefined semi-infinite callback output",
                ));
            }
            return Ok(Err(EvaluationStop::Undefined));
        };
        arrays.push((values, shape));
    }
    let spacing = outputs.next().expect("selected sampling interval output");
    let Value::Tensor(spacing) = spacing else {
        return Err(invalid(
            "seminfcon must return an ntheta-by-2 sampling matrix",
        ));
    };
    if spacing.shape() != [families, 2] {
        return Err(invalid(
            "returned sampling interval dimensions must be ntheta-by-2",
        ));
    }
    let mut samples = runmat_value::exact_capacity_vec(families).map_err(|source| {
        crate::build_runtime_error("fseminf: sample family allocation failed")
            .with_source(source)
            .build()
    })?;
    let mut arrays = arrays.into_iter();
    let inequality = arrays.next().unwrap().0;
    let equality = arrays.next().unwrap().0;
    for (family, (values, shape)) in arrays.enumerate() {
        let column = crate::builtins::common::tensor::tensor_value_f64(&spacing, family);
        let second = crate::builtins::common::tensor::tensor_value_f64(&spacing, families + family);
        let row = if second == 0.0 { None } else { Some(second) };
        let rows = shape.first().copied().unwrap_or(1);
        let columns = shape.get(1).copied().unwrap_or(1);
        samples.push(Arc::new(Samples::new(
            values,
            rows,
            columns,
            Spacing { column, row },
        )?));
        cooperation.completed(1).await?;
    }
    Ok(Ok(Raw {
        inequality,
        equality,
        samples,
    }))
}
pub(in crate::builtins::math::optim::fmincon) async fn point(
    evaluator: &mut Evaluator<'_>,
    x: &[f64],
    mut point: Point,
) -> BuiltinResult<Evaluation> {
    let mut domain = evaluator
        .semi_infinite
        .take()
        .expect("selected semi-infinite domain");
    let result = async {
        let intervals = domain.state.probe_intervals()?;
        let raw = call(Request {
            x,
            shape: &evaluator.shape,
            scalar: evaluator.scalar,
            callback: evaluator
                .constraints
                .ok_or_else(|| invalid("missing original seminfcon callback"))?,
            intervals,
            families: domain.state.family_count(),
            policy: &evaluator.policy,
            cooperation: evaluator.cooperation,
            budget: &mut evaluator.budget,
            reserve_point: false,
            check_values: domain.controls.check_values,
        })
        .await?;
        let raw = match raw {
            Ok(raw) => raw,
            Err(stop) => return Ok(Err(stop)),
        };
        let finite = FiniteExtents {
            inequality: raw.inequality.len(),
            equality: raw.equality.len(),
        };
        let rows = if domain.state.has_current() {
            let rows = domain.state.frozen()?;
            if rows.generation().finite != finite {
                return Err(invalid("finite constraint layout changed in seminfcon"));
            }
            rows
        } else {
            let proposal = domain
                .state
                .propose(
                    raw.samples.clone(),
                    finite,
                    &[],
                    domain.controls.mesh_tolerance,
                    evaluator.cooperation,
                )
                .await?;
            domain.state.commit(proposal)?.0
        };
        point.inequality = rows
            .evaluate(&raw.samples, &raw.inequality, evaluator.cooperation)
            .await?;
        point.equality = raw.equality;
        point.inequality_jacobian = None;
        point.equality_jacobian = None;
        point.semi_infinite = Some(Payload {
            rows,
            samples: raw.samples,
            finite,
        });
        Ok(Ok(point))
    }
    .await;
    evaluator.semi_infinite = Some(domain);
    result
}

pub(in crate::builtins::math::optim::fmincon) async fn probe(
    evaluator: &mut Evaluator<'_>,
    x: &[f64],
) -> BuiltinResult<std::result::Result<evaluation::Probe, EvaluationStop>> {
    let domain = evaluator
        .semi_infinite
        .take()
        .expect("selected semi-infinite probe domain");
    let result = async {
        let rows = domain.state.frozen()?;
        let raw = call(Request {
            x,
            shape: &evaluator.shape,
            scalar: evaluator.scalar,
            callback: evaluator
                .constraints
                .ok_or_else(|| invalid("missing original seminfcon callback"))?,
            intervals: domain.state.probe_intervals()?,
            families: domain.state.family_count(),
            policy: &evaluator.policy,
            cooperation: evaluator.cooperation,
            budget: &mut evaluator.budget,
            reserve_point: true,
            check_values: domain.controls.check_values,
        })
        .await?;
        let raw = match raw {
            Ok(raw) => raw,
            Err(stop) => return Ok(Err(stop)),
        };
        if rows.generation().finite
            != (FiniteExtents {
                inequality: raw.inequality.len(),
                equality: raw.equality.len(),
            })
        {
            return Err(invalid(
                "finite constraints changed extent in a constraint-only probe",
            ));
        }
        let inequality = rows
            .evaluate(&raw.samples, &raw.inequality, evaluator.cooperation)
            .await?;
        Ok(Ok(evaluation::Probe::ConstraintsOnly {
            inequality,
            equality: raw.equality,
        }))
    }
    .await;
    evaluator.semi_infinite = Some(domain);
    result
}

#[cfg(test)]
#[path = "oracle/tests.rs"]
mod tests;
