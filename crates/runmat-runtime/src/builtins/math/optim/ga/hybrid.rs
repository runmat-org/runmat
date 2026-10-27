//! Selected local solver invocation, with no objective replay or public-result
//! success alias. The GA owner still checks its original domain before use.
use super::{common, inputs, invalid, ScalarCooperation, POLICY};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::OptimHybridSolver;
use runmat_value::{NumericDType, Value};

pub(super) fn validate(owner: super::options::Owner, value: &Value) -> BuiltinResult<()> {
    common::global_hybrid::parse(value, |detail: &str| owner.invalid(detail)).map(|_| ())
}
pub(super) struct Selected {
    kind: OptimHybridSolver,
    target: crate::call::descriptor::RepeatableCallable,
    controls: Option<RootedValue>,
}
pub(super) struct Candidate {
    pub point: RootedValue,
    pub fitness: f64,
    pub flag: f64,
    pub evaluations: usize,
}
pub(super) fn select(
    value: Option<&Value>,
    problem: &inputs::Problem,
    population: runmat_builtins::OptimGeneticPopulation,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<Option<Selected>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(selected) = common::global_hybrid::parse(value, |detail: &str| invalid(detail))?
    else {
        return Ok(None);
    };
    if problem.mixed_integer || population != runmat_builtins::OptimGeneticPopulation::DoubleVector
    {
        return Err(invalid(
            "HybridFcn requires a continuous doubleVector problem",
        ));
    }
    if !selected.kind.accepts_bounds()
        && (problem.has_bounds || problem.has_linear || problem.nonlinear.is_some())
    {
        return Err(invalid(
            "a constrained problem requires fmincon or patternsearch as HybridFcn",
        ));
    }
    let target = super::callbacks::select(&selected.target, cooperation)?;
    Ok(Some(Selected {
        kind: selected.kind,
        target,
        controls: selected
            .options
            .map(|value| RootedValue::in_optional_context(value, cooperation.runtime()))
            .transpose()?,
    }))
}
impl Selected {
    pub async fn run_with<I: super::callbacks::Invocation>(
        &self,
        point: &RootedValue,
        problem: &inputs::Problem,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<std::ops::ControlFlow<I::Stop, Candidate>> {
        let mut arguments = POLICY.capacity(
            if self.kind.accepts_bounds() { 10 } else { 3 },
            cooperation.runtime(),
        )?;
        arguments.push(problem.objective.clone());
        arguments.push(point.value().clone());
        if self.kind.accepts_bounds() {
            let Value::Struct(source) = &problem.public else {
                return Err(invalid("original hybrid problem view is unavailable"));
            };
            for field in ["Aineq", "Bineq", "Aeq", "Beq", "lb", "ub", "nonlcon"] {
                arguments.push(
                    source
                        .fields
                        .get(field)
                        .ok_or_else(|| invalid("original hybrid constraint role is unavailable"))?
                        .clone(),
                );
                cooperation.completed(1).await?;
            }
        }
        arguments.push(
            self.controls
                .as_ref()
                .map(|value| value.value().clone())
                .unwrap_or_else(inputs::raw::empty),
        );
        let prepared =
            crate::call::BuiltinArguments::from_borrowed(&arguments, cooperation.runtime())?;
        let returned = match invocation
            .call(
                &self.target,
                prepared,
                4,
                super::callbacks::Event::Work,
                cooperation,
            )
            .await?
        {
            std::ops::ControlFlow::Continue(returned) => returned,
            std::ops::ControlFlow::Break(stop) => return Ok(std::ops::ControlFlow::Break(stop)),
        };
        if returned.len() != 4 {
            return Err(invalid("hybrid solver must return x,fval,exitflag,output"));
        }
        let mut returned = {
            let _scope = cooperation
                .runtime()
                .map(crate::context::RuntimeContext::enter);
            returned.try_into_values()?
        };
        let value = returned.next().expect("four hybrid outputs");
        let value = match value {
            Value::Num(value) if problem.variables == 1 && value.is_finite() => {
                Value::Tensor(runmat_value::Tensor::new(vec![value], vec![1, 1]).map_err(invalid)?)
            }
            Value::Tensor(value)
                if value.numeric_dtype() == NumericDType::F64
                    && value.len() == problem.variables =>
            {
                for &coordinate in value.as_f64_slice().expect("double hybrid point") {
                    if !coordinate.is_finite() {
                        return Err(invalid("hybrid x must be finite"));
                    }
                    cooperation.completed(1).await?;
                }
                Value::Tensor(value.reshape(vec![1, problem.variables]).map_err(invalid)?)
            }
            _ => return Err(invalid("hybrid x must contain nvars double coordinates")),
        };
        let fitness = scalar(&returned.next().expect("fval"))?;
        let flag = scalar(&returned.next().expect("flag"))?;
        let output = returned.next().expect("output");
        let Value::Struct(output) = output else {
            return Err(invalid("hybrid output must be a scalar structure"));
        };
        let count = output
            .fields
            .get("funcCount")
            .or_else(|| output.fields.get("funccount"))
            .ok_or_else(|| invalid("hybrid output requires its function evaluation count"))?;
        let evaluations = common::option_usize_value("ga", Some(count), "hybrid funcCount", 0)?;
        Ok(std::ops::ControlFlow::Continue(Candidate {
            point: RootedValue::in_optional_context(value, cooperation.runtime())?,
            fitness,
            flag,
            evaluations,
        }))
    }
}
fn scalar(value: &Value) -> BuiltinResult<f64> {
    let value = match value {
        Value::Num(value) => *value,
        Value::Tensor(value) if value.numeric_dtype() == NumericDType::F64 && value.len() == 1 => {
            value.as_f64_slice().expect("double scalar")[0]
        }
        _ => return Err(invalid("hybrid fval and exitflag must be double scalars")),
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("hybrid fval and exitflag must be finite"))
    }
}
