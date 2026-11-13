use super::*;
use common::output_callback::{validate_for_builtin, CallbackForm};
use runmat_builtins::FminsearchOptionName as Field;
pub(in crate::builtins::math::optim) struct Options<'a> {
    pub tol_x: f64,
    pub tol_fun: f64,
    pub max_iter: usize,
    pub max_calls: usize,
    pub display: runmat_builtins::OptimDisplay,
    pub check_values: bool,
    pub output: Option<&'a Value>,
    pub plots: Option<&'a Value>,
}
pub(super) fn default_marker(value: &Value) -> bool {
    matches!(value,Value::Tensor(value)if value.is_empty())
}
impl<'a> Options<'a> {
    pub async fn parse(
        value: Option<&'a Value>,
        n: usize,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        let limit = n
            .checked_mul(200)
            .ok_or_else(|| invalid("default budget geometry overflow"))?;
        let mut out = Self {
            tol_x: 1e-4,
            tol_fun: 1e-4,
            max_iter: limit,
            max_calls: limit,
            display: runmat_builtins::OptimDisplay::Notify,
            check_values: false,
            output: None,
            plots: None,
        };
        let fields = match value {
            None => return Ok(out),
            Some(value) if default_marker(value) => return Ok(out),
            Some(Value::Struct(value)) => &value.fields,
            _ => return Err(invalid("options must be a scalar structure or []")),
        };
        admit_in(value, runtime)?;
        for (name, value) in fields {
            if default_marker(value) {
                continue;
            }
            let field = Field::from_public(name)
                .ok_or_else(|| invalid(format!("unsupported option {name}")))?;
            match field {
                Field::OutputFcn => out.output = Some(value),
                Field::PlotFcns => out.plots = Some(value),
                Field::Display => {
                    let text = text(value).ok_or_else(|| invalid("Display must be scalar text"))?;
                    out.display = runmat_builtins::OptimDisplay::parse(&text)
                        .filter(|display| {
                            runmat_builtins::OptimSolver::Generic.accepts_display(*display)
                        })
                        .ok_or_else(|| invalid("Display must be notify/final/iter/off/none"))?;
                }
                Field::FunValCheck => {
                    let text =
                        text(value).ok_or_else(|| invalid("FunValCheck must be on or off"))?;
                    out.check_values = match text.to_ascii_lowercase().as_str() {
                        "on" => true,
                        "off" => false,
                        _ => return Err(invalid("FunValCheck must be on or off")),
                    };
                }
                Field::MaxIter | Field::MaxFunEvals => {
                    let count = if text(value)
                        .is_some_and(|value| value == runmat_builtins::FMINSEARCH_DEFAULT_BUDGET)
                    {
                        limit
                    } else {
                        let value = scalar(value, runtime).await?;
                        let minimum = if field == Field::MaxIter { 0.0 } else { 1.0 };
                        if value < minimum
                            || value.fract() != 0.0
                            || value >= 2f64.powi(usize::BITS as i32)
                        {
                            return Err(invalid("MaxIter must be a nonnegative representable integer and MaxFunEvals a positive representable integer"));
                        }
                        value as usize
                    };
                    match field {
                        Field::MaxIter => out.max_iter = count,
                        _ => out.max_calls = count,
                    }
                }
                Field::TolX | Field::TolFun => {
                    let value = scalar(value, runtime).await?;
                    if value <= 0.0 {
                        return Err(invalid("TolX and TolFun must be positive"));
                    }
                    match field {
                        Field::TolX => out.tol_x = value,
                        _ => out.tol_fun = value,
                    }
                }
            }
        }
        Ok(out)
    }
}
fn text(value: &Value) -> Option<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
}
async fn scalar(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<f64> {
    scalar_metadata(value)?;
    let value = crate::dispatcher::gather_if_needed_in_context_async(value, runtime).await?;
    let tensor =
        crate::builtins::common::tensor::value_into_tensor_for(NAME, value).map_err(invalid)?;
    let value = crate::builtins::common::tensor::tensor_value_f64(&tensor, 0);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("numeric controls must be finite"))
    }
}
fn scalar_metadata(value: &Value) -> BuiltinResult<()> {
    if !matches!(
        value,
        Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_)
    ) {
        return Err(invalid(
            "numeric controls must be finite real floating scalars",
        ));
    }
    let fact = crate::value_fact::value_fact(value);
    if !matches!(
        fact.kind,
        runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::Single | runmat_types::NumericClass::Double,
            domain: runmat_types::NumericDomain::Real
        })
    ) || fact.shape.element_count() != Some(1)
        || fact.storage == runmat_types::StorageFact::Sparse
    {
        return Err(invalid(
            "numeric controls must be finite real floating scalars",
        ));
    }
    Ok(())
}

pub(in crate::builtins::math::optim) fn admit(value: Option<&Value>) -> BuiltinResult<()> {
    let fields = match value {
        None => return Ok(()),
        Some(value) if default_marker(value) => return Ok(()),
        Some(Value::Struct(value)) => &value.fields,
        _ => return Err(invalid("options must be a scalar structure or []")),
    };
    for (name, value) in fields {
        if default_marker(value) {
            continue;
        }
        let field = Field::from_public(name)
            .ok_or_else(|| invalid(format!("unsupported option {name}")))?;
        match field {
            Field::OutputFcn => validate_for_builtin(NAME, value, CallbackForm::Handles)?,
            Field::PlotFcns => {
                validate_for_builtin(NAME, value, CallbackForm::HandlesOrNames)?;
                #[cfg(not(feature = "plot-core"))]
                if common::output_callback::callback_count(
                    Some(value),
                    CallbackForm::HandlesOrNames,
                )? > 0
                {
                    return Err(invalid("PlotFcns layout requires plot-core"));
                }
            }
            Field::Display => {
                if text(value)
                    .and_then(|value| runmat_builtins::OptimDisplay::parse(&value))
                    .filter(|display| {
                        runmat_builtins::OptimSolver::Generic.accepts_display(*display)
                    })
                    .is_none()
                {
                    return Err(invalid("Display must be notify/final/iter/off/none"));
                }
            }
            Field::FunValCheck => {
                if !text(value).is_some_and(|value| {
                    value.eq_ignore_ascii_case("on") || value.eq_ignore_ascii_case("off")
                }) {
                    return Err(invalid("FunValCheck must be on or off"));
                }
            }
            Field::MaxIter | Field::MaxFunEvals
                if text(value)
                    .is_some_and(|value| value == runmat_builtins::FMINSEARCH_DEFAULT_BUDGET) => {}
            _ => {
                scalar_metadata(value)?;
                input::admit_numeric(value)?;
            }
        }
    }
    Ok(())
}

pub(in crate::builtins::math::optim) fn admit_in(
    value: Option<&Value>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    match runtime {
        Some(runtime) => {
            let _entry = runtime.enter();
            admit(value)
        }
        None => {
            let original = crate::context::PreparedUnselectedScope::prepare(None)?;
            let _entry = original.enter();
            admit(value)
        }
    }
}
