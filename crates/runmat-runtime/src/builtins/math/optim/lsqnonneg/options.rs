//! Selected controls only, admitted as a whole before any provider transfer.
use super::*;
use runmat_builtins::{OptimDisplay, OptimOptionField, OptimParameterLookup, OptimParameterName};
pub(super) struct Controls {
    pub tolerance: Option<f64>,
    pub threshold: f64,
    pub limit: usize,
    pub iterations: usize,
    supplied_limit: Option<usize>,
    display: OptimDisplay,
}
impl Controls {
    pub async fn parse(
        value: Value,
        runtime: &crate::context::RuntimeContext,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let source = crate::execution::RootedValue::in_optional_context(value, Some(runtime))?;
        let mut selected = [None; 3];
        let fields = match source.value() {
            Value::Tensor(t) if t.is_empty() => None,
            Value::Struct(s) => Some(&s.fields),
            _ => return Err(invalid("options must be a scalar struct or []")),
        };
        if let Some(fields) = fields {
            for (name, value) in fields {
                let parameter = match OptimParameterName::resolve(name) {
                    OptimParameterLookup::Matched(name) => name,
                    OptimParameterLookup::Ambiguous => {
                        return Err(invalid(format!("ambiguous option {name}")))
                    }
                    OptimParameterLookup::Unknown => {
                        return Err(invalid(format!("unknown option {name}")))
                    }
                };
                // Standard optimset structures contain unused known [] fields.
                if matches!(value,Value::Tensor(t) if t.is_empty()) {
                    continue;
                }
                let field = LsqnonnegOption::from_parameter(parameter)
                    .ok_or_else(|| invalid(format!("unsupported nonempty option {name}")))?;
                let slot = match field {
                    LsqnonnegOption::Tolerance => 0,
                    LsqnonnegOption::MaxIterations => 1,
                    LsqnonnegOption::Display => 2,
                };
                selected[slot].get_or_insert(value);
            }
        }
        // Only numerical controls may gather; display remains text-owned.
        for (slot, field) in [(0, OptimOptionField::TolX), (1, OptimOptionField::MaxIter)] {
            if let Some(value) = selected[slot] {
                if !matches!(
                    value,
                    Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
                ) {
                    return Err(invalid("numeric controls must be real numeric scalars"));
                }
                common::preflight_option_scalar(NAME, value, field)?;
                common::preflight_host_residency(NAME, value, None, Some(runtime))?;
                if crate::builtins::common::validation::value_has_native_integer_class(value) {
                    crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                        Some(runtime),
                        &runmat_builtins::LSQNONNEG_INTEGER_OPTION_EXTENSION,
                        NAME,
                    )?;
                }
            }
        }
        let display = match selected[2] {
            None => OptimDisplay::Notify,
            Some(value) => crate::builtins::common::validation::value_text_scalar_string(value)
                .and_then(|text| OptimDisplay::parse(&text))
                .filter(|value| {
                    matches!(
                        value,
                        OptimDisplay::Off
                            | OptimDisplay::None
                            | OptimDisplay::Final
                            | OptimDisplay::Notify
                    )
                })
                .ok_or_else(|| invalid("Display must be notify, off, none or final"))?,
        };
        let tolerance = if let Some(value) = selected[0] {
            let value =
                crate::dispatcher::gather_if_needed_in_context_async(value, Some(runtime)).await?;
            let value = common::option_f64_value(NAME, Some(&value), "TolX", 0.0)?;
            if value <= 0.0 || !value.is_finite() {
                return Err(invalid("TolX must be positive and finite"));
            }
            Some(value)
        } else {
            None
        };
        let supplied_limit = if let Some(value) = selected[1] {
            let value =
                crate::dispatcher::gather_if_needed_in_context_async(value, Some(runtime)).await?;
            Some(common::option_usize_value(
                NAME,
                Some(&value),
                "MaxIter",
                0,
            )?)
        } else {
            None
        };
        cooperation.completed(0).await?;
        Ok(Self {
            tolerance,
            threshold: 0.0,
            limit: 0,
            iterations: 0,
            supplied_limit,
            display,
        })
    }
    pub fn initialize(&mut self, model: &model::Model) -> BuiltinResult<()> {
        self.limit = match self.supplied_limit {
            Some(limit) => limit,
            None => model
                .c
                .cols
                .checked_mul(3)
                .ok_or_else(|| invalid("default iteration limit overflows addressable size"))?,
        };
        self.threshold = match self.tolerance {
            Some(tolerance) if model.scale > 0.0 => tolerance / model.scale,
            Some(_) => f64::INFINITY,
            None => model.default_threshold,
        };
        Ok(())
    }
    pub fn consume(&mut self) -> bool {
        if self.iterations >= self.limit {
            return false;
        }
        self.iterations += 1;
        true
    }
    pub fn summary(
        &self,
        flag: i32,
        message: &str,
        runtime: &crate::context::RuntimeContext,
    ) -> BuiltinResult<()> {
        if self.display == OptimDisplay::Final
            || (self.display == OptimDisplay::Notify && flag != 1)
        {
            let _scope = runtime.enter();
            crate::console::record_console_line(
                crate::console::ConsoleStream::Stdout,
                format!("lsqnonneg: {message} Iterations: {}.", self.iterations),
            )?;
        }
        Ok(())
    }
}
