//! One typed admission plan for ttest2 controls. Host-visible errors are
//! settled before sample/provider materialization; resident controls retain
//! their ordered position until their owning gather completes.
use super::*;

enum Step {
    Ready(DecodedOption),
    DeferredValue(Ttest2OptionName, Value),
    DeferredName(Value, Value),
}

enum DecodedOption {
    Alpha(f64),
    Dim(usize),
    Tail(Tail),
    Variance(VarType),
}

pub(super) struct Ttest2OptionPlan {
    steps: Vec<Step>,
}

impl Ttest2OptionPlan {
    pub(super) fn admit(rest: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        if !rest.len().is_multiple_of(2) {
            let last = rest.last().expect("odd option list has a last name");
            if !crate::value_contains_gpu(last) {
                let name = scalar_text(last, "option name")?;
                return Err(invalid_argument(format!(
                    "ttest2: option '{name}' requires a value"
                )));
            }
            return Err(invalid_argument(
                "ttest2: name-value options must be supplied in pairs",
            ));
        }
        let mut steps = Vec::with_capacity(rest.len() / 2);
        let mut rest = rest.into_iter();
        while let (Some(name), Some(value)) = (rest.next(), rest.next()) {
            if crate::value_contains_gpu(&name) {
                steps.push(Step::DeferredName(name, value));
                continue;
            }
            let name = scalar_text(&name, "option name")?;
            let kind = Ttest2OptionName::from_public(&name)
                .ok_or_else(|| invalid_argument(format!("ttest2: unsupported option '{name}'")))?;
            Self::gate_integer_option(kind, &value)?;
            if crate::value_contains_gpu(&value) {
                steps.push(Step::DeferredValue(kind, value));
            } else {
                steps.push(Step::Ready(Self::decode_value(kind, &value)?));
            }
        }
        Ok(Self { steps })
    }

    fn gate_integer_option(kind: Ttest2OptionName, value: &Value) -> BuiltinResult<()> {
        if matches!(kind, Ttest2OptionName::Alpha | Ttest2OptionName::Dim)
            && ttest2_is_typed_integer(value)
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TTEST2_INTEGER_OPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        Ok(())
    }

    fn decode_value(kind: Ttest2OptionName, value: &Value) -> BuiltinResult<DecodedOption> {
        match kind {
            Ttest2OptionName::Alpha => {
                ensure_exact_ttest2_integer_scalar(value, "Alpha")?;
                let alpha = scalar_number(value)
                    .ok_or_else(|| invalid_argument("ttest2: Alpha must be numeric"))?;
                if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
                    return Err(invalid_argument(
                        "ttest2: Alpha must be a scalar in the open interval (0,1)",
                    ));
                }
                Ok(DecodedOption::Alpha(alpha))
            }
            Ttest2OptionName::Dim => {
                let dim = scalar_usize(value)
                    .ok_or_else(|| invalid_argument("ttest2: Dim must be a positive integer"))?;
                if dim == 0 {
                    return Err(invalid_argument("ttest2: Dim must be a positive integer"));
                }
                Ok(DecodedOption::Dim(dim - 1))
            }
            Ttest2OptionName::Tail => {
                let text = scalar_text(value, "Tail")?;
                Tail::from_public(&text)
                    .map(DecodedOption::Tail)
                    .ok_or_else(|| invalid_argument(format!("ttest2: unsupported Tail '{text}'")))
            }
            Ttest2OptionName::VarType => {
                let text = scalar_text(value, "Vartype")?;
                VarType::from_public(&text)
                    .map(DecodedOption::Variance)
                    .ok_or_else(|| {
                        invalid_argument(format!("ttest2: unsupported Vartype '{text}'"))
                    })
            }
        }
    }

    pub(super) async fn resolve(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Options> {
        let mut options = Options::default();
        for step in self.steps {
            let decoded = match step {
                Step::Ready(option) => option,
                Step::DeferredValue(kind, value) => {
                    let value = gathered_value(value, runtime).await?;
                    Self::decode_value(kind, &value)?
                }
                Step::DeferredName(name, value) => {
                    let name = gathered_value(name, runtime).await?;
                    let name = scalar_text(&name, "option name")?;
                    let kind = Ttest2OptionName::from_public(&name).ok_or_else(|| {
                        invalid_argument(format!("ttest2: unsupported option '{name}'"))
                    })?;
                    Self::gate_integer_option(kind, &value)?;
                    let value = gathered_value(value, runtime).await?;
                    Self::decode_value(kind, &value)?
                }
            };
            match decoded {
                DecodedOption::Alpha(alpha) => options.alpha = alpha,
                DecodedOption::Dim(dim) => options.dim = Some(dim),
                DecodedOption::Tail(tail) => options.tail = tail,
                DecodedOption::Variance(vartype) => options.vartype = vartype,
            }
        }
        Ok(options)
    }
}
