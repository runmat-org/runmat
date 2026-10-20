//! One admitted typed kstest control plan; host-visible failures precede downloads.
use super::*;

enum Step {
    Ready(BuiltinResult<DecodedOption>),
    DeferredValue(KstestOptionName, Value),
}

enum DecodedOption {
    Alpha(f64),
    Tail(KstestTail),
    Cdf(CdfSpec),
}

pub(super) struct KstestOptionPlan {
    steps: Vec<Step>,
    integer_cdf: bool,
    explicit_control: bool,
}

impl KstestOptionPlan {
    pub(super) fn admit(rest: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        if !rest.len().is_multiple_of(2) {
            return Err(kstest_invalid_argument(
                "kstest: name-value options must be supplied in pairs",
            ));
        }
        let explicit_control = rest.iter().any(kstest_value_contains_explicit_gpu);
        let mut integer_cdf = false;
        let mut steps = Vec::with_capacity(rest.len() / 2);
        let mut rest = rest.into_iter();
        while let (Some(name), Some(value)) = (rest.next(), rest.next()) {
            if crate::value_contains_gpu(&name) {
                // A resident numeric value cannot decode as a public option name.
                // Retain the error until after compatibility gates, but never
                // download the sample or the invalid name to discover it.
                steps.push(Step::Ready(Err(kstest_invalid_argument(
                    "kstest: option name must be a string scalar",
                ))));
                continue;
            }
            let name = kstest_scalar_text(&name, "option name")?;
            let option = KstestOptionName::from_public(&name).ok_or_else(|| {
                kstest_invalid_argument(format!(
                    "kstest: unsupported option '{}'",
                    name.to_ascii_lowercase()
                ))
            })?;
            integer_cdf |=
                option == KstestOptionName::Cdf && kstest_cdf_contains_typed_integer(&value);
            if crate::value_contains_gpu(&value) {
                Self::validate_resident_role(option, &value)?;
                steps.push(Step::DeferredValue(option, value));
            } else {
                Self::validate_resident_role(option, &value)?;
                steps.push(Step::Ready(Self::decode(option, &value)));
            }
        }
        Ok(Self {
            steps,
            integer_cdf,
            explicit_control,
        })
    }

    pub(super) fn ensure_extensions(&self, sample: &Value) -> BuiltinResult<()> {
        if self.integer_cdf {
            crate::compatibility::ensure_builtin_extension_enabled(
                &KSTEST_INTEGER_CDF_EXTENSION,
                KSTEST_NAME,
            )?;
        }
        if self.explicit_control || kstest_value_contains_explicit_gpu(sample) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &KSTEST_RESIDENT_INPUT_EXTENSION,
                KSTEST_NAME,
            )?;
        }
        Ok(())
    }

    pub(super) fn preflight_known(mut self) -> BuiltinResult<Self> {
        let mut admitted = Vec::with_capacity(self.steps.len());
        for step in self.steps {
            match step {
                Step::Ready(Err(error)) => return Err(error),
                step => admitted.push(step),
            }
        }
        self.steps = admitted;
        Ok(self)
    }

    fn validate_resident_role(option: KstestOptionName, value: &Value) -> BuiltinResult<()> {
        match option {
            KstestOptionName::Alpha => validate_kstest_alpha_role(value),
            KstestOptionName::Tail => kstest_scalar_text(value, "Tail").map(|_| ()),
            KstestOptionName::Cdf => validate_kstest_cdf_role(value),
        }
    }

    fn decode(option: KstestOptionName, value: &Value) -> BuiltinResult<DecodedOption> {
        match option {
            KstestOptionName::Alpha => {
                validate_kstest_alpha_role(value)?;
                let alpha = kstest_scalar_number(value)
                    .ok_or_else(|| kstest_invalid_argument("kstest: Alpha must be numeric"))?;
                if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
                    return Err(kstest_invalid_argument(
                        "kstest: Alpha must be a scalar in the open interval (0,1)",
                    ));
                }
                Ok(DecodedOption::Alpha(alpha))
            }
            KstestOptionName::Tail => {
                let tail = kstest_scalar_text(value, "Tail")?;
                KstestTail::from_public(&tail)
                    .map(DecodedOption::Tail)
                    .ok_or_else(|| {
                        kstest_invalid_argument(format!(
                            "kstest: unsupported Tail '{}'",
                            tail.to_ascii_lowercase()
                        ))
                    })
            }
            KstestOptionName::Cdf => parse_kstest_cdf(value).map(DecodedOption::Cdf),
        }
    }

    pub(super) async fn resolve(self) -> BuiltinResult<KstestOptions> {
        let mut options = KstestOptions::default();
        for step in self.steps {
            let decoded = match step {
                Step::Ready(value) => value?,
                Step::DeferredValue(option, value) => {
                    let value = kstest_gather(value).await?;
                    Self::decode(option, &value)?
                }
            };
            match decoded {
                DecodedOption::Alpha(alpha) => options.alpha = alpha,
                DecodedOption::Tail(tail) => options.tail = tail,
                DecodedOption::Cdf(cdf) => options.cdf = cdf,
            }
        }
        Ok(options)
    }
}
