//! Typed, ordered ECDF controls with host-visible admission before transfers.
use super::*;

enum DecodedOption {
    Function(FunctionKind),
    Censoring(Vec<f64>),
    Frequency(Vec<f64>),
    Alpha(f64),
    Bounds(EcdfBounds),
}

enum Step {
    Ready(BuiltinResult<DecodedOption>),
    Deferred(EcdfOptionName, Value),
}

pub(super) struct EcdfOptionPlan {
    steps: Vec<Step>,
    role_extensions: Vec<&'static BuiltinExtensionDescriptor>,
    censoring_count: Option<usize>,
    frequency_count: Option<usize>,
}

impl EcdfOptionPlan {
    pub(super) fn admit(rest: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        if !rest.len().is_multiple_of(2) {
            return Err(invalid_argument(
                ECDF_OWNER,
                "ecdf: name-value options must appear in pairs",
            ));
        }
        let mut steps = Vec::with_capacity(rest.len() / 2);
        let mut role_extensions = Vec::new();
        let mut censoring_count = None;
        let mut frequency_count = None;
        let mut rest = rest.into_iter();
        while let (Some(name), Some(value)) = (rest.next(), rest.next()) {
            // A resident numeric handle cannot be an option name. This remains
            // a host-visible role error even when a sample is resident.
            if crate::value_contains_gpu(&name) {
                steps.push(Step::Ready(Err(invalid_argument(
                    ECDF_OWNER,
                    "ecdf: option names must be text",
                ))));
                continue;
            }
            let name = match keyword_of(&name) {
                Ok(Some(name)) => name,
                Ok(None) => {
                    steps.push(Step::Ready(Err(invalid_argument(
                        ECDF_OWNER,
                        "ecdf: option names must be text",
                    ))));
                    continue;
                }
                Err(error) => {
                    steps.push(Step::Ready(Err(error)));
                    continue;
                }
            };
            let Some(option) = EcdfOptionName::from_public(&name) else {
                steps.push(Step::Ready(Err(invalid_argument(
                    ECDF_OWNER,
                    format!("ecdf: unsupported option '{}'", name.to_ascii_lowercase()),
                ))));
                continue;
            };
            if option.is_interval_only() {
                steps.push(Step::Ready(Err(unsupported(format!(
                    "ecdf: option '{name}' is only valid for interval or double-censored data"
                )))));
                continue;
            }
            match option {
                EcdfOptionName::Frequency => {
                    frequency_count = known_count(&value);
                    if is_typed_integer_value(&value) {
                        role_extensions.push(&ECDF_INTEGER_FREQUENCY_EXTENSION);
                    } else if is_logical_value(&value) {
                        role_extensions.push(&ECDF_LOGICAL_FREQUENCY_EXTENSION);
                    }
                }
                EcdfOptionName::Censoring if is_typed_integer_value(&value) => {
                    censoring_count = known_count(&value);
                    role_extensions.push(&ECDF_INTEGER_CENSORING_EXTENSION);
                }
                EcdfOptionName::Censoring => censoring_count = known_count(&value),
                _ => {}
            }
            if crate::value_contains_gpu(&value) {
                match Self::validate_resident_role(option, &value) {
                    Ok(()) => steps.push(Step::Deferred(option, value)),
                    Err(error) => steps.push(Step::Ready(Err(error))),
                }
            } else {
                steps.push(Step::Ready(Self::decode(option, &value)));
            }
        }
        Ok(Self {
            steps,
            role_extensions,
            censoring_count,
            frequency_count,
        })
    }

    pub(super) fn ensure_extensions(&self) -> BuiltinResult<()> {
        for extension in &self.role_extensions {
            crate::compatibility::ensure_builtin_extension_enabled(extension, ECDF_NAME)?;
        }
        Ok(())
    }

    pub(super) fn preflight_known(mut self) -> BuiltinResult<Self> {
        let mut steps = Vec::with_capacity(self.steps.len());
        for step in self.steps {
            match step {
                Step::Ready(Err(error)) => return Err(error),
                step => steps.push(step),
            }
        }
        self.steps = steps;
        Ok(self)
    }

    pub(super) fn preflight_lengths(self, sample: &Value) -> BuiltinResult<Self> {
        if let Some(sample_count) = known_count(sample) {
            if self
                .censoring_count
                .is_some_and(|count| count != sample_count)
            {
                return Err(invalid_argument(
                    ECDF_OWNER,
                    "ecdf: Censoring vector must have the same number of elements as y",
                ));
            }
            if self
                .frequency_count
                .is_some_and(|count| count != sample_count)
            {
                return Err(invalid_argument(
                    ECDF_OWNER,
                    "ecdf: Frequency vector must have the same number of elements as y",
                ));
            }
        }
        Ok(self)
    }

    pub(super) fn borrowed_handles(
        &self,
    ) -> BuiltinResult<Vec<runmat_accelerate_api::GpuTensorHandle>> {
        let mut handles = Vec::new();
        for step in &self.steps {
            if let Step::Deferred(_, value) = step {
                crate::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
                    handles.push(handle.clone());
                })
                .map_err(|cause| map_error(ECDF_OWNER, cause))?;
            }
        }
        Ok(handles)
    }

    pub(super) async fn resolve(self) -> BuiltinResult<Options> {
        let mut options = Options::default();
        for step in self.steps {
            let decoded = match step {
                Step::Ready(value) => value?,
                Step::Deferred(option, value) => {
                    let gathered = gather_ecdf_value(value).await?;
                    ensure_exact_ecdf_integer_boundary(
                        &gathered,
                        match option {
                            EcdfOptionName::Frequency => "Frequency",
                            EcdfOptionName::Censoring => "Censoring",
                            _ => "option",
                        },
                    )?;
                    Self::decode(option, &gathered)?
                }
            };
            match decoded {
                DecodedOption::Function(value) => options.function = value,
                DecodedOption::Censoring(value) => options.censoring = Some(value),
                DecodedOption::Frequency(value) => options.frequency = Some(value),
                DecodedOption::Alpha(value) => options.alpha = value,
                DecodedOption::Bounds(value) => options.bounds_on = value == EcdfBounds::On,
            }
        }
        Ok(options)
    }

    fn validate_resident_role(option: EcdfOptionName, value: &Value) -> BuiltinResult<()> {
        match option {
            EcdfOptionName::Function => Err(invalid_argument(
                ECDF_OWNER,
                "ecdf: Function option must be text",
            )),
            EcdfOptionName::Bounds => Err(invalid_argument(
                ECDF_OWNER,
                "ecdf: Bounds option must be text",
            )),
            EcdfOptionName::Alpha => {
                let Value::GpuTensor(handle) = value else {
                    return Ok(());
                };
                if runmat_accelerate_api::handle_storage(handle)
                    != runmat_accelerate_api::GpuTensorStorage::Real
                    || handle
                        .shape()
                        .iter()
                        .try_fold(1usize, |n, dim| n.checked_mul(*dim))
                        != Some(1)
                {
                    return Err(invalid_argument(
                        ECDF_OWNER,
                        "ecdf: Alpha must be a numeric scalar",
                    ));
                }
                Ok(())
            }
            EcdfOptionName::Censoring | EcdfOptionName::Frequency => {
                let Value::GpuTensor(handle) = value else {
                    return Ok(());
                };
                if runmat_accelerate_api::handle_storage(handle)
                    != runmat_accelerate_api::GpuTensorStorage::Real
                {
                    return Err(invalid_argument(
                        ECDF_OWNER,
                        "ecdf: observation controls must be real numeric vectors",
                    ));
                }
                let shape = handle.shape();
                let rows = shape.first().copied().unwrap_or(1);
                let cols = shape.get(1).copied().unwrap_or(1);
                if rows > 1 && cols > 1 {
                    return Err(invalid_argument(
                        ECDF_OWNER,
                        "ecdf: observation controls must be vectors",
                    ));
                }
                Ok(())
            }
            _ => unreachable!("interval-only controls were rejected during admission"),
        }
    }

    fn decode(option: EcdfOptionName, value: &Value) -> BuiltinResult<DecodedOption> {
        match option {
            EcdfOptionName::Function => {
                let text = keyword_of(value)?.ok_or_else(|| {
                    invalid_argument(ECDF_OWNER, "ecdf: Function option must be text")
                })?;
                let function = FunctionKind::from_public(&text).ok_or_else(|| {
                    invalid_argument(
                        ECDF_OWNER,
                        format!(
                            "ecdf: unsupported Function value '{}'",
                            text.to_ascii_lowercase()
                        ),
                    )
                })?;
                Ok(DecodedOption::Function(function))
            }
            EcdfOptionName::Censoring => {
                ensure_exact_ecdf_integer_boundary(value, "Censoring")?;
                Ok(DecodedOption::Censoring(numeric_vector(value, ECDF_OWNER)?))
            }
            EcdfOptionName::Frequency => {
                ensure_exact_ecdf_integer_boundary(value, "Frequency")?;
                Ok(DecodedOption::Frequency(numeric_vector(value, ECDF_OWNER)?))
            }
            EcdfOptionName::Alpha => {
                let alpha = scalar_number(value).ok_or_else(|| {
                    invalid_argument(ECDF_OWNER, "ecdf: Alpha must be a numeric scalar")
                })?;
                if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
                    return Err(invalid_argument(
                        ECDF_OWNER,
                        "ecdf: Alpha must be a finite scalar in the open interval (0,1)",
                    ));
                }
                Ok(DecodedOption::Alpha(alpha))
            }
            EcdfOptionName::Bounds => {
                let text = keyword_of(value)?.ok_or_else(|| {
                    invalid_argument(ECDF_OWNER, "ecdf: Bounds option must be text")
                })?;
                let bounds = EcdfBounds::from_public(&text).ok_or_else(|| {
                    invalid_argument(
                        ECDF_OWNER,
                        format!(
                            "ecdf: unsupported Bounds value '{}'",
                            text.to_ascii_lowercase()
                        ),
                    )
                })?;
                Ok(DecodedOption::Bounds(bounds))
            }
            _ => unreachable!("interval-only controls were rejected during admission"),
        }
    }
}

fn known_count(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(tensor) => Some(tensor.len()),
        Value::LogicalArray(array) => Some(array.data.len()),
        Value::GpuTensor(handle) => handle
            .shape()
            .iter()
            .try_fold(1usize, |n, dim| n.checked_mul(*dim)),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(1),
        _ => None,
    }
}
