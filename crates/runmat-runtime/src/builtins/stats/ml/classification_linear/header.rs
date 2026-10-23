use super::*;

/// Inspect public argument positions while preserving strict text decoding failures.
pub(super) struct HeaderInspection {
    pub option_start: usize,
    slots: Vec<OptionSlot>,
}

struct OptionSlot {
    index: usize,
    spelling: Option<String>,
    identity: Option<FitclinearOption>,
}

pub(super) struct FitclinearArguments {
    pub option_start: usize,
    pub options: Vec<PreparedOption>,
}

pub(super) struct PreparedOption {
    pub name: NamedOption,
    pub predecoded: Option<FitclinearAction>,
}

pub(super) struct NamedOption {
    pub index: usize,
    pub spelling: String,
    pub identity: FitclinearOption,
}

impl HeaderInspection {
    pub fn inspect(
        first: &Value,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        let first_name = match rest.first() {
            Some(value) => scalar_text(value)?,
            None => None,
        };
        let first_identity = first_name
            .as_deref()
            .and_then(FitclinearOption::from_public_name);
        let option_start = if is_table_value(first) {
            usize::from(!rest.is_empty() && first_identity.is_none())
        } else {
            1
        };
        let slots = (option_start..rest.len())
            .step_by(2)
            .map(|index| -> BuiltinResult<OptionSlot> {
                let spelling = if index == 0 {
                    first_name.clone()
                } else {
                    scalar_text(&rest[index])?
                };
                let identity = if index == 0 {
                    first_identity
                } else {
                    spelling
                        .as_deref()
                        .and_then(FitclinearOption::from_public_name)
                };
                Ok(OptionSlot {
                    index,
                    spelling,
                    identity,
                })
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        Ok(Self {
            option_start,
            slots,
        })
    }

    pub fn known_options(&self) -> impl Iterator<Item = (usize, FitclinearOption)> + '_ {
        self.slots
            .iter()
            .filter_map(|slot| slot.identity.map(|identity| (slot.index, identity)))
    }

    pub fn validate(self, rest_len: usize) -> BuiltinResult<FitclinearArguments> {
        if rest_len < self.option_start {
            return Err(fitclinear_invalid(
                "fitclinear: matrix input requires response vector Y",
            ));
        }
        if !(rest_len - self.option_start).is_multiple_of(2) {
            return Err(fitclinear_invalid(
                "fitclinear: name-value options must be paired",
            ));
        }
        let mut options = Vec::with_capacity(self.slots.len());
        for slot in self.slots {
            let spelling = slot
                .spelling
                .ok_or_else(|| fitclinear_invalid("fitclinear: option name must be text"))?;
            let identity = slot.identity.ok_or_else(|| {
                fitclinear_invalid(format!(
                    "fitclinear: unknown option '{}'",
                    spelling.to_ascii_lowercase()
                ))
            })?;
            if identity.unsupported() {
                return Err(fitclinear_invalid(format!(
                    "fitclinear: option '{spelling}' is not supported yet"
                )));
            }
            options.push(PreparedOption {
                name: NamedOption {
                    index: slot.index,
                    spelling,
                    identity,
                },
                predecoded: None,
            });
        }
        Ok(FitclinearArguments {
            option_start: self.option_start,
            options,
        })
    }
}

impl FitclinearArguments {
    pub fn preflight_values(
        mut self,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        for option in &mut self.options {
            if option.name.identity.ignored() {
                option.predecoded = Some(FitclinearAction::Ignored);
            } else {
                let value = &rest[option.name.index + 1];
                if !crate::dispatcher::value_contains_gpu(value) {
                    option.predecoded = Some(decode_fit_option(&option.name, value)?);
                }
            }
        }
        Ok(self)
    }
}

pub(super) fn preflight_matrix_rank(first: &Value) -> BuiltinResult<()> {
    if is_table_value(first) {
        return Ok(());
    }
    let rank = match first {
        Value::Tensor(tensor) => Some(tensor.shape().len()),
        Value::LogicalArray(array) => Some(array.shape.len()),
        Value::GpuTensor(handle) => Some(handle.shape().len()),
        _ => None,
    };
    if rank.is_some_and(|rank| rank > 2) {
        return Err(fitclinear_invalid(
            "predict: X must be a 2-D numeric matrix",
        ));
    }
    Ok(())
}

pub(super) fn preflight_response_shape(
    first: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    arguments: &FitclinearArguments,
) -> BuiltinResult<()> {
    let response = if is_table_value(first) {
        if arguments.option_start == 1 {
            match rest.first() {
                Some(value) if scalar_text(value)?.is_none() => Some(value),
                _ => None,
            }
        } else {
            None
        }
    } else {
        rest.first()
    };
    let Some(response) = response else {
        return Ok(());
    };
    let shape = match response {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(array.shape.as_slice()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        _ => None,
    };
    let Some(shape) = shape else {
        return Ok(());
    };
    if shape.len() > 2
        || (shape.first().copied().unwrap_or(1) != 1 && shape.get(1).copied().unwrap_or(1) != 1)
    {
        return Err(fitclinear_invalid("predict: Y must be a vector"));
    }
    if !is_table_value(first) {
        let mut axis = Some(ObservationsIn::Rows);
        for option in &arguments.options {
            if option.name.identity == FitclinearOption::ObservationsIn {
                axis = match option.predecoded.as_ref() {
                    Some(FitclinearAction::ObservationsIn(axis)) => Some(*axis),
                    _ => None,
                };
            }
        }
        let x_shape = match first {
            Value::Tensor(tensor) => Some(tensor.shape()),
            Value::LogicalArray(array) => Some(array.shape.as_slice()),
            Value::GpuTensor(handle) => Some(handle.shape()),
            _ => None,
        };
        if let (Some(axis), Some(x_shape), Some(len)) = (
            axis,
            x_shape,
            shape
                .iter()
                .try_fold(1usize, |product, extent| product.checked_mul(*extent)),
        ) {
            let observations = match axis {
                ObservationsIn::Rows => x_shape.first().copied().unwrap_or(1),
                ObservationsIn::Columns => x_shape.get(1).copied().unwrap_or(1),
            };
            if observations != len {
                return Err(fitclinear_invalid(
                    "fitclinear: length(Y) must match the number of observations in X",
                ));
            }
        }
    }
    Ok(())
}
