use super::*;

pub(super) struct HeaderInspection {
    pub option_start: usize,
    slots: Vec<OptionSlot>,
}

struct OptionSlot {
    index: usize,
    spelling: Option<String>,
    identity: Option<FitlmOption>,
}

pub(super) struct FitlmArguments {
    pub option_start: usize,
    pub options: Vec<PreparedOption>,
}

pub(super) struct PreparedOption {
    pub name: NamedOption,
    pub predecoded: Option<FitlmAction>,
}

pub(super) struct NamedOption {
    pub index: usize,
    pub spelling: String,
    pub identity: FitlmOption,
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
            .and_then(FitlmOption::from_public_name);
        let second_name = match rest.get(1) {
            Some(value) => scalar_text(value)?,
            None => None,
        };
        let second_identity = second_name
            .as_deref()
            .and_then(FitlmOption::from_public_name);
        let option_start = if is_table_value(first) {
            usize::from(!rest.is_empty() && first_identity.is_none())
        } else {
            1 + usize::from(rest.get(1).is_some() && second_identity.is_none())
        };
        let slots = (option_start..rest.len())
            .step_by(2)
            .map(|index| -> BuiltinResult<OptionSlot> {
                let (spelling, identity) = match index {
                    0 => (first_name.clone(), first_identity),
                    1 => (second_name.clone(), second_identity),
                    _ => {
                        let spelling = scalar_text(&rest[index])?;
                        let identity = spelling.as_deref().and_then(FitlmOption::from_public_name);
                        (spelling, identity)
                    }
                };
                Ok(OptionSlot {
                    index,
                    spelling: spelling.map(std::borrow::Cow::into_owned),
                    identity,
                })
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        Ok(Self {
            option_start,
            slots,
        })
    }

    pub fn known_options(&self) -> impl Iterator<Item = (usize, FitlmOption)> + '_ {
        self.slots
            .iter()
            .filter_map(|slot| slot.identity.map(|identity| (slot.index, identity)))
    }

    pub fn validate(self, rest_len: usize) -> BuiltinResult<FitlmArguments> {
        if rest_len < self.option_start {
            return Err(fitlm_invalid(
                "fitlm: matrix input requires response vector y",
            ));
        }
        if !(rest_len - self.option_start).is_multiple_of(2) {
            return Err(fitlm_invalid("fitlm: name-value options must be paired"));
        }
        let mut options = Vec::with_capacity(self.slots.len());
        for slot in self.slots {
            let spelling = slot
                .spelling
                .ok_or_else(|| fitlm_invalid("fitlm: option name must be text"))?;
            let identity = slot
                .identity
                .ok_or_else(|| fitlm_invalid(format!("fitlm: unknown option '{spelling}'")))?;
            if identity.unsupported() {
                return Err(fitlm_invalid(format!(
                    "fitlm: option '{spelling}' is not supported yet"
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
        Ok(FitlmArguments {
            option_start: self.option_start,
            options,
        })
    }
}

impl FitlmArguments {
    pub fn preflight_values(
        mut self,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        for option in &mut self.options {
            let value = &rest[option.name.index + 1];
            if !crate::dispatcher::value_contains_gpu(value) {
                option.predecoded = Some(decode_fit_option(&option.name, value)?);
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
        return Err(fitlm_invalid("fitlm: X must be a 2-D numeric matrix"));
    }
    Ok(())
}

pub(super) fn preflight_response_shape(
    first: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if is_table_value(first) {
        return Ok(());
    }
    let Some(response) = rest.first() else {
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
        return Err(fitlm_invalid("fitlm: y must be a vector"));
    }
    let rows = match first {
        Value::Tensor(tensor) => Some(fitlm_matrix_dimensions(tensor).0),
        Value::LogicalArray(array) => Some(array.shape.first().copied().unwrap_or(1)),
        Value::GpuTensor(handle) => Some(handle.shape().first().copied().unwrap_or(1)),
        _ => None,
    };
    let len = shape
        .iter()
        .try_fold(1usize, |product, extent| product.checked_mul(*extent));
    if let (Some(rows), Some(len)) = (rows, len) {
        if rows != len {
            return Err(fitlm_invalid(
                "fitlm: length(y) must match the number of rows in X",
            ));
        }
    }
    Ok(())
}
