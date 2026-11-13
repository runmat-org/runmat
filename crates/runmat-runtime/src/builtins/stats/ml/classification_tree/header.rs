use super::*;

/// Inspect public argument positions while preserving strict text decoding failures.
pub(super) struct HeaderInspection {
    pub option_start: usize,
    slots: Vec<OptionSlot>,
}

struct OptionSlot {
    index: usize,
    spelling: Option<String>,
    identity: Option<FitctreeOption>,
}

pub(super) struct FitctreeArguments {
    pub option_start: usize,
    pub options: Vec<PreparedOption>,
}

pub(super) struct PreparedOption {
    pub name: NamedOption,
    pub predecoded: Option<FitctreeAction>,
}

pub(super) struct NamedOption {
    pub index: usize,
    pub spelling: String,
    pub identity: FitctreeOption,
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
            .and_then(FitctreeOption::from_public_name);
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
                        .and_then(FitctreeOption::from_public_name)
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

    pub fn known_options(&self) -> impl Iterator<Item = (usize, FitctreeOption)> + '_ {
        self.slots
            .iter()
            .filter_map(|slot| slot.identity.map(|identity| (slot.index, identity)))
    }

    pub fn validate(self, rest_len: usize) -> BuiltinResult<FitctreeArguments> {
        if rest_len < self.option_start {
            return Err(fitctree_invalid(
                "fitctree: matrix input requires response vector Y",
            ));
        }
        if !(rest_len - self.option_start).is_multiple_of(2) {
            return Err(fitctree_invalid(
                "fitctree: name-value options must be paired",
            ));
        }
        let mut options = Vec::with_capacity(self.slots.len());
        for slot in self.slots {
            let spelling = slot
                .spelling
                .ok_or_else(|| fitctree_invalid("fitctree: option name must be text"))?;
            let identity = slot.identity.ok_or_else(|| {
                fitctree_invalid(format!(
                    "fitctree: unknown option '{}'",
                    spelling.to_ascii_lowercase()
                ))
            })?;
            if !identity.supported() {
                return Err(fitctree_invalid(format!(
                    "fitctree: option '{spelling}' is not supported yet"
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
        Ok(FitctreeArguments {
            option_start: self.option_start,
            options,
        })
    }
}

impl FitctreeArguments {
    pub fn preflight_values(
        mut self,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        for option in &mut self.options {
            let value = &rest[option.name.index + 1];
            if !crate::dispatcher::value_contains_gpu(value) {
                option.predecoded = Some(decode_option(&option.name, value)?);
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
        return Err(fitctree_invalid("fitctree: X must be a 2-D numeric matrix"));
    }
    Ok(())
}

pub(super) fn preflight_response_shape(
    first: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    arguments: &FitctreeArguments,
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
        return Err(fitctree_invalid("predict: Y must be a vector"));
    }
    if !is_table_value(first) {
        let rows = match first {
            Value::Tensor(tensor) => Some(classification_tree_matrix_dimensions(tensor).0),
            Value::LogicalArray(array) => Some(array.shape.first().copied().unwrap_or(1)),
            Value::GpuTensor(handle) => Some(handle.shape().first().copied().unwrap_or(1)),
            _ => None,
        };
        let len = shape
            .iter()
            .try_fold(1usize, |product, extent| product.checked_mul(*extent));
        if let (Some(rows), Some(len)) = (rows, len) {
            if rows != len {
                return Err(fitctree_invalid(
                    "fitctree: length(Y) must match the number of rows in X",
                ));
            }
        }
    }
    Ok(())
}
