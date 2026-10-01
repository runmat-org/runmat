use super::*;

/// Inspect public argument positions without failing before compatibility gates run.
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
    pub options: Vec<NamedOption>,
}

pub(super) struct NamedOption {
    pub index: usize,
    pub spelling: String,
    pub identity: FitctreeOption,
}

impl HeaderInspection {
    pub fn inspect(first: &Value, rest: &[Value]) -> Self {
        let first_name = rest.first().and_then(scalar_text);
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
            .map(|index| {
                let spelling = if index == 0 {
                    first_name.clone()
                } else {
                    scalar_text(&rest[index])
                };
                let identity = if index == 0 {
                    first_identity
                } else {
                    spelling
                        .as_deref()
                        .and_then(FitctreeOption::from_public_name)
                };
                OptionSlot {
                    index,
                    spelling,
                    identity,
                }
            })
            .collect();
        Self {
            option_start,
            slots,
        }
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
            options.push(NamedOption {
                index: slot.index,
                spelling,
                identity,
            });
        }
        Ok(FitctreeArguments {
            option_start: self.option_start,
            options,
        })
    }
}

pub(super) fn preflight_matrix_rank(first: &Value) -> BuiltinResult<()> {
    if is_table_value(first) {
        return Ok(());
    }
    let rank = match first {
        Value::Tensor(tensor) => Some(tensor.shape.len()),
        Value::LogicalArray(array) => Some(array.shape.len()),
        Value::GpuTensor(handle) => Some(handle.shape().len()),
        _ => None,
    };
    if rank.is_some_and(|rank| rank > 2) {
        return Err(fitctree_invalid("fitctree: X must be a 2-D numeric matrix"));
    }
    Ok(())
}
