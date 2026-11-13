//! Name-value identity admission for multinomial fitting.

mod runtime;

pub(super) use runtime::prepare_values;

#[derive(Clone, Debug)]
pub(super) struct Options {
    pub(super) constant: bool,
    pub(super) weights: Option<Vec<f64>>,
    pub(super) max_iter: usize,
    pub(super) tol_x: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            constant: true,
            weights: None,
            max_iter: 100,
            tol_x: 1.0e-6,
        }
    }
}

use super::{invalid, scalar_text};
use crate::BuiltinResult;
pub(super) use runmat_builtins::MnrfitOption as OptionName;

pub(super) struct OptionSlot {
    spelling: Option<String>,
    identity: Option<OptionName>,
}

pub(super) fn inspect(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<OptionSlot>> {
    values
        .chunks_exact(2)
        .map(|pair| {
            let spelling = scalar_text(&pair[0])?.map(std::borrow::Cow::into_owned);
            let identity = spelling.as_deref().and_then(OptionName::from_public);
            Ok(OptionSlot { spelling, identity })
        })
        .collect()
}

pub(super) fn identities(slots: &[OptionSlot]) -> impl Iterator<Item = Option<OptionName>> + '_ {
    slots.iter().map(|slot| slot.identity)
}

pub(super) fn validate(
    values: crate::call::FunctionArgumentView<'_>,
    slots: Vec<OptionSlot>,
) -> BuiltinResult<Vec<OptionName>> {
    if !values.len().is_multiple_of(2) {
        return Err(invalid("mnrfit: name-value options must be paired"));
    }
    slots
        .into_iter()
        .map(|slot| {
            let spelling = slot
                .spelling
                .ok_or_else(|| invalid("mnrfit: option name must be text"))?;
            slot.identity.ok_or_else(|| {
                invalid(format!(
                    "mnrfit: unknown option '{}'",
                    spelling.to_ascii_lowercase()
                ))
            })
        })
        .collect()
}
