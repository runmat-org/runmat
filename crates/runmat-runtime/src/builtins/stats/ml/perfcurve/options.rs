//! Typed public option identities for performance curves.

use super::{invalid, scalar_text, unsupported};
use crate::BuiltinResult;

pub(super) use runmat_builtins::PerfcurveOptionName as OptionName;

pub(super) struct OptionSlot {
    name: BuiltinResult<String>,
    identity: Option<OptionName>,
}

pub(super) fn inspect(rest: crate::call::FunctionArgumentView<'_>) -> Vec<OptionSlot> {
    rest.chunks_exact(2)
        .map(|pair| {
            let name = scalar_text(&pair[0], "option name");
            let identity = name
                .as_ref()
                .ok()
                .and_then(|text| OptionName::from_public(text));
            OptionSlot { name, identity }
        })
        .collect()
}

pub(super) fn identities(slots: &[OptionSlot]) -> impl Iterator<Item = Option<OptionName>> + '_ {
    slots.iter().map(|slot| slot.identity)
}

pub(super) fn validate(
    rest: crate::call::FunctionArgumentView<'_>,
    slots: Vec<OptionSlot>,
) -> BuiltinResult<Vec<OptionName>> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid("perfcurve: name-value options must be paired"));
    }
    slots
        .into_iter()
        .map(|slot| {
            let spelling = slot.name?;
            match slot.identity {
                Some(OptionName::UnsupportedBootstrap) => Err(unsupported(
                    "perfcurve: confidence-bound and bootstrap modes are not implemented",
                )),
                Some(name) => Ok(name),
                None => Err(invalid(format!(
                    "perfcurve: unsupported option '{}'",
                    OptionName::canonical_spelling(&spelling)
                ))),
            }
        })
        .collect()
}
