//! Typed public option identities for performance curves.

use super::{canonical, invalid, scalar_text, unsupported};
use crate::BuiltinResult;
use runmat_value::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OptionName {
    XCrit,
    YCrit,
    NegClass,
    Weights,
    Prior,
    Cost,
    TVals,
    XVals,
    UseNearest,
    ProcessNaN,
    UnsupportedBootstrap,
}

impl OptionName {
    fn from_public(text: &str) -> Option<Self> {
        Some(match canonical(text).as_str() {
            "xcrit" => Self::XCrit,
            "ycrit" => Self::YCrit,
            "negclass" => Self::NegClass,
            "weights" => Self::Weights,
            "prior" => Self::Prior,
            "cost" => Self::Cost,
            "tvals" => Self::TVals,
            "xvals" => Self::XVals,
            "usenearest" => Self::UseNearest,
            "processnan" => Self::ProcessNaN,
            "nboot" | "alpha" | "bootstrap" | "bootstraps" | "nbootstraps" => {
                Self::UnsupportedBootstrap
            }
            _ => return None,
        })
    }
}

pub(super) struct OptionSlot {
    name: BuiltinResult<String>,
    identity: Option<OptionName>,
}

pub(super) fn inspect(rest: &[Value]) -> Vec<OptionSlot> {
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

pub(super) fn validate(rest: &[Value], slots: Vec<OptionSlot>) -> BuiltinResult<Vec<OptionName>> {
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
                    canonical(&spelling)
                ))),
            }
        })
        .collect()
}
