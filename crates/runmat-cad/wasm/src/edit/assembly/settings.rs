//! `assembly.document-settings`: the unit system the assembly's readouts use
//! ([`runmat_cad::assembly::settings`]), written as `settings = { units = ips }`.

use super::{model, text_value};
use runmat_cad::assembly::model::AssemblyDocument;
use runmat_cad::assembly::settings::{DocumentSettings, UnitSystem};
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{Map, Value};

pub(super) fn apply(
    document: &mut AssemblyDocument,
    values: &Map<String, Value>,
) -> Result<String, EditFailure> {
    let units = match text_value(values, "units") {
        "mmgs" => UnitSystem::Mmgs,
        "mks" => UnitSystem::Mks,
        "ips" => UnitSystem::Ips,
        "" => return Err(model("choose the units")),
        other => {
            return Err(model(format!(
                "`{other}` isn't a unit system: choose mmgs, mks or ips"
            )))
        }
    };
    if document.settings.as_ref().map(|s| s.units) == Some(units) {
        return Err(model(format!("the assembly already reads in {}", label(units))));
    }
    document
        .settings
        .get_or_insert_with(DocumentSettings::default)
        .units = units;
    Ok(format!("Units set to {}", label(units)))
}

fn label(units: UnitSystem) -> &'static str {
    match units {
        UnitSystem::Mmgs => "millimeter, gram, second",
        UnitSystem::Mks => "meter, kilogram, second",
        UnitSystem::Ips => "inch, pound, second",
    }
}
