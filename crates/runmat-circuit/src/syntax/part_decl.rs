//! A circuit's `part` declarations: catalog parts embedded in the file that uses them.
//!
//! ```text
//! part "BUCK-60V-3A" = {
//!     manufacturer = "Sample Catalog",
//!     description = "60 V 3 A synchronous buck",
//!     category = regulator,
//!     package = "SOT23-6",
//!     datasheet = "https://example.com/buck.pdf",
//!     params = { vin_max = "60 V" },
//!     pins = [
//!         { name = BST, pad = "1", type = passive },
//!         { name = GND, pad = "2", type = power_in }
//!     ],
//!     spice = { model = BUCK60, library = "models/buck.lib", nodes = [VIN, SW, FB, GND] }
//! }
//! ```

use super::printer::{list, name, names, record, PrintError};
use super::*;
use crate::library::catalog::{CatalogPart, CatalogPin, CatalogSpice};

fn text_of(expr: Expr, span: Span) -> Read<String> {
    word_of(Some(expr), span)
}

fn pin_of(expr: Expr, span: Span) -> Read<CatalogPin> {
    let mut pin = CatalogPin {
        name: String::new(),
        pads: vec![],
        kind: "passive".into(),
    };
    for (field, value) in record_of(Some(expr), span)? {
        match field.as_str() {
            "name" => pin.name = text_of(value, span)?,
            "pad" => pin.pads = vec![text_of(value, span)?],
            "pads" => pin.pads = words_of(Some(value), span)?,
            "type" => pin.kind = text_of(value, span)?,
            other => {
                return Err(diagnostic(
                    value.span(),
                    format!(
                    "a part's pin has no `{other}`; it takes `name`, `pad` or `pads`, and `type`"
                ),
                ))
            }
        }
    }
    if pin.name.is_empty() {
        return Err(diagnostic(span, "a part's pin needs its `name`"));
    }
    Ok(pin)
}

fn spice_of(expr: Expr, span: Span) -> Read<CatalogSpice> {
    let mut spice = CatalogSpice::default();
    for (field, value) in record_of(Some(expr), span)? {
        match field.as_str() {
            "model" => spice.model = text_of(value, span)?,
            "library" => spice.library = Some(text_of(value, span)?),
            "nodes" => spice.nodes = words_of(Some(value), span)?,
            "device" => spice.device = Some(text_of(value, span)?),
            other => return Err(diagnostic(value.span(), format!("a part's `spice` has no `{other}`; it takes `model`, `library`, `nodes` and `device`"))),
        }
    }
    if spice.model.is_empty() {
        return Err(diagnostic(span, "a part's `spice` needs its `model`"));
    }
    Ok(spice)
}

/// Reads `part "<mpn>" = { … }`.
pub(super) fn read(mpn: String, expr: Option<Expr>, span: Span) -> Read<CatalogPart> {
    let mut part = CatalogPart {
        mpn,
        ..Default::default()
    };
    for (field, value) in record_of(expr, span)? {
        match field.as_str() {
            "manufacturer" => part.manufacturer = text_of(value, span)?,
            "description" => part.description = text_of(value, span)?,
            "category" => part.category = text_of(value, span)?,
            "prefix" => part.prefix = text_of(value, span)?,
            "package" => part.package = text_of(value, span)?,
            "datasheet" => part.datasheet = Some(text_of(value, span)?),
            "params" => {
                for (k, v) in record_of(Some(value), span)? {
                    part.params.insert(k, text_of(v, span)?);
                }
            }
            "pins" => match value {
                Expr::List(items) => {
                    part.pins = items.items().into_iter().map(|item| pin_of(item, span)).collect::<Read<_>>()?
                }
                other => return Err(diagnostic(other.span(), "a part's `pins` is a list of `{ name, pad, type }` records")),
            },
            "spice" => part.spice = Some(spice_of(value, span)?),
            other => {
                return Err(diagnostic(
                    value.span(),
                    format!("a part has no `{other}`; it takes manufacturer, description, category, prefix, package, datasheet, params, pins and spice"),
                ))
            }
        }
    }
    if part.package.is_empty() {
        return Err(diagnostic(
            span,
            format!("part `{}` needs its `package`", part.mpn),
        ));
    }
    part.check().map_err(|m| diagnostic(span, m))?;
    Ok(part)
}

/// The value written after `part "<mpn>" = `, which [`read`] reads back to `part`.
pub fn print(part: &CatalogPart) -> Result<String, PrintError> {
    let mut entries: Vec<(String, String)> =
        vec![("manufacturer".into(), quote(&part.manufacturer))];
    if !part.description.is_empty() {
        entries.push(("description".into(), quote(&part.description)));
    }
    if !part.category.is_empty() {
        entries.push(("category".into(), name(&part.category)));
    }
    if !part.prefix.is_empty() {
        entries.push(("prefix".into(), name(&part.prefix)));
    }
    entries.push(("package".into(), quote(&part.package)));
    if let Some(d) = &part.datasheet {
        entries.push(("datasheet".into(), quote(d)));
    }
    if !part.params.is_empty() {
        let params: Vec<(String, String)> = part
            .params
            .iter()
            .map(|(k, v)| (k.clone(), quote(v)))
            .collect();
        entries.push(("params".into(), record(&params)));
    }
    let pins: Vec<String> = part
        .pins
        .iter()
        .map(|p| {
            let pads = match p.pads.as_slice() {
                [one] => ("pad".to_string(), quote(one)),
                many => (
                    "pads".to_string(),
                    list(&many.iter().map(|x| quote(x)).collect::<Vec<_>>()),
                ),
            };
            format!(
                "{{ name = {}, {} = {}, type = {} }}",
                name(&p.name),
                pads.0,
                pads.1,
                name(&p.kind)
            )
        })
        .collect();
    if pins.is_empty() {
        return Err(PrintError(format!("part `{}` has no pins", part.mpn)));
    }
    entries.push((
        "pins".into(),
        format!("[\n{INDENT}{}\n]", pins.join(&format!(",\n{INDENT}"))),
    ));
    if let Some(s) = &part.spice {
        let mut e = vec![("model".to_string(), name(&s.model))];
        if let Some(l) = &s.library {
            e.push(("library".into(), quote(l)));
        }
        if !s.nodes.is_empty() {
            e.push(("nodes".into(), names(&s.nodes)));
        }
        if let Some(d) = &s.device {
            e.push(("device".into(), name(d)));
        }
        entries.push(("spice".into(), record(&e)));
    }
    Ok(record(&entries))
}

/// The whole statement, `part "<mpn>" = { … }`.
pub fn statement(part: &CatalogPart) -> Result<String, PrintError> {
    Ok(format!(
        "{} {} = {}",
        words::PART,
        quote(&part.mpn),
        print(part)?
    ))
}
