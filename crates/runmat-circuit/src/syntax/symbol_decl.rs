//! `symbol <name> = { … }`: a schematic symbol a part can draw with in place of the one
//! generated from its pins. A part maps its pins to the symbol's pins by name.
//!
//! ```text
//! symbol opamp_single = {
//!     body = [polygon([[-2.54, -2.54], [-2.54, 2.54], [2.54, 0]])],
//!     pins = [
//!         { name = "IN+", at = [-5.08, -1.27], dir = right, length = 2.54 },
//!         { name = "IN-", at = [-5.08, 1.27], dir = right, length = 2.54 },
//!         { name = OUT, at = [5.08, 0], dir = left, length = 2.54 }
//!     ]
//! }
//! ```
//!
//! Lengths are mm. A pin's `at` is the end wires connect to; `dir` is the way the pin runs
//! from there toward the body (`right`, `up`, `left` or `down`).

use super::footprint_decl::{drawn, graphic, graphics_text, Drawn};
use super::printer::PrintError;
use super::values::{len, mm_text, point, pt_text, text, val, Fields, Val};
use super::*;
use crate::geom::mm;
use crate::ids::Uuid;
use crate::library::make::uuid;
use crate::library::{Graphic, Symbol, SymbolPin};

/// Symbol line width when a drawing gives none, mm.
pub const SYMBOL_WIDTH: f64 = 0.254;
/// A pin's length when it gives none, mm.
pub const PIN_LENGTH: f64 = 2.54;

const DIRS: &[(&str, f64)] = &[("right", 0.0), ("up", 90.0), ("left", 180.0), ("down", 270.0)];

fn pin_of(v: &Val, symbol: &str) -> Read<SymbolPin> {
    let mut f = Fields::of(v, "symbol pin")?;
    let name = text(f.need("name")?)?;
    let dir = f.opt("dir", text)?.unwrap_or_else(|| "right".into());
    let angle = DIRS
        .iter()
        .find(|(w, _)| *w == dir)
        .map(|(_, a)| *a)
        .ok_or_else(|| diagnostic(v.span, format!("symbol `{symbol}`: pin `{name}` has dir `{dir}`; a pin runs right, up, left or down")))?;
    let pin = SymbolPin {
        pin: name,
        at: f.need("at").and_then(point)?,
        angle,
        length: f.or("length", mm(PIN_LENGTH), len)?,
    };
    f.finish()?;
    Ok(pin)
}

/// Reads `symbol <name> = { … }`.
pub(super) fn read(name: String, expr: Option<Expr>, span: Span) -> Read<Symbol> {
    let mut graphics: Vec<Graphic> = Vec::new();
    let mut pins = Vec::new();
    for (field, value) in record_of(expr, span)? {
        match field.as_str() {
            "body" => {
                let d = drawn(&value)?;
                let items = match d {
                    Drawn::List(items) => items,
                    one => vec![one],
                };
                for item in items {
                    let Drawn::Call { name: n, args, named, span } = item else {
                        return Err(diagnostic(value.span(), "a symbol's `body` is a list of drawings"));
                    };
                    graphics.push(graphic(&n, &args, &named, span, SYMBOL_WIDTH)?);
                }
            }
            "pins" => {
                let v = val(&value)?;
                pins = super::values::list(&v)?.iter().map(|p| pin_of(p, &name)).collect::<Read<_>>()?;
            }
            other => return Err(diagnostic(value.span(), format!("a symbol has no `{other}`; it takes body and pins"))),
        }
    }
    if pins.is_empty() {
        return Err(diagnostic(span, format!("symbol `{name}` needs its `pins`")));
    }
    let mut seen = std::collections::BTreeSet::new();
    for p in &pins {
        if !seen.insert(p.pin.as_str()) {
            return Err(diagnostic(span, format!("symbol `{name}` draws pin `{}` twice", p.pin)));
        }
    }
    Ok(Symbol { uuid: uuid("symbol", &name), unit: Uuid::NIL, name, pins, graphics })
}

/// The value written after `symbol <name> = `, which [`read`] reads back.
pub fn print(symbol: &Symbol) -> Result<String, PrintError> {
    let mut entries: Vec<(String, String)> = Vec::new();
    if !symbol.graphics.is_empty() {
        entries.push(("body".into(), graphics_text(&symbol.graphics, SYMBOL_WIDTH)));
    }
    let pins: Vec<String> = symbol
        .pins
        .iter()
        .map(|p| {
            let dir = DIRS.iter().find(|(_, a)| *a == p.angle.rem_euclid(360.0)).map(|(w, _)| *w).ok_or_else(|| {
                PrintError(format!("symbol `{}`: pin `{}` runs at {}°; a pin runs right, up, left or down", symbol.name, p.pin, p.angle))
            })?;
            let mut s = format!("{{ name = {}, at = {}, dir = {dir}", super::printer::name(&p.pin), pt_text(p.at));
            if p.length != mm(PIN_LENGTH) {
                s.push_str(&format!(", length = {}", mm_text(p.length)));
            }
            s.push_str(" }");
            Ok(s)
        })
        .collect::<Result<_, PrintError>>()?;
    if pins.is_empty() {
        return Err(PrintError(format!("symbol `{}` has no pins", symbol.name)));
    }
    entries.push(("pins".into(), format!("[\n{INDENT}{}\n]", pins.join(&format!(",\n{INDENT}")))));
    Ok(super::printer::record(&entries))
}

/// The whole statement, `symbol <name> = { … }`.
pub fn statement(name: &str, symbol: &Symbol) -> Result<String, PrintError> {
    Ok(format!("{} {} = {}", words::SYMBOL, super::printer::name(name), print(symbol)?))
}
