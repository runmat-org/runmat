//! How the views, the selection and edit requests name one design item.
//!
//! A reference is `<kind>:<key>` text, the same string on every side of the wasm boundary:
//!
//! | Text | Item |
//! |---|---|
//! | `board:outline` | the board outline |
//! | `net:GND` | a net, by name |
//! | `component:U1` | a placed component, by designator |
//! | `pad:U1.3` | a footprint pad: designator, then pad number |
//! | `pin:U1.P0.1` | a schematic pin: designator, then pin name (which may hold dots) |
//! | `track:12`, `via:4`, `zone:2` | board copper, by item id |
//! | `room:power` | a placement room, by name |
//! | `sheet:/power` | a schematic sheet, by instance path |
//! | `text:/power/note` | a schematic note: its sheet path, then its name |
//! | `point:12.5,-3` | a picked point, millimetres |
//!
//! [`ItemRef`] owns this grammar: producers print with [`std::fmt::Display`] and edits read with
//! [`ItemRef::parse`] or [`ItemRef::read`], so no module formats or strips a prefix by hand.
//! [`ItemRef::net_on`] and [`ItemRef::centre_on`] resolve a reference against a board.

use crate::geom::Point;
use crate::layout::{Board, ItemId, NetId};
use std::fmt;

/// One design item, as a view picks it and an edit request names it.
#[derive(Clone, Debug, PartialEq)]
pub enum ItemRef {
    BoardOutline,
    Net(String),
    Component(String),
    Pad { component: String, pad: String },
    Pin { component: String, pin: String },
    Track(u64),
    Via(u64),
    Zone(u64),
    Room(String),
    Sheet(String),
    Text { sheet: String, name: String },
    Point { x_mm: f64, y_mm: f64 },
}

impl ItemRef {
    /// The reference `text` spells, or `None` when it has no known kind or a malformed key.
    pub fn parse(text: &str) -> Option<ItemRef> {
        let (kind, key) = text.split_once(':')?;
        let pair = |key: &str| -> Option<(String, String)> {
            let (a, b) = key.split_once('.')?;
            (!a.is_empty() && !b.is_empty()).then(|| (a.to_owned(), b.to_owned()))
        };
        let named = |key: &str| (!key.is_empty()).then(|| key.to_owned());
        Some(match kind {
            "board" if key == "outline" => ItemRef::BoardOutline,
            "net" => ItemRef::Net(named(key)?),
            "component" => ItemRef::Component(named(key)?),
            "pad" => {
                let (component, pad) = pair(key)?;
                ItemRef::Pad { component, pad }
            }
            "pin" => {
                let (component, pin) = pair(key)?;
                ItemRef::Pin { component, pin }
            }
            "track" => ItemRef::Track(key.parse().ok()?),
            "via" => ItemRef::Via(key.parse().ok()?),
            "zone" => ItemRef::Zone(key.parse().ok()?),
            "room" => ItemRef::Room(named(key)?),
            "sheet" => ItemRef::Sheet(named(key)?),
            "text" => {
                let (sheet, name) = key.rsplit_once('/')?;
                ItemRef::Text {
                    sheet: sheet.to_owned(),
                    name: named(name)?,
                }
            }
            "point" => {
                let (x, y) = key.split_once(',')?;
                let (x_mm, y_mm): (f64, f64) = (x.trim().parse().ok()?, y.trim().parse().ok()?);
                if !(x_mm.is_finite() && y_mm.is_finite()) {
                    return None;
                }
                ItemRef::Point { x_mm, y_mm }
            }
            _ => return None,
        })
    }

    /// A field value that takes a reference or a bare name: `text` parsed, or, when it has no
    /// `:`, the item `bare` makes of it (`ItemRef::Net` for a net field).
    pub fn read(text: &str, bare: fn(String) -> ItemRef) -> Option<ItemRef> {
        if text.is_empty() {
            return None;
        }
        match ItemRef::parse(text) {
            Some(r) => Some(r),
            None if !text.contains(':') => Some(bare(text.to_owned())),
            None => None,
        }
    }

    /// A picked point in board or page millimetres.
    pub fn point(x_mm: f64, y_mm: f64) -> ItemRef {
        ItemRef::Point { x_mm, y_mm }
    }

    /// The net's name when this is a net reference.
    pub fn net(&self) -> Option<&str> {
        match self {
            ItemRef::Net(n) => Some(n),
            _ => None,
        }
    }

    /// The designator when this is a component reference.
    pub fn component(&self) -> Option<&str> {
        match self {
            ItemRef::Component(c) => Some(c),
            _ => None,
        }
    }

    /// The point in millimetres when this is a point pick.
    pub fn xy_mm(&self) -> Option<(f64, f64)> {
        match self {
            ItemRef::Point { x_mm, y_mm } => Some((*x_mm, *y_mm)),
            _ => None,
        }
    }

    /// Whether the item is one that carries a net: a net, pad, track, via or pour.
    pub fn carries_net(&self) -> bool {
        matches!(
            self,
            ItemRef::Net(_)
                | ItemRef::Pad { .. }
                | ItemRef::Track(_)
                | ItemRef::Via(_)
                | ItemRef::Zone(_)
        )
    }

    /// The net this item carries on `board`: a net by name, or the net of a picked pad, track,
    /// via or zone.
    pub fn net_on(&self, board: &Board) -> Option<NetId> {
        match self {
            ItemRef::Net(name) => board.net(name),
            ItemRef::Pad { component, pad } => board
                .footprint_by_ref(component)?
                .pad_nets
                .get(pad)
                .copied(),
            ItemRef::Track(id) => board.tracks.get(&ItemId(*id))?.net,
            ItemRef::Via(id) => board.vias.get(&ItemId(*id))?.net,
            ItemRef::Zone(id) => board.zones.get(&ItemId(*id))?.net,
            _ => None,
        }
    }

    /// Where the item sits on `board`, for dimensions and moves between items.
    pub fn centre_on(&self, board: &Board) -> Option<Point> {
        match self {
            ItemRef::Point { x_mm, y_mm } => Some(Point::mm(*x_mm, *y_mm)),
            ItemRef::Component(r) => board.footprint_by_ref(r).map(|f| f.at),
            ItemRef::Pad { component, pad } => {
                let f = board.footprint_by_ref(component)?;
                let i = f.package.pads.iter().position(|p| p.name == *pad)?;
                f.pad(i, board.copper_count()).map(|p| p.center)
            }
            ItemRef::Via(id) => board.vias.get(&ItemId(*id)).map(|v| v.at),
            ItemRef::Track(id) => board.tracks.get(&ItemId(*id)).map(|t| t.endpoints().0),
            _ => None,
        }
    }
}

impl fmt::Display for ItemRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ItemRef::BoardOutline => f.write_str("board:outline"),
            ItemRef::Net(n) => write!(f, "net:{n}"),
            ItemRef::Component(c) => write!(f, "component:{c}"),
            ItemRef::Pad { component, pad } => write!(f, "pad:{component}.{pad}"),
            ItemRef::Pin { component, pin } => write!(f, "pin:{component}.{pin}"),
            ItemRef::Track(id) => write!(f, "track:{id}"),
            ItemRef::Via(id) => write!(f, "via:{id}"),
            ItemRef::Zone(id) => write!(f, "zone:{id}"),
            ItemRef::Room(r) => write!(f, "room:{r}"),
            ItemRef::Sheet(s) => write!(f, "sheet:{s}"),
            ItemRef::Text { sheet, name } => write!(f, "text:{sheet}/{name}"),
            ItemRef::Point { x_mm, y_mm } => write!(f, "point:{},{}", trim(*x_mm), trim(*y_mm)),
        }
    }
}

/// A millimetre value to a micrometre, without trailing zeros (`12.5`, `-3`, `0`).
fn trim(v: f64) -> String {
    let s = format!("{:.6}", (v * 1.0e6).round() / 1.0e6);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".to_owned()
    } else {
        s.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_round_trips_through_its_text() {
        for text in [
            "board:outline",
            "net:GND",
            "component:U1",
            "pad:U1.3",
            "pin:U1.P0.1",
            "track:12",
            "via:4",
            "zone:2",
            "room:power",
            "sheet:/power",
            "text:/power/note",
            "text:/note",
            "text://note",
            "point:12.5,-3",
        ] {
            let r = ItemRef::parse(text).unwrap_or_else(|| panic!("{text} parses"));
            assert_eq!(r.to_string(), text);
        }
    }

    #[test]
    fn a_pin_name_keeps_its_dots_and_a_point_trims_to_a_micrometre() {
        assert_eq!(
            ItemRef::parse("pin:U1.P0.1"),
            Some(ItemRef::Pin {
                component: "U1".into(),
                pin: "P0.1".into()
            })
        );
        assert_eq!(ItemRef::point(1.0000004, -0.0).to_string(), "point:1,0");
    }

    #[test]
    fn malformed_and_unknown_text_reads_as_nothing() {
        for text in [
            "track:x",
            "pad:U1",
            "net:",
            "point:1",
            "point:a,b",
            "wire:3",
            "GND",
        ] {
            assert_eq!(ItemRef::parse(text), None, "{text}");
        }
    }

    #[test]
    fn a_bare_name_takes_the_fields_kind() {
        assert_eq!(
            ItemRef::read("GND", ItemRef::Net),
            Some(ItemRef::Net("GND".into()))
        );
        assert_eq!(
            ItemRef::read("track:7", ItemRef::Net),
            Some(ItemRef::Track(7))
        );
        assert_eq!(ItemRef::read("wire:7", ItemRef::Net), None);
        assert_eq!(ItemRef::read("", ItemRef::Net), None);
    }
}
