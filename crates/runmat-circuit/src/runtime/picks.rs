//! What an edit request picks: a field's values, else the selection, read as [`ItemRef`]s.
//!
//! A field takes item references (`net:GND`, `track:12`, `component:U1`) or bare names, which
//! read as the field's own kind. Every edit reads its picks here, so each net field accepts the
//! same picks: a net by name, or a pad, track, via or pour that carries it.

use crate::item_ref::ItemRef;
use crate::layout::{Board, NetId};
use serde_json::{Map, Value};

/// The strings field `field` holds: one string, or an array's strings.
pub(super) fn listed(values: &Map<String, Value>, field: &str) -> Vec<String> {
    match values.get(field) {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

/// The picks in `field`, or the selection's when the field is empty. A bare name reads as
/// `bare` makes it; text that names no item is left out.
pub(super) fn picks(
    values: &Map<String, Value>,
    selection: &[String],
    field: &str,
    bare: fn(String) -> ItemRef,
) -> Vec<ItemRef> {
    let mut ids = listed(values, field);
    ids.retain(|s| !s.is_empty());
    if ids.is_empty() {
        ids = selection.to_vec();
    }
    ids.iter().filter_map(|s| ItemRef::read(s, bare)).collect()
}

/// The first pick in `field`, or the first selected item.
pub(super) fn first(
    values: &Map<String, Value>,
    selection: &[String],
    field: &str,
    bare: fn(String) -> ItemRef,
) -> Option<ItemRef> {
    picks(values, selection, field, bare).into_iter().next()
}

/// The board net a pick carries: a net by name, or the net of a picked pad, track, via or pour.
pub(super) fn net_on(board: &Board, pick: &ItemRef) -> Result<NetId, String> {
    match pick {
        ItemRef::Net(name) => board
            .net(name)
            .ok_or_else(|| format!("`{name}` is not a net on the board")),
        other => other
            .net_on(board)
            .ok_or_else(|| format!("`{other}` is on no net on the board")),
    }
}

/// The net's name a pick gives on `board`: a net reference keeps its own text (a pair's name
/// reads this way too), any other pick gives the name of the net it carries.
pub(super) fn net_name_on(board: &Board, pick: &ItemRef) -> Result<String, String> {
    match pick {
        ItemRef::Net(name) => Ok(name.clone()),
        other => net_on(board, other).map(|n| board.net_name(n).to_owned()),
    }
}
