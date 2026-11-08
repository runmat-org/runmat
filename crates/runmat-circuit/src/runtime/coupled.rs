//! `board.diff-pair` and `board.tune-length` on a layout's text, through
//! [`crate::pair`]: the pair comes from the circuit's `diff_pair` declarations, and the tracks
//! written are the ones that pass DRC.

use crate::layer_set::CopperIndex;
use super::edit::{failure, EditError, EditOutput, EditRequest};
use super::*;
use crate::geom::mm;
use crate::layout::NetId;

/// The board nets of the circuit's declared pairs, as DRC, zone fills and rules take them.
pub(super) fn pair_nets(project: &Project, board: &Board) -> Vec<(NetId, NetId)> {
    pairs(project, board).into_iter().map(|(_, p, n)| (p, n)).collect()
}

/// The board nets of the circuit's declared pairs, by pair name.
fn pairs(project: &Project, board: &Board) -> Vec<(String, NetId, NetId)> {
    let nl = &project.netlist;
    nl.diff_pairs
        .iter()
        .filter_map(|d| Some((d.name.clone(), board.net(&nl.nets.get(d.p)?.name)?, board.net(&nl.nets.get(d.n)?.name)?)))
        .collect()
}

/// The net a `net:` pick (or the selection) names.
fn picked(request: &EditRequest, field: &str) -> Option<String> {
    let v = request.values.get(field);
    let s = match v {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(a)) => a.first().and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
    .filter(|s| !s.is_empty())
    .or_else(|| request.selection.first().cloned())?;
    Some(s.strip_prefix("net:").unwrap_or(&s).to_owned())
}

/// A copper layer by its stackup name, its file name (`F.Cu`, `In1.Cu`, `B.Cu`), or `Top` /
/// `Bottom`; none is the top.
fn layer_of(board: &Board, name: Option<&str>) -> Result<CopperIndex, String> {
    let names = board.stackup.copper_names();
    let n = names.len() as CopperIndex;
    match name.map(str::trim) {
        None | Some("") | Some("Top") | Some("top") => Ok(0),
        Some("Bottom") | Some("bottom") => Ok(n - 1),
        Some(l) => names
            .iter()
            .position(|x| x == l)
            .map(|i| i as CopperIndex)
            .or_else(|| (0..n).find(|i| crate::layout::copper_name(*i, n) == l))
            .ok_or_else(|| format!("`{l}` is not a copper layer of this board")),
    }
}

/// Runs `board.diff-pair`.
pub(super) fn diff_pair(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let values = &request.values;
    let net = picked(request, "pair").ok_or_else(|| EditError::new("pick a net of the pair"))?;
    let width = values.get("width").and_then(Value::as_f64).map(mm).unwrap_or(mm(0.12));
    let gap = values.get("gap").and_then(Value::as_f64).map(mm).unwrap_or(mm(0.15));
    let mut summary = String::new();
    let (change, ()) = board_edit::edit_board(&layout.path, &layout.text, |board| {
        let (name, p, n) = pairs(project, board)
            .into_iter()
            .find(|(name, p, n)| *name == net || [p, n].iter().any(|id| board.nets[id.0 as usize].name == net))
            .ok_or_else(|| format!("`{net}` is in no differential pair; declare the pair in the circuit with diff_pair"))?;
        let layer = layer_of(board, values.get("layer").and_then(Value::as_str))?;
        let r = crate::pair::route_pair(board, p, n, layer, width, gap)?;
        summary = format!(
            "Routed {name} on {}: {} {:.3} mm, {} {:.3} mm, skew {:.3} mm",
            board.stackup.copper_names()[layer as usize],
            board.nets[p.0 as usize].name,
            r.p_length,
            board.nets[n.0 as usize].name,
            r.n_length,
            r.skew()
        );
        Ok(())
    })
    .map_err(failure)?;
    Ok(EditOutput { summary, files: vec![file_change(&layout.path, &layout.text, change)] })
}

/// Runs `board.tune-length`.
pub(super) fn tune_length(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let values = &request.values;
    let net = picked(request, "net").ok_or_else(|| EditError::new("pick the net to tune"))?;
    let target = values.get("target").and_then(Value::as_f64).ok_or_else(|| EditError::new("give the target length"))?;
    let tolerance = values.get("tolerance").and_then(Value::as_f64).unwrap_or(0.1);
    let mut summary = String::new();
    let (change, ()) = board_edit::edit_board(&layout.path, &layout.text, |board| {
        let id = board.net(&net).ok_or_else(|| format!("`{net}` is not a net on the board"))?;
        let pairs = pair_nets(project, board);
        let t = crate::pair::tune_length(board, id, target, tolerance, &pairs)?;
        summary = format!(
            "Tuned {net} from {:.3} mm to {:.3} mm with {} meander(s) {:.3} mm tall",
            t.before, t.after, t.bumps, t.amplitude
        );
        Ok(())
    })
    .map_err(failure)?;
    Ok(EditOutput { summary, files: vec![file_change(&layout.path, &layout.text, change)] })
}
