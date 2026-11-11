//! `board.diff-pair` and `board.tune-length` on a layout's text, through
//! [`crate::pair`]: the pair comes from the circuit's `diff_pair` declarations, and the tracks
//! written are the ones that pass DRC.

use super::edit::{failure, EditError, EditOutput, EditRequest};
use super::*;
use super::picks;
use crate::geom::mm;
use crate::item_ref::ItemRef;
use crate::layer_set::CopperIndex;
use crate::layout::NetId;

/// The board nets of the circuit's declared pairs, as DRC, zone fills and rules take them.
pub(super) fn pair_nets(project: &Project, board: &Board) -> Vec<(NetId, NetId)> {
    pairs(project, board)
        .into_iter()
        .map(|(_, p, n)| (p, n))
        .collect()
}

/// The board nets of the circuit's declared pairs, by pair name.
fn pairs(project: &Project, board: &Board) -> Vec<(String, NetId, NetId)> {
    let nl = &project.netlist;
    nl.diff_pairs
        .iter()
        .filter_map(|d| {
            Some((
                d.name.clone(),
                board.net(&nl.nets.get(d.p)?.name)?,
                board.net(&nl.nets.get(d.n)?.name)?,
            ))
        })
        .collect()
}

/// The pick in `field` (or the selection): a net by name, or a pad, track or via on one.
fn picked(request: &EditRequest, field: &str) -> Option<ItemRef> {
    picks::first(&request.values, &request.selection, field, ItemRef::Net)
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
    let pick = picked(request, "pair").ok_or_else(|| EditError::new("pick a net of the pair"))?;
    let width = values.get("width").and_then(Value::as_f64).map(mm);
    let gap = values.get("gap").and_then(Value::as_f64).map(mm);
    let mut summary = String::new();
    let (change, ()) = board_edit::edit_board(&layout.path, &layout.text, |board| {
        let net = picks::net_name_on(board, &pick)?;
        let (name, p, n) = pairs(project, board)
            .into_iter()
            .find(|(name, p, n)| *name == net || [p, n].iter().any(|id| board.nets[id.0 as usize].name == net))
            .ok_or_else(|| format!("`{net}` is in no differential pair; declare the pair in the circuit with diff_pair"))?;
        let layer = layer_of(board, values.get("layer").and_then(Value::as_str))?;
        // Empty fields: the pair's width and gap on this layer from its rules (a class's
        // impedance target solves them per layer).
        let (rule_w, rule_s) = crate::board_rules::BoardRules::for_board(board).0.pair_on(board, p, layer);
        let width = width.or(rule_w).unwrap_or(mm(0.12));
        let gap = gap.or(rule_s).unwrap_or(mm(0.15));
        let r = crate::pair::route_pair(board, p, n, layer, width, gap)?;
        summary = format!(
            "Routed {name} on {} at {:.3} mm width, {:.3} mm gap: {} {:.3} mm, {} {:.3} mm, skew {:.3} mm",
            board.stackup.copper_names()[layer as usize],
            width as f64 / 1e6,
            gap as f64 / 1e6,
            board.nets[p.0 as usize].name,
            r.p_length,
            board.nets[n.0 as usize].name,
            r.n_length,
            r.skew()
        );
        Ok(())
    })
    .map_err(failure)?;
    Ok(EditOutput {
        summary,
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

/// The meander pattern a tuning command asks for (trombone when empty).
fn pattern_of(values: &serde_json::Map<String, Value>) -> Result<crate::pair::Pattern, EditError> {
    match values
        .get("pattern")
        .and_then(Value::as_str)
        .filter(|p| !p.is_empty())
    {
        None => Ok(crate::pair::Pattern::Trombone),
        Some(p) => crate::pair::Pattern::parse(p).ok_or_else(|| {
            EditError::new(format!(
                "`{p}` is not a pattern; use trombone, accordion or sawtooth"
            ))
        }),
    }
}

/// A tuning's summary line.
fn tuned_text(t: &crate::high_speed::tuning::Tuned, pattern: crate::pair::Pattern) -> String {
    let unit = t.basis.unit();
    format!(
        "Tuned {} from {:.3} {unit} to {:.3} {unit} with {} {} meander(s) {:.3} mm tall",
        t.net,
        t.before,
        t.after,
        t.tuning.bumps,
        pattern.word(),
        t.tuning.amplitude
    )
}

/// Runs `board.tune-length`: a length target in mm, or a delay target (the time field, in
/// seconds) in ps.
pub(super) fn tune_length(
    project: &Project,
    request: &EditRequest,
) -> Result<EditOutput, EditError> {
    use crate::high_speed::tuning::{tune_net, Target};
    let layout = project.layout_or_err()?;
    let values = &request.values;
    let pick = picked(request, "net").ok_or_else(|| EditError::new("pick the net to tune"))?;
    let pattern = pattern_of(values)?;
    let delay = values.get("basis").and_then(Value::as_str) == Some("delay");
    let (target, tolerance) = if delay {
        let ps = values
            .get("target_delay")
            .and_then(Value::as_f64)
            .ok_or_else(|| EditError::new("give the target delay"))?
            * 1e12;
        let tol = values
            .get("tolerance_delay")
            .and_then(Value::as_f64)
            .map_or(1.0, |s| s * 1e12);
        (Target::Delay(ps), tol)
    } else {
        let mm = values
            .get("target")
            .and_then(Value::as_f64)
            .ok_or_else(|| EditError::new("give the target length"))?;
        (
            Target::Length(mm),
            values
                .get("tolerance")
                .and_then(Value::as_f64)
                .unwrap_or(0.1),
        )
    };
    let mut summary = String::new();
    let (change, ()) = board_edit::edit_board(&layout.path, &layout.text, |board| {
        let id = picks::net_on(board, &pick)?;
        let pairs = pair_nets(project, board);
        let t = tune_net(board, id, target, tolerance, pattern, &pairs)?;
        summary = tuned_text(&t, pattern);
        Ok(())
    })
    .map_err(failure)?;
    Ok(EditOutput {
        summary,
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

/// Runs `board.tune-group`: every short member of a match group to the group's target.
pub(super) fn tune_group(
    project: &Project,
    request: &EditRequest,
) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let values = &request.values;
    let group = values
        .get("group")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|g| !g.is_empty())
        .ok_or_else(|| EditError::new("name the match group to tune"))?
        .to_owned();
    let pattern = pattern_of(values)?;
    let mut summary = String::new();
    let (change, ()) = board_edit::edit_board(&layout.path, &layout.text, |board| {
        let pairs = pair_nets(project, board);
        let results = crate::high_speed::tuning::tune_group(board, &group, pattern, &pairs)?;
        if results.is_empty() {
            return Err(format!("no member of {group} is short of its target"));
        }
        let (done, failed): (Vec<_>, Vec<_>) = results.into_iter().partition(Result::is_ok);
        if done.is_empty() {
            return Err(failed
                .into_iter()
                .filter_map(Result::err)
                .collect::<Vec<_>>()
                .join("; "));
        }
        let mut lines: Vec<String> = done
            .into_iter()
            .filter_map(Result::ok)
            .map(|t| tuned_text(&t, pattern))
            .collect();
        lines.extend(
            failed
                .into_iter()
                .filter_map(Result::err)
                .map(|e| format!("not tuned: {e}")),
        );
        summary = format!("Tuned {group}: {}", lines.join("; "));
        Ok(())
    })
    .map_err(failure)?;
    Ok(EditOutput {
        summary,
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}
