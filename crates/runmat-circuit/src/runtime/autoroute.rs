//! `board.fanout` and `board.autoroute` on a layout's text: the routers of
//! [`crate::autoroute`] run on the board model, and the new copper is written as statements.

use super::edit::{failure, EditError, EditOutput, EditRequest};
use super::*;
use crate::autoroute::{self, FanoutOptions, RouteOptions};
use crate::geom::{mm, Coord};
use crate::layout::NetId;

/// Runs `board.fanout` or `board.autoroute`.
pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let values = &request.values;
    let mut summary = String::new();
    let (change, ()) = board_edit::edit_board(&layout.path, &layout.text, |board| {
        let clearance = clearance(board);
        let widths = widths(board);
        let default_width = mm_of(values, "width").unwrap_or(mm(0.2));
        let via = via_of(values, "via").unwrap_or((mm(0.45), mm(0.2)));
        let planes = plane_nets(board);
        match request.command.as_str() {
            "board.fanout" => {
                let picked = picks(values, "component", &request.selection, "component:");
                if picked.is_empty() {
                    return Err("pick the component to fan out".into());
                }
                let mut footprints = Vec::new();
                for r in &picked {
                    let f = board
                        .footprint_by_ref(r)
                        .ok_or_else(|| format!("`{r}` is not on the board"))?;
                    footprints.push(f.id);
                }
                let mut nets: BTreeSet<NetId> =
                    net_picks(board, values, "nets")?.into_iter().collect();
                if nets.is_empty() {
                    nets = planes.clone();
                }
                if values.get("signals").and_then(Value::as_bool) == Some(true) {
                    for id in &footprints {
                        nets.extend(board.footprints[id].pad_nets.values().copied());
                    }
                }
                let report = autoroute::fanout(
                    board,
                    &FanoutOptions {
                        footprints,
                        nets,
                        via,
                        widths,
                        default_width,
                        clearance,
                    },
                );
                summary = format!(
                    "Fanned out {} pad(s) of {}{}{}",
                    report.fanned,
                    picked.join(", "),
                    if report.kept > 0 {
                        format!("; {} already had a via", report.kept)
                    } else {
                        String::new()
                    },
                    if report.blocked.is_empty() {
                        String::new()
                    } else {
                        format!("; no room for a via at {}", report.blocked.join(", "))
                    }
                );
                Ok(())
            }
            _ => {
                let mut nets = net_picks(board, values, "nets")?;
                if nets.is_empty() {
                    nets = (0..board.nets.len() as u32)
                        .map(NetId)
                        .filter(|n| !planes.contains(n))
                        .collect();
                }
                let layers = match values.get("layers").and_then(Value::as_str) {
                    Some(text) if !text.trim().is_empty() => text
                        .split(',')
                        .map(|l| layer_index(board, l.trim()))
                        .collect::<Result<Vec<u8>, String>>()?,
                    _ => routing_layers(board),
                };
                let mut opts = RouteOptions::new(nets, layers, clearance);
                opts.widths = widths;
                opts.default_width = default_width;
                opts.via = via;
                // Airwires as the board shows them: poured copper counts, so a pad already on a
                // plane through its fanout via has none.
                let mut filled = board.clone();
                let (mut rules, _) = BoardRules::new(&filled.limits, &filled.rules);
                rules.rebuild(&BoardGraph::new(&filled, &[]));
                crate::zone::fill_zones(&mut filled, &rules, &[]);
                let wires = crate::route::ratsnest(&filled);
                let mut space = autoroute::Space::new(&filled, clearance);
                let report = autoroute::autoroute_in(board, &mut space, &opts, wires);
                let total = report.routed as usize + report.failed.len();
                summary = format!(
                    "Routed {} of {} connection(s) with {} track(s) and {} via(s){}",
                    report.routed,
                    total,
                    report.tracks,
                    report.vias,
                    if report.failed.is_empty() {
                        String::new()
                    } else {
                        let names: BTreeSet<&str> =
                            report.failed.iter().map(|f| f.0.as_str()).collect();
                        format!(
                            "; left unrouted: {}",
                            names.into_iter().collect::<Vec<_>>().join(", ")
                        )
                    }
                );
                Ok(())
            }
        }
    })
    .map_err(failure)?;
    Ok(EditOutput {
        summary,
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

/// The largest clearance the board's rules can require, plus 5 µm of rounding margin.
fn clearance(board: &Board) -> Coord {
    let (rules, _) = BoardRules::new(&board.limits, &board.rules);
    rules.max_clearance().max(board.limits.min_clearance) + 5_000
}

/// Track width per net from its net class's `track_width`.
fn widths(board: &Board) -> BTreeMap<NetId, Coord> {
    let mut out = BTreeMap::new();
    for (i, n) in board.nets.iter().enumerate() {
        let Some(class) = &n.class else { continue };
        let width = board.net_class_rules.get(class).and_then(|decls| {
            decls
                .iter()
                .find(|(k, _)| k == crate::board_rules::props::TRACK_WIDTH)
                .and_then(|(_, v)| v.as_len())
        });
        if let Some(w) = width {
            out.insert(NetId(i as u32), w);
        }
    }
    out
}

/// Nets poured on an inner copper layer: they reach their pads through fanout vias.
fn plane_nets(board: &Board) -> BTreeSet<NetId> {
    let n = board.copper_count();
    board
        .zones
        .values()
        .filter(|z| z.layers.iter().any(|l| *l > 0 && *l + 1 < n))
        .filter_map(|z| z.net)
        .collect()
}

/// The outer layers, then inner layers with no pour on them.
fn routing_layers(board: &Board) -> Vec<u8> {
    let n = board.copper_count();
    let poured: BTreeSet<u8> = board
        .zones
        .values()
        .flat_map(|z| z.layers.iter().copied())
        .collect();
    let mut out = vec![0];
    if n > 1 {
        out.push(n - 1);
    }
    out.extend((1..n.saturating_sub(1)).filter(|l| !poured.contains(l)));
    out
}

fn layer_index(board: &Board, name: &str) -> Result<u8, String> {
    let n = board.copper_count();
    (0..n)
        .find(|i| crate::layout::copper_name(*i, n) == name)
        .ok_or_else(|| format!("`{name}` is not a copper layer of this board"))
}

fn mm_of(values: &serde_json::Map<String, Value>, id: &str) -> Option<Coord> {
    values.get(id).and_then(Value::as_f64).map(mm)
}

/// A via given as `drill/diameter` (either order; the smaller is the drill), in mm.
fn via_of(values: &serde_json::Map<String, Value>, id: &str) -> Option<(Coord, Coord)> {
    let text = values.get(id)?.as_str()?;
    let (a, b) = text.split_once('/')?;
    let (a, b) = (a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?);
    Some((mm(a.max(b)), mm(a.min(b))))
}

/// Picked ids with `prefix` stripped, from the field or else the selection.
fn picks(
    values: &serde_json::Map<String, Value>,
    field: &str,
    selection: &[String],
    prefix: &str,
) -> Vec<String> {
    let listed: Vec<String> = match values.get(field) {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_owned).collect(),
        _ => Vec::new(),
    };
    let ids = if listed.is_empty() {
        selection.to_vec()
    } else {
        listed
    };
    ids.iter()
        .filter_map(|id| {
            id.strip_prefix(prefix)
                .map(str::to_owned)
                .or_else(|| (!id.contains(':')).then(|| id.clone()))
        })
        .collect()
}

fn net_picks(
    board: &Board,
    values: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Vec<NetId>, String> {
    picks(values, field, &[], "net:")
        .iter()
        .map(|n| {
            board
                .net(n)
                .ok_or_else(|| format!("`{n}` is not a net on the board"))
        })
        .collect()
}
