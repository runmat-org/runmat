//! `board.fanout`, `board.autoroute` and `board.route-net` on a layout's text: the routers of
//! [`crate::autoroute`] run on the board model, and the new copper is written as statements.
//! Route net is a net's right-click command: the router takes the selected nets' open
//! connections only.

use super::edit::{failure, EditError, EditOutput, EditRequest};
use super::picks;
use super::*;
use crate::autoroute::{self, FanoutOptions, RouteOptions};
use crate::geom::{mm, Coord};
use crate::item_ref::ItemRef;
use crate::layer_set::CopperIndex;
use crate::layout::NetId;

/// Runs `board.fanout`, `board.autoroute` or `board.route-net`.
pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let values = &request.values;
    let mut summary = String::new();
    let (change, ()) = board_edit::edit_board(&layout.path, &layout.text, |board| {
        let clearance = clearance(board);
        let widths = widths(board);
        let default_width = mm_of(values, "width").unwrap_or(mm(0.2));
        let via = via_of(values, "via").unwrap_or_else(|| default_via(&board.limits));
        let planes = plane_nets(board);
        match request.command.as_str() {
            "board.fanout" => {
                let picked: Vec<String> =
                    picks::picks(values, &request.selection, "component", ItemRef::Component)
                        .into_iter()
                        .filter_map(|r| r.component().map(str::to_owned))
                        .collect();
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
                    net_picks(board, values, "nets", &[])?.into_iter().collect();
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
                let route_net = request.command == "board.route-net";
                // Route net takes the nets its menu was opened on; Autoroute only its field.
                let selection: &[String] = if route_net { &request.selection } else { &[] };
                let mut nets = net_picks(board, values, "nets", selection)?;
                if route_net && nets.is_empty() {
                    return Err("select the net to route".into());
                }
                // Every net by default: the airwires come from the filled board, so a plane
                // net only has the ones its pours leave (a pad off the plane, an island).
                if nets.is_empty() {
                    nets = (0..board.nets.len() as u32).map(NetId).collect();
                }
                let layers = match values.get("layers").and_then(Value::as_str) {
                    Some(text) if !text.trim().is_empty() => text
                        .split(',')
                        .map(|l| layer_index(board, l.trim()))
                        .collect::<Result<Vec<CopperIndex>, String>>()?,
                    _ => routing_layers(board),
                };
                let mut opts = RouteOptions::new(nets, layers, clearance);
                opts.layer_widths = layer_widths(board, &widths, default_width);
                opts.widths = widths;
                opts.default_width = default_width;
                opts.via = via;
                // Airwires as the board shows them: poured copper counts, so a pad already on a
                // plane through its fanout via has none. New tracks can cut a pad's only spoke
                // into a pour, so the pours refill after each pass and the airwires that opens
                // are routed in the next (at most four passes, while a pass routes something).
                let mut report = autoroute::RouteReport::default();
                for _ in 0..4 {
                    let mut filled = board.clone();
                    let (mut rules, _) = BoardRules::for_board(&filled);
                    rules.rebuild(&BoardGraph::new(&filled, &[]));
                    crate::zone::fill_zones(&mut filled, &rules, &[]);
                    let wires = crate::route::ratsnest(&filled);
                    let mut space = autoroute::Space::new(&filled, clearance);
                    let pass = autoroute::autoroute_in(board, &mut space, &opts, wires);
                    report.routed += pass.routed;
                    report.tracks += pass.tracks;
                    report.vias += pass.vias;
                    report.failed = pass.failed;
                    if pass.routed == 0 {
                        break;
                    }
                }
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
                        let mut why: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
                        for f in &report.failed {
                            why.entry(f.net.as_str()).or_default().insert(f.why.text());
                        }
                        let parts: Vec<String> = why
                            .into_iter()
                            .map(|(net, w)| {
                                let w: Vec<&str> = w.into_iter().collect();
                                format!("{net} ({})", w.join(", "))
                            })
                            .collect();
                        format!("; left unrouted: {}", parts.join(", "))
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
    let (rules, _) = BoardRules::for_board(board);
    rules.max_clearance().max(board.limits.min_clearance) + 5_000
}

/// Track width per net from its net class's `track_width`.
fn widths(board: &Board) -> BTreeMap<NetId, Coord> {
    let mut out = BTreeMap::new();
    for (i, n) in board.nets.iter().enumerate() {
        let Some(class) = &n.class else { continue };
        let width = board.net_class_rules.get(class).and_then(|c| {
            c.decls
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

/// Per-layer track widths of the nets whose class has an impedance target: each copper
/// layer's width from the board's rules ([`BoardRules::width_on`]), where it differs by layer.
fn layer_widths(
    board: &Board,
    widths: &BTreeMap<NetId, Coord>,
    default_width: Coord,
) -> BTreeMap<NetId, Vec<Coord>> {
    let classes: BTreeSet<&str> = board
        .net_class_rules
        .iter()
        .filter(|(_, c)| crate::high_speed::ImpedanceTarget::of(&c.decls).is_some())
        .map(|(k, _)| k.as_str())
        .collect();
    if classes.is_empty() {
        return BTreeMap::new();
    }
    let (rules, _) = BoardRules::for_board(board);
    let n = board.copper_count();
    let mut out = BTreeMap::new();
    for (i, net) in board.nets.iter().enumerate() {
        if !net.class.as_deref().is_some_and(|c| classes.contains(c)) {
            continue;
        }
        let id = NetId(i as u32);
        let fallback = widths.get(&id).copied().unwrap_or(default_width);
        out.insert(
            id,
            (0..n)
                .map(|l| rules.width_on(board, Some(id), l, fallback))
                .collect(),
        );
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
fn routing_layers(board: &Board) -> Vec<CopperIndex> {
    let n = board.copper_count();
    let poured: BTreeSet<CopperIndex> = board
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

fn layer_index(board: &Board, name: &str) -> Result<CopperIndex, String> {
    let n = board.copper_count();
    (0..n)
        .find(|i| crate::layout::copper_name(*i, n) == name)
        .ok_or_else(|| format!("`{name}` is not a copper layer of this board"))
}

fn mm_of(values: &serde_json::Map<String, Value>, id: &str) -> Option<Coord> {
    values.get(id).and_then(Value::as_f64).map(mm)
}

/// A via given as `drill/diameter` (either order; the smaller is the drill), in mm.
/// The via the routers place when the request names none: 0.45 mm over a 0.2 mm drill, grown to
/// the board's limits (drill, via diameter and annular ring) so new vias pass DRC.
fn default_via(limits: &crate::layout::FabLimits) -> (Coord, Coord) {
    let drill = mm(0.2).max(limits.min_drill);
    let diameter = mm(0.45)
        .max(limits.min_via_diameter)
        .max(drill + 2 * limits.min_annular);
    (diameter, drill)
}

fn via_of(values: &serde_json::Map<String, Value>, id: &str) -> Option<(Coord, Coord)> {
    let text = values.get(id)?.as_str()?;
    let (a, b) = text.split_once('/')?;
    let (a, b) = (a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?);
    Some((mm(a.max(b)), mm(a.min(b))))
}

fn net_picks(
    board: &Board,
    values: &serde_json::Map<String, Value>,
    field: &str,
    selection: &[String],
) -> Result<Vec<NetId>, String> {
    picks::picks(values, selection, field, ItemRef::Net)
        .iter()
        .filter(|pick| pick.carries_net())
        .map(|pick| picks::net_on(board, pick))
        .collect()
}
