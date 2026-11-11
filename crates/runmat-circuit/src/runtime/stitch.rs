//! `board.stitch` on a layout's text: stitching vias through a pour, or a row of vias either
//! side of a net's tracks (a shield), written as via statements.
//!
//! Candidates come from geometry: a stitching via sits on the pour's grid at the pitch, wholly
//! inside the pour's filled copper on each of its layers; a shield via sits beside each track
//! of the shielded net, one track half-width plus the clearance plus the via's radius off its
//! centreline, and inside a pour of the shield net, so it joins that net. The board's own DRC
//! then decides: the vias are added, the pours refilled, and every new via a finding names
//! (clearance to other copper on any layer, the board edge, hole-to-hole, keepouts) is taken
//! out, until a check finds none of them. So the vias written are the ones that pass DRC.

use super::edit::{failure, EditError, EditOutput, EditRequest};
use super::*;
use crate::board_rules::BItem;
use crate::geom::poly::circle_ring;
use crate::geom::{mm, Coord, Point};
use crate::layout::NetId;

/// Runs `board.stitch`.
pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let values = &request.values;
    let mode = values
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("stitch")
        .to_owned();
    let pitch = values
        .get("pitch")
        .and_then(Value::as_f64)
        .map(mm)
        .unwrap_or(mm(1.5));
    if pitch <= 0 {
        return Err(EditError::new("the pitch must be more than zero"));
    }
    let (diameter, drill) = via_of(
        values
            .get("via")
            .and_then(Value::as_str)
            .unwrap_or("0.3/0.6"),
    )
    .ok_or_else(|| EditError::new("write the via as drill/diameter in mm, such as 0.3/0.6"))?;
    let target = values
        .get("target")
        .and_then(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Array(a) => a.first().and_then(Value::as_str).map(str::to_owned),
            _ => None,
        })
        .filter(|s| !s.is_empty())
        .or_else(|| request.selection.first().cloned())
        .ok_or_else(|| {
            EditError::new(if mode == "shield" {
                "pick the net to shield"
            } else {
                "pick the pour to stitch"
            })
        })?;
    let shield_net = values
        .get("shield_net")
        .and_then(Value::as_str)
        .unwrap_or("GND")
        .to_owned();
    let mut summary = String::new();
    let (change, ()) = board_edit::edit_board(&layout.path, &layout.text, |board| {
        let (net, candidates, what) = if mode == "shield" {
            let net = net_named(board, target.strip_prefix("net:").unwrap_or(&target))?;
            let shield = net_named(board, &shield_net)?;
            if shield == net {
                return Err("a net can't shield itself; name another shield net".into());
            }
            (
                shield,
                shield_sites(board, net, shield, pitch, diameter),
                format!("beside {}", board.nets[net.0 as usize].name),
            )
        } else {
            let zone = zone_named(board, target.strip_prefix("zone:").unwrap_or(&target))?;
            let z = &board.zones[&zone];
            let net = z
                .net
                .ok_or("the pour has no net; stitching vias join a pour to its net")?;
            let names = board.stackup.copper_names();
            let layers: Vec<&str> = z
                .layers
                .iter()
                .filter_map(|l| names.get(*l as usize).map(String::as_str))
                .collect();
            let what = format!(
                "in the {} pour on {}",
                board.nets[net.0 as usize].name,
                layers.join(", ")
            );
            (net, stitch_sites(board, zone, pitch, diameter), what)
        };
        let tried = candidates.len();
        let placed = place_clean(board, net, &candidates, diameter, drill);
        summary = format!(
            "Placed {placed} {} via(s) {what} at {:.2} mm pitch{}",
            board.nets[net.0 as usize].name,
            pitch as f64 / 1e6,
            if placed < tried {
                format!("; {} site(s) failed DRC and were left out", tried - placed)
            } else {
                String::new()
            }
        );
        if placed == 0 {
            return Err(format!(
                "no via {what} passes DRC at this pitch and via size"
            ));
        }
        Ok(())
    })
    .map_err(failure)?;
    Ok(EditOutput {
        summary,
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

/// A via as `drill/diameter` (either order; the smaller is the drill), mm → (diameter, drill).
fn via_of(text: &str) -> Option<(Coord, Coord)> {
    let (a, b) = text.split_once('/')?;
    let (a, b) = (a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?);
    (a > 0.0 && b > 0.0 && a != b).then(|| (mm(a.max(b)), mm(a.min(b))))
}

fn net_named(board: &Board, name: &str) -> Result<NetId, String> {
    board
        .net(name)
        .ok_or_else(|| format!("`{name}` is not a net on the board"))
}

/// A pour by its Outline id number or its name.
fn zone_named(board: &Board, key: &str) -> Result<ItemId, String> {
    board
        .zones
        .values()
        .find(|z| key.parse::<u64>().is_ok_and(|n| z.id.0 == n) || z.name == key)
        .map(|z| z.id)
        .ok_or_else(|| format!("`{key}` is not a pour on the board"))
}

/// `board` with its pours filled.
fn filled(board: &Board) -> (Board, BoardRules, Vec<crate::zone::FillReport>) {
    let mut b = board.clone();
    let (mut rules, _) = BoardRules::for_board(&b);
    rules.rebuild(&BoardGraph::new(&b, &[]));
    let fills = fill_zones(&mut b, &rules, &[]);
    (b, rules, fills)
}

/// Whether a via of `diameter` at `p` lies wholly in `fill`.
fn inside(fill: &crate::geom::poly::PolySet, p: Point, diameter: Coord) -> bool {
    circle_ring(p, diameter / 2 + 1_000, 20_000)
        .iter()
        .all(|q| fill.contains(*q) == Some(true))
}

/// The pour's grid at `pitch`, centred in its outline's box, where a via lies wholly inside the
/// pour's fill on each of its layers.
fn stitch_sites(board: &Board, zone: ItemId, pitch: Coord, diameter: Coord) -> Vec<Point> {
    let (b, _, _) = filled(board);
    let z = &b.zones[&zone];
    let bb = z.outline.bbox();
    let (nx, ny) = ((bb.width() / pitch).max(0), (bb.height() / pitch).max(0));
    let (x0, y0) = (
        bb.min.x + (bb.width() - nx * pitch) / 2,
        bb.min.y + (bb.height() - ny * pitch) / 2,
    );
    let mut out = Vec::new();
    for j in 0..=ny {
        for i in 0..=nx {
            let p = Point::new(x0 + i * pitch, y0 + j * pitch);
            if z.layers
                .iter()
                .all(|l| z.fill.get(l).is_some_and(|f| inside(f, p, diameter)))
            {
                out.push(p);
            }
        }
    }
    out
}

/// Sites either side of each of `net`'s tracks at `pitch`, inside a pour of `shield`.
fn shield_sites(
    board: &Board,
    net: NetId,
    shield: NetId,
    pitch: Coord,
    diameter: Coord,
) -> Vec<Point> {
    let (b, rules, _) = filled(board);
    let clearance = rules.max_clearance().max(b.limits.min_clearance) + 5_000;
    let pours: Vec<&crate::geom::poly::PolySet> = b
        .zones
        .values()
        .filter(|z| z.net == Some(shield))
        .flat_map(|z| z.fill.values())
        .collect();
    let mut out: Vec<Point> = Vec::new();
    for t in b.tracks.values().filter(|t| t.net == Some(net)) {
        let (a, e) = t.endpoints();
        let (dx, dy) = ((e.x - a.x) as f64, (e.y - a.y) as f64);
        let len = dx.hypot(dy);
        if len == 0.0 {
            continue;
        }
        let off = (t.width / 2 + clearance + diameter / 2) as f64;
        let (nx, ny) = (-dy / len * off, dx / len * off);
        let steps = (len / pitch as f64).floor() as i64;
        for k in 0..=steps {
            let f = (k as f64 * pitch as f64 + (len - steps as f64 * pitch as f64) / 2.0) / len;
            let c = (a.x as f64 + dx * f, a.y as f64 + dy * f);
            for s in [1.0, -1.0] {
                let p = Point::new(
                    (c.0 + s * nx).round() as Coord,
                    (c.1 + s * ny).round() as Coord,
                );
                let apart = out
                    .iter()
                    .all(|q| ((q.x - p.x) as f64).hypot((q.y - p.y) as f64) >= pitch as f64 * 0.75);
                if apart && pours.iter().any(|f| inside(f, p, diameter)) {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// Adds a via of `net` at each of `sites`, then takes out every new via a DRC finding names,
/// refilling and re-checking until none is named. Returns how many stay.
fn place_clean(
    board: &mut Board,
    net: NetId,
    sites: &[Point],
    diameter: Coord,
    drill: Coord,
) -> usize {
    let mut new: Vec<ItemId> = sites
        .iter()
        .map(|p| board.add_via(Some(net), *p, diameter, drill))
        .collect();
    // Each pass takes out at least one via or ends.
    loop {
        let (b, rules, fills) = filled(board);
        let drc = Drc::run(&b, &rules, &[], &fills);
        let named: BTreeSet<ItemId> = drc
            .findings()
            .iter()
            .flat_map(|f| f.items.iter())
            .filter_map(|i| match i {
                BItem::Via(id) => Some(*id),
                _ => None,
            })
            .filter(|id| new.contains(id))
            .collect();
        if named.is_empty() {
            break;
        }
        for id in &named {
            board.vias.remove(id);
        }
        new.retain(|id| !named.contains(id));
    }
    new.len()
}
