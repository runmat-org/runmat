//! Board commands drawn in the board view: route, via, pour, fill, keepout, room, outline and
//! dimension take their points from canvas picks.
//!
//! The board canvas picks a point as `point:x,y` (board millimetres). A command reads its
//! points from its `points` field, in pick order, or from the selection's point picks when
//! the field is empty. Nets come from the `net` field or from a picked pad, track or via.

use crate::geom::poly::{area2, Polygon};
use crate::geom::{mm, Coord, Point};
use crate::item_ref::ItemRef;
use crate::layer_set::CopperIndex;
use crate::layout::{copper_name, Board, KeepoutRules, Layer, PadConnect};
use crate::syntax::board_edit::BoardEdit;
use serde_json::{Map, Value};

/// `point:12.5,-3` → the point in board nanometres.
pub fn parse_point(id: &str) -> Option<Point> {
    let (x, y) = ItemRef::parse(id)?.xy_mm()?;
    Some(Point::mm(x, y))
}

/// The id a canvas pick of `p` (mm) carries.
pub fn point_id(x_mm: f64, y_mm: f64) -> String {
    ItemRef::point(x_mm, y_mm).to_string()
}

fn ids(values: &Map<String, Value>, field: &str) -> Vec<String> {
    match values.get(field) {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.clone()),
                // `[x, y]` in millimetres.
                Value::Array(xy) if xy.len() == 2 => {
                    let (x, y) = (xy[0].as_f64()?, xy[1].as_f64()?);
                    Some(point_id(x, y))
                }
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// The command's points: its `points` field, else the selection's point picks.
pub fn points(values: &Map<String, Value>, selection: &[String]) -> Vec<Point> {
    let field: Vec<Point> = ids(values, "points")
        .iter()
        .filter_map(|s| parse_point(s))
        .collect();
    if !field.is_empty() {
        return field;
    }
    selection.iter().filter_map(|s| parse_point(s)).collect()
}

fn text<'a>(values: &'a Map<String, Value>, id: &str) -> Option<&'a str> {
    values
        .get(id)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
}

fn number(values: &Map<String, Value>, id: &str) -> Option<f64> {
    values.get(id).and_then(Value::as_f64)
}

fn length(values: &Map<String, Value>, id: &str) -> Option<Coord> {
    number(values, id).map(mm)
}

/// A copper layer as the layer field writes it: `Top`, `Bottom`, `Inner 2`, or a stack-up name.
pub fn copper_layer(board: &Board, name: &str) -> Result<String, String> {
    let n = board.copper_count();
    let lower = name.trim().to_ascii_lowercase();
    let index = match lower.as_str() {
        "top" | "front" => Some(0),
        "bottom" | "back" => Some(n.saturating_sub(1)),
        _ => lower
            .strip_prefix("inner")
            .and_then(|k| k.trim().parse::<CopperIndex>().ok())
            .filter(|k| *k >= 1 && *k + 1 < n),
    };
    if let Some(i) = index {
        return Ok(copper_name(i, n));
    }
    let names = board.stackup.copper_names();
    if names.iter().any(|x| x == name) || (0..n).any(|i| copper_name(i, n) == name) {
        return Ok(name.to_owned());
    }
    Err(format!("`{name}` is not a copper layer of this board"))
}

fn layer_value(board: &Board, values: &Map<String, Value>) -> Result<String, String> {
    copper_layer(board, text(values, "layer").unwrap_or("Top"))
}

/// The net: the `net` field (a net or a pad, track or via pick), else the first picked one.
fn net(board: &Board, values: &Map<String, Value>, selection: &[String]) -> Option<String> {
    let picked = ids(values, "net");
    picked
        .iter()
        .chain(selection.iter())
        .find_map(|id| net_of_pick(board, id))
}

fn net_of_pick(board: &Board, id: &str) -> Option<String> {
    let r = ItemRef::read(id, ItemRef::Net)?;
    if matches!(r, ItemRef::Zone(_)) {
        return None;
    }
    r.net_on(board).map(|n| board.net_name(n).to_owned())
}

/// `0.3/0.6` → (drill, diameter).
fn via_size(s: &str) -> Result<(Coord, Coord), String> {
    let (d, od) = s
        .split_once('/')
        .ok_or_else(|| format!("via size `{s}` is drill/diameter in mm, like 0.3/0.6"))?;
    let parse = |v: &str| {
        v.trim()
            .trim_end_matches("mm")
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("via size `{s}` is drill/diameter in mm, like 0.3/0.6"))
    };
    let (drill, diameter) = (parse(d)?, parse(od)?);
    if drill <= 0.0 || diameter <= drill {
        return Err(format!(
            "via size `{s}`: the diameter must exceed the drill"
        ));
    }
    Ok((mm(drill), mm(diameter)))
}

/// A closed outline from three or more points, counter-clockwise.
fn polygon(points: &[Point], what: &str) -> Result<Polygon, String> {
    let mut ring: Vec<Point> = points.to_vec();
    if ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    }
    ring.dedup();
    if ring.len() < 3 {
        return Err(format!("click at least three points for the {what}"));
    }
    let a = area2(&ring);
    if a == 0 {
        return Err(format!("the {what}'s points are in a line"));
    }
    if a < 0 {
        ring.reverse();
    }
    Ok(Polygon {
        outer: ring,
        holes: Vec::new(),
    })
}

/// A user layer by name (`Mech1` → User 1, `Dwgs.User` → User 0).
fn drawing_layer(name: &str) -> Layer {
    let lower = name.to_ascii_lowercase();
    if lower == "dwgs.user" || lower == "drawing" {
        return Layer::User(0);
    }
    let digits: String = lower.chars().filter(char::is_ascii_digit).collect();
    Layer::User(digits.parse().unwrap_or(1))
}

/// The model edit of a board command drawn in the board view.
pub fn drawn_edit(
    board: &Board,
    command: &str,
    values: &Map<String, Value>,
    selection: &[String],
) -> Result<BoardEdit, String> {
    let pts = points(values, selection);
    match command {
        "board.route" => {
            if pts.len() < 2 {
                return Err(
                    "click the route's start and at least one more point in the board view".into(),
                );
            }
            let net = net(board, values, selection);
            let layer = layer_value(board, values)?;
            // No width written: the net's width on this layer from its rules (an impedance
            // class's solved width, else its class or board width).
            let width = match length(values, "width") {
                Some(w) => w,
                None => {
                    let n = board.copper_count();
                    let names = board.stackup.copper_names();
                    let index = (0..n).find(|i| {
                        copper_name(*i, n) == layer || names.get(*i as usize) == Some(&layer)
                    });
                    let (rules, _) = crate::board_rules::BoardRules::for_board(board);
                    match index {
                        Some(l) => rules.width_on(
                            board,
                            net.as_deref().and_then(|x| board.net(x)),
                            l,
                            mm(0.15),
                        ),
                        None => mm(0.15),
                    }
                }
            };
            Ok(BoardEdit::Route {
                net,
                layer,
                width,
                points: pts,
            })
        }
        "board.place-via" => {
            let at = *pts.first().ok_or("click where the via goes")?;
            let (drill, diameter) = via_size(text(values, "size").unwrap_or("0.3/0.6"))?;
            let n = board.copper_count();
            let span = match text(values, "span").unwrap_or("through") {
                "blind" => Some((copper_name(0, n), copper_name(1.min(n - 1), n))),
                "buried" if n > 3 => Some((copper_name(1, n), copper_name(n - 2, n))),
                "micro" => Some((copper_name(0, n), copper_name(1.min(n - 1), n))),
                _ => None,
            };
            Ok(BoardEdit::Via {
                net: net(board, values, selection),
                at,
                diameter,
                drill,
                micro: text(values, "span") == Some("micro"),
                span,
            })
        }
        "board.pour" | "board.fill" => {
            if command == "board.pour" && text(values, "action").is_some_and(|a| a != "draw") {
                return Err(format!(
                    "{} isn't drawn in the board view",
                    text(values, "action").unwrap_or_default()
                ));
            }
            let fill = command == "board.fill";
            Ok(BoardEdit::Pour {
                net: net(board, values, selection),
                layers: vec![layer_value(board, values)?],
                outline: polygon(&pts, if fill { "fill" } else { "pour" })?,
                clearance: if fill {
                    None
                } else {
                    length(values, "clearance")
                },
                connect: match text(values, "connect") {
                    _ if fill => PadConnect::Solid,
                    Some("solid") => PadConnect::Solid,
                    Some("relief-tht") => PadConnect::ThermalThtOnly,
                    Some("none") => PadConnect::None,
                    _ => PadConnect::Thermal,
                },
                priority: values
                    .get("priority")
                    .and_then(serde_json::Value::as_f64)
                    .map_or(0, |p| p as i32),
            })
        }
        "board.keepout" => {
            let applies = text(values, "applies").unwrap_or("all");
            let all = applies == "all";
            let copper = all || applies == "copper";
            let rules = KeepoutRules {
                tracks: copper || applies == "tracks",
                vias: copper || applies == "vias",
                pads: copper,
                pours: copper,
                footprints: all || applies == "components",
                max_height: None,
            };
            let name = text(values, "name")
                .map(str::to_owned)
                .unwrap_or_else(|| format!("keepout-{}", board.keepouts.len() + 1));
            Ok(BoardEdit::Keepout {
                name,
                outline: polygon(&pts, "keepout")?,
                layers: vec![layer_value(board, values)?],
                rules,
            })
        }
        "board.room" => {
            let name = text(values, "name").ok_or("name the room")?.to_owned();
            let members: Vec<String> = ids(values, "components")
                .into_iter()
                .chain(
                    selection
                        .iter()
                        .filter(|s| ItemRef::parse(s).is_some_and(|r| r.component().is_some()))
                        .cloned(),
                )
                .filter_map(|s| ItemRef::read(&s, ItemRef::Component))
                .filter_map(|r| r.component().map(str::to_owned))
                .fold(Vec::new(), |mut v, s| {
                    if !v.contains(&s) {
                        v.push(s);
                    }
                    v
                });
            // Redrawing an existing room with no components picked keeps its members.
            let members = if members.is_empty() {
                board
                    .rooms
                    .iter()
                    .find(|r| r.name == name)
                    .map(|r| r.members.clone())
                    .unwrap_or_default()
            } else {
                members
            };
            let outline = if pts.len() >= 3 {
                polygon(&pts, "room")?
            } else {
                // Around the members' bodies, with a 1 mm margin.
                let mut r = crate::geom::Rect::EMPTY;
                for m in &members {
                    if let Some(f) = board.footprint_by_ref(m) {
                        r = r.union(f.bbox());
                    }
                }
                if r.is_empty() {
                    return Err("click the room's corners, or pick its components".into());
                }
                let r = r.inflate(mm(1.0));
                polygon(
                    &[
                        r.min,
                        Point::new(r.max.x, r.min.y),
                        r.max,
                        Point::new(r.min.x, r.max.y),
                    ],
                    "room",
                )?
            };
            Ok(BoardEdit::Room {
                name,
                outline,
                members,
            })
        }
        "board.outline" => Ok(BoardEdit::Outline {
            outline: polygon(&pts, "board outline")?,
        }),
        "board.dimension" => {
            // Two clicked points, or the centres of two picked items; a third point sets the offset.
            let mut ends: Vec<Point> = pts.iter().take(2).copied().collect();
            if ends.len() < 2 {
                for id in ids(values, "items").iter().chain(selection.iter()) {
                    if let Some(p) = centre(board, id) {
                        ends.push(p);
                    }
                    if ends.len() == 2 {
                        break;
                    }
                }
            }
            let [a, b] = ends[..] else {
                return Err("click the two points to dimension".into());
            };
            if a == b {
                return Err("the dimension's points coincide".into());
            }
            let offset = match pts.get(2) {
                Some(p) => {
                    let (dx, dy) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
                    let len = dx.hypot(dy);
                    (((p.x - a.x) as f64 * -dy + (p.y - a.y) as f64 * dx) / len).round() as Coord
                }
                None => length(values, "offset").unwrap_or(mm(2.0)),
            };
            Ok(BoardEdit::Dimension {
                layer: drawing_layer(text(values, "layer").unwrap_or("Mech1")),
                a,
                b,
                offset,
            })
        }
        other => Err(format!("{other} isn't drawn in the board view")),
    }
}

/// Where a picked item sits, for dimensions between items.
fn centre(board: &Board, id: &str) -> Option<Point> {
    ItemRef::parse(id)?.centre_on(board)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_ids_round_trip() {
        assert_eq!(point_id(12.5, -3.0), "point:12.5,-3");
        assert_eq!(parse_point("point:12.5,-3"), Some(Point::mm(12.5, -3.0)));
        assert_eq!(parse_point("pad:R1.2"), None);
        assert_eq!(via_size("0.3/0.6"), Ok((mm(0.3), mm(0.6))));
        assert!(via_size("0.6/0.3").is_err());
    }

    #[test]
    fn outlines_wind_counter_clockwise() {
        let cw = [
            Point::mm(0.0, 0.0),
            Point::mm(0.0, 5.0),
            Point::mm(5.0, 5.0),
            Point::mm(5.0, 0.0),
        ];
        let p = polygon(&cw, "pour").unwrap();
        assert!(area2(&p.outer) > 0);
        assert!(polygon(&cw[..2], "pour").is_err());
    }
}
