//! Copy Room Formats: one channel's finished placement and routing, replicated into the rooms
//! of the other channels of the same repeated sheet (the multi-channel step after rooms per
//! channel).
//!
//! Channels match by identity. A room per channel is named for the channel's path (`ch[0]`),
//! and its parts' component paths sit under it (`/ch[0]/R1`). The part `/ch[1]/R1` is the
//! counterpart of `/ch[0]/R1`, whatever their designators are. Nets match through those
//! parts' pads: a source net maps to the net the counterpart pads carry, so channel-local nets
//! map to their own channel's nets, and shared nets (`GND`, a supply) map to themselves.
//!
//! [`copy_room_formats`] translates by the offset between the rooms' lower-left corners, or,
//! with [`RoomFormatOptions::rotate`], turns each copy about the board's center by the quarter
//! turns that carry the source room's center onto the target's (rooms set around the center, as
//! the four channels of a 4-in-1 motor controller are). It:
//! - places each counterpart footprint at the source part's offset position, rotation and
//!   side (a locked counterpart stays where it is and is reported);
//! - copies the tracks and vias inside the source room on mapped nets, onto the mapped nets,
//!   after removing the unlocked copper on those nets inside the target room;
//! - copies the pours inside the source room the same way: a pour on a mapped net goes onto
//!   the mapped net, a pour with no net goes as it is, and the target room's unlocked pours on
//!   those nets are removed first;
//! - optionally gives the target room the source room's shape.
//!
//! Copper whose net maps to more than one target net is not copied and is reported. The edit
//! is all or nothing: on an error the board is unchanged, and on success the report holds the
//! previous board as the one undo step.

use crate::geom::poly::Polygon;
use crate::geom::Point;
use crate::layout::{Board, ItemId, NetId, Room, Track, TrackGeom, Via, Zone};
use std::collections::{BTreeMap, BTreeSet};

/// What to copy.
#[derive(Clone, Debug, PartialEq)]
pub struct RoomFormatOptions {
    /// Footprint position, rotation and side.
    pub placement: bool,
    /// Tracks and vias, replacing the target room's routing on the mapped nets.
    pub routing: bool,
    /// The source room's outline, moved to the target room's lower-left corner.
    pub room_shape: bool,
    /// Copper pours on the mapped nets (and pours with no net) that lie inside the source room,
    /// replacing the target room's unlocked pours on those nets.
    pub pours: bool,
    /// Turn each copy about the board outline's center, by the multiple of 90 degrees between
    /// the source and target rooms' centers as seen from it, instead of only translating.
    pub rotate: bool,
}

impl Default for RoomFormatOptions {
    fn default() -> Self {
        RoomFormatOptions {
            placement: true,
            routing: true,
            room_shape: true,
            pours: true,
            rotate: false,
        }
    }
}

/// Where a copy goes: `p ↦ R(p − from) + to`, with `R` a turn of `quarter` × 90°
/// counter-clockwise (exact on the nanometre grid).
#[derive(Clone, Copy, Debug)]
struct Map {
    quarter: u8,
    from: Point,
    to: Point,
}

impl Map {
    fn apply(&self, p: Point) -> Point {
        let v = p - self.from;
        let r = match self.quarter % 4 {
            0 => v,
            1 => Point::new(-v.y, v.x),
            2 => Point::new(-v.x, -v.y),
            _ => Point::new(v.y, -v.x),
        };
        r + self.to
    }
    fn degrees(&self) -> f64 {
        90.0 * f64::from(self.quarter % 4)
    }
}

fn center(p: &Polygon) -> Point {
    let b = p.bbox();
    Point::new((b.min.x + b.max.x) / 2, (b.min.y + b.max.y) / 2)
}

/// The map from the source room to the target room: a translation between their lower-left
/// corners, or a turn about `pivot` (the board's center) followed by the translation that puts
/// the turned source center on the target center.
fn room_map(src: &Polygon, dst: &Polygon, pivot: Option<Point>) -> Map {
    let Some(c) = pivot else {
        return Map { quarter: 0, from: src.bbox().min, to: dst.bbox().min };
    };
    let (a, b) = (center(src) - c, center(dst) - c);
    let angle = (b.y as f64).atan2(b.x as f64) - (a.y as f64).atan2(a.x as f64);
    let quarter = ((angle.to_degrees() / 90.0).round() as i64).rem_euclid(4) as u8;
    Map { quarter, from: center(src), to: center(dst) }
}

/// What one target room received.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoomCopy {
    pub room: String,
    /// Designators of the footprints placed.
    pub placed: Vec<String>,
    pub tracks_added: Vec<u64>,
    pub vias_added: Vec<u64>,
    pub tracks_removed: Vec<u64>,
    pub vias_removed: Vec<u64>,
    pub pours_added: Vec<u64>,
    pub pours_removed: Vec<u64>,
    /// Items left as they were, with why (`R12: locked`).
    pub skipped: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoomFormatReport {
    pub source: String,
    pub copies: Vec<RoomCopy>,
    /// The board before the edit: the one undo step.
    pub undo: Box<Board>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoomFormatError(pub String);

impl std::fmt::Display for RoomFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RoomFormatError {}

/// `ch[2]` → `ch`: the repeated sheet a channel room belongs to.
fn channel_base(room: &str) -> Option<&str> {
    let open = room.rfind('[')?;
    let index = room.strip_suffix(']')?.get(open + 1..)?;
    (!index.is_empty() && index.chars().all(|c| c.is_ascii_digit())).then(|| &room[..open])
}

/// The rooms of the same repeated sheet as `source`, in board order.
pub fn sibling_rooms(board: &Board, source: &str) -> Vec<String> {
    let Some(base) = channel_base(source) else {
        return Vec::new();
    };
    board
        .rooms
        .iter()
        .filter(|r| r.name != source && channel_base(&r.name) == Some(base))
        .map(|r| r.name.clone())
        .collect()
}

/// The component path under a room's channel (`/ch[0]/R1` in `ch[0]` → `R1`).
fn within<'a>(room: &str, path: &'a str) -> Option<&'a str> {
    path.strip_prefix('/')?
        .strip_prefix(room)?
        .strip_prefix('/')
        .filter(|rest| !rest.is_empty())
}

fn inside(room: &Polygon, p: Point) -> bool {
    room.contains(p) != Some(false)
}

/// A pour lies inside a room when every vertex of its outline does.
fn zone_inside(room: &Polygon, z: &Zone) -> bool {
    z.outline
        .outer
        .iter()
        .chain(z.outline.holes.iter().flatten())
        .all(|p| inside(room, *p))
}

fn track_inside(room: &Polygon, t: &Track) -> bool {
    match t.geom {
        TrackGeom::Segment { a, b } => inside(room, a) && inside(room, b),
        TrackGeom::Arc { start, mid, end } => {
            inside(room, start) && inside(room, mid) && inside(room, end)
        }
    }
}

fn moved(geom: &TrackGeom, d: Map) -> TrackGeom {
    match *geom {
        TrackGeom::Segment { a, b } => TrackGeom::Segment { a: d.apply(a), b: d.apply(b) },
        TrackGeom::Arc { start, mid, end } => TrackGeom::Arc {
            start: d.apply(start),
            mid: d.apply(mid),
            end: d.apply(end),
        },
    }
}

fn moved_outline(p: &Polygon, d: Map) -> Polygon {
    Polygon {
        outer: p.outer.iter().map(|q| d.apply(*q)).collect(),
        holes: p
            .holes
            .iter()
            .map(|h| h.iter().map(|q| d.apply(*q)).collect())
            .collect(),
    }
}

fn room<'a>(board: &'a Board, name: &str) -> Result<&'a Room, RoomFormatError> {
    board
        .rooms
        .iter()
        .find(|r| r.name == name)
        .ok_or_else(|| RoomFormatError(format!("there is no room {name}")))
}

/// Copies `source`'s placement and routing into each of `targets` (every other room of the
/// same repeated sheet when `targets` is empty).
pub fn copy_room_formats(
    board: &mut Board,
    source: &str,
    targets: &[String],
    o: &RoomFormatOptions,
) -> Result<RoomFormatReport, RoomFormatError> {
    let src = room(board, source)?.clone();
    let targets: Vec<String> = if targets.is_empty() {
        sibling_rooms(board, source)
    } else {
        targets.to_vec()
    };
    if targets.is_empty() {
        return Err(RoomFormatError(format!(
            "room {source} has no other channel rooms to copy to"
        )));
    }
    if let Some(same) = targets.iter().find(|t| t.as_str() == source) {
        return Err(RoomFormatError(format!(
            "room {same} cannot copy to itself"
        )));
    }
    for t in &targets {
        room(board, t)?;
    }
    // The source parts: the parts whose component path is under the room's channel.
    let parts: BTreeMap<String, ItemId> = board
        .footprints
        .values()
        .filter_map(|f| Some((within(source, f.component.as_deref()?)?.to_owned(), f.id)))
        .collect();
    if parts.is_empty() {
        return Err(RoomFormatError(format!(
            "room {source} holds no parts linked to its channel"
        )));
    }

    let before = board.clone();
    let mut next = board.clone();
    let mut copies = Vec::new();
    for target in &targets {
        let dst = room(&next, target)?.clone();
        let pivot = o.rotate.then(|| center(&next.outline));
        let d = room_map(&src.outline, &dst.outline, pivot);
        let mut copy = RoomCopy {
            room: target.clone(),
            ..RoomCopy::default()
        };
        let by_path: BTreeMap<&str, ItemId> = next
            .footprints
            .values()
            .filter_map(|f| Some((within(target, f.component.as_deref()?)?, f.id)))
            .collect();
        let pairs: Vec<(ItemId, ItemId)> = parts
            .iter()
            .filter_map(|(rest, s)| match by_path.get(rest.as_str()) {
                Some(t) => Some((*s, *t)),
                None => {
                    copy.skipped.push(format!(
                        "{}: no counterpart in {target}",
                        next.footprints[s].reference
                    ));
                    None
                }
            })
            .collect();
        if pairs.is_empty() {
            return Err(RoomFormatError(format!(
                "room {target} has none of {source}'s parts"
            )));
        }

        // Source net → the target nets its counterpart pads carry.
        let mut nets: BTreeMap<NetId, BTreeSet<NetId>> = BTreeMap::new();
        for (s, t) in &pairs {
            let (sf, tf) = (&next.footprints[s], &next.footprints[t]);
            for (pad, net) in &sf.pad_nets {
                if let Some(tn) = tf.pad_nets.get(pad) {
                    nets.entry(*net).or_default().insert(*tn);
                }
            }
        }
        let mapped: BTreeMap<NetId, NetId> = nets
            .iter()
            .filter(|(_, t)| t.len() == 1)
            .map(|(s, t)| (*s, *t.iter().next().unwrap()))
            .collect();
        for (s, t) in nets.iter().filter(|(_, t)| t.len() > 1) {
            let names: Vec<&str> = t.iter().map(|n| next.net_name(*n)).collect();
            copy.skipped.push(format!(
                "net {}: maps to {} in {target}; its copper is not copied",
                next.net_name(*s),
                names.join(", ")
            ));
        }

        if o.placement {
            for (s, t) in &pairs {
                let sf = next.footprints[s].clone();
                let tf = next.footprints.get_mut(t).unwrap();
                if tf.locked {
                    copy.skipped.push(format!("{}: locked", tf.reference));
                    continue;
                }
                tf.at = d.apply(sf.at);
                tf.rotation = (sf.rotation + d.degrees()).rem_euclid(360.0);
                tf.side = sf.side;
                copy.placed.push(tf.reference.clone());
            }
        }
        let outline = if o.room_shape {
            moved_outline(&src.outline, d)
        } else {
            dst.outline.clone()
        };
        if o.routing {
            let target_nets: BTreeSet<NetId> = mapped.values().copied().collect();
            let stale_tracks: Vec<ItemId> = next
                .tracks
                .values()
                .filter(|t| {
                    !t.locked
                        && t.net.is_some_and(|n| target_nets.contains(&n))
                        && track_inside(&outline, t)
                })
                .map(|t| t.id)
                .collect();
            let stale_vias: Vec<ItemId> = next
                .vias
                .values()
                .filter(|v| {
                    !v.locked
                        && v.net.is_some_and(|n| target_nets.contains(&n))
                        && inside(&outline, v.at)
                })
                .map(|v| v.id)
                .collect();
            for id in &stale_tracks {
                next.tracks.remove(id);
                copy.tracks_removed.push(id.0);
            }
            for id in &stale_vias {
                next.vias.remove(id);
                copy.vias_removed.push(id.0);
            }
            let tracks: Vec<Track> = before
                .tracks
                .values()
                .filter(|t| track_inside(&src.outline, t))
                .filter_map(|t| Some((t, *mapped.get(&t.net?)?)))
                .map(|(t, net)| Track {
                    id: ItemId(0),
                    net: Some(net),
                    geom: moved(&t.geom, d),
                    ..t.clone()
                })
                .collect();
            let vias: Vec<Via> = before
                .vias
                .values()
                .filter(|v| inside(&src.outline, v.at))
                .filter_map(|v| Some((v, *mapped.get(&v.net?)?)))
                .map(|(v, net)| Via {
                    id: ItemId(0),
                    net: Some(net),
                    at: d.apply(v.at),
                    ..v.clone()
                })
                .collect();
            for mut t in tracks {
                t.id = next.alloc();
                copy.tracks_added.push(t.id.0);
                next.tracks.insert(t.id, t);
            }
            for mut v in vias {
                v.id = next.alloc();
                copy.vias_added.push(v.id.0);
                next.vias.insert(v.id, v);
            }
        }
        if o.pours {
            // Pours on a net that maps to more than one target net are not copied (reported
            // with the net above); pours with no net copy as they are.
            let target_nets: BTreeSet<NetId> = mapped.values().copied().collect();
            let stale: Vec<ItemId> = next
                .zones
                .values()
                .filter(|z| {
                    !z.locked
                        && z.net.is_none_or(|n| target_nets.contains(&n))
                        && zone_inside(&outline, z)
                })
                .map(|z| z.id)
                .collect();
            for id in &stale {
                next.zones.remove(id);
                copy.pours_removed.push(id.0);
            }
            let pours: Vec<Zone> = before
                .zones
                .values()
                .filter(|z| zone_inside(&src.outline, z))
                .filter_map(|z| match z.net {
                    None => Some((z, None)),
                    Some(n) => mapped.get(&n).map(|t| (z, Some(*t))),
                })
                .map(|(z, net)| Zone {
                    id: ItemId(0),
                    net,
                    outline: moved_outline(&z.outline, d),
                    fill: BTreeMap::new(),
                    ..z.clone()
                })
                .collect();
            for mut z in pours {
                z.id = next.alloc();
                copy.pours_added.push(z.id.0);
                next.zones.insert(z.id, z);
            }
        }
        if o.room_shape {
            if let Some(r) = next.rooms.iter_mut().find(|r| &r.name == target) {
                r.outline = outline;
            }
        }
        copies.push(copy);
    }
    *board = next;
    Ok(RoomFormatReport {
        source: source.to_owned(),
        copies,
        undo: Box::new(before),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_rooms_by_base() {
        assert_eq!(channel_base("ch[2]"), Some("ch"));
        assert_eq!(channel_base("pwr/ch[10]"), Some("pwr/ch"));
        assert_eq!(channel_base("power"), None);
        assert_eq!(channel_base("ch[]"), None);
        assert_eq!(within("ch[0]", "/ch[0]/R1"), Some("R1"));
        assert_eq!(within("ch[0]", "/ch[0]/amp/U1"), Some("amp/U1"));
        assert_eq!(within("ch[0]", "/ch[01]/R1"), None);
        assert_eq!(within("ch[0]", "/ch[0]"), None);
    }

    fn square(x0: f64, y0: f64, side: f64) -> Polygon {
        let p = |x: f64, y: f64| Point::mm(x, y);
        Polygon {
            outer: vec![p(x0, y0), p(x0 + side, y0), p(x0 + side, y0 + side), p(x0, y0 + side)],
            holes: vec![],
        }
    }

    /// Quadrant rooms around a board centerd on the origin: the lower-right room is the
    /// lower-left one turned a quarter counter-clockwise, the upper-right a half turn, and a
    /// point carries onto its rotated image exactly. Without a pivot the map is the corner offset.
    #[test]
    fn quarter_turns_about_the_board_center() {
        let (ll, lr, ur, ul) = (square(-20.0, -20.0, 18.0), square(2.0, -20.0, 18.0), square(2.0, 2.0, 18.0), square(-20.0, 2.0, 18.0));
        let c = Some(Point::ORIGIN);
        let a = Point::mm(-15.25, -15.25);
        let m = room_map(&ll, &lr, c);
        assert_eq!((m.quarter, m.apply(a)), (1, Point::mm(15.25, -15.25)));
        let m = room_map(&ll, &ur, c);
        assert_eq!((m.quarter, m.apply(a), m.degrees()), (2, Point::mm(15.25, 15.25), 180.0));
        let m = room_map(&ll, &ul, c);
        assert_eq!((m.quarter, m.apply(a)), (3, Point::mm(-15.25, 15.25)));
        let m = room_map(&ll, &lr, None);
        assert_eq!((m.quarter, m.apply(a)), (0, Point::mm(6.75, -15.25)));
    }
}
