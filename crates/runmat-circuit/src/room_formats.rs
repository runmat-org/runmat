//! Copy Room Formats: one channel's finished placement and routing, replicated into the rooms
//! of the other channels of the same repeated sheet (Altium's *Copy Room Formats*, the
//! multi-channel step after rooms per channel).
//!
//! Channels match by identity. A room per channel is named for the channel's path (`ch[0]`),
//! and its parts' component paths sit under it (`/ch[0]/R1`). The part `/ch[1]/R1` is the
//! counterpart of `/ch[0]/R1`, whatever their designators are. Nets match through those
//! parts' pads: a source net maps to the net the counterpart pads carry, so channel-local nets
//! map to their own channel's nets, and shared nets (`GND`, a supply) map to themselves.
//!
//! [`copy_room_formats`] translates by the offset between the rooms' lower-left corners. It:
//! - places each counterpart footprint at the source part's offset position, rotation and
//!   side (a locked counterpart stays where it is and is reported);
//! - copies the tracks and vias inside the source room on mapped nets, onto the mapped nets,
//!   after removing the unlocked copper on those nets inside the target room;
//! - optionally gives the target room the source room's shape.
//!
//! Copper whose net maps to more than one target net is not copied and is reported. The edit
//! is all or nothing: on an error the board is unchanged, and on success the report holds the
//! previous board as the one undo step.

use crate::geom::poly::Polygon;
use crate::geom::Point;
use crate::layout::{Board, ItemId, NetId, Room, Track, TrackGeom, Via};
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
}

impl Default for RoomFormatOptions {
    fn default() -> Self {
        RoomFormatOptions {
            placement: true,
            routing: true,
            room_shape: true,
        }
    }
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

fn track_inside(room: &Polygon, t: &Track) -> bool {
    match t.geom {
        TrackGeom::Segment { a, b } => inside(room, a) && inside(room, b),
        TrackGeom::Arc { start, mid, end } => {
            inside(room, start) && inside(room, mid) && inside(room, end)
        }
    }
}

fn moved(geom: &TrackGeom, d: Point) -> TrackGeom {
    match *geom {
        TrackGeom::Segment { a, b } => TrackGeom::Segment { a: a + d, b: b + d },
        TrackGeom::Arc { start, mid, end } => TrackGeom::Arc {
            start: start + d,
            mid: mid + d,
            end: end + d,
        },
    }
}

fn moved_outline(p: &Polygon, d: Point) -> Polygon {
    Polygon {
        outer: p.outer.iter().map(|q| *q + d).collect(),
        holes: p
            .holes
            .iter()
            .map(|h| h.iter().map(|q| *q + d).collect())
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
        let d = dst.outline.bbox().min - src.outline.bbox().min;
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
                tf.at = sf.at + d;
                tf.rotation = sf.rotation;
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
                    at: v.at + d,
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
}
