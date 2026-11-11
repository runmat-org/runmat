//! `through`: a straight member (a skid tube) seated through two or more holders' bores in one
//! call. Each entry names a holder occurrence, and optionally its bore (`leg_fore.bore`); with
//! no face named the holder's round faces that line up with the other holders' are used. The
//! bores must share one axis: when they don't, the refusal says how far off each is and which
//! holder to seat again with `levelWith`. The member's own round face (its outside, or `own.face`)
//! goes on that axis with its middle midway between the outermost bores (`offset` shifts it
//! along the axis), and it is fastened to the first holder only, so the mates stay solvable.

use super::geometry::*;
use super::{Frame, Placed, PlacedSet, SeatOwn, Seated, ThroughRow};
use crate::ToolError;

/// Bores farther apart than this from one axis aren't coaxial, m.
const OFF_AXIS: f64 = 1e-4;
/// Nor are bores whose axes differ by more than this (cos 0.5°).
const PARALLEL: f64 = 0.999_96;

/// One `through` entry: the holder and, when named, its bore.
pub(super) struct Hole {
    pub(super) holder: String,
    pub(super) face: Option<String>,
}

/// `through` entries as holders and bores: `leg_fore` or `leg_fore.<face>`.
pub(super) fn holes(entries: &[String], name: &str) -> Result<Vec<Hole>, ToolError> {
    if entries.len() < 2 {
        return Err(ToolError::invalid("through takes two or more holders' bores (`leg_fore.bore`, `leg_aft.bore`); for one, give target.face the bore"));
    }
    entries
        .iter()
        .map(|e| {
            let (holder, face) = match e.trim().split_once('.') {
                Some((h, f)) => (h.trim(), Some(f.trim().to_owned()).filter(|f| !f.is_empty())),
                None => (e.trim(), None),
            };
            if holder == name {
                return Err(ToolError::invalid(format!("component `{name}` can't pass through itself; name the holders it goes through")));
            }
            Ok(Hole { holder: holder.to_owned(), face })
        })
        .collect()
}

/// A bore in the assembly.
struct Bore {
    name: String,
    middle: V3,
    axis: V3,
    area: f64,
}

fn bores(hole: &Hole, placed: &Placed) -> Result<Vec<Bore>, ToolError> {
    let world = |r: Round| Bore {
        middle: placed.world.apply(r.middle),
        axis: unit(placed.world.apply_vector(r.axis)).unwrap_or(r.axis),
        area: r.area,
        name: r.name,
    };
    let h = &hole.holder;
    match &hole.face {
        Some(face) => match named_round(&placed.display, face) {
            Some(Some(r)) => Ok(vec![world(r)]),
            Some(None) => Err(ToolError::invalid(format!(
                "`{h}.{face}` isn't round; name the bore the member passes through"
            ))),
            None => Err(ToolError::invalid(format!(
                "`{h}` has no face `{face}`; its round faces are {}",
                round_names(placed)
            ))),
        },
        None => {
            let all: Vec<Bore> = rounds(&placed.display).into_iter().map(world).collect();
            if all.is_empty() {
                return Err(ToolError::invalid(format!(
                    "`{h}` has no round face (a bore) for a member to pass through"
                )));
            }
            Ok(all)
        }
    }
}

fn round_names(placed: &Placed) -> String {
    let names: Vec<String> = rounds(&placed.display)
        .into_iter()
        .map(|r| format!("`{}`", r.name))
        .take(8)
        .collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

/// `b`'s offset square to the axis through `a`.
fn off(a: &Bore, b: &Bore) -> V3 {
    let d = sub(b.middle, a.middle);
    sub(d, scale(a.axis, dot(d, a.axis)))
}

fn on_axis(a: &Bore, b: &Bore) -> bool {
    dot(a.axis, b.axis).abs() >= PARALLEL && norm(off(a, b)) <= OFF_AXIS
}

fn mm(v: f64) -> String {
    format!("{:.2}", v * 1000.0)
}

/// The seating: the two frames (the target's on the first holder), the pose, the member's
/// middle and axis, and a row per bore.
pub(super) struct Through {
    pub(super) target_frame: Frame,
    pub(super) own_frame: Frame,
    pub(super) seated: Seated,
    pub(super) center: V3,
    pub(super) axis: V3,
    pub(super) rows: Vec<ThroughRow>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn through(
    set: &PlacedSet,
    moving: &Placed,
    name: &str,
    holes: &[Hole],
    own: &SeatOwn,
    flip: bool,
    turn_by: f64,
    along: Option<V3>,
    offset: f64,
    notes: &mut Vec<String>,
) -> Result<Through, ToolError> {
    let placed: Vec<Placed> = holes
        .iter()
        .map(|h| set.get(&h.holder, "through holder"))
        .collect::<Result<_, _>>()?;
    let found: Vec<Vec<Bore>> = holes
        .iter()
        .zip(&placed)
        .map(|(h, p)| bores(h, p))
        .collect::<Result<_, _>>()?;

    // Lines through every holder: each of the first holder's bores with, in each other holder,
    // the largest bore on its axis. Bores of one holder on one axis (a bore and its boss) count
    // once.
    let mut chains: Vec<Vec<&Bore>> = Vec::new();
    for first in &found[0] {
        if chains.iter().any(|c| on_axis(c[0], first)) {
            continue;
        }
        let rest: Option<Vec<&Bore>> = found[1..]
            .iter()
            .map(|bs| {
                bs.iter()
                    .filter(|b| on_axis(first, b))
                    .max_by(|a, b| a.area.total_cmp(&b.area))
            })
            .collect();
        if let Some(rest) = rest {
            let mut chain = vec![first];
            chain.extend(rest);
            chains.push(chain);
        }
    }
    let label = |i: usize, b: &Bore| format!("`{}.{}`", holes[i].holder, b.name);
    let chain = match chains.len() {
        1 => chains.remove(0),
        0 => return Err(not_coaxial(holes, &found, name)),
        _ => {
            let lines: Vec<String> = chains
                .iter()
                .map(|c| {
                    format!(
                        "{} at [{}] mm",
                        label(0, c[0]),
                        c[0].middle.map(|v| mm(v)).join(", ")
                    )
                })
                .collect();
            return Err(ToolError::invalid(format!(
                "`{}` has {} bores that line up with the other holders': {}; name the one `{name}` passes through, as `{}.<face>`",
                holes[0].holder,
                chains.len(),
                lines.join("; "),
                holes[0].holder
            )));
        }
    };

    // The axis, and the member's middle midway between the outermost bores.
    let (p0, v) = (chain[0].middle, chain[0].axis);
    let stations: Vec<f64> = chain.iter().map(|b| dot(sub(b.middle, p0), v)).collect();
    let (lo, hi) = stations
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), s| {
            (l.min(*s), h.max(*s))
        });
    let mid = (lo + hi) / 2.0 + offset;
    let center = add(p0, scale(v, mid));
    let rows: Vec<ThroughRow> = chain
        .iter()
        .zip(&stations)
        .enumerate()
        .map(|(i, (b, s))| ThroughRow {
            holder: holes[i].holder.clone(),
            bore: b.name.clone(),
            off_axis_mm: super::round(norm(off(chain[0], b)) * 1000.0, 3),
            station_mm: super::round((s - mid) * 1000.0, 2),
        })
        .collect();

    // The member's own round face: the one named, else its largest (a tube's outside).
    let own_round = match own.face.as_deref().map(str::trim).filter(|f| !f.is_empty()) {
        Some(face) => match named_round(&moving.display, face) {
            Some(Some(r)) => r,
            Some(None) => return Err(ToolError::invalid(format!("own face `{face}` on `{name}` isn't round; through puts the member's round face (its outside) on the bores' axis"))),
            None => return Err(super::refuse_face("own", name, face, &moving.display)),
        },
        None => rounds(&moving.display).into_iter().next().ok_or_else(|| ToolError::invalid(format!("component `{name}` has no round face (a tube's outside) to put on the bores' axis")))?,
    };
    if own_round.length + 1e-6 < hi - lo {
        notes.push(format!(
            "`{name}` is {} mm long, shorter than the {} mm between the outer bores",
            mm(own_round.length),
            mm(hi - lo)
        ));
    }

    let current = moving.world.rotation;
    let same = (dot(apply(&current, own_round.axis), v) >= 0.0) != flip;
    let to_z = if same { v } else { scale(v, -1.0) };
    let own_x = perpendicular(own_round.axis);
    let lean = along.unwrap_or_else(|| apply(&current, own_x));
    let to_x = turn(
        square_to(lean, to_z).unwrap_or_else(|| perpendicular(to_z)),
        to_z,
        turn_by,
    );
    let rotation = frame_rotation(own_x, own_round.axis, to_x, to_z);
    let translation = sub(center, apply(&rotation, own_round.middle));
    let inv = placed[0].world.inverse();
    let target_frame = Frame {
        at: inv.apply(center),
        z: inv.apply_vector(to_z),
        x: inv.apply_vector(to_x),
    };
    let own_frame = Frame {
        at: own_round.middle,
        z: own_round.axis,
        x: own_x,
    };
    let names: Vec<String> = chain.iter().enumerate().map(|(i, b)| label(i, b)).collect();
    notes.push(format!(
        "through {} on one axis, its middle midway between them; fastened to `{}` only",
        names.join(", "),
        holes[0].holder
    ));
    Ok(Through {
        target_frame,
        own_frame,
        seated: Seated {
            rotation,
            translation,
        },
        center,
        axis: to_z,
        rows,
    })
}

/// Why no straight member passes through every holder, with the call that lines them up.
fn not_coaxial(holes: &[Hole], found: &[Vec<Bore>], name: &str) -> ToolError {
    let first = &found[0][0];
    let a = &holes[0].holder;
    let mut lines = Vec::new();
    for (i, bs) in found.iter().enumerate().skip(1) {
        let b = bs
            .iter()
            .max_by(|x, y| {
                dot(x.axis, first.axis)
                    .abs()
                    .total_cmp(&dot(y.axis, first.axis).abs())
                    .then(x.area.total_cmp(&y.area))
            })
            .unwrap_or(&bs[0]);
        let h = &holes[i].holder;
        if dot(b.axis, first.axis).abs() < PARALLEL {
            lines.push(format!(
                "`{h}.{}` runs along [{}], `{a}.{}` along [{}]: seat `{h}` again with along [{}]",
                b.name,
                b.axis.map(|v| format!("{v:.3}")).join(", "),
                first.name,
                first.axis.map(|v| format!("{v:.3}")).join(", "),
                first.axis.map(|v| format!("{v:.3}")).join(", ")
            ));
            continue;
        }
        let d = off(first, b);
        let (higher, lower) = if d[2] > 0.0 {
            (h.as_str(), a.as_str())
        } else {
            (a.as_str(), h.as_str())
        };
        lines.push(format!(
            "`{h}.{}`'s axis is {} mm from `{a}.{}`'s ({} mm {}, {} mm across): seat `{higher}` again with levelWith `{lower}`, so it moves down onto that axis",
            b.name,
            mm(norm(d)),
            first.name,
            mm(d[2].abs()),
            if d[2] > 0.0 { "higher" } else { "lower" },
            mm((d[0] * d[0] + d[1] * d[1]).sqrt()),
        ));
    }
    ToolError::invalid(format!("the bores aren't on one axis, so no straight member passes through them all: {}; then seat `{name}` through them again", lines.join("; ")))
}
