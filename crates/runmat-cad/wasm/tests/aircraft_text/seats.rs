//! The seats: one passenger seat (a leg frame, a fabric cushion and back, armrests between
//! seats), patterned on grids into business doubles and economy triples row by row, and the two
//! pilot seats. Seat coordinates: the origin is on the floor under the cushion's centre, x
//! aft (the seat faces −x, forward), y to starboard, z up.

use super::text::{mm, offset, outline, part, polygon, use_};

/// Economy seat pitch across the aircraft and the aisle's width.
pub const ECONOMY_SEAT: f64 = 450.0;
pub const BUSINESS_SEAT: f64 = 560.0;
/// Half the aisle at the seat cushions, economy and business.
pub const ECONOMY_AISLE: f64 = 210.0;
pub const BUSINESS_AISLE: f64 = 300.0;

/// The side profile of a seat's cushion and back, (x, z): the cushion's front roll at −x, the
/// back raked 9° aft, `back` mm above the floor at its top.
fn cushion_profile(back: f64) -> Vec<(f64, f64)> {
    let rake = 0.16;
    let top = back;
    vec![
        (-250.0, 410.0),
        (-265.0, 505.0),
        (170.0, 525.0),
        (170.0 + rake * (top - 525.0), top - 30.0),
        (260.0 + rake * (top - 525.0), top),
        (300.0 + rake * (top - 525.0), top - 60.0),
        (300.0, 470.0),
        (260.0, 400.0),
    ]
}

/// A cushion and back in one fabric piece, `width` across, its outline's corners rounded.
fn cushion(
    name: &str,
    number: &str,
    material: &str,
    comment: &str,
    width: f64,
    back: f64,
) -> String {
    let corners: Vec<(f64, f64)> = cushion_profile(back).iter().map(|&(x, z)| (z, x)).collect();
    part(
        name,
        number,
        material,
        comment,
        &format!(
            "{}    feature body = extrude(profile = sketch.s, distance = {}, symmetric)\n",
            outline("s", "origin.zx", &corners, 28.0),
            mm(width)
        ),
        "body",
    )
}

/// The seat frame: two leg plates and two cross tubes under the cushion.
fn frame() -> String {
    let leg: Vec<(f64, f64)> = [
        (-210.0, 0.0),
        (-160.0, 0.0),
        (-90.0, 360.0),
        (180.0, 360.0),
        (240.0, 0.0),
        (290.0, 0.0),
        (230.0, 405.0),
        (-200.0, 405.0),
    ]
    .iter()
    .map(|&(x, z)| (z, x))
    .collect();
    let mut body = String::new();
    for (k, y) in [-180.0, 160.0].iter().enumerate() {
        body += &format!(
            "    feature leg{k}_at = plane.offset(base = origin.zx, distance = {})\n",
            mm(*y)
        );
        body += &polygon(&format!("leg{k}_s"), &format!("plane.leg{k}_at"), &leg);
        let mode = if k == 0 { "" } else { ", mode = add" };
        body += &format!(
            "    feature leg{k} = extrude(profile = sketch.leg{k}_s, distance = 20 mm{mode})\n"
        );
    }
    for (k, x) in [-140.0, 200.0].iter().enumerate() {
        body += &format!("    sketch tube{k}_s on origin.zx {{\n        c = circle(center = [385 mm, {}], radius = 18 mm)\n    }}\n", mm(*x));
        body += &format!("    feature tube{k} = extrude(profile = sketch.tube{k}_s, distance = 400 mm, symmetric, mode = add)\n");
    }
    part(
        "seat_frame",
        "CAB-ST-110",
        "aluminum_7075",
        "A seat's frame: two leg plates and two cross tubes under the cushion.",
        &body,
        "leg0",
    )
}

fn armrest() -> String {
    let pts: Vec<(f64, f64)> = [
        (-150.0, 595.0),
        (200.0, 595.0),
        (200.0, 420.0),
        (245.0, 420.0),
        (245.0, 595.0),
        (300.0, 595.0),
        (300.0, 645.0),
        (-150.0, 645.0),
    ]
    .iter()
    .map(|&(x, z)| (z, x))
    .collect();
    part(
        "seat_armrest",
        "CAB-ST-130",
        "polycarbonate",
        "An armrest between two seats, on a post from the seat frame.",
        &format!(
            "{}    feature body = extrude(profile = sketch.s, distance = 40 mm, symmetric)\n",
            polygon("s", "origin.zx", &pts)
        ),
        "body",
    )
}

pub fn parts() -> Vec<(String, String)> {
    vec![
        ("seat_frame.part".into(), frame()),
        ("seat_armrest.part".into(), armrest()),
        (
            "seat_cushion.part".into(),
            cushion(
                "seat_cushion",
                "CAB-ST-120",
                "wool_fabric",
                "An economy seat's cushion and back, 400 mm wide, in fabric.",
                400.0,
                1_130.0,
            ),
        ),
        (
            "seat_cushion_business.part".into(),
            cushion(
                "seat_cushion_business",
                "CAB-ST-220",
                "leather",
                "A business seat's cushion and back, 500 mm wide, in leather.",
                500.0,
                1_180.0,
            ),
        ),
        (
            "pilot_seat_cushion.part".into(),
            cushion(
                "pilot_seat_cushion",
                "FD-ST-320",
                "leather",
                "A pilot seat's cushion, back and headrest, 520 mm wide.",
                520.0,
                1_300.0,
            ),
        ),
    ]
}

/// One block of seats on one side of the aisle, for `cabin.assembly`: a seat frame, a cushion
/// and an armrest at the block's front inboard seat, patterned on a grid aft by `rows` rows
/// (`row_pitch`, an input's name) and across by `seats` seats `seat_pitch` apart, starting at
/// `y0` and running to starboard; an armrest each side of every seat.
pub struct Block<'a> {
    pub name: &'a str,
    pub cushion: &'a str,
    pub x0: f64,
    pub y0: f64,
    pub z: f64,
    pub rows: usize,
    pub row_pitch: &'a str,
    pub seats: usize,
    pub seat_pitch: f64,
}

impl Block<'_> {
    /// The occurrences and their patterns.
    pub fn uses(&self) -> (String, String) {
        let n = self.name;
        let mut s = String::new();
        s += &use_(
            "part",
            "seat_frame",
            &format!("{n}_frame"),
            &[offset(self.x0, self.y0, self.z)],
        );
        s += &use_(
            "part",
            self.cushion,
            &format!("{n}_cushion"),
            &[offset(self.x0, self.y0, self.z)],
        );
        s += &use_(
            "part",
            "seat_armrest",
            &format!("{n}_armrest"),
            &[offset(self.x0, self.y0 - self.seat_pitch / 2.0, self.z)],
        );
        let grid = |seeds: &str, count: usize| {
            format!(
                "    pattern {n}_{seeds}s = grid(seeds = [{n}_{seeds}], first = [1, 0, 0], first_spacing = input.{}, first_count = {}, second = [0, 1, 0], second_spacing = {}, second_count = {count})\n",
                self.row_pitch,
                self.rows,
                mm(self.seat_pitch)
            )
        };
        let p = grid("frame", self.seats)
            + &grid("cushion", self.seats)
            + &grid("armrest", self.seats + 1);
        (s, p)
    }
}
