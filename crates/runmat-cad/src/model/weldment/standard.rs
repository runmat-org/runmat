//! Standard structural sizes for the weldment profile library, from published dimension tables:
//!
//! - IPE, HEA and HEB parallel-flange I-sections: EN 10365:2017 (the dimensions DIN 1025-5,
//!   1025-3 and 1025-2 listed before it). Each row is h × b × tw × tf with the root radius r
//!   between web and flange; the tabulated area A and second moment of area Iy (strong axis)
//!   are kept beside them so the drawn section can be checked against the table.
//! - Pipe, NPS ½ to 12 Schedule 40: ASME B36.10M, outside diameter and nominal wall.
//!
//! A designation reads as the tables print it: `IPE 200`, `HEA 160`, `NPS 2 Sch 40`. Lookup
//! ignores case and spacing (`ipe200`).

use super::ProfileOutline;
use crate::ids::SketchEntityId;
use std::f64::consts::PI;

/// A standard section's shape, in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StandardShape {
    /// A parallel-flange I-section: depth `h`, flange width `b`, web `tw`, flange `tf`, root
    /// radius `r`.
    I {
        h: f64,
        b: f64,
        tw: f64,
        tf: f64,
        r: f64,
    },
    /// A pipe: outside diameter and wall.
    Pipe { d: f64, t: f64 },
}

/// One row of a standard's table.
#[derive(Clone, Debug, PartialEq)]
pub struct StandardSize {
    pub designation: String,
    /// The family as the library groups it (`IPE`, `NPS Sch 40`).
    pub family: &'static str,
    /// The standard the dimensions come from.
    pub standard: &'static str,
    pub shape: StandardShape,
    /// The table's area (m²) and strong-axis second moment of area (m⁴), where it lists them.
    pub table_area: Option<f64>,
    pub table_iy: Option<f64>,
}

const EN10365: &str = "EN 10365:2017";
const B3610: &str = "ASME B36.10M";

/// IPE: h, b, tw, tf, r (mm), A (cm²), Iy (cm⁴).
const IPE: &[(u32, [f64; 5], f64, f64)] = &[
    (80, [80.0, 46.0, 3.8, 5.2, 5.0], 7.64, 80.1),
    (100, [100.0, 55.0, 4.1, 5.7, 7.0], 10.3, 171.0),
    (120, [120.0, 64.0, 4.4, 6.3, 7.0], 13.2, 318.0),
    (140, [140.0, 73.0, 4.7, 6.9, 7.0], 16.4, 541.0),
    (160, [160.0, 82.0, 5.0, 7.4, 9.0], 20.1, 869.0),
    (180, [180.0, 91.0, 5.3, 8.0, 9.0], 23.9, 1317.0),
    (200, [200.0, 100.0, 5.6, 8.5, 12.0], 28.5, 1943.0),
    (220, [220.0, 110.0, 5.9, 9.2, 12.0], 33.4, 2772.0),
    (240, [240.0, 120.0, 6.2, 9.8, 15.0], 39.1, 3892.0),
    (270, [270.0, 135.0, 6.6, 10.2, 15.0], 45.9, 5790.0),
    (300, [300.0, 150.0, 7.1, 10.7, 15.0], 53.8, 8356.0),
    (330, [330.0, 160.0, 7.5, 11.5, 18.0], 62.6, 11770.0),
    (360, [360.0, 170.0, 8.0, 12.7, 18.0], 72.7, 16270.0),
    (400, [400.0, 180.0, 8.6, 13.5, 21.0], 84.5, 23130.0),
    (450, [450.0, 190.0, 9.4, 14.6, 21.0], 98.8, 33740.0),
    (500, [500.0, 200.0, 10.2, 16.0, 21.0], 116.0, 48200.0),
    (550, [550.0, 210.0, 11.1, 17.2, 24.0], 134.0, 67120.0),
    (600, [600.0, 220.0, 12.0, 19.0, 24.0], 156.0, 92080.0),
];

/// HEA: h, b, tw, tf, r (mm), A (cm²), Iy (cm⁴).
const HEA: &[(u32, [f64; 5], f64, f64)] = &[
    (100, [96.0, 100.0, 5.0, 8.0, 12.0], 21.2, 349.0),
    (120, [114.0, 120.0, 5.0, 8.0, 12.0], 25.3, 606.0),
    (140, [133.0, 140.0, 5.5, 8.5, 12.0], 31.4, 1033.0),
    (160, [152.0, 160.0, 6.0, 9.0, 15.0], 38.8, 1673.0),
    (180, [171.0, 180.0, 6.0, 9.5, 15.0], 45.3, 2510.0),
    (200, [190.0, 200.0, 6.5, 10.0, 18.0], 53.8, 3692.0),
    (220, [210.0, 220.0, 7.0, 11.0, 18.0], 64.3, 5410.0),
    (240, [230.0, 240.0, 7.5, 12.0, 21.0], 76.8, 7763.0),
    (260, [250.0, 260.0, 7.5, 12.5, 24.0], 86.8, 10450.0),
    (280, [270.0, 280.0, 8.0, 13.0, 24.0], 97.3, 13670.0),
    (300, [290.0, 300.0, 8.5, 14.0, 27.0], 112.5, 18260.0),
];

/// HEB: h, b, tw, tf, r (mm), A (cm²), Iy (cm⁴).
const HEB: &[(u32, [f64; 5], f64, f64)] = &[
    (100, [100.0, 100.0, 6.0, 10.0, 12.0], 26.0, 450.0),
    (120, [120.0, 120.0, 6.5, 11.0, 12.0], 34.0, 864.0),
    (140, [140.0, 140.0, 7.0, 12.0, 12.0], 43.0, 1509.0),
    (160, [160.0, 160.0, 8.0, 13.0, 15.0], 54.3, 2492.0),
    (180, [180.0, 180.0, 8.5, 14.0, 15.0], 65.3, 3831.0),
    (200, [200.0, 200.0, 9.0, 15.0, 18.0], 78.1, 5696.0),
    (220, [220.0, 220.0, 9.5, 16.0, 18.0], 91.0, 8091.0),
    (240, [240.0, 240.0, 10.0, 17.0, 21.0], 106.0, 11260.0),
    (260, [260.0, 260.0, 10.0, 17.5, 24.0], 118.4, 14920.0),
    (280, [280.0, 280.0, 10.5, 18.0, 24.0], 131.4, 19270.0),
    (300, [300.0, 300.0, 11.0, 19.0, 27.0], 149.1, 25170.0),
];

/// NPS Schedule 40: nominal size, outside diameter and wall (mm).
const NPS40: &[(&str, f64, f64)] = &[
    ("1/2", 21.3, 2.77),
    ("3/4", 26.7, 2.87),
    ("1", 33.4, 3.38),
    ("1-1/4", 42.2, 3.56),
    ("1-1/2", 48.3, 3.68),
    ("2", 60.3, 3.91),
    ("2-1/2", 73.0, 5.16),
    ("3", 88.9, 5.49),
    ("3-1/2", 101.6, 5.74),
    ("4", 114.3, 6.02),
    ("5", 141.3, 6.55),
    ("6", 168.3, 7.11),
    ("8", 219.1, 8.18),
    ("10", 273.0, 9.27),
    ("12", 323.8, 10.31),
];

/// Every size the library offers, family by family.
pub fn all() -> Vec<StandardSize> {
    let mut out = Vec::new();
    for (family, rows) in [("IPE", IPE), ("HEA", HEA), ("HEB", HEB)] {
        for (n, [h, b, tw, tf, r], a, iy) in rows {
            out.push(StandardSize {
                designation: format!("{family} {n}"),
                family,
                standard: EN10365,
                shape: StandardShape::I {
                    h: h / 1e3,
                    b: b / 1e3,
                    tw: tw / 1e3,
                    tf: tf / 1e3,
                    r: r / 1e3,
                },
                table_area: Some(a * 1e-4),
                table_iy: Some(iy * 1e-8),
            });
        }
    }
    for (nps, d, t) in NPS40 {
        out.push(StandardSize {
            designation: format!("NPS {nps} Sch 40"),
            family: "NPS Sch 40",
            standard: B3610,
            shape: StandardShape::Pipe {
                d: d / 1e3,
                t: t / 1e3,
            },
            table_area: None,
            table_iy: None,
        });
    }
    out
}

fn key(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The size a designation names, ignoring case and spacing.
pub fn lookup(designation: &str) -> Option<StandardSize> {
    let want = key(designation);
    all().into_iter().find(|s| key(&s.designation) == want)
}

impl StandardShape {
    /// The section's area, exact: an I-section's four root fillets add (4 − π) r².
    pub fn area(&self) -> f64 {
        match *self {
            StandardShape::I { h, b, tw, tf, r } => {
                2.0 * b * tf + (h - 2.0 * tf) * tw + (4.0 - PI) * r * r
            }
            StandardShape::Pipe { d, t } => PI / 4.0 * (d * d - (d - 2.0 * t).powi(2)),
        }
    }

    /// The outline in the member frame, centred on the section (x across the flanges, y up
    /// the web); each root fillet and a pipe's circles sampled with `segments` sides a turn.
    pub fn outline(&self, segments: usize) -> ProfileOutline {
        let id = |s: &str| SketchEntityId::new(s);
        match *self {
            StandardShape::Pipe { d, t } => {
                let circle = |dia: f64, name: &str| {
                    (0..segments)
                        .map(|k| {
                            let a = 2.0 * PI * k as f64 / segments as f64;
                            (id(name), [dia / 2.0 * a.cos(), dia / 2.0 * a.sin()])
                        })
                        .collect::<Vec<_>>()
                };
                vec![circle(d, "outer"), circle(d - 2.0 * t, "inner")]
            }
            StandardShape::I { h, b, tw, tf, r } => {
                let (hh, bb, w) = (h / 2.0, b / 2.0, tw / 2.0);
                let quarter = (segments / 4).max(2);
                let mut pts: Vec<(SketchEntityId, [f64; 2])> = Vec::new();
                // A concave root fillet about `c` from angle `a0` to `a1` (degrees, decreasing).
                let fillet = |pts: &mut Vec<(SketchEntityId, [f64; 2])>,
                              name: &str,
                              c: [f64; 2],
                              a0: f64| {
                    for k in 0..quarter {
                        let a = (a0 - 90.0 * k as f64 / quarter as f64).to_radians();
                        pts.push((id(name), [c[0] + r * a.cos(), c[1] + r * a.sin()]));
                    }
                };
                pts.push((id("bottom"), [-bb, -hh]));
                pts.push((id("bottom_right_toe"), [bb, -hh]));
                pts.push((id("bottom_right_inner"), [bb, -hh + tf]));
                fillet(&mut pts, "root_bottom_right", [w + r, -hh + tf + r], 270.0);
                pts.push((id("web_right"), [w, -hh + tf + r]));
                fillet(&mut pts, "root_top_right", [w + r, hh - tf - r], 180.0);
                pts.push((id("top_right_inner"), [w + r, hh - tf]));
                pts.push((id("top_right_toe"), [bb, hh - tf]));
                pts.push((id("top"), [bb, hh]));
                pts.push((id("top_left_toe"), [-bb, hh]));
                pts.push((id("top_left_inner"), [-bb, hh - tf]));
                fillet(&mut pts, "root_top_left", [-w - r, hh - tf - r], 90.0);
                pts.push((id("web_left"), [-w, hh - tf - r]));
                fillet(&mut pts, "root_bottom_left", [-w - r, -hh + tf + r], 0.0);
                pts.push((id("bottom_left_inner"), [-w - r, -hh + tf]));
                pts.push((id("bottom_left_toe"), [-bb, -hh + tf]));
                // The fillet's first point and the straight edge before it coincide with the
                // corner points pushed above; drop exact repeats so no edge has zero length.
                pts.dedup_by(|a, b| (a.1[0] - b.1[0]).hypot(a.1[1] - b.1[1]) < 1e-12);
                vec![pts]
            }
        }
    }

    /// The cut list's description of the stock.
    pub fn stock(&self) -> String {
        match *self {
            StandardShape::I { .. } => String::new(),
            StandardShape::Pipe { d, t } => {
                format!(" (⌀{} × {})", super::mm_text(d), super::mm_text(t))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::weldment::outline_section;

    /// The drawn section reproduces each table's area and strong-axis second moment of area:
    /// a typo in a row shows up as a mismatch here.
    #[test]
    fn every_i_section_matches_its_tabulated_area_and_iy() {
        for s in all() {
            let section = outline_section(&s.shape.outline(1024));
            assert!(
                (section.area - s.shape.area()).abs() < 2e-3 * s.shape.area(),
                "{}: drawn {} vs exact {}",
                s.designation,
                section.area,
                s.shape.area()
            );
            if let Some(a) = s.table_area {
                assert!(
                    (s.shape.area() - a).abs() < 0.01 * a,
                    "{}: area {} cm² vs table {} cm²",
                    s.designation,
                    s.shape.area() * 1e4,
                    a * 1e4
                );
            }
            if let Some(iy) = s.table_iy {
                // Iy is about the axis across the web: the outline's x axis, ∫ y² dA.
                assert!(
                    (section.ixx - iy).abs() < 0.01 * iy,
                    "{}: Iy {} cm⁴ vs table {} cm⁴",
                    s.designation,
                    section.ixx * 1e8,
                    iy * 1e8
                );
            }
        }
    }

    #[test]
    fn designations_read_as_printed_or_run_together() {
        assert_eq!(lookup("ipe200").unwrap().designation, "IPE 200");
        assert_eq!(lookup("NPS 2 Sch 40").unwrap().family, "NPS Sch 40");
        assert!(lookup("IPE 210").is_none());
        // IPE 200 by hand: 2·100·8.5 + 183·5.6 + (4 − π)·12² = 2848.4 mm².
        let a = lookup("IPE 200").unwrap().shape.area();
        assert!((a * 1e6 - 2848.4).abs() < 0.1, "{}", a * 1e6);
    }
}
