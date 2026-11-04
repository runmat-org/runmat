//! Hole standards: thread data, tap drills, clearance fits and fastener counterbores and
//! countersinks, for the hole wizard's standard sizes.
//!
//! - ISO metric coarse threads (ISO 261/262, M1.6–M36): pitch, tap drill (major − pitch),
//!   clearance holes by ISO 273 (fine/medium/coarse as close/normal/loose), counterbores for
//!   ISO 4762 socket head cap screws, 90° countersinks for ISO 10642 flat heads (M3–M20).
//!   ANSI metric (ASME B18.3.1M) uses the same table.
//! - Unified inch threads (ASME B1.1, UNC and UNF from #2 to 1"): 75 % tap drills, close and
//!   free clearance drills (Machinery's Handbook; loose is free + 1/32"), socket head
//!   counterbores (ASME B18.3) and 82° flat head countersinks (ASME B18.6.3).
//! - Pipe taps: NPT (ASME B1.20.1, 1/16"–2") and BSPT (ISO 7-1 Rc, 1/8"–1"), 1:16 taper on
//!   diameter, tap drill without reaming, effective thread length.
//!
//! Every value returned is SI (metres, radians).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HoleStandard {
    Iso,
    AnsiMetric,
    AnsiInch,
    Npt,
    Bspt,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HoleFit {
    Close,
    #[default]
    Normal,
    Loose,
}

/// The standard size a hole is for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoleSpec {
    pub standard: HoleStandard,
    /// `M6`, `M6x0.75` (a fine pitch), `1/4-20`, `#10-32`, `1/8` (pipe).
    pub size: String,
    #[serde(default)]
    pub fit: HoleFit,
    /// Thread tolerance class (`6H`, `2B`); the standard's default when left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
}

/// A thread for a tapped or pipe-tapped hole.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadData {
    /// `M6x1`, `1/4-20 UNC`, `1/8-27 NPT`.
    pub designation: String,
    pub class: String,
    pub major: f64,
    pub pitch: f64,
    pub tap_drill: f64,
    /// Half-angle of a tapered (pipe) thread.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taper: Option<f64>,
    /// Effective thread length of a pipe thread.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<f64>,
}

impl ThreadData {
    /// Radial thread depth: the basic profile height 5H/8 = 0.5413 P of a 60° thread.
    pub fn depth(&self) -> f64 {
        0.541_265_877 * self.pitch
    }
}

/// Holes for a fastener of one size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastenerHoles {
    /// Close, normal and loose clearance diameters.
    pub clearance: [f64; 3],
    /// Socket head counterbore (diameter, depth).
    pub counterbore: Option<(f64, f64)>,
    /// Flat head countersink (diameter, included angle).
    pub countersink: Option<(f64, f64)>,
}

impl FastenerHoles {
    pub fn clearance(&self, fit: HoleFit) -> f64 {
        self.clearance[match fit {
            HoleFit::Close => 0,
            HoleFit::Normal => 1,
            HoleFit::Loose => 2,
        }]
    }
}

const MM: f64 = 1e-3;
const IN: f64 = 0.0254;

/// An ISO metric row: size, pitch, clearance close/normal/loose, counterbore Ø/depth,
/// countersink Ø.
type IsoRow = (&'static str, f64, [f64; 3], (f64, f64), f64);

/// A unified inch row: designation, major, TPI, series, tap drill, close and free clearance,
/// counterbore Ø/depth, countersink Ø.
type UnifiedRow = (
    &'static str,
    f64,
    f64,
    &'static str,
    f64,
    [f64; 2],
    (f64, f64),
    f64,
);

/// ISO metric coarse: size, pitch, clearance close/normal/loose, counterbore Ø/depth,
/// countersink Ø (90°; 0 where ISO 10642 has no size), all mm.
#[rustfmt::skip]
const ISO: &[IsoRow] = &[
    ("M1.6", 0.35, [1.7, 1.8, 2.0], (3.5, 1.8), 0.0),
    ("M2", 0.40, [2.2, 2.4, 2.6], (4.4, 2.3), 0.0),
    ("M2.5", 0.45, [2.7, 2.9, 3.1], (5.5, 2.8), 0.0),
    ("M3", 0.50, [3.2, 3.4, 3.6], (6.5, 3.3), 6.72),
    ("M4", 0.70, [4.3, 4.5, 4.8], (8.0, 4.4), 8.96),
    ("M5", 0.80, [5.3, 5.5, 5.8], (9.5, 5.4), 11.2),
    ("M6", 1.00, [6.4, 6.6, 7.0], (11.0, 6.5), 13.44),
    ("M8", 1.25, [8.4, 9.0, 10.0], (14.0, 8.6), 17.92),
    ("M10", 1.50, [10.5, 11.0, 12.0], (17.5, 10.8), 22.4),
    ("M12", 1.75, [13.0, 13.5, 14.5], (20.0, 13.0), 26.88),
    ("M14", 2.00, [15.0, 15.5, 16.5], (23.0, 15.2), 30.8),
    ("M16", 2.00, [17.0, 17.5, 18.5], (26.0, 17.5), 33.6),
    ("M20", 2.50, [21.0, 22.0, 24.0], (32.0, 21.5), 40.32),
    ("M24", 3.00, [25.0, 26.0, 28.0], (39.0, 25.5), 0.0),
    ("M30", 3.50, [31.0, 33.0, 35.0], (48.0, 32.0), 0.0),
    ("M36", 4.00, [37.0, 39.0, 42.0], (58.0, 38.0), 0.0),
];

/// Unified inch: designation, major, TPI, series, tap drill, close and free clearance,
/// counterbore Ø/depth, 82° countersink Ø, all inches.
#[rustfmt::skip]
const UNIFIED: &[UnifiedRow] = &[
    ("#2-56", 0.086, 56.0, "UNC", 0.0700, [0.0890, 0.1015], (0.1875, 0.086), 0.172),
    ("#4-40", 0.112, 40.0, "UNC", 0.0890, [0.1160, 0.1285], (0.218, 0.112), 0.225),
    ("#6-32", 0.138, 32.0, "UNC", 0.1065, [0.1440, 0.1695], (0.250, 0.138), 0.279),
    ("#8-32", 0.164, 32.0, "UNC", 0.1360, [0.1695, 0.1960], (0.312, 0.164), 0.332),
    ("#10-24", 0.190, 24.0, "UNC", 0.1495, [0.1960, 0.2210], (0.375, 0.190), 0.385),
    ("#10-32", 0.190, 32.0, "UNF", 0.1590, [0.1960, 0.2210], (0.375, 0.190), 0.385),
    ("1/4-20", 0.250, 20.0, "UNC", 0.2010, [0.2570, 0.2660], (0.4375, 0.250), 0.507),
    ("1/4-28", 0.250, 28.0, "UNF", 0.2130, [0.2570, 0.2660], (0.4375, 0.250), 0.507),
    ("5/16-18", 0.3125, 18.0, "UNC", 0.2570, [0.3230, 0.3320], (0.531, 0.3125), 0.635),
    ("3/8-16", 0.375, 16.0, "UNC", 0.3125, [0.3860, 0.3970], (0.625, 0.375), 0.762),
    ("3/8-24", 0.375, 24.0, "UNF", 0.3320, [0.3860, 0.3970], (0.625, 0.375), 0.762),
    ("1/2-13", 0.500, 13.0, "UNC", 0.4219, [0.5156, 0.5312], (0.8125, 0.500), 1.016),
    ("1/2-20", 0.500, 20.0, "UNF", 0.4531, [0.5156, 0.5312], (0.8125, 0.500), 1.016),
    ("5/8-11", 0.625, 11.0, "UNC", 0.5312, [0.6406, 0.6562], (1.000, 0.625), 1.270),
    ("3/4-10", 0.750, 10.0, "UNC", 0.6562, [0.7656, 0.7812], (1.1875, 0.750), 1.524),
    ("1-8", 1.000, 8.0, "UNC", 0.8750, [1.0156, 1.0312], (1.625, 1.000), 2.032),
];

/// Pipe threads: size, TPI, outside diameter, tap drill, effective thread length (inches;
/// BSPT in mm).
#[rustfmt::skip]
const NPT: &[(&str, f64, f64, f64, f64)] = &[
    ("1/16", 27.0, 0.3125, 0.2500, 0.2611),
    ("1/8", 27.0, 0.405, 0.3390, 0.2639),
    ("1/4", 18.0, 0.540, 0.4375, 0.4018),
    ("3/8", 18.0, 0.675, 0.5781, 0.4078),
    ("1/2", 14.0, 0.840, 0.7188, 0.5337),
    ("3/4", 14.0, 1.050, 0.9219, 0.5457),
    ("1", 11.5, 1.315, 1.1563, 0.6828),
    ("1-1/4", 11.5, 1.660, 1.5000, 0.7068),
    ("1-1/2", 11.5, 1.900, 1.7344, 0.7235),
    ("2", 11.5, 2.375, 2.2188, 0.7565),
];

#[rustfmt::skip]
const BSPT: &[(&str, f64, f64, f64, f64)] = &[
    ("1/8", 28.0, 9.728, 8.4, 6.5),
    ("1/4", 19.0, 13.157, 11.2, 9.7),
    ("3/8", 19.0, 16.662, 14.75, 10.1),
    ("1/2", 14.0, 20.955, 18.25, 13.2),
    ("3/4", 14.0, 26.441, 23.75, 14.5),
    ("1", 11.0, 33.249, 30.0, 16.8),
];

/// Half-angle of a 1:16 taper on diameter.
fn pipe_taper() -> f64 {
    (1.0f64 / 32.0).atan()
}

fn norm(s: &str) -> String {
    s.trim()
        .trim_end_matches(" NPT")
        .trim_end_matches(" BSPT")
        .replace(' ', "")
        .to_ascii_uppercase()
}

fn iso_row(size: &str) -> Option<(f64, &'static IsoRow)> {
    let s = norm(size);
    let (base, fine) = match s.split_once('X') {
        Some((b, p)) => (b.to_string(), p.parse::<f64>().ok()),
        None => (s.clone(), None),
    };
    let row = ISO.iter().find(|r| r.0.eq_ignore_ascii_case(&base))?;
    Some((fine.unwrap_or(row.1), row))
}

fn unified_row(size: &str) -> Option<&'static UnifiedRow> {
    let s = norm(size);
    let s = s
        .trim_end_matches("UNC")
        .trim_end_matches("UNF")
        .to_string();
    UNIFIED.iter().find(|r| r.0.eq_ignore_ascii_case(&s))
}

fn unknown(spec: &HoleSpec) -> String {
    format!(
        "`{}` is not a {} size in the hole tables",
        spec.size,
        match spec.standard {
            HoleStandard::Iso => "ISO metric",
            HoleStandard::AnsiMetric => "ANSI metric",
            HoleStandard::AnsiInch => "unified inch",
            HoleStandard::Npt => "NPT",
            HoleStandard::Bspt => "BSPT",
        }
    )
}

/// The thread a tapped (or pipe-tapped) hole of this size gets.
pub fn thread(spec: &HoleSpec) -> Result<ThreadData, String> {
    match spec.standard {
        HoleStandard::Iso | HoleStandard::AnsiMetric => {
            let (pitch, row) = iso_row(&spec.size).ok_or_else(|| unknown(spec))?;
            let major: f64 = row.0[1..].parse().map_err(|_| unknown(spec))?;
            Ok(ThreadData {
                designation: format!("{}x{}", row.0, pitch),
                class: spec.class.clone().unwrap_or_else(|| "6H".into()),
                major: major * MM,
                pitch: pitch * MM,
                tap_drill: (major - pitch) * MM,
                taper: None,
                length: None,
            })
        }
        HoleStandard::AnsiInch => {
            let r = unified_row(&spec.size).ok_or_else(|| unknown(spec))?;
            Ok(ThreadData {
                designation: format!("{} {}", r.0, r.3),
                class: spec.class.clone().unwrap_or_else(|| "2B".into()),
                major: r.1 * IN,
                pitch: IN / r.2,
                tap_drill: r.4 * IN,
                taper: None,
                length: None,
            })
        }
        HoleStandard::Npt | HoleStandard::Bspt => {
            let (table, unit, name) = if spec.standard == HoleStandard::Npt {
                (NPT, IN, "NPT")
            } else {
                (BSPT, MM, "BSPT")
            };
            let s = norm(&spec.size);
            let r = table
                .iter()
                .find(|r| r.0 == s)
                .ok_or_else(|| unknown(spec))?;
            Ok(ThreadData {
                designation: format!("{}-{} {}", r.0, r.1, name),
                class: spec.class.clone().unwrap_or_default(),
                major: r.2 * unit,
                pitch: IN / r.1,
                tap_drill: r.3 * unit,
                taper: Some(pipe_taper()),
                length: Some(r.4 * unit),
            })
        }
    }
}

/// Clearance, counterbore and countersink holes for a fastener of this size.
pub fn fastener(spec: &HoleSpec) -> Result<FastenerHoles, String> {
    match spec.standard {
        HoleStandard::Iso | HoleStandard::AnsiMetric => {
            let (_, r) = iso_row(&spec.size).ok_or_else(|| unknown(spec))?;
            Ok(FastenerHoles {
                clearance: [r.2[0] * MM, r.2[1] * MM, r.2[2] * MM],
                counterbore: Some((r.3 .0 * MM, r.3 .1 * MM)),
                countersink: (r.4 > 0.0).then_some((r.4 * MM, std::f64::consts::FRAC_PI_2)),
            })
        }
        HoleStandard::AnsiInch => {
            let r = unified_row(&spec.size).ok_or_else(|| unknown(spec))?;
            let free = r.5[1];
            Ok(FastenerHoles {
                clearance: [r.5[0] * IN, free * IN, (free + 1.0 / 32.0) * IN],
                counterbore: Some((r.6 .0 * IN, r.6 .1 * IN)),
                countersink: Some((r.7 * IN, 82f64.to_radians())),
            })
        }
        HoleStandard::Npt | HoleStandard::Bspt => Err(format!(
            "`{}` is a pipe thread size; it has no fastener clearance",
            spec.size
        )),
    }
}

/// Every size a standard lists, in table order.
pub fn sizes(standard: HoleStandard) -> Vec<&'static str> {
    match standard {
        HoleStandard::Iso | HoleStandard::AnsiMetric => ISO.iter().map(|r| r.0).collect(),
        HoleStandard::AnsiInch => UNIFIED.iter().map(|r| r.0).collect(),
        HoleStandard::Npt => NPT.iter().map(|r| r.0).collect(),
        HoleStandard::Bspt => BSPT.iter().map(|r| r.0).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(standard: HoleStandard, size: &str) -> HoleSpec {
        HoleSpec {
            standard,
            size: size.into(),
            fit: HoleFit::Normal,
            class: None,
        }
    }

    #[test]
    fn metric_tap_drill_is_major_less_pitch() {
        let t = thread(&spec(HoleStandard::Iso, "M6")).unwrap();
        assert_eq!(t.designation, "M6x1");
        assert!((t.tap_drill - 5.0e-3).abs() < 1e-12);
        let fine = thread(&spec(HoleStandard::Iso, "M8x1")).unwrap();
        assert!((fine.tap_drill - 7.0e-3).abs() < 1e-12);
        assert_eq!(fine.class, "6H");
    }

    #[test]
    fn clearance_fits_and_heads() {
        let h = fastener(&spec(HoleStandard::Iso, "M6")).unwrap();
        assert!((h.clearance(HoleFit::Normal) - 6.6e-3).abs() < 1e-12);
        assert!((h.clearance(HoleFit::Close) - 6.4e-3).abs() < 1e-12);
        let (cd, ch) = h.counterbore.unwrap();
        assert!((cd - 11.0e-3).abs() < 1e-12 && (ch - 6.5e-3).abs() < 1e-12);
        assert!(fastener(&spec(HoleStandard::Iso, "M2"))
            .unwrap()
            .countersink
            .is_none());
        let u = fastener(&spec(HoleStandard::AnsiInch, "1/4-20")).unwrap();
        assert!((u.clearance(HoleFit::Normal) - 0.266 * IN).abs() < 1e-12);
        assert!((u.countersink.unwrap().1 - 82f64.to_radians()).abs() < 1e-12);
    }

    #[test]
    fn unified_and_pipe_threads() {
        let t = thread(&spec(HoleStandard::AnsiInch, "1/4-20 UNC")).unwrap();
        assert_eq!(t.designation, "1/4-20 UNC");
        assert_eq!(t.class, "2B");
        assert!((t.pitch - IN / 20.0).abs() < 1e-15);
        let p = thread(&spec(HoleStandard::Npt, "1/8")).unwrap();
        assert_eq!(p.designation, "1/8-27 NPT");
        assert!((p.taper.unwrap().tan() - 1.0 / 32.0).abs() < 1e-12);
        assert!(fastener(&spec(HoleStandard::Npt, "1/8")).is_err());
        assert!(thread(&spec(HoleStandard::Iso, "M7")).is_err());
    }
}
