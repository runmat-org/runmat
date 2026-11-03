//! `.kicad_pro` project files (JSON): net classes in and out.
//!
//! The project file has no published specification; this module follows the structure KiCad
//! 7–10 project files carry, as observed in files: `net_settings` holds `classes` (one object
//! per class: `name`, `clearance`, `track_width`, `via_diameter`, `via_drill`,
//! `microvia_diameter`, `microvia_drill`, `diff_pair_width`, `diff_pair_gap`,
//! `diff_pair_via_gap`, `wire_width`, `bus_width`, colours…, lengths in millimetres),
//! `netclass_patterns` (`{"netclass", "pattern"}` with `*`/`?` wildcards) and
//! `netclass_assignments` (net name → class name, or a list of class names).
//!
//! Editing is lossless: the file is held as an order-preserving JSON tree whose numbers keep
//! their source text, only `net_settings` members we model are replaced, and an unedited file
//! is written back byte for byte.

use crate::board_rules::{props, RuleSpec, RuleValue};
use crate::geom::Coord;
use crate::ir::{NetClass, Param, ParamValue, Tolerance};
use runmat_rules::{Dim, Quantity};
use std::collections::BTreeMap;
use std::fmt::Write as _;

// ---------------------------------------------------------------- JSON

/// A JSON value that keeps object key order and number text.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// The number as written.
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Json> {
        match self {
            Json::Obj(m) => m.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// Sets `key` in an object, keeping its position when present (appending otherwise).
    pub fn set(&mut self, key: &str, v: Json) {
        if let Json::Obj(m) = self {
            match m.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = v,
                None => m.push((key.to_string(), v)),
            }
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => n.parse().ok(),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn num(v: f64) -> Json {
        let mut s = format!("{v}");
        if !s.contains(['.', 'e', 'E']) {
            s.push_str(".0");
        }
        Json::Num(s)
    }
    pub fn int(v: i64) -> Json {
        Json::Num(v.to_string())
    }
    pub fn str(s: &str) -> Json {
        Json::Str(s.to_string())
    }
}

/// Parses JSON text.
pub fn parse_json(text: &str) -> Result<Json, String> {
    let b = text.as_bytes();
    let mut i = 0;
    let v = value(b, &mut i)?;
    skip_ws(b, &mut i);
    if i != b.len() {
        return Err(format!("trailing characters at byte {i}"));
    }
    Ok(v)
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn value(b: &[u8], i: &mut usize) -> Result<Json, String> {
    skip_ws(b, i);
    let err = |i: usize, what: &str| format!("{what} at byte {i}");
    match b.get(*i) {
        None => Err(err(*i, "unexpected end")),
        Some(b'{') => {
            *i += 1;
            let mut m = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Json::Obj(m));
            }
            loop {
                skip_ws(b, i);
                let Json::Str(k) = string(b, i)? else {
                    unreachable!()
                };
                skip_ws(b, i);
                if b.get(*i) != Some(&b':') {
                    return Err(err(*i, "expected ':'"));
                }
                *i += 1;
                let v = value(b, i)?;
                m.push((k, v));
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => {
                        *i += 1;
                        return Ok(Json::Obj(m));
                    }
                    _ => return Err(err(*i, "expected ',' or '}'")),
                }
            }
        }
        Some(b'[') => {
            *i += 1;
            let mut a = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Json::Arr(a));
            }
            loop {
                a.push(value(b, i)?);
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => {
                        *i += 1;
                        return Ok(Json::Arr(a));
                    }
                    _ => return Err(err(*i, "expected ',' or ']'")),
                }
            }
        }
        Some(b'"') => string(b, i),
        Some(b't') if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(Json::Bool(true))
        }
        Some(b'f') if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(Json::Bool(false))
        }
        Some(b'n') if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(Json::Null)
        }
        Some(c) if *c == b'-' || c.is_ascii_digit() => {
            let s = *i;
            *i += 1;
            while *i < b.len() && matches!(b[*i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
                *i += 1;
            }
            let t = std::str::from_utf8(&b[s..*i]).unwrap().to_string();
            t.parse::<f64>()
                .map_err(|_| err(s, "bad number"))
                .map(|_| Json::Num(t))
        }
        _ => Err(err(*i, "unexpected character")),
    }
}

fn string(b: &[u8], i: &mut usize) -> Result<Json, String> {
    if b.get(*i) != Some(&b'"') {
        return Err(format!("expected string at byte {i}"));
    }
    *i += 1;
    let mut out: Vec<u8> = Vec::new();
    while let Some(&c) = b.get(*i) {
        *i += 1;
        match c {
            b'"' => {
                return String::from_utf8(out)
                    .map(Json::Str)
                    .map_err(|e| e.to_string())
            }
            b'\\' => {
                let e = *b.get(*i).ok_or("unterminated escape")?;
                *i += 1;
                let ch = match e {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        let hex = std::str::from_utf8(b.get(*i..*i + 4).ok_or("short \\u")?)
                            .map_err(|e| e.to_string())?;
                        *i += 4;
                        let mut cp = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                        if (0xD800..0xDC00).contains(&cp) && b[*i..].starts_with(b"\\u") {
                            let lo = std::str::from_utf8(&b[*i + 2..*i + 6]).unwrap_or("0");
                            let lo = u32::from_str_radix(lo, 16).unwrap_or(0);
                            *i += 6;
                            cp =
                                0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF);
                        }
                        char::from_u32(cp).unwrap_or('\u{FFFD}')
                    }
                    _ => return Err(format!("bad escape at byte {i}")),
                };
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
            _ => out.push(c),
        }
    }
    Err("unterminated string".into())
}

/// Writes JSON the way KiCad lays project files out: two-space indentation, one member
/// per line, empty containers inline.
pub fn write_json(v: &Json) -> String {
    let mut s = String::new();
    write_value(v, 0, &mut s);
    s.push('\n');
    s
}

fn quote(t: &str, s: &mut String) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(s, "\\u{:04x}", c as u32);
            }
            c => s.push(c),
        }
    }
    s.push('"');
}

fn write_value(v: &Json, depth: usize, s: &mut String) {
    let pad = |d: usize, s: &mut String| s.extend(std::iter::repeat_n("  ", d));
    match v {
        Json::Null => s.push_str("null"),
        Json::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => s.push_str(n),
        Json::Str(t) => quote(t, s),
        Json::Arr(a) if a.is_empty() => s.push_str("[]"),
        Json::Obj(m) if m.is_empty() => s.push_str("{}"),
        Json::Arr(a) => {
            s.push_str("[\n");
            for (k, x) in a.iter().enumerate() {
                pad(depth + 1, s);
                write_value(x, depth + 1, s);
                if k + 1 < a.len() {
                    s.push(',');
                }
                s.push('\n');
            }
            pad(depth, s);
            s.push(']');
        }
        Json::Obj(m) => {
            s.push_str("{\n");
            for (k, (key, x)) in m.iter().enumerate() {
                pad(depth + 1, s);
                quote(key, s);
                s.push_str(": ");
                write_value(x, depth + 1, s);
                if k + 1 < m.len() {
                    s.push(',');
                }
                s.push('\n');
            }
            pad(depth, s);
            s.push('}');
        }
    }
}

// ---------------------------------------------------------------- net classes

/// One KiCad net class (lengths in nm; `None` when the file leaves the field out).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KiNetClass {
    pub name: String,
    pub clearance: Option<Coord>,
    pub track_width: Option<Coord>,
    pub via_diameter: Option<Coord>,
    pub via_drill: Option<Coord>,
    pub microvia_diameter: Option<Coord>,
    pub microvia_drill: Option<Coord>,
    pub diff_pair_width: Option<Coord>,
    pub diff_pair_gap: Option<Coord>,
    pub diff_pair_via_gap: Option<Coord>,
    /// Members not modelled (colours, wire and bus widths, line style…), kept as written.
    pub other: Vec<(String, Json)>,
}

const LEN_FIELDS: [&str; 9] = [
    "clearance",
    "track_width",
    "via_diameter",
    "via_drill",
    "microvia_diameter",
    "microvia_drill",
    "diff_pair_width",
    "diff_pair_gap",
    "diff_pair_via_gap",
];

impl KiNetClass {
    fn field(&mut self, k: &str) -> Option<&mut Option<Coord>> {
        Some(match k {
            "clearance" => &mut self.clearance,
            "track_width" => &mut self.track_width,
            "via_diameter" => &mut self.via_diameter,
            "via_drill" => &mut self.via_drill,
            "microvia_diameter" => &mut self.microvia_diameter,
            "microvia_drill" => &mut self.microvia_drill,
            "diff_pair_width" => &mut self.diff_pair_width,
            "diff_pair_gap" => &mut self.diff_pair_gap,
            "diff_pair_via_gap" => &mut self.diff_pair_via_gap,
            _ => return None,
        })
    }
    fn get(&self, k: &str) -> Option<Coord> {
        let mut c = self.clone();
        c.field(k).and_then(|f| *f)
    }
    fn from_json(v: &Json) -> KiNetClass {
        let mut c = KiNetClass::default();
        if let Json::Obj(m) = v {
            for (k, x) in m {
                if k == "name" {
                    c.name = x.as_str().unwrap_or("").to_string();
                } else if let Some(f) = c.field(k) {
                    *f = x.as_f64().map(|mm| (mm * 1e6).round() as Coord);
                } else {
                    c.other.push((k.clone(), x.clone()));
                }
            }
        }
        c
    }
    fn to_json(&self) -> Json {
        // KiCad writes members alphabetically; merge ours into that order.
        let mut m: BTreeMap<String, Json> = self.other.iter().cloned().collect();
        m.insert("name".into(), Json::str(&self.name));
        for k in LEN_FIELDS {
            if let Some(v) = self.get(k) {
                m.insert(k.into(), Json::num(v as f64 * 1e-6));
            }
        }
        Json::Obj(m.into_iter().collect())
    }
}

/// The net-class part of a project.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NetSettings {
    pub classes: Vec<KiNetClass>,
    /// `(pattern, class)` in file order; the first match wins.
    pub patterns: Vec<(String, String)>,
    /// Net name → class names (KiCad 8+ allows several; the first is the effective one).
    pub assignments: BTreeMap<String, Vec<String>>,
}

impl NetSettings {
    /// The class of a net: explicit assignment, else the first matching pattern, else
    /// `Default`.
    pub fn class_of(&self, net: &str) -> &str {
        if let Some(c) = self.assignments.get(net).and_then(|v| v.first()) {
            return c;
        }
        for (p, c) in &self.patterns {
            if glob(p, net) {
                return c;
            }
        }
        "Default"
    }

    /// Classes as design net classes (lengths as SI quantities under their KiCad names).
    pub fn to_design(&self) -> BTreeMap<String, NetClass> {
        self.classes
            .iter()
            .map(|c| {
                let mut params = BTreeMap::new();
                for k in LEN_FIELDS {
                    if let Some(v) = c.get(k) {
                        params.insert(
                            k.to_string(),
                            Param::quantity(Quantity::length_m(v as f64 * 1e-9), Tolerance::Exact),
                        );
                    }
                }
                (
                    c.name.clone(),
                    NetClass {
                        name: c.name.clone(),
                        params,
                        source: Default::default(),
                    },
                )
            })
            .collect()
    }

    /// From design net classes (parameters named as in [`KiNetClass`]; lengths in metres).
    pub fn from_design(classes: &BTreeMap<String, NetClass>) -> NetSettings {
        let mut out = NetSettings::default();
        for c in classes.values() {
            let mut k = KiNetClass {
                name: c.name.clone(),
                ..Default::default()
            };
            for f in LEN_FIELDS {
                if let Some(Param {
                    value: ParamValue::Quantity { nominal, .. },
                    ..
                }) = c.params.get(f)
                {
                    if nominal.dim == Dim::LENGTH {
                        *k.field(f).unwrap() = Some((nominal.value * 1e9).round() as Coord);
                    }
                }
            }
            out.classes.push(k);
        }
        out
    }

    /// Board rules equivalent to the classes: per class a `track_width` (and via, diff-pair
    /// values) for its tracks and nets, and clearances between classes as KiCad applies
    /// them, the larger of the two classes' values: one pair rule per class against any
    /// copper, and one more specific rule per pair of classes.
    pub fn rules(&self) -> Vec<RuleSpec> {
        let mut out = Vec::new();
        let sel = |c: &str| format!(":net-class(\"{c}\")");
        for c in &self.classes {
            let n = &c.name;
            if let Some(v) = c.clearance {
                out.push(
                    RuleSpec::pair(
                        &format!("netclass {n} clearance"),
                        &format!("copper{}", sel(n)),
                        "copper",
                    )
                    .set(props::CLEARANCE, RuleValue::Len(v)),
                );
            }
            let mut single =
                RuleSpec::single(&format!("netclass {n}"), &format!("copper{}", sel(n)));
            for (f, p) in [
                ("track_width", props::TRACK_WIDTH),
                ("via_diameter", props::VIA_DIAMETER),
                ("via_drill", props::VIA_DRILL),
                ("diff_pair_width", props::DIFF_WIDTH),
            ] {
                if let Some(v) = c.get(f) {
                    single = single.set(p, RuleValue::Len(v));
                }
            }
            if !single.decls.is_empty() {
                out.push(single);
            }
            if let Some(v) = c.diff_pair_gap {
                out.push(
                    RuleSpec::single(&format!("netclass {n} pair"), &format!("net{}", sel(n)))
                        .set(props::DIFF_GAP, RuleValue::Len(v)),
                );
            }
        }
        for (i, a) in self.classes.iter().enumerate() {
            for b in &self.classes[i + 1..] {
                if let (Some(x), Some(y)) = (a.clearance, b.clearance) {
                    out.push(
                        RuleSpec::pair(
                            &format!("netclass {} / {} clearance", a.name, b.name),
                            &format!("copper{}", sel(&a.name)),
                            &format!("copper{}", sel(&b.name)),
                        )
                        .set(props::CLEARANCE, RuleValue::Len(x.max(y))),
                    );
                }
            }
        }
        out
    }
}

/// `*` and `?` wildcard match (KiCad net-class patterns).
pub fn glob(pattern: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// A project file held for lossless editing.
#[derive(Clone, Debug, PartialEq)]
pub struct Project {
    text: String,
    root: Json,
    edited: bool,
}

impl Project {
    pub fn read(text: &str) -> Result<Project, String> {
        let root = parse_json(text)?;
        if !matches!(root, Json::Obj(_)) {
            return Err("a project file is a JSON object".into());
        }
        Ok(Project {
            text: text.to_string(),
            root,
            edited: false,
        })
    }
    /// A minimal new project.
    pub fn new(name: &str) -> Project {
        let root = Json::Obj(vec![
            (
                "meta".into(),
                Json::Obj(vec![
                    ("filename".into(), Json::str(&format!("{name}.kicad_pro"))),
                    ("version".into(), Json::int(1)),
                ]),
            ),
            (
                "net_settings".into(),
                Json::Obj(vec![
                    ("classes".into(), Json::Arr(vec![])),
                    (
                        "meta".into(),
                        Json::Obj(vec![("version".into(), Json::int(3))]),
                    ),
                    ("net_colors".into(), Json::Null),
                    ("netclass_assignments".into(), Json::Null),
                    ("netclass_patterns".into(), Json::Arr(vec![])),
                ]),
            ),
        ]);
        Project {
            text: String::new(),
            root,
            edited: true,
        }
    }
    pub fn root(&self) -> &Json {
        &self.root
    }
    pub fn net_settings(&self) -> NetSettings {
        let mut out = NetSettings::default();
        let Some(ns) = self.root.get("net_settings") else {
            return out;
        };
        if let Some(Json::Arr(cs)) = ns.get("classes") {
            out.classes = cs.iter().map(KiNetClass::from_json).collect();
        }
        if let Some(Json::Arr(ps)) = ns.get("netclass_patterns") {
            for p in ps {
                if let (Some(c), Some(pat)) = (
                    p.get("netclass").and_then(Json::as_str),
                    p.get("pattern").and_then(Json::as_str),
                ) {
                    out.patterns.push((pat.to_string(), c.to_string()));
                }
            }
        }
        if let Some(Json::Obj(m)) = ns.get("netclass_assignments") {
            for (net, v) in m {
                let classes = match v {
                    Json::Str(c) => vec![c.clone()],
                    Json::Arr(a) => a
                        .iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect(),
                    _ => vec![],
                };
                out.assignments.insert(net.clone(), classes);
            }
        }
        out
    }
    /// Replaces the classes, patterns and assignments; other members stay as they are.
    /// Unmodelled members of a class that keeps its name are kept.
    pub fn set_net_settings(&mut self, s: &NetSettings) {
        let old = self.net_settings();
        if !matches!(self.root.get("net_settings"), Some(Json::Obj(_))) {
            self.root.set("net_settings", Json::Obj(vec![]));
        }
        let ns = self.root.get_mut("net_settings").unwrap();
        let classes = s
            .classes
            .iter()
            .map(|c| {
                let mut c = c.clone();
                if c.other.is_empty() {
                    if let Some(o) = old.classes.iter().find(|o| o.name == c.name) {
                        c.other = o.other.clone();
                    }
                }
                c.to_json()
            })
            .collect();
        ns.set("classes", Json::Arr(classes));
        ns.set(
            "netclass_assignments",
            if s.assignments.is_empty() {
                Json::Null
            } else {
                Json::Obj(
                    s.assignments
                        .iter()
                        .map(|(n, cs)| {
                            let v = if cs.len() == 1 {
                                Json::str(&cs[0])
                            } else {
                                Json::Arr(cs.iter().map(|c| Json::str(c)).collect())
                            };
                            (n.clone(), v)
                        })
                        .collect(),
                )
            },
        );
        ns.set(
            "netclass_patterns",
            Json::Arr(
                s.patterns
                    .iter()
                    .map(|(p, c)| {
                        Json::Obj(vec![
                            ("netclass".into(), Json::str(c)),
                            ("pattern".into(), Json::str(p)),
                        ])
                    })
                    .collect(),
            ),
        );
        self.edited = true;
    }
    /// The file text: the original bytes when nothing was edited.
    pub fn write(&self) -> String {
        if self.edited {
            write_json(&self.root)
        } else {
            self.text.clone()
        }
    }
}

/// Net-class assignments for a board's nets from its own classes (for export).
pub fn assignments_of(board: &crate::layout::Board) -> BTreeMap<String, Vec<String>> {
    board
        .nets
        .iter()
        .filter_map(|n| n.class.clone().map(|c| (n.name.clone(), vec![c])))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRO: &str = r#"{
  "board": {
    "design_settings": {
      "rules": {
        "min_clearance": 0.0
      }
    }
  },
  "meta": {
    "filename": "demo.kicad_pro",
    "version": 1
  },
  "net_settings": {
    "classes": [
      {
        "bus_width": 12,
        "clearance": 0.2,
        "diff_pair_gap": 0.25,
        "diff_pair_via_gap": 0.25,
        "diff_pair_width": 0.2,
        "line_style": 0,
        "microvia_diameter": 0.3,
        "microvia_drill": 0.1,
        "name": "Default",
        "pcb_color": "rgba(0, 0, 0, 0.000)",
        "schematic_color": "rgba(0, 0, 0, 0.000)",
        "track_width": 0.25,
        "via_diameter": 0.8,
        "via_drill": 0.4,
        "wire_width": 6
      },
      {
        "clearance": 0.3,
        "name": "Power",
        "track_width": 0.5
      }
    ],
    "meta": {
      "version": 3
    },
    "net_colors": null,
    "netclass_assignments": {
      "/VBUS": "Power",
      "/SENSE": ["Default", "Analog"]
    },
    "netclass_patterns": [
      {
        "netclass": "Power",
        "pattern": "+*"
      }
    ]
  },
  "text_variables": {"Té": "x\n"}
}
"#;

    #[test]
    fn json_round_trip_and_unedited_bytes() {
        let v = parse_json(PRO).unwrap();
        let again = parse_json(&write_json(&v)).unwrap();
        assert_eq!(v, again);
        let p = Project::read(PRO).unwrap();
        assert_eq!(
            p.write(),
            PRO,
            "unedited file is written back byte for byte"
        );
        assert!(parse_json("{\"a\": [1, 2,]}").is_err());
    }

    #[test]
    fn net_classes_in_and_out() {
        let mut p = Project::read(PRO).unwrap();
        let ns = p.net_settings();
        assert_eq!(ns.classes.len(), 2);
        let d = &ns.classes[0];
        assert_eq!(d.clearance, Some(200_000));
        assert_eq!(d.via_drill, Some(400_000));
        assert!(d.other.iter().any(|(k, _)| k == "pcb_color"));
        assert_eq!(ns.class_of("/VBUS"), "Power");
        assert_eq!(ns.class_of("/SENSE"), "Default");
        assert_eq!(ns.class_of("+3V3"), "Power");
        assert_eq!(ns.class_of("/D0"), "Default");
        // Rules: per-class clearance, widths and the class-pair maximum.
        let rules = ns.rules();
        let pair = rules
            .iter()
            .find(|r| r.id == "netclass Default / Power clearance")
            .unwrap();
        assert_eq!(pair.decls[0].1, RuleValue::Len(300_000));
        let dc = ns.to_design();
        assert_eq!(dc["Power"].params.len(), 2);
        let back = NetSettings::from_design(&dc);
        assert_eq!(back.classes[1].track_width, Some(500_000));
        // Edit: widen Power, drop the pattern; colours and other members are kept.
        let mut ns2 = ns.clone();
        ns2.classes[1].clearance = Some(350_000);
        ns2.patterns.clear();
        p.set_net_settings(&ns2);
        let text = p.write();
        let p2 = Project::read(&text).unwrap();
        let ns3 = p2.net_settings();
        assert_eq!(ns3.classes[1].clearance, Some(350_000));
        assert!(ns3.patterns.is_empty());
        assert_eq!(ns3.assignments, ns.assignments);
        assert!(text.contains("\"pcb_color\": \"rgba(0, 0, 0, 0.000)\""));
        assert!(text.contains("\"min_clearance\": 0.0"));
        assert!(text.contains("\"T\u{e9}\": \"x\\n\""));
        // Key order of untouched members is preserved.
        let i_board = text.find("\"board\"").unwrap();
        let i_meta = text.find("\"meta\"").unwrap();
        assert!(i_board < i_meta);
    }

    #[test]
    fn globs() {
        assert!(glob("+*", "+5V"));
        assert!(glob("/USB_D?", "/USB_DP"));
        assert!(!glob("/USB_D?", "/USB_DPX"));
        assert!(glob("*CLK*", "/sys/CLK_IN"));
        assert!(!glob("GND", "AGND"));
    }
}
