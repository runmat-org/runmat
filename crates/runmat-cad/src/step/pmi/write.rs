//! Semantic PMI written into a STEP AP242 file (see the module docs of [`super`]).

use super::{entity_has_datums, modifier_enum, Limits, StepPmi, CHARACTERISTICS, CYLINDRICAL_ZONE};
use crate::step::p21::{encode_string as s, parse, real, Appender, Data, Instance, Param};
use std::collections::BTreeMap;

/// Where the part's shape lives in the file: its `product_definition_shape`, the B-rep
/// representation holding the faces and that representation's context, and the length unit.
pub(super) struct Anchor {
    pub pds: u64,
    pub rep: u64,
    pub ctx: u64,
    /// The length unit instance and how many millimetres one of it is.
    pub unit: Option<(u64, f64)>,
}

const BREP_REPS: &[&str] = &[
    "ADVANCED_BREP_SHAPE_REPRESENTATION",
    "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
    "FACETED_BREP_SHAPE_REPRESENTATION",
    "SHAPE_REPRESENTATION",
];

fn first_ref(i: &Instance, k: usize) -> Option<u64> {
    i.parts
        .first()
        .and_then(|(_, p)| p.get(k))
        .and_then(Param::as_ref)
}

/// The length unit: a complex `(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(prefix,.METRE.))`, or a
/// conversion-based inch.
pub(super) fn length_unit(d: &Data) -> Option<(u64, f64)> {
    for (id, i) in d.of("LENGTH_UNIT") {
        if let Some(si) = i.part("SI_UNIT") {
            let scale = match si.first().and_then(Param::as_enum) {
                Some("MILLI") => 1.0,
                Some("CENTI") => 10.0,
                Some("MICRO") => 1e-3,
                Some("KILO") => 1e6,
                _ => 1000.0,
            };
            return Some((id, scale));
        }
        if let Some(c) = i.part("CONVERSION_BASED_UNIT") {
            let name = c
                .first()
                .and_then(Param::as_str)
                .unwrap_or("")
                .to_ascii_lowercase();
            let scale = match name.as_str() {
                "inch" => 25.4,
                "foot" => 304.8,
                _ => continue,
            };
            return Some((id, scale));
        }
    }
    None
}

/// Every face name and the face instances that carry it.
pub(super) fn named_faces(d: &Data) -> BTreeMap<String, Vec<u64>> {
    let mut out: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for kind in ["ADVANCED_FACE", "FACE_SURFACE"] {
        for (id, i) in d.of(kind) {
            if let Some(n) = i.parts[0]
                .1
                .first()
                .and_then(Param::as_str)
                .filter(|n| !n.is_empty())
            {
                out.entry(n.to_owned()).or_default().push(id);
            }
        }
    }
    out
}

pub(super) fn anchor(d: &Data) -> Result<Anchor, String> {
    let rep_of = |id: u64| {
        d.get(id)
            .and_then(|i| i.name())
            .is_some_and(|n| BREP_REPS.contains(&n))
    };
    // The brep representation: one that a face belongs to through its solid or shell.
    let brep = d
        .instances
        .iter()
        .find(|(_, i)| i.name().is_some_and(|n| BREP_REPS[..3].contains(&n)))
        .map(|(k, _)| *k)
        .ok_or("the file has no B-rep shape representation")?;
    let mut pds = None;
    for (_, sdr) in d.of("SHAPE_DEFINITION_REPRESENTATION") {
        let (Some(def), Some(rep)) = (first_ref(sdr, 0), first_ref(sdr, 1)) else {
            continue;
        };
        let linked = rep == brep
            || d.of("SHAPE_REPRESENTATION_RELATIONSHIP").any(|(_, r)| {
                let p = r.part("SHAPE_REPRESENTATION_RELATIONSHIP").unwrap_or(&[]);
                let (a, b) = (
                    p.get(2).and_then(Param::as_ref),
                    p.get(3).and_then(Param::as_ref),
                );
                (a == Some(rep) && b == Some(brep)) || (a == Some(brep) && b == Some(rep))
            });
        if linked && rep_of(rep) {
            pds = Some(def);
            break;
        }
    }
    let pds = pds.ok_or("no product_definition_shape owns the B-rep")?;
    let ctx = first_ref(d.get(brep).expect("found"), 2).ok_or("the B-rep has no context")?;
    Ok(Anchor {
        pds,
        rep: brep,
        ctx,
        unit: length_unit(d),
    })
}

struct Writer<'a> {
    a: Appender,
    anchor: &'a Anchor,
    faces: &'a BTreeMap<String, Vec<u64>>,
    unit: u64,
    /// File units per millimetre.
    per_mm: f64,
    missing: Vec<String>,
}

impl Writer<'_> {
    fn len(&self, mm: f64) -> String {
        real(mm * self.per_mm)
    }
    /// GISUs from a shape aspect to each named face.
    fn link(&mut self, aspect: u64, names: &[String], what: &str) {
        for n in names {
            match self.faces.get(n) {
                Some(ids) => {
                    for f in ids.clone() {
                        self.a.add(format!(
                            "GEOMETRIC_ITEM_SPECIFIC_USAGE('','',#{aspect},#{},#{f})",
                            self.anchor.rep
                        ));
                    }
                }
                None => self.missing.push(format!("{what}: no face named {n}")),
            }
        }
    }
    fn aspect(&mut self, name: &str, faces: &[String], what: &str) -> u64 {
        let id = self.a.add(format!(
            "SHAPE_ASPECT({},'',#{},.T.)",
            s(name),
            self.anchor.pds
        ));
        self.link(id, faces, what);
        id
    }
    fn measure(&mut self, mm: f64, role: &str) -> u64 {
        self.a.add(format!(
            "(LENGTH_MEASURE_WITH_UNIT() MEASURE_REPRESENTATION_ITEM() MEASURE_WITH_UNIT(LENGTH_MEASURE({}),#{}) REPRESENTATION_ITEM({}))",
            self.len(mm),
            self.unit,
            s(role)
        ))
    }
    /// The nominal value and the tolerance of a size or location.
    fn limits(&mut self, of: u64, l: &Limits) {
        let v = self.measure(l.nominal, "nominal value");
        let sdr = self.a.add(format!(
            "SHAPE_DIMENSION_REPRESENTATION('',(#{v}),#{})",
            self.anchor.ctx
        ));
        self.a.add(format!(
            "DIMENSIONAL_CHARACTERISTIC_REPRESENTATION(#{of},#{sdr})"
        ));
        let range = match &l.fit {
            Some(fit) => {
                let split = fit.find(|c: char| c.is_ascii_digit()).unwrap_or(fit.len());
                self.a.add(format!(
                    "LIMITS_AND_FITS({},'',{},'ISO 286')",
                    s(&fit[..split]),
                    s(&fit[split..])
                ))
            }
            None => {
                let lo = self.measure(l.lower, "lower limit");
                let hi = self.measure(l.upper, "upper limit");
                self.a.add(format!("TOLERANCE_VALUE(#{lo},#{hi})"))
            }
        };
        self.a.add(format!("PLUS_MINUS_TOLERANCE(#{range},#{of})"));
    }
}

/// `step` with `pmi` written into it. Errors when the file has no part shape to attach to;
/// `Ok` carries the problems with individual items (a face name the file doesn't have).
pub fn embed(step: &str, pmi: &StepPmi) -> Result<(String, Vec<String>), String> {
    if pmi.is_empty() {
        return Ok((step.to_owned(), Vec::new()));
    }
    let d = parse(step)?;
    let anchor = anchor(&d)?;
    let faces = named_faces(&d);
    let mut a = Appender::new(&d);
    let (unit, per_mm) = match anchor.unit {
        Some((u, mm_per_unit)) => (u, 1.0 / mm_per_unit),
        None => (
            a.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))"),
            1.0,
        ),
    };
    let mut w = Writer {
        a,
        anchor: &anchor,
        faces: &faces,
        unit,
        per_mm,
        missing: Vec::new(),
    };
    let pds = anchor.pds;

    let mut datum_ids = BTreeMap::new();
    for dt in &pmi.datums {
        let df = w.a.add(format!(
            "DATUM_FEATURE({},'',#{pds},.T.)",
            s(&format!("datum feature {}", dt.label))
        ));
        w.link(df, &dt.faces, &format!("datum {}", dt.label));
        let datum = w.a.add(format!(
            "DATUM({},'',#{pds},.F.,{})",
            s(&dt.label),
            s(&dt.label)
        ));
        w.a.add(format!("SHAPE_ASPECT_RELATIONSHIP('','',#{df},#{datum})"));
        datum_ids.insert(dt.label.clone(), datum);
    }

    for t in &pmi.tolerances {
        let target = w.aspect(&t.name, &t.faces, &t.name);
        let magnitude = w.a.add(format!(
            "LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE({}),#{unit})",
            w.len(t.value)
        ));
        let entity = CHARACTERISTICS
            .iter()
            .find(|(c, _)| *c == t.characteristic)
            .map(|(_, e)| *e)
            .expect("every characteristic");
        let system = if t.datums.is_empty() {
            None
        } else {
            let mut compartments = Vec::new();
            for r in &t.datums {
                let Some(datum) = datum_ids.get(&r.label) else {
                    w.missing
                        .push(format!("{}: datum {} is not declared", t.name, r.label));
                    continue;
                };
                let mods = modifier_enum(r.modifier)
                    .map(|m| format!("(SIMPLE_DATUM_REFERENCE_MODIFIER(.{m}.))"))
                    .unwrap_or_else(|| "$".into());
                compartments.push(w.a.add(format!(
                    "DATUM_REFERENCE_COMPARTMENT('','',#{pds},.F.,#{datum},{mods})"
                )));
            }
            let list = compartments
                .iter()
                .map(|c| format!("#{c}"))
                .collect::<Vec<_>>()
                .join(",");
            Some(w.a.add(format!("DATUM_SYSTEM('','',#{pds},.F.,({list}))")))
        };
        let gt = format!("{},'',#{magnitude},#{target}", s(&t.name));
        let modifier = modifier_enum(t.modifier);
        let inherent = entity_has_datums(t.characteristic);
        let simple = modifier.is_none() && (system.is_some() == inherent);
        let tol = if simple {
            match system {
                Some(ds) => w.a.add(format!("{entity}({gt},(#{ds}))")),
                None => w.a.add(format!("{entity}({gt})")),
            }
        } else {
            let mut parts = vec![
                (entity.to_owned(), String::new()),
                ("GEOMETRIC_TOLERANCE".to_owned(), gt.clone()),
            ];
            if let Some(ds) = system {
                parts.push((
                    "GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE".into(),
                    format!("(#{ds})"),
                ));
            }
            if let Some(m) = modifier {
                parts.push((
                    "GEOMETRIC_TOLERANCE_WITH_MODIFIERS".into(),
                    format!("(.{m}.)"),
                ));
            }
            parts.sort();
            let body = parts
                .iter()
                .map(|(n, p)| format!("{n}({p})"))
                .collect::<Vec<_>>()
                .join(" ");
            w.a.add(format!("({body})"))
        };
        if t.diametral {
            let form =
                w.a.add(format!("TOLERANCE_ZONE_FORM({})", s(CYLINDRICAL_ZONE)));
            w.a.add(format!("TOLERANCE_ZONE('','',#{pds},.F.,(#{tol}),#{form})"));
        }
    }

    for z in &pmi.sizes {
        let aspect = w.aspect(&z.name, &z.faces, &z.name);
        let size =
            w.a.add(format!("DIMENSIONAL_SIZE(#{aspect},{})", s(&z.kind)));
        w.limits(size, &z.limits);
    }

    for l in &pmi.locations {
        let from = w.aspect(&format!("{} from", l.name), &l.from, &l.name);
        let to = w.aspect(&format!("{} to", l.name), &l.to, &l.name);
        let loc = w.a.add(format!(
            "DIMENSIONAL_LOCATION('linear distance',{},#{from},#{to})",
            s(&l.name)
        ));
        w.limits(loc, &l.limits);
    }

    let missing = std::mem::take(&mut w.missing);
    Ok((w.a.finish(step)?, missing))
}
