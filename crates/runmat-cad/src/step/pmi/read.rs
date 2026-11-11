//! Semantic PMI read from a STEP AP242 file into [`StepPmi`], by face names. Reads what
//! [`super::embed`] writes and the same structures from other AP242 writers (simple or complex
//! instances, simple or complex measures, limits named or in order).

use super::write::{length_unit, named_faces};
use super::{
    Limits, StepDatum, StepLocation, StepPmi, StepSize, StepTolerance, CHARACTERISTICS,
    CYLINDRICAL_ZONE,
};
use crate::pmi::{DatumReference, MaterialCondition};
use crate::step::p21::{parse, Data, Instance, Param};
use std::collections::BTreeMap;

struct Reader<'a> {
    d: &'a Data,
    /// Shape aspect → the faces its GISUs name.
    aspect_faces: BTreeMap<u64, Vec<String>>,
    /// Millimetres per file length unit.
    mm: f64,
}

fn params<'a>(i: &'a Instance, entity: &str) -> &'a [Param] {
    i.part(entity)
        .or_else(|| (!i.is_complex()).then(|| i.parts[0].1.as_slice()))
        .unwrap_or(&[])
}

fn condition(e: Option<&str>) -> MaterialCondition {
    match e {
        Some("MAXIMUM_MATERIAL_REQUIREMENT") => MaterialCondition::Mmc,
        Some("LEAST_MATERIAL_REQUIREMENT") => MaterialCondition::Lmc,
        _ => MaterialCondition::Rfs,
    }
}

impl Reader<'_> {
    fn str_at(&self, i: &Instance, entity: &str, k: usize) -> String {
        params(i, entity)
            .get(k)
            .and_then(Param::as_str)
            .unwrap_or("")
            .to_owned()
    }
    fn faces(&self, aspect: u64) -> Vec<String> {
        self.aspect_faces.get(&aspect).cloned().unwrap_or_default()
    }
    fn aspect_name(&self, aspect: u64) -> String {
        self.d
            .get(aspect)
            .map(|i| self.str_at(i, "SHAPE_ASPECT", 0))
            .unwrap_or_default()
    }
    /// A measure (simple `LENGTH_MEASURE_WITH_UNIT` or the complex measure item) in mm, and
    /// its representation item name when it has one.
    fn measure(&self, id: u64) -> Option<(f64, String)> {
        let i = self.d.get(id)?;
        let p = i
            .part("MEASURE_WITH_UNIT")
            .or_else(|| i.part("LENGTH_MEASURE_WITH_UNIT").filter(|p| !p.is_empty()))?;
        let v = p.first()?.as_f64()?;
        let role = i
            .part("REPRESENTATION_ITEM")
            .and_then(|r| r.first())
            .and_then(Param::as_str)
            .unwrap_or("")
            .to_owned();
        Some((v * self.mm, role))
    }
    /// The nominal value of a size or location and its tolerance.
    fn limits(&self, of: u64) -> Limits {
        let mut l = Limits {
            nominal: 0.0,
            lower: 0.0,
            upper: 0.0,
            fit: None,
        };
        for (_, dcr) in self.d.of("DIMENSIONAL_CHARACTERISTIC_REPRESENTATION") {
            let p = params(dcr, "DIMENSIONAL_CHARACTERISTIC_REPRESENTATION");
            if p.first().and_then(Param::as_ref) != Some(of) {
                continue;
            }
            let Some(sdr) = p.get(1).and_then(Param::as_ref).and_then(|r| self.d.get(r)) else {
                continue;
            };
            for item in params(sdr, "SHAPE_DIMENSION_REPRESENTATION")
                .get(1)
                .map(Param::as_list)
                .unwrap_or(&[])
            {
                if let Some((v, role)) = item.as_ref().and_then(|r| self.measure(r)) {
                    if role.is_empty() || role == "nominal value" {
                        l.nominal = v;
                    }
                }
            }
        }
        for (_, pm) in self.d.of("PLUS_MINUS_TOLERANCE") {
            let p = params(pm, "PLUS_MINUS_TOLERANCE");
            if p.get(1).and_then(Param::as_ref) != Some(of) {
                continue;
            }
            let Some(range) = p
                .first()
                .and_then(Param::as_ref)
                .and_then(|r| self.d.get(r))
            else {
                continue;
            };
            if range.has("LIMITS_AND_FITS") {
                let f = params(range, "LIMITS_AND_FITS");
                let get = |k: usize| f.get(k).and_then(Param::as_str).unwrap_or("");
                l.fit = Some(format!("{}{}{}", get(0), get(1), get(2)));
            } else if range.has("TOLERANCE_VALUE") {
                let t = params(range, "TOLERANCE_VALUE");
                let a = t
                    .first()
                    .and_then(Param::as_ref)
                    .and_then(|r| self.measure(r));
                let b = t
                    .get(1)
                    .and_then(Param::as_ref)
                    .and_then(|r| self.measure(r));
                if let (Some((lo, _)), Some((hi, _))) = (a, b) {
                    l.lower = lo;
                    l.upper = hi;
                }
            }
        }
        l
    }
}

/// The PMI in `step`.
pub fn read(step: &str) -> Result<StepPmi, String> {
    let d = parse(step)?;
    let mut face_name = BTreeMap::new();
    for (n, ids) in named_faces(&d) {
        for id in ids {
            face_name.insert(id, n.clone());
        }
    }
    let mut aspect_faces: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    for (_, g) in d.of("GEOMETRIC_ITEM_SPECIFIC_USAGE") {
        let p = params(g, "GEOMETRIC_ITEM_SPECIFIC_USAGE");
        if let (Some(a), Some(f)) = (
            p.get(2).and_then(Param::as_ref),
            p.get(4).and_then(Param::as_ref),
        ) {
            if let Some(n) = face_name.get(&f) {
                let v = aspect_faces.entry(a).or_default();
                if !v.contains(n) {
                    v.push(n.clone());
                }
            }
        }
    }
    let r = Reader {
        d: &d,
        aspect_faces,
        mm: length_unit(&d).map(|u| u.1).unwrap_or(1.0),
    };
    let mut out = StepPmi::default();

    // Datums, and their features through shape_aspect_relationship.
    let mut label_of: BTreeMap<u64, String> = BTreeMap::new();
    for (id, i) in d.of("DATUM") {
        let p = params(i, "DATUM");
        let label = p
            .get(4)
            .and_then(Param::as_str)
            .or_else(|| p.first().and_then(Param::as_str))
            .unwrap_or("")
            .to_owned();
        label_of.insert(id, label.clone());
        let mut faces = Vec::new();
        for (_, rel) in d.of("SHAPE_ASPECT_RELATIONSHIP") {
            let q = params(rel, "SHAPE_ASPECT_RELATIONSHIP");
            let (a, b) = (
                q.get(2).and_then(Param::as_ref),
                q.get(3).and_then(Param::as_ref),
            );
            let other = if b == Some(id) {
                a
            } else if a == Some(id) {
                b
            } else {
                None
            };
            if let Some(df) = other.filter(|o| d.get(*o).is_some_and(|x| x.has("DATUM_FEATURE"))) {
                faces.extend(r.faces(df));
            }
        }
        out.datums.push(StepDatum { label, faces });
    }

    // Diametral zones by the tolerance they define.
    let mut diametral = std::collections::BTreeSet::new();
    for (_, z) in d.of("TOLERANCE_ZONE") {
        let p = params(z, "TOLERANCE_ZONE");
        let form = p
            .get(5)
            .and_then(Param::as_ref)
            .and_then(|f| d.get(f))
            .map(|f| r.str_at(f, "TOLERANCE_ZONE_FORM", 0));
        if form.as_deref() == Some(CYLINDRICAL_ZONE) {
            diametral.extend(
                p.get(4)
                    .map(Param::as_list)
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(Param::as_ref),
            );
        }
    }

    for (id, i) in &d.instances {
        let Some((c, _)) = CHARACTERISTICS.iter().find(|(_, e)| i.has(e)) else {
            continue;
        };
        let gt = if i.is_complex() {
            i.part("GEOMETRIC_TOLERANCE").unwrap_or(&[])
        } else {
            &i.parts[0].1[..]
        };
        let name = gt.first().and_then(Param::as_str).unwrap_or("").to_owned();
        let value = gt
            .get(2)
            .and_then(Param::as_ref)
            .and_then(|m| r.measure(m))
            .map(|m| m.0)
            .unwrap_or(0.0);
        let target = gt.get(3).and_then(Param::as_ref);
        let systems: Vec<u64> = if i.is_complex() {
            i.part("GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE")
                .and_then(|p| p.first())
                .map(Param::as_list)
                .unwrap_or(&[])
                .iter()
                .filter_map(Param::as_ref)
                .collect()
        } else {
            gt.get(4)
                .map(Param::as_list)
                .unwrap_or(&[])
                .iter()
                .filter_map(Param::as_ref)
                .collect()
        };
        let mut datums = Vec::new();
        for ds in systems {
            let Some(sys) = d.get(ds) else { continue };
            for c in params(sys, "DATUM_SYSTEM")
                .get(4)
                .map(Param::as_list)
                .unwrap_or(&[])
            {
                let Some(comp) = c.as_ref().and_then(|c| d.get(c)) else {
                    continue;
                };
                let p = params(comp, "DATUM_REFERENCE_COMPARTMENT");
                let label = p
                    .get(4)
                    .and_then(Param::as_ref)
                    .and_then(|b| label_of.get(&b))
                    .cloned()
                    .unwrap_or_default();
                let modifier = condition(
                    p.get(5)
                        .map(Param::as_list)
                        .unwrap_or(&[])
                        .iter()
                        .find_map(|m| match m {
                            Param::Typed(_, e) => e.as_enum(),
                            other => other.as_enum(),
                        }),
                );
                datums.push(DatumReference { label, modifier });
            }
        }
        let modifier = condition(
            i.part("GEOMETRIC_TOLERANCE_WITH_MODIFIERS")
                .and_then(|p| p.first())
                .map(Param::as_list)
                .unwrap_or(&[])
                .iter()
                .find_map(|m| {
                    let e = m.as_enum()?;
                    e.ends_with("MATERIAL_REQUIREMENT").then_some(e)
                }),
        );
        out.tolerances.push(StepTolerance {
            name,
            characteristic: *c,
            faces: target.map(|t| r.faces(t)).unwrap_or_default(),
            value,
            diametral: diametral.contains(id),
            modifier,
            datums,
        });
    }

    for (id, i) in d.of("DIMENSIONAL_SIZE") {
        let p = params(i, "DIMENSIONAL_SIZE");
        let aspect = p.first().and_then(Param::as_ref).unwrap_or(0);
        out.sizes.push(StepSize {
            name: r.aspect_name(aspect),
            kind: p.get(1).and_then(Param::as_str).unwrap_or("").to_owned(),
            faces: r.faces(aspect),
            limits: r.limits(id),
        });
    }

    for (id, i) in d.of("DIMENSIONAL_LOCATION") {
        let p = params(i, "DIMENSIONAL_LOCATION");
        let (a, b) = (
            p.get(2).and_then(Param::as_ref).unwrap_or(0),
            p.get(3).and_then(Param::as_ref).unwrap_or(0),
        );
        let mut name = p.get(1).and_then(Param::as_str).unwrap_or("").to_owned();
        if name.is_empty() {
            name = r.aspect_name(a).trim_end_matches(" from").to_owned();
        }
        out.locations.push(StepLocation {
            name,
            from: r.faces(a),
            to: r.faces(b),
            limits: r.limits(id),
        });
    }
    Ok(out)
}
