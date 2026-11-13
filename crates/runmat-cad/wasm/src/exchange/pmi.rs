//! A part's semantic PMI to and from STEP AP242: [`step_pmi`] gathers the PMI of a
//! regeneration by the faces it resolved to (named as the STEP writer names them), and
//! [`part_statements`] turns PMI read from a STEP file into `.part` statements on the
//! imported faces.

use runmat_cad::display::PartDisplay;
use runmat_cad::model::PartDocument;
use runmat_cad::pmi::Characteristic;
use runmat_cad::refs::history::SurfaceKind;
use runmat_cad::refs::name::{EntityName, TopoKind};
use runmat_cad::refs::reference::Reference;
use runmat_cad::regen::RegenResult;
use runmat_cad::step::pmi::{Limits, StepDatum, StepLocation, StepPmi, StepSize, StepTolerance};
use runmat_cad::syntax::refs::print_reference;
use runmat_cad::units::StubUnits;
use runmat_cad::value::{Evaluator, NoDynamicValues, ValueEnv, ValueExpr};
use std::fmt::Write;

fn face_names(entities: &[Vec<EntityName>]) -> Vec<String> {
    let mut out = Vec::new();
    for names in entities {
        // One face per resolved entity: the first of its names that prints (the one the
        // display and the STEP writer use).
        if let Some(t) = names.iter().find_map(|n| {
            print_reference(&Reference::Name {
                kind: TopoKind::Face,
                name: n.clone(),
            })
        }) {
            if !out.contains(&t) {
                out.push(t);
            }
        }
    }
    out
}

struct Values<'a> {
    env: &'a ValueEnv,
}

impl Values<'_> {
    fn eval(&self, v: &ValueExpr) -> Option<f64> {
        let ev = Evaluator {
            units: &StubUnits,
            dynamic: &NoDynamicValues,
        };
        ev.eval(v, self.env).ok().map(|q| q.si * 1e3)
    }
    /// Millimetres per unit of a bound written without a unit: the nominal's unit.
    fn bare_unit(&self, nominal: &ValueExpr) -> f64 {
        match nominal {
            ValueExpr::Quantity { value, .. } if *value != 0.0 => {
                self.eval(nominal).map(|mm| mm / value).unwrap_or(1.0)
            }
            _ => 1.0,
        }
    }
    fn bound(&self, b: &ValueExpr, unit: f64) -> f64 {
        match b {
            ValueExpr::Number { value } => value * unit,
            other => self.eval(other).unwrap_or(0.0),
        }
    }
    fn limits(&self, v: &ValueExpr, fit: Option<String>) -> Option<Limits> {
        let nominal = self.eval(v)?;
        let (lower, upper) = match v {
            ValueExpr::Toleranced {
                nominal: n,
                minus,
                plus,
            } => {
                let u = self.bare_unit(n);
                (-self.bound(minus, u).abs(), self.bound(plus, u))
            }
            _ => (0.0, 0.0),
        };
        Some(Limits {
            nominal,
            lower,
            upper,
            fit,
        })
    }
}

/// The size kind CAx-IF names for the faces a size is on.
fn size_kind(faces: &[String], display: &PartDisplay) -> &'static str {
    let cylinder = display.bodies.iter().flat_map(|b| &b.faces).any(|f| {
        f.reference.as_ref().is_some_and(|r| faces.contains(r))
            && matches!(f.surface, Some(SurfaceKind::Cylinder { .. }))
    });
    if cylinder {
        "diameter"
    } else {
        "thickness"
    }
}

/// The PMI of a regenerated part by face names, and what could not be written (an item whose
/// value doesn't evaluate).
pub fn step_pmi(
    doc: &PartDocument,
    result: &RegenResult,
    display: &PartDisplay,
) -> (StepPmi, Vec<String>) {
    let mut out = StepPmi::default();
    let mut notes = Vec::new();
    let v = Values {
        env: &result.values,
    };
    let resolved = &result.pmi;
    for d in &doc.pmi.datums {
        out.datums.push(StepDatum {
            label: d.label.clone(),
            faces: face_names(
                resolved
                    .datums
                    .get(&d.label)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            ),
        });
    }
    for t in &doc.pmi.tolerances {
        let Some(value) = v.eval(&t.tolerance) else {
            notes.push(format!("tolerance {}: its value doesn't evaluate", t.name));
            continue;
        };
        out.tolerances.push(StepTolerance {
            name: t.name.clone(),
            characteristic: t.characteristic,
            faces: face_names(
                resolved
                    .tolerances
                    .get(&t.name)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            ),
            value,
            diametral: t.diametral,
            modifier: t.modifier,
            datums: t.datums.clone(),
        });
    }
    for z in &doc.pmi.sizes {
        let Some(limits) = v.limits(&z.size, z.fit.clone()) else {
            notes.push(format!("size {}: its value doesn't evaluate", z.name));
            continue;
        };
        let faces = face_names(
            resolved
                .sizes
                .get(&z.name)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        );
        out.sizes.push(StepSize {
            name: z.name.clone(),
            kind: size_kind(&faces, display).into(),
            faces,
            limits,
        });
    }
    for l in &doc.pmi.locations {
        let Some(limits) = v.limits(&l.distance, None) else {
            notes.push(format!(
                "location {}: its distance doesn't evaluate",
                l.name
            ));
            continue;
        };
        let [from, to] = resolved.locations.get(&l.name).cloned().unwrap_or_default();
        out.locations.push(StepLocation {
            name: l.name.clone(),
            from: face_names(&from),
            to: face_names(&to),
            limits,
        });
    }
    // The faces by the names the STEP writer gives them (a face in a region travels under the
    // region's name).
    let exported = |faces: &mut Vec<String>| {
        for name in faces.iter_mut() {
            let face = display
                .bodies
                .iter()
                .flat_map(|b| b.faces.iter().map(move |f| (b, f)))
                .find(|(_, f)| f.reference.as_deref() == Some(name.as_str()));
            if let Some(e) = face.and_then(|(b, f)| super::exported_face_name(doc, b, f)) {
                *name = e;
            }
        }
    };
    out.datums.iter_mut().for_each(|d| exported(&mut d.faces));
    out.tolerances
        .iter_mut()
        .for_each(|t| exported(&mut t.faces));
    out.sizes.iter_mut().for_each(|z| exported(&mut z.faces));
    out.locations.iter_mut().for_each(|l| {
        exported(&mut l.from);
        exported(&mut l.to);
    });
    (out, notes)
}

fn mm(v: f64) -> String {
    let s = format!("{:.6}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".into()
    } else {
        s.to_owned()
    }
}

fn ident(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.starts_with(|c: char| c.is_ascii_digit()) || s.is_empty() {
        format!("pmi_{s}")
    } else {
        s
    }
}

fn quoted(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn characteristic(c: Characteristic) -> String {
    serde_json::to_value(c)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn condition(m: runmat_cad::pmi::MaterialCondition) -> String {
    serde_json::to_value(m)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn limits_text(l: &Limits) -> String {
    if l.fit.is_some() || (l.lower == 0.0 && l.upper == 0.0) {
        format!("{} mm", mm(l.nominal))
    } else {
        // The `.part` tolerance form: `± b` when symmetric, else `± [plus, minus]` (magnitudes).
        let (plus, minus) = (mm(l.upper), mm(-l.lower));
        if plus == minus {
            format!("{} mm ± {plus} mm", mm(l.nominal))
        } else {
            format!("{} mm ± [{plus} mm, {minus} mm]", mm(l.nominal))
        }
    }
}

/// `.part` PMI statements for PMI read from a STEP file whose solid is the import feature
/// `feature`: each face by its name in the file (`feature.step.face("…")`).
pub fn part_statements(pmi: &StepPmi, feature: &str) -> String {
    let face = |n: &String| format!("feature.{feature}.face({})", quoted(n));
    let list = |fs: &[String]| format!("[{}]", fs.iter().map(face).collect::<Vec<_>>().join(", "));
    let mut s = String::new();
    for d in &pmi.datums {
        if let Some(f) = d.faces.first() {
            let _ = writeln!(s, "    datum {} = {{ on = {} }}", ident(&d.label), face(f));
        }
    }
    for t in pmi.tolerances.iter().filter(|t| !t.faces.is_empty()) {
        let _ = write!(
            s,
            "    tolerance {} = {{ characteristic = {}, on = {}, tolerance = {} mm",
            ident(&t.name),
            characteristic(t.characteristic),
            list(&t.faces),
            mm(t.value)
        );
        if t.diametral {
            s.push_str(", diametral = true");
        }
        if t.modifier != Default::default() {
            let _ = write!(s, ", modifier = {}", condition(t.modifier));
        }
        if !t.datums.is_empty() {
            let refs: Vec<String> = t
                .datums
                .iter()
                .map(|d| {
                    if d.modifier == Default::default() {
                        format!("{{ label = {} }}", quoted(&d.label))
                    } else {
                        format!(
                            "{{ label = {}, modifier = {} }}",
                            quoted(&d.label),
                            condition(d.modifier)
                        )
                    }
                })
                .collect();
            let _ = write!(s, ", datums = [{}]", refs.join(", "));
        }
        s.push_str(" }\n");
    }
    for z in pmi.sizes.iter().filter(|z| !z.faces.is_empty()) {
        let _ = write!(
            s,
            "    size {} = {{ on = {}, size = {}",
            ident(&z.name),
            list(&z.faces),
            limits_text(&z.limits)
        );
        if let Some(f) = &z.limits.fit {
            let _ = write!(s, ", fit = {}", quoted(f));
        }
        s.push_str(" }\n");
    }
    for l in pmi
        .locations
        .iter()
        .filter(|l| !l.from.is_empty() && !l.to.is_empty())
    {
        let _ = writeln!(
            s,
            "    location {} = {{ from = {}, to = {}, distance = {} }}",
            ident(&l.name),
            face(&l.from[0]),
            face(&l.to[0]),
            limits_text(&l.limits)
        );
    }
    s
}
