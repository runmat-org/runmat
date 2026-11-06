//! Mass properties of parts and assemblies, from their exact bodies and materials.
//!
//! A part's mass is its first output's bodies (the part the file defines; a fluid volume
//! output is left out) at the output's density: its `density` property, else its material's
//! ([`runmat_cad::model::materials`]). The kernel integrates each body's volume, centroid and inertia
//! at unit density, and the density scales them.
//!
//! An assembly's mass properties add every placed part occurrence, each regenerated once per
//! file and configuration and moved to its world placement: the centre of gravity is the
//! mass-weighted centroid, and the inertia tensor about it (world axes) sums each part's
//! rotated tensor and its parallel-axis term. A part with no density is listed, carries no
//! mass, and makes the result incomplete.

use crate::configured::regenerate_configured;
use crate::runtime::{regenerate_part_in, resolve_assembly_in, Files};
use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::assembly::source::ConfigKey;
use runmat_cad::display::part_display;
use runmat_cad::evaluate::{AnalysisRequest, AnalysisResult};
use runmat_cad::model::materials::{part_density, DensitySource};
use runmat_cad::model::PartDocument;
use runmat_cad::regen::kernel::{TessellationOptions, Transform};
use runmat_cad::regen::RegenResult;
use serde::Serialize;
use std::collections::BTreeMap;

/// Mass properties in SI: kg, m, kg·m².
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Mass {
    pub volume: f64,
    pub mass: f64,
    pub center: [f64; 3],
    /// About `center`, rows.
    pub inertia: [[f64; 3]; 3],
}

/// One part occurrence's share of an assembly's mass.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PartMass {
    /// The occurrence path (the part file for a part on its own).
    pub path: String,
    /// What selecting the occurrence selects (`occurrence:<uuid>`, as the Outline and canvas
    /// name it); none for a part on its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item: Option<String>,
    pub file: String,
    pub configuration: String,
    pub material: Option<String>,
    /// kg/m³, when known.
    pub density: Option<f64>,
    pub density_source: Option<DensitySource>,
    pub mass: Option<f64>,
    pub volume: f64,
    /// The occurrence's centre of mass in the assembly's frame (m), so a balance check can
    /// read which parts move the CG.
    pub center: [f64; 3],
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MassReport {
    pub total: Mass,
    pub parts: Vec<PartMass>,
    /// Parts without a density, or whose bodies the kernel could not integrate.
    pub problems: Vec<String>,
}

impl MassReport {
    /// Whether every part carried a mass.
    pub fn complete(&self) -> bool {
        self.problems.is_empty()
    }
}

/// A part's integrated bodies at unit density in its own frame, and its density.
struct Integrated {
    unit: Mass,
    material: Option<String>,
    density: Option<(f64, DensitySource)>,
}

/// The first output's solid bodies of a regenerated part (all of them when it has one output).
/// Surface bodies (sheets, with laminar edges) hold no volume and are left out.
fn output_bodies<'a>(
    document: &PartDocument,
    result: &'a RegenResult,
    kernel: &dyn crate::kernel::DesignKernel,
) -> Vec<&'a runmat_cad::regen::kernel::BodyHandle> {
    let display = part_display(result, kernel, &TessellationOptions::default());
    let first = document.parts.first().map(|o| o.body.to_string());
    display
        .bodies
        .iter()
        .filter(|b| document.parts.len() < 2 || b.reference.as_deref() == first.as_deref())
        .filter_map(|b| result.state.bodies.get(b.index as usize))
        .filter(|s| !s.topology.edges.iter().any(|e| e.laminar))
        .map(|s| &s.handle)
        .collect()
}

fn integrate(
    document: &PartDocument,
    result: &RegenResult,
    kernel: &dyn crate::kernel::DesignKernel,
) -> Result<Integrated, String> {
    let bodies = output_bodies(document, result, kernel);
    if bodies.is_empty() {
        return Err("it has no solid body".into());
    }
    let unit = match kernel.analyze(&AnalysisRequest::MassProperties { density: 1.0 }, &bodies) {
        Ok(AnalysisResult::MassProperties(m)) => Mass {
            volume: m.volume,
            mass: m.volume,
            center: m.centroid,
            inertia: m.inertia,
        },
        Ok(other) => return Err(format!("the kernel answered {other:?}")),
        Err(e) => return Err(format!("the kernel could not integrate it: {e:?}")),
    };
    let output = document.parts.first();
    Ok(Integrated {
        unit,
        material: output
            .and_then(|o| o.material.as_ref())
            .map(|m| m.as_str().to_owned()),
        density: output.and_then(part_density),
    })
}

fn scaled(m: &Mass, density: f64) -> Mass {
    Mass {
        volume: m.volume,
        mass: m.volume * density,
        center: m.center,
        inertia: m.inertia.map(|row| row.map(|v| v * density)),
    }
}

/// `m` moved by `t`: centre placed, tensor rotated (R I Rᵀ).
fn placed(m: &Mass, t: &Transform) -> Mass {
    let r = t.rotation;
    let mut inertia = [[0.0; 3]; 3];
    for (i, row) in inertia.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3)
                .flat_map(|k| (0..3).map(move |l| (k, l)))
                .map(|(k, l)| r[i][k] * m.inertia[k][l] * r[j][l])
                .sum();
        }
    }
    Mass {
        volume: m.volume,
        mass: m.mass,
        center: t.apply(m.center),
        inertia,
    }
}

/// The sum of masses about their common centre (parallel axis theorem).
pub fn combine(parts: &[Mass]) -> Mass {
    let mass: f64 = parts.iter().map(|m| m.mass).sum();
    let volume: f64 = parts.iter().map(|m| m.volume).sum();
    if mass <= 0.0 {
        return Mass {
            volume,
            ..Mass::default()
        };
    }
    let mut center = [0.0; 3];
    for m in parts {
        for (c, x) in center.iter_mut().zip(m.center) {
            *c += m.mass * x / mass;
        }
    }
    let mut inertia = [[0.0; 3]; 3];
    for m in parts {
        let d = [
            m.center[0] - center[0],
            m.center[1] - center[1],
            m.center[2] - center[2],
        ];
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for i in 0..3 {
            for j in 0..3 {
                let delta = if i == j { 1.0 } else { 0.0 };
                inertia[i][j] += m.inertia[i][j] + m.mass * (d2 * delta - d[i] * d[j]);
            }
        }
    }
    Mass {
        volume,
        mass,
        center,
        inertia,
    }
}

fn share(path: String, item: Option<String>, file: String, configuration: String, part: &Integrated, center: [f64; 3]) -> PartMass {
    PartMass {
        path,
        item,
        file,
        configuration,
        material: part.material.clone(),
        density: part.density.map(|d| d.0),
        density_source: part.density.map(|d| d.1),
        mass: part.density.map(|d| part.unit.volume * d.0),
        volume: part.unit.volume,
        center,
    }
}

fn no_density(path: &str, material: Option<&str>) -> String {
    match material {
        Some(m) => format!("{path}: no density for {m}; give the part a `density` property"),
        None => format!("{path}: no material or `density` property, so it has no mass"),
    }
}

/// The mass properties of the part `path` in its named configuration (the default for `None`).
pub fn part_mass(
    path: &str,
    text: &str,
    configuration: Option<&str>,
) -> Result<MassReport, String> {
    let part = regenerate_part_in(path, text, configuration, &TessellationOptions::default())
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let integrated = integrate(&part.document, &part.result, part.kernel.as_ref())?;
    let mut report = MassReport::default();
    let configuration = configuration.unwrap_or("default").to_owned();
    match integrated.density {
        Some((d, _)) => report.total = scaled(&integrated.unit, d),
        None => {
            report
                .problems
                .push(no_density(path, integrated.material.as_deref()));
            report.total = Mass {
                volume: integrated.unit.volume,
                center: integrated.unit.center,
                ..Mass::default()
            };
        }
    }
    report.parts.push(share(
        path.to_owned(),
        None,
        path.to_owned(),
        configuration,
        &integrated,
        integrated.unit.center,
    ));
    Ok(report)
}

/// The mass properties of the assembly `path` in its named configuration, over every placed
/// part occurrence.
pub fn assembly_mass(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
) -> Result<MassReport, String> {
    let resolution = resolve_assembly_in(path, text, files, configuration)
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let resolved = &resolution.resolved;
    let mut integrated: BTreeMap<(String, String), Result<Integrated, String>> = BTreeMap::new();
    let mut report = MassReport::default();
    let mut placed_masses = Vec::new();
    for (id, node) in resolved.nodes.iter().enumerate() {
        if node.parent.is_none() || node.failed || node.represented_by_ancestor {
            continue;
        }
        if node.definition.kind != DefinitionKind::Part {
            continue;
        }
        let file = node.definition.file.0.clone();
        let occurrence = resolved.path(id as u32);
        let key = (file.clone(), node.configuration.digest());
        let entry = integrated.entry(key).or_insert_with(|| {
            let Some(document) = resolution.documents.get(&file) else {
                return Err("its file isn't in the workspace".into());
            };
            let configured = document.in_configuration(node.configuration.name.as_ref());
            let (result, kernel) = regenerate_configured(document, &node.configuration);
            integrate(&configured, &result, kernel.as_ref())
        });
        match entry {
            Err(e) => report.problems.push(format!("{occurrence}: {e}")),
            Ok(part) => {
                report.parts.push(share(
                    occurrence.clone(),
                    Some(format!("occurrence:{}", node.uuid)),
                    file.clone(),
                    if node.configuration == ConfigKey::default() {
                        "default".into()
                    } else {
                        node.configuration.label()
                    },
                    part,
                    node.world.apply(part.unit.center),
                ));
                match part.density {
                    Some((d, _)) => placed_masses.push(placed(&scaled(&part.unit, d), &node.world)),
                    None => report
                        .problems
                        .push(no_density(&occurrence, part.material.as_deref())),
                }
            }
        }
    }
    report.total = combine(&placed_masses);
    report.total.volume = report.parts.iter().map(|p| p.volume).sum();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_point_masses_combine_about_their_centre() {
        let a = Mass {
            volume: 1.0,
            mass: 1.0,
            center: [-1.0, 0.0, 0.0],
            inertia: [[0.0; 3]; 3],
        };
        let b = Mass {
            center: [1.0, 0.0, 0.0],
            ..a.clone()
        };
        let m = combine(&[a, b]);
        assert_eq!(m.mass, 2.0);
        assert_eq!(m.center, [0.0, 0.0, 0.0]);
        // Two unit masses 1 m either side of the centre: 2 kg·m² about y and z, none about x.
        assert_eq!(m.inertia[0][0], 0.0);
        assert_eq!(m.inertia[1][1], 2.0);
        assert_eq!(m.inertia[2][2], 2.0);
    }

    #[test]
    fn a_rotated_part_turns_its_tensor() {
        let m = Mass {
            volume: 1.0,
            mass: 1.0,
            center: [0.0; 3],
            inertia: [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        };
        // A quarter turn about z swaps the x and y moments.
        let t = Transform {
            rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };
        let r = placed(&m, &t);
        assert_eq!(r.inertia[0][0], 2.0);
        assert_eq!(r.inertia[1][1], 1.0);
    }

    const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 10 mm])\n        l3 = line(start = [100 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output block = { body = feature.body.body, material = \"material.steel\" }\n}\n";

    #[test]
    fn a_steel_block_and_two_placed_copies() {
        let part = part_mass("/p/block.part", BLOCK, None).unwrap();
        assert!(part.complete(), "{:?}", part.problems);
        // 100 × 10 × 10 mm of steel: 1e-5 m³ at 7850 kg/m³.
        assert!((part.total.mass - 0.0785).abs() < 1e-3, "{:?}", part.total);
        let assembly = "runmat 0.7\nstd 1\n\nassembly pair {\n    use part block as a\n    use part block as b {\n        placement = { placement = offset, offset = { translation = [0 mm, 100 mm, 0 mm] } }\n    }\n}\n";
        let mut files = Files::new();
        files.insert("/p/block.part".into(), BLOCK.into());
        let report = assembly_mass("/p/pair.assembly", assembly, &files, None).unwrap();
        assert_eq!(report.parts.len(), 2, "{report:?}");
        assert!((report.total.mass - 2.0 * part.total.mass).abs() < 1e-9);
        // The centre sits halfway between the copies.
        assert!(
            (report.total.center[1] - (part.total.center[1] + 0.05)).abs() < 1e-6,
            "{:?}",
            report.total
        );
    }

    #[test]
    fn a_part_without_a_density_says_so() {
        let text = BLOCK.replace(", material = \"material.steel\"", "");
        let report = part_mass("/p/block.part", &text, None).unwrap();
        assert!(!report.complete());
        assert!(
            report.problems[0].contains("no material"),
            "{:?}",
            report.problems
        );
    }
}
