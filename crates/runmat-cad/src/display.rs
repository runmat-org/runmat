//! The display projection: what a design viewport draws, joined with persistent names.
//!
//! A regenerated part's bodies are tessellated through the kernel's [`Tessellator`] and each
//! face, edge and vertex carries the reference text a command field takes
//! (`feature.body.face.end`, `edges_between(…)`, `vertex_of(…)`), so a pick on the canvas maps
//! straight to a reference and to the Outline nodes (features, bodies, regions) it belongs to.
//! Entity indices match the part index's `bodies[i].faces[j]`.
//!
//! An assembly's display lists one prototype per configured definition and every part
//! occurrence as a world transform over its prototype, so the viewport instances repeated
//! parts. A prototype is a part's mesh when the part is open to the caller, else its
//! configured bounding box.
//!
//! Positions are metres, written as `f32` in flat arrays (`[x, y, z, x, y, z, …]`).

use crate::assembly::model::DefinitionKind;
use crate::assembly::resolve::ResolvedAssembly;
use crate::assembly::source::DesignSource;
use crate::model::sketch::SketchGeometry;
use crate::model::{FeatureKind, PartDocument};
use crate::refs::name::{EntityName, NameTable, TopoKind};
use crate::refs::Reference;
use crate::regen::engine::RegenResult;
use crate::regen::kernel::{BodyMesh, KernelError, TessellationOptions, Tessellator};
use crate::syntax::refs::print_reference;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub mod references;
pub mod wire;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FaceDisplay {
    pub index: u32,
    /// The face's reference text, when its name has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// Every name the face carries (a merged face has several), as reference text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub names: Vec<String>,
    /// The file's face regions that hold it (`region mount : face = …`), by name: what the
    /// Outline's Regions folder and a simulation's named regions call it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<String>,
    /// The face's name for people, from how it was made (`Body side 2`, `Hollow inner wall ·
    /// body side 1`): [`NameTable::label`] of the name its reference shows. Forms, Evaluate
    /// tables and the agent's picks all read faces by this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// The features that made it.
    pub features: Vec<String>,
    #[serde(serialize_with = "wire::f32s", deserialize_with = "wire::f32s_de")]
    pub positions: Vec<f32>,
    #[serde(serialize_with = "wire::u32s", deserialize_with = "wire::u32s_de")]
    pub triangles: Vec<u32>,
    /// The surface's outward unit normal at each position (x, y, z), when the kernel knows it;
    /// empty otherwise (the renderer then averages triangle normals over shared vertices).
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "wire::f32s",
        deserialize_with = "wire::f32s_de"
    )]
    pub normals: Vec<f32>,
    /// The face's surface where the kernel knows it: a plane's normal, a cylinder's axis and
    /// radius (drawing centerlines, mates on axes and faces).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<crate::refs::history::SurfaceKind>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EdgeDisplay {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    pub faces: [u32; 2],
    #[serde(serialize_with = "wire::f32s", deserialize_with = "wire::f32s_de")]
    pub points: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VertexDisplay {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    pub point: [f32; 3],
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyDisplay {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    pub faces: Vec<FaceDisplay>,
    pub edges: Vec<EdgeDisplay>,
    pub vertices: Vec<VertexDisplay>,
    pub triangle_count: u32,
    /// The body's own color (a circuit board's mask, copper, silkscreen and component bodies);
    /// absent: the part's appearance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 4]>,
    /// The sheet metal's flat pattern (`sheet_metal.flat_pattern`), kept apart from the model: the
    /// folded model hides it, its bounds leave it out and an assembly never draws it; the flat
    /// pattern drawing and an on-demand show draw it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub flat_pattern: bool,
    /// A flat pattern's or an unfolded sheet's bend lines on both faces of the blank: segments
    /// as start and end points (x, y, z, metres), six numbers each. Empty for every other body.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "wire::f32s",
        deserialize_with = "wire::f32s_de"
    )]
    pub bend_lines: Vec<f32>,
}

/// A body the kernel could not draw, with why.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DisplayProblem {
    pub body: u32,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PartDisplay {
    pub bodies: Vec<BodyDisplay>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds: Option<Bounds>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub problems: Vec<DisplayProblem>,
    /// Solved sketches on their planes (see [`sketch_display`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sketches: Vec<SketchDisplay>,
    /// Datum planes, axes, points and frames, mate connectors and reference curves as lines
    /// (see [`references::reference_display`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<references::ReferenceDisplay>,
    /// The color the part draws in: its material's appearance ([`crate::appearance`]), sRGB
    /// and opacity. Absent: the CAD face color.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub appearance: Option<[f32; 4]>,
    /// Coarser tessellations of the same bodies, finest first, for a viewer's level of detail:
    /// an assembly's prototypes carry them ([`DISPLAY_LEVELS`]) so a part small on screen
    /// draws with fewer triangles. Faces only; edges and names stay on `bodies`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub levels: Vec<DisplayLevel>,
}

/// One level of detail: every face of every body as one mesh (part coordinates, metres).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DisplayLevel {
    #[serde(serialize_with = "wire::f32s", deserialize_with = "wire::f32s_de")]
    pub positions: Vec<f32>,
    #[serde(serialize_with = "wire::u32s", deserialize_with = "wire::u32s_de")]
    pub triangles: Vec<u32>,
    /// The surface normal at each position, when the kernel knows it.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "wire::f32s",
        deserialize_with = "wire::f32s_de"
    )]
    pub normals: Vec<f32>,
}

/// How much coarser each level of an assembly prototype tessellates than its full display:
/// chords and the size fraction grow by these factors, the angle per segment to 20° and 32°.
pub const DISPLAY_LEVELS: [(f64, f64); 2] = [(4.0, 20.0), (16.0, 32.0)];

/// The chord ceiling the coarse levels scale from (metres): the full display's ceiling on OCCT.
const DISPLAY_CHORD_CEILING: f64 = 1.5e-3;

/// The part's coarser levels ([`DISPLAY_LEVELS`]) under `options` (the full display's).
pub fn display_levels(
    result: &RegenResult,
    kernel: &dyn Tessellator,
    options: &TessellationOptions,
) -> Vec<DisplayLevel> {
    DISPLAY_LEVELS
        .iter()
        .map(|&(scale, degrees)| {
            let coarse = TessellationOptions {
                chord_tolerance: options.chord_tolerance * scale,
                angle_tolerance: degrees.to_radians().max(options.angle_tolerance),
                size_fraction: options.size_fraction * scale,
                // Past the full display's 1.5 mm ceiling, by the same factor: a level drawn
                // small on screen sags well under a pixel.
                chord_max: Some(DISPLAY_CHORD_CEILING * scale),
            };
            let mut level = DisplayLevel::default();
            for (index, body) in result.state.bodies.iter().enumerate() {
                // Levels draw an assembly's occurrences: the folded part only.
                if is_flat_pattern_body(result, index) {
                    continue;
                }
                let Ok(mesh) = kernel.tessellate(&body.handle, &coarse) else {
                    continue;
                };
                for face in &mesh.faces {
                    let base = (level.positions.len() / 3) as u32;
                    level
                        .positions
                        .extend(face.positions.iter().flat_map(|p| p.map(|v| v as f32)));
                    level
                        .triangles
                        .extend(face.triangles.iter().flat_map(|t| t.map(|i| i + base)));
                    if face.normals.len() == face.positions.len() {
                        level
                            .normals
                            .extend(face.normals.iter().flat_map(|n| n.map(|v| v as f32)));
                    }
                }
            }
            if level.normals.len() != level.positions.len() {
                level.normals.clear();
            }
            level
        })
        .collect()
}

/// A sketch plane in part coordinates.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameDisplay {
    pub origin: [f32; 3],
    pub x: [f32; 3],
    pub y: [f32; 3],
    pub z: [f32; 3],
}

/// One solved sketch entity, in sketch-plane coordinates (metres).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SketchEntityDisplay {
    pub id: String,
    /// `point`, `line`, `arc`, `circle` or `spline`.
    pub kind: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub construction: bool,
    /// Solved points by role (`at`, `start`, `end`, `center`, and `p<i>` for a spline's
    /// interior through-points).
    pub points: BTreeMap<String, [f32; 2]>,
    /// A spline's curve sampled as a polyline from start to end (empty for other kinds).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub samples: Vec<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f32>,
    /// Remaining degrees of freedom, where the solver reports them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dof: Option<u32>,
}

/// A sketch as sketch mode draws it: the plane, the solved entities and the constraints as
/// authored (dimensions carry their value expressions).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchDisplay {
    pub id: String,
    pub frame: FrameDisplay,
    pub status: String,
    /// Degrees of freedom the solver left (0 when fully defined); none for explicit sketches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dof: Option<u32>,
    pub entities: Vec<SketchEntityDisplay>,
    pub constraints: Vec<serde_json::Value>,
    /// Constraint indices the solver reported redundant or conflicting.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redundant: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conflicting: Vec<usize>,
}

fn flat(points: &[[f64; 3]]) -> Vec<f32> {
    points.iter().flatten().map(|v| *v as f32).collect()
}

fn reference_text(kind: TopoKind, name: &EntityName) -> Option<String> {
    print_reference(&Reference::Name {
        kind,
        name: name.clone(),
    })
}

fn is_split_side(name: &EntityName) -> bool {
    matches!(
        name,
        EntityName::Generated(g) if matches!(g.tag, crate::refs::name::SourceTag::Side { .. })
    )
}

fn feature_of(name: &EntityName) -> String {
    name.maker().0.clone()
}

fn union(a: Option<Bounds>, lo: [f64; 3], hi: [f64; 3]) -> Bounds {
    let lo = lo.map(|v| v as f32);
    let hi = hi.map(|v| v as f32);
    match a {
        None => Bounds { min: lo, max: hi },
        Some(b) => Bounds {
            min: [0, 1, 2].map(|i| b.min[i].min(lo[i])),
            max: [0, 1, 2].map(|i| b.max[i].max(hi[i])),
        },
    }
}

/// Draws one body: its mesh joined with the names the regeneration gave each face.
fn body_display(
    index: u32,
    mesh: &BodyMesh,
    body_names: &BTreeSet<EntityName>,
    face_names: &[BTreeSet<EntityName>],
    topology: &crate::refs::history::Topology,
    regions: &BTreeMap<&EntityName, Vec<String>>,
    table: &NameTable,
) -> BodyDisplay {
    let face_ref: Vec<Option<String>> = (0..mesh.faces.len())
        .map(|i| {
            face_names
                .get(i)
                .and_then(|names| names.iter().find_map(|n| reference_text(TopoKind::Face, n)))
        })
        .collect();
    let faces = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let names = face_names.get(i);
            let shown = names.and_then(|set| {
                set.iter()
                    .find(|n| {
                        face_ref[i].is_some() && reference_text(TopoKind::Face, n) == face_ref[i]
                    })
                    .or_else(|| set.iter().next())
            });
            FaceDisplay {
                index: i as u32,
                reference: face_ref[i].clone(),
                label: shown.map(|n| table.label(n)),
                regions: names
                    .map(|set| {
                        set.iter()
                            .flat_map(|n| regions.get(n).into_iter().flatten().cloned())
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect()
                    })
                    .unwrap_or_default(),
                names: names
                    .map(|set| {
                        set.iter()
                            .filter_map(|n| reference_text(TopoKind::Face, n))
                            .collect()
                    })
                    .unwrap_or_default(),
                features: names
                    .map(|set| {
                        set.iter()
                            .map(feature_of)
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect()
                    })
                    .unwrap_or_default(),
                positions: flat(&f.positions),
                triangles: f.triangles.iter().flatten().copied().collect(),
                normals: flat(&f.normals),
                surface: topology
                    .faces
                    .get(i)
                    .map(|t| t.geom.surface)
                    .filter(|k| !matches!(k, crate::refs::history::SurfaceKind::Other)),
            }
        })
        .collect();
    let between = |a: u32, b: u32| -> Option<String> {
        let (fa, fb) = (
            face_ref.get(a as usize)?.as_ref()?,
            face_ref.get(b as usize)?.as_ref()?,
        );
        Some(format!("edges_between({fa}, {fb})"))
    };
    let edges = topology
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let [a, b] = e.faces;
            EdgeDisplay {
                index: i as u32,
                reference: between(a, b),
                faces: [a, b],
                points: mesh.edges.get(i).map(|p| flat(p)).unwrap_or_default(),
            }
        })
        .collect();
    let vertices = topology
        .vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            // The faces around the vertex, through its edges.
            let around: BTreeSet<u32> = v
                .edges
                .iter()
                .filter_map(|e| topology.edges.get(*e as usize))
                .flat_map(|e| e.faces)
                .collect();
            let refs: Option<Vec<String>> = around
                .iter()
                .take(3)
                .map(|f| face_ref.get(*f as usize).cloned().flatten())
                .collect();
            VertexDisplay {
                index: i as u32,
                reference: refs
                    .filter(|r| !r.is_empty())
                    .map(|r| format!("vertex_of({})", r.join(", "))),
                point: mesh
                    .vertices
                    .get(i)
                    .copied()
                    .unwrap_or(v.point)
                    .map(|x| x as f32),
            }
        })
        .collect();
    BodyDisplay {
        index,
        // A split piece also carries the split body's name; its side name tells the pieces apart.
        reference: body_names
            .iter()
            .filter(|n| is_split_side(n))
            .chain(body_names.iter())
            .find_map(|n| reference_text(TopoKind::Body, n)),
        faces,
        edges,
        vertices,
        triangle_count: mesh.triangle_count() as u32,
        color: None,
        flat_pattern: false,
        bend_lines: Vec::new(),
    }
}

/// The bend lines of body `index` when a flat pattern or an unfold made it (see
/// [`BodyDisplay::bend_lines`]): each line of its feature's layout on the blank's two faces.
pub fn flat_pattern_bend_lines(result: &RegenResult, index: usize) -> Vec<f32> {
    let Some(body) = result.state.bodies.get(index) else {
        return Vec::new();
    };
    let layout = body
        .names
        .body
        .iter()
        .chain(body.names.faces.iter().flatten())
        .find_map(|name| match name {
            EntityName::Generated(g) => match result.state.records.get(&g.feature) {
                Some(crate::regen::records::FeatureRecord::FlatPattern { layout, .. }) => {
                    Some(layout)
                }
                _ => None,
            },
            _ => None,
        });
    let Some(layout) = layout else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for face in [0.0, layout.depth] {
        let frame = layout.frame.offset(face);
        for line in &layout.bend_lines {
            for p in [line.start, line.end] {
                out.extend(frame.to_world(p).map(|c| c as f32));
            }
        }
    }
    out
}

/// Whether the regeneration's body `index` is a sheet metal flat pattern: the body a
/// `sheet_metal.flat_pattern` feature made.
pub fn is_flat_pattern_body(result: &RegenResult, index: usize) -> bool {
    let Some(body) = result.state.bodies.get(index) else {
        return false;
    };
    let flat = |name: &EntityName| match name {
        EntityName::Generated(g) => matches!(
            result.state.records.get(&g.feature),
            Some(crate::regen::records::FeatureRecord::FlatPattern {
                in_place: false,
                ..
            })
        ),
        _ => false,
    };
    body.names.body.iter().any(flat)
        || (!body.names.faces.is_empty()
            && body.names.faces.iter().all(|names| names.iter().any(flat)))
}

/// The display projection of a regenerated part. A body the kernel cannot draw is listed in
/// `problems` and left out; the others still draw.
pub fn part_display(
    result: &RegenResult,
    kernel: &dyn Tessellator,
    options: &TessellationOptions,
) -> PartDisplay {
    let mut out = PartDisplay::default();
    // Each name the file's regions resolved to, with those regions' names (a face matches
    // through its own face names only).
    let mut regions: BTreeMap<&EntityName, Vec<String>> = BTreeMap::new();
    for (region, outcome) in &result.regions {
        for name in outcome.entities.iter().flatten() {
            regions
                .entry(name)
                .or_default()
                .push(region.as_str().to_string());
        }
    }
    // The flat pattern's bounds count only when nothing folded draws.
    let mut flat_bounds = None;
    for (i, body) in result.state.bodies.iter().enumerate() {
        match kernel.tessellate(&body.handle, options) {
            Ok(mesh) => {
                let flat_pattern = is_flat_pattern_body(result, i);
                if let Some((lo, hi)) = mesh.bounds() {
                    let into = if flat_pattern {
                        &mut flat_bounds
                    } else {
                        &mut out.bounds
                    };
                    *into = Some(union(into.take(), lo, hi));
                }
                let mut display = body_display(
                    i as u32,
                    &mesh,
                    &body.names.body,
                    &body.names.faces,
                    &body.topology,
                    &regions,
                    &result.state.names,
                );
                display.flat_pattern = flat_pattern;
                display.bend_lines = flat_pattern_bend_lines(result, i);
                out.bodies.push(display);
            }
            Err(error) => out.problems.push(DisplayProblem {
                body: i as u32,
                message: display_error(&error),
            }),
        }
    }
    if out.bounds.is_none() {
        out.bounds = flat_bounds;
    }
    out
}

fn display_error(error: &KernelError) -> String {
    error.to_string()
}

// ---------------------------------------------------------------------------- assemblies

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "shape")]
pub enum PrototypeShape {
    /// The part's own display.
    Mesh { part: Box<PartDisplay> },
    /// The configured bounding box (the part is not open to the caller).
    Box { bounds: Bounds },
    /// Nothing to draw: the definition has no geometry summary.
    Empty,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prototype {
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<String>,
    /// The material its occurrences are made of (the part's, or the occurrence's override):
    /// what the canvas shades it by (glass, rubber, a metal, a composite).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    /// The color its occurrences draw in: the material's appearance ([`crate::appearance`]).
    pub appearance: [f32; 4],
    #[serde(flatten)]
    pub shape: PrototypeShape,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccurrenceDisplay {
    /// The display path, the same as the assembly index's occurrence path.
    pub path: String,
    pub uuid: String,
    pub prototype: u32,
    /// World transform: the rotation's rows, then the translation (metres).
    pub transform: [f32; 12],
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AssemblyDisplay {
    pub prototypes: Vec<Prototype>,
    pub occurrences: Vec<OccurrenceDisplay>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds: Option<Bounds>,
}

/// The display projection of a resolved assembly. `parts` holds the displays of part files the
/// caller has regenerated (by file path); other parts draw as their configured boxes.
pub fn assembly_display(
    resolved: &ResolvedAssembly,
    source: &dyn DesignSource,
    parts: &BTreeMap<String, PartDisplay>,
) -> AssemblyDisplay {
    assembly_display_configured(resolved, source, parts, &BTreeMap::new())
}

/// The key of a part occurrence's configuration in [`assembly_display_configured`]'s
/// `configured` map: the file and the resolved configuration as JSON.
pub fn configuration_key(node: &crate::assembly::resolve::OccNode) -> (String, String) {
    (
        node.definition.file.as_str().to_string(),
        serde_json::to_string(&node.configuration).unwrap_or_default(),
    )
}

/// [`assembly_display`] where a part occurrence in a configuration other than the default
/// draws that configuration's display from `configured` (keyed by [`configuration_key`]); the
/// others draw the file's default display from `parts`.
pub fn assembly_display_configured(
    resolved: &ResolvedAssembly,
    source: &dyn DesignSource,
    parts: &BTreeMap<String, PartDisplay>,
    configured: &BTreeMap<(String, String), PartDisplay>,
) -> AssemblyDisplay {
    let mut out = AssemblyDisplay::default();
    let mut prototype_of: BTreeMap<(String, String), u32> = BTreeMap::new();
    for (id, node) in resolved.nodes.iter().enumerate() {
        // Parts and circuit boards are leaves that draw; a board draws its 3D model.
        let drawn = matches!(
            node.definition.kind,
            DefinitionKind::Part | DefinitionKind::Circuit
        );
        if !drawn || node.represented_by_ancestor {
            continue;
        }
        let file = node.definition.file.as_str().to_string();
        let configuration = node.configuration.name.as_ref().map(|c| c.to_string());
        let key = configuration_key(node);
        let prototype = match prototype_of.get(&key) {
            Some(p) => *p,
            None => {
                let shape = match configured.get(&key).or_else(|| parts.get(&file)) {
                    // An occurrence is the folded part: a sheet metal flat pattern stays out.
                    Some(part) => PrototypeShape::Mesh {
                        part: Box::new(PartDisplay {
                            bodies: part
                                .bodies
                                .iter()
                                .filter(|b| !b.flat_pattern)
                                .cloned()
                                .collect(),
                            ..part.clone()
                        }),
                    },
                    None => match source
                        .configured(
                            &node.definition,
                            node.revision.as_ref(),
                            &node.configuration,
                        )
                        .ok()
                        .and_then(|c| c.bbox)
                    {
                        Some(b) => PrototypeShape::Box {
                            bounds: Bounds {
                                min: b.min.map(|v| v as f32),
                                max: b.max.map(|v| v as f32),
                            },
                        },
                        None => PrototypeShape::Empty,
                    },
                };
                // The part's own appearance (its color, else its material's) unless the
                // occurrence overrides the material.
                let own = match &shape {
                    PrototypeShape::Mesh { part } => part.appearance,
                    _ => None,
                };
                let part_material = source
                    .definition(&node.definition, node.revision.as_ref())
                    .ok()
                    .and_then(|s| s.material.clone());
                let appearance = match own {
                    Some(c) if node.material == part_material => c,
                    _ => {
                        crate::appearance::material_appearance(
                            node.material.as_ref().map(|m| m.0.as_str()),
                        )
                        .1
                    }
                };
                out.prototypes.push(Prototype {
                    file,
                    configuration,
                    material: node.material.as_ref().map(|m| m.0.clone()),
                    appearance,
                    shape,
                });
                let p = (out.prototypes.len() - 1) as u32;
                prototype_of.insert(key, p);
                p
            }
        };
        let w = node.world;
        let mut transform = [0.0f32; 12];
        for r in 0..3 {
            for c in 0..3 {
                transform[r * 3 + c] = w.rotation[r][c] as f32;
            }
            transform[9 + r] = w.translation[r] as f32;
        }
        if let Some(local) = prototype_bounds(&out.prototypes[prototype as usize].shape) {
            let corners = (0..8).map(|k| {
                let p = [0, 1, 2].map(|a| {
                    if k & (1 << a) == 0 {
                        local.min[a] as f64
                    } else {
                        local.max[a] as f64
                    }
                });
                w.apply(p)
            });
            for c in corners {
                out.bounds = Some(union(out.bounds.take(), c, c));
            }
        }
        out.occurrences.push(OccurrenceDisplay {
            path: resolved.path(id as u32),
            uuid: node.uuid.to_string(),
            prototype,
            transform,
        });
    }
    out
}

fn prototype_bounds(shape: &PrototypeShape) -> Option<Bounds> {
    match shape {
        PrototypeShape::Mesh { part } => part.bounds.clone(),
        PrototypeShape::Box { bounds } => Some(bounds.clone()),
        PrototypeShape::Empty => None,
    }
}

/// A sketch spline's curve as a polyline (16 spans per through-point), from its solved
/// points; empty when it doesn't solve.
fn spline_samples(
    id: &crate::ids::SketchEntityId,
    count: usize,
    closed: bool,
    solved: &crate::model::sketch::SolvedSketch,
) -> Vec<[f32; 2]> {
    let points: Option<Vec<[f64; 2]>> = SketchGeometry::spline_roles(count, closed)
        .into_iter()
        .map(|r| solved.point(id, r))
        .collect();
    let Some(points) = points else {
        return Vec::new();
    };
    let Some(kind) = runmat_geometry_kernel::interpolate::interpolate(&points, closed) else {
        return Vec::new();
    };
    let segment = runmat_geometry_kernel::op::ProfileSegment {
        curve: id.clone(),
        kind,
        start: points[0],
        end: if closed {
            points[0]
        } else {
            points[points.len() - 1]
        },
    };
    segment
        .polyline(16 * count.max(2))
        .into_iter()
        .map(|p| p.map(|x| x as f32))
        .collect()
}

/// The solved sketches of a regenerated part, in document order.
pub fn sketch_display(document: &PartDocument, result: &RegenResult) -> Vec<SketchDisplay> {
    let f32s = |v: [f64; 3]| v.map(|x| x as f32);
    document
        .features
        .iter()
        .filter_map(|feature| {
            let FeatureKind::Sketch(sketch) = &feature.kind else {
                return None;
            };
            let (solved, frame) = result
                .state
                .sketches
                .get(&feature.id)
                .or_else(|| result.state.over_defined_sketches.get(&feature.id))?;
            let entities = sketch
                .entities
                .iter()
                .map(|entity| {
                    let kind = match entity.geometry {
                        SketchGeometry::Point { .. } => "point",
                        SketchGeometry::Line { .. } => "line",
                        SketchGeometry::Arc { .. } => "arc",
                        SketchGeometry::Circle { .. } => "circle",
                        SketchGeometry::Spline { .. } => "spline",
                    };
                    let points = entity
                        .geometry
                        .points()
                        .into_iter()
                        .filter_map(|(role, _)| {
                            solved
                                .point(&entity.id, role)
                                .map(|p| (role.word(), p.map(|x| x as f32)))
                        })
                        .collect();
                    let samples = match &entity.geometry {
                        SketchGeometry::Spline { points, closed } => {
                            spline_samples(&entity.id, points.len(), *closed, solved)
                        }
                        _ => Vec::new(),
                    };
                    SketchEntityDisplay {
                        id: entity.id.0.clone(),
                        kind: kind.into(),
                        construction: entity.construction,
                        points,
                        samples,
                        radius: solved.radii.get(&entity.id).map(|r| *r as f32),
                        dof: solved.entity_dof.get(&entity.id).copied(),
                    }
                })
                .collect();
            Some(SketchDisplay {
                id: feature.id.0.clone(),
                frame: FrameDisplay {
                    origin: f32s(frame.origin),
                    x: f32s(frame.x),
                    y: f32s(frame.y),
                    z: f32s(frame.z),
                },
                status: serde_json::to_value(solved.status)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default(),
                dof: solved.dof,
                entities,
                constraints: sketch
                    .constraints
                    .iter()
                    .filter_map(|c| serde_json::to_value(c).ok())
                    .collect(),
                redundant: solved.redundant.clone(),
                conflicting: solved.conflicting.clone(),
            })
        })
        .collect()
}

/// The part display as JSON, for the Desktop.
pub fn part_display_json(display: &PartDisplay) -> String {
    serde_json::to_string(display).expect("display serializes")
}

/// The assembly display as JSON, for the Desktop.
pub fn assembly_display_json(display: &AssemblyDisplay) -> String {
    serde_json::to_string(display).expect("display serializes")
}
