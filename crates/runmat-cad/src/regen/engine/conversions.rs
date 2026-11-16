//! A part sketch's converted entities (Use edges): each is read again from the part's bodies as
//! they stand when the sketch regenerates, carried onto the sketch's plane and held fixed, so
//! it follows the edge or face it came from ([`crate::sketch_conversion`]).

use super::*;
use crate::model::sketch::{
    Sketch, SketchConstraint, SketchConstraintKind, SketchProjection, SketchTarget,
};
use crate::sketch_conversion::{
    fit_curve, in_plane, join_offset_corners, offset_on_face, plane_section, silhouette, Curve,
    FaceSurface,
};

/// Fine enough that a cut through a curved face reads as its circle or line: chords within
/// 10 µm and 4° per segment.
const CONVERSION_MESH: TessellationOptions = TessellationOptions {
    chord_tolerance: 1.0e-5,
    angle_tolerance: 4.0 * std::f64::consts::PI / 180.0,
    size_fraction: 5.0e-4,
    chord_max: None,
};

impl<'a, 'b> Run<'a, 'b> {
    /// `sketch` with each converted entity re-read and fixed, in the sketch's `frame`.
    pub(super) fn with_conversions(
        &mut self,
        i: usize,
        f: &FeatureId,
        sketch: &Sketch,
        frame: &Frame,
    ) -> Result<Sketch, Halt> {
        let mut out = sketch.clone();
        let mut meshes: BTreeMap<u32, BodyMesh> = BTreeMap::new();
        // Offsets by the face they cross, with the ends of their edges, to meet at corners.
        let mut offsets: Vec<(String, usize, Curve, [[f64; 2]; 2])> = Vec::new();
        for (k, entity) in sketch.entities.iter().enumerate() {
            let Some(from) = &entity.projected else {
                continue;
            };
            let param = format!("entities.{}.from", entity.id.0);
            out.constraints.push(SketchConstraint {
                id: None,
                kind: SketchConstraintKind::Fix {
                    target: SketchTarget::Entity {
                        entity: entity.id.clone(),
                    },
                },
            });
            if from.is_published() {
                return Err(self.error(
                    f,
                    Some(&param),
                    DiagCode::RefKindMismatch,
                    format!(
                        "`{}` reads a component's published edges, which only an assembly \
                         sketch has; in a part, convert the part's own edges and faces",
                        entity.id
                    ),
                    Vec::new(),
                ));
            }
            // Without a display tessellation the entity keeps its written position.
            let Some(tessellator) = self.ctx.tessellator else {
                continue;
            };
            let mut mesh_of = |run: &mut Self, body: u32| -> Result<BodyMesh, Halt> {
                if let Some(m) = meshes.get(&body) {
                    return Ok(m.clone());
                }
                let handle = run.st.bodies[body as usize].handle.clone();
                let m = tessellator
                    .tessellate(&handle, &CONVERSION_MESH)
                    .map_err(|e| {
                        run.error(
                            f,
                            Some(&param),
                            DiagCode::KernelFailed,
                            e.to_string(),
                            Vec::new(),
                        )
                    })?;
                meshes.insert(body, m.clone());
                Ok(m)
            };
            let fail = |run: &mut Self, message: String| {
                run.error(
                    f,
                    Some(&param),
                    DiagCode::RefKindMismatch,
                    message,
                    Vec::new(),
                )
            };
            let chains: Vec<Vec<[f64; 3]>> = match from {
                SketchProjection::Published { .. } => unreachable!("refused above"),
                SketchProjection::Project { edge } => {
                    let h = self.handle(i, f, &param, edge, TopoKind::Edge)?;
                    let mesh = mesh_of(self, h.body)?;
                    vec![mesh
                        .edges
                        .get(h.index as usize)
                        .cloned()
                        .unwrap_or_default()]
                }
                SketchProjection::Silhouette { face, chain }
                | SketchProjection::Intersection { face, chain } => {
                    let h = self.handle(i, f, &param, face, TopoKind::Face)?;
                    let mesh = mesh_of(self, h.body)?;
                    let surface = Some(
                        self.st.bodies[h.body as usize].topology.faces[h.index as usize]
                            .geom
                            .surface,
                    );
                    let Some(face_mesh) = mesh.faces.get(h.index as usize) else {
                        return Err(fail(self, format!("`{face}` isn't drawn")));
                    };
                    let face_surface = FaceSurface {
                        mesh: face_mesh,
                        surface,
                    };
                    let all = match from {
                        SketchProjection::Silhouette { .. } => silhouette(&face_surface, frame),
                        _ => plane_section(&face_surface, frame),
                    }
                    .map_err(|m| fail(self, format!("`{face}`: {m}")))?;
                    let n = all.len();
                    match all.into_iter().nth(*chain as usize) {
                        Some(c) => vec![c],
                        None => {
                            return Err(fail(
                                self,
                                format!(
                                    "`{face}` gives {n} chain(s) here, so it has no chain {chain}"
                                ),
                            ))
                        }
                    }
                }
                SketchProjection::OffsetOnSurface {
                    edge,
                    face,
                    distance,
                } => {
                    let d = self.value(f, &param, distance, Dimension::LENGTH)?;
                    let e = self.handle(i, f, &param, edge, TopoKind::Edge)?;
                    let fh = self.handle(i, f, &format!("{param}.face"), face, TopoKind::Face)?;
                    if e.body != fh.body {
                        return Err(fail(self, format!("`{edge}` isn't an edge of `{face}`")));
                    }
                    let mesh = mesh_of(self, e.body)?;
                    let topology = &self.st.bodies[e.body as usize].topology;
                    let curve = topology.edges[e.index as usize].geom.curve;
                    let surface = Some(topology.faces[fh.index as usize].geom.surface);
                    let points = mesh
                        .edges
                        .get(e.index as usize)
                        .cloned()
                        .unwrap_or_default();
                    let Some(face_mesh) = mesh.faces.get(fh.index as usize) else {
                        return Err(fail(self, format!("`{face}` isn't drawn")));
                    };
                    let moved = offset_on_face(
                        &points,
                        curve,
                        &FaceSurface {
                            mesh: face_mesh,
                            surface,
                        },
                        d,
                    )
                    .map_err(|m| fail(self, format!("`{edge}`: {m}")))?;
                    let local: Vec<[f64; 2]> = moved.iter().map(|p| in_plane(frame, *p)).collect();
                    let Some(curve) = fit_curve(&local) else {
                        return Err(fail(
                            self,
                            format!("`{edge}` offset projects to a point on this sketch's plane"),
                        ));
                    };
                    let ends = [
                        in_plane(frame, points[0]),
                        in_plane(frame, points[points.len() - 1]),
                    ];
                    offsets.push((format!("{face}"), k, curve, ends));
                    continue;
                }
            };
            let [chain] = chains.as_slice() else {
                unreachable!("one chain per entity");
            };
            let local: Vec<[f64; 2]> = chain.iter().map(|p| in_plane(frame, *p)).collect();
            let Some(curve) = fit_curve(&local) else {
                return Err(fail(
                    self,
                    format!("`{}` projects to a point on this sketch's plane", entity.id),
                ));
            };
            out.entities[k].geometry = curve.geometry();
        }
        // Offsets across one face meet where their edges met.
        let mut faces: Vec<String> = offsets.iter().map(|o| o.0.clone()).collect();
        faces.sort();
        faces.dedup();
        for face in faces {
            let mut group: Vec<(Curve, [[f64; 2]; 2])> = offsets
                .iter()
                .filter(|o| o.0 == face)
                .map(|o| (o.2.clone(), o.3))
                .collect();
            join_offset_corners(&mut group);
            for (o, (curve, _)) in offsets.iter().filter(|o| o.0 == face).zip(group) {
                out.entities[o.1].geometry = curve.geometry();
            }
        }
        Ok(out)
    }

    /// The one entity of `kind` that `r` names, as the sketch's features left it.
    fn handle(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        r: &Reference,
        kind: TopoKind,
    ) -> Result<EntityHandle, Halt> {
        let resolved = self.single(i, f, param, r, Some(kind))?;
        Ok(*resolved.entities().iter().next().expect("exactly one"))
    }
}
