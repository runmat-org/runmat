//! A circuit board as exact STEP AP242 (`board.export`): the board the 3D tab shows, one named
//! product per solid, so an MCAD assembly or an FEA study addresses each by name.
//!
//! - `board`: the substrate, its outline extruded to the stack-up's thickness with its cut-outs
//!   and drilled holes.
//! - `solder mask top` and `solder mask bottom`: the mask's extent on each face, the outline with
//!   its cut-outs and holes, 20 µm thick (on by default).
//! - `copper F.Cu`, `copper B.Cu` (the stack-up's layer names): the outer tracks, via rings and
//!   pads as thin solids at the layer's thickness, standing on each face (off by default: a
//!   mechanical model needs the envelope, not the artwork).
//! - One product per fitted component, named by its reference designator (`U1`): its package's
//!   STEP model when the package names one that is in the project, placed by the package's model
//!   offset, rotation and scale and the footprint's position, rotation and side; otherwise its
//!   body box, as the 3D tab draws it.
//!
//! Millimetres in the file, z = 0 at the board's bottom face, x and y as the layout's. The kernel
//! that writes no STEP (the grid test double) gives `None`.

use crate::board_solid::{prism, ring_loop};
use crate::kernel::DesignKernel;
use crate::runtime::{resolve_path, Files};
use runmat_cad::regen::kernel::{
    BodyHandle, Combine, ExchangeDocument, ExchangeFormat, ExchangePart, KernelOp, Transform,
};
use runmat_circuit::board3d::{colors, PlacedModel};

/// What the export writes beside the substrate and the component bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoardStepOptions {
    /// The outer copper (tracks, via rings, pads) as thin solids.
    pub copper: bool,
    /// The solder mask's extent on each face.
    pub mask: bool,
}

impl Default for BoardStepOptions {
    fn default() -> Self {
        BoardStepOptions {
            copper: false,
            mask: true,
        }
    }
}

/// The STEP text, how each component's body was written, and what the export couldn't write (a
/// model that doesn't read, copper items that don't extrude).
#[derive(Clone, Debug)]
pub struct BoardStep {
    pub step: Vec<u8>,
    /// Components written from their package's STEP model.
    pub models: usize,
    /// Components written as their body box (no model, or one outside the project).
    pub boxes: usize,
    pub notes: Vec<String>,
}

struct Product {
    name: String,
    bodies: Vec<BodyHandle>,
    color: [f64; 4],
}

fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.split('.').next().unwrap_or(name).to_string()
}

fn mm_loops(
    loops: &[Vec<(f64, f64)>],
    name: &str,
) -> Vec<Vec<runmat_cad::regen::kernel::ProfileSegment>> {
    loops
        .iter()
        .enumerate()
        .map(|(i, r)| ring_loop(r, &format!("{name}{i}")))
        .collect()
}

/// The package model `m` as bodies placed on the board, or `None` (with a note) when it isn't in
/// the project or doesn't read.
fn model_bodies(
    kernel: &dyn DesignKernel,
    m: &PlacedModel,
    layout: &str,
    files: &Files,
    designator: &str,
    notes: &mut Vec<String>,
) -> Option<Vec<BodyHandle>> {
    // `${…}/x.step` names a model library outside the project: the body box stands in, as in
    // the 3D tab.
    if m.path.starts_with('$') {
        return None;
    }
    let source = resolve_path(layout, &m.path, files);
    if !files.contains_key(&source) {
        notes.push(format!(
            "{designator}: its model {} isn't in the project; its body box is written",
            m.path
        ));
        return None;
    }
    let read = KernelOp::Import {
        source: source.clone(),
        digest: "unrecorded".into(),
    };
    let mut bodies: Vec<BodyHandle> = match kernel.execute(&read, &[]) {
        Ok(out) if !out.bodies.is_empty() => out.bodies.into_iter().map(|b| b.handle).collect(),
        Ok(_) => {
            notes.push(format!(
                "{designator}: {} holds no solid; its body box is written",
                m.path
            ));
            return None;
        }
        Err(e) => {
            notes.push(format!(
                "{designator}: {} doesn't read ({e}); its body box is written",
                m.path
            ));
            return None;
        }
    };
    let mut step = |op: KernelOp, bodies: Vec<BodyHandle>| -> Option<Vec<BodyHandle>> {
        let mut out = Vec::new();
        for b in &bodies {
            match kernel.execute(&op, &[b]) {
                Ok(r) => out.extend(r.bodies.into_iter().map(|b| b.handle)),
                Err(e) => {
                    notes.push(format!(
                        "{designator}: its model doesn't place ({e}); its body box is written"
                    ));
                    return None;
                }
            }
        }
        Some(out)
    };
    if m.scale != [1.0; 3] {
        bodies = step(
            KernelOp::Scale {
                center: [0.0; 3],
                factors: m.scale,
            },
            bodies,
        )?;
    }
    let transform = Transform {
        rotation: m.rotation,
        translation: m.translation.map(|v| v * 1e-3),
    };
    step(KernelOp::Transform { transform }, bodies)
}

/// The board that `file` (a `.circuit` or its `.circuit.layout`) lays out, as STEP AP242.
pub fn board_step(
    file: &str,
    files: &Files,
    options: BoardStepOptions,
) -> Result<Option<BoardStep>, String> {
    let session = crate::board::compiled_board(file, files)?;
    let solid = session.board_solid().ok_or("no board")?;
    let (copper, mask) = session.board_surfaces().ok_or("no board")?;
    let layout = crate::board::layout_path(file);
    let kernel = crate::kernel::make();
    let k = kernel.as_ref();
    let mut notes = Vec::new();
    let mut products = Vec::new();

    // The substrate, with its cut-outs and drilled holes as inner loops of one profile.
    let mut sheet = solid.outline.clone();
    sheet.extend(solid.holes.iter().cloned());
    let slab = |z0: f64, z1: f64, name: &str, notes: &mut Vec<String>| match prism(
        k,
        mm_loops(&sheet, name),
        z0 * 1e-3,
        z1 * 1e-3,
        Combine::New,
        &[],
    ) {
        Ok(b) => Ok(b),
        Err(e) if !solid.holes.is_empty() => {
            notes.push(format!(
                "{name}: the {} drilled holes don't cut ({e}); it is written without them",
                solid.holes.len()
            ));
            prism(
                k,
                mm_loops(&solid.outline, name),
                z0 * 1e-3,
                z1 * 1e-3,
                Combine::New,
                &[],
            )
            .map_err(|e| format!("the board outline doesn't extrude: {e}"))
        }
        Err(e) => Err(format!("the board outline doesn't extrude: {e}")),
    };
    products.push(Product {
        name: "board".into(),
        bodies: vec![slab(0.0, solid.thickness, "board", &mut notes)?],
        color: colors::SUBSTRATE,
    });
    if options.mask {
        for (name, (z0, z1)) in [
            ("solder mask top", mask[0]),
            ("solder mask bottom", mask[1]),
        ] {
            products.push(Product {
                name: name.into(),
                bodies: vec![slab(z0, z1, name, &mut notes)?],
                color: colors::MASK,
            });
        }
    }
    if options.copper {
        for layer in &copper {
            let name = format!("copper {}", layer.layer);
            let mut bodies = Vec::new();
            let mut lost = 0usize;
            for (i, item) in layer.items.iter().enumerate() {
                let loops = mm_loops(item, &format!("cu{i}_"));
                match prism(
                    k,
                    loops,
                    layer.z0 * 1e-3,
                    layer.z1 * 1e-3,
                    Combine::New,
                    &[],
                ) {
                    Ok(b) => bodies.push(b),
                    Err(_) => lost += 1,
                }
            }
            if lost > 0 {
                notes.push(format!(
                    "{name}: {lost} of {} copper items don't extrude and are left out",
                    layer.items.len()
                ));
            }
            if !bodies.is_empty() {
                products.push(Product {
                    name,
                    bodies,
                    color: colors::FINISH,
                });
            }
        }
    }
    let (mut models, mut boxes) = (0, 0);
    for b in &solid.bodies {
        let model = b
            .model
            .as_ref()
            .and_then(|m| model_bodies(k, m, &layout, files, &b.designator, &mut notes));
        let bodies = match model {
            Some(bodies) => {
                models += 1;
                bodies
            }
            None => match prism(
                k,
                vec![ring_loop(&b.ring, &b.designator)],
                b.z0 * 1e-3,
                b.z1 * 1e-3,
                Combine::New,
                &[],
            ) {
                Ok(body) => {
                    boxes += 1;
                    vec![body]
                }
                Err(e) => {
                    notes.push(format!(
                        "{}: its body doesn't extrude ({e}); left out",
                        b.designator
                    ));
                    continue;
                }
            },
        };
        products.push(Product {
            name: b.designator.clone(),
            bodies,
            color: b.color,
        });
    }
    let doc = ExchangeDocument {
        name: stem(file),
        parts: products
            .iter()
            .map(|p| ExchangePart {
                name: p.name.clone(),
                bodies: p.bodies.iter().collect(),
                color: Some([p.color[0], p.color[1], p.color[2]]),
                material: None,
                face_names: Vec::new(),
                face_colors: Vec::new(),
            })
            .collect(),
        occurrences: Vec::new(),
    };
    Ok(kernel
        .export(&doc, ExchangeFormat::Ap242)
        .ok()
        .map(|step| BoardStep {
            step,
            models,
            boxes,
            notes,
        }))
}
