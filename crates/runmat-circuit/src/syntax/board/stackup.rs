//! Stack-ups: built-in materials and the layer stack, read and printed.

use super::*;
use runmat_design_syntax::ast::{self, Headed};

/* --- stackups ------------------------------------------------------------------------------ */

/// A built-in material by its written name.
pub(super) type Builtin = (&'static str, fn() -> Material);

pub(super) const MATERIALS: [Builtin; 4] = [
    ("copper", Material::copper),
    ("fr4", Material::fr4),
    ("polyimide", Material::polyimide),
    ("solder_mask", Material::solder_mask),
];

pub(super) const MATERIAL_KEYS: [&str; 8] = [
    "dk",
    "df",
    "conductivity",
    "k_inplane",
    "k_through",
    "density",
    "cp",
    "cte_ppm",
];

pub(super) fn material_fields(m: &mut Material) -> [&mut f64; 8] {
    [
        &mut m.dk,
        &mut m.df,
        &mut m.conductivity,
        &mut m.k_inplane,
        &mut m.k_through,
        &mut m.density,
        &mut m.cp,
        &mut m.cte_ppm,
    ]
}

pub(super) fn read_material(v: &Val) -> Read<Material> {
    if let V::Word(w) = &v.v {
        return MATERIALS
            .iter()
            .find(|(n, _)| n == w)
            .map(|(_, f)| f())
            .ok_or_else(|| {
                diagnostic(
                    v.span,
                    format!(
                        "`{w}` is not a built-in material; write `{{ name = \"…\", dk = …, … }}`"
                    ),
                )
            });
    }
    let mut f = Fields::of(v, "material")?;
    let mut m = Material {
        name: text(f.need("name")?)?,
        ..Material::fr4()
    };
    for (key, field) in MATERIAL_KEYS.iter().zip(material_fields(&mut m)) {
        *field = num(f.need(key)?)?;
    }
    f.finish()?;
    Ok(m)
}

pub(super) fn material_text(m: &Material) -> String {
    if let Some((n, _)) = MATERIALS.iter().find(|(_, f)| f() == *m) {
        return (*n).to_owned();
    }
    let mut r = Rec::new();
    r.put("name", name_text(&m.name));
    let mut m = m.clone();
    for (key, field) in MATERIAL_KEYS.iter().zip(material_fields(&mut m)) {
        r.put(key, f64_text(*field));
    }
    r.text()
}

pub(super) fn read_stackup(b: &ast::BlockStatement, name: String) -> Read<Stackup> {
    let mut s = Stackup {
        name,
        layers: Vec::new(),
        finish: String::new(),
    };
    for statement in b.block().map(|b| b.statements()).unwrap_or_default() {
        let Some(h) = head(&statement)? else {
            return Err(diagnostic(statement.span(), "expected `layer` or `finish`"));
        };
        match h.word.as_str() {
            words::FINISH if h.name.is_none() => s.finish = text(need_value(&h)?)?,
            words::LAYER => {
                let name = need_name(&h)?;
                let mut f = Fields::of(need_value(&h)?, "stackup layer")?;
                let kv = f.need("kind")?;
                let kind = match text(kv)?.as_str() {
                    "copper" => StackKind::Copper {
                        weight_oz: f.or("weight", 1.0, num)?,
                    },
                    "core" => StackKind::Dielectric { core: true },
                    "prepreg" => StackKind::Dielectric { core: false },
                    "mask" => StackKind::SolderMask,
                    "silk" => StackKind::Silkscreen,
                    "paste" => StackKind::Paste,
                    other => {
                        return Err(diagnostic(
                            kv.span,
                            format!("`{other}` is not a layer kind; expected copper, core, prepreg, mask, silk or paste"),
                        ))
                    }
                };
                let thickness = len(f.need("thickness")?)?;
                let material = read_material(f.need("material")?)?;
                f.finish()?;
                s.layers.push(StackLayer {
                    name,
                    kind,
                    thickness,
                    material,
                });
            }
            other => {
                return Err(diagnostic(
                    h.span,
                    format!("`{other}` does not belong in a stackup; expected `layer` or `finish`"),
                ))
            }
        }
    }
    Ok(s)
}
