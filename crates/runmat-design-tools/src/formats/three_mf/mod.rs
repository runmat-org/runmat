//! 3MF (3D Manufacturing Format): the core specification 1.3 with base materials, and the
//! production extension 1.1 (UUIDs on the build, its items, objects and components).
//!
//! - [`write`] packages a [`Model`] as an OPC ZIP: `[Content_Types].xml`, `_rels/.rels` and
//!   `3D/3dmodel.model`.
//! - [`read`] reads a package back into a [`Model`] (stored or deflated entries).
//! - [`check`] checks a model against the core specification's rules: references resolve,
//!   indices are in range, triangles are non-degenerate, and every mesh is a closed, consistently
//!   oriented 2-manifold.
//!
//! Units are the model's (`millimeter` from RunMat). Transforms are the specification's
//! 3x4 matrices: `[m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32]`, applied to row vectors
//! (`p' = [x y z 1] · M`), so `m30..m32` is the translation.

pub mod check;
pub mod read;
pub mod write;

pub use check::check;
pub use read::read;
pub use write::write;

pub const CORE_NS: &str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";
pub const PRODUCTION_NS: &str = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06";
pub const MODEL_REL: &str = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel";
pub const MODEL_CONTENT_TYPE: &str = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml";

/// A 3MF transform (see the module docs).
pub type Transform = [f64; 12];

pub const IDENTITY: Transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

#[derive(Clone, Debug, PartialEq)]
pub struct BaseMaterial {
    pub name: String,
    /// `#RRGGBBAA` (sRGB).
    pub color: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Component {
    pub object: u32,
    pub transform: Option<Transform>,
    pub uuid: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    Mesh {
        vertices: Vec<[f64; 3]>,
        triangles: Vec<[u32; 3]>,
    },
    Components(Vec<Component>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Object {
    pub id: u32,
    pub name: Option<String>,
    pub part_number: Option<String>,
    /// `model`, `solidsupport`, `support`, `surface` or `other`.
    pub kind: String,
    /// The base-material group and index, when the object has one.
    pub material: Option<(u32, u32)>,
    pub uuid: Option<String>,
    pub content: Content,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub object: u32,
    pub transform: Option<Transform>,
    pub part_number: Option<String>,
    pub uuid: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// `micron`, `millimeter`, `centimeter`, `inch`, `foot` or `meter`.
    pub unit: String,
    pub metadata: Vec<(String, String)>,
    /// Base-material groups: (resource id, materials).
    pub materials: Vec<(u32, Vec<BaseMaterial>)>,
    pub objects: Vec<Object>,
    pub build_uuid: Option<String>,
    pub items: Vec<Item>,
}

impl Model {
    pub fn object(&self, id: u32) -> Option<&Object> {
        self.objects.iter().find(|o| o.id == id)
    }

    /// Every build item's triangles in model units, transforms applied through components.
    pub fn world_triangles(&self) -> Vec<[[f64; 3]; 3]> {
        let mut out = Vec::new();
        for item in &self.items {
            self.collect(
                item.object,
                &item.transform.unwrap_or(IDENTITY),
                &mut out,
                0,
            );
        }
        out
    }

    fn collect(&self, id: u32, t: &Transform, out: &mut Vec<[[f64; 3]; 3]>, depth: usize) {
        let Some(o) = self.object(id) else { return };
        if depth > 64 {
            return;
        }
        match &o.content {
            Content::Mesh {
                vertices,
                triangles,
            } => {
                for tri in triangles {
                    out.push(tri.map(|i| apply(t, vertices[i as usize])));
                }
            }
            Content::Components(cs) => {
                for c in cs {
                    let m = compose(&c.transform.unwrap_or(IDENTITY), t);
                    self.collect(c.object, &m, out, depth + 1);
                }
            }
        }
    }
}

/// `p · M` for a row vector `p`.
pub fn apply(m: &Transform, p: [f64; 3]) -> [f64; 3] {
    [
        p[0] * m[0] + p[1] * m[3] + p[2] * m[6] + m[9],
        p[0] * m[1] + p[1] * m[4] + p[2] * m[7] + m[10],
        p[0] * m[2] + p[1] * m[5] + p[2] * m[8] + m[11],
    ]
}

/// `a` then `b` (`p · a · b`).
pub fn compose(a: &Transform, b: &Transform) -> Transform {
    let mut m = [0.0; 12];
    for r in 0..4 {
        for c in 0..3 {
            let mut s = if r == 3 { b[9 + c] } else { 0.0 };
            for k in 0..3 {
                s += a[r * 3 + k] * b[k * 3 + c];
            }
            m[r * 3 + c] = s;
        }
    }
    m
}

/// A 3MF transform from a rotation's rows and a translation (`p' = R p + t`).
pub fn from_rows(r: [[f64; 3]; 3], t: [f64; 3]) -> Transform {
    [
        r[0][0], r[1][0], r[2][0], r[0][1], r[1][1], r[2][1], r[0][2], r[1][2], r[2][2], t[0],
        t[1], t[2],
    ]
}
