//! Volume views of a solved field on its tetrahedral solver mesh: section caps, streamlines,
//! arrow glyphs and the parts of a phasor (amplitude, level, instantaneous value). They are
//! computed once, where the solution lives, and handed to the producer as plain triangles and
//! polylines with one value per vertex, which it bakes into ordinary scene chunks.
//!
//! Values come from the solution itself, never from a smoothed copy:
//! - an element field is constant on each element, so a section shows each element's own value;
//! - a nodal field is evaluated with the element's own basis (linear on Tet4, quadratic on
//!   Tet10, midside nodes included) at every point drawn;
//! - element values are never averaged onto nodes.
//!
//! Each view reports its error: how far the drawn (linearly interpolated) colors can depart
//! from the field between the points where it was evaluated, how far curved quadratic elements
//! depart from the straight-sided tetrahedra used to cut and locate, and for streamlines the
//! estimated integration error.

mod glyph;
mod locate;
mod section;
mod streamline;

pub use glyph::{arrow_glyphs, GlyphOptions, Glyphs};
pub use locate::{Located, TetrahedronLocator};
pub use section::{section_cap, SectionCap, SectionOptions};
pub use streamline::{streamlines, StreamlineOptions, Streamlines};

/// Polynomial order of the mesh's tetrahedra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TetOrder {
    /// Four corner nodes.
    Linear,
    /// Four corners, then six midside nodes on edges 01, 12, 20, 03, 13, 23.
    Quadratic,
}

impl TetOrder {
    pub fn node_count(self) -> usize {
        match self {
            Self::Linear => 4,
            Self::Quadratic => 10,
        }
    }
}

/// Corner pairs of the quadratic midside nodes, in their stored order after the corners.
pub const QUADRATIC_EDGE_CORNERS: [[usize; 2]; 6] =
    [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FieldVolumeError {
    #[error("the mesh has no tetrahedra")]
    Empty,
    #[error("element node list has {actual} entries, not a multiple of {per_element}")]
    ElementLength { per_element: usize, actual: usize },
    #[error("element {element} references node {node}, but the mesh has {nodes} nodes")]
    NodeOutOfRange {
        element: usize,
        node: u32,
        nodes: usize,
    },
    #[error("node {node} has a non-finite position")]
    NonFinitePosition { node: usize },
    #[error("the field has {components} components; views draw scalars and 3-vectors")]
    Components { components: usize },
    #[error("the field has {actual} values for {expected} {support}")]
    FieldLength {
        support: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("the field contains a non-finite value")]
    NonFinite,
}

/// Tetrahedral mesh in metres, nodes in solver order.
#[derive(Debug, Clone, PartialEq)]
pub struct TetMesh {
    positions: Vec<[f64; 3]>,
    order: TetOrder,
    /// `order.node_count()` node indices per element.
    elements: Vec<u32>,
}

impl TetMesh {
    pub fn new(
        positions: Vec<[f64; 3]>,
        order: TetOrder,
        elements: Vec<u32>,
    ) -> Result<Self, FieldVolumeError> {
        let per_element = order.node_count();
        if elements.is_empty() {
            return Err(FieldVolumeError::Empty);
        }
        if elements.len() % per_element != 0 {
            return Err(FieldVolumeError::ElementLength {
                per_element,
                actual: elements.len(),
            });
        }
        if let Some(node) = positions
            .iter()
            .position(|p| !p.iter().all(|c| c.is_finite()))
        {
            return Err(FieldVolumeError::NonFinitePosition { node });
        }
        if let Some(index) = elements
            .iter()
            .position(|&node| node as usize >= positions.len())
        {
            return Err(FieldVolumeError::NodeOutOfRange {
                element: index / per_element,
                node: elements[index],
                nodes: positions.len(),
            });
        }
        Ok(Self {
            positions,
            order,
            elements,
        })
    }

    pub fn order(&self) -> TetOrder {
        self.order
    }

    pub fn node_count(&self) -> usize {
        self.positions.len()
    }

    pub fn element_count(&self) -> usize {
        self.elements.len() / self.order.node_count()
    }

    pub fn element_nodes(&self, element: usize) -> &[u32] {
        let n = self.order.node_count();
        &self.elements[element * n..(element + 1) * n]
    }

    pub fn corners(&self, element: usize) -> [[f64; 3]; 4] {
        let nodes = self.element_nodes(element);
        [0, 1, 2, 3].map(|k| self.positions[nodes[k] as usize])
    }

    /// The point at barycentric coordinates `weights` of the straight-sided tetrahedron.
    pub fn point(&self, element: usize, weights: [f64; 4]) -> [f64; 3] {
        let corners = self.corners(element);
        let mut p = [0.0; 3];
        for (corner, weight) in corners.iter().zip(weights) {
            for axis in 0..3 {
                p[axis] += corner[axis] * weight;
            }
        }
        p
    }

    /// How far a curved quadratic element departs from its straight-sided tetrahedron: the
    /// largest distance from a midside node to the midpoint of its edge (zero for Tet4).
    pub fn curvature_deviation(&self, element: usize) -> f64 {
        if self.order == TetOrder::Linear {
            return 0.0;
        }
        let nodes = self.element_nodes(element);
        QUADRATIC_EDGE_CORNERS
            .iter()
            .enumerate()
            .map(|(edge, [a, b])| {
                let (a, b) = (
                    self.positions[nodes[*a] as usize],
                    self.positions[nodes[*b] as usize],
                );
                let midside = self.positions[nodes[4 + edge] as usize];
                distance(midside, [0, 1, 2].map(|axis| 0.5 * (a[axis] + b[axis])))
            })
            .fold(0.0, f64::max)
    }

    pub fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in &self.positions {
            for axis in 0..3 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
        }
        (min, max)
    }

    /// Mean corner-to-corner edge length over (at most) the first 20,000 elements.
    pub fn mean_edge(&self) -> f64 {
        let (mut total, mut count) = (0.0, 0usize);
        for element in 0..self.element_count().min(20_000) {
            let c = self.corners(element);
            for (a, b) in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)] {
                total += distance(c[a], c[b]);
                count += 1;
            }
        }
        if count == 0 {
            1.0
        } else {
            total / count as f64
        }
    }

    pub fn element_volume(&self, element: usize) -> f64 {
        let c = self.corners(element);
        (dot(sub(c[1], c[0]), cross(sub(c[2], c[0]), sub(c[3], c[0]))) / 6.0).abs()
    }
}

/// Where a field's values live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSupport {
    /// One value set per mesh node, interpolated with the element's basis.
    Node,
    /// One value set per element, constant on it.
    Element,
}

/// A solved field on a [`TetMesh`]: entity-major values (`entity * components + component`),
/// real and, for a phasor, imaginary parts.
#[derive(Debug, Clone, PartialEq)]
pub struct SolvedField {
    pub support: FieldSupport,
    pub components: usize,
    pub real: Vec<f64>,
    pub imaginary: Option<Vec<f64>>,
}

/// The field at one point: up to three components, real and imaginary parts.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PointValue {
    pub components: usize,
    pub real: [f64; 3],
    pub imaginary: [f64; 3],
}

impl SolvedField {
    /// Checks the field against the mesh, so evaluation never indexes out of range.
    pub fn validate(&self, mesh: &TetMesh) -> Result<(), FieldVolumeError> {
        if !matches!(self.components, 1 | 3) {
            return Err(FieldVolumeError::Components {
                components: self.components,
            });
        }
        let (support, entities) = match self.support {
            FieldSupport::Node => ("nodes", mesh.node_count()),
            FieldSupport::Element => ("elements", mesh.element_count()),
        };
        let expected = entities * self.components;
        for values in std::iter::once(&self.real).chain(self.imaginary.iter()) {
            if values.len() != expected {
                return Err(FieldVolumeError::FieldLength {
                    support,
                    expected,
                    actual: values.len(),
                });
            }
            if !values.iter().all(|v| v.is_finite()) {
                return Err(FieldVolumeError::NonFinite);
            }
        }
        Ok(())
    }

    /// The stored value set of one node (nodal field) or element (element field).
    pub fn entity_value(&self, entity: usize) -> PointValue {
        let mut value = PointValue {
            components: self.components,
            ..PointValue::default()
        };
        let start = entity * self.components;
        for component in 0..self.components {
            value.real[component] = self.real[start + component];
            if let Some(imaginary) = &self.imaginary {
                value.imaginary[component] = imaginary[start + component];
            }
        }
        value
    }

    pub fn is_complex(&self) -> bool {
        self.imaginary.is_some()
    }

    /// The field at barycentric coordinates `weights` of `element`: the element's own value for
    /// an element field; the element's basis applied to its nodal values for a nodal field.
    pub fn at(&self, mesh: &TetMesh, element: usize, weights: [f64; 4]) -> PointValue {
        let mut value = PointValue {
            components: self.components,
            ..PointValue::default()
        };
        let mut accumulate = |entity: usize, weight: f64| {
            let start = entity * self.components;
            for component in 0..self.components {
                value.real[component] += weight * self.real[start + component];
                if let Some(imaginary) = &self.imaginary {
                    value.imaginary[component] += weight * imaginary[start + component];
                }
            }
        };
        match self.support {
            FieldSupport::Element => accumulate(element, 1.0),
            FieldSupport::Node => {
                let nodes = mesh.element_nodes(element);
                let (basis, count) = basis(mesh.order, weights);
                for k in 0..count {
                    accumulate(nodes[k] as usize, basis[k]);
                }
            }
        }
        value
    }

    /// The largest magnitude (vector norm, phasor modulus) over the stored value sets: the
    /// field's largest value, since basis interpolation of Tet4 and constant element values
    /// cannot exceed it (a Tet10 interpolant can, slightly, between nodes).
    pub fn largest_magnitude(&self) -> f64 {
        let entities = self.real.len() / self.components.max(1);
        (0..entities)
            .map(|entity| {
                let start = entity * self.components;
                (0..self.components)
                    .map(|c| {
                        let re = self.real[start + c];
                        let im = self.imaginary.as_ref().map_or(0.0, |im| im[start + c]);
                        re * re + im * im
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0, f64::max)
    }
}

/// Shape functions of a tetrahedron at barycentric `l`, and how many there are.
pub fn basis(order: TetOrder, l: [f64; 4]) -> ([f64; 10], usize) {
    let mut n = [0.0; 10];
    match order {
        TetOrder::Linear => {
            n[..4].copy_from_slice(&l);
            (n, 4)
        }
        TetOrder::Quadratic => {
            for corner in 0..4 {
                n[corner] = l[corner] * (2.0 * l[corner] - 1.0);
            }
            for (edge, [a, b]) in QUADRATIC_EDGE_CORNERS.iter().enumerate() {
                n[4 + edge] = 4.0 * l[*a] * l[*b];
            }
            (n, 10)
        }
    }
}

/// What a view draws from the field at each point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reduction {
    /// The vector norm of a real field; the amplitude (modulus) of a phasor, |z| for a scalar
    /// and sqrt(sum |z_i|^2) for a vector.
    Magnitude,
    /// The instantaneous value Re(z e^{i phase}) of a phasor: signed for a scalar, its norm for
    /// a vector. For a real field this is the value itself (signed for a scalar).
    InstantaneousAt { phase_radians: f64 },
    /// Sound pressure level of a pressure phasor: 20 log10(|p| / (sqrt(2) p_ref)) dB, peak
    /// amplitude to RMS.
    SoundPressureLevel { reference_pa: f64 },
}

impl Reduction {
    pub fn apply(self, value: &PointValue) -> f64 {
        let n = value.components;
        match self {
            Self::Magnitude => (0..n)
                .map(|c| value.real[c] * value.real[c] + value.imaginary[c] * value.imaginary[c])
                .sum::<f64>()
                .sqrt(),
            Self::InstantaneousAt { phase_radians } => {
                let (s, c) = phase_radians.sin_cos();
                let instantaneous = |k: usize| value.real[k] * c - value.imaginary[k] * s;
                if n == 1 {
                    instantaneous(0)
                } else {
                    (0..n).map(|k| instantaneous(k).powi(2)).sum::<f64>().sqrt()
                }
            }
            Self::SoundPressureLevel { reference_pa } => {
                let amplitude = Self::Magnitude.apply(value);
                20.0 * (amplitude.max(f64::MIN_POSITIVE)
                    / (std::f64::consts::SQRT_2 * reference_pa))
                    .log10()
            }
        }
    }
}

/// Triangles with one value per vertex.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FieldTriangles {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub values: Vec<f64>,
    pub indices: Vec<u32>,
}

impl FieldTriangles {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub(crate) fn push_vertex(&mut self, position: [f64; 3], normal: [f64; 3], value: f64) -> u32 {
        let index = self.positions.len() as u32;
        self.positions.push(position.map(|v| v as f32));
        self.normals.push(normal.map(|v| v as f32));
        self.values.push(value);
        index
    }

    /// The smallest and largest vertex value.
    pub fn value_range(&self) -> Option<(f64, f64)> {
        extent(self.values.iter().copied())
    }
}

/// A polyline with one value per point.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FieldPolyline {
    pub points: Vec<[f32; 3]>,
    pub values: Vec<f64>,
}

pub(crate) fn extent(values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for value in values {
        min = min.min(value);
        max = max.max(value);
    }
    (min <= max).then_some((min, max))
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn add_scaled(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn normalize(a: [f64; 3]) -> [f64; 3] {
    let length = norm(a);
    if length > 0.0 {
        a.map(|v| v / length)
    } else {
        [0.0, 0.0, 1.0]
    }
}

pub(crate) fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// A unit cube split into six tetrahedra around the main diagonal (Kuhn triangulation).
    pub fn unit_cube(order: TetOrder) -> TetMesh {
        let mut positions: Vec<[f64; 3]> = (0..8)
            .map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64])
            .collect();
        let tetrahedra: [[u32; 4]; 6] = [
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [0, 4, 5, 7],
            [0, 5, 1, 7],
        ];
        let mut elements = Vec::new();
        let mut midsides = std::collections::HashMap::new();
        for tet in tetrahedra {
            elements.extend_from_slice(&tet);
            if order == TetOrder::Quadratic {
                for [a, b] in QUADRATIC_EDGE_CORNERS {
                    let key = (tet[a].min(tet[b]), tet[a].max(tet[b]));
                    let node = *midsides.entry(key).or_insert_with(|| {
                        let (pa, pb) = (positions[tet[a] as usize], positions[tet[b] as usize]);
                        positions.push([0, 1, 2].map(|k| 0.5 * (pa[k] + pb[k])));
                        positions.len() as u32 - 1
                    });
                    elements.push(node);
                }
            }
        }
        TetMesh::new(positions, order, elements).unwrap()
    }

    pub fn nodal(mesh: &TetMesh, f: impl Fn([f64; 3]) -> Vec<f64>) -> SolvedField {
        let values: Vec<Vec<f64>> = mesh.positions.iter().map(|p| f(*p)).collect();
        SolvedField {
            support: FieldSupport::Node,
            components: values[0].len(),
            real: values.concat(),
            imaginary: None,
        }
    }

    pub fn positions(mesh: &TetMesh) -> &[[f64; 3]] {
        &mesh.positions
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn quadratic_basis_reproduces_quadratic_fields_at_any_point() {
        let mesh = unit_cube(TetOrder::Quadratic);
        let f = |p: [f64; 3]| 1.0 + p[0] * p[0] - 2.0 * p[1] * p[2] + 0.5 * p[2];
        let field = nodal(&mesh, |p| vec![f(p)]);
        for element in 0..mesh.element_count() {
            for weights in [[0.25; 4], [0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]] {
                let value = field.at(&mesh, element, weights);
                let exact = f(mesh.point(element, weights));
                assert!(
                    (value.real[0] - exact).abs() < 1e-12,
                    "{} vs {exact}",
                    value.real[0]
                );
            }
        }
    }

    #[test]
    fn element_fields_are_the_element_value_everywhere_in_it() {
        let mesh = unit_cube(TetOrder::Linear);
        let field = SolvedField {
            support: FieldSupport::Element,
            components: 1,
            real: (0..6).map(f64::from).collect(),
            imaginary: None,
        };
        field.validate(&mesh).unwrap();
        for element in 0..6 {
            for weights in [[1.0, 0.0, 0.0, 0.0], [0.25; 4]] {
                assert_eq!(field.at(&mesh, element, weights).real[0], element as f64);
            }
        }
    }

    #[test]
    fn phasor_reductions_follow_their_definitions() {
        let value = PointValue {
            components: 1,
            real: [3.0, 0.0, 0.0],
            imaginary: [4.0, 0.0, 0.0],
        };
        assert!((Reduction::Magnitude.apply(&value) - 5.0).abs() < 1e-12);
        let quarter = Reduction::InstantaneousAt {
            phase_radians: std::f64::consts::FRAC_PI_2,
        };
        assert!((quarter.apply(&value) + 4.0).abs() < 1e-12);
        let level = Reduction::SoundPressureLevel {
            reference_pa: 2.0e-5,
        }
        .apply(&value);
        let expected = 20.0 * (5.0 / (std::f64::consts::SQRT_2 * 2.0e-5)).log10();
        assert!((level - expected).abs() < 1e-9);
    }

    #[test]
    fn validation_rejects_mismatched_fields() {
        let mesh = unit_cube(TetOrder::Linear);
        let mut field = nodal(&mesh, |p| vec![p[0]]);
        field.real.pop();
        assert!(matches!(
            field.validate(&mesh),
            Err(FieldVolumeError::FieldLength { .. })
        ));
        let field = nodal(&mesh, |p| vec![p[0], p[1]]);
        assert!(matches!(
            field.validate(&mesh),
            Err(FieldVolumeError::Components { .. })
        ));
    }
}
