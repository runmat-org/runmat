//! GPU instancing: one mesh (a CAD prototype's faces or its edges) drawn at many placements in
//! one draw call, with a per-placement color paint and a level-of-detail band.
//!
//! A scene node that carries [`NodeInstances`] holds its mesh in the prototype's own
//! coordinates. Each frame the renderer keeps the placements whose projected size falls in the
//! node's band ([`NodeInstances::lod_pixels`]) and inside the view, writes them to an instance
//! buffer and draws the mesh once per placement. Payload and vertex memory scale with the
//! prototypes, not the occurrences.
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

/// One placement as the instanced shaders read it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct InstanceRaw {
    /// The placement's 3x4 transform by rows: `p' = (row0·p, row1·p, row2·p)` with `p.w = 1`.
    pub rows: [[f32; 4]; 3],
    /// A face paint: the color moves toward `rgb` by `a` (0 leaves it; 1 fills it). Edge
    /// instances use it the same way with `a = 1` for the selection and hover colors.
    pub tint: [f32; 4],
    /// `x`: the stroke weight of an edge instance (1 normally; above 1 selected or hovered).
    pub extra: [f32; 4],
}

impl InstanceRaw {
    /// A placement from a rotation's rows then a translation (the CAD display's layout).
    pub fn from_rows_translation(t: &[f32; 12]) -> Self {
        Self {
            rows: [
                [t[0], t[1], t[2], t[9]],
                [t[3], t[4], t[5], t[10]],
                [t[6], t[7], t[8], t[11]],
            ],
            tint: [0.0; 4],
            extra: [1.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn matrix(&self) -> Mat4 {
        let [r0, r1, r2] = self.rows;
        Mat4::from_cols(
            Vec4::new(r0[0], r1[0], r2[0], 0.0),
            Vec4::new(r0[1], r1[1], r2[1], 0.0),
            Vec4::new(r0[2], r1[2], r2[2], 0.0),
            Vec4::new(r0[3], r1[3], r2[3], 1.0),
        )
    }

    pub fn apply(&self, p: Vec3) -> Vec3 {
        let row = |r: [f32; 4]| r[0] * p.x + r[1] * p.y + r[2] * p.z + r[3];
        Vec3::new(row(self.rows[0]), row(self.rows[1]), row(self.rows[2]))
    }

    /// Instance attributes at shader locations 4 (row 0) to 8 (extra).
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            4 => Float32x4,
            5 => Float32x4,
            6 => Float32x4,
            7 => Float32x4,
            8 => Float32x4
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRIBUTES,
        }
    }
}

/// A node's placements, with what the per-frame selection needs.
#[derive(Debug, Clone, Default)]
pub struct NodeInstances {
    pub raw: Vec<InstanceRaw>,
    /// Each placement's bounding sphere in world coordinates (center, radius).
    pub spheres: Vec<(Vec3, f32)>,
    /// The band of projected diameters, in device pixels, the node draws placements in:
    /// `[min, max)`. `None` draws every placement in view.
    pub lod_pixels: Option<[f32; 2]>,
}

impl NodeInstances {
    /// Placements of a mesh whose own bounding sphere is `local` (center, radius).
    pub fn new(raw: Vec<InstanceRaw>, local: (Vec3, f32), lod_pixels: Option<[f32; 2]>) -> Self {
        let spheres = raw.iter().map(|i| (i.apply(local.0), local.1)).collect();
        Self {
            raw,
            spheres,
            lod_pixels,
        }
    }

    /// The placements to draw this frame: inside the view and, when the node has a band, with
    /// a projected diameter in it. `view_proj` includes the axes' model matrix; `viewport` is
    /// in device pixels.
    pub fn visible(&self, view_proj: Mat4, viewport: Vec2) -> Vec<InstanceRaw> {
        // The clip-space y row's length is the projection's vertical scale (the view's rows are
        // unit vectors), so a sphere of radius r at clip w spans r·scale/w of half the viewport.
        let row_y = view_proj.row(1).truncate().length();
        let row_x = view_proj.row(0).truncate().length();
        let half_h = viewport.y.max(1.0) * 0.5;
        let mut out = Vec::with_capacity(self.raw.len());
        for (raw, &(center, radius)) in self.raw.iter().zip(&self.spheres) {
            let clip = view_proj * center.extend(1.0);
            // In front of the eye (perspective) or any depth (parallel, w = 1).
            let w = clip.w;
            if w <= -radius * row_y.max(row_x) {
                continue;
            }
            let w_safe = w.abs().max(1.0e-6);
            // Outside the view by more than the sphere's own size.
            let reach_x = radius * row_x;
            let reach_y = radius * row_y;
            if w > 0.0
                && (clip.x - reach_x > w
                    || clip.x + reach_x < -w
                    || clip.y - reach_y > w
                    || clip.y + reach_y < -w)
            {
                continue;
            }
            if let Some([min, max]) = self.lod_pixels {
                let diameter = 2.0 * radius * row_y / w_safe * half_h;
                if w > 0.0 && (diameter < min || diameter >= max) {
                    continue;
                }
            }
            out.push(*raw);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f32) -> InstanceRaw {
        InstanceRaw::from_rows_translation(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, x, 0.0, 0.0])
    }

    #[test]
    fn a_placement_moves_points_by_its_rows_and_translation() {
        // A quarter turn about z, then 2 m along x.
        let r = InstanceRaw::from_rows_translation(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]);
        let p = r.apply(Vec3::X);
        assert!((p - Vec3::new(2.0, 1.0, 0.0)).length() < 1e-6, "{p}");
        let m = r.matrix().transform_point3(Vec3::X);
        assert!((m - p).length() < 1e-6);
    }

    #[test]
    fn placements_draw_only_inside_their_pixel_band_and_the_view() {
        let proj = Mat4::perspective_rh(45f32.to_radians(), 1.0, 0.1, 1000.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let vp = proj * view;
        // A 0.1 m sphere 10 m away on a 1000 px tall view: 2·0.1·2.414/10·500 ≈ 24 px.
        let set = NodeInstances::new(vec![at(0.0), at(500.0)], (Vec3::ZERO, 0.1), Some([10.0, 40.0]));
        let shown = set.visible(vp, Vec2::new(1000.0, 1000.0));
        assert_eq!(shown, vec![at(0.0)], "the far-off placement is out of view");
        let coarse = NodeInstances::new(vec![at(0.0)], (Vec3::ZERO, 0.1), Some([40.0, f32::INFINITY]));
        assert!(coarse.visible(vp, Vec2::new(1000.0, 1000.0)).is_empty(), "24 px is under the band");
        let all = NodeInstances::new(vec![at(0.0)], (Vec3::ZERO, 0.1), None);
        assert_eq!(all.visible(vp, Vec2::new(1000.0, 1000.0)).len(), 1);
    }
}
