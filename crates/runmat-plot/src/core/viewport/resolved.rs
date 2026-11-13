use crate::core::BoundingBox;

/// Pixel geometry is generated against ordered display limits, never axis direction.
/// A missing view preserves the domain's existing data-bounds/camera policy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedViewport {
    pub pixels: Option<(u32, u32)>,
    pub view: Option<(f64, f64, f64, f64)>,
}

impl ResolvedViewport {
    pub fn bounds(self, data: BoundingBox) -> BoundingBox {
        let Some((left, right, bottom, top)) = self.view else {
            return data;
        };
        let (left, right, bottom, top) = (left as f32, right as f32, bottom as f32, top as f32);
        if ![left, right, bottom, top].iter().all(|v| v.is_finite())
            || (right - left).abs() < 1e-12
            || (top - bottom).abs() < 1e-12
        {
            return data;
        }
        BoundingBox {
            min: glam::Vec3::new(left.min(right), bottom.min(top), data.min.z),
            max: glam::Vec3::new(left.max(right), bottom.max(top), data.max.z),
        }
    }
}
