use super::SurfacePlot;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SurfaceColorMapping {
    #[default]
    Scaled,
    Direct,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface() -> SurfacePlot {
        SurfacePlot::new(
            vec![100., 200.],
            vec![300., 400.],
            vec![vec![20., 30.], vec![40., f64::NAN]],
        )
        .unwrap()
    }

    #[test]
    fn scalar_color_extent_is_not_geometric_position_or_explicit_palette_limits() {
        let surface = surface()
            .with_flatten_z(true)
            .with_color_limits(Some((-1., 1.)));
        assert_eq!(
            surface.scalar_color_range(),
            ScalarColorRange::Known(20., 40.)
        );
        assert_eq!(
            surface
                .clone()
                .with_color_mapping(SurfaceColorMapping::Direct)
                .scalar_color_range(),
            ScalarColorRange::NotMapped
        );
        assert_eq!(
            surface
                .with_color_grid(vec![vec![glam::Vec4::ONE; 2]; 2])
                .scalar_color_range(),
            ScalarColorRange::NotMapped
        );
    }

    #[test]
    fn retained_resident_extent_does_not_need_host_payload() {
        let mut surface = surface().with_scalar_color_extent(20., 40.);
        surface.z_data = None;
        assert_eq!(
            surface.scalar_color_range(),
            ScalarColorRange::Known(20., 40.)
        );
        surface.scalar_color_extent = None;
        assert_eq!(surface.scalar_color_range(), ScalarColorRange::Unavailable);
    }

    #[test]
    fn empty_or_nonfinite_color_data_has_no_extent() {
        let empty = SurfacePlot::new(vec![], vec![], vec![]).unwrap();
        assert_eq!(empty.scalar_color_range(), ScalarColorRange::Empty);
        let invalid = SurfacePlot::new(vec![1.], vec![1.], vec![vec![f64::NAN]]).unwrap();
        assert_eq!(invalid.scalar_color_range(), ScalarColorRange::Empty);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarColorRange {
    NotMapped,
    Empty,
    Known(f64, f64),
    Unavailable,
}

impl SurfacePlot {
    /// Retain extrema already computed while preparing resident color data.
    /// This metadata does not request a GPU download.
    pub fn with_scalar_color_extent(mut self, lower: f64, upper: f64) -> Self {
        self.scalar_color_extent = Some((lower, upper));
        self
    }

    pub fn with_color_mapping(mut self, mapping: SurfaceColorMapping) -> Self {
        self.color_mapping = mapping;
        self
    }

    pub fn scalar_color_range(&self) -> ScalarColorRange {
        if self.color_mapping == SurfaceColorMapping::Direct
            || self.color_grid.is_some()
            || self.gpu_color_grid_source.is_some()
        {
            return ScalarColorRange::NotMapped;
        }
        let extent = if let Some(source) = &self.source_c {
            if self.source_c_rgb {
                return ScalarColorRange::NotMapped;
            }
            let mut lower = f64::INFINITY;
            let mut upper = f64::NEG_INFINITY;
            for index in 0..source.len() {
                let value = source
                    .storage()
                    .value_at(index)
                    .expect("validated surface CData")
                    .materialize_f64();
                if value.is_finite() {
                    lower = lower.min(value);
                    upper = upper.max(value);
                }
            }
            Some((lower, upper))
        } else if let Some(rows) = &self.z_data {
            let mut lower = f64::INFINITY;
            let mut upper = f64::NEG_INFINITY;
            for value in rows
                .iter()
                .flatten()
                .copied()
                .filter(|value| value.is_finite())
            {
                lower = lower.min(value);
                upper = upper.max(value);
            }
            Some((lower, upper))
        } else {
            self.scalar_color_extent
        };
        match extent {
            Some((lower, upper)) if lower.is_finite() && upper.is_finite() => {
                ScalarColorRange::Known(lower, upper)
            }
            Some(_) => ScalarColorRange::Empty,
            None => ScalarColorRange::Unavailable,
        }
    }
}
