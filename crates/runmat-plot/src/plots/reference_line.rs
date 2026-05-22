//! Reference line plot implementation for MATLAB-compatible xline/yline.

use crate::core::{BoundingBox, RenderData};
use crate::plots::line::{LinePlot, LineStyle};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceLineOrientation {
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone)]
pub struct ReferenceLine {
    pub orientation: ReferenceLineOrientation,
    pub value: f64,
    pub color: Vec4,
    pub line_width: f32,
    pub line_style: LineStyle,
    pub label: Option<String>,
    pub display_name: Option<String>,
    pub label_orientation: String,
    pub visible: bool,
}

impl ReferenceLine {
    pub fn new(orientation: ReferenceLineOrientation, value: f64) -> Result<Self, String> {
        if !value.is_finite() {
            return Err("reference line coordinate must be finite".to_string());
        }
        Ok(Self {
            orientation,
            value,
            color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            line_width: 1.0,
            line_style: LineStyle::Solid,
            label: None,
            display_name: None,
            label_orientation: "aligned".to_string(),
            visible: true,
        })
    }

    pub fn with_style(mut self, color: Vec4, line_width: f32, line_style: LineStyle) -> Self {
        self.color = color;
        self.line_width = line_width;
        self.line_style = line_style;
        self
    }

    pub fn label_for_legend(&self) -> Option<String> {
        self.display_name.clone().or_else(|| self.label.clone())
    }

    pub fn coordinate_bounds(&self) -> BoundingBox {
        match self.orientation {
            ReferenceLineOrientation::Vertical => BoundingBox::new(
                Vec3::new(self.value as f32, 0.0, 0.0),
                Vec3::new(self.value as f32, 0.0, 0.0),
            ),
            ReferenceLineOrientation::Horizontal => BoundingBox::new(
                Vec3::new(0.0, self.value as f32, 0.0),
                Vec3::new(0.0, self.value as f32, 0.0),
            ),
        }
    }

    pub fn render_data_with_range(
        &self,
        x_range: (f64, f64),
        y_range: (f64, f64),
        viewport_px: Option<(u32, u32)>,
    ) -> RenderData {
        let (x, y) = match self.orientation {
            ReferenceLineOrientation::Vertical => {
                (vec![self.value, self.value], vec![y_range.0, y_range.1])
            }
            ReferenceLineOrientation::Horizontal => {
                (vec![x_range.0, x_range.1], vec![self.value, self.value])
            }
        };
        let mut line = LinePlot::new(x, y)
            .expect("reference line builds finite two-point line")
            .with_style(self.color, self.line_width, self.line_style);
        if let Some(label) = self.label_for_legend() {
            line = line.with_label(label);
        }
        line.render_data_with_viewport(viewport_px)
    }

    pub fn estimated_memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.label.as_ref().map_or(0, String::len)
            + self.display_name.as_ref().map_or(0, String::len)
            + self.label_orientation.len()
    }
}
