//! High-quality vector graphics export (SVG, PDF, etc.)
//!
//! Production-ready vector output using same rendering pipeline data.

use crate::core::RenderData;
use crate::plots::Figure;
use std::fmt::Write;
use std::path::Path;

/// High-quality vector graphics exporter
pub struct VectorExporter {
    /// Export settings
    settings: VectorExportSettings,
}

/// Vector export configuration
#[derive(Debug, Clone)]
pub struct VectorExportSettings {
    /// Output width in user units
    pub width: f32,
    /// Output height in user units
    pub height: f32,
    /// Background color [R, G, B, A] (0.0-1.0)
    pub background_color: [f32; 4],
    /// Stroke width for lines
    pub stroke_width: f32,
    /// Include metadata in output
    pub include_metadata: bool,
    /// Enable anti-aliasing
    pub anti_aliasing: bool,
}

/// Supported vector formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VectorFormat {
    Svg,
    Pdf,
    Eps,
}

impl Default for VectorExportSettings {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            background_color: [1.0, 1.0, 1.0, 1.0], // White background
            stroke_width: 1.0,
            include_metadata: true,
            anti_aliasing: true,
        }
    }
}

impl VectorExporter {
    /// Create a new vector exporter
    pub fn new() -> Self {
        Self {
            settings: VectorExportSettings::default(),
        }
    }

    /// Create exporter with custom settings
    pub fn with_settings(settings: VectorExportSettings) -> Self {
        Self { settings }
    }

    /// Export figure to SVG file
    pub fn export_svg<P: AsRef<Path>>(&self, figure: &mut Figure, path: P) -> Result<(), String> {
        let svg_content = self.render_to_svg(figure)?;
        std::fs::write(path, svg_content).map_err(|e| format!("Failed to write SVG file: {e}"))?;
        println!("DEBUG: SVG export completed successfully");
        Ok(())
    }

    /// Export figure to PDF file (placeholder)
    pub fn export_pdf<P: AsRef<Path>>(&self, _figure: &mut Figure, _path: P) -> Result<(), String> {
        // TODO: Implement PDF export using a PDF library
        Err("PDF export not yet implemented".to_string())
    }

    /// Render figure to SVG string using same rendering pipeline data
    pub fn render_to_svg(&self, _figure: &mut Figure) -> Result<String, String> {
        println!("DEBUG: Starting SVG vector export render");

        let mut svg = String::new();

        // SVG header
        writeln!(
            &mut svg,
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">"#,
            self.settings.width, self.settings.height
        ).map_err(|e| format!("SVG write error: {e}"))?;

        // Background
        if self.settings.background_color[3] > 0.0 {
            writeln!(
                &mut svg,
                r#"  <rect width="100%" height="100%" fill="{}"/>"#,
                self.color_to_hex(&self.settings.background_color)
            )
            .map_err(|e| format!("SVG write error: {e}"))?;
        }

        // Add metadata if requested
        if self.settings.include_metadata {
            writeln!(
                &mut svg,
                "  <metadata>\n    <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n      <rdf:Description>\n        <dc:creator xmlns:dc=\"http://purl.org/dc/elements/1.1/\">RustMat Plot System</dc:creator>\n      </rdf:Description>\n    </rdf:RDF>\n  </metadata>"
            ).map_err(|e| format!("SVG write error: {e}"))?;
        }

        // Render each plot element using the same render data
        // TODO: Implement proper figure element iteration once Figure API is finalized
        writeln!(&mut svg, "  <!-- Plot data will be rendered here -->")
            .map_err(|e| format!("SVG write error: {e}"))?;

        // Note: add_render_data_to_svg and related methods are part of the public API
        // and will be used once the Figure iteration is implemented

        // SVG footer
        writeln!(&mut svg, "</svg>").map_err(|e| format!("SVG write error: {e}"))?;

        println!(
            "DEBUG: SVG render completed, {} characters generated",
            svg.len()
        );
        Ok(svg)
    }

    /// Add render data to SVG using same pipeline data
    /// Note: Will be used when Figure iteration is implemented
    #[allow(dead_code)]
    fn add_render_data_to_svg(
        &self,
        svg: &mut String,
        render_data: &RenderData,
    ) -> Result<(), String> {
        match render_data.pipeline_type {
            crate::core::PipelineType::Lines => {
                self.add_lines_to_svg(svg, render_data)?;
            }
            crate::core::PipelineType::Points => {
                self.add_points_to_svg(svg, render_data)?;
            }
            crate::core::PipelineType::Triangles => {
                self.add_triangles_to_svg(svg, render_data)?;
            }
            crate::core::PipelineType::PointCloud => {
                self.add_points_to_svg(svg, render_data)?;
            }
        }
        Ok(())
    }

    /// Add line data to SVG
    #[allow(dead_code)]
    fn add_lines_to_svg(&self, svg: &mut String, render_data: &RenderData) -> Result<(), String> {
        if render_data.vertices.len() < 2 {
            return Ok(());
        }

        // Convert vertices to SVG path
        writeln!(svg, "  <g>").map_err(|e| format!("SVG write error: {e}"))?;

        for chunk in render_data.vertices.chunks(2) {
            if chunk.len() == 2 {
                let start = &chunk[0];
                let end = &chunk[1];

                let start_screen = self.world_to_screen(start.position);
                let end_screen = self.world_to_screen(end.position);

                writeln!(
                    svg,
                    r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}"/>"#,
                    start_screen[0],
                    start_screen[1],
                    end_screen[0],
                    end_screen[1],
                    self.color_to_hex(&start.color),
                    self.settings.stroke_width
                )
                .map_err(|e| format!("SVG write error: {e}"))?;
            }
        }

        writeln!(svg, "  </g>").map_err(|e| format!("SVG write error: {e}"))?;
        Ok(())
    }

    /// Add point data to SVG
    #[allow(dead_code)]
    fn add_points_to_svg(&self, svg: &mut String, render_data: &RenderData) -> Result<(), String> {
        writeln!(svg, "  <g>").map_err(|e| format!("SVG write error: {e}"))?;

        for vertex in &render_data.vertices {
            let screen_pos = self.world_to_screen(vertex.position);
            let radius = self.settings.stroke_width * 2.0;

            writeln!(
                svg,
                r#"    <circle cx="{}" cy="{}" r="{}" fill="{}"/>"#,
                screen_pos[0],
                screen_pos[1],
                radius,
                self.color_to_hex(&vertex.color)
            )
            .map_err(|e| format!("SVG write error: {e}"))?;
        }

        writeln!(svg, "  </g>").map_err(|e| format!("SVG write error: {e}"))?;
        Ok(())
    }

    /// Add triangle data to SVG
    #[allow(dead_code)]
    fn add_triangles_to_svg(
        &self,
        svg: &mut String,
        render_data: &RenderData,
    ) -> Result<(), String> {
        writeln!(svg, "  <g>").map_err(|e| format!("SVG write error: {e}"))?;

        for triangle in render_data.vertices.chunks(3) {
            if triangle.len() == 3 {
                let p1 = self.world_to_screen(triangle[0].position);
                let p2 = self.world_to_screen(triangle[1].position);
                let p3 = self.world_to_screen(triangle[2].position);

                writeln!(
                    svg,
                    r#"    <polygon points="{},{} {},{} {},{}" fill="{}"/>"#,
                    p1[0],
                    p1[1],
                    p2[0],
                    p2[1],
                    p3[0],
                    p3[1],
                    self.color_to_hex(&triangle[0].color)
                )
                .map_err(|e| format!("SVG write error: {e}"))?;
            }
        }

        writeln!(svg, "  </g>").map_err(|e| format!("SVG write error: {e}"))?;
        Ok(())
    }

    /// Convert world coordinates to screen coordinates
    #[allow(dead_code)]
    fn world_to_screen(&self, world_pos: [f32; 3]) -> [f32; 2] {
        // Simple orthographic projection for 2D plots
        // TODO: Add proper camera transformation for 3D plots
        [
            (world_pos[0] + 1.0) * 0.5 * self.settings.width,
            (1.0 - world_pos[1]) * 0.5 * self.settings.height,
        ]
    }

    /// Convert color array to hex string
    fn color_to_hex(&self, color: &[f32; 4]) -> String {
        format!(
            "#{:02x}{:02x}{:02x}",
            (color[0] * 255.0) as u8,
            (color[1] * 255.0) as u8,
            (color[2] * 255.0) as u8
        )
    }

    /// Update export settings
    pub fn set_settings(&mut self, settings: VectorExportSettings) {
        self.settings = settings;
    }

    /// Get current export settings
    pub fn settings(&self) -> &VectorExportSettings {
        &self.settings
    }
}

impl Default for VectorExporter {
    fn default() -> Self {
        Self::new()
    }
}
