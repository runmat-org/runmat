//! High-quality vector graphics export (SVG, PDF, etc.)
//!
//! Production-ready vector output using same rendering pipeline data.

use crate::core::RenderData;
use base64::Engine;
use image::ImageEncoder;
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
        log::debug!(target: "runmat_plot", "svg export completed");
        Ok(())
    }

    /// Export figure to PDF file (placeholder)
    pub fn export_pdf<P: AsRef<Path>>(&self, _figure: &mut Figure, _path: P) -> Result<(), String> {
        // TODO: Implement PDF export using a PDF library
        Err("PDF export not yet implemented".to_string())
    }

    /// Render figure to SVG string using same rendering pipeline data
    pub fn render_to_svg(&self, figure: &mut Figure) -> Result<String, String> {
        log::debug!(target: "runmat_plot", "svg export render start");

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
                "  <metadata>\n    <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n      <rdf:Description>\n        <dc:creator xmlns:dc=\"http://purl.org/dc/elements/1.1/\">RunMat Plot System</dc:creator>\n      </rdf:Description>\n    </rdf:RDF>\n  </metadata>"
            ).map_err(|e| format!("SVG write error: {e}"))?;
        }

        // Figure title (if any)
        if let Some(title) = &figure.title {
            let fs = 18;
            writeln!(
                &mut svg,
                "  <text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"{}\" fill=\"#000000\" font-family=\"sans-serif\">{}</text>",
                self.settings.width * 0.5,
                24,
                fs,
                xml_escape(title)
            ).map_err(|e| format!("SVG write error: {e}"))?;
        }

        // Render each plot element grouped by axes (subplots)
        let (rows, cols) = figure.axes_grid();
        let gaps = (8.0f32, 8.0f32);
        let (hgap, vgap) = gaps;
        let total_hgap = hgap * (cols.saturating_sub(1) as f32);
        let total_vgap = vgap * (rows.saturating_sub(1) as f32);
        let cell_w = (self.settings.width - total_hgap).max(1.0) / (cols.max(1) as f32);
        let cell_h = (self.settings.height - total_vgap).max(1.0) / (rows.max(1) as f32);

        let axes_vps: Vec<(f32, f32, f32, f32)> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c as f32 * (cell_w + hgap), r as f32 * (cell_h + vgap), cell_w, cell_h)))
            .collect();

        let axes_map = figure.plot_axes_indices().to_vec();
        let rds = figure.render_data();
        for (i, rd) in rds.iter().enumerate() {
            let ax = axes_map.get(i).copied().unwrap_or(0).min(rows*cols - 1);
            let vp = axes_vps[ax];
            // Axes labels
            if let Some(lbl) = &figure.x_label {
                let cx = vp.0 + vp.2 * 0.5; let cy = vp.1 + vp.3 + 20.0;
                writeln!(
                    &mut svg,
                    "  <text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#000000\" font-family=\"sans-serif\">{}</text>",
                    cx, cy, xml_escape(lbl)
                ).map_err(|e| format!("SVG write error: {e}"))?;
            }
            if let Some(lbl) = &figure.y_label {
                let cx = vp.0 - 24.0; let cy = vp.1 + vp.3 * 0.5;
                writeln!(
                    &mut svg,
                    "  <text x=\"{}\" y=\"{}\" transform=\"rotate(-90, {}, {})\" text-anchor=\"middle\" font-size=\"12\" fill=\"#000000\" font-family=\"sans-serif\">{}</text>",
                    cx, cy, cx, cy, xml_escape(lbl)
                ).map_err(|e| format!("SVG write error: {e}"))?;
            }
            self.add_render_data_to_svg_viewport(&mut svg, rd, vp)?;
        }

        // SVG footer
        writeln!(&mut svg, "</svg>").map_err(|e| format!("SVG write error: {e}"))?;

        log::debug!(target: "runmat_plot", "svg export size chars={}", svg.len());
        Ok(svg)
    }

    /// Add render data to SVG using same pipeline data
    /// Note: Will be used when Figure iteration is implemented
    #[allow(dead_code)]
    fn add_render_data_to_svg_viewport(
        &self,
        svg: &mut String,
        render_data: &RenderData,
        viewport: (f32, f32, f32, f32),
    ) -> Result<(), String> {
        match render_data.pipeline_type {
            crate::core::PipelineType::Lines => {
                self.add_lines_to_svg_viewport(svg, render_data, viewport)?;
            }
            crate::core::PipelineType::Points => {
                self.add_points_to_svg_viewport(svg, render_data, viewport)?;
            }
            crate::core::PipelineType::Triangles => {
                self.add_triangles_to_svg_viewport(svg, render_data, viewport)?;
            }
            crate::core::PipelineType::Textured => {
                self.add_textured_to_svg_viewport(svg, render_data, viewport)?;
            }
            crate::core::PipelineType::PointCloud => {
                self.add_points_to_svg_viewport(svg, render_data, viewport)?;
            }
        }
        Ok(())
    }

    /// Add line data to SVG
    #[allow(dead_code)]
    fn add_lines_to_svg_viewport(&self, svg: &mut String, render_data: &RenderData, vp: (f32,f32,f32,f32)) -> Result<(), String> {
        if render_data.vertices.len() < 2 {
            return Ok(());
        }

        // Convert vertices to SVG path
        writeln!(svg, "  <g>").map_err(|e| format!("SVG write error: {e}"))?;

        // Material encodes style: roughness=line_width, metallic=style, emissive.x=cap, emissive.y=join
        let lw = render_data.material.roughness.max(0.5);
        let style_code = render_data.material.metallic as i32;
        let cap_code = render_data.material.emissive.x as i32;
        let join_code = render_data.material.emissive.y as i32;
        let stroke = self.color_to_hex(&render_data.vertices.get(0).map(|v| v.color).unwrap_or([0.0,0.0,0.0,1.0]));
        let stroke_linecap = match cap_code { 1 => "square", 2 => "round", _ => "butt" };
        let stroke_linejoin = match join_code { 1 => "bevel", 2 => "round", _ => "miter" };
        // Normalize dash spacing in SVG to on-screen units scaled by stroke width
        // Patterns:
        //  - Dashed:   on=6*lw, off=6*lw
        //  - Dotted:   on=1*lw, off=6*lw
        //  - DashDot:  on=6*lw, off=4*lw, dot=1*lw, gap=4*lw
        let stroke_dasharray: Option<String> = match style_code {
            1 => Some(format!("{},{}", 6.0 * lw, 6.0 * lw)),
            2 => Some(format!("{},{}", 1.0 * lw, 6.0 * lw)),
            3 => Some(format!("{},{},{},{}", 6.0 * lw, 4.0 * lw, 1.0 * lw, 4.0 * lw)),
            _ => None,
        };

        for chunk in render_data.vertices.chunks(2) {
            if chunk.len() == 2 {
                let start = &chunk[0];
                let end = &chunk[1];

                let start_screen = self.world_to_screen_viewport(start.position, vp);
                let end_screen = self.world_to_screen_viewport(end.position, vp);

                if let Some(ref dash) = stroke_dasharray {
                    writeln!(
                        svg,
                        r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" stroke-linecap="{}" stroke-linejoin="{}" stroke-dasharray="{}"/>"#,
                        start_screen[0], start_screen[1], end_screen[0], end_screen[1], stroke, lw, stroke_linecap, stroke_linejoin, dash
                    ).map_err(|e| format!("SVG write error: {e}"))?;
                } else {
                    writeln!(
                        svg,
                        r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" stroke-linecap="{}" stroke-linejoin="{}"/>"#,
                        start_screen[0], start_screen[1], end_screen[0], end_screen[1], stroke, lw, stroke_linecap, stroke_linejoin
                    ).map_err(|e| format!("SVG write error: {e}"))?;
                }
            }
        }

        writeln!(svg, "  </g>").map_err(|e| format!("SVG write error: {e}"))?;
        Ok(())
    }

    /// Add point data to SVG
    #[allow(dead_code)]
    fn add_points_to_svg_viewport(&self, svg: &mut String, render_data: &RenderData, vp: (f32,f32,f32,f32)) -> Result<(), String> {
        writeln!(svg, "  <g>").map_err(|e| format!("SVG write error: {e}"))?;

        let marker_shape = render_data.material.metallic as u32; // 0 circle, 1 square
        let edge_color = render_data.material.emissive;
        let edge_width = render_data.material.roughness.max(0.0);
        for vertex in &render_data.vertices {
            let screen_pos = self.world_to_screen_viewport(vertex.position, vp);
            let radius = (vertex.normal[2] * 0.5).max(1.0);
            // Map to viewport scale roughly (already in pixels; SVG units are px)

            if marker_shape == 1u32 {
                // square
                let x = screen_pos[0] - radius;
                let y = screen_pos[1] - radius;
                writeln!(
                    svg,
                    r#"    <rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="{}"/>"#,
                    x,
                    y,
                    radius*2.0,
                    radius*2.0,
                    self.color_to_hex(&vertex.color),
                    self.color_to_hex(&edge_color.to_array()),
                    edge_width
                ).map_err(|e| format!("SVG write error: {e}"))?;
            } else {
                // circle default
                writeln!(
                    svg,
                    r#"    <circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="{}"/>"#,
                    screen_pos[0],
                    screen_pos[1],
                    radius,
                    self.color_to_hex(&vertex.color),
                    self.color_to_hex(&edge_color.to_array()),
                    edge_width
                ).map_err(|e| format!("SVG write error: {e}"))?;
            }
        }

        writeln!(svg, "  </g>").map_err(|e| format!("SVG write error: {e}"))?;
        Ok(())
    }

    /// Add triangle data to SVG
    #[allow(dead_code)]
    fn add_triangles_to_svg_viewport(
        &self,
        svg: &mut String,
        render_data: &RenderData,
        vp: (f32,f32,f32,f32),
    ) -> Result<(), String> {
        writeln!(svg, "  <g>").map_err(|e| format!("SVG write error: {e}"))?;

        for triangle in render_data.vertices.chunks(3) {
            if triangle.len() == 3 {
                let p1 = self.world_to_screen_viewport(triangle[0].position, vp);
                let p2 = self.world_to_screen_viewport(triangle[1].position, vp);
                let p3 = self.world_to_screen_viewport(triangle[2].position, vp);

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
    fn world_to_screen_viewport(&self, world_pos: [f32; 3], vp: (f32,f32,f32,f32)) -> [f32; 2] {
        let (vx, vy, vw, vh) = vp;
        [
            vx + (world_pos[0] + 1.0) * 0.5 * vw,
            vy + (1.0 - (world_pos[1] + 1.0) * 0.5) * vh,
        ]
    }

    /// Draw textured quad as a filled rectangle (placeholder), sampling average color if available
    fn add_textured_to_svg_viewport(&self, svg: &mut String, render_data: &RenderData, vp: (f32,f32,f32,f32)) -> Result<(), String> {
        // Compute screen-space bounding box from vertices
        if render_data.vertices.is_empty() { return Ok(()); }
        let mut min_x = f32::INFINITY; let mut min_y = f32::INFINITY; let mut max_x = f32::NEG_INFINITY; let mut max_y = f32::NEG_INFINITY;
        for v in &render_data.vertices {
            let p = self.world_to_screen_viewport(v.position, vp);
            min_x = min_x.min(p[0]); max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]); max_y = max_y.max(p[1]);
        }
        let w = (max_x - min_x).max(1.0); let h = (max_y - min_y).max(1.0);
        // Embed image as base64 data URI if available
        if let Some(crate::core::scene::ImageData::Rgba8 { width, height, data }) = &render_data.image {
            if !data.is_empty() {
                // Encode RGBA8 as PNG in-memory
                let mut png_buf: Vec<u8> = Vec::new();
                {
                    let encoder = image::codecs::png::PngEncoder::new(&mut png_buf);
                    encoder.write_image(&data, *width, *height, image::ColorType::Rgba8).map_err(|e| format!("PNG encode failed: {e}"))?;
                }
                let b64 = base64::engine::general_purpose::STANDARD.encode(&png_buf);
                let href = format!("data:image/png;base64,{}", b64);
                writeln!(
                    svg,
                    r#"  <image x="{}" y="{}" width="{}" height="{}" xlink:href="{}" preserveAspectRatio="none"/>"#,
                    min_x, min_y, w, h, href
                ).map_err(|e| format!("SVG write error: {e}"))?;
                return Ok(());
            }
        }
        // Fallback: gray rect
        writeln!(
            svg,
            "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#999999\"/>",
            min_x, min_y, w, h
        )
            .map_err(|e| format!("SVG write error: {e}"))?;
        Ok(())
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
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

impl Default for VectorExporter {
    fn default() -> Self {
        Self::new()
    }
}
