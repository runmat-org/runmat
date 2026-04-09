//! Image export (PNG and raw RGBA).
//!
//! This module routes through `export::native_surface` to produce static image
//! output that matches interactive/webgpu composition (scene + overlay layout).

use crate::core::Camera;
use crate::plots::Figure;
use crate::styling::PlotThemeConfig;
use std::io::Cursor;
use std::path::Path;

/// High-fidelity image exporter using the interactive render path.
pub struct ImageExporter {
    /// Export settings
    settings: ImageExportSettings,
    /// Optional theme override for export composition.
    theme: Option<PlotThemeConfig>,
}

/// Image export configuration
#[derive(Debug, Clone)]
pub struct ImageExportSettings {
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels
    pub height: u32,
    /// Samples for anti-aliasing (reserved; interactive path currently chooses its own)
    pub samples: u32,
    /// Background color [R, G, B, A] (reserved; theme/figure-driven in interactive path)
    pub background_color: [f32; 4],
    /// Image quality (0.0-1.0) for lossy formats (reserved)
    pub quality: f32,
    /// Include metadata in output (reserved)
    pub include_metadata: bool,
}

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    WebP,
    Bmp,
}

impl Default for ImageExportSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            samples: 4,
            background_color: [1.0, 1.0, 1.0, 1.0],
            quality: 0.95,
            include_metadata: true,
        }
    }
}

impl ImageExporter {
    /// Create a new image exporter.
    pub async fn new() -> Result<Self, String> {
        Self::with_settings(ImageExportSettings::default()).await
    }

    /// Create exporter with custom settings.
    pub async fn with_settings(settings: ImageExportSettings) -> Result<Self, String> {
        Ok(Self {
            settings,
            theme: None,
        })
    }

    /// Set the export theme configuration.
    pub fn set_theme_config(&mut self, theme: PlotThemeConfig) {
        self.theme = Some(theme);
    }

    /// Export figure to PNG file.
    pub async fn export_png<P: AsRef<Path>>(
        &self,
        figure: &mut Figure,
        path: P,
    ) -> Result<(), String> {
        let bytes = self.render_png_bytes(figure).await?;
        std::fs::write(path, bytes).map_err(|e| format!("Failed to save PNG: {e}"))
    }

    /// Render figure into a PNG buffer.
    pub async fn render_png_bytes(&self, figure: &mut Figure) -> Result<Vec<u8>, String> {
        if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_png_bytes_interactive_and_theme(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                theme.clone(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_png_bytes_interactive(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
            )
            .await
        }
    }

    /// Render figure into raw RGBA8 bytes.
    pub async fn render_rgba_bytes(&self, figure: &mut Figure) -> Result<Vec<u8>, String> {
        if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_and_theme(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                theme.clone(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_rgba_bytes_interactive(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
            )
            .await
        }
    }

    /// Render figure into a PNG buffer using an explicit camera override.
    pub async fn render_png_bytes_with_camera(
        &self,
        figure: &mut Figure,
        camera: &Camera,
    ) -> Result<Vec<u8>, String> {
        if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_png_bytes_interactive_with_camera_and_theme(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                camera,
                theme.clone(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_png_bytes_interactive_with_camera(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                camera,
            )
            .await
        }
    }

    /// Render figure into raw RGBA8 bytes using an explicit camera override.
    pub async fn render_rgba_bytes_with_camera(
        &self,
        figure: &mut Figure,
        camera: &Camera,
    ) -> Result<Vec<u8>, String> {
        if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_with_camera_and_theme(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                camera,
                theme.clone(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_with_camera(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                camera,
            )
            .await
        }
    }

    /// Render figure into a PNG buffer using per-axes camera overrides.
    pub async fn render_png_bytes_with_axes_cameras(
        &self,
        figure: &mut Figure,
        axes_cameras: &[Camera],
    ) -> Result<Vec<u8>, String> {
        if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_png_bytes_interactive_with_axes_cameras_and_theme(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                axes_cameras,
                theme.clone(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_png_bytes_interactive_with_axes_cameras(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                axes_cameras,
            )
            .await
        }
    }

    /// Render figure into raw RGBA8 bytes using per-axes camera overrides.
    pub async fn render_rgba_bytes_with_axes_cameras(
        &self,
        figure: &mut Figure,
        axes_cameras: &[Camera],
    ) -> Result<Vec<u8>, String> {
        if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_with_axes_cameras_and_theme(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                axes_cameras,
                theme.clone(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_with_axes_cameras(
                figure.clone(),
                self.settings.width.max(1),
                self.settings.height.max(1),
                axes_cameras,
            )
            .await
        }
    }

    /// Encode RGBA bytes as PNG.
    pub fn encode_png_bytes(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        use image::{ImageBuffer, ImageOutputFormat, Rgba};

        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
            self.settings.width.max(1),
            self.settings.height.max(1),
            data.to_vec(),
        )
        .ok_or("Failed to create image buffer")?;

        let mut cursor = Cursor::new(Vec::new());
        image
            .write_to(&mut cursor, ImageOutputFormat::Png)
            .map_err(|e| format!("Failed to encode PNG: {e}"))?;
        Ok(cursor.into_inner())
    }

    /// Update export settings.
    pub fn set_settings(&mut self, settings: ImageExportSettings) {
        self.settings = settings;
    }

    /// Get current export settings.
    pub fn settings(&self) -> &ImageExportSettings {
        &self.settings
    }
}
