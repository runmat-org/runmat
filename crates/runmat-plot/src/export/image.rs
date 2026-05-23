//! Image export (PNG and raw RGBA).
//!
//! This module renders with the native GPU surface path first and falls back to
//! a CPU raster path only when headless GPU initialization is unavailable.

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
    /// Optional export textmark.
    textmark: Option<String>,
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
            textmark: None,
        })
    }

    /// Set the export theme configuration.
    pub fn set_theme_config(&mut self, theme: PlotThemeConfig) {
        self.theme = Some(theme);
    }

    /// Set an optional textmark rendered in export output.
    pub fn set_textmark(&mut self, textmark: Option<&str>) {
        self.textmark = textmark
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned);
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
        let width = self.settings.width.max(1);
        let height = self.settings.height.max(1);
        let gpu = if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_png_bytes_interactive_and_theme_and_textmark(
                figure.clone(),
                width,
                height,
                theme.clone(),
                self.textmark.as_deref(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_png_bytes_interactive(
                figure.clone(),
                width,
                height,
            )
            .await
        };

        match gpu {
            Ok(bytes) => Ok(bytes),
            Err(err) if crate::export::native_surface::is_headless_gpu_unavailable_error(&err) => {
                crate::export::cpu_surface::render_figure_png_bytes(
                    figure.clone(),
                    width,
                    height,
                    self.theme.clone(),
                    None,
                    None,
                    self.textmark.as_deref(),
                )
                .await
            }
            Err(err) => Err(err),
        }
    }

    /// Render figure into raw RGBA8 bytes.
    pub async fn render_rgba_bytes(&self, figure: &mut Figure) -> Result<Vec<u8>, String> {
        let width = self.settings.width.max(1);
        let height = self.settings.height.max(1);
        let gpu = if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_and_theme(
                figure.clone(),
                width,
                height,
                theme.clone(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_rgba_bytes_interactive(
                figure.clone(),
                width,
                height,
            )
            .await
        };

        match gpu {
            Ok(bytes) => Ok(bytes),
            Err(err) if crate::export::native_surface::is_headless_gpu_unavailable_error(&err) => {
                crate::export::cpu_surface::render_figure_rgba_bytes(
                    figure.clone(),
                    width,
                    height,
                    self.theme.clone(),
                    None,
                    None,
                    self.textmark.as_deref(),
                )
                .await
            }
            Err(err) => Err(err),
        }
    }

    /// Render figure into a PNG buffer using an explicit camera override.
    pub async fn render_png_bytes_with_camera(
        &self,
        figure: &mut Figure,
        camera: &Camera,
    ) -> Result<Vec<u8>, String> {
        let width = self.settings.width.max(1);
        let height = self.settings.height.max(1);
        let gpu = if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_png_bytes_interactive_with_camera_and_theme_and_textmark(
                figure.clone(),
                width,
                height,
                camera,
                theme.clone(),
                self.textmark.as_deref(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_png_bytes_interactive_with_camera(
                figure.clone(),
                width,
                height,
                camera,
            )
            .await
        };

        match gpu {
            Ok(bytes) => Ok(bytes),
            Err(err) if crate::export::native_surface::is_headless_gpu_unavailable_error(&err) => {
                crate::export::cpu_surface::render_figure_png_bytes(
                    figure.clone(),
                    width,
                    height,
                    self.theme.clone(),
                    Some(camera),
                    None,
                    self.textmark.as_deref(),
                )
                .await
            }
            Err(err) => Err(err),
        }
    }

    /// Render figure into raw RGBA8 bytes using an explicit camera override.
    pub async fn render_rgba_bytes_with_camera(
        &self,
        figure: &mut Figure,
        camera: &Camera,
    ) -> Result<Vec<u8>, String> {
        let width = self.settings.width.max(1);
        let height = self.settings.height.max(1);
        let gpu = if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_with_camera_and_theme(
                figure.clone(),
                width,
                height,
                camera,
                theme.clone(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_with_camera(
                figure.clone(),
                width,
                height,
                camera,
            )
            .await
        };

        match gpu {
            Ok(bytes) => Ok(bytes),
            Err(err) if crate::export::native_surface::is_headless_gpu_unavailable_error(&err) => {
                crate::export::cpu_surface::render_figure_rgba_bytes(
                    figure.clone(),
                    width,
                    height,
                    self.theme.clone(),
                    Some(camera),
                    None,
                    self.textmark.as_deref(),
                )
                .await
            }
            Err(err) => Err(err),
        }
    }

    /// Render figure into a PNG buffer using per-axes camera overrides.
    pub async fn render_png_bytes_with_axes_cameras(
        &self,
        figure: &mut Figure,
        axes_cameras: &[Camera],
    ) -> Result<Vec<u8>, String> {
        let width = self.settings.width.max(1);
        let height = self.settings.height.max(1);
        let gpu = if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_png_bytes_interactive_with_axes_cameras_and_theme_and_textmark(
                figure.clone(),
                width,
                height,
                axes_cameras,
                theme.clone(),
                self.textmark.as_deref(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_png_bytes_interactive_with_axes_cameras(
                figure.clone(),
                width,
                height,
                axes_cameras,
            )
            .await
        };

        match gpu {
            Ok(bytes) => Ok(bytes),
            Err(err) if crate::export::native_surface::is_headless_gpu_unavailable_error(&err) => {
                crate::export::cpu_surface::render_figure_png_bytes(
                    figure.clone(),
                    width,
                    height,
                    self.theme.clone(),
                    None,
                    Some(axes_cameras),
                    self.textmark.as_deref(),
                )
                .await
            }
            Err(err) => Err(err),
        }
    }

    /// Render figure into raw RGBA8 bytes using per-axes camera overrides.
    pub async fn render_rgba_bytes_with_axes_cameras(
        &self,
        figure: &mut Figure,
        axes_cameras: &[Camera],
    ) -> Result<Vec<u8>, String> {
        let width = self.settings.width.max(1);
        let height = self.settings.height.max(1);
        let gpu = if let Some(theme) = &self.theme {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_with_axes_cameras_and_theme(
                figure.clone(),
                width,
                height,
                axes_cameras,
                theme.clone(),
            )
            .await
        } else {
            crate::export::native_surface::render_figure_rgba_bytes_interactive_with_axes_cameras(
                figure.clone(),
                width,
                height,
                axes_cameras,
            )
            .await
        };

        match gpu {
            Ok(bytes) => Ok(bytes),
            Err(err) if crate::export::native_surface::is_headless_gpu_unavailable_error(&err) => {
                crate::export::cpu_surface::render_figure_rgba_bytes(
                    figure.clone(),
                    width,
                    height,
                    self.theme.clone(),
                    None,
                    Some(axes_cameras),
                    self.textmark.as_deref(),
                )
                .await
            }
            Err(err) => Err(err),
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
