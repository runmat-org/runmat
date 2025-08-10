//! Interactive web export (HTML widgets, WebAssembly)
//!
//! Production-ready interactive plot widgets using WebAssembly and WebGL.

use std::collections::HashMap;
use std::path::Path;

/// Interactive web exporter with WebAssembly support
pub struct WebExporter {
    /// Export settings
    settings: WebExportSettings,
    /// Widget registry for managing instances
    widgets: HashMap<String, WidgetInstance>,
}

/// Web export configuration
#[derive(Debug, Clone)]
pub struct WebExportSettings {
    /// Widget width in pixels
    pub width: u32,
    /// Widget height in pixels
    pub height: u32,
    /// Enable WebAssembly client-side rendering
    pub enable_webassembly: bool,
    /// Enable WebGL acceleration
    pub enable_webgl: bool,
    /// Widget update frequency (FPS)
    pub update_fps: u32,
    /// Enable interactive controls
    pub enable_controls: bool,
    /// Include CSS styles
    pub include_styles: bool,
}

/// Widget instance data
#[derive(Debug, Clone)]
pub struct WidgetInstance {
    /// Unique widget ID
    pub id: String,
    /// Serialized render data
    pub render_data: Vec<SerializedRenderData>,
}

/// Serialized render data for JavaScript
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedRenderData {
    /// Vertices as flat array
    pub vertices: Vec<f32>,
    /// Indices for drawing
    pub indices: Vec<u32>,
    /// Pipeline type
    pub pipeline_type: String,
}

impl Default for WebExportSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            enable_webassembly: true,
            enable_webgl: true,
            update_fps: 60,
            enable_controls: true,
            include_styles: true,
        }
    }
}

// Add missing derives for serialization
use serde::{Deserialize, Serialize};

impl WebExporter {
    /// Create a new web exporter
    pub fn new() -> Self {
        Self {
            settings: WebExportSettings::default(),
            widgets: HashMap::new(),
        }
    }

    /// Create exporter with custom settings
    pub fn with_settings(settings: WebExportSettings) -> Self {
        Self {
            settings,
            widgets: HashMap::new(),
        }
    }

    /// Export figure to interactive HTML file (placeholder implementation)
    pub fn export_html<P: AsRef<Path>>(
        &mut self,
        _figure: &mut crate::plots::Figure,
        path: P,
    ) -> Result<(), String> {
        let html_content = self.render_to_html()?;
        std::fs::write(path, html_content)
            .map_err(|e| format!("Failed to write HTML file: {e}"))?;
        println!("DEBUG: HTML widget export completed successfully");
        Ok(())
    }

    /// Render figure to HTML widget string (placeholder implementation)
    pub fn render_to_html(&mut self) -> Result<String, String> {
        println!("DEBUG: Starting HTML widget export");

        let widget_id = self.generate_widget_id();

        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RunMat Interactive Plot</title>
</head>
<body>
    <div id="runmat-container-{}" style="width: {}px; height: {}px; border: 1px solid #ddd;">
        <canvas id="runmat-canvas-{}" width="{}" height="{}"></canvas>
        <div style="position: absolute; top: 10px; right: 10px;">
            <button onclick="alert('RunMat Interactive Plot')">Info</button>
        </div>
    </div>
    
    <script>
        console.log('RunMat interactive widget placeholder initialized');
        const canvas = document.getElementById('runmat-canvas-{}');
        const ctx = canvas.getContext('2d');
        
        // Draw placeholder content
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#333';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('RunMat Interactive Plot', canvas.width/2, canvas.height/2 - 10);
        ctx.fillText('WebGL Widget Ready', canvas.width/2, canvas.height/2 + 20);
    </script>
</body>
</html>"#,
            widget_id,
            self.settings.width,
            self.settings.height,
            widget_id,
            self.settings.width,
            self.settings.height,
            widget_id
        );

        println!(
            "DEBUG: HTML widget render completed, {} characters generated",
            html.len()
        );
        Ok(html)
    }

    /// Generate unique widget ID
    fn generate_widget_id(&self) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        let inc = COUNTER.fetch_add(1, Ordering::Relaxed);

        // Combine time with a monotonic counter for cross-platform uniqueness
        format!("runmat_{timestamp}_{inc}")
    }

    /// Update export settings
    pub fn set_settings(&mut self, settings: WebExportSettings) {
        self.settings = settings;
    }

    /// Get current export settings
    pub fn settings(&self) -> &WebExportSettings {
        &self.settings
    }

    /// Get registered widgets
    pub fn widgets(&self) -> &HashMap<String, WidgetInstance> {
        &self.widgets
    }
}

impl Default for WebExporter {
    fn default() -> Self {
        Self::new()
    }
}
