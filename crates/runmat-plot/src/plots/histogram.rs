//! Histogram implementation
//!
//! High-performance histograms with GPU acceleration.

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};

/// High-performance GPU-accelerated histogram
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Raw data values
    pub data: Vec<f64>,

    /// Histogram configuration
    pub bins: usize,
    pub bin_edges: Vec<f64>,
    pub bin_counts: Vec<u64>,

    /// Visual styling
    pub color: Vec4,
    pub outline_color: Option<Vec4>,
    pub outline_width: f32,
    pub normalize: bool,

    /// Metadata
    pub label: Option<String>,
    pub visible: bool,

    /// Generated rendering data (cached)
    vertices: Option<Vec<Vertex>>,
    indices: Option<Vec<u32>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

impl Histogram {
    /// Create a new histogram with data and number of bins
    pub fn new(data: Vec<f64>, bins: usize) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Cannot create histogram with empty data".to_string());
        }

        if bins == 0 {
            return Err("Number of bins must be greater than zero".to_string());
        }

        let mut histogram = Self {
            data,
            bins,
            bin_edges: Vec::new(),
            bin_counts: Vec::new(),
            color: Vec4::new(0.0, 0.5, 1.0, 1.0), // Default blue
            outline_color: Some(Vec4::new(0.0, 0.0, 0.0, 1.0)), // Default black outline
            outline_width: 1.0,
            normalize: false,
            label: None,
            visible: true,
            vertices: None,
            indices: None,
            bounds: None,
            dirty: true,
        };

        histogram.compute_histogram();
        Ok(histogram)
    }

    /// Create a histogram with custom bin edges
    pub fn with_bin_edges(data: Vec<f64>, bin_edges: Vec<f64>) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Cannot create histogram with empty data".to_string());
        }

        if bin_edges.len() < 2 {
            return Err("Must have at least 2 bin edges".to_string());
        }

        // Verify bin edges are sorted
        for i in 1..bin_edges.len() {
            if bin_edges[i] <= bin_edges[i - 1] {
                return Err("Bin edges must be strictly increasing".to_string());
            }
        }

        let bins = bin_edges.len() - 1;
        let mut histogram = Self {
            data,
            bins,
            bin_edges,
            bin_counts: Vec::new(),
            color: Vec4::new(0.0, 0.5, 1.0, 1.0),
            outline_color: Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
            outline_width: 1.0,
            normalize: false,
            label: None,
            visible: true,
            vertices: None,
            indices: None,
            bounds: None,
            dirty: true,
        };

        histogram.compute_histogram();
        Ok(histogram)
    }

    /// Set styling options
    pub fn with_style(mut self, color: Vec4, normalize: bool) -> Self {
        self.color = color;
        self.normalize = normalize;
        self.dirty = true;
        self
    }

    /// Add outline to bars
    pub fn with_outline(mut self, outline_color: Vec4, outline_width: f32) -> Self {
        self.outline_color = Some(outline_color);
        self.outline_width = outline_width.max(0.1);
        self.dirty = true;
        self
    }

    /// Remove outline
    pub fn without_outline(mut self) -> Self {
        self.outline_color = None;
        self.dirty = true;
        self
    }

    /// Set the histogram label for legends
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Update the data and recompute histogram
    pub fn update_data(&mut self, data: Vec<f64>) -> Result<(), String> {
        if data.is_empty() {
            return Err("Cannot update with empty data".to_string());
        }

        self.data = data;
        self.compute_histogram();
        self.dirty = true;
        Ok(())
    }

    /// Set the number of bins and recompute
    pub fn set_bins(&mut self, bins: usize) -> Result<(), String> {
        if bins == 0 {
            return Err("Number of bins must be greater than zero".to_string());
        }

        self.bins = bins;
        self.compute_histogram();
        self.dirty = true;
        Ok(())
    }

    /// Set the histogram color
    pub fn set_color(&mut self, color: Vec4) {
        self.color = color;
        self.dirty = true;
    }

    /// Enable or disable normalization
    pub fn set_normalize(&mut self, normalize: bool) {
        self.normalize = normalize;
        self.dirty = true;
    }

    /// Show or hide the histogram
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Get the number of bins
    pub fn len(&self) -> usize {
        self.bins
    }

    /// Check if the histogram has no data
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compute the histogram from the data
    fn compute_histogram(&mut self) {
        if self.data.is_empty() {
            return;
        }

        // Generate bin edges if not provided
        if self.bin_edges.is_empty() {
            let min_val = self.data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = self.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Handle case where all values are the same
            let (min_val, max_val) = if (max_val - min_val).abs() < f64::EPSILON {
                (min_val - 0.5, max_val + 0.5)
            } else {
                (min_val, max_val)
            };

            let bin_width = (max_val - min_val) / self.bins as f64;
            self.bin_edges = (0..=self.bins)
                .map(|i| min_val + i as f64 * bin_width)
                .collect();
        }

        // Count values in each bin
        self.bin_counts = vec![0; self.bins];
        for &value in &self.data {
            // Find which bin this value belongs to
            let mut bin_index = self.bins; // Default to overflow

            for i in 0..self.bins {
                if value >= self.bin_edges[i] && value < self.bin_edges[i + 1] {
                    bin_index = i;
                    break;
                }
            }

            // Handle the last bin edge (inclusive)
            if bin_index == self.bins && value == self.bin_edges[self.bins] {
                bin_index = self.bins - 1;
            }

            // Count the value if it's within bounds
            if bin_index < self.bins {
                self.bin_counts[bin_index] += 1;
            }
        }
    }

    /// Get bin heights (counts or normalized densities)
    fn get_bin_heights(&self) -> Vec<f64> {
        if self.normalize {
            let total_count: u64 = self.bin_counts.iter().sum();

            if total_count == 0 {
                return vec![0.0; self.bin_counts.len()];
            }

            self.bin_counts
                .iter()
                .zip(self.bin_edges.windows(2))
                .map(|(&count, edges)| {
                    let bin_width = edges[1] - edges[0];
                    count as f64 / (total_count as f64 * bin_width)
                })
                .collect()
        } else {
            self.bin_counts.iter().map(|&c| c as f64).collect()
        }
    }

    /// Generate vertices for GPU rendering
    pub fn generate_vertices(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
        if self.dirty || self.vertices.is_none() {
            let (vertices, indices) = self.create_histogram_geometry();
            self.vertices = Some(vertices);
            self.indices = Some(indices);
            self.dirty = false;
        }
        (
            self.vertices.as_ref().unwrap(),
            self.indices.as_ref().unwrap(),
        )
    }

    /// Create the geometry for the histogram bars
    fn create_histogram_geometry(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let heights = self.get_bin_heights();

        for (&height, edges) in heights.iter().zip(self.bin_edges.windows(2)) {
            let left = edges[0] as f32;
            let right = edges[1] as f32;
            let bottom = 0.0;
            let top = height as f32;

            // Create vertices for this bar (rectangle)
            let base_vertex_index = vertices.len() as u32;

            // Four corners of the rectangle
            vertices.push(Vertex::new(Vec3::new(left, bottom, 0.0), self.color)); // Bottom-left
            vertices.push(Vertex::new(Vec3::new(right, bottom, 0.0), self.color)); // Bottom-right
            vertices.push(Vertex::new(Vec3::new(right, top, 0.0), self.color)); // Top-right
            vertices.push(Vertex::new(Vec3::new(left, top, 0.0), self.color)); // Top-left

            // Two triangles to form the rectangle
            indices.extend_from_slice(&[
                base_vertex_index,
                base_vertex_index + 1,
                base_vertex_index + 2, // Bottom-right triangle
                base_vertex_index,
                base_vertex_index + 2,
                base_vertex_index + 3, // Top-left triangle
            ]);
        }

        (vertices, indices)
    }

    /// Get the bounding box of the histogram
    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            if self.bin_edges.is_empty() {
                self.bounds = Some(BoundingBox::default());
                return self.bounds.unwrap();
            }

            let min_x = *self.bin_edges.first().unwrap() as f32;
            let max_x = *self.bin_edges.last().unwrap() as f32;

            let heights = self.get_bin_heights();
            let max_height = heights.iter().fold(0.0f64, |a, &b| a.max(b)) as f32;

            self.bounds = Some(BoundingBox::new(
                Vec3::new(min_x, 0.0, 0.0),
                Vec3::new(max_x, max_height, 0.0),
            ));
        }
        self.bounds.unwrap()
    }

    /// Generate complete render data for the graphics pipeline
    pub fn render_data(&mut self) -> RenderData {
        let (vertices, indices) = self.generate_vertices();
        let vertices = vertices.clone();
        let indices = indices.clone();

        let material = Material {
            albedo: self.color,
            ..Default::default()
        };

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count: vertices.len(),
            index_offset: Some(0),
            index_count: Some(indices.len()),
            instance_count: 1,
        };

        RenderData {
            pipeline_type: PipelineType::Triangles,
            vertices,
            indices: Some(indices),
            material,
            draw_calls: vec![draw_call],
        }
    }

    /// Get histogram statistics
    pub fn statistics(&self) -> HistogramStatistics {
        let data_range = if self.data.is_empty() {
            (0.0, 0.0)
        } else {
            let min_val = self.data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = self.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            (min_val, max_val)
        };

        let total_count: u64 = self.bin_counts.iter().sum();
        let max_count = self.bin_counts.iter().max().copied().unwrap_or(0);

        HistogramStatistics {
            data_count: self.data.len(),
            bin_count: self.bins,
            data_range,
            total_count,
            max_bin_count: max_count,
            memory_usage: self.estimated_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let data_size = self.data.len() * std::mem::size_of::<f64>();
        let edges_size = self.bin_edges.len() * std::mem::size_of::<f64>();
        let counts_size = self.bin_counts.len() * std::mem::size_of::<u64>();
        let vertices_size = self
            .vertices
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>());
        let indices_size = self
            .indices
            .as_ref()
            .map_or(0, |i| i.len() * std::mem::size_of::<u32>());

        data_size + edges_size + counts_size + vertices_size + indices_size
    }
}

/// Histogram statistics
#[derive(Debug, Clone)]
pub struct HistogramStatistics {
    pub data_count: usize,
    pub bin_count: usize,
    pub data_range: (f64, f64),
    pub total_count: u64,
    pub max_bin_count: u64,
    pub memory_usage: usize,
}

/// MATLAB-compatible histogram creation utilities
pub mod matlab_compat {
    use super::*;

    /// Create a histogram (equivalent to MATLAB's `hist(data, bins)`)
    pub fn hist(data: Vec<f64>, bins: usize) -> Result<Histogram, String> {
        Histogram::new(data, bins)
    }

    /// Create a histogram with custom bin edges (`hist(data, edges)`)
    pub fn hist_with_edges(data: Vec<f64>, edges: Vec<f64>) -> Result<Histogram, String> {
        Histogram::with_bin_edges(data, edges)
    }

    /// Create a normalized histogram (density)
    pub fn histogram_normalized(data: Vec<f64>, bins: usize) -> Result<Histogram, String> {
        Ok(Histogram::new(data, bins)?.with_style(
            Vec4::new(0.0, 0.5, 1.0, 1.0),
            true, // normalize
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_creation() {
        let data = vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let hist = Histogram::new(data.clone(), 5).unwrap();

        assert_eq!(hist.data, data);
        assert_eq!(hist.bins, 5);
        assert_eq!(hist.bin_edges.len(), 6); // n+1 edges for n bins
        assert_eq!(hist.bin_counts.len(), 5);
        assert!(!hist.is_empty());
    }

    #[test]
    fn test_histogram_validation() {
        // Empty data should fail
        assert!(Histogram::new(vec![], 5).is_err());

        // Zero bins should fail
        assert!(Histogram::new(vec![1.0, 2.0], 0).is_err());

        // Invalid bin edges should fail
        assert!(Histogram::with_bin_edges(vec![1.0, 2.0], vec![1.0]).is_err()); // Too few edges
        assert!(Histogram::with_bin_edges(vec![1.0, 2.0], vec![2.0, 1.0]).is_err());
        // Not sorted
    }

    #[test]
    fn test_histogram_computation() {
        let data = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let hist = Histogram::new(data, 3).unwrap();

        // Should have created appropriate bin edges
        assert!(hist.bin_edges[0] <= 1.0);
        assert!(hist.bin_edges.last().unwrap() >= &3.0);

        // Should have counted all data points
        let total_count: u64 = hist.bin_counts.iter().sum();
        assert_eq!(total_count, 5);
    }

    #[test]
    fn test_histogram_custom_edges() {
        let data = vec![0.5, 1.5, 2.5, 3.5];
        let edges = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let hist = Histogram::with_bin_edges(data, edges.clone()).unwrap();

        assert_eq!(hist.bin_edges, edges);
        assert_eq!(hist.bins, 4);

        // Each bin should have exactly one value
        assert_eq!(hist.bin_counts, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_histogram_normalization() {
        let data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let hist = Histogram::new(data, 3).unwrap().with_style(Vec4::ONE, true);

        let heights = hist.get_bin_heights();

        // For normalized histogram, the sum of (height * width) should equal 1
        let total_area: f64 = heights
            .iter()
            .zip(hist.bin_edges.windows(2))
            .map(|(&height, edges)| height * (edges[1] - edges[0]))
            .sum();

        assert!((total_area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_bounds() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut hist = Histogram::new(data, 4).unwrap();
        let bounds = hist.bounds();

        // X bounds should span all bin edges
        assert!(bounds.min.x <= 1.0);
        assert!(bounds.max.x >= 5.0);

        // Y should start at 0 and go to max count
        assert_eq!(bounds.min.y, 0.0);
        assert!(bounds.max.y > 0.0);
    }

    #[test]
    fn test_histogram_vertex_generation() {
        let data = vec![1.0, 2.0];
        let mut hist = Histogram::new(data, 2).unwrap();
        let (vertices, indices) = hist.generate_vertices();

        // Should have 4 vertices per bin (rectangle)
        assert_eq!(vertices.len(), 8);

        // Should have 6 indices per bin (2 triangles)
        assert_eq!(indices.len(), 12);
    }

    #[test]
    fn test_histogram_render_data() {
        let data = vec![1.0, 1.5, 2.0];
        let mut hist = Histogram::new(data, 2).unwrap();
        let render_data = hist.render_data();

        assert_eq!(render_data.pipeline_type, PipelineType::Triangles);
        assert!(!render_data.vertices.is_empty());
        assert!(render_data.indices.is_some());
    }

    #[test]
    fn test_histogram_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let hist = Histogram::new(data, 3).unwrap();
        let stats = hist.statistics();

        assert_eq!(stats.data_count, 5);
        assert_eq!(stats.bin_count, 3);
        assert_eq!(stats.data_range, (1.0, 5.0));
        assert_eq!(stats.total_count, 5);
        assert!(stats.memory_usage > 0);
    }

    #[test]
    fn test_matlab_compat_hist() {
        use super::matlab_compat::*;

        let data = vec![1.0, 2.0, 3.0];

        let hist1 = hist(data.clone(), 2).unwrap();
        assert_eq!(hist1.len(), 2);

        let edges = vec![0.0, 1.5, 3.5];
        let hist2 = hist_with_edges(data.clone(), edges).unwrap();
        assert_eq!(hist2.bins, 2);

        let hist3 = histogram_normalized(data, 3).unwrap();
        assert!(hist3.normalize);
    }
}
