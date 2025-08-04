//! Scene graph system for organizing and managing plot objects
//! 
//! Provides hierarchical organization of plot elements with efficient
//! culling, level-of-detail, and batch rendering capabilities.

use std::collections::HashMap;
use glam::{Mat4, Vec3, Vec4};
use crate::core::renderer::{Vertex, PipelineType};

/// Unique identifier for scene nodes
pub type NodeId = u64;

/// Scene node representing a renderable object
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub transform: Mat4,
    pub visible: bool,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
    
    // Hierarchy
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    
    // Rendering data
    pub render_data: Option<RenderData>,
    
    // Bounding box for culling
    pub bounds: BoundingBox,
    
    // Level of detail settings
    pub lod_levels: Vec<LodLevel>,
    pub current_lod: usize,
}

/// Rendering data for a scene node
#[derive(Debug, Clone)]
pub struct RenderData {
    pub pipeline_type: PipelineType,
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
    pub material: Material,
    pub draw_calls: Vec<DrawCall>,
}

/// Material properties for rendering
#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: Vec4,
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: Vec4,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: Vec4::new(1.0, 1.0, 1.0, 1.0),
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec4::ZERO,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
        }
    }
}

/// Alpha blending mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask { cutoff: u8 },
    Blend,
}

/// Level of detail configuration
#[derive(Debug, Clone)]
pub struct LodLevel {
    pub distance: f32,
    pub vertex_count: usize,
    pub index_count: Option<usize>,
    pub simplification_ratio: f32,
}

/// Draw call for efficient batching
#[derive(Debug, Clone)]
pub struct DrawCall {
    pub vertex_offset: usize,
    pub vertex_count: usize,
    pub index_offset: Option<usize>,
    pub index_count: Option<usize>,
    pub instance_count: usize,
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }
}

impl BoundingBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
    
    pub fn from_points(points: &[Vec3]) -> Self {
        if points.is_empty() {
            return Self::default();
        }
        
        let mut min = points[0];
        let mut max = points[0];
        
        for &point in points.iter().skip(1) {
            min = min.min(point);
            max = max.max(point);
        }
        
        Self { min, max }
    }
    
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) / 2.0
    }
    
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
    
    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }
    
    pub fn expand_by_box(&mut self, other: &BoundingBox) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
    
    pub fn transform(&self, transform: &Mat4) -> Self {
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];
        
        let transformed_corners: Vec<Vec3> = corners
            .iter()
            .map(|&corner| (*transform * corner.extend(1.0)).truncate())
            .collect();
        
        Self::from_points(&transformed_corners)
    }
    
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
    }
    
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }
}

/// Scene graph managing hierarchical plot objects
pub struct Scene {
    nodes: HashMap<NodeId, SceneNode>,
    root_nodes: Vec<NodeId>,
    next_id: NodeId,
    
    // Cached data for optimization
    world_bounds: BoundingBox,
    bounds_dirty: bool,
    
    // Culling and LOD
    frustum: Option<Frustum>,
    camera_position: Vec3,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            root_nodes: Vec::new(),
            next_id: 1,
            world_bounds: BoundingBox::default(),
            bounds_dirty: true,
            frustum: None,
            camera_position: Vec3::ZERO,
        }
    }
    
    /// Add a new node to the scene
    pub fn add_node(&mut self, mut node: SceneNode) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        
        node.id = id;
        
        // Add to parent's children if specified
        if let Some(parent_id) = node.parent {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.children.push(id);
            }
        } else {
            self.root_nodes.push(id);
        }
        
        self.nodes.insert(id, node);
        self.bounds_dirty = true;
        id
    }
    
    /// Remove a node and all its children
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        // Get the node data first to avoid borrowing conflicts
        let (parent_id, children) = if let Some(node) = self.nodes.get(&id) {
            (node.parent, node.children.clone())
        } else {
            return false;
        };
        
        // Remove from parent's children
        if let Some(parent_id) = parent_id {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.children.retain(|&child_id| child_id != id);
            }
        } else {
            self.root_nodes.retain(|&root_id| root_id != id);
        }
        
        // Recursively remove children
        for child_id in children {
            self.remove_node(child_id);
        }
        
        self.nodes.remove(&id);
        self.bounds_dirty = true;
        true
    }
    
    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }
    
    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        if self.nodes.contains_key(&id) {
            self.bounds_dirty = true;
        }
        self.nodes.get_mut(&id)
    }
    
    /// Update world transform for a node and its children
    pub fn update_transforms(&mut self, root_transform: Mat4) {
        for &root_id in &self.root_nodes.clone() {
            self.update_node_transform(root_id, root_transform);
        }
    }
    
    fn update_node_transform(&mut self, node_id: NodeId, parent_transform: Mat4) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            let world_transform = parent_transform * node.transform;
            
            // Update bounding box
            if let Some(render_data) = &node.render_data {
                let local_bounds = BoundingBox::from_points(
                    &render_data.vertices.iter()
                        .map(|v| Vec3::from_array(v.position))
                        .collect::<Vec<_>>()
                );
                node.bounds = local_bounds.transform(&world_transform);
            }
            
            // Recursively update children
            let children = node.children.clone();
            for child_id in children {
                self.update_node_transform(child_id, world_transform);
            }
        }
    }
    
    /// Get the overall bounding box of the scene
    pub fn world_bounds(&mut self) -> BoundingBox {
        if self.bounds_dirty {
            self.update_world_bounds();
        }
        self.world_bounds
    }
    
    fn update_world_bounds(&mut self) {
        self.world_bounds = BoundingBox::default();
        
        for node in self.nodes.values() {
            if node.visible {
                self.world_bounds.expand_by_box(&node.bounds);
            }
        }
        
        self.bounds_dirty = false;
    }
    
    /// Set camera position for LOD calculations
    pub fn set_camera_position(&mut self, position: Vec3) {
        self.camera_position = position;
        self.update_lod();
    }
    
    /// Update level of detail for all nodes based on camera distance
    fn update_lod(&mut self) {
        for node in self.nodes.values_mut() {
            if !node.lod_levels.is_empty() {
                let distance = node.bounds.center().distance(self.camera_position);
                
                // Find appropriate LOD level
                let mut lod_index = node.lod_levels.len() - 1;
                for (i, lod) in node.lod_levels.iter().enumerate() {
                    if distance <= lod.distance {
                        lod_index = i;
                        break;
                    }
                }
                
                node.current_lod = lod_index;
            }
        }
    }
    
    /// Get visible nodes for rendering (with frustum culling)
    pub fn get_visible_nodes(&self) -> Vec<&SceneNode> {
        self.nodes.values()
            .filter(|node| {
                node.visible && 
                node.render_data.is_some() &&
                self.is_node_in_frustum(node)
            })
            .collect()
    }
    
    fn is_node_in_frustum(&self, node: &SceneNode) -> bool {
        // If no frustum is set, all nodes are visible
        if let Some(ref frustum) = self.frustum {
            frustum.intersects_box(&node.bounds)
        } else {
            true
        }
    }
    
    /// Set frustum for culling
    pub fn set_frustum(&mut self, frustum: Frustum) {
        self.frustum = Some(frustum);
    }
    
    /// Clear all nodes
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root_nodes.clear();
        self.bounds_dirty = true;
    }
    
    /// Get statistics about the scene
    pub fn statistics(&self) -> SceneStatistics {
        let visible_nodes = self.nodes.values().filter(|n| n.visible).count();
        let total_vertices: usize = self.nodes.values()
            .filter_map(|n| n.render_data.as_ref())
            .map(|rd| rd.vertices.len())
            .sum();
        let total_triangles: usize = self.nodes.values()
            .filter_map(|n| n.render_data.as_ref())
            .filter(|rd| rd.pipeline_type == PipelineType::Triangles)
            .map(|rd| rd.indices.as_ref().map_or(rd.vertices.len() / 3, |i| i.len() / 3))
            .sum();
        
        SceneStatistics {
            total_nodes: self.nodes.len(),
            visible_nodes,
            total_vertices,
            total_triangles,
        }
    }
}

/// View frustum for culling
#[derive(Debug, Clone)]
pub struct Frustum {
    pub planes: [Plane; 6], // left, right, bottom, top, near, far
}

impl Frustum {
    pub fn from_view_proj(view_proj: Mat4) -> Self {
        let m = view_proj.to_cols_array_2d();
        
        // Extract frustum planes from view-projection matrix
        let planes = [
            // Left plane
            Plane::new(
                m[0][3] + m[0][0],
                m[1][3] + m[1][0],
                m[2][3] + m[2][0],
                m[3][3] + m[3][0],
            ),
            // Right plane
            Plane::new(
                m[0][3] - m[0][0],
                m[1][3] - m[1][0],
                m[2][3] - m[2][0],
                m[3][3] - m[3][0],
            ),
            // Bottom plane
            Plane::new(
                m[0][3] + m[0][1],
                m[1][3] + m[1][1],
                m[2][3] + m[2][1],
                m[3][3] + m[3][1],
            ),
            // Top plane
            Plane::new(
                m[0][3] - m[0][1],
                m[1][3] - m[1][1],
                m[2][3] - m[2][1],
                m[3][3] - m[3][1],
            ),
            // Near plane
            Plane::new(
                m[0][3] + m[0][2],
                m[1][3] + m[1][2],
                m[2][3] + m[2][2],
                m[3][3] + m[3][2],
            ),
            // Far plane
            Plane::new(
                m[0][3] - m[0][2],
                m[1][3] - m[1][2],
                m[2][3] - m[2][2],
                m[3][3] - m[3][2],
            ),
        ];
        
        Self { planes }
    }
    
    pub fn intersects_box(&self, bbox: &BoundingBox) -> bool {
        for plane in &self.planes {
            if plane.distance_to_box(bbox) > 0.0 {
                return false; // Box is outside this plane
            }
        }
        true // Box intersects or is inside frustum
    }
}

/// Plane equation ax + by + cz + d = 0
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f32,
}

impl Plane {
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        let normal = Vec3::new(a, b, c);
        let length = normal.length();
        
        Self {
            normal: normal / length,
            distance: d / length,
        }
    }
    
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.distance
    }
    
    pub fn distance_to_box(&self, bbox: &BoundingBox) -> f32 {
        // Find the positive vertex (farthest in direction of normal)
        let positive_vertex = Vec3::new(
            if self.normal.x >= 0.0 { bbox.max.x } else { bbox.min.x },
            if self.normal.y >= 0.0 { bbox.max.y } else { bbox.min.y },
            if self.normal.z >= 0.0 { bbox.max.z } else { bbox.min.z },
        );
        
        self.distance_to_point(positive_vertex)
    }
}

/// Scene statistics for debugging and optimization
#[derive(Debug, Clone)]
pub struct SceneStatistics {
    pub total_nodes: usize,
    pub visible_nodes: usize,
    pub total_vertices: usize,
    pub total_triangles: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bounding_box_creation() {
        let points = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        
        let bbox = BoundingBox::from_points(&points);
        assert_eq!(bbox.min, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(bbox.max, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(bbox.center(), Vec3::ZERO);
    }
    
    #[test]
    fn test_scene_node_hierarchy() {
        let mut scene = Scene::new();
        
        let parent_node = SceneNode {
            id: 0,
            name: "Parent".to_string(),
            transform: Mat4::IDENTITY,
            visible: true,
            cast_shadows: true,
            receive_shadows: true,
            parent: None,
            children: Vec::new(),
            render_data: None,
            bounds: BoundingBox::default(),
            lod_levels: Vec::new(),
            current_lod: 0,
        };
        
        let parent_id = scene.add_node(parent_node);
        
        let child_node = SceneNode {
            id: 0,
            name: "Child".to_string(),
            transform: Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            visible: true,
            cast_shadows: true,
            receive_shadows: true,
            parent: Some(parent_id),
            children: Vec::new(),
            render_data: None,
            bounds: BoundingBox::default(),
            lod_levels: Vec::new(),
            current_lod: 0,
        };
        
        let child_id = scene.add_node(child_node);
        
        // Verify hierarchy
        let parent = scene.get_node(parent_id).unwrap();
        assert!(parent.children.contains(&child_id));
        
        let child = scene.get_node(child_id).unwrap();
        assert_eq!(child.parent, Some(parent_id));
    }
}