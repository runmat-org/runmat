//! The solved bodies a result view draws. Each volume element of the solver mesh belongs to the
//! study region (the CAD name) the model assigns its material to; a view's surface carries those
//! regions as scene regions, so a viewer can hide or isolate a body, and the study's display
//! (`display <output> { hide = […]; context = […] }`) leaves bodies out or draws them as grey
//! context. Only the bodies drawn in field colour count toward the legend's extremes.
pub use crate::analysis::contracts::AnalysisStudyRegion;
use runmat_analysis_core::AuthoredOutputDisplay;
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::HashMap;

/// The scene region tag of a solved body's surface.
pub const BODY_REGION_TAG: &str = "solid";

/// The study region of each volume element of one run's solver mesh.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BodyRegions {
    /// Region names (CAD names), in the study's assignment order.
    pub names: Vec<String>,
    /// For each volume element in solver order, its index in `names`.
    elements: Vec<Option<u32>>,
}

impl BodyRegions {
    /// Each element's region: the first study region whose exact entities include the element's
    /// mesh region or one of the CAD solids it was meshed from.
    pub fn from_solver_mesh(mesh: &SolverMeshArtifact, regions: &[AnalysisStudyRegion]) -> Self {
        let mut owner = HashMap::new();
        for (index, region) in regions.iter().enumerate() {
            for entity in &region.entity_ids {
                owner.entry(entity).or_insert(index as u32);
            }
        }
        let elements = mesh
            .topology
            .volume_elements
            .iter()
            .map(|element| {
                std::iter::once(&element.region_id)
                    .chain(&element.provenance)
                    .filter_map(|entity| owner.get(entity).copied())
                    .min()
            })
            .collect();
        Self {
            names: regions
                .iter()
                .map(|region| region.region_id.clone())
                .collect(),
            elements,
        }
    }

    /// The bodies of the run whose canonical solver mesh is at `path`.
    pub fn load(path: &str, regions: &[AnalysisStudyRegion]) -> Result<Self, String> {
        let mesh = super::super::solver_mesh_artifact::load_solver_mesh_artifact(
            std::path::Path::new(path),
        )
        .map_err(|error| error.to_string())?;
        Ok(Self::from_solver_mesh(&mesh, regions))
    }

    /// Bodies given element by element, for tests of the views drawn from them.
    #[cfg(test)]
    pub(crate) fn of_elements(names: &[&str], elements: Vec<Option<u32>>) -> Self {
        Self {
            names: names.iter().map(|name| name.to_string()).collect(),
            elements,
        }
    }

    /// The region index of volume element `element`, when the study names its body.
    pub fn of_element(&self, element: usize) -> Option<usize> {
        self.elements
            .get(element)
            .copied()
            .flatten()
            .map(|index| index as usize)
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// How a view draws one solved body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionDraw {
    /// In field colour; its values count toward the legend's extremes.
    Field,
    /// Translucent grey with no field colour; its values do not count.
    Context,
    /// Not drawn; its values do not count.
    Hidden,
}

/// What one view draws of a run's solved bodies.
#[derive(Debug, Clone, Default)]
pub struct ViewRegions {
    pub bodies: Option<std::sync::Arc<BodyRegions>>,
    /// Region names the study's display hides.
    pub hide: Vec<String>,
    /// Region names the study's display draws as context.
    pub context: Vec<String>,
}

impl ViewRegions {
    /// The view of `bodies` that `display` (the study's display of the shown output) asks for.
    pub fn of_display(
        bodies: Option<std::sync::Arc<BodyRegions>>,
        display: Option<&AuthoredOutputDisplay>,
    ) -> Self {
        Self {
            bodies,
            hide: display
                .map(|display| display.hide.clone())
                .unwrap_or_default(),
            context: display
                .map(|display| display.context.clone())
                .unwrap_or_default(),
        }
    }

    /// Whether some body is drawn other than in field colour.
    pub fn filters(&self) -> bool {
        self.bodies.is_some() && !(self.hide.is_empty() && self.context.is_empty())
    }

    /// How the view draws the body of volume element `element` (an element the study names no
    /// body for is drawn in field colour).
    pub fn draw_of_element(&self, element: Option<usize>) -> RegionDraw {
        let name = element
            .zip(self.bodies.as_deref())
            .and_then(|(element, bodies)| Some(&bodies.names[bodies.of_element(element)?]));
        match name {
            Some(name) if self.hide.contains(name) => RegionDraw::Hidden,
            Some(name) if self.context.contains(name) => RegionDraw::Context,
            _ => RegionDraw::Field,
        }
    }

    /// The volume elements whose values count toward the legend's extremes, and the nodes of
    /// those elements (`element_nodes` gives each element's node indices), when the view leaves
    /// some body out of them.
    pub fn counted_entities(&self, element_nodes: &[Vec<usize>]) -> Option<CountedEntities> {
        if !self.filters() {
            return None;
        }
        let mut counted = CountedEntities {
            elements: Vec::with_capacity(element_nodes.len()),
            nodes: Vec::new(),
        };
        for (element, nodes) in element_nodes.iter().enumerate() {
            let shown = self.draw_of_element(Some(element)) == RegionDraw::Field;
            counted.elements.push(shown);
            if shown {
                for &node in nodes {
                    if counted.nodes.len() <= node {
                        counted.nodes.resize(node + 1, false);
                    }
                    counted.nodes[node] = true;
                }
            }
        }
        Some(counted)
    }

    /// Region names the display lists that no solved body of the run carries.
    pub fn unknown_regions(&self) -> Vec<&str> {
        let Some(bodies) = self.bodies.as_deref() else {
            return Vec::new();
        };
        self.hide
            .iter()
            .chain(&self.context)
            .filter(|name| !bodies.names.contains(name))
            .map(String::as_str)
            .collect()
    }
}

/// The mesh entities whose values a view counts toward its legend's extremes.
#[derive(Debug, Clone, PartialEq)]
pub struct CountedEntities {
    /// One flag per volume element in solver order.
    pub elements: Vec<bool>,
    /// One flag per node in solver order (missing trailing nodes are not counted).
    pub nodes: Vec<bool>,
}

impl CountedEntities {
    pub fn counts_node(&self, node: usize) -> bool {
        self.nodes.get(node).copied().unwrap_or(false)
    }

    pub fn counts_element(&self, element: usize) -> bool {
        self.elements.get(element).copied().unwrap_or(false)
    }
}
