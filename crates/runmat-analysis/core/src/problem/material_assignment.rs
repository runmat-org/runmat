use serde::{Deserialize, Serialize};

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceConfidence {
    Verified,
    Probable,
    Inferred,
}

/// A material-bearing analysis domain. This is deliberately distinct from a boundary selection:
/// a CAD face may define a shell domain or a boundary face, and those meanings are not
/// interchangeable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyRegionTarget {
    Beam(PropertyRegionScope),
    Shell(PropertyRegionScope),
    Solid(PropertyRegionScope),
    Cell(PropertyRegionScope),
}

/// The coverage scope inside one material-bearing formulation domain.
///
/// `All` is an explicit authored choice, never an implicit fallback inferred from the number of
/// declared materials. `Region` may name one persistent entity or an authored semantic group and
/// is resolved against geometry/mesh provenance before assembly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyRegionScope {
    All,
    Region(PropertyRegionSelection),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropertyRegionSelection {
    region_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    resolved_entities: Vec<PersistentEntityId>,
}

impl PropertyRegionTarget {
    pub fn beam(region_id: impl Into<String>) -> Self {
        Self::Beam(PropertyRegionScope::region(region_id))
    }

    pub fn shell(region_id: impl Into<String>) -> Self {
        Self::Shell(PropertyRegionScope::region(region_id))
    }

    pub fn solid(region_id: impl Into<String>) -> Self {
        Self::Solid(PropertyRegionScope::region(region_id))
    }

    pub fn cell(region_id: impl Into<String>) -> Self {
        Self::Cell(PropertyRegionScope::region(region_id))
    }

    pub const fn all_beams() -> Self {
        Self::Beam(PropertyRegionScope::All)
    }

    pub const fn all_shells() -> Self {
        Self::Shell(PropertyRegionScope::All)
    }

    pub const fn all_solids() -> Self {
        Self::Solid(PropertyRegionScope::All)
    }

    pub const fn all_cells() -> Self {
        Self::Cell(PropertyRegionScope::All)
    }

    pub const fn scope(&self) -> &PropertyRegionScope {
        match self {
            Self::Beam(scope) | Self::Shell(scope) | Self::Solid(scope) | Self::Cell(scope) => {
                scope
            }
        }
    }

    pub fn scope_mut(&mut self) -> &mut PropertyRegionScope {
        match self {
            Self::Beam(scope) | Self::Shell(scope) | Self::Solid(scope) | Self::Cell(scope) => {
                scope
            }
        }
    }

    pub fn selected_region_id(&self) -> Option<&str> {
        self.scope().selected_region_id()
    }

    pub fn selected_region_id_mut(&mut self) -> Option<&mut String> {
        self.scope_mut().selected_region_id_mut()
    }

    pub const fn selects_all(&self) -> bool {
        self.scope().selects_all()
    }

    pub fn bind_resolved_entities(
        &mut self,
        entities: Vec<PersistentEntityId>,
    ) -> Result<(), PropertyRegionResolutionError> {
        let expected = match self {
            Self::Beam(_) => PersistentEntityKind::Edge,
            Self::Shell(_) => PersistentEntityKind::Face,
            Self::Solid(_) => PersistentEntityKind::Solid,
            Self::Cell(_) => PersistentEntityKind::Region,
        };
        let PropertyRegionScope::Region(selection) = self.scope_mut() else {
            return Ok(());
        };
        selection.bind_resolved_entities(expected, entities)
    }

    pub fn validate_resolution(&self) -> Result<(), PropertyRegionResolutionError> {
        let expected = match self {
            Self::Beam(_) => PersistentEntityKind::Edge,
            Self::Shell(_) => PersistentEntityKind::Face,
            Self::Solid(_) => PersistentEntityKind::Solid,
            Self::Cell(_) => PersistentEntityKind::Region,
        };
        match self.scope() {
            PropertyRegionScope::All => Ok(()),
            PropertyRegionScope::Region(selection) => selection.validate_resolution(expected),
        }
    }
}

impl PropertyRegionScope {
    fn region(region_id: impl Into<String>) -> Self {
        Self::Region(PropertyRegionSelection::new(region_id))
    }

    pub fn selected_region_id(&self) -> Option<&str> {
        match self {
            Self::All => None,
            Self::Region(selection) => Some(&selection.region_id),
        }
    }

    pub fn selected_region_id_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::All => None,
            Self::Region(selection) => Some(&mut selection.region_id),
        }
    }

    pub const fn selects_all(&self) -> bool {
        matches!(self, Self::All)
    }
}

impl PropertyRegionSelection {
    fn new(region_id: impl Into<String>) -> Self {
        Self {
            region_id: region_id.into(),
            resolved_entities: Vec::new(),
        }
    }

    pub fn region_id(&self) -> &str {
        &self.region_id
    }

    pub fn resolved_entities(&self) -> &[PersistentEntityId] {
        &self.resolved_entities
    }

    pub fn bind_resolved_entities(
        &mut self,
        expected: PersistentEntityKind,
        mut entities: Vec<PersistentEntityId>,
    ) -> Result<(), PropertyRegionResolutionError> {
        entities.sort();
        entities.dedup();
        if entities.is_empty()
            || entities
                .iter()
                .any(|entity| entity.validate().is_err() || entity.kind != expected)
        {
            return Err(PropertyRegionResolutionError {
                region_id: self.region_id.clone(),
                expected,
                count: entities.len(),
            });
        }
        self.resolved_entities = entities;
        Ok(())
    }

    pub fn matches_persistent_entity(&self, entity: &PersistentEntityId) -> bool {
        if self.resolved_entities.is_empty() {
            entity.matches_analysis_selection(&self.region_id)
        } else {
            self.resolved_entities.binary_search(entity).is_ok()
        }
    }

    fn validate_resolution(
        &self,
        expected: PersistentEntityKind,
    ) -> Result<(), PropertyRegionResolutionError> {
        if self.resolved_entities.is_empty() {
            return Ok(());
        }
        if !self
            .resolved_entities
            .windows(2)
            .all(|pair| pair[0] < pair[1])
            || self
                .resolved_entities
                .iter()
                .any(|entity| entity.validate().is_err() || entity.kind != expected)
        {
            return Err(PropertyRegionResolutionError {
                region_id: self.region_id.clone(),
                expected,
                count: self.resolved_entities.len(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "property region `{region_id}` resolved {count} entities incompatible with required {expected:?} domain"
)]
pub struct PropertyRegionResolutionError {
    pub region_id: String,
    pub expected: PersistentEntityKind,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialAssignment {
    pub target: PropertyRegionTarget,
    pub expected_material_id: String,
    pub assigned_material_id: String,
    pub confidence: EvidenceConfidence,
}

impl MaterialAssignment {
    pub fn to_solid_region(
        region_id: impl Into<String>,
        material_id: impl Into<String>,
        confidence: EvidenceConfidence,
    ) -> Self {
        let material_id = material_id.into();
        Self {
            target: PropertyRegionTarget::solid(region_id),
            expected_material_id: material_id.clone(),
            assigned_material_id: material_id,
            confidence,
        }
    }

    pub fn selected_region_id(&self) -> Option<&str> {
        self.target.selected_region_id()
    }

    pub fn target_label(&self) -> &str {
        self.selected_region_id().unwrap_or(match &self.target {
            PropertyRegionTarget::Beam(_) => "<all beams>",
            PropertyRegionTarget::Shell(_) => "<all shells>",
            PropertyRegionTarget::Solid(_) => "<all solids>",
            PropertyRegionTarget::Cell(_) => "<all cells>",
        })
    }
}
