//! Borrowed solver-semantic collections; serialization projects one original item at a time.
use super::*;
use serde::ser::SerializeSeq;

pub struct SolverMaterials<'a>(pub(super) &'a [MaterialModel]);
pub struct SolverMaterialAssignments<'a>(pub(super) &'a [MaterialAssignment]);
pub struct SolverSteps<'a>(pub(super) &'a [ProcedureStep]);

fn serialize_items<'a, T: 'a, U: Serialize, S: serde::Serializer>(
    items: &'a [T],
    serializer: S,
    mut project: impl FnMut(&'a T) -> U,
) -> Result<S::Ok, S::Error> {
    let mut sequence = serializer.serialize_seq(Some(items.len()))?;
    for item in items {
        sequence.serialize_element(&project(item))?;
    }
    sequence.end()
}

impl Serialize for SolverMaterials<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_items(self.0, serializer, MaterialModel::solver_identity_contract)
    }
}
impl Serialize for SolverMaterialAssignments<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_items(self.0, serializer, |assignment| SolverMaterialAssignment {
            target: &assignment.target,
            assigned_material_id: &assignment.assigned_material_id,
        })
    }
}
impl Serialize for SolverSteps<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_items(self.0, serializer, ProcedureStep::solver_identity_contract)
    }
}
