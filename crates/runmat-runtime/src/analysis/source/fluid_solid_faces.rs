//! Check-time fluid–solid face admission on the imported CAD.
//!
//! A fluid–structure study's run classifies every mesh face that joins a fluid cell and a solid
//! element with [`classify_fluid_solid_face`]. The check applies the same rule to the faces two
//! solids of the CAD share (`interfaces = shared`): a fluid solid is one assigned a fluid
//! material, a structural solid one assigned an elastic material. Regions are matched by the
//! CAD names their `select name` selectors choose; when a region the rule needs selects by
//! anything else, the check leaves the decision to the run.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AuthoredConstraintKind, AuthoredInterfaceKind, AuthoredMaterialSource, AuthoredModel,
    AuthoredPropertyTarget, AuthoredRegionSource, CoupledPhysicsFamily, PhysicsFamily,
    RealizedStudy,
};
use runmat_analysis_fea::admission::{
    classify_fluid_solid_face, is_authored_fluid_wall, AdmissionIssue,
};
use runmat_analysis_language::{GeometryInspection, ResolvedAnalysisStudy};
use runmat_geometry_core::GeometryRegionSelector;

/// Issues for the CAD faces that join a fluid and a solid in a fluid–structure study. Empty when
/// the study has no fluid–structure step, the geometry has no shared faces, or a region the rule
/// needs is not selected by name.
pub(super) fn fluid_solid_face_issues(
    resolved: &ResolvedAnalysisStudy,
    realized: &RealizedStudy,
    inspection: Option<&GeometryInspection>,
) -> Vec<AdmissionIssue> {
    let fsi = PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction);
    if !realized
        .sequence
        .steps
        .iter()
        .any(|step| step.family == fsi)
    {
        return Vec::new();
    }
    let Some(inspection) = inspection.filter(|i| !i.shared_faces.is_empty()) else {
        return Vec::new();
    };
    classify(resolved, realized, inspection).unwrap_or_default()
}

/// `None` when a region the rule needs cannot be named at check.
fn classify(
    resolved: &ResolvedAnalysisStudy,
    realized: &RealizedStudy,
    inspection: &GeometryInspection,
) -> Option<Vec<AdmissionIssue>> {
    let geometry = &resolved.geometry;
    let region_names = geometry
        .regions
        .iter()
        .map(|region| {
            let name = match &region.selector {
                GeometryRegionSelector::Name(name) => Some(name.as_str()),
                _ => None,
            };
            (region.region_id.as_str(), name)
        })
        .collect::<BTreeMap<_, _>>();
    let name_of = |source: &AuthoredRegionSource| -> Option<&str> {
        match source {
            AuthoredRegionSource::Geometry(region)
                if region.module_path == geometry.module_path
                    && region.geometry_id == geometry.geometry_id =>
            {
                region_names
                    .get(region.region_id.as_str())
                    .copied()
                    .flatten()
            }
            _ => None,
        }
    };
    let models = std::iter::once(&realized.model)
        .chain(&realized.included_models)
        .collect::<Vec<&AuthoredModel>>();
    let mut fluid_solids = BTreeSet::new();
    let mut structural_solids = BTreeSet::new();
    for assignment in models.iter().flat_map(|m| &m.material_assignments) {
        let (AuthoredPropertyTarget::Solid(source) | AuthoredPropertyTarget::Cell(source)) =
            &assignment.target
        else {
            continue;
        };
        let AuthoredMaterialSource::Declaration(reference) = &assignment.material else {
            return None;
        };
        let material = resolved.material_catalog.get(reference)?;
        let name = name_of(source)?;
        if material.fluid.is_some() {
            fluid_solids.insert(name);
        } else if material.elasticity.is_some() {
            structural_solids.insert(name);
        }
    }
    let mut interfaces = Vec::new();
    for interface in models.iter().flat_map(|m| &m.interfaces) {
        if interface.kind == AuthoredInterfaceKind::FluidStructureNoSlip {
            interfaces.push((
                interface.interface_id.as_str(),
                [name_of(&interface.primary)?, name_of(&interface.secondary)?],
            ));
        }
    }
    let mut walls = Vec::new();
    for constraint in models.iter().flat_map(|m| &m.constraints) {
        if !is_authored_fluid_wall(&constraint.kind) {
            continue;
        }
        let (AuthoredConstraintKind::FluidNoSlipWall { target }
        | AuthoredConstraintKind::FluidSlipWall { target }
        | AuthoredConstraintKind::FluidMovingWall { target, .. }) = &constraint.kind
        else {
            continue;
        };
        walls.push((constraint.constraint_id.as_str(), name_of(&target.region)?));
    }
    let is =
        |names: &[String], set: &BTreeSet<&str>| names.iter().any(|n| set.contains(n.as_str()));
    let mut issues = Vec::<AdmissionIssue>::new();
    for face in &inspection.shared_faces {
        let [a, b] = &face.solid_names;
        let joins = (is(a, &fluid_solids) && is(b, &structural_solids))
            || (is(b, &fluid_solids) && is(a, &structural_solids));
        if !joins {
            continue;
        }
        let carries = |name: &str| face.face_names.iter().any(|n| n == name);
        let coupling = interfaces
            .iter()
            .filter(|(_, names)| names.iter().any(|n| carries(n)))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let wall = walls
            .iter()
            .filter(|(_, name)| carries(name))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let label = if face.face_names.is_empty() {
            format!(
                "`{}` between solids {} and {}",
                face.face,
                quoted(a),
                quoted(b)
            )
        } else {
            quoted(&face.face_names)
        };
        if let Err(issue) = classify_fluid_solid_face(&label, &coupling, &wall) {
            if !issues.contains(&issue) {
                issues.push(issue);
            }
        }
    }
    Some(issues)
}

fn quoted(names: &[String]) -> String {
    if names.is_empty() {
        return "(unnamed)".into();
    }
    names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(", ")
}
