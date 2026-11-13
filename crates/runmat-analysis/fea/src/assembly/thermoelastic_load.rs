//! Physical equivalent nodal loads for element-owned prescribed thermoelastic strain.

use std::collections::BTreeSet;

use runmat_meshing_core::SolverMeshArtifact;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use thiserror::Error;

use crate::{
    properties::{CompiledElementMaterial, CompiledPropertyMap},
    topology::AnalysisProjection,
    FeaThermoMechanicalContext,
};

#[derive(Debug, Error)]
pub(crate) enum ThermoelasticLoadError {
    #[error("thermoelastic numerical accounting failed: {0}")]
    Accounting(#[from] LedgerError),
    #[error("thermoelastic projection cancelled")]
    Cancelled,
    #[error("thermoelastic element projection is invalid: {0}")]
    Projection(String),
    #[error("thermoelastic solid element {0} has no compiled material owner")]
    MissingOwner(u64),
    #[error("thermoelastic solid element {0} has invalid constitutive coefficients")]
    InvalidConstitutive(u64),
    #[error("thermoelastic prescribed temperature is invalid for region {0}")]
    InvalidTemperature(String),
    #[error("thermoelastic temperature targets duplicate region {0}")]
    DuplicateRegion(String),
    #[error("thermoelastic temperature targets region {0} absent from the solver mesh")]
    UnknownRegion(String),
    #[error("solved thermoelastic temperature field is incomplete or inconsistent with the mesh")]
    InvalidNodalField,
}

fn projection_error(error: crate::topology::AnalysisProjectionError) -> ThermoelasticLoadError {
    match error {
        crate::topology::AnalysisProjectionError::Accounting(cause) => {
            ThermoelasticLoadError::Accounting(cause)
        }
        crate::topology::AnalysisProjectionError::Cancelled => ThermoelasticLoadError::Cancelled,
        other => ThermoelasticLoadError::Projection(other.to_string()),
    }
}

#[derive(Debug)]
pub(crate) enum ThermoelasticLoadCapacityError<A> {
    Numerical(ThermoelasticLoadError),
    Admission(A),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ThermoelasticCapacityPhase {
    Projection,
    Element { element: usize },
}

impl<A> From<ThermoelasticLoadError> for ThermoelasticLoadCapacityError<A> {
    fn from(error: ThermoelasticLoadError) -> Self {
        Self::Numerical(error)
    }
}

impl<A> From<LedgerError> for ThermoelasticLoadCapacityError<A> {
    fn from(error: LedgerError) -> Self {
        Self::Numerical(ThermoelasticLoadError::Accounting(error))
    }
}

fn quadrature_work(order: runmat_meshing_core::ElementOrder, prescribed: bool) -> OperationCount {
    let per_point = match order {
        runmat_meshing_core::ElementOrder::Tet4 => 39 * 4 + 105,
        runmat_meshing_core::ElementOrder::Tet10 => 116 + 39 * 10 + 105,
    };
    OperationCount::from_operations(
        crate::topology::quadrature::KEAST_FIFTEEN.len() as u64 * per_point
            + if prescribed { 1 + 2 + 4 } else { 0 },
    )
}

#[cfg(test)]
fn immediate<R>(
    future: impl std::future::Future<
        Output = Result<R, ThermoelasticLoadCapacityError<std::convert::Infallible>>,
    >,
) -> Result<R, ThermoelasticLoadError> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => result.map_err(|error| match error {
            ThermoelasticLoadCapacityError::Numerical(error) => error,
            ThermoelasticLoadCapacityError::Admission(never) => match never {},
        }),
        std::task::Poll::Pending => unreachable!("immediate thermal load cannot suspend"),
    }
}

pub(super) fn validate_temperature_regions(
    mesh: &SolverMeshArtifact,
    context: &FeaThermoMechanicalContext,
) -> Result<(), ThermoelasticLoadError> {
    let mut targeted = BTreeSet::new();
    for target in &context.region_temperature_deltas {
        if !targeted.insert(target.region_id()) {
            return Err(ThermoelasticLoadError::DuplicateRegion(
                target.region_id().into(),
            ));
        }
        if !mesh
            .topology
            .volume_elements
            .iter()
            .any(|element| target.contains_element(&element.region_id, &element.provenance))
        {
            return Err(ThermoelasticLoadError::UnknownRegion(
                target.region_id().into(),
            ));
        }
        let temperature =
            context.reference_temperature_k.canonical_value() + target.temperature_delta_k;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(ThermoelasticLoadError::InvalidTemperature(
                target.region_id().into(),
            ));
        }
    }
    // Each element takes at most one prescribed regional temperature.
    for element in &mesh.topology.volume_elements {
        let owners = context
            .region_temperature_deltas
            .iter()
            .filter(|target| target.contains_element(&element.region_id, &element.provenance))
            .count();
        if owners > 1 {
            return Err(ThermoelasticLoadError::DuplicateRegion(
                element.region_id.source_topology_id.clone(),
            ));
        }
    }
    Ok(())
}

/// Prescribed temperature of a solver element: its region's temperature when a regional
/// field covers it, otherwise the whole-model temperature.
pub(super) fn element_temperature_k(
    region: &runmat_meshing_core::PersistentEntityId,
    provenance: &[runmat_meshing_core::PersistentEntityId],
    context: &FeaThermoMechanicalContext,
) -> Result<f64, ThermoelasticLoadError> {
    let delta = context
        .region_temperature_deltas
        .iter()
        .find(|target| target.contains_element(region, provenance))
        .map_or(
            context.applied_temperature_delta_k.canonical_value(),
            |target| target.temperature_delta_k,
        );
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Assembly, OperationCount::from_operations(1))
        })
        .transpose()?;
    let temperature = context.reference_temperature_k.canonical_value() + delta;
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(1))?;
    }
    if temperature.is_finite() && temperature > 0.0 {
        Ok(temperature)
    } else {
        Err(ThermoelasticLoadError::InvalidTemperature(
            region.source_topology_id.clone(),
        ))
    }
}

pub(super) fn element_thermal_eigenstrain(
    owner: &CompiledElementMaterial,
    region: &runmat_meshing_core::PersistentEntityId,
    provenance: &[runmat_meshing_core::PersistentEntityId],
    context: &FeaThermoMechanicalContext,
) -> Result<f64, ThermoelasticLoadError> {
    let thermal = owner
        .material
        .require_thermal()
        .map_err(|_| ThermoelasticLoadError::InvalidConstitutive(owner.element_id))?;
    let temperature_k = element_temperature_k(region, provenance, context)?;
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Assembly, OperationCount::from_operations(2))
        })
        .transpose()?;
    let strain = thermal.expansion_coefficient_per_k.canonical_value()
        * (temperature_k - thermal.reference_temperature_k.canonical_value());
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(2))?;
    }
    if strain.is_finite() {
        Ok(strain)
    } else {
        Err(ThermoelasticLoadError::InvalidConstitutive(
            owner.element_id,
        ))
    }
}

/// Assemble ∫ Bᵀ D εₜₕ dV with element-constant εₜₕ = α(T − Tref)[1,1,1,0,0,0].
/// The isotropic hydrostatic stress is E α(T − Tref)/(1−2ν), so each nodal force component is
/// its volume integral against the corresponding physical shape gradient. No material average or
/// ordinal DOF distribution enters this load.
#[cfg(test)]
pub(super) fn assemble_prescribed_temperature_load(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    context: &FeaThermoMechanicalContext,
) -> Result<Vec<f64>, ThermoelasticLoadError> {
    let mut owner = |_, _| std::future::ready(Ok::<(), std::convert::Infallible>(()));
    immediate(assemble_prescribed_temperature_load_with_owner(
        mesh, properties, context, &mut owner,
    ))
}

pub(super) async fn assemble_prescribed_temperature_load_with_owner<O, F, A>(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    context: &FeaThermoMechanicalContext,
    before: &mut O,
) -> Result<Vec<f64>, ThermoelasticLoadCapacityError<A>>
where
    O: FnMut(ThermoelasticCapacityPhase, OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    validate_temperature_regions(mesh, context)?;
    let projection_work =
        AnalysisProjection::derivation_work_upper_bound(mesh).map_err(projection_error)?;
    if projection_work > OperationCount::ZERO {
        before(ThermoelasticCapacityPhase::Projection, projection_work)
            .await
            .map_err(ThermoelasticLoadCapacityError::Admission)?;
    }
    let projection = AnalysisProjection::derive(mesh).map_err(projection_error)?;
    let mut load = vec![0.0; projection.nodes.len() * 3];
    for (element_index, element) in projection.elements.iter().enumerate() {
        before(
            ThermoelasticCapacityPhase::Element {
                element: element_index,
            },
            quadrature_work(element.order, true),
        )
        .await
        .map_err(ThermoelasticLoadCapacityError::Admission)?;
        let owner = properties
            .solid_material(element.stable_identity)
            .ok_or(ThermoelasticLoadError::MissingOwner(element.element_id))?;
        let mechanical = owner
            .material
            .require_mechanical()
            .map_err(|_| ThermoelasticLoadError::InvalidConstitutive(element.element_id))?;
        let eigenstrain =
            element_thermal_eigenstrain(owner, &element.region_id, &element.provenance, context)?;
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Assembly, OperationCount::from_operations(4))
            })
            .transpose()?;
        let hydrostatic_stress_pa = mechanical.youngs_modulus_pa.canonical_value() * eigenstrain
            / (1.0 - 2.0 * mechanical.poisson_ratio.canonical_value());
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(4))?;
        }
        if !hydrostatic_stress_pa.is_finite() {
            return Err(ThermoelasticLoadError::InvalidConstitutive(element.element_id).into());
        }
        projection
            .visit_element_quadrature(element, |point| {
                for (&node_index, gradient) in element
                    .node_indices
                    .iter()
                    .zip(point.physical_shape_gradients())
                {
                    for axis in 0..3 {
                        load[node_index * 3 + axis] +=
                            hydrostatic_stress_pa * gradient[axis] * point.volume_weight_m3;
                    }
                }
            })
            .map_err(projection_error)?;
    }
    Ok(load)
}

/// Integrate the transferred nodal solution as a physical temperature field at each solid
/// quadrature point. No element-average or ordinal region proxy replaces the field interpolation.
#[cfg(test)]
pub(crate) fn assemble_solved_temperature_load(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    temperature: &crate::execution::SolvedTemperatureField,
) -> Result<Vec<f64>, ThermoelasticLoadError> {
    let mut owner = |_, _| std::future::ready(Ok::<(), std::convert::Infallible>(()));
    immediate(assemble_solved_temperature_load_with_owner(
        mesh,
        properties,
        temperature,
        &mut owner,
    ))
}

pub(super) async fn assemble_solved_temperature_load_with_owner<O, F, A>(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    temperature: &crate::execution::SolvedTemperatureField,
    before: &mut O,
) -> Result<Vec<f64>, ThermoelasticLoadCapacityError<A>>
where
    O: FnMut(ThermoelasticCapacityPhase, OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    let projection_work =
        AnalysisProjection::derivation_work_upper_bound(mesh).map_err(projection_error)?;
    if projection_work > OperationCount::ZERO {
        before(ThermoelasticCapacityPhase::Projection, projection_work)
            .await
            .map_err(ThermoelasticLoadCapacityError::Admission)?;
    }
    let projection = AnalysisProjection::derive(mesh).map_err(projection_error)?;
    if projection.nodes.len() != temperature.node_count()
        || projection.nodes.len() != mesh.topology.nodes.len()
        || projection
            .nodes
            .iter()
            .zip(&mesh.topology.nodes)
            .any(|(projected, canonical)| projected.node_id != canonical.node_id)
        || !temperature.all_positive_finite()
    {
        return Err(ThermoelasticLoadError::InvalidNodalField.into());
    }
    let mut load = vec![0.0; projection.nodes.len() * 3];
    for (element_index, element) in projection.elements.iter().enumerate() {
        before(
            ThermoelasticCapacityPhase::Element {
                element: element_index,
            },
            quadrature_work(element.order, false),
        )
        .await
        .map_err(ThermoelasticLoadCapacityError::Admission)?;
        let owner = properties
            .solid_material(element.stable_identity)
            .ok_or(ThermoelasticLoadError::MissingOwner(element.element_id))?;
        let mechanical = owner
            .material
            .require_mechanical()
            .map_err(|_| ThermoelasticLoadError::InvalidConstitutive(element.element_id))?;
        let thermal = owner
            .material
            .require_thermal()
            .map_err(|_| ThermoelasticLoadError::InvalidConstitutive(element.element_id))?;
        let coefficient = mechanical.youngs_modulus_pa.canonical_value()
            * thermal.expansion_coefficient_per_k.canonical_value()
            / (1.0 - 2.0 * mechanical.poisson_ratio.canonical_value());
        if !coefficient.is_finite() {
            return Err(ThermoelasticLoadError::InvalidConstitutive(element.element_id).into());
        }
        projection
            .visit_element_quadrature(element, |point| {
                let temperature_k = element
                    .node_indices
                    .iter()
                    .zip(point.shape_values())
                    .map(|(&node_index, &shape)| shape * temperature.at(element_index, node_index))
                    .sum::<f64>();
                let stress_pa = coefficient
                    * (temperature_k - thermal.reference_temperature_k.canonical_value());
                for (&node_index, gradient) in element
                    .node_indices
                    .iter()
                    .zip(point.physical_shape_gradients())
                {
                    for axis in 0..3 {
                        load[node_index * 3 + axis] +=
                            stress_pa * gradient[axis] * point.volume_weight_m3;
                    }
                }
            })
            .map_err(projection_error)?;
    }
    if load.iter().any(|value| !value.is_finite()) {
        return Err(ThermoelasticLoadError::InvalidNodalField.into());
    }
    Ok(load)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::{
            assemble_linear_system_impl, solver_solid::assemble_solver_solid_operators,
            AssemblyExecution,
        },
        excitation::RealizedExcitations,
        fixtures::{fixture_model, FixtureId},
        properties::compile_structural_property_map,
    };
    use runmat_meshing_core::{
        fixtures::{canonical_tetrahedron_solver_mesh, three_region_tetrahedron_solver_mesh},
        ElementOrder,
    };

    fn context() -> FeaThermoMechanicalContext {
        FeaThermoMechanicalContext {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(10.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(0.0),
            region_temperature_deltas: Vec::new(),
        }
    }

    #[test]
    fn awaited_fixed_thermoelastic_rules_match_native_and_keep_denial_typed() {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let mut model = fixture_model(FixtureId::CantileverLinearStatic);
            model.materials[0]
                .thermal
                .as_mut()
                .unwrap()
                .expansion_coefficient_per_k =
                runmat_types::physical::ThermalExpansion::new(1.0e-5);
            let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
            let native =
                assemble_prescribed_temperature_load(&mesh, &properties, &context()).unwrap();
            let quotes = std::cell::RefCell::new(Vec::new());
            let mut grant = |element, count| {
                quotes.borrow_mut().push((element, count));
                std::future::ready(Ok::<(), &'static str>(()))
            };
            let awaited =
                futures::executor::block_on(assemble_prescribed_temperature_load_with_owner(
                    &mesh,
                    &properties,
                    &context(),
                    &mut grant,
                ))
                .unwrap();
            assert_eq!(awaited, native);
            let quotes = quotes.borrow();
            assert_eq!(quotes.len(), 1 + mesh.topology.volume_elements.len());
            assert_eq!(
                quotes[0],
                (
                    ThermoelasticCapacityPhase::Projection,
                    AnalysisProjection::derivation_work_upper_bound(&mesh).unwrap(),
                )
            );
            assert!(quotes[1..].iter().enumerate().all(|(element, quote)| {
                *quote
                    == (
                        ThermoelasticCapacityPhase::Element { element },
                        quadrature_work(order, true),
                    )
            }));
            let mut deny = |_, _| std::future::ready(Err::<(), _>("denied"));
            let denied =
                futures::executor::block_on(assemble_prescribed_temperature_load_with_owner(
                    &mesh,
                    &properties,
                    &context(),
                    &mut deny,
                ));
            assert!(matches!(
                denied,
                Err(ThermoelasticLoadCapacityError::Admission("denied"))
            ));
        }
    }

    #[test]
    fn uniform_tet4_thermal_load_matches_closed_form_nodal_forces() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(1.0e-5);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let load = assemble_prescribed_temperature_load(&mesh, &properties, &context()).unwrap();
        let mechanical = model.materials[0].require_mechanical().unwrap();
        let stress = mechanical.youngs_modulus_pa.canonical_value() * 1.0e-4
            / (1.0 - 2.0 * mechanical.poisson_ratio.canonical_value());
        let expected = [
            -stress / 6.0,
            -stress / 6.0,
            -stress / 6.0,
            stress / 6.0,
            0.0,
            0.0,
            0.0,
            stress / 6.0,
            0.0,
            0.0,
            0.0,
            stress / 6.0,
        ];
        for (actual, expected) in load.iter().zip(expected) {
            assert!((actual - expected).abs() <= stress.abs() * 1.0e-12);
        }
        for axis in 0..3 {
            let resultant = load.chunks_exact(3).map(|force| force[axis]).sum::<f64>();
            assert!(resultant.abs() <= stress.abs() * 1.0e-12);
        }
    }

    #[test]
    fn solved_linear_tet4_temperature_integrates_closed_form_thermal_force() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(1.0e-5);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let temperature = mesh
            .topology
            .nodes
            .iter()
            .map(|node| {
                293.15
                    + 10.0 * node.coordinates_m.canonical_value()[0]
                    + 20.0 * node.coordinates_m.canonical_value()[1]
                    - 5.0 * node.coordinates_m.canonical_value()[2]
            })
            .collect::<Vec<_>>();
        let force = assemble_solved_temperature_load(
            &mesh,
            &properties,
            &crate::execution::SolvedTemperatureField::nodal(temperature.clone()),
        )
        .unwrap();
        let mechanical = model.materials[0].require_mechanical().unwrap();
        let average_delta = 0.25 * (10.0 + 20.0 - 5.0);
        let hydrostatic_stress =
            mechanical.youngs_modulus_pa.canonical_value() * 1.0e-5 * average_delta
                / (1.0 - 2.0 * mechanical.poisson_ratio.canonical_value());
        let expected = [
            -hydrostatic_stress / 6.0,
            -hydrostatic_stress / 6.0,
            -hydrostatic_stress / 6.0,
            hydrostatic_stress / 6.0,
            0.0,
            0.0,
            0.0,
            hydrostatic_stress / 6.0,
            0.0,
            0.0,
            0.0,
            hydrostatic_stress / 6.0,
        ];
        for (actual, reference) in force.iter().zip(expected) {
            assert!((actual - reference).abs() <= hydrostatic_stress.abs() * 1.0e-12);
        }
        let mut invalid = temperature;
        invalid[0] = -1.0;
        assert!(matches!(
            assemble_solved_temperature_load(
                &mesh,
                &properties,
                &crate::execution::SolvedTemperatureField::nodal(invalid),
            ),
            Err(ThermoelasticLoadError::InvalidNodalField)
        ));
    }

    #[test]
    fn uniform_tet10_load_has_zero_resultant_and_rejects_unknown_region_field() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let load = assemble_prescribed_temperature_load(&mesh, &properties, &context()).unwrap();
        assert_eq!(load.len(), 30);
        assert!(load.iter().any(|force| force.abs() > 0.0));
        let scale = load.iter().map(|force| force.abs()).sum::<f64>();
        for axis in 0..3 {
            let resultant = load.chunks_exact(3).map(|force| force[axis]).sum::<f64>();
            assert!(resultant.abs() <= scale * 1.0e-12);
        }
        let mut unknown = context();
        unknown
            .region_temperature_deltas
            .push(crate::FeaThermoRegionTemperatureDelta {
                target: runmat_analysis_core::RegionTarget::volume(
                    runmat_meshing_core::fixtures::fixture_region("missing_region"),
                ),
                temperature_delta_k: 10.0,
            });
        assert!(matches!(
            assemble_prescribed_temperature_load(&mesh, &properties, &unknown),
            Err(ThermoelasticLoadError::UnknownRegion(_))
        ));
    }

    #[test]
    fn tet4_and_tet10_free_expansion_satisfy_independent_mechanical_equilibrium() {
        // For a homogeneous traction-free solid, u = alpha * delta_T * x gives
        // epsilon(u) = epsilon_thermal at every integration point, hence K u = f_thermal.
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let mut model = fixture_model(FixtureId::CantileverLinearStatic);
            model.materials[0]
                .thermal
                .as_mut()
                .unwrap()
                .expansion_coefficient_per_k =
                runmat_types::physical::ThermalExpansion::new(1.0e-5);
            let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
            let projected = AnalysisProjection::derive(&mesh).unwrap();
            let coefficient = model.materials[0]
                .require_thermal()
                .unwrap()
                .expansion_coefficient_per_k
                .canonical_value()
                * context().applied_temperature_delta_k.canonical_value();
            let displacement = projected
                .nodes
                .iter()
                .flat_map(|node| {
                    node.coordinates_m
                        .canonical_value()
                        .map(|coordinate| coefficient * coordinate)
                })
                .collect::<Vec<_>>();
            let thermal_load =
                assemble_prescribed_temperature_load(&mesh, &properties, &context()).unwrap();
            let stiffness = assemble_solver_solid_operators(&mesh, &properties)
                .unwrap()
                .stiffness;
            let dense = stiffness.to_dense();
            let n = stiffness.rows();
            assert_eq!(n, displacement.len());
            let scale = thermal_load
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            assert!(scale > 0.0);
            for row in 0..n {
                let internal_force = dense[row * n..(row + 1) * n]
                    .iter()
                    .zip(&displacement)
                    .map(|(stiffness, value)| stiffness * value)
                    .sum::<f64>();
                assert!(
                    (internal_force - thermal_load[row]).abs() <= scale * 1.0e-10,
                    "{order:?} row {row}: internal={internal_force}, thermal={}",
                    thermal_load[row]
                );
            }
        }
    }

    #[test]
    fn heterogeneous_regions_use_their_exact_constitutive_owner_independent_of_authored_order() {
        let mesh = three_region_tetrahedron_solver_mesh();
        let mut model = fixture_model(FixtureId::MultiMaterialAssembly);
        model
            .material_assignments
            .iter_mut()
            .find(|assignment| {
                assignment.selected_region_id()
                    == Some(runmat_meshing_core::fixtures::fixture_region("mid_aluminum").as_str())
            })
            .unwrap()
            .assigned_material_id = "mat_aluminum".into();
        for (material, coefficient) in model.materials.iter_mut().zip([1.0e-5, 2.0e-5, 4.0e-5]) {
            material
                .thermal
                .as_mut()
                .unwrap()
                .expansion_coefficient_per_k =
                runmat_types::physical::ThermalExpansion::new(coefficient);
        }
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let mut prescribed = context();
        prescribed
            .region_temperature_deltas
            .push(crate::FeaThermoRegionTemperatureDelta {
                target: runmat_analysis_core::RegionTarget::volume(
                    runmat_meshing_core::fixtures::fixture_region("mid_aluminum"),
                ),
                temperature_delta_k: 30.0,
            });
        let load = assemble_prescribed_temperature_load(&mesh, &properties, &prescribed).unwrap();
        let recovery =
            crate::assembly::solid_recovery::build(&mesh, &properties, Some(&prescribed), None)
                .unwrap();
        assert!(crate::assembly::solid_recovery::build(
            &mesh,
            &properties,
            Some(&prescribed),
            Some(&crate::execution::SolvedTemperatureField::nodal(vec![
                300.0;
                mesh.topology.nodes.len()
            ])),
        )
        .is_err());
        for element in &recovery {
            let expected = if element.region_id == "mid_aluminum" {
                323.15
            } else {
                303.15
            };
            assert_eq!(
                element
                    .prescribed_temperature_k
                    .map(|value| value.canonical_value()),
                Some(expected)
            );
        }
        let projected = AnalysisProjection::derive(&mesh).unwrap();
        for element in &projected.elements {
            let region = element.region_id.source_topology_id.as_str();
            let (youngs, poisson, expansion) = match region {
                "tip_steel" => (200.0e9, 0.3, 1.0e-5),
                "mid_aluminum" => (69.0e9, 0.33, 2.0e-5),
                "polymer_segment" => (3.2e9, 0.37, 4.0e-5),
                _ => panic!("unexpected independent material region {region}"),
            };
            let delta = if region == "mid_aluminum" { 30.0 } else { 10.0 };
            let stress = youngs * expansion * delta / (1.0 - 2.0 * poisson);
            let expected = [
                -stress / 6.0,
                -stress / 6.0,
                -stress / 6.0,
                stress / 6.0,
                0.0,
                0.0,
                0.0,
                stress / 6.0,
                0.0,
                0.0,
                0.0,
                stress / 6.0,
            ];
            for (&node, expected_force) in element.node_indices.iter().zip(expected.chunks_exact(3))
            {
                for axis in 0..3 {
                    assert!(
                        (load[node * 3 + axis] - expected_force[axis]).abs()
                            <= stress.abs() * 1.0e-12,
                        "region {region}, node {node}, axis {axis}"
                    );
                }
            }
        }
        model.materials.reverse();
        model.material_assignments.reverse();
        let reordered = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        assert_eq!(
            assemble_prescribed_temperature_load(&mesh, &reordered, &prescribed).unwrap(),
            load
        );
        prescribed
            .region_temperature_deltas
            .push(crate::FeaThermoRegionTemperatureDelta {
                target: runmat_analysis_core::RegionTarget::volume(
                    runmat_meshing_core::fixtures::fixture_region("mid_aluminum"),
                ),
                temperature_delta_k: 40.0,
            });
        assert!(matches!(
            assemble_prescribed_temperature_load(&mesh, &reordered, &prescribed),
            Err(ThermoelasticLoadError::DuplicateRegion(_))
        ));
        prescribed.region_temperature_deltas.pop();
        prescribed.region_temperature_deltas[0].temperature_delta_k = -400.0;
        assert!(matches!(
            assemble_prescribed_temperature_load(&mesh, &reordered, &prescribed),
            Err(ThermoelasticLoadError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn physical_structural_assembly_uses_element_thermal_load_not_proxy_pattern() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let context = context();
        let expected = assemble_prescribed_temperature_load(&mesh, &properties, &context).unwrap();
        let summary = assemble_linear_system_impl(
            &model,
            &RealizedExcitations::default(),
            Some(&mesh),
            Some(context),
            None,
            AssemblyExecution::serial_physical(),
        )
        .expect("physical thermoelastic assembly");
        assert_eq!(
            &summary.physical_load[..expected.len()],
            expected.as_slice()
        );
        assert!(summary.physical_load.iter().any(|value| value.abs() > 0.0));
        assert!(summary.constrained_dof_count > 0);
        assert!(summary
            .operator
            .constrained_dofs()
            .iter()
            .enumerate()
            .any(|(row, constrained)| *constrained && summary.physical_load[row] != 0.0));
        for (row, is_constrained) in summary.operator.constrained_dofs().iter().enumerate() {
            if *is_constrained {
                assert_eq!(summary.operator.load()[row], 0.0);
            }
        }
    }
}
