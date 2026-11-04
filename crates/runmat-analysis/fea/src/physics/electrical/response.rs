use super::{assembly::ElectricalAssembly, recovery, ElectricalStaticError};
use crate::algebra::{
    spmv, AssembledRealSystem, CanonicalRealOperators, MatrixProperties,
    NullspaceEliminationEvidence, OperatorRole, RealSparseMatrix,
};
use crate::solve::backend::linear_algebra::LinearAlgebraBackend;
use runmat_analysis_core::StaticElectricalEquation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ElectricalConstitutiveResponse {
    Electrostatics {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        displacement_c_per_m2: runmat_types::physical::Quantity<
            runmat_types::physical::kind::ElectricDisplacement,
            Vec<[f64; 3]>,
        >,
        #[serde(with = "runmat_types::physical::canonical_storage")]
        energy_density_j_per_m3:
            runmat_types::physical::Quantity<runmat_types::physical::kind::EnergyDensity, Vec<f64>>,
        #[serde(with = "runmat_types::physical::canonical_storage")]
        total_energy_j: runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
    },
    SteadyConduction {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        current_density_a_per_m2: runmat_types::physical::Quantity<
            runmat_types::physical::kind::ElectricCurrentDensity,
            Vec<[f64; 3]>,
        >,
        #[serde(with = "runmat_types::physical::canonical_storage")]
        joule_density_w_per_m3: runmat_types::physical::Quantity<
            runmat_types::physical::kind::HeatGenerationDensity,
            Vec<f64>,
        >,
        #[serde(with = "runmat_types::physical::canonical_storage")]
        total_joule_power_w:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Power, f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalTerminalResponse {
    pub constraint_id: String,
    pub ordered_node_ids: Vec<u64>,
    /// Outward dielectric flux (C) or conduction current (A), from the original weak
    /// operator, not a field-average extrapolation. Electrode charge has opposite sign.
    pub outward_flux:
        runmat_types::physical::Quantity<runmat_types::physical::DynamicDimension, f64>,
}

/// What certified the electrical operator before its solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElectricalOperatorProof {
    /// The anchored local-energy bound separated the assembled operator from singularity.
    LocalEnergy,
    /// The local-energy bound was indeterminate; the sparse Cholesky factorization decided.
    SparseCholesky,
    /// Periodic relations reduced the system, which the shared SPD solver factored or iterated.
    ReducedRelations,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalElectricalResponse {
    pub equation: StaticElectricalEquation,
    pub mesh_digest: runmat_meshing_core::StableDigest,
    pub ordered_node_ids: Vec<u64>,
    pub ordered_element_ids: Vec<u64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub potential_v:
        runmat_types::physical::Quantity<runmat_types::physical::kind::ElectricPotential, Vec<f64>>,
    /// Component-wise gauge departure retained before adding absolute potentials.
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub potential_increment_v:
        runmat_types::physical::Quantity<runmat_types::physical::kind::ElectricPotential, Vec<f64>>,
    /// Volume averages. Quadratic products below are integrated BEFORE averaging.
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub element_electric_field_v_per_m: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ElectricField,
        Vec<[f64; 3]>,
    >,
    pub constitutive: ElectricalConstitutiveResponse,
    pub terminals: Vec<ElectricalTerminalResponse>,
    pub nodal_outward_flux:
        runmat_types::physical::Quantity<runmat_types::physical::DynamicDimension, Vec<f64>>,
    pub charge_sources: Option<super::ElectricalChargeSources>,
    /// Gauge-departure work (J electrostatic, W conduction), not stored energy/2.
    pub volume_source_work:
        runmat_types::physical::Quantity<runmat_types::physical::DynamicDimension, f64>,
    pub terminal_work:
        runmat_types::physical::Quantity<runmat_types::physical::DynamicDimension, f64>,
    pub relative_flux_imbalance: f64,
    pub relative_energy_identity_error: f64,
    /// Absolute L2 residual of the dimensionless Dirichlet-lifted system.
    pub linear_absolute_residual: f64,
    /// Absolute residual divided by the same lifted RHS norm used by shared PCG.
    pub linear_relative_residual: f64,
    pub iterations: u32,
    /// What certified the operator before the solve, retained as evidence of the run.
    pub operator_proof: ElectricalOperatorProof,
}

pub(super) fn solve(
    mut assembly: ElectricalAssembly,
    equation: StaticElectricalEquation,
    residual_target: f64,
    planned: Option<super::PlannedElectricalSolve<'_>>,
) -> Result<CanonicalElectricalResponse, ElectricalStaticError> {
    let count = assembly.projection.nodes.len();
    assembly
        .coefficients
        .require_kind(equation.constitutive_coefficient_kind())
        .map_err(|error| ElectricalStaticError::Material(error.to_string()))?;
    let constrained = assembly
        .prescribed
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    // Material-coefficient normalization prevents identity Dirichlet rows in volts
    // from overwhelming free electrostatic rows (epsilon is ~1e-11 SI).
    // Free rows retain Potential×Length units; prescribed rows retain Potential.
    // This is a mixed-row numerical system, not a dimensionless physical operator.
    // The original physical operator is retained for flux and energy accounting.
    let scale = assembly
        .coefficients
        .canonical_values()
        .iter()
        .copied()
        .reduce(f64::max)
        .ok_or_else(|| ElectricalStaticError::Material("empty electrical domain".into()))?;
    let normalized = RealSparseMatrix::scaled_sum(
        &[(scale.recip(), &assembly.stiffness)],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
    let proof = NullspaceEliminationEvidence::prove_scalar_diffusion(
        &assembly.projection,
        &normalized,
        &constrained,
        &vec![0.0; count],
    )
    .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
    // Solve component-wise increments so arbitrary potential gauge offsets do not
    // create roundoff fields, spurious terminal flux or a false relative energy error.
    let components = crate::algebra::connected_components(&assembly.projection.node_neighbors)
        .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
    let mut reference = vec![0.0; count];
    for nodes in components {
        let value = nodes
            .iter()
            .find_map(|&node| assembly.prescribed[node])
            .ok_or_else(|| ElectricalStaticError::Boundary("unanchored component".into()))?;
        for node in nodes {
            reference[node] = value;
        }
    }
    let zero = |role| {
        RealSparseMatrix::diagonal(
            vec![0.0; count],
            role,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|error| ElectricalStaticError::Operator(error.to_string()))
    };
    let reduced_operator = (!assembly.relations.is_empty()).then(|| normalized.clone());
    let operators = CanonicalRealOperators::try_new(
        normalized,
        zero(OperatorRole::Mass)?,
        zero(OperatorRole::Damping)?,
    )
    .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
    let rhs: Vec<f64> = assembly
        .prescribed
        .iter()
        .zip(&reference)
        .enumerate()
        .map(|(node, (value, reference))| {
            value.map_or_else(
                || {
                    assembly
                        .charge_sources
                        .as_ref()
                        .map_or(0.0, |source| source.nodal_source[node] / scale)
                },
                |value| value - reference,
            )
        })
        .collect();
    let physical_flux =
        runmat_types::physical::DynamicDimension::quantity_type(equation.conjugate_flux_kind())
            .with_semantic_kind(Some(equation.conjugate_flux_kind()))
            .expect("electrical conjugate flux");
    let (increment, linear_absolute_residual, relative_residual, iterations, operator_proof) =
        if let Some(operator) = reduced_operator {
            // Periodic relations: eliminate the secondary nodes and solve the reduced system.
            // Relations hold absolute potentials; in increments they gain the reference shift.
            let relations = crate::physics::scalar_constraints::about_reference(
                &assembly.relations,
                &reference,
            );
            let known: Vec<Option<f64>> = assembly
                .prescribed
                .iter()
                .zip(&reference)
                .map(|(value, reference)| value.map(|value| value - reference))
                .collect();
            let strategy = planned.map_or(
                crate::solve::preconditioner::SpdPreconditionerKind::Auto,
                |planned| planned.strategy(),
            );
            let reduced_solve = |tolerance| {
                crate::physics::scalar_constraints::solve(
                    &operator, &rhs, &known, &relations, tolerance, strategy,
                )
                .map_err(|error| match error {
                    crate::physics::scalar_constraints::ScalarSolveError::Usage(error) => {
                        ElectricalStaticError::Accounting(error)
                    }
                    crate::physics::scalar_constraints::ScalarSolveError::Detail(detail) => {
                        ElectricalStaticError::Operator(detail)
                    }
                })
            };
            let mut reduced = reduced_solve(residual_target)?;
            let mut iterations = reduced.iterations;
            let (mut tolerance, mut refinements) = (residual_target, 0);
            while let Some(next) = conservation_tolerance(
                &assembly,
                &reduced.solution,
                physical_flux,
                residual_target,
                tolerance,
                refinements,
            )? {
                tolerance = next;
                refinements += 1;
                reduced = reduced_solve(tolerance)?;
                iterations += reduced.iterations;
            }
            let (solution, relative_residual) = (reduced.solution, reduced.relative_residual);
            if !relative_residual.is_finite() || relative_residual > residual_target {
                return Err(ElectricalStaticError::Operator(
                    "electrical relative residual target not satisfied".into(),
                ));
            }
            drop(constrained);
            drop(proof);
            drop(operators);
            let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
            (
                solution,
                relative_residual * rhs_norm,
                relative_residual,
                iterations,
                ElectricalOperatorProof::ReducedRelations,
            )
        } else {
            let system = AssembledRealSystem::try_new_with_nullspace_evidence(
                operators,
                constrained.clone(),
                rhs,
                Some(proof),
            )
            .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
            let normalized_equation = physical_flux
                .divide(assembly.coefficients.descriptor())
                .ok_or_else(|| {
                    ElectricalStaticError::Operator(
                        "invalid electrical normalization descriptor".into(),
                    )
                })?;
            let system = crate::algebra::PhysicalScalarSystem::<
                runmat_types::physical::kind::ElectricPotential,
            >::try_new(system, normalized_equation)
            .map_err(|error| ElectricalStaticError::Operator(error.into()))?;
            system
                .require_equation(normalized_equation)
                .map_err(|error| ElectricalStaticError::Operator(error.into()))?;
            system
                .physical_operator()
                .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
            let backend = crate::solve::backend::cpu_reference::CpuReferenceBackend;
            let lifted_rhs = system.linear_rhs();
            let rhs_norm = backend.dot(&lifted_rhs, &lifted_rhs).sqrt();
            drop(lifted_rhs);
            let mut preparation = crate::algebra::LinearPreparationReuse::with_local_energy(
                assembly.local_energy.take().ok_or_else(|| {
                    ElectricalStaticError::Operator("missing original local-energy source".into())
                })?,
                constrained,
                scale.recip(),
            );
            let strategy = planned.map_or(
                crate::solve::preconditioner::SpdPreconditionerKind::Auto,
                |planned| planned.strategy(),
            );
            let mut linear_solve = |guess: Option<&[f64]>, tolerance: f64| {
                let solution = system
                    .with_canonical_system(|numerical| {
                        crate::solve::linear::solve_operator_system_scoped(
                            numerical,
                            strategy,
                            crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                            &backend,
                            guess,
                            crate::solve::linear::ScopedLinearSolveSettings {
                                plan: crate::planning::resource::LocalExecutionPlan::serial(),
                                admission: planned.map(|planned| planned.admission),
                                relative_tolerance: tolerance,
                                preparation: Some(&mut preparation),
                            },
                        )
                    })
                    .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
                let solution = system
                    .admit_solution(solution)
                    .map_err(|error| ElectricalStaticError::Operator(error.into()))?;
                if !solution.converged
                    || solution
                        .solution
                        .canonical_values()
                        .iter()
                        .any(|value| !value.is_finite())
                {
                    return Err(ElectricalStaticError::NotConverged {
                        iterations: solution.iterations,
                        residual: solution.residual_norm,
                        diagnostics: solution.diagnostics,
                    });
                }
                Ok(solution)
            };
            let mut solution = linear_solve(None, residual_target)?;
            let mut iterations = solution.iterations;
            let (mut tolerance, mut refinements) = (residual_target, 0);
            while let Some(next) = conservation_tolerance(
                &assembly,
                solution.solution.canonical_values(),
                physical_flux,
                residual_target,
                tolerance,
                refinements,
            )? {
                tolerance = next;
                refinements += 1;
                solution = linear_solve(Some(solution.solution.canonical_values()), tolerance)?;
                iterations += solution.iterations;
            }
            let relative_residual = if rhs_norm > 0.0 {
                solution.residual_norm / rhs_norm
            } else if solution.residual_norm == 0.0 {
                0.0
            } else {
                return Err(ElectricalStaticError::Operator(
                    "nonzero residual for zero electrical RHS".into(),
                ));
            };
            if !relative_residual.is_finite() || relative_residual > residual_target {
                return Err(ElectricalStaticError::Operator(
                    "electrical relative residual target not satisfied".into(),
                ));
            }
            let operator_proof = if preparation
                .certificate()
                .is_some_and(|certificate| certificate.proved_by_factorization())
            {
                ElectricalOperatorProof::SparseCholesky
            } else {
                ElectricalOperatorProof::LocalEnergy
            };
            (
                solution.solution.into_canonical_values(),
                solution.residual_norm,
                relative_residual,
                iterations,
                operator_proof,
            )
        };
    let potential = increment
        .iter()
        .zip(&reference)
        .map(|(value, reference)| value + reference)
        .collect();
    let TerminalBalance {
        reactions,
        nodal_source,
        nodal_outward_flux,
        terminals,
        relative_flux_imbalance,
    } = terminal_balance(&assembly, &increment, physical_flux)?;
    if let (Some(sources), Some(folded)) = (assembly.charge_sources.as_mut(), nodal_source) {
        sources.nodal_source = folded;
    }
    let recovery = recovery::recover(&assembly, &increment)?;
    let integrals = recovery
        .squared_field_integral
        .iter()
        .zip(assembly.coefficients.canonical_values())
        .map(|(square, coefficient)| square * coefficient)
        .collect::<Vec<_>>();
    let total = integrals.iter().sum::<f64>();
    let volume_source_work = assembly.charge_sources.as_ref().map_or(0.0, |source| {
        increment
            .iter()
            .zip(&source.nodal_source)
            .map(|(value, charge)| value * charge)
            .sum::<f64>()
    });
    let terminal_work = assembly
        .prescribed
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_some())
        .map(|(node, _)| {
            let charge = assembly
                .charge_sources
                .as_ref()
                .map_or(0.0, |source| source.nodal_source[node]);
            increment[node] * (reactions[node] - charge)
        })
        .sum::<f64>();
    let weak_total = volume_source_work + terminal_work;
    let relative_energy_identity_error =
        (total - weak_total).abs() / total.abs().max(weak_total.abs()).max(f64::MIN_POSITIVE);
    let constitutive = match equation {
        StaticElectricalEquation::Electrostatics => {
            ElectricalConstitutiveResponse::Electrostatics {
                displacement_c_per_m2: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::ElectricDisplacement,
                    _,
                >::new(recovery.flux),
                energy_density_j_per_m3: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::EnergyDensity,
                    _,
                >::new(
                    integrals
                        .iter()
                        .zip(&recovery.volumes)
                        .map(|(value, volume)| 0.5 * value / volume)
                        .collect(),
                ),
                total_energy_j: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Energy,
                    _,
                >::new(0.5 * total),
            }
        }
        StaticElectricalEquation::SteadyConduction => {
            ElectricalConstitutiveResponse::SteadyConduction {
                current_density_a_per_m2: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::ElectricCurrentDensity,
                    _,
                >::new(recovery.flux),
                joule_density_w_per_m3: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::HeatGenerationDensity,
                    _,
                >::new(
                    integrals
                        .iter()
                        .zip(&recovery.volumes)
                        .map(|(value, volume)| value / volume)
                        .collect(),
                ),
                total_joule_power_w: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Power,
                    _,
                >::new(total),
            }
        }
    };
    if !total.is_finite()
        || !relative_energy_identity_error.is_finite()
        || !relative_flux_imbalance.is_finite()
        || reactions.iter().any(|value| !value.is_finite())
        || integrals.iter().any(|value| !value.is_finite())
    {
        return Err(ElectricalStaticError::Operator(
            "nonfinite conservation accounting".into(),
        ));
    }
    if relative_energy_identity_error > closure_target(residual_target)
        || relative_flux_imbalance > closure_target(residual_target)
    {
        return Err(ElectricalStaticError::Operator(format!(
            "electrical conservation target not satisfied (energy identity {relative_energy_identity_error:e}, flux imbalance {relative_flux_imbalance:e})"
        )));
    }
    Ok(CanonicalElectricalResponse {
        operator_proof,
        equation,
        mesh_digest: assembly.projection.mesh_digest,
        ordered_node_ids: assembly
            .projection
            .nodes
            .iter()
            .map(|node| node.node_id)
            .collect(),
        ordered_element_ids: assembly
            .projection
            .elements
            .iter()
            .map(|element| element.element_id)
            .collect(),
        potential_v: runmat_types::physical::Quantity::try_new(potential)
            .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?,
        potential_increment_v: runmat_types::physical::Quantity::try_new(increment)
            .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?,
        element_electric_field_v_per_m: runmat_types::physical::Quantity::try_new(
            recovery.electric_field,
        )
        .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?,
        constitutive,
        terminals,
        nodal_outward_flux: runmat_types::physical::Quantity::try_from_descriptor(
            physical_flux,
            nodal_outward_flux,
        )
        .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?,
        charge_sources: assembly.charge_sources,
        volume_source_work: runmat_types::physical::Quantity::try_from_dimension(
            equation.work_kind().dimension(),
            Some(equation.work_kind()),
            volume_source_work,
        )
        .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?,
        terminal_work: runmat_types::physical::Quantity::try_from_dimension(
            equation.work_kind().dimension(),
            Some(equation.work_kind()),
            terminal_work,
        )
        .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?,
        relative_flux_imbalance,
        relative_energy_identity_error,
        linear_absolute_residual,
        linear_relative_residual: relative_residual,
        iterations,
    })
}

/// Closure tolerance for the energy identity and terminal flux balance (the receipt's
/// `closure_relative_tolerance`).
pub(crate) fn closure_target(residual_target: f64) -> f64 {
    residual_target.max(1e-10) * 10.0
}

/// Linear refinements allowed to close the terminal flux balance.
const CONSERVATION_REFINEMENTS: u32 = 4;

/// The terminal imbalance equals the sum of the free-node algebraic residual, which a
/// relative 2-norm residual bounds only up to the square root of the free-node count. When
/// the balance misses the closure target, return a tighter linear tolerance in proportion to
/// the miss so the solver, not the gate, closes the balance. Returns `None` once it closes,
/// or when the refinements or attainable precision run out (the gate then reports the miss).
fn conservation_tolerance(
    assembly: &ElectricalAssembly,
    increment: &[f64],
    physical_flux: runmat_types::physical::DynamicDimension,
    residual_target: f64,
    tolerance: f64,
    refinements: u32,
) -> Result<Option<f64>, ElectricalStaticError> {
    let imbalance = terminal_balance(assembly, increment, physical_flux)?.relative_flux_imbalance;
    let target = closure_target(residual_target);
    let floor = 64.0 * f64::EPSILON;
    if !imbalance.is_finite()
        || imbalance <= target
        || tolerance <= floor
        || refinements >= CONSERVATION_REFINEMENTS
    {
        return Ok(None);
    }
    Ok(Some((tolerance * 0.25 * target / imbalance).max(floor)))
}

/// Terminal flux accounting of a potential increment, with periodic secondaries folded onto
/// their primaries: the flux leaving one face of a periodic pair enters through the other.
struct TerminalBalance {
    reactions: Vec<f64>,
    nodal_source: Option<Vec<f64>>,
    nodal_outward_flux: Vec<f64>,
    terminals: Vec<ElectricalTerminalResponse>,
    relative_flux_imbalance: f64,
}

fn terminal_balance(
    assembly: &ElectricalAssembly,
    increment: &[f64],
    physical_flux: runmat_types::physical::DynamicDimension,
) -> Result<TerminalBalance, ElectricalStaticError> {
    let mut reactions = spmv(&assembly.stiffness, increment)
        .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
    let mut nodal_source = assembly
        .charge_sources
        .as_ref()
        .map(|source| source.nodal_source.clone());
    for relation in &assembly.relations {
        let reaction = std::mem::take(&mut reactions[relation.secondary]);
        let source = nodal_source.as_mut().map_or(0.0, |source| {
            std::mem::take(&mut source[relation.secondary])
        });
        for &(primary, weight) in &relation.primaries {
            reactions[primary] += weight * reaction;
            if let Some(sources) = nodal_source.as_mut() {
                sources[primary] += weight * source;
            }
        }
    }
    let nodal_outward_flux = reactions
        .iter()
        .enumerate()
        .map(|(node, value)| nodal_source.as_ref().map_or(0.0, |source| source[node]) - value)
        .collect::<Vec<_>>();
    let terminals = assembly
        .terminal_nodes
        .iter()
        .map(|(id, nodes)| {
            Ok::<_, ElectricalStaticError>(ElectricalTerminalResponse {
                constraint_id: id.clone(),
                ordered_node_ids: nodes
                    .iter()
                    .map(|&node| assembly.projection.nodes[node].node_id)
                    .collect(),
                outward_flux: runmat_types::physical::Quantity::try_from_descriptor(
                    physical_flux,
                    nodes.iter().map(|&node| nodal_outward_flux[node]).sum(),
                )
                .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let flux_scale = terminals
        .iter()
        .map(|terminal| terminal.outward_flux.canonical_value().abs())
        .sum::<f64>();
    let integrated_charge = assembly
        .charge_sources
        .as_ref()
        .map_or(0.0, |source| source.integrated_source);
    let source_scale = assembly
        .charge_sources
        .as_ref()
        .map_or(0.0, |source| source.absolute_source);
    let relative_flux_imbalance = (terminals
        .iter()
        .map(|terminal| terminal.outward_flux.canonical_value())
        .sum::<f64>()
        - integrated_charge)
        .abs()
        / flux_scale.max(source_scale).max(f64::MIN_POSITIVE);
    Ok(TerminalBalance {
        reactions,
        nodal_source,
        nodal_outward_flux,
        terminals,
        relative_flux_imbalance,
    })
}
