//! One authority for the discrete harmonic operator: assembly, direct solve, field recovery
//! and cold re-evaluation of a retained solution all go through `MaxwellDiscretization`.
//!
//! Unknowns: complex circulations of E on free (ungrounded) Nédélec edges, plus one terminal
//! voltage U per current-driven conductor. Equations, in SI with exp(+i omega t):
//!   (nu curl E, curl w) + ((i omega sigma - omega^2 eps) E, w)
//!       = -i omega (J_vol + sum_k U_k (sigma g_k + j_k), w)            for every free edge w
//!   (sigma g_k, E) + U_k (sigma g_k, g_k) = I_k                         for every current drive
//! The current row is scaled by -i omega so the augmented block is near complex-symmetric.
use super::{
    complex::{dot_conjugate, vector_norm_squared},
    continuity::ChargeContinuity,
    geometry, invalid, topology, ComplexPhasor, ComplexVector3, MaxwellDriveExcitation,
    MaxwellError, MaxwellFieldSample, MaxwellInput, MaxwellResponse, MaxwellSystemPlan,
    MaxwellVolumeData,
};
use crate::solve::sparse_direct::{self, ComplexEntry};
use crate::topology::{AnalysisProjection, TetrahedralEdgeSpace};
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use runmat_types::physical::Quantity;
use std::collections::{BTreeMap, BTreeSet};

/// Terminal response of one driven conductor (peak phasors).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellDriveResponse {
    pub drive_id: String,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub voltage_v: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ElectricPotential,
        ComplexPhasor,
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub current_a: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ElectricCurrent,
        ComplexPhasor,
    >,
    /// (1/2) U conj(I): real part is the power the drive delivers.
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub complex_power_va: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ApparentPower,
        ComplexPhasor,
    >,
}

pub(super) struct MaxwellDiscretization<'a> {
    pub projection: AnalysisProjection,
    pub space: TetrahedralEdgeSpace,
    pub input: &'a MaxwellInput,
    pub volumes: Vec<MaxwellVolumeData>,
    pub grounded_nodes: BTreeSet<usize>,
    /// edge index -> row for free edges
    rows: BTreeMap<usize, usize>,
    free: usize,
    /// per drive: element index -> unit generator field g (V/m per V)
    generator: Vec<BTreeMap<usize, [f64; 3]>>,
    /// per drive: element index -> unit closure current (A/m^2 per V)
    closure: Vec<BTreeMap<usize, [f64; 3]>>,
    /// drive index -> augmented column for current drives
    unknown_voltage: BTreeMap<usize, usize>,
}

impl<'a> MaxwellDiscretization<'a> {
    pub fn new(
        mesh: &SolverMeshArtifact,
        input: &'a MaxwellInput,
        plan: &MaxwellSystemPlan,
    ) -> Result<Self, MaxwellError> {
        input.validate_for_mesh(mesh)?;
        if plan.grounded_faces() != input.grounded_faces.as_slice() {
            return Err(invalid(
                "Maxwell plan grounds different faces than its input",
            ));
        }
        plan.validate_for(mesh)?;
        let volumes = input.ordered_volumes(mesh)?;
        let projection = AnalysisProjection::derive(mesh).map_err(topology)?;
        let space = geometry::spaces(&projection)?;
        let rows: BTreeMap<_, _> = plan
            .free_edge_indices()
            .iter()
            .enumerate()
            .map(|(row, &edge)| (edge, row))
            .collect();
        let free = rows.len();
        let element_index: BTreeMap<StableDigest, usize> = projection
            .elements
            .iter()
            .enumerate()
            .map(|(index, element)| (element.stable_identity, index))
            .collect();
        let lookup = |values: &[super::MaxwellElementVector]| {
            values
                .iter()
                .map(|value| {
                    element_index
                        .get(&value.element_identity)
                        .map(|&index| (index, value.vector))
                        .ok_or_else(|| invalid("drive references a foreign element"))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()
        };
        let mut generator = Vec::new();
        let mut closure = Vec::new();
        let mut unknown_voltage = BTreeMap::new();
        let omega = input.angular_frequency_rad_per_s.canonical_value();
        for (k, drive) in input.drives.iter().enumerate() {
            let g = lookup(&drive.unit_generator_field)?;
            for &e in g.keys() {
                let law = &volumes[e];
                if omega * law.permittivity_f_per_m.canonical_value()
                    > 1e-3 * law.conductivity_s_per_m.canonical_value()
                {
                    return Err(MaxwellError::Unsupported(format!(
                        "driven conductor {} is not a good conductor at this frequency (omega eps / sigma > 1e-3)",
                        drive.drive_id
                    )));
                }
            }
            generator.push(g);
            closure.push(lookup(&drive.unit_closure_current)?);
            if matches!(drive.excitation, MaxwellDriveExcitation::Current { .. }) {
                unknown_voltage.insert(k, free + unknown_voltage.len());
            }
        }
        let grounded_nodes = geometry::grounded_nodes(&projection, &input.grounded_faces);
        Ok(Self {
            projection,
            space,
            input,
            volumes,
            grounded_nodes,
            rows,
            free,
            generator,
            closure,
            unknown_voltage,
        })
    }

    pub fn dimension(&self) -> usize {
        self.free + self.unknown_voltage.len()
    }

    /// Unit impressed current density of drive k in element e (A/m^2 per V).
    fn unit_current(&self, k: usize, e: usize) -> [f64; 3] {
        let sigma = self.volumes[e].conductivity_s_per_m.canonical_value();
        let g = self.generator[k].get(&e).copied().unwrap_or([0.0; 3]);
        let j = self.closure[k].get(&e).copied().unwrap_or([0.0; 3]);
        std::array::from_fn(|axis| sigma * g[axis] + j[axis])
    }

    /// Assemble the augmented complex system.
    pub fn assemble(&self) -> Result<(Vec<ComplexEntry>, Vec<(f64, f64)>), MaxwellError> {
        let omega = self.input.angular_frequency_rad_per_s.canonical_value();
        let mut entries = Vec::with_capacity(self.projection.elements.len() * 36);
        let mut rhs = vec![ComplexPhasor::ZERO; self.dimension()];
        let push = |entries: &mut Vec<ComplexEntry>, row, column, value: ComplexPhasor| {
            entries.push(ComplexEntry {
                row,
                column,
                real: value.real,
                imaginary: value.imaginary,
            })
        };
        for (e, law) in self.volumes.iter().enumerate() {
            let moments = self
                .space
                .element_moments(&self.projection, e)
                .map_err(topology)?;
            let indices = self
                .space
                .element_edge_indices(e)
                .ok_or_else(|| topology("missing element edge inventory"))?;
            for (i, edge) in indices.iter().enumerate() {
                let Some(&row) = self.rows.get(edge) else {
                    continue;
                };
                for (j, other) in indices.iter().enumerate() {
                    if let Some(&column) = self.rows.get(other) {
                        push(
                            &mut entries,
                            row,
                            column,
                            ComplexPhasor::new(
                                moments.curl_curl[i][j]
                                    / law.permeability_h_per_m.canonical_value()
                                    - omega
                                        * omega
                                        * law.permittivity_f_per_m.canonical_value()
                                        * moments.mass[i][j],
                                omega
                                    * law.conductivity_s_per_m.canonical_value()
                                    * moments.mass[i][j],
                            ),
                        );
                    }
                }
                // Drive loads: b_k,i = -i omega (J_unit_k . integral w_i).
                for (k, drive) in self.input.drives.iter().enumerate() {
                    let unit = self.unit_current(k, e);
                    let projected: f64 = (0..3)
                        .map(|axis| unit[axis] * moments.constant_vector_load[i][axis])
                        .sum();
                    if projected == 0.0 {
                        continue;
                    }
                    let load = ComplexPhasor::new(0.0, -omega * projected);
                    match self.unknown_voltage.get(&k) {
                        Some(&column) => push(&mut entries, row, column, -load),
                        None => {
                            let MaxwellDriveExcitation::Voltage { volts } = drive.excitation else {
                                unreachable!()
                            };
                            rhs[row] += load * volts;
                        }
                    }
                }
            }
            // Current rows: -i omega [ (sigma g, w_i) e_i + (sigma g, g) U ] = -i omega I.
            for (&k, &drive_row) in &self.unknown_voltage {
                let Some(g) = self.generator[k].get(&e) else {
                    continue;
                };
                let sigma = law.conductivity_s_per_m.canonical_value();
                for (i, edge) in indices.iter().enumerate() {
                    if let Some(&column) = self.rows.get(edge) {
                        let c: f64 = (0..3)
                            .map(|axis| sigma * g[axis] * moments.constant_vector_load[i][axis])
                            .sum();
                        push(
                            &mut entries,
                            drive_row,
                            column,
                            ComplexPhasor::new(0.0, -omega * c),
                        );
                    }
                }
                let volume = element_volume(&self.projection, e)?;
                let gg = sigma * volume * (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
                push(
                    &mut entries,
                    drive_row,
                    drive_row,
                    ComplexPhasor::new(0.0, -omega * gg),
                );
            }
            // Impressed volume current: -i omega (J, w_i) by Keast15 quadrature.
            let samples = &self.input.source_current[e * 15..(e + 1) * 15];
            let mut ordinal = 0;
            self.space
                .visit_element_quadrature(&self.projection, e, |point| {
                    let source = &samples[ordinal];
                    ordinal += 1;
                    for (i, edge) in indices.iter().enumerate() {
                        let Some(&row) = self.rows.get(edge) else {
                            continue;
                        };
                        for axis in 0..3 {
                            let weight = point.volume_weight_m3 * point.values[i][axis] * omega;
                            // -i omega J: real = +omega Im J, imaginary = -omega Re J.
                            rhs[row] += ComplexPhasor::new(
                                weight
                                    * source.current_density_a_per_m2.canonical_value()[axis]
                                        .imaginary,
                                -weight
                                    * source.current_density_a_per_m2.canonical_value()[axis].real,
                            );
                        }
                    }
                })
                .map_err(topology)?;
            if ordinal != 15 {
                return Err(topology("Maxwell rule cardinality mismatch"));
            }
        }
        for (&k, &drive_row) in &self.unknown_voltage {
            let MaxwellDriveExcitation::Current { amperes } = self.input.drives[k].excitation
            else {
                unreachable!()
            };
            rhs[drive_row] += ComplexPhasor::new(0.0, -omega) * amperes;
        }
        if rhs.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite Maxwell integrated source"));
        }
        Ok((
            entries,
            rhs.into_iter().map(|v| (v.real, v.imaginary)).collect(),
        ))
    }

    /// Terminal voltages for every drive given the augmented solution.
    fn voltages(&self, solution: &[ComplexPhasor]) -> Vec<ComplexPhasor> {
        self.input
            .drives
            .iter()
            .enumerate()
            .map(|(k, drive)| match drive.excitation {
                MaxwellDriveExcitation::Voltage { volts } => volts,
                MaxwellDriveExcitation::Current { .. } => solution[self.unknown_voltage[&k]],
            })
            .collect()
    }

    /// Recover fields and every acceptance quantity from edge circulations and voltages.
    /// Used after the solve and, unchanged, to re-validate a retained result.
    pub fn evaluate(
        &self,
        edges: &[ComplexPhasor],
        voltages: &[ComplexPhasor],
    ) -> Result<MaxwellEvaluation, MaxwellError> {
        let omega = self.input.angular_frequency_rad_per_s.canonical_value();
        let mut samples = Vec::with_capacity(self.input.sampling.integration_sample_count());
        let mut continuity = ChargeContinuity::new(&self.projection, &self.grounded_nodes);
        let mut electric_energy = 0.0;
        let mut magnetic_energy = 0.0;
        let mut ohmic = 0.0;
        let mut source_power = ComplexPhasor::ZERO;
        let mut currents = vec![ComplexPhasor::ZERO; self.input.drives.len()];
        for (e, law) in self.volumes.iter().enumerate() {
            let indices = self
                .space
                .element_edge_indices(e)
                .ok_or_else(|| topology("missing Maxwell element edges"))?;
            let mut generator = [ComplexPhasor::ZERO; 3];
            let mut impressed_drive = [ComplexPhasor::ZERO; 3];
            for (k, voltage) in voltages.iter().enumerate() {
                if let Some(g) = self.generator[k].get(&e) {
                    for axis in 0..3 {
                        generator[axis] += *voltage * g[axis];
                    }
                }
                let unit = self.unit_current(k, e);
                for axis in 0..3 {
                    impressed_drive[axis] += *voltage * unit[axis];
                }
            }
            let mut ordinal = 0;
            let element_identity = self.input.sampling.element_identities()[e];
            let sample_identities = (0..15)
                .map(|o| self.input.sampling.integration_sample_identity(e * 15 + o))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| topology(error.to_string()))?;
            self.space
                .visit_element_quadrature(&self.projection, e, |point| {
                    let position = e * 15 + ordinal;
                    let volume_source = self.input.source_current[position]
                        .current_density_a_per_m2
                        .canonical_value();
                    let edge_field: ComplexVector3 = std::array::from_fn(|axis| {
                        (0..6).fold(ComplexPhasor::ZERO, |sum, i| {
                            sum + edges[indices[i]] * point.values[i][axis]
                        })
                    });
                    let curl: ComplexVector3 = std::array::from_fn(|axis| {
                        (0..6).fold(ComplexPhasor::ZERO, |sum, i| {
                            sum + edges[indices[i]] * point.curls[i][axis]
                        })
                    });
                    let electric: ComplexVector3 =
                        std::array::from_fn(|axis| edge_field[axis] + generator[axis]);
                    let flux = curl.map(|value| value.times_i() / omega);
                    let magnetic =
                        flux.map(|value| value / law.permeability_h_per_m.canonical_value());
                    // The generator field's displacement current is neglected (driven
                    // conductors satisfy omega eps / sigma <= 1e-3).
                    let displacement =
                        edge_field.map(|value| value * law.permittivity_f_per_m.canonical_value());
                    let conduction =
                        electric.map(|value| value * law.conductivity_s_per_m.canonical_value());
                    let weight = point.volume_weight_m3;
                    electric_energy += 0.25
                        * weight
                        * law.permittivity_f_per_m.canonical_value()
                        * vector_norm_squared(edge_field);
                    magnetic_energy += 0.25
                        * weight
                        * law.permeability_h_per_m.canonical_value()
                        * vector_norm_squared(magnetic);
                    ohmic += 0.5
                        * weight
                        * law.conductivity_s_per_m.canonical_value()
                        * vector_norm_squared(electric);
                    let closure_source: ComplexVector3 = std::array::from_fn(|axis| {
                        impressed_drive[axis]
                            - generator[axis] * law.conductivity_s_per_m.canonical_value()
                    });
                    let free_source: ComplexVector3 =
                        std::array::from_fn(|axis| volume_source[axis] + closure_source[axis]);
                    source_power += dot_conjugate(edge_field, free_source) * (-0.5 * weight);
                    let impressed: ComplexVector3 =
                        std::array::from_fn(|axis| volume_source[axis] + impressed_drive[axis]);
                    continuity.accumulate(
                        &self.projection.elements[e].node_indices[..4],
                        &point,
                        edge_field.map(|value| value * law.conductivity_s_per_m.canonical_value()),
                        displacement,
                        impressed,
                        omega,
                    );
                    for (k, current) in currents.iter_mut().enumerate() {
                        if let Some(g) = self.generator[k].get(&e) {
                            for axis in 0..3 {
                                *current += conduction[axis] * (g[axis] * weight);
                            }
                        }
                    }
                    samples.push(MaxwellFieldSample {
                        sample_identity: sample_identities[ordinal],
                        element_identity,
                        ordinal,
                        position_m: point.position_m,
                        volume_weight_m3: weight,
                        electric_field_v_per_m: Quantity::new(electric),
                        flux_density_t: Quantity::new(flux),
                        magnetic_field_a_per_m: Quantity::new(magnetic),
                        electric_displacement_c_per_m2: Quantity::new(displacement),
                        conduction_current_density_a_per_m2: Quantity::new(conduction),
                    });
                    ordinal += 1;
                })
                .map_err(topology)?;
        }
        let drives: Vec<MaxwellDriveResponse> = self
            .input
            .drives
            .iter()
            .zip(voltages)
            .zip(&currents)
            .map(|((drive, voltage), current)| MaxwellDriveResponse {
                drive_id: drive.drive_id.clone(),
                voltage_v: runmat_types::physical::Quantity::new(*voltage),
                current_a: runmat_types::physical::Quantity::new(*current),
                complex_power_va: runmat_types::physical::Quantity::new(
                    *voltage * current.conjugate() * 0.5,
                ),
            })
            .collect();
        for drive in &drives {
            source_power += drive.complex_power_va.canonical_value();
        }
        let reactive = 2.0 * omega * (magnetic_energy - electric_energy);
        let predicted = ComplexPhasor::new(ohmic, reactive);
        let scale = predicted.abs().max(source_power.abs());
        let closure = if scale > 0.0 {
            (source_power - predicted).abs() / scale
        } else {
            (source_power - predicted).abs()
        };
        Ok(MaxwellEvaluation {
            samples,
            drives,
            electric_energy_j: electric_energy,
            magnetic_energy_j: magnetic_energy,
            ohmic_power_w: ohmic,
            reactive_power_var: reactive,
            source_power_va: source_power,
            relative_complex_power_error: closure,
            relative_charge_continuity_residual: continuity.relative_defect(),
        })
    }

    /// Relative residual of the augmented equations for a retained edge/voltage
    /// solution, certified on the same coalesced operator as the hot LU solve.
    pub fn residual(
        &self,
        entries: &[ComplexEntry],
        rhs: &[(f64, f64)],
        edges: &[ComplexPhasor],
        voltages: &[ComplexPhasor],
    ) -> Result<f64, MaxwellError> {
        let mut unknowns = vec![(0.0, 0.0); self.dimension()];
        for (&edge, &row) in &self.rows {
            let value = edges[edge];
            unknowns[row] = (value.real, value.imaginary);
        }
        for (&k, &column) in &self.unknown_voltage {
            let value = voltages[k];
            unknowns[column] = (value.real, value.imaginary);
        }
        sparse_direct::complex_relative_residual(self.dimension(), entries, rhs, &unknowns)
            .map_err(|error| MaxwellError::NotConverged(error.to_string()))
    }

    pub fn edge_count(&self) -> usize {
        self.space.edges().len()
    }
}

pub(super) struct MaxwellEvaluation {
    pub samples: Vec<MaxwellFieldSample>,
    pub drives: Vec<MaxwellDriveResponse>,
    pub electric_energy_j: f64,
    pub magnetic_energy_j: f64,
    pub ohmic_power_w: f64,
    pub reactive_power_var: f64,
    pub source_power_va: ComplexPhasor,
    pub relative_complex_power_error: f64,
    pub relative_charge_continuity_residual: f64,
}

fn element_volume(projection: &AnalysisProjection, e: usize) -> Result<f64, MaxwellError> {
    let mut volume = 0.0;
    projection
        .visit_element_quadrature(&projection.elements[e], |point| {
            volume += point.volume_weight_m3
        })
        .map_err(topology)?;
    Ok(volume)
}

/// Assemble, factor and solve, then recover and accept the response.
pub(super) fn solve(
    mesh: &SolverMeshArtifact,
    input: &MaxwellInput,
    tolerance: f64,
    plan: &MaxwellSystemPlan,
) -> Result<MaxwellResponse, MaxwellError> {
    if !tolerance.is_finite() || tolerance <= 0.0 || tolerance >= 1.0 {
        return Err(invalid("relative residual target"));
    }
    let system = MaxwellDiscretization::new(mesh, input, plan)?;
    let (entries, rhs) = system.assemble()?;
    let solved = sparse_direct::solve_complex(system.dimension(), &entries, &rhs)
        .map_err(|error| MaxwellError::NotConverged(error.to_string()))?;
    let solution: Vec<ComplexPhasor> = solved
        .solution
        .iter()
        .map(|(a, b)| ComplexPhasor::new(*a, *b))
        .collect();
    let mut edges = vec![ComplexPhasor::ZERO; system.edge_count()];
    for (&edge, &row) in &system.rows {
        edges[edge] = solution[row];
    }
    let voltages = system.voltages(&solution);
    // The factorized, duplicate-coalesced operator and compensated residual are
    // the same authority used by shared LU refinement and cold revalidation.
    let physical = solved.relative_residual;
    drop(entries);
    if !physical.is_finite() || physical > tolerance {
        return Err(MaxwellError::NotConverged(format!(
            "physical complex residual={physical:e}, LU refinement residual={:e} after {} steps",
            solved.refinement_relative_residual, solved.refinement_steps,
        )));
    }
    let evaluation = system.evaluate(&edges, &voltages)?;
    accept(&evaluation, tolerance)?;
    Ok(MaxwellResponse {
        input_identity: input.identity()?,
        system_plan_identity: plan.identity()?,
        mesh_digest: system.projection.mesh_digest,
        projection_digest: system.projection.projection_digest,
        sampling_digest: input.sampling.sampling_digest(),
        angular_frequency_rad_per_s: input.angular_frequency_rad_per_s,
        edge_node_identities: system
            .space
            .edges()
            .iter()
            .map(|edge| edge.node_identities)
            .collect(),
        edge_circulations_v: Quantity::new(edges),
        samples: evaluation.samples,
        drives: evaluation.drives,
        electric_energy_j: Quantity::new(evaluation.electric_energy_j),
        magnetic_energy_j: Quantity::new(evaluation.magnetic_energy_j),
        source_power_va: Quantity::new(evaluation.source_power_va),
        ohmic_power_w: Quantity::new(evaluation.ohmic_power_w),
        reactive_power_var: Quantity::new(evaluation.reactive_power_var),
        relative_physical_residual: physical,
        relative_complex_power_error: evaluation.relative_complex_power_error,
        relative_charge_continuity_residual: evaluation.relative_charge_continuity_residual,
    })
}

fn accept(evaluation: &MaxwellEvaluation, tolerance: f64) -> Result<(), MaxwellError> {
    let closure = evaluation.relative_complex_power_error;
    let charge = evaluation.relative_charge_continuity_residual;
    if [
        evaluation.electric_energy_j,
        evaluation.magnetic_energy_j,
        evaluation.ohmic_power_w,
        evaluation.reactive_power_var,
        closure,
        charge,
    ]
    .iter()
    .any(|v| !v.is_finite())
        || !evaluation.source_power_va.is_finite()
        || evaluation.ohmic_power_w < 0.0
        || closure > tolerance * 16.0
        || charge > tolerance * 16.0
        || evaluation.samples.iter().any(|s| {
            [
                s.electric_field_v_per_m.canonical_value(),
                s.flux_density_t.canonical_value(),
                s.magnetic_field_a_per_m.canonical_value(),
                s.electric_displacement_c_per_m2.canonical_value(),
                s.conduction_current_density_a_per_m2.canonical_value(),
            ]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite())
        })
    {
        return Err(MaxwellError::NotConverged(format!(
            "complex power error={closure:e}, harmonic charge continuity={charge:e}"
        )));
    }
    Ok(())
}

/// Re-validate a retained response against its original input without solving.
pub(crate) fn validate_retained_consistency(
    mesh: &SolverMeshArtifact,
    input: &MaxwellInput,
    response: &MaxwellResponse,
    tolerance: f64,
) -> Result<(), MaxwellError> {
    let plan = MaxwellSystemPlan::derive(mesh, &input.grounded_faces)?;
    let system = MaxwellDiscretization::new(mesh, input, &plan)?;
    let edges = response.edge_circulations_v.canonical_values();
    if edges.len() != system.edge_count()
        || !response.edge_node_identities.iter().copied().eq(system
            .space
            .edges()
            .iter()
            .map(|edge| edge.node_identities))
        || (0..edges.len())
            .any(|edge| !system.rows.contains_key(&edge) && edges[edge] != ComplexPhasor::ZERO)
        || response.drives.len() != input.drives.len()
        || response
            .drives
            .iter()
            .zip(&input.drives)
            .any(|(retained, drive)| retained.drive_id != drive.drive_id)
    {
        return Err(invalid(
            "retained Maxwell edges/drives differ from the original grounded space",
        ));
    }
    let voltages: Vec<_> = response
        .drives
        .iter()
        .map(|d| d.voltage_v.canonical_value())
        .collect();
    let (entries, rhs) = system.assemble()?;
    let residual = system.residual(&entries, &rhs, edges, &voltages)?;
    drop(entries);
    let evaluation = system.evaluate(edges, &voltages)?;
    if !residual.is_finite()
        || residual > tolerance
        || (residual - response.relative_physical_residual).abs() > 1e-6 * tolerance.max(residual)
        || evaluation.samples != response.samples
        || evaluation.drives != response.drives
        || evaluation.relative_charge_continuity_residual
            != response.relative_charge_continuity_residual
        || evaluation.relative_complex_power_error != response.relative_complex_power_error
        || evaluation.ohmic_power_w != response.ohmic_power_w.canonical_value()
    {
        return Err(invalid(
            "retained Maxwell response differs from re-evaluation of its original equations",
        ));
    }
    accept(&evaluation, tolerance)
}
