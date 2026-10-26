//! Actual Piola curl-curl/mass/source integration into the shared planned CSR.
use super::{
    geometry, invalid, operator, topology, MaxwellError, MaxwellInput, MaxwellSystemPlan,
    MaxwellVolumeData,
};
use crate::{
    algebra::{
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    topology::{AnalysisProjection, TetrahedralEdgeSpace},
};
use runmat_meshing_core::SolverMeshArtifact;
use runmat_types::physical::{
    kind, DynamicDimension, Quantity, QuantityKind, QuantityLinearOperator, UniformQuantityLayout,
};
use std::collections::BTreeMap;

pub(super) struct MaxwellAssembly<'a> {
    pub projection: AnalysisProjection,
    pub edge_space: TetrahedralEdgeSpace,
    pub plan: &'a MaxwellSystemPlan,
    pub input: &'a MaxwellInput,
    pub volumes: Vec<MaxwellVolumeData>,
    /// [Re E free edges, Im E free edges], physical SI operator, no damping.
    pub block: QuantityLinearOperator<
        UniformQuantityLayout<kind::ElectricPotential>,
        UniformQuantityLayout<DynamicDimension>,
        RealSparseMatrix,
    >,
    pub rhs: Quantity<DynamicDimension, Vec<f64>>,
}
impl<'a> MaxwellAssembly<'a> {
    pub fn assemble(
        mesh: &SolverMeshArtifact,
        input: &'a MaxwellInput,
        plan: &'a MaxwellSystemPlan,
    ) -> Result<Self, MaxwellError> {
        plan.validate_for(mesh)?;
        let volumes = input.ordered_volumes(mesh)?;
        let projection = AnalysisProjection::derive(mesh).map_err(topology)?;
        let (edge_space, _) = geometry::spaces(&projection)?;
        let rows: BTreeMap<_, _> = plan
            .free_edge_indices()
            .iter()
            .enumerate()
            .map(|(row, &edge)| (edge, row))
            .collect();
        let free = rows.len();
        let mut block = RealSparseAssembler::with_pattern(
            plan.block_pattern().clone(),
            OperatorRole::EffectiveTangent,
            MatrixProperties::GENERAL,
        )
        .map_err(operator)?;
        block.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Electromagnetic,
            contribution_id: ContributionId(
                "maxwell_physical_complex_curl_curl_mass_and_volume_current".into(),
            ),
        });
        let mut rhs = vec![
            0.0;
            free.checked_mul(2)
                .ok_or_else(|| invalid("Maxwell dimension overflow"))?
        ];
        // exp(i phase) differentiates with respect to phase before time: the
        // inverse-radian generator cancels nominal radians explicitly.
        let derivative_rate = input.angular_frequency_rad_per_s.phasor_derivative_rate();
        let omega = derivative_rate.canonical_value();
        for (index, data) in volumes.iter().enumerate() {
            let moments = edge_space
                .element_moments(&projection, index)
                .map_err(topology)?;
            let indices = edge_space
                .element_edge_indices(index)
                .ok_or_else(|| topology("missing element edge inventory"))?;
            for (i, edge) in indices.iter().enumerate() {
                let Some(&row) = rows.get(edge) else {
                    continue;
                };
                for (j, other) in indices.iter().enumerate() {
                    if let Some(&column) = rows.get(other) {
                        let real = moments.curl_curl[i][j]
                            / data.permeability_h_per_m.canonical_value()
                            - omega
                                * omega
                                * data.permittivity_f_per_m.canonical_value()
                                * moments.mass[i][j];
                        let imaginary = omega
                            * data.conductivity_s_per_m.canonical_value()
                            * moments.mass[i][j];
                        // A=Ar+iAi -> [[Ar,-Ai],[Ai,Ar]], NOT an SPD operator.
                        block.add(row, column, real);
                        block.add(row, column + free, -imaginary);
                        block.add(row + free, column, imaginary);
                        block.add(row + free, column + free, real);
                    }
                }
            }
            let samples = &input.source_current[index * 15..(index + 1) * 15];
            let mut ordinal = 0;
            edge_space
                .visit_element_quadrature(&projection, index, |point| {
                    let source = &samples[ordinal];
                    ordinal += 1;
                    for (i, edge) in indices.iter().enumerate() {
                        let Some(&row) = rows.get(edge) else {
                            continue;
                        };
                        for axis in 0..3 {
                            let load = point.volume_weight_m3 * point.values[i][axis] * omega;
                            // -iωJ: real=+ω Im J, imaginary=-ω Re J.
                            rhs[row] += load
                                * source.current_density_a_per_m2.canonical_value()[axis].imaginary;
                            rhs[row + free] -=
                                load * source.current_density_a_per_m2.canonical_value()[axis].real;
                        }
                    }
                })
                .map_err(topology)?;
            if ordinal != 15 {
                return Err(topology("Maxwell rule cardinality mismatch"));
            }
        }
        if rhs.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite Maxwell integrated source"));
        }
        let forcing = DynamicDimension::try_new(
            QuantityKind::ElectricCurrent.dimension(),
            Some(QuantityKind::ElectricCurrent),
        )
        .expect("static current")
        .divide(
            DynamicDimension::try_new(QuantityKind::Time.dimension(), Some(QuantityKind::Time))
                .expect("static time"),
        )
        .expect("current time derivative descriptor");
        let block = QuantityLinearOperator::try_new(
            UniformQuantityLayout::new(rhs.len()),
            UniformQuantityLayout::from_descriptor(rhs.len(), forcing),
            block.finish().map_err(operator)?,
        )
        .map_err(operator)?;
        let rhs = Quantity::try_from_descriptor(forcing, rhs).map_err(invalid)?;
        Ok(Self {
            projection,
            edge_space,
            plan,
            input,
            volumes,
            block,
            rhs,
        })
    }
}
