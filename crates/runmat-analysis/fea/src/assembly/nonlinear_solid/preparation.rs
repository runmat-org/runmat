use super::*;

impl CanonicalNonlinearSolidBasis {
    pub(crate) fn prepare(
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        initial: &AssemblySummary,
    ) -> Result<Self, NonlinearSolidError> {
        Self::prepare_composed(model, mesh, initial, None)
    }

    pub(crate) fn prepare_composed(
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        initial: &AssemblySummary,
        contact: Option<&crate::physics::structural::contact::CanonicalContactOperator<'_>>,
    ) -> Result<Self, NonlinearSolidError> {
        if crate::progress::is_cancelled() {
            return Err(NonlinearSolidError::Cancelled);
        }
        let plan =
            CanonicalSolidIntegrationPlan::derive(mesh, CanonicalSolidIntegrationRule::Keast15)
                .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?;
        match contact {
            Some(contact) => {
                Self::prepare_with_integration_composed(model, mesh, initial, &plan, Some(contact))
            }
            None => Self::prepare_with_integration(model, mesh, initial, &plan),
        }
    }

    pub(crate) fn prepare_with_integration(
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        initial: &AssemblySummary,
        plan: &CanonicalSolidIntegrationPlan,
    ) -> Result<Self, NonlinearSolidError> {
        Self::prepare_with_integration_composed(model, mesh, initial, plan, None)
    }

    fn prepare_with_integration_composed(
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        initial: &AssemblySummary,
        plan: &CanonicalSolidIntegrationPlan,
        contact: Option<&crate::physics::structural::contact::CanonicalContactOperator<'_>>,
    ) -> Result<Self, NonlinearSolidError> {
        if crate::progress::is_cancelled() {
            return Err(NonlinearSolidError::Cancelled);
        }
        plan.validate(mesh, plan.rule())
            .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?;
        if model.structural.is_some() || !model.connectors.is_empty() || (contact.is_none() && !model.interfaces.is_empty())
            || model.thermo_mechanical.is_some() || model.electro_thermal.is_some() || model.thermal.is_some()
            || model.electromagnetic.is_some() || model.cfd.is_some()
            || initial.thermo_mechanical.is_some() || initial.electro_thermal.is_some()
            || initial.structural_beam_element_count != 0 || initial.structural_shell_element_count != 0
            || initial.structural_connector_stiffness_count != 0 || initial.structural_connector_damping_count != 0
            || initial.structural_dof_layout.has_rotational_dofs()
            || initial.structural_formulations.len() != 1
            || initial.structural_formulations.iter().any(|f| !matches!(f.operator,runmat_analysis_core::StructuralOperatorImplementation::Tet4LinearElasticity|runmat_analysis_core::StructuralOperatorImplementation::Tet10LinearElasticity)) {
            return Err(NonlinearSolidError::Unsupported("only uncoupled canonical isotropic Tet4/Tet10 solids; no explicit/mixed/contact/connectors or thermal input".into()));
        }
        let projection = AnalysisProjection::derive(mesh).map_err(NonlinearSolidError::from)?;
        if let Some(contact) = contact {
            if !contact.matches_sources(&projection, &model.interfaces) {
                return Err(NonlinearSolidError::Authority(
                    "contact operator differs from full authored interface/geometry source".into(),
                ));
            }
        }
        let constraints = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
            model,
            Some(mesh),
        )
        .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?;
        let symbolic = crate::planning::resource::StructuralSymbolicOperators::derive(
            &projection,
            &constraints,
            model,
            mesh,
        )
        .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?;
        let tangent_pattern = symbolic
            .stiffness_upper_bound()
            .try_clone()
            .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?;
        let properties = compile_structural_property_map(model, Some(mesh))
            .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?;
        if Some(&properties.identity) != initial.structural_property_map_identity.as_ref()
            || initial.structural_solid_element_count != projection.elements.len()
            || initial.structural_dof_layout.total_dof_count() != initial.dof_count
        {
            return Err(NonlinearSolidError::Authority(
                "initial assembly does not authenticate this mesh, properties or layout".into(),
            ));
        }
        let node_space = CanonicalStructuralNodeSpace::derive_solid(mesh)
            .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?;
        if initial.structural_dof_layout.node_count() != node_space.nodes.len() {
            return Err(NonlinearSolidError::Authority(
                "initial layout node inventory differs".into(),
            ));
        }
        let canonical_nodes = node_space
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.identity, i))
            .collect::<BTreeMap<_, _>>();
        let recovery = initial
            .structural_solid_recovery
            .iter()
            .map(|e| (e.element_id.as_str(), e))
            .collect::<BTreeMap<_, _>>();
        if recovery.len() != projection.elements.len() {
            return Err(NonlinearSolidError::Authority(
                "initial recovery element inventory differs".into(),
            ));
        }
        let mut elements = Vec::with_capacity(projection.elements.len());
        for (element, inventory) in projection.elements.iter().zip(plan.elements()) {
            if element.stable_identity != inventory.stable_identity
                || element.element_id != inventory.element_id
                || element.order != inventory.order
                || element.node_indices.len() != inventory.node_count
            {
                return Err(NonlinearSolidError::Authority(
                    "integration/projection inventory differs".into(),
                ));
            }
            if !matches!(element.order, ElementOrder::Tet4 | ElementOrder::Tet10) {
                return Err(NonlinearSolidError::Unsupported(
                    "unsupported solid order".into(),
                ));
            }
            let owned = properties
                .solid_material(element.stable_identity)
                .ok_or_else(|| {
                    NonlinearSolidError::Authority("missing element material owner".into())
                })?;
            let constitutive = IsotropicConstitutiveModel::from_material(&owned.material)
                .map_err(|e| NonlinearSolidError::Constitutive(e.to_string()))?;
            let mut dofs = Vec::with_capacity(element.node_indices.len());
            let mut canonical_indices = Vec::with_capacity(element.node_indices.len());
            for &index in &element.node_indices {
                let projected = &projection.nodes[index];
                // Match stable mesh node identity to canonical node space before resolving rows.
                let canonical =
                    *canonical_nodes
                        .get(&projected.stable_identity)
                        .ok_or_else(|| {
                            NonlinearSolidError::Authority(
                                "projection node is not canonical".into(),
                            )
                        })?;
                canonical_indices.push(canonical);
                let mut rows = [0; 3];
                for (axis, kind) in [
                    StructuralDofKind::Ux,
                    StructuralDofKind::Uy,
                    StructuralDofKind::Uz,
                ]
                .into_iter()
                .enumerate()
                {
                    rows[axis] = initial
                        .structural_dof_layout
                        .index(canonical, kind)
                        .ok_or_else(|| {
                            NonlinearSolidError::Authority(
                                "missing physical translation row".into(),
                            )
                        })?;
                }
                dofs.push(rows);
            }
            let id = element.element_id.to_string();
            let initial_element = recovery
                .get(id.as_str())
                .ok_or_else(|| NonlinearSolidError::Authority("initial element missing".into()))?;
            if initial_element.geometry.node_indices() != canonical_indices
                || initial_element.thermal_eigenstrain.is_some()
                || initial_element.solved_thermal_quadrature.is_some()
                || initial_element.prescribed_temperature_k.is_some()
            {
                return Err(NonlinearSolidError::Authority(
                    "initial solid mapping or thermal authority differs".into(),
                ));
            }
            let coordinates: &[[f64; 3]] = match &initial_element.geometry {
                super::super::SolidRecoveryGeometry::Tetrahedron4 { coordinates_m, .. } => {
                    coordinates_m.canonical_values()
                }
                super::super::SolidRecoveryGeometry::Tetrahedron10 { coordinates_m, .. } => {
                    coordinates_m.canonical_values()
                }
            };
            let mechanical = owned
                .material
                .require_mechanical()
                .map_err(|e| NonlinearSolidError::Constitutive(e.to_string()))?;
            if coordinates
                .iter()
                .zip(&element.node_indices)
                .any(|(x, &node)| *x != projection.nodes[node].coordinates_m.canonical_value())
                || initial_element.material.youngs_modulus_pa.canonical_value()
                    != mechanical.youngs_modulus_pa.canonical_value()
                || initial_element.material.poisson_ratio.canonical_value()
                    != mechanical.poisson_ratio.canonical_value()
            {
                return Err(NonlinearSolidError::Authority(
                    "initial recovery geometry or elasticity differs from compiled owner".into(),
                ));
            }
            let mut points = Vec::with_capacity(inventory.point_count);
            crate::topology::quadrature::visit_volume_rule(
                element.order,
                coordinates,
                // The element's own rule: a curved Tet10 integrates on the converged points of
                // the shared Tet10 element-validity rule.
                inventory.rule.reference_points(),
                |point| {
                    points.push(PreparedPoint {
                        identity: NonlinearSolidPointIdentity {
                            element_identity: element.stable_identity,
                            quadrature_ordinal: points.len(),
                        },
                        position_m: point.position_m,
                        weight_m3: point.volume_weight_m3,
                        gradients: point.physical_shape_gradients().to_vec(),
                    });
                },
            )
            .map_err(NonlinearSolidError::Accounting)?
            .ok_or_else(|| {
                NonlinearSolidError::Authority("invalid physical integration geometry".into())
            })?;
            if points.len() != inventory.point_count
                || points.iter().any(|p| {
                    !p.weight_m3.is_finite()
                        || p.weight_m3 <= 0.0
                        || p.gradients
                            .iter()
                            .flatten()
                            .chain(p.position_m.iter())
                            .any(|v| !v.is_finite())
                })
            {
                return Err(NonlinearSolidError::Authority(
                    "invalid physical quadrature".into(),
                ));
            }
            elements.push(PreparedElement {
                identity: element.stable_identity,
                element_id: element.element_id,
                dofs,
                constitutive,
                points,
            });
        }
        let rows = elements
            .iter()
            .flat_map(|e| e.dofs.iter().flatten().copied())
            .collect::<BTreeSet<_>>();
        if rows.len() != initial.structural_dof_layout.node_dof_count() {
            return Err(NonlinearSolidError::Authority(
                "solid elements do not cover the physical node layout".into(),
            ));
        }
        let nodes = super::contact_layout::prepare_nodes(&projection, &canonical_nodes, initial)?;
        let authority = super::authority::BasisAuthority {
            integration_identity: plan.identity(),
            tangent_pattern_identity: tangent_pattern
                .identity()
                .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?,
            properties: &properties.identity,
            layout: &initial.structural_dof_layout,
            elements: elements
                .iter()
                .map(|e| (e.identity, e.dofs.as_slice(), e.points.as_slice()))
                .collect(),
        };
        // Streamed: the mesh-sized basis authority is hashed without materializing it.
        let basis_digest = crate::contracts::canonical_digest::canonical_contract_digest(
            b"runmat-fea-nonlinear-solid-basis/v1\0",
            "nonlinear-solid-basis",
            &authority,
            CanonicalLimits::new(0, 0, 4096, 64),
        )
        .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?;
        // The kinematic description is part of the operator: total Lagrangian solids publish a
        // distinct basis identity, while small-strain identities keep their established schema.
        let kinematics = model
            .sequence
            .steps
            .iter()
            .find(|step| step.procedure == AnalysisProcedure::Nonlinear)
            .map_or(StructuralKinematics::SmallStrain, |step| {
                step.structural_kinematics()
            });
        let identity = match kinematics {
            StructuralKinematics::SmallStrain => basis_digest,
            StructuralKinematics::LargeDeformation => Digest::sha256(
                encode_contract(
                    b"runmat-fea-nonlinear-solid-basis-total-lagrangian/v1\0",
                    "nonlinear-solid-basis-total-lagrangian",
                    &basis_digest,
                    CanonicalLimits::new(4096, 64, 64, 8),
                )
                .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?,
            ),
        };
        Ok(Self {
            identity,
            kinematics,
            integration_identity: plan.identity(),
            mesh_digest: mesh.canonical_digest,
            property_identity: properties.identity,
            dof_count: initial.dof_count,
            tangent_pattern,
            nodes,
            elements,
        })
    }
}
