use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::frequency::FrequencyResponsePlan;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ConstraintPlanIdentity,
    DiscretizationIdentity, StructuralDampingOperatorIdentity, StructuralDofLayoutIdentity,
    StructuralExcitationSetIdentity, StructuralFormulationIdentity,
    StructuralFrequencyProblemIdentity, StructuralFrequencyScopeIdentity,
    StructuralKinematicsIdentity, StructuralMassOperatorIdentity,
    StructuralStiffnessOperatorIdentity,
};

const FREQUENCY_SCOPE_PREFIX: &[u8] = b"runmat-structural-frequency-scope/v1\0";
const FREQUENCY_SCOPE_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 32));

/// Meaning of authored complex amplitudes at the structural frequency boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralPhasorConvention {
    PeakAmplitude,
}

/// Method-neutral identity boundary for one physical structural frequency response.
///
/// Direct and modal solvers can produce the same scope and physical result. Solver choice, modal
/// basis, numerical options, batching, placement, and attempts belong to numerical receipts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralFrequencyScopeWire")]
pub struct StructuralFrequencyScope {
    schema_version: u16,
    identity: StructuralFrequencyScopeIdentity,
    problem: StructuralFrequencyProblemIdentity,
    plan: FrequencyResponsePlan,
    discretization: DiscretizationIdentity,
    constraint_plan: ConstraintPlanIdentity,
    stiffness_operator: StructuralStiffnessOperatorIdentity,
    mass_operator: StructuralMassOperatorIdentity,
    damping_operator: StructuralDampingOperatorIdentity,
    operator_digest: Digest,
    excitations: StructuralExcitationSetIdentity,
    dof_layout: StructuralDofLayoutIdentity,
    formulation: StructuralFormulationIdentity,
    kinematics: StructuralKinematicsIdentity,
    physical_dof_count: usize,
    phasor_convention: StructuralPhasorConvention,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyScopeWire {
    schema_version: u16,
    identity: StructuralFrequencyScopeIdentity,
    problem: StructuralFrequencyProblemIdentity,
    plan: FrequencyResponsePlan,
    discretization: DiscretizationIdentity,
    constraint_plan: ConstraintPlanIdentity,
    stiffness_operator: StructuralStiffnessOperatorIdentity,
    mass_operator: StructuralMassOperatorIdentity,
    damping_operator: StructuralDampingOperatorIdentity,
    operator_digest: Digest,
    excitations: StructuralExcitationSetIdentity,
    dof_layout: StructuralDofLayoutIdentity,
    formulation: StructuralFormulationIdentity,
    kinematics: StructuralKinematicsIdentity,
    physical_dof_count: usize,
    phasor_convention: StructuralPhasorConvention,
}

impl TryFrom<StructuralFrequencyScopeWire> for StructuralFrequencyScope {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencyScopeWire) -> Result<Self, Self::Error> {
        let scope = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            problem: wire.problem,
            plan: wire.plan,
            discretization: wire.discretization,
            constraint_plan: wire.constraint_plan,
            stiffness_operator: wire.stiffness_operator,
            mass_operator: wire.mass_operator,
            damping_operator: wire.damping_operator,
            operator_digest: wire.operator_digest,
            excitations: wire.excitations,
            dof_layout: wire.dof_layout,
            formulation: wire.formulation,
            kinematics: wire.kinematics,
            physical_dof_count: wire.physical_dof_count,
            phasor_convention: wire.phasor_convention,
        };
        scope.validate()?;
        Ok(scope)
    }
}

pub struct StructuralFrequencyScopeInput {
    pub problem: StructuralFrequencyProblemIdentity,
    pub plan: FrequencyResponsePlan,
    pub discretization: DiscretizationIdentity,
    pub constraint_plan: ConstraintPlanIdentity,
    pub stiffness_operator: StructuralStiffnessOperatorIdentity,
    pub mass_operator: StructuralMassOperatorIdentity,
    pub damping_operator: StructuralDampingOperatorIdentity,
    pub operator_digest: Digest,
    pub excitations: StructuralExcitationSetIdentity,
    pub dof_layout: StructuralDofLayoutIdentity,
    pub formulation: StructuralFormulationIdentity,
    pub kinematics: StructuralKinematicsIdentity,
    pub physical_dof_count: usize,
    pub phasor_convention: StructuralPhasorConvention,
}

impl StructuralFrequencyScope {
    pub fn new(input: StructuralFrequencyScopeInput) -> AnalysisExecutionResult<Self> {
        let mut scope = Self {
            schema_version: 1,
            identity: StructuralFrequencyScopeIdentity::derive(
                "structural-frequency-scope-placeholder",
                &1_u8,
            )?,
            problem: input.problem,
            plan: input.plan,
            discretization: input.discretization,
            constraint_plan: input.constraint_plan,
            stiffness_operator: input.stiffness_operator,
            mass_operator: input.mass_operator,
            damping_operator: input.damping_operator,
            operator_digest: input.operator_digest,
            excitations: input.excitations,
            dof_layout: input.dof_layout,
            formulation: input.formulation,
            kinematics: input.kinematics,
            physical_dof_count: input.physical_dof_count,
            phasor_convention: input.phasor_convention,
        };
        scope.identity = scope.derive_identity()?;
        scope.validate()?;
        Ok(scope)
    }

    pub const fn identity(&self) -> StructuralFrequencyScopeIdentity {
        self.identity
    }

    pub const fn physical_dof_count(&self) -> usize {
        self.physical_dof_count
    }

    pub const fn stiffness_operator(&self) -> StructuralStiffnessOperatorIdentity {
        self.stiffness_operator
    }

    pub const fn operator_digest(&self) -> Digest {
        self.operator_digest
    }

    pub fn plan(&self) -> &FrequencyResponsePlan {
        &self.plan
    }

    pub const fn phasor_convention(&self) -> StructuralPhasorConvention {
        self.phasor_convention
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            FREQUENCY_SCOPE_PREFIX,
            "structural-frequency-scope",
            self,
            FREQUENCY_SCOPE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            FREQUENCY_SCOPE_PREFIX,
            "structural-frequency-scope",
            bytes,
            FREQUENCY_SCOPE_LIMITS,
        )?)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.plan
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.schema_version != 1 || self.physical_dof_count == 0 {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency scope has an invalid schema or physical DOF count".into(),
            ));
        }
        if self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency scope identity does not converge",
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StructuralFrequencyScopeIdentity> {
        StructuralFrequencyScopeIdentity::derive(
            "structural-frequency-scope",
            &(
                self.schema_version,
                self.problem,
                &self.plan,
                self.discretization,
                self.constraint_plan,
                self.stiffness_operator,
                self.mass_operator,
                self.damping_operator,
                self.operator_digest,
                self.excitations,
                self.dof_layout,
                self.formulation,
                self.kinematics,
                self.physical_dof_count,
                self.phasor_convention,
            ),
        )
    }
}
