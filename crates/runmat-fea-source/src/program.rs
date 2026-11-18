use runmat_analysis_core::{
    AuthoredExecutionPolicy, AuthoredExecutionPolicyRef, AuthoredModel, AuthoredStudy,
    AuthoredSweep, EvaluatedParameterizedSignal, MaterialModel,
};
use runmat_analysis_language::LogicalSourcePath;
use runmat_geometry_core::GeometryDefinition;
use runmat_meshing_core::MeshDefinition;
use serde::{Deserialize, Serialize};

/// Serializable semantic product of checking a `.fea` module graph.
///
/// Every definition stored here is the canonical domain object produced by the checker. This
/// envelope only replaces map keys that cannot be represented as JSON object keys; it does not
/// translate domain objects into a display-specific schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSemanticProgram {
    pub studies: Vec<AuthoredStudy>,
    pub sweeps: Vec<AuthoredSweep>,
    pub execution_policies: Vec<AuthoredExecutionPolicy>,
    pub execution_bindings: Vec<FeaExecutionBinding>,
    pub materials: Vec<FeaDeclaredMaterial>,
    pub geometries: Vec<GeometryDefinition>,
    pub meshes: Vec<MeshDefinition>,
    pub models: Vec<AuthoredModel>,
    /// Derived from `studies` and `models`: each study's parameterized-signal values at the
    /// study's own inputs. It never enters the checked revision or any study, model or run
    /// identity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluated_signals: Vec<FeaStudyEvaluatedSignals>,
}

/// One study's parameterized signals evaluated at the study's own input values. Sweep cases
/// never change these values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaStudyEvaluatedSignals {
    pub module_path: LogicalSourcePath,
    pub study_id: String,
    pub signals: Vec<EvaluatedParameterizedSignal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaDeclaredMaterial {
    pub module_path: LogicalSourcePath,
    pub material: MaterialModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaExecutionBinding {
    pub target: FeaExecutionTarget,
    pub policy: AuthoredExecutionPolicyRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeaExecutionTarget {
    Study {
        module_path: LogicalSourcePath,
        study_id: String,
    },
    Sweep {
        module_path: LogicalSourcePath,
        sweep_id: String,
    },
}
