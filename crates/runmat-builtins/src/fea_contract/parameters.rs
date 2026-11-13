//! Parameter descriptors for the FEA operation surface. These are the single
//! signature authority; runtime entrypoints re-export them.

use crate::BuiltinParamArity;
use crate::BuiltinParamDescriptor;
use crate::BuiltinParamType;

pub const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "FEA object or operation result.",
}];
pub const IN_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "study",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A fea.Study or fea.Sweep object.",
}];
pub const IN_STUDY_ARGS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Study id.",
    },
    BuiltinParamDescriptor {
        name: "geometry",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "geometry.Asset returned by geometry.load.",
    },
    BuiltinParamDescriptor {
        name: "model",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Explicit fea.Model bound to the supplied geometry revision.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional Mesh and Execution settings.",
    },
];
pub const IN_VARIADIC_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Constructor or query arguments.",
}];
pub const MODEL_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Model id.",
    },
    BuiltinParamDescriptor {
        name: "geometry",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing geometry.Asset object.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Explicit frame, materials, assignments, constraints, boundaries, signals, excitations, steps or sequence, domains, and interfaces.",
    },
];
pub const MATERIAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Material id.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Mechanical, thermal, acoustic, electrical, and plastic material fields.",
    },
];
pub const SWEEP_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sweep id.",
    },
    BuiltinParamDescriptor {
        name: "studies",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A fea.Study or cell array of fea.Study objects.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional logical FailFast control.",
    },
];
pub const RESULTS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "runOrRunId",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Persisted run id, fea.RunResult, or fea.Results object.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Field, diagnostic, one-based mode/snapshot selector, and inclusion options.",
    },
];
pub const TRENDS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Name, Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional positive integer WindowSize.",
}];
pub const FIELD_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "resultsOrRun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "fea.Results, fea.RunResult, or persisted run id.",
    },
    BuiltinParamDescriptor {
        name: "field",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed quantity or authored-output selector, such as fea.StructuralField.stress() or fea.OutputField.named(\"peak_stress\").",
    },
    BuiltinParamDescriptor {
        name: "sample",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Typed independent-domain sample, such as fea.Frequency.hertz(120).",
    },
];
pub const FIELD_PAGE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "field",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Artifact-backed fea.Field returned by fea.field.",
    },
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "One-based logical value index.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("256"),
        description: "Positive maximum logical value count, capped at 4096.",
    },
];
pub const STEP_RESULT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "run",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Aggregate fea.RunResult returned by fea.run.",
    },
    BuiltinParamDescriptor {
        name: "stepId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exact authored procedure-step id.",
    },
];
pub const PLOT_CONTEXT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "field",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "FEA field carrying result and study context.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Visualization options such as Edges, Overlay, or Deformed.",
    },
];
pub const MESH_CONTEXT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "context",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "FEA run, results, or field carrying mesh context.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Mesh display options such as Source or Edges.",
    },
];
pub const COMPARE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "baselineRunId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Baseline persisted run id.",
    },
    BuiltinParamDescriptor {
        name: "candidateRunId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Candidate persisted run id.",
    },
];
pub const MATERIAL_ASSIGNMENT_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed beam, shell, solid, or cell property-region target.",
    },
    BuiltinParamDescriptor {
        name: "material",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Material value.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Expected-material and confidence options.",
    },
];
