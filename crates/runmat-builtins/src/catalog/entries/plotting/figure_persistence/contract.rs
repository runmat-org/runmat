use crate::*;

pub const FIGURE_PERSISTENCE_STATUS_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "figure-persistence-status-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Request a logical completion status from a figure persistence sink",
        error_identifier: Some("RunMat:compatibility:FigurePersistenceStatusOutputExtension"),
    };
pub const FIGURE_PERSISTENCE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [FIGURE_PERSISTENCE_STATUS_OUTPUT_EXTENSION];

pub const FIGURE_PERSISTENCE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.FIGURE_PERSISTENCE.INVALID_ARGUMENT",
        identifier: Some("RunMat:figurePersistence:InvalidArgument"),
        when: "Arguments are missing, malformed, or incompatible with the requested figure persistence operation.",
        message: "figure persistence: invalid argument",
    };
pub const FIGURE_PERSISTENCE_ERROR_UNSUPPORTED_FORMAT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.FIGURE_PERSISTENCE.UNSUPPORTED_FORMAT",
        identifier: Some("RunMat:figurePersistence:UnsupportedFormat"),
        when: "The requested save/load format is recognized but not supported by RunMat's active plotting exporter.",
        message: "figure persistence: unsupported format",
    };
pub const FIGURE_PERSISTENCE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.IO",
    identifier: Some("RunMat:figurePersistence:IoFailure"),
    when: "The target file cannot be read, written, flushed, synced, or atomically replaced.",
    message: "figure persistence: file I/O failed",
};
pub const FIGURE_PERSISTENCE_ERROR_RENDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.RENDER",
    identifier: Some("RunMat:figurePersistence:RenderFailed"),
    when: "The plotting renderer or scene exporter cannot serialize the requested figure.",
    message: "figure persistence: export failed",
};
pub const FIGURE_PERSISTENCE_ERROR_IMPORT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.IMPORT",
    identifier: Some("RunMat:figurePersistence:ImportFailed"),
    when: "The figure file is missing, corrupt, unsupported, or incompatible with RunMat's scene importer.",
    message: "figure persistence: import failed",
};
pub const FIGURE_PERSISTENCE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE_PERSISTENCE.INTERNAL",
    identifier: Some("RunMat:figurePersistence:Internal"),
    when: "Internal runtime control-flow or conversion fails.",
    message: "figure persistence: internal operation failed",
};

pub(super) const FIGURE_PERSISTENCE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    FIGURE_PERSISTENCE_ERROR_INVALID_ARGUMENT,
    FIGURE_PERSISTENCE_ERROR_UNSUPPORTED_FORMAT,
    FIGURE_PERSISTENCE_ERROR_IO,
    FIGURE_PERSISTENCE_ERROR_RENDER,
    FIGURE_PERSISTENCE_ERROR_IMPORT,
    FIGURE_PERSISTENCE_ERROR_INTERNAL,
];

pub(super) const PLOT_CORE: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::Any,
    engine: BuiltinExampleEngine::Default,
    compiler: &[],
    runtime: &[BuiltinRuntimeCapability::PlotCore],
    toolchain: &[],
};

pub(super) const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};

pub(super) const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: runmat_types::ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

pub(super) const SAVE_EFFECTS: &[runmat_types::EffectKind] = &[
    runmat_types::EffectKind::EnvironmentRead,
    runmat_types::EffectKind::FilesystemWrite,
    runmat_types::EffectKind::UserInterface,
    runmat_types::EffectKind::HostCallback,
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];

pub(super) const LOAD_EFFECTS: &[runmat_types::EffectKind] = &[
    runmat_types::EffectKind::EnvironmentRead,
    runmat_types::EffectKind::FilesystemRead,
    runmat_types::EffectKind::UserInterface,
    runmat_types::EffectKind::HostCallback,
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];

pub(super) fn infer_fixed(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
    outputs: Vec<runmat_types::ValueFact>,
    minimum: usize,
    maximum: usize,
) -> runmat_types::CallInference {
    use crate::catalog::inference::{argument_error, finish_fixed_outputs};
    let mut diagnostics = Vec::new();
    if !(minimum..=maximum).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FIGURE-PERSISTENCE-ARITY",
            format!(
                "{} accepts {minimum} to {maximum} inputs",
                entry.identity.name
            ),
            request.arguments.len().min(maximum),
        ));
    }
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}

pub(super) fn handle_fact() -> runmat_types::ValueFact {
    runmat_types::ValueFact::proven(
        runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::Double,
            domain: runmat_types::NumericDomain::Real,
        }),
        runmat_types::ShapeFact::from(vec![Some(1), None]),
        runmat_types::StorageFact::Dense,
    )
}

pub(super) fn status_fact() -> runmat_types::ValueFact {
    runmat_types::ValueFact::scalar(runmat_types::ValueKindFact::Logical)
}
