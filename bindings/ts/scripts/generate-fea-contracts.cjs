#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const {splitTopLevel} = require("./fea-semantic-typegen/syntax.cjs");
const { ADAPTER: WIRE_INTEGER_ADAPTER, renderAdaptedField, wideIntegerError } = require("./fea-semantic-typegen/wire-integers.cjs");

const bindingsRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(bindingsRoot, "../..");
const rustAnalysisContractsPath = path.resolve(
  repoRoot,
  "crates/runmat-runtime/src/analysis/contracts.rs"
);
const rustMaxwellRunOptionsPath = path.resolve(
  repoRoot,
  "crates/runmat-runtime/src/analysis/maxwell_run_options.rs"
);
const rustFeaProgressPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/fea/src/progress.rs"
);
const rustFeaSourceReportPath = path.resolve(
  repoRoot,
  "crates/runmat-fea-source/src/report.rs"
);
const rustFeaDiagnosticPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/language/src/diagnostic.rs"
);
const rustCoreExecutionPolicyPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/core/src/execution_policy.rs"
);
const rustCoreIdentityPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/core/src/study/identity.rs"
);
const rustCoreFormulationPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/core/src/study/formulation.rs"
);
const rustCoreOutputPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/core/src/study/output/quantity.rs"
);
const rustFieldLocationPath = path.resolve(repoRoot, "crates/runmat-analysis/core/src/study/output.rs");
const rustSignalPath = path.resolve(repoRoot, "crates/runmat-analysis/core/src/problem/signal.rs");
const rustExcitationPath = path.resolve(repoRoot, "crates/runmat-analysis/core/src/problem/excitation.rs");
const rustGeometryUnitsPath = path.resolve(
  repoRoot,
  "crates/runmat-geometry/core/src/model/units.rs"
);
const rustGeometryAuthoringPath = path.resolve(
  repoRoot,
  "crates/runmat-geometry/core/src/model/authoring.rs"
);
const rustMeshingRequestPath = path.resolve(
  repoRoot,
  "crates/runmat-meshing/core/src/contracts/canonical/request.rs"
);
const rustFeaContractsPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/fea/src/contracts/mod.rs"
);
const generatedRoot = path.resolve(bindingsRoot, "src/generated");
const outputPath = path.resolve(generatedRoot, "fea-analysis-contracts.ts");
const analysisConstants = [
  ["ANALYSIS_RUN_DATASET_SCHEMA_VERSION", "FEA_RUN_DATASET_SCHEMA_VERSION"],
  ["ANALYSIS_FIELD_BINDINGS_SCHEMA_VERSION", "FEA_FIELD_BINDINGS_SCHEMA_VERSION"],
  ["ANALYSIS_DIAGNOSTICS_SCHEMA_VERSION", "FEA_DIAGNOSTICS_SCHEMA_VERSION"],
  ["ANALYSIS_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION", "FEA_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION"],
  ["ANALYSIS_RUN_DATASET_KIND", "FEA_RUN_DATASET_KIND"],
  ["ANALYSIS_DATASET_ARTIFACT_KIND", "FEA_DATASET_ARTIFACT_KIND"],
  ["ANALYSIS_FIELD_BINDINGS_ARTIFACT_KIND", "FEA_FIELD_BINDINGS_ARTIFACT_KIND"],
  ["ANALYSIS_DIAGNOSTICS_ARTIFACT_KIND", "FEA_DIAGNOSTICS_ARTIFACT_KIND"],
  ["ANALYSIS_ARTIFACT_MANIFEST_KIND", "FEA_ARTIFACT_MANIFEST_KIND"],
  ["ANALYSIS_FIELD_DEFAULT_PAGE_SIZE", "FEA_FIELD_DEFAULT_PAGE_SIZE"],
  ["ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT", "FEA_FIELD_DEFAULT_MATERIALIZE_LIMIT"],
];
const analysisEnumTypes = [
  ["AnalysisExecutionMode", "FeaExecutionMode"],
  ["AnalysisExecutionReadiness", "FeaExecutionReadiness"],
  ["AnalysisDocumentKind", "FeaDocumentKind"],
  ["AnalysisFieldKind", "FeaFieldKind"],
  ["AnalysisFieldStorage", "FeaFieldStorage"],
  ["AnalysisFieldLocation", "FeaFieldLocation"],
  ["PrecisionMode", "FeaPrecisionMode"],
  ["PreconditionerMode", "FeaPreconditionerMode"],
  ["QualityPolicy", "FeaQualityPolicy"],
  ["PressureVelocityCoupling", "FeaPressureVelocityCoupling"],
];
const progressEnumTypes = [
  ["FeaProgressPhase", "FeaProgressPhase"],
  ["FeaProgressStatus", "FeaProgressStatus"],
];
const analysisStructTypes = [
  ["AnalysisExecutionDuration", "FeaExecutionDuration", { renameAll: "camelCase" }],
  ["AnalysisExecutionCandidate", "FeaExecutionCandidate", { renameAll: "camelCase" }],
  ["AnalysisExecutionChoices", "FeaExecutionChoices", { renameAll: "camelCase" }],
  ["AnalysisRuntimeCapabilities", "FeaCapabilities", { renameAll: "camelCase" }],
  ["AnalysisDocumentRunResult", "FeaRunResult", { renameAll: "camelCase" }],
  ["AnalysisFieldBinding", "FeaFieldBinding", { renameAll: "camelCase" }],
  ["AnalysisFieldRequestOptions", "FeaFieldRequestOptions", { renameAll: "camelCase" }],
  ["AnalysisFieldPageResult", "FeaFieldResult", { renameAll: "camelCase" }],
  ["AnalysisRunDatasetStudyRef", "FeaRunDatasetStudyRef", { renameAll: "camelCase" }],
  ["AnalysisRunDatasetFieldPagingPolicy", "FeaRunDatasetFieldPagingPolicy", { renameAll: "camelCase" }],
  ["AnalysisRunDatasetPayload", "FeaRunDatasetPayload", { renameAll: "camelCase" }],
  ["AnalysisFieldBindingsArtifactPayload", "FeaFieldBindingsArtifactPayload", { renameAll: "camelCase" }],
  ["AnalysisDiagnosticsArtifactPayload", "FeaDiagnosticsArtifactPayload", { renameAll: "camelCase" }],
  ["AnalysisObjectArtifactMetadata", "FeaObjectArtifactMetadata", { renameAll: "camelCase" }],
  ["AnalysisFieldPagingDescriptor", "FeaFieldPagingDescriptor", { renameAll: "camelCase" }],
  ["AnalysisFieldStorageRef", "FeaFieldStorageRef", { renameAll: "camelCase" }],
  ["AnalysisFieldDescriptor", "FeaFieldDescriptor", {}],
  ["AnalysisRunOptions", "FeaStructuralLinearOptions", {}],
  ["AnalysisModalRunOptions", "FeaStructuralModesOptions", {}],
  ["AnalysisAcousticRunOptions", "FeaAcousticHelmholtzOptions", {}],
  ["AnalysisThermalRunOptions", "FeaThermalDiffusionOptions", {}],
  ["AnalysisTransientRunOptions", "FeaStructuralTransientOptions", {}],
  ["AnalysisCfdRunOptions", "FeaIncompressibleFlowOptions", {}],
  ["CanonicalFluidPolicy", "FeaCanonicalFluidPolicy", {}],
  ["AndersonOnePolicy", "FeaAndersonOnePolicy", {}],
  ["FiniteVolumeProjectionPolicy", "FeaFiniteVolumeProjectionPolicy", {}],
  ["ContactEnforcementOptions", "FeaContactEnforcementOptions", {}],
  ["AnalysisNonlinearRunOptions", "FeaStructuralNonlinearOptions", {}],
  ["AnalysisMaxwellRunOptions", "FeaMaxwellHarmonicOptions", {}],
];
const progressStructTypes = [
  ["FeaProgressEvent", "FeaProgressEvent", { renameAll: "camelCase" }],
];
function readAnalysisConstant(source, rustName) {
  const match = source.match(
    new RegExp(`pub const ${rustName}:\\s*[^=]+?=\\s*(?<value>[^;]+);`)
  );
  if (!match?.groups?.value) {
    throw new Error(`Could not find Rust analysis constant ${rustName}.`);
  }
  const raw = match.groups.value.trim();
  if (/^"/.test(raw)) {
    return raw;
  }
  if (/^\d+$/.test(raw)) {
    return raw;
  }
  throw new Error(`Unsupported Rust analysis constant value for ${rustName}: ${raw}`);
}

function snakeToCamel(value) {
  return value.replace(/_([a-z])/g, (_, char) => char.toUpperCase());
}

function snakeToPascal(value) {
  const camel = snakeToCamel(value);
  return camel.charAt(0).toUpperCase() + camel.slice(1);
}

function rustTypeToTs(type) {
  const trimmed = type.trim();
  const array = trimmed.match(/^\[([\s\S]+);\s*(\d+)\]$/);
  if (array) return { type: `[${Array.from({length:Number(array[2])},()=>rustTypeToTs(array[1]).type).join(", ")}]`, nullable:false };
  const optionMatch = trimmed.match(/^Option<(.+)>$/);
  if (optionMatch) {
    return { type: `${rustTypeToTs(optionMatch[1]).type} | null`, nullable: true };
  }

  const vectorMatch = trimmed.match(/^Vec<(.+)>$/);
  if (vectorMatch) {
    return { type: `${rustTypeToTs(vectorMatch[1]).type}[]`, nullable: false };
  }

  switch (trimmed) {
    case "super::assessment::PublishedCaseAssessments":
      return { type: "PublishedCaseAssessments", nullable: false };
    case "super::assessment::PublishedStudyAssessment":
      return { type: "PublishedStudyAssessment", nullable: false };
    case "super::assessment::PublishedStudyReassessment":
      return { type: "PublishedStudyReassessment", nullable: false };
    case "runmat_execution::Digest":
      return { type: '`sha256:${string}`', nullable: false };
    case "String":
    case "&'static str":
      return { type: "string", nullable: false };
    case "JsonValue":
    case "serde_json::Value":
      return { type: "unknown", nullable: false };
    case "bool":
      return { type: "boolean", nullable: false };
    case "usize":
    case "u64":
      throw wideIntegerError(trimmed);
    case "u8":
    case "u16":
    case "u32":
    case "f64":
      return { type: "number", nullable: false };
    case "LengthUnit":
      return { type: "FeaGeometryUnits", nullable: false };
    case "ComputeBackend":
      return { type: "FeaComputeBackend", nullable: false };
    case "ElementOrder":
      return { type: "FeaElementOrder", nullable: false };
    case "FeaRegionSelector":
      return { type: "FeaRegionSelector", nullable: false };
    case "PhysicsFamily":
      return { type: "FeaPhysicsFamily", nullable: false };
    case "AnalysisProcedure":
      return { type: "FeaAnalysisProcedure", nullable: false };
    case "ProcedureStep":
      return { type: "FeaProcedureStep", nullable: false };
    case "runmat_analysis_core::AnalysisFieldLayout":
      return { type: "FeaFieldLayout", nullable: false };
    case "ProcedureStepId":
      return { type: "string", nullable: false };
    case "AnalysisFieldPageValues":
      return {
        type: '{ numericRepresentation: "real" | "complex_interleaved"; values: number[] }',
        nullable: false,
      };
    case "runmat_analysis_fea::results::field::RecoveredFieldSample":
      return {
        type: '"final" | { domain: "time_seconds" | "mode_index" | "frequency_hertz" | "load_factor" | "study_parameter_value"; coordinate: number }',
        nullable: false,
      };
    case "runmat_analysis_fea::results::field::RecoveredFieldNumericRepresentation":
      return { type: '"real" | "complex_interleaved"', nullable: false };
    case "runmat_analysis_core::FieldStatistic":
      return {
        type: '"component_magnitude_envelope" | "root_mean_square" | "variance"',
        nullable: false,
      };
    case "runmat_analysis_core::SpectrumDirection":
      return {
        type: "{ frame: FeaCoordinateFrame; vector: [number, number, number] }",
        nullable: false,
      };
    case "AnalysisStructuralTimeStepPolicy":
      return { type: "FeaStructuralTimeStepPolicy", nullable: false };
    case "runmat_analysis_fea::ModalTruncationPolicy":
      return { type: "FeaModalTruncationPolicy", nullable: false };
    case "runmat_analysis_fea::ModalConvergencePolicy":
      return { type: "FeaModalConvergencePolicy", nullable: false };
    case "runmat_analysis_fea::physics::cfd::canonical::CanonicalFluidPolicy":
      return { type: "FeaCanonicalFluidPolicy", nullable: false };
    case "PressureVelocityCoupling":
      return { type: "FeaPressureVelocityCoupling", nullable: false };
    case "crate::solve::fixed_point::AndersonOnePolicy":
      return { type: "FeaAndersonOnePolicy", nullable: false };
    case "runmat_analysis_fea::topology::FiniteVolumeProjectionPolicy":
      return { type: "FeaFiniteVolumeProjectionPolicy", nullable: false };
    default:
      if (/^BTreeMap<String,\s*String>$/.test(trimmed)) {
        return { type: "Record<string, string>", nullable: false };
      }
      for (const [rustName, tsName] of analysisStructTypes) {
        if (trimmed === rustName) {
          return { type: tsName, nullable: false };
        }
      }
      for (const [rustName, tsName] of analysisEnumTypes) {
        if (trimmed === rustName) {
          return { type: tsName, nullable: false };
        }
      }
      for (const [rustName, tsName] of progressStructTypes) {
        if (trimmed === rustName) {
          return { type: tsName, nullable: false };
        }
      }
      for (const [rustName, tsName] of progressEnumTypes) {
        if (trimmed === rustName) {
          return { type: tsName, nullable: false };
        }
      }
      throw new Error(`Unsupported Rust type in FEA study contract: ${trimmed}`);
  }
}

function readEnumBody(source, name) {
  const match = source.match(new RegExp(`pub enum ${name} \\{(?<body>[\\s\\S]*?)\\n\\}`));
  if (!match?.groups?.body) {
    throw new Error(`Could not find Rust enum ${name}.`);
  }
  return match.groups.body;
}

function rustVariantToSnake(value) {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2")
    .toLowerCase();
}

function readEnumValues(source, name) {
  const body = readEnumBody(source, name);
  const values = [];
  for (const rawLine of body.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#[") || line.startsWith("//")) {
      continue;
    }
    const variantMatch = line.match(/^([A-Za-z0-9_]+),$/);
    if (!variantMatch) {
      throw new Error(`Could not parse enum variant in ${name}: ${line}`);
    }
    values.push(rustVariantToSnake(variantMatch[1]));
  }
  if (values.length === 0) {
    throw new Error(`Rust enum ${name} did not contain variants.`);
  }
  return values;
}

function readTupleEnumValues(source, name) {
  const body = readEnumBody(source, name);
  const values = [];
  for (const match of body.matchAll(/^\s*([A-Za-z0-9_]+)\([^\n]+\),\s*$/gm)) {
    values.push(rustVariantToSnake(match[1]));
  }
  if (values.length === 0) {
    throw new Error(`Rust enum ${name} did not contain tuple variants.`);
  }
  return values;
}

function readStringIdStructEnumValues(source, name) {
  const body = readEnumBody(source, name);
  const values = [];
  for (const rawLine of body.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#[")) {
      continue;
    }
    const variantMatch = line.match(/^([A-Za-z0-9_]+)\s*\{\s*id:\s*String\s*\},$/);
    if (!variantMatch) {
      throw new Error(`Could not parse string-id enum variant in ${name}: ${line}`);
    }
    values.push(rustVariantToSnake(variantMatch[1]));
  }
  if (values.length === 0) {
    throw new Error(`Rust enum ${name} did not contain string-id variants.`);
  }
  return values;
}

function externallyTaggedUnion(name, variants, bodyName) {
  return `export type ${name} = ${variants.map((variant) => `{ ${variant}: ${bodyName} }`).join(" | ")};`;
}

function formatTypeUnion(name, values) {
  return `export type ${name} = ${values.map((value) => JSON.stringify(value)).join(" | ")};`;
}

function readStructBody(source, name) {
  const match = source.match(new RegExp(`pub struct ${name} \\{(?<body>[\\s\\S]*?)\\n\\}`));
  if (!match?.groups?.body) {
    throw new Error(`Could not find Rust struct ${name}.`);
  }
  return match.groups.body;
}

const unboundedWideIntegers = [];

function readStructFields(source, name, options = {}) {
  const body = readStructBody(source, name);
  const fields = [];
  const lines = [];
  let pendingField = "";
  for (const rawLine of body.split("\n")) {
    const line = rawLine.trim();
    if (line.startsWith("//")) continue;
    if (pendingField) {
      pendingField += ` ${line}`;
      if (line.endsWith(",")) {
        lines.push(pendingField);
        pendingField = "";
      }
    } else if (line.startsWith("pub ") && !line.endsWith(",")) {
      pendingField = line;
    } else {
      lines.push(line);
    }
  }
  if (pendingField) {
    throw new Error(`Unterminated field declaration in ${name}: ${pendingField}`);
  }
  let rename = null;
  let skipIfNone = false;
  let canonicalScalar = false;
  let canonicalStorage = false;
  let wireAdapter = null;
  const renameAll = options.renameAll ?? null;

  for (const line of lines) {
    if (!line) {
      continue;
    }
    const renameMatch = line.match(/^#\[serde\(rename = "([^"]+)"\)\]$/);
    if (renameMatch) {
      rename = renameMatch[1];
      continue;
    }
    if (line.startsWith('#[serde(') && line.includes('skip_serializing_if = "Option::is_none"')) {
      skipIfNone = true;
      continue;
    }
    if (line === '#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]') {
      skipIfNone = true;
      continue;
    }
    if (['#[serde(with = "runmat_types::physical::canonical_scalar")]', '#[serde(with = "runmat_types::physical::canonical_scalar::optional")]'].includes(line)) { canonicalScalar = true; continue; }
    if (line === '#[serde(with = "runmat_types::physical::canonical_storage")]') { canonicalStorage = true; continue; }
    const wireMatch = line.match(/^#\[serde\(with = "(runmat_types::wire_integer(?:::option|::vec)?)"\)\]$/);
    if (wireMatch) { wireAdapter = wireMatch[1]; continue; }
    if (line.startsWith("#[")) {
      continue;
    }

    const fieldMatch = line.match(/^pub\s+([a-zA-Z0-9_]+):\s+(.+),$/);
    if (!fieldMatch) {
      throw new Error(`Could not parse field line in ${name}: ${line}`);
    }
    const rustName = fieldMatch[1];
    const rustType = fieldMatch[2];
    const storageMatch = canonicalStorage ? rustType.match(/^(?:[A-Za-z_][A-Za-z0-9_]*::)*Quantity<([\s\S]+)>$/) : null;
    if (canonicalStorage && !storageMatch) throw new Error(`Canonical storage requires shared quantity shape: ${rustType}`);
    const storageType = storageMatch ? splitTopLevel(storageMatch[1])[1] : null;
    // Wide integers are bounded to 2^53 - 1 by the adapter; this file declares no guards, so they stay `number`.
    let tsType;
    try {
    tsType = wireAdapter ? { type: renderAdaptedField(rustType, wireAdapter).replace(/^Wire[UI]64/, "number") + (wireAdapter.endsWith("::option") ? " | null" : ""), nullable: wireAdapter.endsWith("::option") } : canonicalStorage ? rustTypeToTs(storageType) : canonicalScalar ? (/^Option\s*</.test(rustType) ? { type: "number | null", nullable: true } : { type: "number", nullable: false }) : rustTypeToTs(rustType);
    } catch (error) {
      unboundedWideIntegers.push(`${name}.${rustName}: ${rustType}`);
      continue;
    }
    const defaultName = renameAll === "camelCase" ? snakeToCamel(rustName) : rustName;
    fields.push({
      name: rename ?? defaultName,
      optional: skipIfNone,
      type: skipIfNone && tsType.nullable ? tsType.type.replace(/ \| null$/, "") : tsType.type,
    });
    rename = null;
    skipIfNone = false;
    canonicalScalar = false;
    canonicalStorage = false;
    wireAdapter = null;
  }

  return fields;
}

function formatInterface(name, fields) {
  return [
    `export interface ${name} {`,
    ...fields.map((field) => `  ${field.name}${field.optional ? "?" : ""}: ${field.type};`),
    "}",
  ].join("\n");
}

function main() {
  const analysisSource = fs.readFileSync(rustAnalysisContractsPath, "utf8")
    + "\n" + fs.readFileSync(path.resolve(repoRoot, "crates/runmat-runtime/src/analysis/contracts/nonlinear_run_options.rs"), "utf8")
    + "\n" + fs.readFileSync(rustMaxwellRunOptionsPath, "utf8")
    + "\n" + fs.readFileSync(path.resolve(repoRoot, "crates/runmat-analysis/fea/src/physics/cfd/canonical.rs"), "utf8")
    + "\n" + fs.readFileSync(path.resolve(repoRoot, "crates/runmat-analysis/fea/src/solve/fixed_point.rs"), "utf8")
    + "\n" + fs.readFileSync(path.resolve(repoRoot, "crates/runmat-analysis/fea/src/topology/finite_volume.rs"), "utf8");
  const progressSource = fs.readFileSync(rustFeaProgressPath, "utf8");
  const sourceReportSource = fs.readFileSync(rustFeaSourceReportPath, "utf8");
  const diagnosticSource = fs.readFileSync(rustFeaDiagnosticPath, "utf8");
  const coreIdentitySource = fs.readFileSync(rustCoreIdentityPath, "utf8");
  const coreExecutionPolicySource = fs.readFileSync(rustCoreExecutionPolicyPath, "utf8");
  const coreFormulationSource = fs.readFileSync(rustCoreFormulationPath, "utf8");
  const coreOutputSource = fs.readFileSync(rustCoreOutputPath, "utf8");
  const fieldLocationSource = fs.readFileSync(rustFieldLocationPath, "utf8");
  const signalSource = fs.readFileSync(rustSignalPath, "utf8");
  const excitationSource = fs.readFileSync(rustExcitationPath, "utf8");
  const geometryUnitsSource = fs.readFileSync(rustGeometryUnitsPath, "utf8");
  const geometryAuthoringSource = fs.readFileSync(rustGeometryAuthoringPath, "utf8");
  const meshingRequestSource = fs.readFileSync(rustMeshingRequestPath, "utf8");
  const feaContractsSource = fs.readFileSync(rustFeaContractsPath, "utf8");
  const constants = analysisConstants.map(([rustName, tsName]) => ({
    tsName,
    value: readAnalysisConstant(analysisSource, rustName),
  }));
  const analysisEnumValues = analysisEnumTypes.map(([rustName, tsName]) => ({
    rustName,
    tsName,
    values: readEnumValues(analysisSource, rustName),
  }));
  const enumTypes = analysisEnumValues.map(({ tsName, values }) => formatTypeUnion(tsName, values));
  const progressEnums = progressEnumTypes.map(([rustName, tsName]) =>
    formatTypeUnion(tsName, readEnumValues(progressSource, rustName))
  );
  const sourceCheckContracts = [
    formatTypeUnion("FeaCheckStatus", readEnumValues(sourceReportSource, "FeaCheckStatus")),
    formatTypeUnion(
      "FeaSourceAvailability",
      readEnumValues(diagnosticSource, "SourceAvailability").map(snakeToPascal)
    ),
    formatTypeUnion(
      "FeaDiagnosticSeverity",
      readEnumValues(diagnosticSource, "FeaDiagnosticSeverity").map(snakeToPascal)
    ),
    formatTypeUnion(
      "FeaDiagnosticPhase",
      readEnumValues(diagnosticSource, "FeaDiagnosticPhase").map(snakeToPascal)
    ),
    `export interface FeaSourceLocation {
  path: string;
  span: { start: number; end: number };
}`,
    `export interface FeaRelatedDiagnostic {
  message: string;
  location: FeaSourceLocation;
}`,
    `export interface FeaDiagnostic {
  code: string;
  message: string;
  severity: FeaDiagnosticSeverity;
  phase: FeaDiagnosticPhase;
  location: FeaSourceLocation | null;
  label?: string;
  related: FeaRelatedDiagnostic[];
  availability: FeaSourceAvailability;
}`,
    `export type { FeaSourceCheckResult } from "./fea-semantic-contracts.js";`,
  ];
  const authoringEnums = [
    `export type FeaGeometryUnits = import("./fea-semantic-contracts.js").LengthUnit;`,
    formatTypeUnion("FeaComputeBackend", readEnumValues(feaContractsSource, "ComputeBackend")),
    formatTypeUnion(
      "FeaAcceleratorAdmissionPolicy",
      readEnumValues(coreExecutionPolicySource, "AcceleratorAdmissionPolicy")
    ),
    formatTypeUnion("FeaElementOrder", readEnumValues(meshingRequestSource, "ElementOrder")),
    formatTypeUnion("FeaAnalysisProcedure", readEnumValues(coreIdentitySource, "AnalysisProcedure")),
  ];
  const coupledFamilies = readEnumValues(coreIdentitySource, "CoupledPhysicsFamily");
  const observationQuantities = readEnumValues(coreOutputSource, "ObservationQuantity");
  const physicsFamily = `export type FeaPhysicsFamily = "structural" | "thermal" | "electromagnetic" | "acoustic" | "fluid" | { coupled: ${coupledFamilies.map((value) => JSON.stringify(value)).join(" | ")} };`;
  const procedureStep = `export type FeaIndependentDomain =
  | "static"
  | { time_seconds: { origin: number; start: number; end: number } }
  | { frequency_hertz: number[] }
  | { load_factor: { start: number; end: number } }
  | { eigenvalue: { requested_count: number } }
  | { study_parameter: { parameter_id: string; canonical_values: number[] } };
export type FeaStepObjectRef =
  | { constraint: string } | { boundary: string } | { connector: string }
  | { interface: string } | { excitation: string } | { initial_condition: string }
  | { material_parameter: string };
export type FeaActivationLifecycle = "activate"
  | { continue: { from_step: string } }
  | { modify: { prior_step: string; prior_object: FeaStepObjectRef } }
  | { deactivate: { from_step: string } };
export interface FeaStepObjectActivation { object: FeaStepObjectRef; lifecycle: FeaActivationLifecycle; }
export type FeaStateKind = "structural_primary" | "structural_kinematics" | "structural_stress_strain" | "geometric_stiffness" | "material_history" | "contact_history" | "thermal" | "electromagnetic_static" | "electromagnetic_complex" | "acoustic_complex" | "fluid" | "moving_mesh" | "modal_basis" | "coupled_interface" | "integrator";
export type FeaStateSource = "initial_conditions" | { upstream: { producer_step: string; state_id: string } };
export type FeaStateUse = "physical_initial_state" | "linearization_point" | "prestress" | "initial_guess" | "modal_basis" | "coupled_interface" | "restart";
export type FeaInitialStateConflictPolicy = "reject" | "replace_initial_condition" | "augment_disjoint_quantities";
export interface FeaStateInput { input_id: string; required_kind: FeaStateKind; source: FeaStateSource; use_as: FeaStateUse; conflict: FeaInitialStateConflictPolicy; }
export interface FeaStateOutput { state_id: string; kind: FeaStateKind; completeness: "complete" | "checkpoint" | "partial"; persistence: "canonical" | "until_dependents_complete" | "checkpoint_only"; }
export type FeaRestartPolicy = "fresh" | { continue_from_completed_state: { producer_step: string; state_id: string } } | { resume_checkpoint: { checkpoint_id: string; required_state_digest: string } };
export type FeaObservationQuantity = ${observationQuantities.map((value) => JSON.stringify(value)).join(" | ")};
export type FeaSamplingPolicy = "final" | "every_accepted_increment" | { time_seconds: number[] } | { time_interval_seconds: { start: number; end: number; interval: number } } | { frequency_hertz: number[] } | { frequency_interval_hertz: { start: number; end: number; interval: number } } | { load_factors: number[] } | { load_factor_interval: { start: number; end: number; interval: number } } | { mode_indices: number[] } | { event: "first_yield" | "contact_status_change" | "peak_response" | "convergence_failure" | "accuracy_criterion_crossing" } | { adaptive: { minimum_interval: number; maximum_samples: number; relative_change: number } };
export type FeaRetentionPolicy = "canonical" | "until_dependents_complete" | { latest_samples: { count: number } } | { explicit_samples: { maximum_count: number } };
export interface FeaObservationSpec {
  observation_id: string; quantity: FeaObservationQuantity;
  value_shape: FeaObservationValueShape;
  target: FeaObservationTarget; source_location: FeaFieldLocationContract; output_location: FeaFieldLocationContract;
  recovery: "native" | "extrapolate" | "volume_weighted_average" | "area_weighted_average" | "conservative_projection" | "shape_function_interpolation";
  component: "all" | "x" | "y" | "z" | "normal" | "tangent1" | "tangent2" | "trace" | "hydrostatic" | "deviatoric" | "von_mises" | "maximum_principal" | "minimum_principal" | "magnitude";
  frame: FeaCoordinateFrame; sign_convention: "native" | "tension_positive" | "compression_positive" | "outward_positive" | "inward_positive";
  complex_representation: FeaComplexRepresentation;
  sampling: FeaSamplingPolicy; reduction: FeaReductionPolicy | null; storage: FeaObservationStoragePolicy;
  purpose: "publication" | "solver_control" | "visualization" | "downstream_state"; cost: FeaObservationCostEstimate;
}
${formatTypeUnion("FeaFieldLocationContract", readEnumValues(fieldLocationSource, "FieldLocation"))}
export type FeaObservationValueShape = "scalar" | "vector3" | "symmetric_tensor3" | { tensor: { extents: number[] } };
${formatTypeUnion("FeaComplexRepresentation", readEnumValues(fieldLocationSource, "ComplexRepresentation"))}
export type FeaFieldOrdinal = { kind: "single" } | { kind: "step"; index: number } | { kind: "mode"; index: number };
export interface FeaFieldLayout { quantity: FeaObservationQuantity; location: FeaFieldLocationContract; value_shape: FeaObservationValueShape; representation?: FeaComplexRepresentation; ordinal: FeaFieldOrdinal; }
export type FeaObservationTarget = { spatial: FeaExcitationTarget } | { point: { position: FeaPosition } } | { path: { positions: FeaPosition[] } } | { connector: { connector_id: string } } | { constraint: { constraint_id: string } } | "whole_model";
export interface FeaReductionPolicy { spatial: "none" | "integral" | "average" | "root_mean_square" | "euclidean_norm" | "minimum" | "maximum" | "resultant" | "moment"; series: "none" | "minimum" | "maximum" | "absolute_maximum" | "root_mean_square" | "mean" | "variance" | "envelope"; extrema_ties: "first_canonical" | "all_locations"; }
export interface FeaObservationStoragePolicy { precision: FeaSignalPrecision; compression: "lossless" | { error_bounded: { maximum_relative_error: number } } | "none"; retention: FeaRetentionPolicy; partial_access: "manifest_and_chunks" | "whole_artifact"; }
export interface FeaObservationCostEstimate { operation_count: number; peak_resident_bytes: number; stored_bytes: number; transfer_bytes: number; collective_count: number; }
export type FeaObservationRequest = { field: FeaObservationSpec } | { history: FeaObservationSpec } | { probe: FeaObservationSpec } | { reduction: FeaObservationSpec } | { diagnostic: FeaDiagnosticObservation } | { monitor: FeaMonitorObservation };
export interface FeaDiagnosticObservation { observation_id: string; diagnostic: "convergence" | "balance" | "constraint" | "contact" | "adaptation" | "resource" | "execution_equivalence"; sampling: FeaSamplingPolicy; retention: FeaRetentionPolicy; }
export interface FeaMonitorObservation { observation_id: string; source_observation_id: string; cadence: { accepted_increments: { interval: number } } | { wall_seconds: { interval: number } }; stop: FeaMonitorStopCondition | null; }
export type FeaMonitorStopCondition = { less_than: { threshold: FeaObservationThreshold } } | { greater_than: { threshold: FeaObservationThreshold } } | { absolute_change_below: { threshold: FeaObservationThreshold; consecutive: number } };
export interface FeaObservationThreshold { quantity: FeaObservationQuantity; canonical_value: number; }
export type FeaAccuracyTarget = { observation: string } | { quantity: FeaObservationQuantity } | "global_equilibrium" | "interface_closure" | "execution_equivalence";
export type FeaAccuracyTolerance = { relative: number } | { absolute: { quantity: FeaObservationQuantity; canonical_value: number } };
export interface FeaAccuracyCriterion { criterion_id: string; target: FeaAccuracyTarget; contribution: FeaErrorContributionKind; tolerance: FeaAccuracyTolerance; evidence: "a_posteriori_estimator" | "convergence_study" | "conservation_balance" | "residual_bound" | "reference_comparison" | "statistical_confidence" | "transfer_bound"; mandatory: boolean; }
export type FeaErrorContributionKind = "geometry_admission" | "boundary_approximation" | "spatial_discretization" | "algebraic" | "nonlinear" | "temporal" | "frequency_sampling" | "modal_truncation" | "stochastic" | "coupling" | "interface_projection" | "field_transfer" | "signal_interpolation" | "recovery" | "observation_sampling" | "precision" | "execution_equivalence";
export interface FeaAccuracyPlan { accuracy_plan_id: string; criteria: FeaAccuracyCriterion[]; unavailable_evidence: "fail" | "mark_indeterminate"; exceeded_limit: "refine" | "tighten_solver" | "extend_sampling" | "fail"; equivalence: "deterministic" | { relative_tolerance: number } | { absolute_tolerance: number }; resources: { maximum_refinement_cycles: number; maximum_solver_tightenings: number; maximum_samples: number; maximum_wall_time_milliseconds: number; maximum_resident_bytes: number | null }; }
export interface FeaProcedureStep {
  step_id: string; family: FeaPhysicsFamily; procedure: FeaAnalysisProcedure; solution_space: "direct" | "modal_reduced";
  excitation_case_selection: "all" | { only: string[] };
  independent_domain: FeaIndependentDomain; activations: FeaStepObjectActivation[]; state_inputs: FeaStateInput[];
  state_outputs: FeaStateOutput[]; observations: FeaObservationRequest[]; accuracy: FeaAccuracyPlan; restart: FeaRestartPolicy;
}
export interface FeaSequenceDependency { predecessor: string; successor: string; reason: "state_transfer" | "authored_ordering" | "shared_interface" | "adaptation"; }
export interface FeaStateTransition { transition_id: string; producer_step: string; consumer_step: string; state_id: string; method: "reuse" | "conservative_projection" | "interpolation" | "linearize" | "derive_operator_contribution" | "initial_guess_only"; topology: "require_identical" | "project_with_evidence" | "adapted_mesh_lineage"; reference: "preserve" | "receiving_step_origin" | "explicit_linearization_point"; }
export interface FeaSequenceBranch { branch_id: string; producer_step: string; state_ids: string[]; consumer_steps: string[]; }`;
  const signalQuantityTypes = {
    displacement: "{ meters: number }", rotation: "{ radians: number }", velocity: "{ meters_per_second: number }",
    acceleration: "{ meters_per_second_squared: number }", angular_velocity: "{ radians_per_second: number }",
    angular_acceleration: "{ radians_per_second_squared: number }", force: "{ newtons: number }",
    moment: "{ newton_meters: number }", traction: "{ pascals: number }", pressure: "{ pascals: number }",
    body_force_density: "{ newtons_per_cubic_meter: number }", temperature: "{ kelvin: number }",
    temperature_difference: "{ kelvin: number }",
    heat_rate: "{ watts: number }",
    heat_flux: "{ watts_per_square_meter: number }", heat_generation_density: "{ watts_per_cubic_meter: number }",
    electric_potential: "{ volts: number }", electric_current: "{ amperes: number }", electric_charge: "{ coulombs: number }",
    electric_current_density: "{ amperes_per_square_meter: number }", electric_charge_density: "{ coulombs_per_cubic_meter: number }",
    magnetic_vector_potential: "{ webers_per_meter: number }", mass_flow_rate: "{ kilograms_per_second: number }",
    volume_flow_rate: "{ cubic_meters_per_second: number }",
  };
  const signalInput = `export type FeaSignalInput = ${readTupleEnumValues(signalSource, "PhysicalSignal").map((variant) => {
    const quantity = signalQuantityTypes[variant];
    if (!quantity) throw new Error(`Missing TypeScript quantity contract for PhysicalSignal::${variant}`);
    return `{ ${variant}: FeaSignalBody<${quantity}> }`;
  }).join(" | ")};`;
  const canonicalInputs = [
    `export type FeaRegionSelector = ${readTupleEnumValues(geometryAuthoringSource, "GeometryRegionSelector")
      .map((variant) => `{ kind: ${JSON.stringify(variant)}; value: string }`)
      .concat('{ kind: "result" }')
      .join(" | ")};`,
    "export interface FeaRegionInput { region_id: string; selector: FeaRegionSelector; }",
    "export interface FeaRegionTarget {",
    "  region_id: string;",
    "  entity_dimension: \"point\" | \"edge\" | \"face\" | \"volume\";",
    "  expected_topology: \"geometry_entity\" | \"mesh_node\" | \"mesh_element\" | \"boundary_face\" | \"cell\";",
    "  cardinality: \"one\" | \"one_or_more\";",
    "}",
    "export interface FeaPosition { meters: [number, number, number]; }",
    "export type FeaCoordinateFrame = \"global\" | \"material\" | \"element\" | \"boundary_normal_tangent\" | { cartesian: { frame_id: string; origin: FeaPosition; x_axis: [number, number, number]; y_axis: [number, number, number] } } | { cylindrical: { frame_id: string; origin: FeaPosition; axis: [number, number, number]; radial_reference: [number, number, number] } } | { spherical: { frame_id: string; origin: FeaPosition; polar_axis: [number, number, number]; azimuth_reference: [number, number, number] } };",
    "export type FeaPrescribedValue<Q> = { constant: Q } | { signal: string };",
    "export interface FeaPairedConstraint { primary: FeaRegionTarget; secondary: FeaRegionTarget; pairing: \"matching\" | { geometric_tolerance: { meters: number } }; }",
    "export type FeaStructuralConstraintKind = { fixed: { target: FeaRegionTarget } } | { prescribed_translations: { target: FeaRegionTarget; values: { dof: \"ux\" | \"uy\" | \"uz\"; value: FeaPrescribedValue<{ meters: number }> }[] } } | { prescribed_rotations: { target: FeaRegionTarget; values: { dof: \"rx\" | \"ry\" | \"rz\"; value: FeaPrescribedValue<{ radians: number }> }[] } } | { normal_displacement: { target: FeaRegionTarget; value: FeaPrescribedValue<{ meters: number }> } } | { symmetry: { target: FeaRegionTarget } } | { antisymmetry: { target: FeaRegionTarget } } | { rigid_coupling: { target: FeaRegionTarget; reference_point: { reference_point_id: string; position: FeaPosition } } } | { distributing_coupling: { target: FeaRegionTarget; reference_point: { reference_point_id: string; position: FeaPosition } } } | { tie: FeaPairedConstraint } | { periodic: FeaPairedConstraint } | { equation: { terms: { target: FeaRegionTarget; dof: \"ux\" | \"uy\" | \"uz\"; coefficient: number }[]; right_hand_side: { meters: number } } };",
    "export type FeaConstraintInput = { structural: { constraint_id: string; frame: FeaCoordinateFrame; kind: FeaStructuralConstraintKind } } | { thermal: { constraint_id: string; kind: { prescribed: { target: FeaRegionTarget; value: FeaPrescribedValue<{ kelvin: number }> } } | { periodic: FeaPairedConstraint } | { tie: FeaPairedConstraint } | { affine_jump: { pair: FeaPairedConstraint; temperature_jump: FeaPrescribedValue<{ kelvin: number }> } } } } | { electrical: { constraint_id: string; kind: { prescribed_potential: { target: FeaRegionTarget; value: FeaPrescribedValue<{ volts: number }> } } | { equipotential: { target: FeaRegionTarget } } | { floating_conductor: { target: FeaRegionTarget } } | { integral_current: { target: FeaRegionTarget; value: FeaPrescribedValue<{ amperes: number }> } } | { periodic: FeaPairedConstraint } | { potential_reference: { target: FeaRegionTarget } } } } | { magnetic: { constraint_id: string; kind: { prescribed_vector_potential: { target: FeaRegionTarget; value: FeaPrescribedValue<{ webers_per_meter: number }> } } | { periodic: FeaPairedConstraint } | { vector_potential_ground: { target: FeaRegionTarget } } | { coulomb_gauge: { target: FeaRegionTarget } } } } | { acoustic: { constraint_id: string; kind: { prescribed_pressure: { target: FeaRegionTarget; value: FeaPrescribedValue<{ pascals: number }> } } | { periodic: FeaPairedConstraint } } } | { fluid: { constraint_id: string; kind: { prescribed_velocity: { target: FeaRegionTarget; value: FeaPrescribedValue<[{ meters_per_second: number }, { meters_per_second: number }, { meters_per_second: number }]> } } | { moving_wall: { target: FeaRegionTarget; velocity: FeaPrescribedValue<[{ meters_per_second: number }, { meters_per_second: number }, { meters_per_second: number }]> } } | { no_slip_wall: { target: FeaRegionTarget } } | { slip_wall: { target: FeaRegionTarget } } | { symmetry: { target: FeaRegionTarget } } | { periodic: FeaPairedConstraint } | { pressure_reference: { target: FeaRegionTarget; value: FeaPrescribedValue<{ pascals: number }> } } } };",
    "export type FeaBoundaryOperator = { structural_pressure: { value: FeaPrescribedValue<{ pascals: number }> } } | { thermal_heat_flux: { value: FeaPrescribedValue<{ watts_per_square_meter: number }> } } | { thermal_convection: { ambient_temperature: FeaPrescribedValue<{ kelvin: number }>; coefficient: { watts_per_square_meter_kelvin: number } } } | { thermal_radiation: { ambient_temperature: FeaPrescribedValue<{ kelvin: number }>; emissivity: { fraction: number } } } | \"magnetic_insulation\" | \"acoustic_rigid_wall\" | \"acoustic_radiation\" | { acoustic_impedance: { impedance: { pascal_seconds_per_meter: number } } } | { fluid_outlet_pressure: { value: FeaPrescribedValue<{ pascals: number }> } };",
    "export interface FeaBoundaryInput { boundary_id: string; target: FeaRegionTarget; operator: FeaBoundaryOperator; }",
    "export type FeaConnectorKind = { translational_spring: { stiffness: { newtons_per_meter: number } } } | { translational_dashpot: { damping: { newton_seconds_per_meter: number } } } | { elastic_foundation: { stiffness: { newtons_per_meter: number } } };",
    "export interface FeaConnectorInput { connector_id: string; primary: FeaRegionTarget; secondary: FeaRegionTarget | null; kind: FeaConnectorKind; }",
    "export type FeaInterfaceKind = { contact: { friction: \"frictionless\" | { coulomb: { coefficient: { ratio: number } } } } } | { fluid_structure: { kinematics: \"no_slip\" } } | { conjugate_heat_transfer: { transfer: \"perfect\" | { conductance: { watts_per_square_meter_kelvin: number } } | { resistance: { square_meter_kelvin_per_watt: number } } } };",
    "export interface FeaInterfaceInput { interface_id: string; primary: FeaRegionTarget; secondary: FeaRegionTarget; kind: FeaInterfaceKind; }",
    "export interface FeaMaterialInput { material_id: string; name?: string; mechanical: { youngs_modulus_pa: number; poisson_ratio: number; density_kg_per_m3?: number }; thermal?: { reference_temperature_k?: number; modulus_temp_coeff_per_k?: number; conductivity_w_per_mk?: number; specific_heat_j_per_kgk?: number; expansion_coefficient_per_k?: number }; acoustic?: { density_kg_per_m3?: number; speed_of_sound_m_per_s?: number; damping_ratio?: number }; electrical?: { reference_temperature_k?: number; conductivity_s_per_m?: number; resistive_heating_coefficient?: number; relative_permittivity?: number; relative_permeability?: number; conductivity_frequency_response?: { frequency_hz: number; conductivity_scale: number; dispersive_loss_scale?: number; relative_permittivity_scale?: number; relative_permeability_scale?: number }[] }; plastic?: { yield_strain: number; hardening_modulus_ratio: number; saturation_exponent: number }; }",
    "export interface FeaMaterialAssignmentInput { region: string; material: string; expected_material?: string; confidence?: \"verified\" | \"probable\" | \"inferred\"; }",
    "export type FeaSignalShape = \"scalar\" | \"vector3\" | { tensor: number[] } | { topology_field: { topology_id: string; location: FeaFieldLocationContract; component_count: number } };",
    "export type FeaSignalValue<Q> = { scalar: Q } | { vector3: [Q, Q, Q] } | { tensor: { shape: number[]; values: Q[] } };",
    "export type FeaSignalOrdinate<Q> = { real: FeaSignalValue<Q> } | { complex: { real: FeaSignalValue<Q>; imaginary: FeaSignalValue<Q> } };",
    "export type FeaTabulatedSignalDomain = \"time_seconds\" | \"frequency_hertz\" | { study_parameter: { parameter_id: string } };",
    "export type FeaSignalPointContinuity = \"left\" | \"right\";",
    "export type FeaSignalKnotValue<Q> = { continuous: FeaSignalOrdinate<Q> } | { jump: { left_limit: FeaSignalOrdinate<Q>; right_limit: FeaSignalOrdinate<Q>; continuity: FeaSignalPointContinuity } };",
    "export interface FeaSignalKnot<Q> { abscissa: number; value: FeaSignalKnotValue<Q>; }",
    "export interface FeaSignalArtifact { artifact_id: string; content_digest: string; schema: string; domain: \"constant\" | { time_seconds: { start: number; end: number } } | { frequency_hertz: { start: number; end: number } } | { study_parameter: { parameter_id: string; start: number; end: number } }; topology_digest: string | null; sample_count: number; }",
    "export type FeaGeneratorArgument = { boolean: boolean } | { signed_integer: number } | { unsigned_integer: number } | { real: number } | { text: string } | { real_vector: number[] } | { artifact: FeaSignalArtifact };",
    "export interface FeaSignalGenerator { qualified_function: string; arguments: FeaGeneratorArgument[]; domain: \"time_seconds\" | \"frequency_hertz\" | { study_parameter: { parameter_id: string } }; sampling: { uniform: { start: number; end: number; sample_count: number } } | { adaptive: { start: number; end: number; max_absolute_error: number; max_samples: number } } | { canonical_points: number[] }; effects: \"pure\" | \"declared_immutable_artifacts\" | { seeded_random: { seed: number; distribution: \"uniform\" | \"normal\" | \"log_normal\" } }; }",
    "export type FeaSignalRepresentation<Q> = { constant: FeaSignalValue<Q> } | { tabulated: { domain: FeaTabulatedSignalDomain; knots: FeaSignalKnot<Q>[]; interpolation: \"left_endpoint\" | \"right_endpoint\" | \"linear\"; extrapolation: \"reject\" | \"hold_endpoint\" | \"zero\" } } | { waveform: { kind: \"step\" | \"ramp\" | \"sine\" | \"half_sine_pulse\" | \"triangular_pulse\" | \"rectangular_pulse\"; amplitude: FeaSignalValue<Q>; offset: FeaSignalValue<Q> | null; start_seconds: number; duration_seconds: number | null; frequency_hz: number | null; phase_radians: number; duty_cycle: number | null } } | { artifact: FeaSignalArtifact } | { generator: FeaSignalGenerator };",
    "export type FeaSignalPrecision = \"real32\" | \"real64\" | \"complex64\" | \"complex128\";",
    "export interface FeaSignalBody<Q> { signal_id: string; shape: FeaSignalShape; frame: FeaCoordinateFrame | null; representation: FeaSignalRepresentation<Q>; precision: FeaSignalPrecision; evidence: { max_absolute_error: number | null; max_relative_error: number | null; derivative_max_absolute_error: number | null; sample_count: number }; provenance: \"authored\" | { imported: { source_digest: string } } | { generated: { project_revision: string; callable_id: string; frozen_input_digest: string } }; }",
    signalInput,
    "export type FeaExcitationTarget = { region: FeaRegionTarget } | { reference_point: { reference_point_id: string; position: FeaPosition } } | { port: { port_id: string } } | { interface: { interface_id: string } } | \"whole_model\";",
    "export interface FeaExcitationBody { excitation_id: string; case_id: string; target: FeaExcitationTarget; frame: FeaCoordinateFrame; direction: { vector: [number, number, number] } | null; distribution: \"concentrated\" | \"preserve_resultant\" | \"intensity_per_area\" | \"intensity_per_volume\" | \"uniform_kinematic\" | \"topology_field\"; kinematics: \"fixed_frame\" | \"follower\"; signal_id: string; active_step_ids: string[]; combination_group: string | null; projection: { preserve_resultant: boolean; preserve_moment: boolean; relative_tolerance: number }; }",
    externallyTaggedUnion("FeaExcitationInput", readTupleEnumValues(excitationSource, "ExcitationAction"), "FeaExcitationBody"),
    "export interface FeaExcitationCaseInput { case_id: string; action_ids: string[]; activation: \"always\" | { steps: string[] } | { mutually_exclusive_group: { group_id: string } }; }",
    "export type FeaInitialValue<Q> = { uniform: Q } | { uniform_vector3: [Q, Q, Q] } | { signal_sample: { signal_id: string; abscissa: number; approach: \"at\" | \"left_limit\" | \"right_limit\" } } | { field_artifact: { artifact_id: string; content_digest: string; topology_digest: string; field_id: string } };",
    "export interface FeaInitialConditionBody<Q> { initial_condition_id: string; target: { region: FeaRegionTarget } | \"whole_model\"; frame: FeaCoordinateFrame; value: FeaInitialValue<Q>; applies_before_step_id: string; }",
    "export type FeaInitialConditionInput = { displacement: FeaInitialConditionBody<{ meters: number }> } | { velocity: FeaInitialConditionBody<{ meters_per_second: number }> } | { angular_velocity: FeaInitialConditionBody<{ radians_per_second: number }> } | { temperature: FeaInitialConditionBody<{ kelvin: number }> } | { electric_potential: FeaInitialConditionBody<{ volts: number }> } | { magnetic_vector_potential: FeaInitialConditionBody<{ webers_per_meter: number }> } | { acoustic_pressure: FeaInitialConditionBody<{ pascals: number }> } | { fluid_pressure: FeaInitialConditionBody<{ pascals: number }> } | { fluid_velocity: FeaInitialConditionBody<{ meters_per_second: number }> };",
    "export interface FeaResponseSpectrumInput { spectrum_id: string; abscissa: { frequency_hertz: number[] } | { period_seconds: number[] }; ordinates: number[]; damping: { ratio: number } | { quality_factor: number }; response: \"relative_displacement\" | \"relative_velocity\" | \"relative_acceleration\" | \"absolute_acceleration\" | \"pseudo_velocity\" | \"pseudo_acceleration\"; envelope: \"positive\" | \"negative\" | \"maximax\"; interpolation: \"linear_frequency\" | \"linear_period\" | \"log_log\"; extrapolation: \"reject\" | \"hold_endpoint\"; direction: { frame: FeaCoordinateFrame; vector: [number, number, number] }; modal_combination: \"srss\" | \"complete_quadratic_combination\" | \"ten_percent\" | \"absolute_sum\"; directional_combination: \"srss\" | \"absolute_sum\" | { percentage_rule: { primary: number; secondary: number } }; truncation: { minimum_cumulative_mass_ratio: number; include_residual_mode: boolean; include_rigid_response: boolean; include_missing_mass: boolean }; construction: \"authored_design\" | { derived_from_acceleration_history: { source_signal_id: string; method: \"exact_piecewise_linear\" | \"recursive_filter\" } }; evidence: { sampling_adequacy_ratio: number; max_interpolation_error: number | null }; provenance_digest: string; }",
    "export interface FeaDomainsInput { thermo_mechanical?: { reference_temperature: { kelvin: number } }; electro_thermal?: { reference_temperature: { kelvin: number } }; electromagnetic?: { static: { equation: \"electrostatics\" | \"steady_conduction\" } } | { harmonic: { reference_frequency: { hertz: number } } }; cfd?: { density: { kilograms_per_cubic_meter: number }; dynamic_viscosity: { pascal_seconds: number }; turbulence_intensity: { ratio: number } }; }",
    "export interface FeaCanonicalSequence { sequence_id: string; steps: FeaProcedureStep[]; dependencies: FeaSequenceDependency[]; transitions: FeaStateTransition[]; branches: FeaSequenceBranch[]; }",
    "export type FeaExecutionInput = { step: string; backend?: FeaComputeBackend; accelerator_admission?: FeaAcceleratorAdmissionPolicy } & ({ solver: \"structural_linear\"; options: FeaStructuralLinearOptions } | { solver: \"structural_modes\"; options: FeaStructuralModesOptions } | { solver: \"acoustic_helmholtz\"; options: FeaAcousticHelmholtzOptions } | { solver: \"thermal_diffusion\"; options: FeaThermalDiffusionOptions } | { solver: \"structural_transient\"; options: FeaStructuralTransientOptions } | { solver: \"incompressible_flow\"; options: FeaIncompressibleFlowOptions } | { solver: \"structural_nonlinear\"; options: FeaStructuralNonlinearOptions } | { solver: \"maxwell_harmonic\"; options: FeaMaxwellHarmonicOptions } | { solver?: never; options?: never });",
  ].join("\n\n");
  const analysisInterfaces = analysisStructTypes.map(([rustName, tsName, options]) =>
    formatInterface(tsName, readStructFields(analysisSource, rustName, options))
  );
  const progressInterfaces = progressStructTypes.map(([rustName, tsName, options]) =>
    formatInterface(tsName, readStructFields(progressSource, rustName, options))
  );

  fs.mkdirSync(generatedRoot, { recursive: true });
  const contents = [
    "// @generated by scripts/generate-fea-contracts.cjs",
    "// Do not edit by hand.",
    "",
    'import type { FeaSemanticProgram, FeaSourceIndex, PublishedStudyAssessment, PublishedCaseAssessments, PublishedStudyReassessment } from "./fea-semantic-contracts.js";',
    "",
    ...constants.map(({ tsName, value }) => `export const ${tsName} = ${value} as const;`),
    "",
    ...enumTypes.flatMap((typeSource) => [typeSource, ""]),
    ...progressEnums.flatMap((typeSource) => [typeSource, ""]),
    ...sourceCheckContracts.flatMap((typeSource) => [typeSource, ""]),
    ...authoringEnums.flatMap((typeSource) => [typeSource, ""]),
    "export type FeaStructuralTimeStepPolicy = { fixed: { interval_count: number } };",
    "export interface FeaModalTruncationPolicy { requested_mode_count: number; eigenspace_relative_tolerance: number; }",
    "export interface FeaModalConvergencePolicy { relative_residual_tolerance: number; maximum_subspace_iterations: number; }",
    "",
    physicsFamily,
    "",
    procedureStep,
    "",
    ...progressInterfaces.flatMap((interfaceSource) => [interfaceSource, ""]),
    ...analysisInterfaces.flatMap((interfaceSource) => [interfaceSource, ""]),
    canonicalInputs,
  ].join("\n");
  if (unboundedWideIntegers.length) {
    throw new Error(`FEA analysis contract fields lack the 2^53 - 1 wire-integer adapter (#[serde(with = "${WIRE_INTEGER_ADAPTER}[::option|::vec]")]):\n  ${unboundedWideIntegers.join("\n  ")}`);
  }
  fs.writeFileSync(outputPath, contents);
  console.log(
    `Generated ${constants.length} FEA analysis constants, ${enumTypes.length + progressEnums.length + authoringEnums.length} type unions, and ${progressInterfaces.length + analysisInterfaces.length} analysis interfaces.`
  );
}

main();
