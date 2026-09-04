#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const bindingsRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(bindingsRoot, "../..");
const rustAuthoringPath = path.resolve(
  repoRoot,
  "crates/runmat-runtime/src/analysis/fea_document_authoring.rs"
);
const rustAuthoringContractsPath = path.resolve(
  repoRoot,
  "crates/runmat-runtime/src/analysis/fea_document_authoring/contracts.rs"
);
const rustFeaDocumentPath = path.resolve(
  repoRoot,
  "crates/runmat-runtime/src/analysis/fea_document.rs"
);
const rustAnalysisContractsPath = path.resolve(
  repoRoot,
  "crates/runmat-runtime/src/analysis/contracts.rs"
);
const rustFeaProgressPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/fea/src/progress.rs"
);
const rustCoreIdentityPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/core/src/study/identity.rs"
);
const rustCoreFormulationPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/core/src/study/formulation.rs"
);
const rustSignalPath = path.resolve(repoRoot, "crates/runmat-analysis/core/src/problem/signal.rs");
const rustExcitationPath = path.resolve(repoRoot, "crates/runmat-analysis/core/src/problem/excitation.rs");
const rustGeometryUnitsPath = path.resolve(
  repoRoot,
  "crates/runmat-geometry/core/src/model/units.rs"
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
const outputPath = path.resolve(generatedRoot, "fea-study-document-contracts.ts");
const analysisConstants = [
  ["ANALYSIS_RUN_DATASET_SCHEMA_VERSION", "FEA_RUN_DATASET_SCHEMA_VERSION"],
  ["ANALYSIS_FIELD_DESCRIPTORS_SCHEMA_VERSION", "FEA_FIELD_DESCRIPTORS_SCHEMA_VERSION"],
  ["ANALYSIS_DIAGNOSTICS_SCHEMA_VERSION", "FEA_DIAGNOSTICS_SCHEMA_VERSION"],
  ["ANALYSIS_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION", "FEA_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION"],
  ["ANALYSIS_RUN_DATASET_KIND", "FEA_RUN_DATASET_KIND"],
  ["ANALYSIS_DATASET_ARTIFACT_KIND", "FEA_DATASET_ARTIFACT_KIND"],
  ["ANALYSIS_FIELD_DESCRIPTORS_ARTIFACT_KIND", "FEA_FIELD_DESCRIPTORS_ARTIFACT_KIND"],
  ["ANALYSIS_DIAGNOSTICS_ARTIFACT_KIND", "FEA_DIAGNOSTICS_ARTIFACT_KIND"],
  ["ANALYSIS_ARTIFACT_MANIFEST_KIND", "FEA_ARTIFACT_MANIFEST_KIND"],
  ["ANALYSIS_FIELD_DEFAULT_PAGE_SIZE", "FEA_FIELD_DEFAULT_PAGE_SIZE"],
  ["ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT", "FEA_FIELD_DEFAULT_MATERIALIZE_LIMIT"],
];
const analysisEnumTypes = [
  ["AnalysisModelTemplate", "FeaModelTemplate"],
  ["AnalysisDocumentKind", "FeaDocumentKind"],
  ["AnalysisFieldKind", "FeaFieldKind"],
  ["AnalysisFieldStorage", "FeaFieldStorage"],
  ["AnalysisFieldLocation", "FeaFieldLocation"],
  ["PrecisionMode", "FeaPrecisionMode"],
  ["PreconditionerMode", "FeaPreconditionerMode"],
  ["QualityPolicy", "FeaQualityPolicy"],
];
const progressEnumTypes = [
  ["FeaProgressPhase", "FeaProgressPhase"],
  ["FeaProgressStatus", "FeaProgressStatus"],
];
const analysisStructTypes = [
  ["AnalysisRuntimeCapabilities", "FeaCapabilities", { renameAll: "camelCase" }],
  ["AnalysisDocumentCheckResult", "FeaCheckResult", { renameAll: "camelCase" }],
  ["AnalysisDocumentRunResult", "FeaRunResult", { renameAll: "camelCase" }],
  ["AnalysisFieldRequestOptions", "FeaFieldRequestOptions", { renameAll: "camelCase" }],
  ["AnalysisFieldPageResult", "FeaFieldResult", { renameAll: "camelCase" }],
  ["AnalysisRunDatasetStudyRef", "FeaRunDatasetStudyRef", { renameAll: "camelCase" }],
  ["AnalysisRunDatasetFieldPagingPolicy", "FeaRunDatasetFieldPagingPolicy", { renameAll: "camelCase" }],
  ["AnalysisRunDatasetPayload", "FeaRunDatasetPayload", { renameAll: "camelCase" }],
  ["AnalysisFieldDescriptorsArtifactPayload", "FeaFieldDescriptorsArtifactPayload", { renameAll: "camelCase" }],
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
  ["AnalysisChtRunOptions", "FeaConjugateHeatTransferOptions", {}],
  ["AnalysisFsiRunOptions", "FeaFluidStructureInteractionOptions", {}],
  ["AnalysisNonlinearRunOptions", "FeaStructuralNonlinearOptions", {}],
  ["AnalysisElectromagneticRunOptions", "FeaElectromagneticPotentialOptions", {}],
];
const progressStructTypes = [
  ["FeaProgressEvent", "FeaProgressEvent", { renameAll: "camelCase" }],
];
const studyDocumentStructTypes = [
  ["FeaStudyDocumentOperationOutput", {}],
  ["FeaStudySummary", { renameAll: "camelCase" }],
  ["FeaStudyMeshInput", {}],
  ["FeaStudyRegionEntry", { renameAll: "camelCase" }],
  ["FeaStudyMaterialEntry", { renameAll: "camelCase" }],
  ["FeaStudyMaterialAssignmentEntry", { renameAll: "camelCase" }],
  ["FeaStudyNamedEntry", { renameAll: "camelCase" }],
  ["FeaStudySequenceSummary", { renameAll: "camelCase" }],
  ["FeaStudyReadiness", { renameAll: "camelCase" }],
  ["FeaStudyDocumentDiff", { renameAll: "camelCase" }],
  ["FeaStudyDocumentCounts", { renameAll: "camelCase" }],
  ["FeaStudyDocumentOperationResult", { renameAll: "camelCase" }],
];
const studyDocumentStructs = studyDocumentStructTypes.map(([name]) => name);

function readOperationNames(source) {
  const match = source.match(
    /pub const FEA_STUDY_DOCUMENT_OPERATION_NAMES:\s*&\[&str\]\s*=\s*&\[(?<body>[\s\S]*?)\];/
  );
  if (!match?.groups?.body) {
    throw new Error("Could not find FEA_STUDY_DOCUMENT_OPERATION_NAMES in Rust authoring source.");
  }

  const names = [];
  for (const stringMatch of match.groups.body.matchAll(/"([^"]+)"/g)) {
    names.push(stringMatch[1]);
  }
  if (names.length === 0) {
    throw new Error("FEA_STUDY_DOCUMENT_OPERATION_NAMES did not contain any operation names.");
  }
  return names;
}

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

function rustTypeToTs(type) {
  const trimmed = type.trim();
  const optionMatch = trimmed.match(/^Option<(.+)>$/);
  if (optionMatch) {
    return { type: `${rustTypeToTs(optionMatch[1]).type} | null`, nullable: true };
  }

  const vectorMatch = trimmed.match(/^Vec<(.+)>$/);
  if (vectorMatch) {
    return { type: `${rustTypeToTs(vectorMatch[1]).type}[]`, nullable: false };
  }

  switch (trimmed) {
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
    case "u32":
    case "f64":
      return { type: "number", nullable: false };
    case "UnitSystem":
      return { type: "FeaGeometryUnits", nullable: false };
    case "AnalysisModelTemplate":
      return { type: "FeaModelTemplate", nullable: false };
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
    case "AnalysisRuntimeModelTemplateCatalogEntry":
      return { type: "FeaModelTemplateCatalogEntry", nullable: false };
    case "AnalysisRuntimeModelTemplateDefaultObservation":
      return { type: "{ field: string; location: string }", nullable: false };
    default:
      if (/^BTreeMap<String,\s*String>$/.test(trimmed)) {
        return { type: "Record<string, string>", nullable: false };
      }
      if (studyDocumentStructs.includes(trimmed)) {
        return { type: trimmed, nullable: false };
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
    if (!line || line.startsWith("#[")) {
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

function externallyTaggedUnion(name, variants, bodyName) {
  return `export type ${name} = ${variants.map((variant) => `{ ${variant}: ${bodyName} }`).join(" | ")};`;
}

function formatTypeUnion(name, values) {
  return `export type ${name} = ${values.map((value) => JSON.stringify(value)).join(" | ")};`;
}

function formatConstArray(name, values) {
  return [
    `export const ${name} = [`,
    ...values.map((value) => `  ${JSON.stringify(value)},`),
    "] as const;",
  ].join("\n");
}

function findMatching(source, startIndex, openChar, closeChar) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = startIndex; index < source.length; index += 1) {
    const char = source[index];
    if (inString) {
      escaped = char === "\\" && !escaped;
      if (char === "\"" && !escaped) {
        inString = false;
      }
      if (char !== "\\") {
        escaped = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
      continue;
    }
    if (char === openChar) {
      depth += 1;
    } else if (char === closeChar) {
      depth -= 1;
      if (depth === 0) {
        return index;
      }
    }
  }
  throw new Error(`Could not find matching ${closeChar} for ${openChar}.`);
}

function readConstArrayBody(source, constName) {
  const marker = `pub const ${constName}`;
  const constIndex = source.indexOf(marker);
  if (constIndex === -1) {
    throw new Error(`Could not find Rust const ${constName}.`);
  }
  const assignmentIndex = source.indexOf("=", constIndex);
  if (assignmentIndex === -1) {
    throw new Error(`Could not find assignment for Rust const ${constName}.`);
  }
  const arrayStart = source.indexOf("&[", assignmentIndex);
  if (arrayStart === -1) {
    throw new Error(`Could not find array initializer for Rust const ${constName}.`);
  }
  const bodyStart = arrayStart + 1;
  const bodyEnd = findMatching(source, bodyStart, "[", "]");
  return source.slice(bodyStart + 1, bodyEnd);
}

function readNamedStringField(body, fieldName) {
  const match = body.match(new RegExp(`${fieldName}:\\s*"([^"]*)"`));
  if (!match) {
    throw new Error(`Could not parse string field ${fieldName} in model template catalog entry.`);
  }
  return match[1];
}

function readNamedTemplateField(body) {
  const match = body.match(/template:\s*AnalysisModelTemplate::([A-Za-z0-9_]+)/);
  if (!match) {
    throw new Error("Could not parse template field in physics template catalog entry.");
  }
  return rustVariantToSnake(match[1]);
}

function splitStructInitializers(body, structName) {
  const entries = [];
  let searchIndex = 0;
  const marker = `${structName} {`;
  while (searchIndex < body.length) {
    const entryStart = body.indexOf(marker, searchIndex);
    if (entryStart === -1) {
      break;
    }
    const braceStart = body.indexOf("{", entryStart);
    const braceEnd = findMatching(body, braceStart, "{", "}");
    entries.push(body.slice(braceStart + 1, braceEnd));
    searchIndex = braceEnd + 1;
  }
  return entries;
}

function readModelTemplateCatalog(source) {
  const body = readConstArrayBody(source, "ANALYSIS_MODEL_TEMPLATE_CATALOG");
  const entries = splitStructInitializers(body, "AnalysisModelTemplateCatalogEntry");
  if (entries.length === 0) {
    throw new Error("ANALYSIS_MODEL_TEMPLATE_CATALOG did not contain any entries.");
  }
  return entries.map((entry) => ({
    template: readNamedTemplateField(entry),
    label: readNamedStringField(entry, "label"),
    family: readNamedStringField(entry, "family"),
    target: readNamedStringField(entry, "target"),
    value: readNamedStringField(entry, "value"),
    defaultObservations: splitStructInitializers(entry, "AnalysisModelTemplateDefaultObservation").map((observation) => ({
      field: readNamedStringField(observation, "field"),
      location: readNamedStringField(observation, "location"),
    })),
  }));
}

function formatModelTemplateCatalog(entries) {
  return [
    "export interface FeaModelTemplateCatalogEntry {",
    "  template: FeaModelTemplate;",
    "  label: string;",
    "  family: string;",
    "  target: string;",
    "  value: string;",
    "  defaultObservations: readonly {",
    "    field: string;",
    "    location: string;",
    "  }[];",
    "}",
    "",
    "export const FEA_SUPPORTED_MODEL_TEMPLATES: readonly FeaModelTemplateCatalogEntry[] = [",
    ...entries.flatMap((entry) => [
      "  {",
      `    template: ${JSON.stringify(entry.template)},`,
      `    label: ${JSON.stringify(entry.label)},`,
      `    family: ${JSON.stringify(entry.family)},`,
      `    target: ${JSON.stringify(entry.target)},`,
      `    value: ${JSON.stringify(entry.value)},`,
      "    defaultObservations: [",
      ...entry.defaultObservations.map(
        (observation) =>
          `      { field: ${JSON.stringify(observation.field)}, location: ${JSON.stringify(observation.location)} },`
      ),
      "    ],",
      "  },",
    ]),
    "] as const;",
    "",
    "export const FEA_SUPPORTED_PHYSICS_FAMILIES: readonly string[] = Array.from(",
    "  new Set(FEA_SUPPORTED_MODEL_TEMPLATES.map((template) => template.family))",
    ");",
  ].join("\n");
}

function readStructBody(source, name) {
  const match = source.match(new RegExp(`pub struct ${name} \\{(?<body>[\\s\\S]*?)\\n\\}`));
  if (!match?.groups?.body) {
    throw new Error(`Could not find Rust struct ${name}.`);
  }
  return match.groups.body;
}

function readStructFields(source, name, options = {}) {
  const body = readStructBody(source, name);
  const fields = [];
  let rename = null;
  let skipIfNone = false;
  const renameAll = options.renameAll ?? null;

  for (const rawLine of body.split("\n")) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    const renameMatch = line.match(/^#\[serde\(rename = "([^"]+)"\)\]$/);
    if (renameMatch) {
      rename = renameMatch[1];
      continue;
    }
    if (line === '#[serde(skip_serializing_if = "Option::is_none")]') {
      skipIfNone = true;
      continue;
    }
    if (line === '#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]') {
      skipIfNone = true;
      continue;
    }
    if (line.startsWith("#[")) {
      continue;
    }

    const fieldMatch = line.match(/^pub\s+([a-zA-Z0-9_]+):\s+(.+),$/);
    if (!fieldMatch) {
      throw new Error(`Could not parse field line in ${name}: ${line}`);
    }
    const rustName = fieldMatch[1];
    const rustType = fieldMatch[2];
    const tsType = rustTypeToTs(rustType);
    const defaultName = renameAll === "camelCase" ? snakeToCamel(rustName) : rustName;
    fields.push({
      name: rename ?? defaultName,
      optional: skipIfNone,
      type: skipIfNone && tsType.nullable ? tsType.type.replace(/ \| null$/, "") : tsType.type,
    });
    rename = null;
    skipIfNone = false;
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
  const authoringSource = [rustAuthoringPath, rustAuthoringContractsPath]
    .map((sourcePath) => fs.readFileSync(sourcePath, "utf8"))
    .join("\n");
  const analysisSource = fs.readFileSync(rustAnalysisContractsPath, "utf8");
  const feaDocumentSource = fs.readFileSync(rustFeaDocumentPath, "utf8");
  const progressSource = fs.readFileSync(rustFeaProgressPath, "utf8");
  const coreIdentitySource = fs.readFileSync(rustCoreIdentityPath, "utf8");
  const coreFormulationSource = fs.readFileSync(rustCoreFormulationPath, "utf8");
  const signalSource = fs.readFileSync(rustSignalPath, "utf8");
  const excitationSource = fs.readFileSync(rustExcitationPath, "utf8");
  const geometryUnitsSource = fs.readFileSync(rustGeometryUnitsPath, "utf8");
  const meshingRequestSource = fs.readFileSync(rustMeshingRequestPath, "utf8");
  const feaContractsSource = fs.readFileSync(rustFeaContractsPath, "utf8");
  const operations = readOperationNames(authoringSource);
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
  const authoringEnums = [
    formatTypeUnion("FeaGeometryUnits", readEnumValues(geometryUnitsSource, "UnitSystem")),
    formatTypeUnion("FeaComputeBackend", readEnumValues(feaContractsSource, "ComputeBackend")),
    formatTypeUnion("FeaElementOrder", readEnumValues(meshingRequestSource, "ElementOrder")),
    formatTypeUnion("FeaAnalysisProcedure", readEnumValues(coreIdentitySource, "AnalysisProcedure")),
    formatTypeUnion(
      "FeaFormulationRequirement",
      readEnumValues(coreFormulationSource, "FormulationRequirement")
    ),
  ];
  const coupledFamilies = readEnumValues(coreIdentitySource, "CoupledPhysicsFamily");
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
export type FeaObservationQuantity = "displacement" | "rotation" | "velocity" | "acceleration" | "strain" | "stress" | "reaction_force" | "reaction_moment" | "strain_energy" | "temperature" | "temperature_gradient" | "heat_flux" | "heat_rate" | "electric_potential" | "electric_field" | "electric_current" | "electric_current_density" | "electric_charge" | "magnetic_vector_potential" | "magnetic_flux_density" | "magnetic_field_intensity" | "acoustic_pressure" | "acoustic_particle_velocity" | "sound_pressure_level" | "fluid_pressure" | "fluid_velocity" | "vorticity" | "mass_flow_rate" | "wall_shear_stress" | "modal_frequency" | "modal_participation" | "eigenvalue" | "residual" | "balance" | "constraint_violation" | "contact_gap" | "contact_pressure";
export type FeaSamplingPolicy = "final" | "every_accepted_increment" | { time_seconds: number[] } | { time_interval_seconds: { start: number; end: number; interval: number } } | { frequency_hertz: number[] } | { frequency_interval_hertz: { start: number; end: number; interval: number } } | { load_factors: number[] } | { load_factor_interval: { start: number; end: number; interval: number } } | { mode_indices: number[] } | { event: "first_yield" | "contact_status_change" | "peak_response" | "convergence_failure" | "accuracy_criterion_crossing" } | { adaptive: { minimum_interval: number; maximum_samples: number; relative_change: number } };
export type FeaRetentionPolicy = "canonical" | "until_dependents_complete" | { latest_samples: { count: number } } | { explicit_samples: { maximum_count: number } };
export interface FeaObservationSpec {
  observation_id: string; quantity: FeaObservationQuantity;
  value_shape: "scalar" | "vector3" | "symmetric_tensor3" | { tensor: { extents: number[] } };
  target: FeaObservationTarget; source_location: FeaFieldLocationContract; output_location: FeaFieldLocationContract;
  recovery: "native" | "extrapolate" | "volume_weighted_average" | "area_weighted_average" | "conservative_projection" | "shape_function_interpolation";
  component: "all" | "x" | "y" | "z" | "normal" | "tangent1" | "tangent2" | "trace" | "hydrostatic" | "deviatoric" | "von_mises" | "maximum_principal" | "minimum_principal" | "magnitude";
  frame: FeaCoordinateFrame; sign_convention: "native" | "tension_positive" | "compression_positive" | "outward_positive" | "inward_positive";
  complex_representation: "real" | "imaginary" | "magnitude" | "phase_radians" | "complex";
  sampling: FeaSamplingPolicy; reduction: FeaReductionPolicy | null; storage: FeaObservationStoragePolicy;
  purpose: "publication" | "solver_control" | "visualization" | "downstream_state"; cost: FeaObservationCostEstimate;
}
export type FeaFieldLocationContract = "node" | "cell" | "element" | "integration_point" | "face" | "edge" | "interface" | "connector" | "reference_point" | "mode" | "global";
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
export interface FeaAccuracyPlan { accuracy_plan_id: string; criteria: FeaAccuracyCriterion[]; unavailable_evidence: "fail" | "mark_indeterminate_experimental"; exceeded_limit: "refine" | "tighten_solver" | "extend_sampling" | "fail" | "warn_experimental"; equivalence: "deterministic" | { relative_tolerance: number } | { absolute_tolerance: number }; resources: { maximum_refinement_cycles: number; maximum_solver_tightenings: number; maximum_samples: number; maximum_wall_time_milliseconds: number; maximum_resident_bytes: number }; }
export interface FeaProcedureStep {
  step_id: string; family: FeaPhysicsFamily; procedure: FeaAnalysisProcedure; solution_space: "direct" | "modal_reduced";
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
    heat_flux: "{ watts_per_square_meter: number }", heat_generation_density: "{ watts_per_cubic_meter: number }",
    electric_potential: "{ volts: number }", electric_current: "{ amperes: number }", electric_charge: "{ coulombs: number }",
    electric_current_density: "{ amperes_per_square_meter: number }", electric_charge_density: "{ coulombs_per_cubic_meter: number }",
    magnetic_vector_potential: "{ webers_per_meter: number }", mass_flow_rate: "{ kilograms_per_second: number }",
  };
  const signalInput = `export type FeaSignalInput = ${readTupleEnumValues(signalSource, "PhysicalSignal").map((variant) => {
    const quantity = signalQuantityTypes[variant];
    if (!quantity) throw new Error(`Missing TypeScript quantity contract for PhysicalSignal::${variant}`);
    return `{ ${variant}: FeaSignalBody<${quantity}> }`;
  }).join(" | ")};`;
  const canonicalInputs = [
    externallyTaggedUnion(
      "FeaRegionSelector",
      readTupleEnumValues(feaDocumentSource, "FeaRegionSelector"),
      "string"
    ),
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
    "export type FeaConstraintInput = { structural: { constraint_id: string; frame: FeaCoordinateFrame; kind: FeaStructuralConstraintKind } } | { thermal: { constraint_id: string; kind: { prescribed: { target: FeaRegionTarget; value: FeaPrescribedValue<{ kelvin: number }> } } | { periodic: FeaPairedConstraint } | { tie: FeaPairedConstraint } | { affine_jump: { pair: FeaPairedConstraint; temperature_jump: FeaPrescribedValue<{ kelvin: number }> } } } } | { electrical: { constraint_id: string; kind: { prescribed_potential: { target: FeaRegionTarget; value: FeaPrescribedValue<{ volts: number }> } } | { equipotential: { target: FeaRegionTarget } } | { floating_conductor: { target: FeaRegionTarget } } | { integral_current: { target: FeaRegionTarget; value: FeaPrescribedValue<{ amperes: number }> } } | { periodic: FeaPairedConstraint } | { potential_reference: { target: FeaRegionTarget } } } } | { magnetic: { constraint_id: string; kind: { prescribed_vector_potential: { target: FeaRegionTarget; value: FeaPrescribedValue<{ webers_per_meter: number }> } } | { periodic: FeaPairedConstraint } | { vector_potential_ground: { target: FeaRegionTarget } } | { coulomb_gauge: { target: FeaRegionTarget } } } } | { acoustic: { constraint_id: string; kind: { prescribed_pressure: { target: FeaRegionTarget; value: FeaPrescribedValue<{ pascals: number }> } } | { periodic: FeaPairedConstraint } } } | { fluid: { constraint_id: string; kind: { prescribed_velocity: { target: FeaRegionTarget; value: FeaPrescribedValue<{ meters_per_second: number }> } } | { moving_wall: { target: FeaRegionTarget; velocity: FeaPrescribedValue<{ meters_per_second: number }> } } | { no_slip_wall: { target: FeaRegionTarget } } | { slip_wall: { target: FeaRegionTarget } } | { symmetry: { target: FeaRegionTarget } } | { periodic: FeaPairedConstraint } | { pressure_reference: { target: FeaRegionTarget; value: FeaPrescribedValue<{ pascals: number }> } } } };",
    "export type FeaBoundaryOperator = { structural_pressure: { value: FeaPrescribedValue<{ pascals: number }> } } | { thermal_heat_flux: { value: FeaPrescribedValue<{ watts_per_square_meter: number }> } } | { thermal_convection: { ambient_temperature: FeaPrescribedValue<{ kelvin: number }>; coefficient: { watts_per_square_meter_kelvin: number } } } | \"magnetic_insulation\" | \"acoustic_rigid_wall\" | \"acoustic_radiation\" | { acoustic_impedance: { impedance: { pascal_seconds_per_meter: number } } } | { fluid_inlet_velocity: { value: FeaPrescribedValue<{ meters_per_second: number }> } } | { fluid_outlet_pressure: { value: FeaPrescribedValue<{ pascals: number }> } } | \"fluid_no_slip_wall\" | \"fluid_slip_wall\" | \"fluid_symmetry\";",
    "export interface FeaBoundaryInput { boundary_id: string; target: FeaRegionTarget; operator: FeaBoundaryOperator; }",
    "export type FeaConnectorKind = { translational_spring: { stiffness: { newtons_per_meter: number } } } | { translational_dashpot: { damping: { newton_seconds_per_meter: number } } } | { elastic_foundation: { stiffness: { newtons_per_meter: number } } };",
    "export interface FeaConnectorInput { connector_id: string; primary: FeaRegionTarget; secondary: FeaRegionTarget | null; kind: FeaConnectorKind; }",
    "export type FeaInterfaceKind = { contact: { penalty_stiffness_scale: number; max_penetration_ratio: number; friction_coefficient: number } } | { fluid_structure: { normal_stiffness_pa_per_m: number; damping_ratio: number; relaxation_factor: number } } | { conjugate_heat_transfer: { thermal_conductance_w_per_m2k: number; contact_resistance_m2k_per_w: number; relaxation_factor: number } };",
    "export interface FeaInterfaceInput { interface_id: string; primary_region_id: string; secondary_region_id: string; kind: FeaInterfaceKind; }",
    "export interface FeaMaterialInput { material_id: string; name?: string; mechanical: { youngs_modulus_pa: number; poisson_ratio: number; density_kg_per_m3?: number }; thermal?: { reference_temperature_k?: number; modulus_temp_coeff_per_k?: number; conductivity_w_per_mk?: number; specific_heat_j_per_kgk?: number; expansion_coefficient_per_k?: number }; acoustic?: { density_kg_per_m3?: number; speed_of_sound_m_per_s?: number; damping_ratio?: number }; electrical?: { reference_temperature_k?: number; conductivity_s_per_m?: number; resistive_heating_coefficient?: number; relative_permittivity?: number; relative_permeability?: number; conductivity_frequency_response?: { frequency_hz: number; conductivity_scale: number; dispersive_loss_scale?: number; relative_permittivity_scale?: number; relative_permeability_scale?: number }[] }; plastic?: { yield_strain: number; hardening_modulus_ratio: number; saturation_exponent: number }; }",
    "export interface FeaMaterialAssignmentInput { region: string; material: string; expected_material?: string; confidence?: \"verified\" | \"probable\" | \"inferred\"; }",
    "export type FeaSignalShape = \"scalar\" | \"vector3\" | { tensor: number[] } | { topology_field: { topology_id: string; component_count: number } };",
    "export type FeaSignalValue<Q> = { scalar: Q } | { vector3: [Q, Q, Q] } | { tensor: { shape: number[]; values: Q[] } };",
    "export type FeaSignalOrdinate<Q> = { real: FeaSignalValue<Q> } | { complex: { real: FeaSignalValue<Q>; imaginary: FeaSignalValue<Q> } };",
    "export type FeaSignalAxis = { time_seconds: number[] } | { frequency_hertz: number[] } | { study_parameter: { parameter_id: string; canonical_values: number[] } };",
    "export interface FeaSignalArtifact { artifact_id: string; content_digest: string; schema: string; domain: \"constant\" | { time_seconds: { start: number; end: number } } | { frequency_hertz: { start: number; end: number } } | { study_parameter: { parameter_id: string; start: number; end: number } }; topology_digest: string | null; sample_count: number; }",
    "export type FeaGeneratorArgument = { boolean: boolean } | { signed_integer: number } | { unsigned_integer: number } | { real: number } | { text: string } | { real_vector: number[] } | { artifact: FeaSignalArtifact };",
    "export interface FeaSignalGenerator { qualified_function: string; arguments: FeaGeneratorArgument[]; domain: \"time_seconds\" | \"frequency_hertz\" | { study_parameter: { parameter_id: string } }; sampling: { uniform: { start: number; end: number; sample_count: number } } | { adaptive: { start: number; end: number; max_absolute_error: number; max_samples: number } } | { canonical_points: number[] }; effects: \"pure\" | \"declared_immutable_artifacts\" | { seeded_random: { seed: number; distribution: \"uniform\" | \"normal\" | \"log_normal\" } }; }",
    "export type FeaSignalRepresentation<Q> = { constant: FeaSignalValue<Q> } | { tabulated: { axis: FeaSignalAxis; values: FeaSignalOrdinate<Q>[]; interpolation: \"step_left\" | \"step_right\" | \"linear\" | \"cubic_hermite\"; extrapolation: \"reject\" | \"hold_endpoint\" | \"zero\"; breakpoints: { abscissa: number; side: \"left\" | \"right\" | \"both\" }[] } } | { waveform: { kind: \"step\" | \"ramp\" | \"sine\" | \"half_sine_pulse\" | \"triangular_pulse\" | \"rectangular_pulse\"; amplitude: FeaSignalValue<Q>; offset: FeaSignalValue<Q> | null; start_seconds: number; duration_seconds: number | null; frequency_hz: number | null; phase_radians: number; duty_cycle: number | null } } | { artifact: FeaSignalArtifact } | { generator: FeaSignalGenerator };",
    "export type FeaSignalPrecision = \"real32\" | \"real64\" | \"complex64\" | \"complex128\";",
    "export interface FeaSignalBody<Q> { signal_id: string; shape: FeaSignalShape; frame: FeaCoordinateFrame | null; representation: FeaSignalRepresentation<Q>; precision: FeaSignalPrecision; evidence: { max_absolute_error: number | null; max_relative_error: number | null; derivative_max_absolute_error: number | null; sample_count: number }; provenance: \"authored\" | { imported: { source_digest: string } } | { generated: { project_revision: string; callable_id: string; frozen_input_digest: string } }; }",
    signalInput,
    "export type FeaExcitationTarget = { region: FeaRegionTarget } | { reference_point: { reference_point_id: string; position: FeaPosition } } | { port: { port_id: string } } | { interface: { interface_id: string } } | \"whole_model\";",
    "export interface FeaExcitationBody { excitation_id: string; case_id: string; target: FeaExcitationTarget; frame: FeaCoordinateFrame; direction: { vector: [number, number, number] } | null; distribution: \"concentrated\" | \"preserve_resultant\" | \"intensity_per_area\" | \"intensity_per_volume\" | \"uniform_kinematic\" | \"topology_field\"; kinematics: \"fixed_frame\" | \"follower\"; signal_id: string; active_step_ids: string[]; combination_group: string | null; projection: { preserve_resultant: boolean; preserve_moment: boolean; relative_tolerance: number }; }",
    externallyTaggedUnion("FeaExcitationInput", readTupleEnumValues(excitationSource, "ExcitationAction"), "FeaExcitationBody"),
    "export interface FeaExcitationCaseInput { case_id: string; action_ids: string[]; activation: \"always\" | { steps: string[] } | { mutually_exclusive_group: { group_id: string } }; }",
    "export type FeaInitialValue<Q> = { uniform: Q } | { uniform_vector3: [Q, Q, Q] } | { signal_sample: { signal_id: string; abscissa: number } } | { field_artifact: { artifact_id: string; content_digest: string; topology_digest: string; field_id: string } };",
    "export interface FeaInitialConditionBody<Q> { initial_condition_id: string; target: { region: FeaRegionTarget } | \"whole_model\"; frame: FeaCoordinateFrame; value: FeaInitialValue<Q>; applies_before_step_id: string; }",
    "export type FeaInitialConditionInput = { displacement: FeaInitialConditionBody<{ meters: number }> } | { velocity: FeaInitialConditionBody<{ meters_per_second: number }> } | { angular_velocity: FeaInitialConditionBody<{ radians_per_second: number }> } | { temperature: FeaInitialConditionBody<{ kelvin: number }> } | { electric_potential: FeaInitialConditionBody<{ volts: number }> } | { magnetic_vector_potential: FeaInitialConditionBody<{ webers_per_meter: number }> } | { acoustic_pressure: FeaInitialConditionBody<{ pascals: number }> } | { fluid_pressure: FeaInitialConditionBody<{ pascals: number }> } | { fluid_velocity: FeaInitialConditionBody<{ meters_per_second: number }> };",
    "export interface FeaResponseSpectrumInput { spectrum_id: string; abscissa: { frequency_hertz: number[] } | { period_seconds: number[] }; ordinates: number[]; damping: { ratio: number } | { quality_factor: number }; response: \"relative_displacement\" | \"relative_velocity\" | \"relative_acceleration\" | \"absolute_acceleration\" | \"pseudo_velocity\" | \"pseudo_acceleration\"; envelope: \"positive\" | \"negative\" | \"maximax\"; interpolation: \"linear_frequency\" | \"linear_period\" | \"log_log\"; extrapolation: \"reject\" | \"hold_endpoint\"; direction: { frame: FeaCoordinateFrame; vector: [number, number, number] }; modal_combination: \"srss\" | \"complete_quadratic_combination\" | \"ten_percent\" | \"absolute_sum\"; directional_combination: \"srss\" | \"absolute_sum\" | { percentage_rule: { primary: number; secondary: number } }; truncation: { minimum_cumulative_mass_ratio: number; include_residual_mode: boolean; include_rigid_response: boolean; include_missing_mass: boolean }; construction: \"authored_design\" | { derived_from_acceleration_history: { source_signal_id: string; method: \"exact_piecewise_linear\" | \"recursive_filter\" } }; evidence: { sampling_adequacy_ratio: number; max_interpolation_error: number | null }; provenance_digest: string; }",
    "export interface FeaDomainsInput { thermo_mechanical?: { enabled: boolean; reference_temperature_k: number; applied_temperature_delta_k: number; field_artifact_id?: string | null; field_source?: { source_id: string; revision: number; interpolation_mode?: \"linear\" | \"step\" | null; expected_region_ids?: string[] } | null; region_temperature_deltas?: { region_id: string; temperature_delta_k: number }[]; temperature_signal_id?: string | null }; electro_thermal?: { enabled: boolean; reference_temperature_k: number; applied_voltage_v: number; region_conductivity_scales?: { region_id: string; conductivity_scale: number }[]; electric_potential_signal_id?: string | null }; electromagnetic?: { enabled: boolean; reference_frequency_hz: number; applied_current_a: number }; cfd?: { enabled: boolean; solve_family: \"steady_state\" | \"transient\"; reference_density_kg_per_m3: number; dynamic_viscosity_pa_s: number; inlet_velocity_m_per_s: number; turbulence_intensity: number; inlet_velocity_signal_id?: string | null }; }",
    "export type FeaMeshSettingsInput = Partial<FeaStudyMeshInput>;",
    "export interface FeaStudyCreateInput { study_id?: string; geometry: { path: string; units?: FeaGeometryUnits; import?: { max_triangles?: number } }; model: { id?: string; template: FeaModelTemplate; frame?: FeaCoordinateFrame; defaults?: \"template_scaffold\" | \"none\" }; mesh?: FeaMeshSettingsInput; sequence?: FeaCanonicalSequence; execution?: FeaExecutionInput[]; }",
    "export interface FeaCanonicalSequence { sequence_id: string; steps: FeaProcedureStep[]; dependencies: FeaSequenceDependency[]; transitions: FeaStateTransition[]; branches: FeaSequenceBranch[]; }",
    "export type FeaExecutionInput = { step: string; backend?: FeaComputeBackend; requirement?: FeaFormulationRequirement } & ({ solver: \"structural_linear\"; options: FeaStructuralLinearOptions } | { solver: \"structural_modes\"; options: FeaStructuralModesOptions } | { solver: \"acoustic_helmholtz\"; options: FeaAcousticHelmholtzOptions } | { solver: \"thermal_diffusion\"; options: FeaThermalDiffusionOptions } | { solver: \"structural_transient\"; options: FeaStructuralTransientOptions } | { solver: \"incompressible_flow\"; options: FeaIncompressibleFlowOptions } | { solver: \"conjugate_heat_transfer\"; options: FeaConjugateHeatTransferOptions } | { solver: \"fluid_structure_interaction\"; options: FeaFluidStructureInteractionOptions } | { solver: \"structural_nonlinear\"; options: FeaStructuralNonlinearOptions } | { solver: \"electromagnetic_potential\"; options: FeaElectromagneticPotentialOptions } | { solver?: never; options?: never });",
    "export interface FeaStudyOperationInputs {",
    "  get_summary: Record<string, never>; create: FeaStudyCreateInput;",
    "  add_region: FeaRegionInput; update_region: FeaRegionInput; remove_region: { id: string };",
    "  add_material: FeaMaterialInput; update_material: FeaMaterialInput; remove_material: { id: string };",
    "  assign_material: FeaMaterialAssignmentInput; remove_material_assignment: { region: string; material: string };",
    "  add_constraint: FeaConstraintInput; update_constraint: FeaConstraintInput; remove_constraint: { id: string };",
    "  add_boundary: FeaBoundaryInput; update_boundary: FeaBoundaryInput; remove_boundary: { id: string };",
    "  add_connector: FeaConnectorInput; update_connector: FeaConnectorInput; remove_connector: { id: string };",
    "  add_interface: FeaInterfaceInput; update_interface: FeaInterfaceInput; remove_interface: { id: string };",
    "  add_signal: FeaSignalInput; update_signal: FeaSignalInput; remove_signal: { id: string };",
    "  add_excitation_case: FeaExcitationCaseInput; update_excitation_case: FeaExcitationCaseInput; remove_excitation_case: { id: string };",
    "  add_excitation: FeaExcitationInput; update_excitation: FeaExcitationInput; remove_excitation: { id: string };",
    "  add_initial_condition: FeaInitialConditionInput; update_initial_condition: FeaInitialConditionInput; remove_initial_condition: { id: string };",
    "  add_response_spectrum: FeaResponseSpectrumInput; update_response_spectrum: FeaResponseSpectrumInput; remove_response_spectrum: { id: string };",
    "  set_sequence: FeaCanonicalSequence; set_domains: FeaDomainsInput; set_mesh: FeaMeshSettingsInput; clear_mesh: Record<string, never>;",
    "}",
    "export type FeaStudyOperationInput<O extends FeaStudyDocumentOperation> = FeaStudyOperationInputs[O];",
  ].join("\n\n");
  const modelTemplateCatalog = readModelTemplateCatalog(analysisSource);
  const analysisInterfaces = analysisStructTypes.map(([rustName, tsName, options]) =>
    formatInterface(tsName, readStructFields(analysisSource, rustName, options))
  );
  const progressInterfaces = progressStructTypes.map(([rustName, tsName, options]) =>
    formatInterface(tsName, readStructFields(progressSource, rustName, options))
  );
  const interfaces = studyDocumentStructTypes.map(([name, options]) =>
    formatInterface(name, readStructFields(authoringSource, name, options))
  );

  fs.mkdirSync(generatedRoot, { recursive: true });
  const contents = [
    "// @generated by scripts/generate-fea-contracts.cjs",
    "// Do not edit by hand.",
    "",
    ...constants.map(({ tsName, value }) => `export const ${tsName} = ${value} as const;`),
    "",
    formatConstArray(
      "FEA_MODEL_TEMPLATES",
      analysisEnumValues.find(({ rustName }) => rustName === "AnalysisModelTemplate")?.values ?? []
    ),
    "",
    ...enumTypes.flatMap((typeSource) => [typeSource, ""]),
    ...progressEnums.flatMap((typeSource) => [typeSource, ""]),
    ...authoringEnums.flatMap((typeSource) => [typeSource, ""]),
    physicsFamily,
    "",
    procedureStep,
    "",
    formatModelTemplateCatalog(modelTemplateCatalog),
    "",
    ...progressInterfaces.flatMap((interfaceSource) => [interfaceSource, ""]),
    ...analysisInterfaces.flatMap((interfaceSource) => [interfaceSource, ""]),
    "export const FEA_STUDY_DOCUMENT_OPERATIONS = [",
    ...operations.map((operation) => `  ${JSON.stringify(operation)},`),
    "] as const;",
    "",
    "export type FeaStudyDocumentOperation = typeof FEA_STUDY_DOCUMENT_OPERATIONS[number];",
    "",
    canonicalInputs,
    "",
    ...interfaces.flatMap((interfaceSource) => [interfaceSource, ""]),
  ].join("\n");
  fs.writeFileSync(outputPath, contents);
  console.log(
    `Generated ${constants.length} FEA analysis constants, ${enumTypes.length + progressEnums.length + authoringEnums.length} type unions, ${modelTemplateCatalog.length} model template catalog entries, ${progressInterfaces.length + analysisInterfaces.length} analysis interfaces, ${operations.length} FEA study document operations, and ${interfaces.length} study interfaces.`
  );
}

main();
