#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const bindingsRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(bindingsRoot, "../..");
const rustAuthoringPath = path.resolve(
  repoRoot,
  "crates/runmat-runtime/src/analysis/fea_document_authoring.rs"
);
const rustAnalysisContractsPath = path.resolve(
  repoRoot,
  "crates/runmat-runtime/src/analysis/contracts.rs"
);
const rustFeaProgressPath = path.resolve(
  repoRoot,
  "crates/runmat-analysis/fea/src/progress.rs"
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
  ["AnalysisCreateModelProfile", "FeaAnalysisProfile"],
  ["AnalysisRunKind", "FeaAnalysisRunKind"],
  ["AnalysisDocumentKind", "FeaDocumentKind"],
  ["AnalysisFieldKind", "FeaFieldKind"],
  ["AnalysisFieldStorage", "FeaFieldStorage"],
  ["AnalysisFieldLocation", "FeaFieldLocation"],
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
];
const progressStructTypes = [
  ["FeaProgressEvent", "FeaProgressEvent", { renameAll: "camelCase" }],
];
const studyDocumentStructs = [
  "FeaStudyDocumentOperationOutput",
  "FeaStudySummary",
  "FeaStudyRegionEntry",
  "FeaStudyMaterialEntry",
  "FeaStudyMaterialAssignmentEntry",
  "FeaStudyBoundaryConditionEntry",
  "FeaStudyDrivingConditionEntry",
  "FeaStudyStepEntry",
  "FeaStudyOutputEntry",
  "FeaStudyReadiness",
  "FeaStudyDocumentDiff",
  "FeaStudyDocumentCounts",
  "FeaStudyDocumentOperationResult",
];

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
    case "AnalysisRuntimePhysicsProfileCatalogEntry":
      return { type: "FeaPhysicsProfileCatalogEntry", nullable: false };
    case "AnalysisRuntimePhysicsProfileDefaultOutput":
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
    throw new Error(`Could not parse string field ${fieldName} in physics profile catalog entry.`);
  }
  return match[1];
}

function readNamedProfileField(body) {
  const match = body.match(/profile:\s*AnalysisCreateModelProfile::([A-Za-z0-9_]+)/);
  if (!match) {
    throw new Error("Could not parse profile field in physics profile catalog entry.");
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

function readPhysicsProfileCatalog(source) {
  const body = readConstArrayBody(source, "ANALYSIS_PHYSICS_PROFILE_CATALOG");
  const entries = splitStructInitializers(body, "AnalysisPhysicsProfileCatalogEntry");
  if (entries.length === 0) {
    throw new Error("ANALYSIS_PHYSICS_PROFILE_CATALOG did not contain any entries.");
  }
  return entries.map((entry) => ({
    profile: readNamedProfileField(entry),
    label: readNamedStringField(entry, "label"),
    family: readNamedStringField(entry, "family"),
    target: readNamedStringField(entry, "target"),
    value: readNamedStringField(entry, "value"),
    defaultOutputs: splitStructInitializers(entry, "AnalysisPhysicsProfileDefaultOutput").map((output) => ({
      field: readNamedStringField(output, "field"),
      location: readNamedStringField(output, "location"),
    })),
  }));
}

function formatPhysicsProfileCatalog(entries) {
  return [
    "export interface FeaPhysicsProfileCatalogEntry {",
    "  profile: FeaAnalysisProfile;",
    "  label: string;",
    "  family: string;",
    "  target: string;",
    "  value: string;",
    "  defaultOutputs: readonly {",
    "    field: string;",
    "    location: string;",
    "  }[];",
    "}",
    "",
    "export const FEA_SUPPORTED_PHYSICS_PROFILES: readonly FeaPhysicsProfileCatalogEntry[] = [",
    ...entries.flatMap((entry) => [
      "  {",
      `    profile: ${JSON.stringify(entry.profile)},`,
      `    label: ${JSON.stringify(entry.label)},`,
      `    family: ${JSON.stringify(entry.family)},`,
      `    target: ${JSON.stringify(entry.target)},`,
      `    value: ${JSON.stringify(entry.value)},`,
      "    defaultOutputs: [",
      ...entry.defaultOutputs.map(
        (output) =>
          `      { field: ${JSON.stringify(output.field)}, location: ${JSON.stringify(output.location)} },`
      ),
      "    ],",
      "  },",
    ]),
    "] as const;",
    "",
    "export const FEA_SUPPORTED_PHYSICS_FAMILIES: readonly string[] = Array.from(",
    "  new Set(FEA_SUPPORTED_PHYSICS_PROFILES.map((profile) => profile.family))",
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
  const authoringSource = fs.readFileSync(rustAuthoringPath, "utf8");
  const analysisSource = fs.readFileSync(rustAnalysisContractsPath, "utf8");
  const progressSource = fs.readFileSync(rustFeaProgressPath, "utf8");
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
  const physicsProfileCatalog = readPhysicsProfileCatalog(analysisSource);
  const analysisInterfaces = analysisStructTypes.map(([rustName, tsName, options]) =>
    formatInterface(tsName, readStructFields(analysisSource, rustName, options))
  );
  const progressInterfaces = progressStructTypes.map(([rustName, tsName, options]) =>
    formatInterface(tsName, readStructFields(progressSource, rustName, options))
  );
  const interfaces = studyDocumentStructs.map((name) =>
    formatInterface(name, readStructFields(authoringSource, name))
  );

  fs.mkdirSync(generatedRoot, { recursive: true });
  const contents = [
    "// @generated by scripts/generate-fea-contracts.cjs",
    "// Do not edit by hand.",
    "",
    ...constants.map(({ tsName, value }) => `export const ${tsName} = ${value} as const;`),
    "",
    formatConstArray(
      "FEA_ANALYSIS_PROFILES",
      analysisEnumValues.find(({ rustName }) => rustName === "AnalysisCreateModelProfile")?.values ?? []
    ),
    "",
    formatConstArray(
      "FEA_ANALYSIS_RUN_KINDS",
      analysisEnumValues.find(({ rustName }) => rustName === "AnalysisRunKind")?.values ?? []
    ),
    "",
    ...enumTypes.flatMap((typeSource) => [typeSource, ""]),
    ...progressEnums.flatMap((typeSource) => [typeSource, ""]),
    formatPhysicsProfileCatalog(physicsProfileCatalog),
    "",
    ...progressInterfaces.flatMap((interfaceSource) => [interfaceSource, ""]),
    ...analysisInterfaces.flatMap((interfaceSource) => [interfaceSource, ""]),
    "export const FEA_STUDY_DOCUMENT_OPERATIONS = [",
    ...operations.map((operation) => `  ${JSON.stringify(operation)},`),
    "] as const;",
    "",
    "export type FeaStudyDocumentOperation = typeof FEA_STUDY_DOCUMENT_OPERATIONS[number];",
    "",
    ...interfaces.flatMap((interfaceSource) => [interfaceSource, ""]),
  ].join("\n");
  fs.writeFileSync(outputPath, contents);
  console.log(
    `Generated ${constants.length} FEA analysis constants, ${enumTypes.length + progressEnums.length} field unions, ${physicsProfileCatalog.length} physics profile catalog entries, ${progressInterfaces.length + analysisInterfaces.length} field interfaces, ${operations.length} FEA study document operations, and ${interfaces.length} study interfaces.`
  );
}

main();
