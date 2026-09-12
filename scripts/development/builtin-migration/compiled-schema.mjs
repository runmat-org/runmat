import { compareCodePoint } from "./constants.mjs";
import {
  array, boolean, enumValue, exact, identity, integer, nonempty, object,
  repositoryPath, rustModulePath, uniqueStrings,
} from "./schema.mjs";
import { validateBuiltinExampleFixture, validateBuiltinExampleRequirements } from "../../metadata/BuiltinExampleFixtureSchema.mjs";

const NUMERIC_CLASSES = ["Double", "Single", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64"];
const CAPABILITIES = ["HostRuntime", "Filesystem", "Network", "UserInterface", "Accelerator", "NativeCode", "ForeignRuntime", "ParallelRuntime", "DistributedRuntime"];
const EFFECTS = ["WorkspaceRead", "WorkspaceWrite", "EnvironmentRead", "EnvironmentWrite", "FilesystemRead", "FilesystemWrite", "Network", "UserInterface", "Randomness", "Clock", "HostCallback", "MaySuspend", "MayThrow", "Unknown"];

export function catalogEntry(value) {
  exact(value, ["identity", "category", "documentation", "descriptor", "contract", "placement", "link", "bindings", "extensions", "integer_capabilities", "integer_audit", "suppress_auto_output"], "compiled catalog entry");
  exact(value.identity, ["name"], "catalog identity"); identity(value.identity.name, "catalog name"); nonempty(value.category, "catalog category");
  documentation(value.documentation); descriptor(value.descriptor); contract(value.contract); placement(value.placement); link(value.link);
  array(value.bindings, "catalog bindings").forEach(binding); uniqueBy(value.bindings, (entry) => entry.variant, "catalog bindings");
  array(value.extensions, "catalog extensions", { empty: true }).forEach(extension);
  array(value.integer_capabilities, "catalog integer capabilities", { empty: true }).forEach(integerCapability);
  nullable(value.integer_audit, integerAudit, "catalog integer audit"); boolean(value.suppress_auto_output, "catalog suppress_auto_output");
}

export function catalogProvenance(value) {
  exact(value, ["identity", "provenance"], "compiled catalog provenance");
  exact(value.identity, ["builtin", "variant"], "compiled binding identity");
  exact(value.identity.builtin, ["name"], "compiled provenance builtin"); identity(value.identity.builtin.name, "compiled provenance name"); nonempty(value.identity.variant, "compiled provenance variant");
  exact(value.provenance, ["source_file", "module_path"], "compiled catalog source provenance"); repositoryPath(value.provenance.source_file, "catalog provenance source file"); rustModulePath(value.provenance.module_path, "catalog provenance module path");
}

export function catalogAlias(value) {
  exact(value, ["alias", "canonical", "provenance"], "compiled catalog alias");
  exact(value.alias, ["name"], "catalog alias identity");
  exact(value.canonical, ["name"], "catalog alias target");
  const alias = identity(value.alias.name, "catalog alias name").toLowerCase();
  const canonical = identity(value.canonical.name, "catalog alias canonical name").toLowerCase();
  if (alias === canonical) throw new Error("catalog alias cannot target itself");
  exact(value.provenance, ["source_file", "module_path"], "catalog alias provenance");
  repositoryPath(value.provenance.source_file, "catalog alias source file");
  rustModulePath(value.provenance.module_path, "catalog alias module path");
}

export function constant(value) {
  exact(value, ["name", "kind", "provenance"], "compiled constant");
  identity(value.name, "constant name");
  enumValue(value.kind, ["real_double", "complex_double", "logical"], "constant kind");
  exact(value.provenance, ["source_file", "module_path"], "constant catalog provenance");
  repositoryPath(value.provenance.source_file, "constant catalog source file");
  rustModulePath(value.provenance.module_path, "constant catalog module path");
}

export function legacyFunction(value) {
  exact(value, ["name", "description", "category", "parameter_types", "return_type", "resolver", "semantic_authority", "semantics", "accelerator_tags", "is_sink", "suppress_auto_output", "execution_stack", "required_capabilities", "descriptor", "extensions", "integer_capabilities", "integer_audit"], "compiled legacy function");
  identity(value.name, "legacy function name"); text(value.description, "legacy description"); text(value.category, "legacy category");
  array(value.parameter_types, "legacy parameter types", { empty: true }).forEach(typeValue); typeValue(value.return_type);
  enumValue(value.resolver, ["none", "simple", "with_context"], "legacy resolver"); enumValue(value.semantic_authority, ["catalog", "name_table", "derived"], "legacy semantic authority"); semantics(value.semantics);
  uniqueStrings(value.accelerator_tags, "legacy accelerator tags", { empty: true }).forEach((entry) => enumValue(entry, ["unary", "elementwise", "reduction", "matmul", "transpose", "array_construct"], "accelerator tag"));
  boolean(value.is_sink, "legacy is_sink"); boolean(value.suppress_auto_output, "legacy suppress_auto_output"); enumValue(value.execution_stack, ["Any", "Process"], "legacy execution stack");
  array(value.required_capabilities, "legacy capabilities", { empty: true }).forEach((entry) => enumValue(entry, CAPABILITIES, "capability"));
  nullable(value.descriptor, descriptor, "legacy descriptor"); array(value.extensions, "legacy extensions", { empty: true }).forEach(extension); array(value.integer_capabilities, "legacy integer capabilities", { empty: true }).forEach(integerCapability); nullable(value.integer_audit, integerAudit, "legacy integer audit");
}

export function legacyDocumentation(value) {
  exact(value, ["name", "category", "summary", "keywords", "errors", "related", "introduced", "status", "examples"], "compiled legacy documentation"); identity(value.name, "legacy documentation name");
  for (const field of ["category", "summary", "keywords", "errors", "related", "introduced", "status", "examples"]) nullable(value[field], (entry) => text(entry, `legacy documentation ${field}`), `legacy documentation ${field}`);
}

function documentation(value) {
  exact(value, ["authority", "title", "slug", "summary", "description", "keywords", "related", "sections", "examples", "example_exemption", "faqs", "links", "media", "evidence", "introduced", "status"], "catalog documentation");
  enumValue(value.authority, ["Catalog", "LegacySidecar"], "documentation authority"); optionalText(value.title, "documentation title"); optionalText(value.slug, "documentation slug"); text(value.summary, "documentation summary"); text(value.description, "documentation description");
  strings(value.keywords, "documentation keywords"); strings(value.related, "documentation related");
  array(value.sections, "documentation sections", { empty: true }).forEach((entry) => { exact(entry, ["heading", "paragraphs"], "documentation section"); nonempty(entry.heading, "section heading"); strings(entry.paragraphs, "section paragraphs"); });
  array(value.examples, "documentation examples", { empty: true }).forEach(example); optionalText(value.example_exemption, "example exemption");
  array(value.faqs, "documentation faqs", { empty: true }).forEach((entry) => { exact(entry, ["question", "answer"], "documentation faq"); nonempty(entry.question, "faq question"); nonempty(entry.answer, "faq answer"); });
  array(value.links, "documentation links", { empty: true }).forEach(docLink); array(value.media, "documentation media", { empty: true }).forEach((entry) => { exact(entry, ["kind", "url", "alt"], "documentation media"); enumValue(entry.kind, ["Image", "Video"], "media kind"); nonempty(entry.url, "media url"); nonempty(entry.alt, "media alt"); });
  exact(value.evidence, ["implementation", "verification", "notes"], "documentation evidence"); array(value.evidence.implementation, "implementation links", { empty: true }).forEach(docLink); array(value.evidence.verification, "verification evidence", { empty: true }).forEach((entry) => { exact(entry, ["kind", "label", "location"], "verification reference"); enumValue(entry.kind, ["UnitTest", "IntegrationTest", "BrowserTest", "ProviderTest", "WgpuTest", "ConformanceTest", "Validation"], "evidence kind"); nonempty(entry.label, "evidence label"); nonempty(entry.location, "evidence location"); }); strings(value.evidence.notes, "evidence notes"); optionalText(value.introduced, "documentation introduced"); if (value.status !== null) enumValue(value.status, ["Stable", "Experimental", "Partial", "Deprecated"], "documentation status");
}

function descriptor(value) { exact(value, ["signatures", "output_mode", "completion_policy", "errors"], "builtin descriptor"); array(value.signatures, "signatures", { empty: true }).forEach((entry) => { exact(entry, ["label", "inputs", "outputs"], "signature"); nonempty(entry.label, "signature label"); array(entry.inputs, "signature inputs", { empty: true }).forEach(parameter); array(entry.outputs, "signature outputs", { empty: true }).forEach(parameter); }); enumValue(value.output_mode, ["Fixed", "ByRequestedOutputCount"], "output mode"); enumValue(value.completion_policy, ["Public", "MethodOnly", "HiddenInternal"], "completion policy"); array(value.errors, "descriptor errors", { empty: true }).forEach((entry) => { exact(entry, ["code", "identifier", "when", "message"], "error descriptor"); nonempty(entry.code, "error code"); optionalText(entry.identifier, "error identifier"); nonempty(entry.when, "error condition"); nonempty(entry.message, "error message"); }); }
function parameter(value) { exact(value, ["name", "ty", "arity", "default", "description"], "parameter"); nonempty(value.name, "parameter name"); enumValue(value.ty, ["Any", "NumericScalar", "IntegerScalar", "StringScalar", "NumericArray", "LogicalArray", "SizeArg", "LikePrototype", "AxesHandle", "StyleSpec", "PropertyName", "PropertyValue", "Callable"], "parameter type"); enumValue(value.arity, ["Required", "Optional", "Variadic"], "parameter arity"); optionalText(value.default, "parameter default"); text(value.description, "parameter description"); }

function contract(value) { exact(value, ["maturity", "inference_rule", "compatibility", "async_behavior", "purity", "semantic_kind", "workspace_effect", "environment_effect", "effects", "capabilities"], "builtin contract"); enumValue(value.maturity, ["Complete", "DynamicByDesign", "LegacyResolver", "Incomplete"], "contract maturity"); inference(value.inference_rule); enumValue(value.compatibility, ["Matlab", "InteractiveOnly"], "compatibility"); enumValue(value.async_behavior, ["NeverSuspends", "MaySuspend", "RequiresAsyncRuntime"], "async behavior"); enumValue(value.purity, ["Pure", "DeterministicReadOnly", "Impure"], "purity"); rustEnum(value.semantic_kind, SEMANTIC_ENUM, "semantic kind"); nullableEnum(value.workspace_effect, ["ReadsWorkspace", "CreatesBinding", "ClearsBinding", "ClearsFunctionCache", "LoadsExternalBindings", "DynamicEval"], "workspace effect"); nullableEnum(value.environment_effect, ["PathMutation", "WorkingDirectoryMutation", "FunctionCacheInvalidation", "DynamicLookupInvalidation"], "environment effect"); array(value.effects, "effects", { empty: true }).forEach((entry) => enumValue(entry, EFFECTS, "effect")); array(value.capabilities, "capabilities", { empty: true }).forEach((entry) => enumValue(entry, CAPABILITIES, "capability")); }

const SEMANTIC_ENUM = { units: ["General", "Elementwise", "ArrayConstructor", "ParameterizedArrayConstructor", "PermutationConstructor", "RangeConstructor", "EmptyConstructor", "Reduction", "LinearAlgebra", "Plotting", "Filesystem", "Network", "Workspace"], children: { ShapeTransform: { units: ["General", "Reshape", "Permute", "Repmat", "Dot", "Transpose"], children: { Concatenate: { units: ["Dimension", "Horizontal", "Vertical"] } } } } };

function inference(value) { rustEnum(value, INFERENCE_ENUM, "inference rule"); }
const units = (...values) => ({ units: values });
const INFERENCE_ENUM = { children: {
  Identity: { empty: true }, Acceleration: units("Arrayfun", "Gather", "GpuArray"), Aggregate: units("Struct"), Introspection: units("Feval"), Parallel: units("Barrier", "Broadcast", "Cat", "Codistributed", "CodistributedBuild", "Codistributor", "Codistributor1d", "Codistributor2dbc", "CodistributorIsComplete", "Distributed", "FetchNext", "FetchOutputs", "FunctionalReduce", "Gcp", "GetCodistributor", "GetCurrentJob", "GetCurrentTask", "GetCurrentWorker", "GlobalIndices", "Gplus", "Iscodistributed", "LocalPart", "Parfeval", "ParfevalOnAll", "Parpool", "Probe", "Receive", "Redistribute", "Send", "SendReceive", "SpmdIndex", "SpmdSize"),
  Stats: { children: { Random: units("Binomial", "Gamma", "Weibull") } },
  Array: { children: { Accumulation: units("Indexed"), Binning: units("Discretize"), Combinatorics: units("CartesianProduct", "Permutations", "SelectionCombinations"), Creation: units("Full", "Zeros"), Grouping: units("Counts", "GroupedApply", "IndexLabels", "SortedGroups"), Introspection: { children: { ShapePredicate: units("Empty", "Scalar", "Vector", "Matrix", "Row", "Column"), ShapeQuery: units("Size", "ElementCount"), ShapeScalarQuery: units("Length", "Rank", "Height", "Width") } } } },
  Io: { children: { Console: units("ClearConsole"), ReplFs: { children: { WorkingDirectory: units("Change", "Current"), Environment: units("Read", "Exists", "Set", "Remove"), SourceInventory: units("FolderContents"), Directory: { children: { Lifecycle: units("Create", "Remove"), Listing: units("Metadata", "Names") } }, File: { children: { Transfer: units("Copy", "Move") } }, Path: { children: { Installation: units("Root"), Predicate: units("File", "Folder"), Syntax: units("Join", "Split", "FileSeparator", "PathListSeparator"), Search: units("QueryOrReplace", "Add", "Remove", "Generate", "Persist"), Temporary: units("Directory", "UniqueName") } } } } } },
  Logical: { children: { Elementwise: { children: { Binary: units("And", "Or", "Xor"), Unary: units("Not") } }, MetadataPredicate: units("Cell", "CellString", "GpuArray", "Logical", "Numeric", "Real", "Sparse"), NumericClassification: units("Finite", "Infinite", "Nan"), Relational: units("Equal", "NotEqual", "LessThan", "LessThanOrEqual", "GreaterThan", "GreaterThanOrEqual"), ScalarReduction: units("AllFinite") } },
  Math: { units: ["Atan2", "Bsxfun", "ComplexConstruction", "Heaviside", "IntegerDivide", "Hypot", "Rescale", "Typecast", "Round"], children: { AngleConversion: units("DegreesToRadians", "RadiansToDegrees"), BinaryArithmetic: units("Add", "Subtract", "Multiply", "RightDivide", "LeftDivide", "Power"), MatrixArithmetic: units("Multiply", "LeftDivide", "RightDivide", "Power"), MagnitudePhaseSign: units("Magnitude", "Phase", "Sign"), Exponential: units("Natural", "MinusOne"), Logarithm: units("Natural", "OnePlus", "Binary", "Common"), LogicalReduction: units("All", "Any"), Root: units("Principal", "RealOnly"), NumericConversion: units(...NUMERIC_CLASSES), NumericConversionWithLike: units(...NUMERIC_CLASSES), NumericComponent: units("Conjugate", "ImaginaryPart", "RealPart"), Rounding: units("Ceil", "Fix", "Floor"), Remainder: units("Modulus", "Remainder"), Trigonometric: units("Sin", "Cos", "Tan"), Hyperbolic: units("Sine", "Cosine", "Tangent"), PiScaledTrigonometric: units("Sin", "Cos"), PowerOfTwo: units("NextExponent", "Power"), DegreeTrigonometric: units("Sin", "Cos", "Tan"), InverseTrigonometric: units("Sine", "Cosine", "Tangent"), InverseHyperbolic: units("Cosine", "Sine", "Tangent"), ErrorFunction: units("Erf", "InverseComplementary"), GammaFunction: units("Gamma", "LogGamma"), Discrete: { units: ["Factor", "Factorial", "IsPrime", "Primes"], children: { Binary: units("Gcd", "Lcm") } }, Bitwise: { units: ["Complement", "Get", "Set", "Shift", "SwapBytes"], children: { Binary: units("And", "Or", "Xor") } }, NumericLimit: { children: { Integer: units("Minimum", "Maximum"), Floating: units("SmallestNormal", "LargestFinite", "LargestConsecutiveInteger") } } } }
} };

function placement(value) { exact(value, ["portability", "accelerator", "residency", "fusion", "distributed"], "placement"); enumValue(value.portability, ["NativeAndWasm", "NativeOnly", "WasmHostBridge"], "portability"); enumValue(value.accelerator, ["Forbidden", "Optional", "Required"], "accelerator policy"); enumValue(value.residency, ["Host", "PreserveInputs", "ProduceResident", "GatherToHost", "Dynamic"], "residency"); enumValue(value.fusion, ["Never", "Candidate", "Boundary"], "fusion"); rustEnum(value.distributed, { units: ["Unsupported", "InspectHandles", "MaterializeArguments", "MapUnary", "ScalarLikePrototype"], children: { MapUnaryConstrained: { object: (entry) => { exact(entry, ["numeric_classes"], "distributed map contract"); array(entry.numeric_classes, "distributed numeric classes", { empty: true }).forEach((item) => enumValue(item, NUMERIC_CLASSES, "numeric class")); } } } }, "distributed policy"); }
function link(value) { exact(value, ["reachability", "policy", "execution_stack", "artifact_dependencies"], "link contract"); rustEnum(value.reachability, { units: ["Always", "Dynamic"], children: { Feature: { scalar: nonempty } } }, "reachability"); enumValue(value.policy, ["PortableRuntime", "HostRuntime", "NativeSymbol", "ForeignRuntime"], "link policy"); enumValue(value.execution_stack, ["Any", "Process"], "link execution stack"); strings(value.artifact_dependencies, "artifact dependencies"); }
function binding(value) { exact(value, ["variant", "availability"], "binding declaration"); nonempty(value.variant, "binding variant"); enumValue(value.availability, ["Required", "TargetConditional"], "binding availability"); }
function extension(value) { exact(value, ["id", "mode", "description", "error_identifier"], "extension"); nonempty(value.id, "extension id"); if (value.mode !== "RunMatOnly") throw new Error("unsupported extension mode"); nonempty(value.description, "extension description"); optionalText(value.error_identifier, "extension error identifier"); }

function integerCapability(value) { exact(value, ["form", "inputs", "computation_domain", "output_class", "overflow", "backend", "overload", "notes"], "integer capability"); nonempty(value.form, "integer form"); array(value.inputs, "integer inputs", { empty: true }).forEach((entry) => { exact(entry, ["name", "classes", "availability", "scalar_double", "notes"], "integer input"); nonempty(entry.name, "integer input name"); array(entry.classes, "integer classes", { empty: true }).forEach((item) => enumValue(item, ["Int8", "Int16", "Int32", "Int64", "Uint8", "Uint16", "Uint32", "Uint64"], "integer class")); enumValue(entry.availability, ["Documented", "RunMatOnly", "Rejected"], "integer availability"); enumValue(entry.scalar_double, ["NotApplicable", "Allowed", "AllowedExceptWith64BitInteger", "Rejected"], "scalar double rule"); text(entry.notes, "integer input notes"); }); enumValue(value.computation_domain, ["ExactInteger", "FloatingPoint", "Predicate", "Structural", "FunctionSpecific"], "integer computation domain"); enumValue(value.output_class, ["PreserveInput", "PreserveNondoubleInput", "Double", "Logical", "OptionDependent", "NotApplicable", "FunctionSpecific"], "integer output class"); enumValue(value.overflow, ["Saturate", "Error", "NotApplicable", "EvidenceOpen", "FunctionSpecific"], "integer overflow"); enumValue(value.backend, ["HostOnly", "HostAndGpu", "GatherFallback", "GpuRestricted", "FunctionSpecific"], "integer backend"); enumValue(value.overload, ["ScalarOnly", "ElementwiseShapePreserving", "SameSizeOrScalar", "BroadcastCompatible", "StructuralParameter", "Multiple", "FunctionSpecific"], "integer overload"); text(value.notes, "integer notes"); }
function integerAudit(value) { const fields = value.canonical_builtin === undefined ? ["kind", "notes"] : ["kind", "canonical_builtin", "notes"]; exact(value, fields, "integer audit"); enumValue(value.kind, ["AliasOf", "NotApplicable"], "integer audit kind"); if (value.canonical_builtin !== undefined) identity(value.canonical_builtin, "integer canonical builtin"); text(value.notes, "integer audit notes"); }
function semantics(value) { exact(value, ["compatibility", "async_behavior", "effects", "workspace_effect", "environment_effect", "purity", "semantic_kind"], "legacy semantics"); enumValue(value.compatibility, ["Matlab", "InteractiveOnly"], "semantic compatibility"); enumValue(value.async_behavior, ["NeverSuspends", "MaySuspend", "RequiresAsyncRuntime"], "semantic async"); exact(value.effects, ["workspace", "environment", "filesystem", "network", "ui", "random", "time", "host_callback", "unknown"], "semantic effects"); Object.values(value.effects).forEach((entry) => boolean(entry, "semantic effect")); nullableEnum(value.workspace_effect, ["ReadsWorkspace", "CreatesBinding", "ClearsBinding", "ClearsFunctionCache", "LoadsExternalBindings", "DynamicEval"], "workspace effect"); nullableEnum(value.environment_effect, ["PathMutation", "WorkingDirectoryMutation", "FunctionCacheInvalidation", "DynamicLookupInvalidation"], "environment effect"); enumValue(value.purity, ["Pure", "DeterministicReadOnly", "Impure"], "semantic purity"); rustEnum(value.semantic_kind, SEMANTIC_ENUM, "semantic kind"); }

const TYPE_UNITS = Object.freeze(["Int", "Num", "Bool", "String", "Symbolic", "Void", "Unknown"]);
const TYPE_VARIANTS = Object.freeze({
  Logical: (value) => shapedType(value, "Logical"),
  Tensor: (value) => shapedType(value, "Tensor"),
  SymbolicArray: (value) => shapedType(value, "SymbolicArray"),
  Object: objectType,
  Cell: cellType,
  Function: functionType,
  Union: (value) => typeList(value, "Union"),
  OutputList: (value) => typeList(value, "OutputList"),
  Struct: structType,
});

function typeValue(value) {
  if (typeof value === "string") { enumValue(value, TYPE_UNITS, "type"); return; }
  object(value, "type");
  const tags = Object.keys(value);
  if (tags.length !== 1) throw new Error("type enum must contain exactly one variant");
  const parser = TYPE_VARIANTS[tags[0]];
  if (!parser) throw new Error(`unsupported type variant ${tags[0]}`);
  parser(value[tags[0]]);
}

function shapedType(value, tag) { exact(value, ["shape"], `${tag} type`); shape(value.shape, `${tag} shape`); }
function objectType(value) { exact(value, ["class_name", "shape"], "Object type"); shape(value.shape, "Object shape"); if (value.class_name !== null) nonempty(value.class_name, "object class identity"); }
function cellType(value) { exact(value, ["element_type", "length"], "cell type"); nullable(value.element_type, typeValue, "cell element type"); optionalInteger(value.length, "cell length"); }
function functionType(value) { exact(value, ["params", "returns"], "function type"); array(value.params, "function params", { empty: true }).forEach(typeValue); typeValue(value.returns); }
function typeList(value, tag) { array(value, `${tag} types`, { empty: true }).forEach(typeValue); }
function structType(value) { exact(value, ["known_fields"], "struct type"); if (value.known_fields !== null) strings(value.known_fields, "known fields"); }
function shape(value, label) { if (value === null) return; array(value, label, { empty: true }).forEach((entry) => { if (entry !== null) integer(entry, label); }); }
function example(value) { exact(value, ["id", "title", "program", "display_output", "compatibility", "harness", "fixture", "requirements", "verification"], "example"); nonempty(value.id, "example id"); nonempty(value.title, "example title"); nonempty(value.program, "example program"); optionalText(value.display_output, "example output"); enumValue(value.compatibility, ["RunMat", "Matlab", "Strict"], "example compatibility"); enumValue(value.harness, ["Portable", "Native", "Browser", "BrowserGraphics", "NativeFilesystem", "NativeLoopbackNetwork", "Wgpu", "NativeForeignRuntime", "InteractiveHost"], "example harness"); validateBuiltinExampleRequirements(value.requirements); validateBuiltinExampleFixture(value.fixture, "example fixture", { program: value.program, harness: value.harness, requirements: value.requirements }); rustEnum(value.verification, { units: ["Succeeds"], children: { Assertions: { object: (entry) => { exact(entry, ["source"], "assertions verification"); nonempty(entry.source, "assertions source"); } }, ExpectedError: { object: (entry) => { exact(entry, ["identifier"], "expected error verification"); nonempty(entry.identifier, "expected error identifier"); } }, Figure: { object: (entry) => { exact(entry, ["minimum_figures", "assertions"], "figure verification"); integer(entry.minimum_figures, "minimum figures"); nonempty(entry.assertions, "figure assertions"); } } } }, "example verification"); }
function docLink(value) { exact(value, ["label", "target"], "documentation link"); nonempty(value.label, "link label"); rustEnum(value.target, { children: { Builtin: { scalar: nonempty }, Documentation: { scalar: nonempty }, Source: { scalar: nonempty }, External: { scalar: nonempty } } }, "link target"); }

function rustEnum(value, grammar, label) { if (typeof value === "string") { enumValue(value, grammar.units ?? [], label); return; } object(value, label); const keys = Object.keys(value); if (keys.length !== 1) throw new Error(`${label} must contain exactly one enum variant`); const tag = keys[0]; const child = grammar.children?.[tag]; if (!child) throw new Error(`${label} has unsupported variant ${tag}`); const inner = value[tag]; if (child.empty) exact(inner, [], `${label} ${tag}`); else if (child.scalar) child.scalar(inner, `${label} ${tag}`); else if (child.object) child.object(inner); else rustEnum(inner, child, `${label} ${tag}`); }
function strings(value, label) { array(value, label, { empty: true }).forEach((entry) => text(entry, label)); }
function text(value, label) { if (typeof value !== "string") throw new Error(`${label} must be a string`); return value; }
function optionalText(value, label) { if (value !== null) text(value, label); }
function optionalInteger(value, label) { if (value !== null) integer(value, label); }
function nullable(value, parser, label) { if (value !== null) parser(value); }
function nullableEnum(value, allowed, label) { if (value !== null) enumValue(value, allowed, label); }
export function uniqueBy(values, key, label) { const seen = new Set(); for (const entry of values) { const id = key(entry); if (seen.has(id)) throw new Error(`${label} must be unique`); seen.add(id); } }
export function sortedBy(values, key, label) { for (let index = 1; index < values.length; index += 1) if (compareCodePoint(key(values[index - 1]), key(values[index])) > 0) throw new Error(`${label} must use canonical order`); }
