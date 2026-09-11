// Closed wire-schema validation shared by catalog and example-inventory consumers.

export const NO_EXAMPLE_FIXTURE = "None";
export const NO_EXAMPLE_REQUIREMENTS = Object.freeze({
  host: "Any",
  engine: "Default",
  compiler: Object.freeze([]),
  runtime: Object.freeze([]),
  toolchain: Object.freeze([])
});

const MAX_FILESYSTEM_ENTRIES = 64;
const MAX_LOOPBACK_EXCHANGES = 32;
const MAX_PAYLOAD_BYTES = 1024 * 1024;
const MAX_TRANSCRIPT_STEPS = 128;
const MAX_RELATIVE_PATH_BYTES = 1024;
const MAX_PATH_COMPONENT_BYTES = 240;
const MAX_HTTP_HEADERS = 64;
const MAX_HTTP_PATH_BYTES = 2048;
const MAX_HTTP_HEADER_NAME_BYTES = 256;
const MAX_HTTP_HEADER_VALUE_BYTES = 8192;
const ENDPOINT_TOKENS = {
  HttpBaseUrl: "__RUNMAT_HTTP_BASE_URL__",
  LoopbackHost: "__RUNMAT_LOOPBACK_HOST__",
  LoopbackPort: "__RUNMAT_LOOPBACK_PORT__"
};

export function validateBuiltinExampleFixture(value, label = "example fixture", context = null) {
  if (value === "None") {
    if (context) compatibility(value, context, label);
    return value;
  }
  const [tag, inner] = tagged(value, ["Filesystem", "Loopback", "ForeignAdapter", "CliInteraction", "DesktopHostOnly"], label);
  if (tag === "Filesystem") filesystem(inner, label);
  else if (tag === "Loopback") loopback(inner, label);
  else if (tag === "ForeignAdapter") foreignAdapter(inner, label);
  else if (tag === "CliInteraction") cliInteraction(inner, label);
  else desktopHost(inner, label);
  if (context) compatibility(value, context, label);
  return value;
}

export function validateBuiltinExampleRequirements(value, label = "example requirements") {
  exact(value, ["host", "engine", "compiler", "runtime", "toolchain"], label);
  member(value.host, ["Any", "NativeOnly", "DesktopHostOnly"], `${label} host`);
  member(value.engine, ["Default", "Interpreter", "Jit", "Aot"], `${label} engine`);
  orderedMembers(value.compiler, ["C", "Cxx", "Fortran", "Cuda", "JavaBytecode", "PythonExtension"], `${label} compiler capabilities`);
  orderedMembers(value.runtime, ["NativeDynamicLoader", "Mex", "NativeFfi", "JavaVirtualMachine", "Python", "NumPy"], `${label} runtime capabilities`);
  orderedMembers(value.toolchain, ["CCompiler", "CxxCompiler", "FortranCompiler", "CudaToolkit", "JavaDevelopmentKit", "PythonInterpreter", "PythonDevelopmentHeaders"], `${label} toolchain capabilities`);
  return value;
}

function filesystem(value, label) {
  exact(value, ["id", "root", "entries"], `${label} filesystem`);
  fixtureId(value.id, label);
  member(value.root, ["IsolatedWorkspace"], `${label} filesystem root`);
  const entries = bounded(value.entries, 1, MAX_FILESYSTEM_ENTRIES, `${label} filesystem entries`);
  const paths = new Set();
  const orderedPaths = [];
  let aggregateBytes = 0;
  entries.forEach((entry) => {
    const [tag, inner] = tagged(entry, ["Directory", "File"], `${label} filesystem entry`);
    if (tag === "Directory") exact(inner, ["relative_path"], `${label} directory`);
    else {
      exact(inner, ["relative_path", "content"], `${label} file`);
      const [contentTag, content] = tagged(inner.content, ["Utf8", "Bytes"], `${label} file content`);
      if (contentTag === "Utf8") string(content, `${label} UTF-8 content`);
      else bytes(content, `${label} byte content`);
    }
    const path = string(inner.relative_path, `${label} relative path`);
    if (!safeRelativePath(path)) throw new Error(`${label} relative path must be normalized and relative`);
    if (["example.m", "runmat.toml"].includes(path)) throw new Error(`${label} relative path is reserved by the example runner`);
    if (paths.has(path)) throw new Error(`${label} filesystem paths must be unique`);
    paths.add(path);
    orderedPaths.push([path, tag === "File"]);
    if (tag === "File") aggregateBytes += inner.content.Bytes?.length ?? new TextEncoder().encode(inner.content.Utf8).length;
  });
  if (!orderedPaths.every(([path], index) => index === 0 || compareUtf8(orderedPaths[index - 1][0], path) < 0)) throw new Error(`${label} filesystem entries must use canonical path order`);
  if (orderedPaths.some(([file, isFile]) => isFile && orderedPaths.some(([path]) => path.startsWith(`${file}/`)))) throw new Error(`${label} cannot place entries below a file`);
  if (aggregateBytes > MAX_PAYLOAD_BYTES) throw new Error(`${label} aggregate payload exceeds its bound`);
}

function loopback(value, label) {
  exact(value, ["id", "scenario", "endpoint_substitutions"], `${label} loopback`);
  fixtureId(value.id, label);
  orderedMembers(value.endpoint_substitutions, ["HttpBaseUrl", "LoopbackHost", "LoopbackPort"], `${label} endpoint substitutions`);
  const [tag, scenario] = tagged(value.scenario, ["Http", "Tcp"], `${label} loopback scenario`);
  if (tag === "Tcp" && value.endpoint_substitutions.includes("HttpBaseUrl")) throw new Error(`${label} TCP scenario cannot declare an HTTP base URL`);
  exact(scenario, ["exchanges"], `${label} ${tag} scenario`);
  let aggregateBytes = 0;
  bounded(scenario.exchanges, 1, MAX_LOOPBACK_EXCHANGES, `${label} ${tag} exchanges`).forEach((exchange) => {
    if (tag === "Http") {
      httpExchange(exchange, label);
      aggregateBytes += (exchange.request.body?.length ?? 0) + exchange.response.body.length;
    }
    else {
      exact(exchange, ["client_bytes", "server_bytes"], `${label} TCP exchange`);
      bytes(exchange.client_bytes, `${label} TCP client bytes`);
      bytes(exchange.server_bytes, `${label} TCP server bytes`);
      aggregateBytes += exchange.client_bytes.length + exchange.server_bytes.length;
    }
  });
  if (aggregateBytes > MAX_PAYLOAD_BYTES) throw new Error(`${label} aggregate loopback payload exceeds its bound`);
}

function httpExchange(value, label) {
  exact(value, ["request", "response"], `${label} HTTP exchange`);
  exact(value.request, ["method", "path", "body"], `${label} HTTP request`);
  member(value.request.method, ["Get", "Head", "Post", "Put", "Patch", "Delete"], `${label} HTTP method`);
  string(value.request.path, `${label} HTTP path`);
  if (value.request.body !== null) bytes(value.request.body, `${label} HTTP request body`);
  exact(value.response, ["status", "headers", "body"], `${label} HTTP response`);
  integer(value.response.status, `${label} HTTP status`);
  if (value.response.status < 100 || value.response.status > 599) throw new Error(`${label} HTTP status is outside 100..599`);
  if (!value.request.path.startsWith("/") || /\s/u.test(value.request.path) || byteLength(value.request.path) > MAX_HTTP_PATH_BYTES) throw new Error(`${label} HTTP path is invalid`);
  const headers = new Set();
  bounded(value.response.headers, 0, MAX_HTTP_HEADERS, `${label} HTTP headers`).forEach((header) => {
    exact(header, ["name", "value"], `${label} HTTP header`);
    string(header.name, `${label} HTTP header name`);
    string(header.value, `${label} HTTP header value`);
    const folded = header.name.toLowerCase();
    if (!/^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/u.test(header.name)
        || byteLength(header.name) > MAX_HTTP_HEADER_NAME_BYTES
        || byteLength(header.value) > MAX_HTTP_HEADER_VALUE_BYTES
        || [...header.value].some((character) => character === "\u007f" || (character < "\u0020" && character !== "\t"))
        || headers.has(folded)) throw new Error(`${label} HTTP header is invalid`);
    headers.add(folded);
  });
  bytes(value.response.body, `${label} HTTP response body`);
}

function foreignAdapter(value, label) {
  exact(value, ["id", "files", "preparation"], `${label} foreign adapter`);
  fixtureId(value.id, label);
  filesystem(value.files, `${label} foreign files`);
  foreignPreparation(value.preparation, value.files, label);
}

function foreignPreparation(value, files, label) {
  const [tag, preparation] = tagged(value, ["Mex", "NativeFfi", "Java", "Python"], `${label} foreign preparation`);
  if (tag === "Mex") {
    exact(preparation, ["module_name", "api", "translation_units", "include_directories", "definitions"], `${label} MEX preparation`);
    identifier(preparation.module_name, `${label} MEX module name`);
    member(preparation.api, ["R2017b", "R2018a", "LargeArrayDims", "CompatibleArrayDims"], `${label} MEX API`);
    nativeBuild(preparation, files, label);
  } else if (tag === "NativeFfi") {
    exact(preparation, ["isolation", "library_name", "translation_units", "include_directories", "definitions", "interface"], `${label} native FFI preparation`);
    member(preparation.isolation, ["InProcess", "OutOfProcess"], `${label} native FFI isolation`);
    identifier(preparation.library_name, `${label} native library name`);
    nativeBuild(preparation, files, label);
    exact(preparation.interface, ["interface_name", "primary_header", "additional_headers", "include_directories", "definitions"], `${label} native interface`);
    identifier(preparation.interface.interface_name, `${label} native interface name`);
    fixtureFile(preparation.interface.primary_header, files, `${label} native interface primary header`);
    orderedPaths(preparation.interface.additional_headers, `${label} native interface additional headers`, (path) => fixtureFile(path, files, `${label} native interface additional header`));
    orderedPaths(preparation.interface.include_directories, `${label} native interface include directories`);
    definitions(preparation.interface.definitions, label);
  } else if (tag === "Java") {
    exact(preparation, ["artifact_name", "release", "source_files", "resources", "compile_classpath"], `${label} Java preparation`);
    artifactName(preparation.artifact_name, `${label} Java artifact name`);
    integer(preparation.release, `${label} Java release`);
    if (preparation.release < 8 || preparation.release > 99) throw new Error(`${label} Java release is outside its supported schema range`);
    if (!Array.isArray(preparation.source_files) || preparation.source_files.length === 0) throw new Error(`${label} Java preparation requires a source file`);
    orderedPaths(preparation.source_files, `${label} Java source files`, (path) => fixtureFile(path, files, `${label} Java source file`));
    const resourcePaths = list(preparation.resources, `${label} Java resources`).map((resource) => {
      exact(resource, ["source_path", "artifact_path"], `${label} Java resource`);
      fixtureFile(resource.source_path, files, `${label} Java resource source`);
      relativePath(resource.artifact_path, `${label} Java resource artifact path`);
      return resource.artifact_path;
    });
    canonicalStrings(resourcePaths, `${label} Java resource artifact paths`);
    orderedPaths(preparation.compile_classpath, `${label} Java compile classpath`, (path) => fixtureFile(path, files, `${label} Java classpath file`));
    if (preparation.source_files.some((path) => preparation.resources.some((resource) => resource.source_path === path))) throw new Error(`${label} Java source and resource files must be disjoint`);
  } else {
    exact(preparation, ["isolation", "environment", "artifact"], `${label} Python preparation`);
    member(preparation.isolation, ["InProcess", "OutOfProcess"], `${label} Python isolation`);
    exact(preparation.environment, ["implementation", "major", "minor"], `${label} Python environment`);
    member(preparation.environment.implementation, ["Cpython"], `${label} Python implementation`);
    integer(preparation.environment.major, `${label} Python major version`);
    integer(preparation.environment.minor, `${label} Python minor version`);
    if (preparation.environment.major !== 3 || preparation.environment.minor > 99) throw new Error(`${label} requires an explicit CPython 3 minor version`);
    const [artifactTag, artifact] = tagged(preparation.artifact, ["SourceTree", "Wheel"], `${label} Python artifact`);
    if (artifactTag === "SourceTree") {
      exact(artifact, ["module_root", "modules"], `${label} Python source tree`);
      relativePath(artifact.module_root, `${label} Python module root`);
      if (!Array.isArray(artifact.modules) || artifact.modules.length === 0) throw new Error(`${label} Python source tree requires a module`);
      orderedStrings(artifact.modules, `${label} Python modules`);
      artifact.modules.forEach((module) => qualifiedIdentifier(module, `${label} Python module`));
      if (!fixtureFilePaths(files).some((path) => path.startsWith(`${artifact.module_root}/`))) throw new Error(`${label} Python module root contains no fixture file`);
    } else {
      exact(artifact, ["artifact_name", "relative_path", "module", "compatibility"], `${label} Python wheel`);
      artifactName(artifact.artifact_name, `${label} Python artifact name`);
      fixtureFile(artifact.relative_path, files, `${label} Python wheel`);
      if (!artifact.relative_path.endsWith(".whl")) throw new Error(`${label} Python wheel path must end in .whl`);
      qualifiedIdentifier(artifact.module, `${label} Python wheel module`);
      if (artifact.compatibility !== "Pure") {
        const [compatTag, native] = tagged(artifact.compatibility, ["Native"], `${label} Python wheel compatibility`);
        if (compatTag !== "Native") throw new Error(`${label} Python wheel compatibility is invalid`);
        exact(native, ["abi_tag", "platform_tag"], `${label} native Python wheel compatibility`);
        tag(native.abi_tag, `${label} Python ABI tag`);
        tag(native.platform_tag, `${label} Python platform tag`);
      }
    }
  }
}

function nativeBuild(value, files, label) {
  const units = bounded(value.translation_units, 1, MAX_FILESYSTEM_ENTRIES, `${label} native translation units`);
  const paths = units.map((unit) => {
    exact(unit, ["relative_path", "language"], `${label} native translation unit`);
    fixtureFile(unit.relative_path, files, `${label} native translation unit`);
    member(unit.language, ["C", "Cxx", "Fortran", "Cuda"], `${label} native source language`);
    return unit.relative_path;
  });
  canonicalStrings(paths, `${label} native translation units`);
  orderedPaths(value.include_directories, `${label} native include directories`);
  definitions(value.definitions, label);
}

function definitions(values, label) {
  const names = list(values, `${label} preprocessor definitions`).map((definition) => {
    exact(definition, ["name", "value"], `${label} preprocessor definition`);
    identifier(definition.name, `${label} preprocessor definition name`);
    if (definition.value !== null && (typeof definition.value !== "string" || /[\0\r\n]/u.test(definition.value))) throw new Error(`${label} preprocessor definition value is invalid`);
    return definition.name;
  });
  canonicalStrings(names, `${label} preprocessor definitions`);
}

function cliInteraction(value, label) {
  exact(value, ["id", "transcript"], `${label} CLI interaction`);
  fixtureId(value.id, label);
  const transcript = bounded(value.transcript, 1, MAX_TRANSCRIPT_STEPS, `${label} CLI transcript`);
  let aggregateBytes = 0;
  transcript.forEach((step) => {
    if (["SendEndOfInput", "SendInterrupt"].includes(step)) return;
    const [tag, content] = tagged(step, ["ExpectOutput", "SendLine", "SendBytes"], `${label} CLI transcript step`);
    if (tag === "SendBytes") {
      bytes(content, `${label} CLI bytes`);
      if (content.length === 0) throw new Error(`${label} CLI bytes must not be empty`);
      aggregateBytes += content.length;
    } else {
      if (!string(content, `${label} CLI text`)) throw new Error(`${label} CLI text must not be empty`);
      aggregateBytes += byteLength(content);
    }
  });
  if (aggregateBytes > MAX_PAYLOAD_BYTES) throw new Error(`${label} CLI transcript exceeds its aggregate byte bound`);
  const terminals = transcript.flatMap((step, index) => ["SendEndOfInput", "SendInterrupt"].includes(step) ? [index] : []);
  if (terminals.length !== 1 || transcript.slice(terminals[0] + 1).some((step) => typeof step !== "object" || !("ExpectOutput" in step))) {
    throw new Error(`${label} CLI transcript requires one terminal action followed only by output expectations`);
  }
}

function desktopHost(value, label) {
  exact(value, ["id", "scenario"], `${label} desktop host`);
  fixtureId(value.id, label);
  member(value.scenario, ["FilePicker", "FigureWindow", "InteractivePrompt"], `${label} desktop scenario`);
}

function fixtureId(value, label) {
  exact(value, ["local_name"], `${label} id`);
  string(value.local_name, `${label} local name`);
  if (!/^[a-z0-9-]+$/u.test(value.local_name) || byteLength(value.local_name) > 64) throw new Error(`${label} local name is invalid`);
}

function compatibility(fixture, context, label) {
  const { harness, program, requirements } = context;
  validateBuiltinExampleRequirements(requirements, `${label} requirements`);
  const nativeHarnesses = ["Native", "NativeFilesystem", "NativeLoopbackNetwork", "NativeForeignRuntime", "InteractiveHost"];
  if (requirements.host === "NativeOnly" && !nativeHarnesses.includes(harness)) throw new Error(`${label} native host is incompatible with its harness`);
  if (requirements.host === "DesktopHostOnly" && harness !== "InteractiveHost") throw new Error(`${label} desktop host is incompatible with its harness`);
  if (requirements.engine !== "Default" && requirements.host !== "NativeOnly") throw new Error(`${label} explicit engine requires a native host`);
  if (fixture === "None") {
    if (Object.values(ENDPOINT_TOKENS).some((token) => program.includes(token))) throw new Error(`${label} endpoint token requires a loopback fixture`);
    return;
  }
  const [tag, value] = Object.entries(fixture)[0];
  const expectedHarnesses = { Filesystem: ["Portable", "Browser", "NativeFilesystem"], Loopback: ["NativeLoopbackNetwork"], ForeignAdapter: ["NativeForeignRuntime"], CliInteraction: ["InteractiveHost"], DesktopHostOnly: ["InteractiveHost"] };
  if (!expectedHarnesses[tag].includes(harness)) throw new Error(`${label} is incompatible with its harness`);
  if (["Loopback", "ForeignAdapter", "CliInteraction"].includes(tag) && requirements.host !== "NativeOnly") throw new Error(`${label} requires a native-only host`);
  if (tag === "DesktopHostOnly" && requirements.host !== "DesktopHostOnly") throw new Error(`${label} requires a desktop-host-only boundary`);
  if (tag !== "Loopback" && Object.values(ENDPOINT_TOKENS).some((token) => program.includes(token))) throw new Error(`${label} endpoint token requires a loopback fixture`);
  if (tag === "Loopback") for (const substitution of value.endpoint_substitutions) {
    if (!program.includes(ENDPOINT_TOKENS[substitution])) throw new Error(`${label} endpoint substitution token is absent from the program`);
  }
  if (tag === "Loopback") for (const [substitution, token] of Object.entries(ENDPOINT_TOKENS)) {
    if (program.includes(token) && !value.endpoint_substitutions.includes(substitution)) throw new Error(`${label} program uses an undeclared endpoint substitution token`);
  }
  if (tag === "ForeignAdapter") foreignCapabilities(value.preparation, requirements, label);
}

function foreignCapabilities(preparation, requirements, label) {
  const [adapter, value] = Object.entries(preparation)[0];
  let runtime;
  if (adapter === "Java") {
    runtime = "JavaVirtualMachine";
    requireCapabilities(requirements, "JavaBytecode", runtime, "JavaDevelopmentKit", label);
    return;
  }
  if (adapter === "Python") {
    runtime = "Python";
    requireCapabilities(requirements, null, runtime, "PythonInterpreter", label);
    return;
  }
  runtime = adapter === "Mex" ? "Mex" : "NativeFfi";
  const languages = [...new Set(value.translation_units.map((unit) => unit.language))];
  for (const language of languages) {
    const [compiler, toolchain] = { C: ["C", "CCompiler"], Cxx: ["Cxx", "CxxCompiler"], Fortran: ["Fortran", "FortranCompiler"], Cuda: ["Cuda", "CudaToolkit"] }[language];
    requireCapabilities(requirements, compiler, runtime, toolchain, label);
  }
}

function requireCapabilities(requirements, compiler, runtime, toolchain, label) {
  if (!requirements.runtime.includes(runtime) || (compiler !== null && !requirements.compiler.includes(compiler)) || !requirements.toolchain.includes(toolchain)) throw new Error(`${label} is missing its adapter runtime, compiler, or toolchain capability`);
}

function tagged(value, allowed, label) {
  object(value, label);
  const keys = Object.keys(value);
  if (keys.length !== 1 || !allowed.includes(keys[0])) throw new Error(`${label} must contain one supported variant`);
  return [keys[0], value[keys[0]]];
}

function orderedMembers(value, allowed, label) {
  list(value, label).forEach((entry) => member(entry, allowed, label));
  if (new Set(value).size !== value.length || !value.every((entry, index) => index === 0 || allowed.indexOf(value[index - 1]) < allowed.indexOf(entry))) {
    throw new Error(`${label} must be sorted and unique`);
  }
}

function orderedPaths(value, label, validate = () => {}) {
  const paths = list(value, label);
  canonicalStrings(paths, label);
  paths.forEach((path) => {
    relativePath(path, label);
    validate(path);
  });
}

function orderedStrings(value, label) {
  const values = list(value, label);
  values.forEach((entry) => string(entry, label));
  canonicalStrings(values, label);
}

function canonicalStrings(values, label) {
  if (new Set(values).size !== values.length || !values.every((entry, index) => index === 0 || compareUtf8(values[index - 1], entry) < 0)) {
    throw new Error(`${label} must be sorted and unique`);
  }
}

function fixtureFile(path, files, label) {
  relativePath(path, label);
  if (!fixtureFilePaths(files).includes(path)) throw new Error(`${label} is not a declared fixture file`);
}

function fixtureFilePaths(files) {
  return files.entries.flatMap((entry) => "File" in entry ? [entry.File.relative_path] : []);
}

function relativePath(value, label) {
  string(value, label);
  if (!safeRelativePath(value)) throw new Error(`${label} must be a portable relative path`);
}

function identifier(value, label) {
  string(value, label);
  if (!/^[A-Za-z_][A-Za-z0-9_]{0,127}$/u.test(value)) throw new Error(`${label} is invalid`);
}

function qualifiedIdentifier(value, label) {
  string(value, label);
  value.split(".").forEach((part) => identifier(part, label));
}

function artifactName(value, label) {
  string(value, label);
  if (!/^[a-z0-9-]{1,64}$/u.test(value)) throw new Error(`${label} is invalid`);
}

function tag(value, label) {
  string(value, label);
  if (!/^[A-Za-z0-9_.-]{1,128}$/u.test(value)) throw new Error(`${label} is invalid`);
}

function bytes(value, label) {
  const values = list(value, label);
  if (values.length > MAX_PAYLOAD_BYTES) throw new Error(`${label} exceeds its byte bound`);
  values.forEach((entry) => {
    integer(entry, label);
    if (entry > 255) throw new Error(`${label} values must fit in one byte`);
  });
}

function exact(value, fields, label) {
  object(value, label);
  const actual = Object.keys(value).sort();
  const expected = [...fields].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`${label} has invalid fields`);
}

function list(value, label) {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value;
}

function bounded(value, minimum, maximum, label) {
  const values = list(value, label);
  if (values.length < minimum || values.length > maximum) throw new Error(`${label} is outside its length bound`);
  return values;
}

function safeRelativePath(value) {
  return value.length > 0 && byteLength(value) <= MAX_RELATIVE_PATH_BYTES && !value.startsWith("/") && !value.includes("\\") && !value.includes("\0")
    && value.split("/").every((part) => {
      if (!part || byteLength(part) > MAX_PATH_COMPONENT_BYTES || part === "." || part === ".." || /[. ]$/u.test(part) || /[:*?"<>|\u0000-\u001f]/u.test(part)) return false;
      return !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/iu.test(part);
    });
}

function byteLength(value) {
  return new TextEncoder().encode(value).length;
}

function compareUtf8(left, right) {
  const encoder = new TextEncoder();
  const a = encoder.encode(left);
  const b = encoder.encode(right);
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return a.length - b.length;
}

function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
}

function string(value, label) {
  if (typeof value !== "string") throw new Error(`${label} must be a string`);
  return value;
}

function integer(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} must be a nonnegative integer`);
}

function member(value, allowed, label) {
  if (!allowed.includes(value)) throw new Error(`${label} has an unsupported value`);
}
