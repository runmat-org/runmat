export type ScalarPrecision = "f32" | "f64";

export type BroadcastingMode = "none" | "elementwise" | "fixed";

export interface GpuSupport {
  elementwise: boolean;
  reduction: boolean;
  precisions: ScalarPrecision[];
  broadcasting: BroadcastingMode;
  notes?: string;
}

export interface FusionSpec {
  elementwise: boolean;
  reduction: boolean;
  max_inputs: number;
  constants: "inline" | "external";
  notes?: string;
}

export interface Tested {
  unit: string;
  integration: string;
  wgpu?: string;
}

export interface Example {
  id?: string;
  description: string;
  input: string;
  output?: string;
  compatibility?: "RunMat" | "Matlab" | "Strict";
  harness?:
    | "Portable"
    | "Native"
    | "Browser"
    | "BrowserGraphics"
    | "NativeFilesystem"
    | "NativeLoopbackNetwork"
    | "Wgpu"
    | "NativeForeignRuntime"
    | "InteractiveHost";
  fixture?: BuiltinExampleFixture;
  requirements?: BuiltinExampleRequirements;
  verification?:
    | "Succeeds"
    | { Assertions: { source: string } }
    | { ExpectedError: { identifier: string } }
    | { Figure: { minimum_figures: number; assertions: string } };
}

export type BuiltinExampleFixture =
  | "None"
  | { Filesystem: BuiltinFilesystemFixture }
  | { Loopback: BuiltinLoopbackFixture }
  | { ForeignAdapter: BuiltinForeignAdapterFixture }
  | { CliInteraction: BuiltinCliInteractionFixture }
  | { DesktopHostOnly: BuiltinDesktopHostFixture };

export interface BuiltinFilesystemFixture {
  id: { local_name: string };
  root: "IsolatedWorkspace";
  entries: Array<
    | { Directory: { relative_path: string } }
    | { File: { relative_path: string; content: { Utf8: string } | { Bytes: number[] } } }
  >;
}

export interface BuiltinLoopbackFixture {
  id: { local_name: string };
  scenario:
    | { Http: { exchanges: Array<{ request: { method: "Get" | "Head" | "Post" | "Put" | "Patch" | "Delete"; path: string; body: number[] | null }; response: { status: number; headers: Array<{ name: string; value: string }>; body: number[] } }> } }
    | { Tcp: { exchanges: Array<{ client_bytes: number[]; server_bytes: number[] }> } };
  endpoint_substitutions: Array<"HttpBaseUrl" | "LoopbackHost" | "LoopbackPort">;
}

export interface BuiltinForeignAdapterFixture {
  id: { local_name: string };
  files: BuiltinFilesystemFixture;
  preparation: BuiltinForeignPreparation;
}

export type BuiltinNativeSourceLanguage = "C" | "Cxx" | "Fortran" | "Cuda";

export type BuiltinForeignPreparation =
  | { Mex: {
      module_name: string;
      api: "R2017b" | "R2018a" | "LargeArrayDims" | "CompatibleArrayDims";
      translation_units: BuiltinNativeTranslationUnit[];
      include_directories: string[];
      definitions: BuiltinPreprocessorDefinition[];
    } }
  | { NativeFfi: {
      isolation: "InProcess" | "OutOfProcess";
      library_name: string;
      translation_units: BuiltinNativeTranslationUnit[];
      include_directories: string[];
      definitions: BuiltinPreprocessorDefinition[];
      interface: {
        interface_name: string;
        primary_header: string;
        additional_headers: string[];
        include_directories: string[];
        definitions: BuiltinPreprocessorDefinition[];
      };
    } }
  | { Java: {
      artifact_name: string;
      release: number;
      source_files: string[];
      resources: Array<{ source_path: string; artifact_path: string }>;
      compile_classpath: string[];
    } }
  | { Python: {
      isolation: "InProcess" | "OutOfProcess";
      environment: { implementation: "Cpython"; major: number; minor: number };
      artifact:
        | { SourceTree: { module_root: string; modules: string[] } }
        | { Wheel: {
            artifact_name: string;
            relative_path: string;
            module: string;
            compatibility: "Pure" | { Native: { abi_tag: string; platform_tag: string } };
          } };
    } };

export interface BuiltinNativeTranslationUnit {
  relative_path: string;
  language: BuiltinNativeSourceLanguage;
}

export interface BuiltinPreprocessorDefinition {
  name: string;
  value: string | null;
}

export interface BuiltinCliInteractionFixture {
  id: { local_name: string };
  transcript: Array<{ ExpectOutput: string } | { SendLine: string } | { SendBytes: number[] } | "SendEndOfInput" | "SendInterrupt">;
}

export interface BuiltinDesktopHostFixture {
  id: { local_name: string };
  scenario: "FilePicker" | "FigureWindow" | "InteractivePrompt";
}

export interface BuiltinExampleRequirements {
  host: "Any" | "NativeOnly" | "DesktopHostOnly";
  engine: "Default" | "Interpreter" | "Jit" | "Aot";
  compiler: Array<"C" | "Cxx" | "Fortran" | "Cuda" | "JavaBytecode" | "PythonExtension">;
  runtime: Array<"NativeDynamicLoader" | "Mex" | "NativeFfi" | "JavaVirtualMachine" | "Python" | "NumPy">;
  toolchain: Array<"CCompiler" | "CxxCompiler" | "FortranCompiler" | "CudaToolkit" | "JavaDevelopmentKit" | "PythonInterpreter" | "PythonDevelopmentHeaders">;
}

export interface FAQ {
    question: string;
    answer: string;
}

export interface Link {
    label: string;
    url: string;
}

export interface JsonEncodeOptions {
    name: string;
    type: string;
    default: string;
    description: string;
}

export interface BuiltinMetadata {
  key?: string;
  module_stem?: string;
  authority?: "catalog" | "legacy_sidecar";
  title: string;
  category: string;
  keywords: string[];
  summary: string;
  gpu_support: GpuSupport;
  fusion: FusionSpec;
  requires_feature: string | null;
  tested: Tested;
  description: string;
  behaviors: string[];
  examples: Example[];
  gpu_residency?: string;
  gpu_behavior?: string[];
  faqs: FAQ[];
  links: Link[];
  source: Link;
  options?: string[];
  syntax?: {
      example: Example;
      points: string[];
  }
  jsonencode_options?: JsonEncodeOptions
}
