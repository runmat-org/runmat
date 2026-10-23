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
  output?: string | null;
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
  | { ApplicationSettings: { id: { local_name: string } } }
  | { MailOutbox: { id: { local_name: string }; expected_message: string } }
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
  entries: BuiltinFilesystemFixture["entries"];
  interactions: BuiltinDesktopHostInteraction[];
}

export type BuiltinDialogFilterExpectation = {
  patterns: string[];
  description: string | null;
};

export type BuiltinDesktopHostInteraction =
  | { Document:
      | { OpenFile: { relative_path: string } }
      | { OpenVariable: { name: string } }
      | { Navigate: { relative_path: string; line: number; column: number; selection: "Caret" | "Select" } }
    }
  | { OpenFileDialog: {
      request: { title: string | null; default_path: string | null; filters: BuiltinDialogFilterExpectation[]; multiselect: boolean };
      outcome: { Selection: { paths: string[]; filter_index: number | null } } | "Cancel" | { Error: string };
    } }
  | { SaveFileDialog: {
      request: { title: string | null; default_path: string | null; filters: BuiltinDialogFilterExpectation[] };
      outcome: { Selection: { path: string; filter_index: number | null } } | "Cancel" | { Error: string };
    } }
  | { DirectoryDialog: {
      request: { title: string | null; default_path: string | null };
      outcome: { Selection: { path: string } } | "Cancel" | { Error: string };
    } }
  | { LineInput: {
      prompt: string;
      echo: boolean;
      outcome: { Line: string } | { Error: string };
    } }
  | { KeyInput: {
      prompt: string;
      outcome: "KeyPress" | { Error: string };
    } }
  | { FigurePresentation: {
      figure_ordinal: number;
      event: "Created" | "Updated" | "Cleared" | "Closed";
      snapshot: { title: string | null; axes_rows: number; axes_cols: number } | null;
    } };

export interface BuiltinExampleRequirements {
  host: "Any" | "NativeOnly" | "DesktopHostOnly";
  engine: "Default" | "Interpreter" | "Jit" | "Aot";
  compiler: Array<"C" | "Cxx" | "Fortran" | "Cuda" | "JavaBytecode" | "PythonExtension">;
  runtime: Array<"NativeDynamicLoader" | "Mex" | "NativeFfi" | "JavaVirtualMachine" | "Python" | "NumPy" | "PlotCore" | "NativeCad" | "NativeMeshing">;
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

/** JSON presentation emitted by documentation_export/projection.rs, not
 * another semantic catalog. Legacy sidecar fields remain a separate variant. */
export type BuiltinMetadata = CanonicalBuiltinMetadata | LegacyBuiltinMetadata;

export interface BuiltinDocumentationSection {
  heading: string;
  paragraphs: string[];
}

export interface BuiltinDocumentationMedia {
  kind: "Image" | "Video";
  url: string;
  alt: string;
}

/** Evidence implementation links retain Rust's externally tagged target.
 * Public links/source/related are separately projected to { label, url }. */
export interface BuiltinDocumentationEvidence {
  implementation: Array<{
    label: string;
    target: { Builtin: string } | { Documentation: string } | { Source: string } | { External: string };
  }>;
  verification: Array<{
    kind: "UnitTest" | "IntegrationTest" | "BrowserTest" | "ProviderTest" | "WgpuTest" | "ConformanceTest" | "Validation";
    label: string;
    location: string;
  }>;
  notes: string[];
}

/** Catalog semantics are opaque to example consumers. Preserve the exact JSON
 * object rather than re-declaring descriptor/contract/placement authorities. */
export type BuiltinCatalogJson = null | boolean | number | string | BuiltinCatalogJson[] | { [key: string]: BuiltinCatalogJson };

/** Canonical rows always carry the explicit typed execution declarations. */
export interface CanonicalBuiltinExample extends Example {
  id: string;
  output: string | null;
  compatibility: NonNullable<Example["compatibility"]>;
  harness: NonNullable<Example["harness"]>;
  fixture: BuiltinExampleFixture;
  requirements: BuiltinExampleRequirements;
  verification: NonNullable<Example["verification"]>;
}

export interface CanonicalBuiltinMetadata {
  key: string;
  module_stem: string;
  authority: "catalog";
  title: string;
  slug: string;
  category: string;
  keywords: string[];
  summary: string;
  description: string;
  sections: BuiltinDocumentationSection[];
  examples: CanonicalBuiltinExample[];
  example_exemption: string | null;
  faqs: FAQ[];
  links: Link[];
  related: Link[];
  media: BuiltinDocumentationMedia[];
  evidence: BuiltinDocumentationEvidence;
  source?: Link;
  introduced: string | null;
  status: "Stable" | "Experimental" | "Partial" | "Deprecated" | null;
  catalog: { [key: string]: BuiltinCatalogJson };
}

/** Historical sidecar projection remains supported during explicit transition.
 * Do not require these legacy-only fields on canonical catalog documents. */
export interface LegacyBuiltinMetadata {
  key?: string;
  module_stem?: string;
  authority?: "legacy_sidecar";
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
