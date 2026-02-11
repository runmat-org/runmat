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
  description: string;
  input: string;
  output?: string;
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
  title: string;
  category: string;
  keywords: string[];
  summary: string;
  references: string[];
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