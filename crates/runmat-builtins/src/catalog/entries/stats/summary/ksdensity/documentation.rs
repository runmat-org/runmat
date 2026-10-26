use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation {
    authority:BuiltinDocumentationAuthority::Catalog,title:Some("ksdensity"),slug:Some("ksdensity"),summary:"Normal-kernel estimates for univariate observations.",
    description:"Estimate a probability density, cumulative distribution or survivor function and optionally draw the resulting curve.",
    keywords:&["ksdensity","kernel density","KDE","bandwidth","normal kernel"],related:&["histogram","normpdf","normcdf","median"],
    sections:&[
        BuiltinDocumentationSection{heading:"Samples, points and outputs",paragraphs:&[
            "[f,xi,bw]=ksdensity(x) accepts real single/double univariate vectors. NaN samples are omitted; at least one observation is required and infinite samples are invalid. Bivariate data, integer/logical/complex samples, sparse arrays and distributed execution are unsupported, not silently converted.",
            "ksdensity(x,pts,...) evaluates at an explicit floating vector. xi retains pts' original class, values and shape, including a single-precision scalar point. Numeric [] requests the automatic grid. Estimates f and the reusable positive bandwidth bw are double, even for single samples; no MATLAB single-output-class parity is claimed. NaN evaluation points propagate NaN; infinite evaluation points use the normal-kernel limiting values.",
            "The RunMat automatic grid has100 double row-vector points from min(x)-3*bw through max(x)+3*bw. NumPoints changes that count; one point uses the upper endpoint, following linspace endpoint semantics. Collapsed coordinates at floating precision are not expanded with an arbitrary extra scale. If padded endpoints are not representable, supply explicit points. This grid policy is stated explicitly rather than claimed to reproduce undocumented MATLAB grid details.",
        ]},
        BuiltinDocumentationSection{heading:"Bandwidth and supported options",paragraphs:&[
            "Bandwidth accepts a finite positive single/double scalar or normal-approx. The normal-reference rule uses h=(MAD/Phi^-1(0.75))*(4/(3*n))^(1/5), with the median absolute deviation about the sample median. If ties give zero MAD but samples are nonconstant, their actual sample standard deviation supplies the scale. If no positive finite automatic scale is identifiable, including constant or singleton samples, provide Bandwidth explicitly; no arbitrary unit width is substituted. Explicit positive Bandwidth supports those samples normally.",
            "Case-insensitive options are Bandwidth, NumPoints, Function, Kernel and Support. Function is pdf (default), cdf or survivor. Kernel must be normal and Support must be unbounded; these select the implemented kernel/domain, not aliases for other kernels or boundary corrections. NumPoints is a positive single/double scalar integer and cannot be combined with nonempty explicit points. Later duplicate controls replace earlier ones.",
            "Plug-in bandwidths, custom/non-normal kernels, censoring, weights, bounded/positive support, BoundaryCorrection, inverse CDF and cumulative hazard are unsupported and rejected. None is accepted and ignored. PDF estimates are not clipped to one; densities above one and representationally infinite peaks can be valid. Log-domain normal-density reduction preserves recoverable small estimates rather than dividing an already-underflowed kernel value by bandwidth.",
        ]},
        BuiltinDocumentationSection{heading:"Plotting and execution",paragraphs:&[
            "With no outputs, ksdensity draws its curve through RunMat's ordinary plot owner and requires plot-core. A typed leading axes target is supported only in that zero-output form. Numeric handle IDs never reinterpret a numeric scalar sample based on ambient UI state. Requested numerical outputs do not create plots or require UI capability.",
            "Only authenticated automatic residency can use the original context's host gather boundary for samples, points and controls. Explicit or unproven device intent is rejected before any argument transfer; use gather explicitly for host estimates. There is no provider-native KDE kernel, numerical fusion producer or separate explicit-host-fallback extension.",
            "Finite host-memory execution is unsupported and refuses before gather, sorting or estimate allocation because the workspaces have no complete allocation certificate. Sample scans, the shared resumable sort and kernel reduction cooperate with the original execution context. Plotting uses the existing exact lazy continuation owner; no new thread or stack-size policy is introduced. Original provider and typed cancellation failures are not replayed or relabeled.",
        ]},
    ],examples:&[
        BuiltinExample{id:"fixed-bandwidth",title:"Evaluate a two-point Gaussian mixture",program:"[f,xi,bw] = ksdensity([-1;1],[0;1],'Bandwidth',1);",display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(isequal(xi,[0;1])); assert(bw==1); assert(abs(f(1)-exp(-0.5)/sqrt(2*pi))<1e-12); assert(abs(f(2)-(1+exp(-2))/(2*sqrt(2*pi)))<1e-12);"}},
        BuiltinExample{id:"normal-reference",title:"Inspect the normal-reference estimate",program:"[f,xi,bw] = ksdensity([-2;-1;0;1;2]);",display_output:None,compatibility:BuiltinExampleCompatibility::RunMat,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(numel(f)==100 && numel(xi)==100); assert(abs(bw-(4/15)^(1/5)/norminv(0.75))<1e-12); assert(all(f>=0)); assert(abs(xi(1)+2+3*bw)<1e-10); assert(abs(xi(end)-2-3*bw)<1e-10);"}},
    ],example_exemption:None,faqs:&[],links:&[
        BuiltinDocumentationLink{label:"MATLAB ksdensity",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/stats/ksdensity.html")},
        BuiltinDocumentationLink{label:"Implementation",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/ksdensity.rs")},
    ],media:&[],evidence:BuiltinDocumentationEvidence{implementation:&[],verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Analytic normal mixtures, scale and public boundaries",location:"builtins::stats::summary::ksdensity::tests"}],notes:&[]},
    introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
