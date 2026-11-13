use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation {
    authority:BuiltinDocumentationAuthority::Catalog,title:Some("ttest"),slug:Some("ttest"),
    summary:"One-sample and paired-sample Student-t tests.",
    description:"Test a population mean with unknown variance, or test the mean of paired differences.",
    keywords:&["ttest","Student t","paired","mean","confidence interval"],related:&["ttest2","tcdf","tinv"],
    sections:&[
        BuiltinDocumentationSection{heading:"Samples and hypotheses",paragraphs:&[
            "ttest(x) tests mean zero. ttest(x,m) tests a finite scalar population mean m. A nonscalar second sample selects the paired test and must have exactly the same size as x; paired observations are subtracted before estimating variance. Samples and numeric controls accept real single or double values. Native integer, logical, complex, object, and distributed sample inputs are unsupported rather than implicitly coerced.",
            "NaN observations are omitted; a paired observation is omitted if either member is NaN. Each lane uses its observed count n, sample standard deviation sd, and n-1 degrees of freedom. Fewer than two observations produce NaN probabilities and a false decision. Computed confidence endpoints are NaN in that case; the untested endpoint of a one-sided interval remains infinite. Zero standard error gives the formula's signed infinite or undefined t statistic, and finite confidence bounds can collapse to the observed mean. Infinite observations can produce undefined statistics.",
            "The default tests the first nonsingleton dimension. Dim selects a positive, platform-representable 1-based integer dimension; dimensions beyond the input rank are singleton dimensions. Dimension-derived metadata must have representable geometry and a successful allocation. Result arrays have length one in that dimension; confidence limits have length two there. No broadcasting is performed for nonscalar paired samples.",
        ]},
        BuiltinDocumentationSection{heading:"Options and results",paragraphs:&[
            "Name/value options are case-insensitive Alpha, Dim, and Tail. Alpha defaults to 0.05 and must lie strictly between zero and one. Tail is both (default), right, or left. Unknown options, incomplete pairs, and unsupported values raise an error; later duplicate options replace earlier ones.",
            "[h,p,ci,stats] returns a logical decision h (p < Alpha), the selected Student-t tail probability, the confidence interval for the original population mean or paired mean difference, and a scalar structure with tstat, df, and sd arrays. The one-sided interval has an infinite bound on its untested side. Numeric outputs use host binary64, including single-input tests; this is not a claim of MATLAB single-output-class parity.",
        ]},
        BuiltinDocumentationSection{heading:"Execution and resource limits",paragraphs:&[
            "Computation runs on the host. Only authenticated automatic residency may use the context-owned host gather fallback, for samples and numeric controls alike. Explicit gpuArray or unproven device intent is rejected before any argument transfer; call gather explicitly to request host analysis. There is no provider-native test kernel, fusion producer, distributed implementation, or separately enabled explicit-host-fallback extension. Gather errors and typed cancellation propagate without retry or relabeling.",
            "Input scans and lane statistics cooperate with the owning execution context. Finite host-memory requests are rejected before gather and workspace allocation because this implementation has no complete lane/result allocation certificate; it does not fabricate a finite-memory grant.",
        ]},
    ],
    examples:&[
        BuiltinExample{id:"one-sample",title:"Test a population mean",program:"[h,p,ci,s] = ttest([1;2],0);",display_output:None,
            compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(~h); assert(abs(p-2*atan(1/3)/pi)<1e-10); assert(s.df==1); assert(abs(s.tstat-3)<1e-10); assert(ci(1)<1.5 && ci(2)>1.5);"}},
        BuiltinExample{id:"paired",title:"Test paired differences",program:"x = [2;4;6]; y = [1;2;3]; [h,p,ci,s] = ttest(x,y,'Tail','right');",display_output:None,
            compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(h); assert(p<0.05); assert(s.df==2); assert(abs(s.sd-1)<1e-10); assert(isinf(ci(2)));"}},
    ],example_exemption:None,faqs:&[],
    links:&[
        BuiltinDocumentationLink{label:"MATLAB ttest",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/stats/ttest.html")},
        BuiltinDocumentationLink{label:"Implementation",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/ttest.rs")},
    ],media:&[],evidence:BuiltinDocumentationEvidence{implementation:&[],verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Analytic probabilities, pairing, dimensions and error boundaries",location:"builtins::stats::summary::ttest::tests"}],notes:&[]},
    introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
