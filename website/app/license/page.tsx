import { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, AlertCircle, HelpCircle, Mail } from "lucide-react";
import Link from "next/link";

export const metadata: Metadata = {
  title: "RunMat License",
  description: "RunMat is licensed under the MIT License with Attribution Requirements and Commercial Scientific Software Company Copyleft Provisions. Free for everyone with clear terms.",
};

export default function LicensePage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-4xl px-4 md:px-6 py-16 md:py-24">
        
        {/* Header */}
        <div className="mb-12">
          <Badge variant="secondary" className="mb-4">Legal</Badge>
          <h1 className="text-4xl lg:text-5xl font-bold mb-6 text-foreground">
            RunMat License
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed">
            RunMat is free and open source software with clear, fair licensing terms. 
            This page explains what you can and cannot do with RunMat.
          </p>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <Card className="border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-950/50">
            <CardHeader className="text-center pb-3">
              <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-600" />
              <CardTitle className="text-lg text-green-800 dark:text-green-200">Free for Most Uses</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-sm text-green-800 dark:text-green-100 font-medium">
                Individuals, researchers, students, and most companies can use RunMat freely
              </p>
            </CardContent>
          </Card>

          <Card className="border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-950/50">
            <CardHeader className="text-center pb-3">
              <AlertCircle className="h-8 w-8 mx-auto mb-2 text-orange-600" />
              <CardTitle className="text-lg text-orange-800 dark:text-orange-200">Attribution Required</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-sm text-orange-800 dark:text-orange-100 font-medium">
                Must credit &ldquo;RunMat by Dystr&rdquo; in distributions and derivative works
              </p>
            </CardContent>
          </Card>

          <Card className="border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/50">
            <CardHeader className="text-center pb-3">
              <HelpCircle className="h-8 w-8 mx-auto mb-2 text-blue-600" />
              <CardTitle className="text-lg text-blue-800 dark:text-blue-200">Special Rules</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-sm text-blue-800 dark:text-blue-100 font-medium">
                Commercial scientific software companies must keep modifications open source
              </p>
            </CardContent>
          </Card>
        </div>

        {/* FAQ Section */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-8 text-foreground">
            Frequently Asked Questions
          </h2>
          
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-xl text-foreground">
                  Can I use RunMat for free?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground mb-3">
                  <strong>Yes!</strong> RunMat is completely free for:
                </p>
                <ul className="list-disc list-inside space-y-1 text-foreground/80 ml-4">
                  <li>Individual researchers and scientists</li>
                  <li>Academic institutions and educational organizations</li>
                  <li>Students for learning and coursework</li>
                  <li>Most commercial companies and startups</li>
                  <li>Open source projects</li>
                  <li>Government agencies and non-profits</li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-xl text-foreground">
                  What does &quot;attribution required&quot; mean?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground mb-3">
                  Any distribution or modification of RunMat must credit <strong>&quot;RunMat by Dystr&quot;</strong>. This includes:
                </p>
                <ul className="list-disc list-inside space-y-1 text-foreground/80 ml-4">
                  <li>Startup messages or about dialogs</li>
                  <li>Documentation and README files</li>
                  <li>Package names or project titles</li>
                  <li>Web interfaces showing &quot;Powered by RunMat by Dystr&quot;</li>
                </ul>
                <p className="text-foreground/90 mt-3">
                  This ensures Dystr receives appropriate credit for creating RunMat.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-xl text-foreground">
                  What are the special rules for scientific software companies?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground mb-3">
                  Companies whose <strong>primary business</strong> involves developing, licensing, or selling scientific computing software 
                  (like MathWorks, Ansys, COMSOL, etc.) must distribute any RunMat modifications as open source under the same license.
                </p>
                <p className="text-foreground mb-3">
                  <strong>This does NOT apply to:</strong>
                </p>
                <ul className="list-disc list-inside space-y-1 text-foreground/80 ml-4">
                  <li>Companies using RunMat without modification</li>
                  <li>Companies whose primary business is not scientific computing software</li>
                  <li>Internal modifications not distributed to third parties</li>
                  <li>Academic institutions or research organizations</li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-xl text-foreground">
                  Can I create proprietary software that uses RunMat?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground mb-3">
                  <strong>Yes!</strong> Most users can create proprietary software that uses or embeds RunMat. 
                  The license only requires that if you distribute or modify RunMat itself, you must:
                </p>
                <ul className="list-disc list-inside space-y-1 text-foreground/80 ml-4">
                  <li>You provide attribution to &quot;RunMat by Dystr&quot;</li>
                  <li>You include the license notice</li>
                  <li>Any modifications to RunMat itself are shared (if you&apos;re a scientific software company)</li>
                </ul>
                <p className="text-foreground/90 mt-3">
                  Your own code that calls RunMat functions remains proprietary.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-xl text-foreground">
                  How does this compare to other open source licenses?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground mb-3">
                  RunMat&apos;s license is based on the MIT License with two additional requirements:
                </p>
                <ul className="list-disc list-inside space-y-1 text-foreground/80 ml-4">
                  <li><strong>Attribution:</strong> Similar to BSD licenses, ensures credit is maintained</li>
                  <li><strong>Targeted copyleft:</strong> Only applies to large scientific software companies, ensuring community contributions</li>
                </ul>
                <p className="text-foreground/90 mt-3">
                  For most users, it&apos;s as permissive as MIT. For the few companies it affects, 
                  it encourages open source contribution rather than proprietary appropriation.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-xl text-foreground">
                  I&apos;m still not sure if my use case is allowed. What should I do?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-foreground/90 mb-4">
                  When in doubt, reach out! We&apos;re happy to clarify licensing questions and work with you 
                  to ensure your use case is properly covered.
                </p>
                <div className="flex items-center space-x-2 text-blue-600 dark:text-blue-400">
                  <Mail className="h-4 w-4" />
                  <Link href="mailto:legal@dystr.com" className="hover:underline">
                    legal@dystr.com
                  </Link>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Full License Text */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-8 text-foreground">
            Full License Text         
          </h2>
          
          <Card className="p-0">
            <CardContent className="p-6">
              <div className="font-mono text-sm text-foreground/90 whitespace-pre-line leading-relaxed">
{`# License

RunMat is licensed under the **MIT License** with **Attribution Requirements** and **Commercial Scientific Software Company Copyleft** Provisions.

## MIT License with Additional Terms

Copyright (c) 2025 Dystr Inc. and RunMat contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

### Attribution Requirement

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. Any distribution, fork, modification, or derivative work of this Software must maintain attribution to "RunMat by Dystr" in a prominent location visible to end users. This includes but is not limited to:

1. **Startup messages** or **about dialogs** displaying "RunMat by Dystr"
2. **Documentation** and **README files** crediting "RunMat by Dystr"  
3. **Package names** or **project titles** indicating the RunMat origin
4. **Web interfaces** showing "Powered by RunMat by Dystr"

### Commercial Scientific Software Company Copyleft

Any entity whose **primary business purpose** involves the development, licensing, or sale of scientific computing software, engineering simulation software, mathematical computing environments, or technical computing platforms (including but not limited to companies like MathWorks, Ansys, COMSOL Multiphysics, Dassault Systèmes, Autodesk, PTC, Siemens Digital Industries Software, Altair, and similar organizations) that creates a fork, derivative work, or modification of this Software **MUST** distribute their modified version under the same license terms as this license, making their modifications available as open source.

This copyleft provision does **NOT** apply to:
- Individual researchers, academic institutions, or educational organizations
- Companies using RunMat as a dependency without modification
- Companies whose primary business is not scientific/technical computing software
- Internal modifications not distributed to third parties
- Other open source projects and their maintainers

### Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.`}
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Contact Section */}
        <section>
          <Card className="bg-blue-50 dark:bg-blue-950/50 border-blue-200 dark:border-blue-800">
            <CardContent className="p-6 text-center">
              <h3 className="text-lg font-semibold mb-3 text-muted-foreground">
                Need Legal Clarification?
              </h3>
              <p className="text-muted-foreground mb-4">
                Our legal team is happy to help clarify licensing questions or discuss commercial licensing options.
              </p>
              <div className="flex items-center justify-center space-x-2">
                <Mail className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                <Link 
                  href="mailto:legal@dystr.com" 
                  className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                >
                  legal@dystr.com
                </Link>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}