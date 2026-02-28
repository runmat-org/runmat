import { Metadata } from "next";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ContentCard } from "@/components/content-card";
import { getPublicBlogPosts } from "@/lib/blog";

export const metadata: Metadata = {
  title: "RunMat Blog - Stories and Insights",
  description: "Stories, insights, and updates from the RunMat development team. Learn about MATLAB compatibility, performance optimization, and the future of scientific computing.",
  openGraph: {
    title: "RunMat Blog - Stories and Insights",
    description: "Stories, insights, and updates from the RunMat development team.",
    type: "website",
    url: "https://runmat.com/blog",
  },
  alternates: { canonical: "https://runmat.com/blog" },
};

export default function BlogPage() {
  const blogPosts = getPublicBlogPosts();

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16 md:px-6 md:py-24">
        <div className="mx-auto max-w-[58rem] text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            RunMat Blog
          </h1>
          <p className="mt-6 text-base text-muted-foreground sm:text-lg break-words">
            Stories, insights, and updates from the RunMat development team
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {blogPosts.map((post) => (
            <ContentCard
              key={post.slug}
              href={`/blog/${post.slug}`}
              title={post.title}
              image={post.image}
              imageAlt={post.imageAlt}
              excerpt={post.description}
              date={post.date}
              ctaLabel="Read"
            />
          ))}
        </div>

        {/* Call to Action */}
        <div className="mt-16 text-center">
          <Card className="inline-block">
            <CardContent className="p-6">
              <h3 className="text-lg font-semibold mb-2">
                Stay Updated
              </h3>
              <p className="text-muted-foreground mb-4">
                Follow our development progress and get notified of new releases
              </p>
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button variant="outline" size="sm" asChild>
                  <Link href="https://github.com/runmat-org/runmat" target="_blank">
                    Star on GitHub
                  </Link>
                </Button>
                <Button variant="outline" size="sm" asChild>
                  <Link href="/download">
                    Try RunMat
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
