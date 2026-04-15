import { Metadata } from "next";
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
          <h1 className="text-3xl sm:text-3xl md:text-5xl font-bold tracking-tight">
            RunMat Blog
          </h1>
          <p className="mt-6 text-foreground text-[0.938rem] break-words">
            Stories, insights, and updates from the RunMat development team
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {blogPosts.map((post, i) => (
            <ContentCard
              key={post.slug}
              href={`/blog/${post.slug}`}
              title={post.title}
              image={post.image}
              imageAlt={post.imageAlt}
              excerpt={post.description}
              date={post.dateModified || post.date}
              ctaLabel="Read"
              index={i}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
