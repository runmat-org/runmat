import { Metadata } from "next";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { Calendar, Clock, User } from "lucide-react";
import { getPublicBlogPosts } from '@/lib/blog';

export const metadata: Metadata = {
  title: "RunMat Blog - Stories and Insights",
  description: "Stories, insights, and updates from the RunMat development team. Learn about MATLAB compatibility, performance optimization, and the future of scientific computing.",
  openGraph: {
    title: "RunMat Blog - Stories and Insights",
    description: "Stories, insights, and updates from the RunMat development team.",
    type: "website",
    url: "https://runmat.org/blog",
  },
  alternates: { canonical: "https://runmat.org/blog" },
};

export default function BlogPage() {
  const blogPosts = getPublicBlogPosts();

  const renderAuthors = (authors?: { name: string; url?: string }[], fallback?: string) => {
    const list =
      authors && authors.length > 0
        ? authors
        : fallback
          ? [{ name: fallback }]
          : [{ name: 'RunMat Team' }];

    return (
      <div className="flex flex-wrap items-center gap-1">
        {list.map((author, index) => (
          <span key={author.name + index} className="flex items-center gap-1">
            {author.url ? (
              <Link
                href={author.url}
                target="_blank"
                rel="noreferrer"
                className="underline underline-offset-2 hover:text-foreground"
              >
                {author.name}
              </Link>
            ) : (
              <span>{author.name}</span>
            )}
            {index < list.length - 1 && <span>,</span>}
          </span>
        ))}
      </div>
    );
  };

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

        <div className="mt-16 space-y-8">
          {blogPosts.map((post) => (
            <Card key={post.slug} className="group overflow-hidden transition-colors hover:bg-muted/50">
              <div className="flex flex-col md:flex-row">
                <div className="flex-1 p-6">
                  <CardHeader className="p-0">
                    <CardTitle className="text-2xl leading-tight sm:text-3xl break-words">
                      <Link
                        href={`/blog/${post.slug}`}
                        className="hover:underline underline-offset-4"
                      >
                        {post.title}
                      </Link>
                    </CardTitle>
                    <CardDescription className="text-base sm:text-lg leading-relaxed break-words">
                      {post.description}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-0 pt-6">
                    <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground mb-4">
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4" />
                        {new Date(post.date).toLocaleDateString('en-US', { 
                          year: 'numeric', 
                          month: 'long', 
                          day: 'numeric' 
                        })}
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4" />
                        {post.readTime}
                      </div>
                      <div className="flex items-center gap-2">
                        <User className="h-4 w-4" />
                        {renderAuthors(post.authors, post.author)}
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      <Link
                        href={`/blog/${post.slug}`}
                        className="hover:underline underline-offset-4"
                      >
                        Read Article →
                      </Link>
                    </div>
                  </CardContent>
                </div>
                {post.image && (
                  <div className="relative w-full md:w-64 lg:w-80 h-48 md:h-48 lg:h-56 flex-shrink-0 overflow-hidden">
                    <Image
                      src={post.image}
                      alt={post.imageAlt || post.title}
                      fill
                      className="object-contain transition-transform duration-300 group-hover:scale-105"
                      sizes="(max-width: 768px) 100vw, (max-width: 1024px) 256px, 320px"
                    />
                  </div>
                )}
              </div>
            </Card>
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
