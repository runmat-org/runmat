import { Metadata } from "next";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { Calendar, Clock, User } from "lucide-react";
import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import matter from 'gray-matter';

export const metadata: Metadata = {
  title: "RunMat Blog - Stories and Insights",
  description: "Stories, insights, and updates from the RunMat development team. Learn about MATLAB compatibility, performance optimization, and the future of scientific computing.",
  openGraph: {
    title: "RunMat Blog - Stories and Insights",
    description: "Stories, insights, and updates from the RunMat development team.",
    type: "website",
  },
};

interface BlogPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  readTime: string;
  author: string;
  tags: string[];
}

function getAllBlogPosts(): BlogPost[] {
  try {
    const blogDir = join(process.cwd(), 'content/blog');
    const files = readdirSync(blogDir).filter(file => file.endsWith('.md'));
    
    const posts = files.map(file => {
      const slug = file.replace(/\.md$/, '');
      const filePath = join(blogDir, file);
      const fileContent = readFileSync(filePath, 'utf-8');
      const { data: frontmatter } = matter(fileContent);
      
      return {
        slug,
        title: frontmatter.title || 'Untitled',
        description: frontmatter.description || frontmatter.excerpt || '',
        date: frontmatter.date || new Date().toISOString(),
        readTime: frontmatter.readTime || '5 min read',
        author: frontmatter.author || 'RunMat Team',
        tags: frontmatter.tags || []
      };
    });
    
    // Sort posts by date (newest first)
    return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  } catch (error) {
    console.error('Error reading blog posts:', error);
    return [];
  }
}

export default function BlogPage() {
  const blogPosts = getAllBlogPosts();

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16 md:px-6 md:py-24">
        <div className="mx-auto max-w-[58rem] text-center">
          <Badge variant="secondary" className="mb-4">
            Blog
          </Badge>
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            RunMat Blog
          </h1>
          <p className="mt-6 text-xl text-muted-foreground">
            Stories, insights, and updates from the RunMat development team
          </p>
        </div>

        <div className="mt-16 space-y-8">
          {blogPosts.map((post) => (
            <Link key={post.slug} href={`/blog/${post.slug}`} className="block">
              <Card className="group overflow-hidden transition-colors hover:bg-muted/50 cursor-pointer">
                <CardHeader>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {post.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                  <CardTitle className="text-2xl leading-tight sm:text-3xl">
                    {post.title}
                  </CardTitle>
                  <CardDescription className="text-lg leading-relaxed">
                    {post.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
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
                      {post.author}
                    </div>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Read Article →
                  </div>
                </CardContent>
              </Card>
            </Link>
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