import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowLeft, Calendar, Clock, User } from "lucide-react";

interface BlogLayoutProps {
  children: React.ReactNode;
  title: string;
  description: string;
  date: string;
  readTime: string;
  author: string;
  tags: string[];
  rightAside?: React.ReactNode;
}

export function BlogLayout({
  children,
  title,
  description,
  date,
  readTime,
  author,
  tags,
  rightAside
}: BlogLayoutProps) {
  return (
    <div className="min-h-screen bg-background">
      <div className="container px-4 py-8 md:px-6 md:py-16">
        {rightAside ? (
          <div className="mx-auto lg:max-w-7xl grid gap-8 lg:grid-cols-[minmax(0,1fr)_260px]">
            <div>
              {/* Back to Blog */}
              <Button variant="ghost" size="sm" className="mb-8" asChild>
                <Link href="/blog">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Blog
                </Link>
              </Button>

              {/* Article Header */}
              <header className="mb-12 max-w-4xl">
                <div className="mb-4 flex flex-wrap gap-2">
                  {tags.map((tag) => (
                    <Badge key={tag} variant="secondary">
                      {tag}
                    </Badge>
                  ))}
                </div>
                
                <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
                  {title}
                </h1>
                
                <p className="mb-6 text-xl text-muted-foreground leading-relaxed">
                  {description}
                </p>
                
                <div className="flex flex-wrap items-center gap-6 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    {date}
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    {readTime}
                  </div>
                  <div className="flex items-center gap-2">
                    <User className="h-4 w-4" />
                    By {author}
                  </div>
                </div>
              </header>

              {/* Article Content */}
              <article className="max-w-4xl">
                <div className="blog-content w-full">
                  {children}
                </div>
              </article>
            </div>
            {rightAside}
          </div>
        ) : (
          <div className="mx-auto max-w-4xl">
            {/* Back to Blog */}
            <Button variant="ghost" size="sm" className="mb-8" asChild>
              <Link href="/blog">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Blog
              </Link>
            </Button>

            {/* Article Header */}
            <header className="mb-12">
              <div className="mb-4 flex flex-wrap gap-2">
                {tags.map((tag) => (
                  <Badge key={tag} variant="secondary">
                    {tag}
                  </Badge>
                ))}
              </div>
              
              <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
                {title}
              </h1>
              
              <p className="mb-6 text-xl text-muted-foreground leading-relaxed">
                {description}
              </p>
              
              <div className="flex flex-wrap items-center gap-6 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  {date}
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  {readTime}
                </div>
                <div className="flex items-center gap-2">
                  <User className="h-4 w-4" />
                  By {author}
                </div>
              </div>
            </header>

            {/* Article Content */}
            <article className="max-w-none">
              <div className="blog-content w-full">
                {children}
              </div>
            </article>
          </div>
        )}
      </div>
    </div>
  );
}