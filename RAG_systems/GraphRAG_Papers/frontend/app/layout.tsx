import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Graph-RAG arXiv",
  description: "Graph-first retrieval for arXiv research papers",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-white">
          {children}
        </div>
      </body>
    </html>
  );
}
