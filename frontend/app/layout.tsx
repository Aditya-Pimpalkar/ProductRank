import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "ProductRank — retrieval & ranking, measured",
  description: "Multi-stage hybrid retrieval with an evaluation engine over BEIR/FiQA.",
};

const NAV = [
  { href: "/", label: "Search Comparison" },
  { href: "/experiments", label: "Experiments" },
  { href: "/analytics", label: "Analytics" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-slate-200 bg-white">
          <div className="mx-auto flex max-w-7xl items-center gap-6 px-6 py-3">
            <Link href="/" className="text-lg font-bold tracking-tight">
              Product<span className="text-indigo-600">Rank</span>
            </Link>
            <nav className="flex gap-1 text-sm">
              {NAV.map((n) => (
                <Link
                  key={n.href}
                  href={n.href}
                  className="rounded-md px-3 py-1.5 text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                >
                  {n.label}
                </Link>
              ))}
            </nav>
            <span className="ml-auto text-xs text-slate-400">BEIR / MS MARCO · 51,070 passages</span>
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
