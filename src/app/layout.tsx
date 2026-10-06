import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Mathematics of Computer Vision Language Models",
  description:
    "Interactive educational tool visualizing the mathematical foundations of Vision-Language Models: image tensors, convolution, attention, embeddings, cross-modal attention, autoregressive generation.",
  keywords: [
    "Computer Vision",
    "Language Models",
    "Transformers",
    "Self-Attention",
    "Vision Transformer",
    "CLIP",
    "Mathematics",
    "Undergraduate Thesis",
  ],
  authors: [{ name: "Thesis Tool" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
