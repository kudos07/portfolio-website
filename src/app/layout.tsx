import type { Metadata } from "next";
import { Sora } from "next/font/google";
import "./globals.css";
import Background from "./components/Background";

const sora = Sora({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "Saransh Surana | Portfolio",
  description: "Portfolio of Saransh Surana - Data Science, Machine Learning, AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`scroll-smooth ${sora.variable}`}>
      <head>
        <script
          defer
          src="https://cloud.umami.is/script.js"
          data-website-id="707eb819-269b-4d44-b441-cf7008915528"
        ></script>
      </head>
      <body className="m-0 antialiased text-white">
        <Background />
        {children}
      </body>
    </html>
  );
}
