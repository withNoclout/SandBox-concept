import type { Metadata } from "next";
import { IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const ibmPlexMono = IBM_Plex_Mono({
    weight: ['400', '500', '600', '700'],
    subsets: ["latin"],
    variable: "--font-ibm",
});

export const metadata: Metadata = {
    title: "The Forgotten Sketchbook // Mono",
    description: "A curated system for analog input.",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className={`${ibmPlexMono.variable} antialiased`}>
                <div className="bg-noise"></div>
                {children}
            </body>
        </html>
    );
}
