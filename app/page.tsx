'use client';

import React, { useState } from 'react';
import { Icon } from '@/components/ui/Icon';

export default function Home() {
    const [prompt, setPrompt] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleGenerate = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!prompt.trim()) return;

        setIsLoading(true);
        console.log("Generating dashboard for:", prompt);
        // TODO: Implement API call
        setTimeout(() => setIsLoading(false), 2000);
    };

    return (
        <div className="relative z-10 max-w-5xl mx-auto px-6 py-12 md:py-20 flex flex-col gap-16 md:gap-24">

            {/* Navigation */}
            <nav className="flex justify-between items-center animate-drift">
                <div className="font-bold text-lg tracking-tight hover:text-white transition-colors duration-300 cursor-pointer animate-breathe uppercase">
                    FS_SYS.01
                </div>
                <div className="flex gap-8 text-xs md:text-sm text-[#8f9196] uppercase tracking-widest font-medium">
                    <a href="#" className="hover:text-[#d9d7c5] transition-colors duration-300 inline-block">[Journal]</a>
                    <a href="#" className="hover:text-[#d9d7c5] transition-colors duration-300 inline-block">[Archive]</a>
                    <a href="#" className="hover:text-[#d9d7c5] transition-colors duration-300 inline-block">[About]</a>
                </div>
            </nav>

            {/* Hero Section */}
            <header className="flex flex-col gap-6 animate-drift delay-100 text-center items-center">
                <div className="mb-4">
                    <Icon icon="lucide:terminal" className="text-[#8f9196] w-10 h-10 md:w-12 md:h-12 animate-breathe" />
                </div>
                <h1 className="md:text-6xl lg:text-7xl leading-none text-3xl font-medium text-[#d9d7c5] tracking-tighter">
                    INK_&_MEMORY
                </h1>
                <p className="max-w-xl mx-auto text-[#8f9196] text-xs md:text-sm leading-loose tracking-wide">
                    &gt; INITIALIZING DIGITAL SPACE...<br />
                    &gt; STATUS: IMPERFECT.<br />
                    Where thoughts settle like dust on parchment. A curated system for analog input.
                </p>

                {/* Wobbly Divider Line (SVG) */}
                <div className="w-full max-w-xs mx-auto py-8 opacity-60">
                    <svg viewBox="0 0 300 10" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto">
                        <path d="M2 5C50 2 60 8 100 5C140 2 160 8 200 5C240 2 260 7 298 5" stroke="#8f9196" strokeWidth="1.5" strokeLinecap="square"></path>
                    </svg>
                </div>

                <button
                    onClick={() => handleGenerate()}
                    disabled={isLoading}
                    className="group relative inline-flex items-center gap-3 px-8 py-3 text-[#d9d7c5] hover:text-white transition-colors duration-300 hover-sketch disabled:opacity-50"
                >
                    {/* Custom Sketch Border for Button */}
                    <div className="absolute inset-0 border border-[#8f9196] group-hover:border-[#d9d7c5] sketch-border bg-transparent transition-colors duration-300"></div>
                    <span className="relative uppercase text-xs tracking-widest font-semibold">
                        {isLoading ? 'Processing...' : 'Execute_Collection'}
                    </span>
                </button>
            </header>

            {/* Gallery / Features Grid */}
            <section className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-6 animate-drift delay-200">

                {/* Card 1 */}
                <div className="group relative p-8 flex flex-col gap-4 hover-sketch cursor-default">
                    <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500"></div>
                    <div className="flex justify-between items-start">
                        <Icon icon="lucide:file-code" className="text-[#d9d7c5] w-6 h-6" />
                        <span className="text-xs text-[#8f9196] font-bold">01_LOG</span>
                    </div>
                    <h3 className="text-base font-bold tracking-tight mt-2 text-[#d9d7c5] uppercase">Daily_Input</h3>
                    <p className="text-xs text-[#8f9196] leading-relaxed tracking-wide">
                    // Ephemeral notes captured in the runtime. Fleeting thoughts preserved in digital ink.
                    </p>
                </div>

                {/* Card 2 */}
                <div className="group relative p-8 flex flex-col gap-4 hover-sketch cursor-default mt-4 md:mt-0">
                    <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500" style={{ borderRadius: '225px 15px 255px 15px / 15px 255px 15px 225px' }}></div>
                    <div className="flex justify-between items-start">
                        <Icon icon="lucide:pen-tool" className="text-[#d9d7c5] w-6 h-6" />
                        <span className="text-xs text-[#8f9196] font-bold">02_DRAFT</span>
                    </div>
                    <h3 className="text-base font-bold tracking-tight mt-2 text-[#d9d7c5] uppercase">Rough_Sketches</h3>
                    <p className="text-xs text-[#8f9196] leading-relaxed tracking-wide">
                    // Imperfect lines and unstructured ideas. A repository for uncompiled drafts.
                    </p>
                </div>

                {/* Card 3 */}
                <div className="group relative p-8 flex flex-col gap-4 hover-sketch cursor-default">
                    <div className="absolute inset-0 border border-[#8f9196] opacity-30 group-hover:opacity-100 sketch-border transition-all duration-500" style={{ borderRadius: '20px 225px 20px 230px / 230px 20px 225px 20px' }}></div>
                    <div className="flex justify-between items-start">
                        <Icon icon="lucide:database" className="text-[#d9d7c5] w-6 h-6" />
                        <span className="text-xs text-[#8f9196] font-bold">03_STORE</span>
                    </div>
                    <h3 className="text-base font-bold tracking-tight mt-2 text-[#d9d7c5] uppercase">Timeless_Archives</h3>
                    <p className="text-xs text-[#8f9196] leading-relaxed tracking-wide">
                    // Curated collections. Analyzing the foundation to predict the next vector.
                    </p>
                </div>

            </section>

            {/* Divider */}
            <div className="w-full opacity-30 animate-drift delay-300">
                <svg viewBox="0 0 800 12" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto" preserveAspectRatio="none">
                    <path d="M2 6C150 2 200 10 300 6C400 2 500 11 600 6C700 1 750 9 798 6" stroke="#8f9196" strokeWidth="1" strokeDasharray="4 4"></path>
                </svg>
            </div>

            {/* Interactive Section (Newsletter/Input) */}
            <section className="flex flex-col md:flex-row items-center justify-between gap-12 animate-drift delay-500">
                <div className="max-w-sm text-center md:text-left">
                    <h2 className="text-xl tracking-tight text-[#d9d7c5] mb-2 font-bold uppercase">Correspondence_</h2>
                    <p className="text-xs text-[#8f9196] tracking-wide">Leave a trace. Write when the buffer is ready.</p>
                </div>

                <div className="w-full max-w-md relative">
                    {/* Hand drawn input box container */}
                    <form onSubmit={handleGenerate} className="relative">
                        <input
                            type="text"
                            placeholder="user@signature.sys"
                            className="w-full bg-transparent border-b border-[#8f9196] text-[#d9d7c5] py-3 px-2 placeholder-[#8f9196]/50 focus:border-[#d9d7c5] transition-colors duration-300 font-ibm text-sm tracking-wide"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                        />
                        <button type="submit" className="absolute right-0 top-2 group">
                            <Icon icon="lucide:arrow-right" className="text-[#8f9196] group-hover:text-white transition-colors duration-300 w-5 h-5" />
                        </button>
                    </form>
                    {/* Decorative underline sketch */}
                    <svg viewBox="0 0 400 10" className="w-full h-3 mt-1 opacity-50 absolute pointer-events-none" preserveAspectRatio="none">
                        <path d="M2 2C100 5 200 1 398 3" stroke="#8f9196" strokeWidth="0.5"></path>
                    </svg>
                </div>
            </section>

            {/* Footer */}
            <footer className="pt-12 md:pt-20 pb-8 flex flex-col items-center gap-6 animate-drift delay-500">
                <div className="w-12 h-12 flex items-center justify-center border border-[#8f9196] sketch-border opacity-50 hover:opacity-100 transition-opacity duration-300 cursor-pointer animate-breathe">
                    <span className="text-xs tracking-tighter font-bold">FS</span>
                </div>
                <div className="text-[#8f9196] text-[10px] md:text-xs text-center tracking-widest uppercase">
                    <p>© 2024 The Forgotten Sketchbook. All rights reserved.</p>
                    <div className="flex justify-center gap-4 mt-2 opacity-60">
                        <a href="#" className="hover:text-[#d9d7c5]">Imprint</a>
                        <span>//</span>
                        <a href="#" className="hover:text-[#d9d7c5]">Privacy</a>
                    </div>
                </div>
            </footer>

        </div>
    );
}
