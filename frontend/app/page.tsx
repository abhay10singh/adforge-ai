"use client";

import React, { useState, useRef } from "react";
import { UploadCloud, Image as ImageIcon, Sparkles, Ratio, Loader2, X, Wand2, Download, RefreshCw, RectangleHorizontal, Square } from "lucide-react";

// Aspect Ratios with better icons
const ASPECT_RATIOS = [
  { label: "Landscape", value: "1536x1024", icon: <RectangleHorizontal className="w-4 h-4" /> },
  { label: "Square", value: "1024x1024", icon: <Square className="w-4 h-4" /> },
];

export default function AdForgeHome() {
  // State Management
  const [prompt, setPrompt] = useState("");
  const [selectedRatio, setSelectedRatio] = useState(ASPECT_RATIOS[0].value);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handlers
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedImage(file);
      // Create a temporary URL for preview
      setImagePreviewUrl(URL.createObjectURL(file));
      // Reset generated image if a new upload happens
      setGeneratedImageUrl(null); 
    }
  };

  const clearUpload = () => {
      setUploadedImage(null);
      setImagePreviewUrl(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
  }

// ...existing code...

const handleGenerate = async () => {
  if (!prompt) return alert("Please enter a prompt");
  if (!uploadedImage) return alert("Please upload an image");
  
  setIsLoading(true);
  setGeneratedImageUrl(null);

  const formData = new FormData();
  formData.append("image", uploadedImage);
  formData.append("prompt", prompt);

  try {
    const response = await fetch("https://abhi10s-adforge.hf.space/generate", {
      method: "POST",
      headers: {
        "x-api-key": process.env.NEXT_PUBLIC_API_KEY || "",
      },
      body: formData,
    });

    const data = await response.json();
    
    if (data.status === "success") {
      setGeneratedImageUrl(data.image_url);
    } else {
      alert("Generation failed: " + data.detail);
    }
  } catch (error) {
    console.error("Error:", error);
    alert("Failed to connect to server");
  } finally {
    setIsLoading(false);
  }
};

// ...existing code...

  return (
    <main className="min-h-screen bg-white selection:bg-blue-100 overflow-hidden relative">
      {/* Background Pattern & Orbs */}
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none" aria-hidden="true" />
      <div className="fixed inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
        <div className="absolute -top-40 -right-40 w-[600px] h-[600px] bg-orb-blue rounded-full blur-3xl animate-float" />
        <div className="absolute top-1/2 -left-40 w-[500px] h-[500px] bg-orb-violet rounded-full blur-3xl" style={{ animationDelay: '1s' }} />
        <div className="absolute -bottom-20 right-1/3 w-[400px] h-[400px] bg-orb-cyan rounded-full blur-3xl" />
      </div>

      {/* Navbar */}
      <header className="relative z-20 border-b border-slate-100 glass-card">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            
            <h1 className="text-xl font-bold text-gradient">
              AdForge AI
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-xs font-medium px-3 py-1.5 bg-slate-100 text-slate-500 rounded-full">Beta</span>
          </div>
        </div>
      </header>

      {/* Main Content Layout */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-start">
        
        {/* LEFT COLUMN: Controls (Inputs) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="space-y-1">
            <h2 className="text-2xl font-semibold text-slate-900 tracking-tight">Create your ad</h2>
            <p className="text-slate-500 text-sm">Upload an image, describe your vision, and let AI craft your creative.</p>
          </div>

          <div className="glass-card border border-slate-200/80 rounded-2xl p-6 shadow-soft-lg space-y-6">
            
            {/* 1. Image Upload Section */}
            <div className="relative">
              <div className="space-y-3">
                <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
                  <span className="flex items-center justify-center w-5 h-5 text-xs font-semibold bg-slate-100 text-slate-500 rounded-full">1</span>
                  Reference Image
                </label>
                
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  className={`group relative flex justify-center rounded-xl border-2 border-dashed px-6 py-8 transition-all cursor-pointer ${
                    imagePreviewUrl 
                      ? "border-green-300 bg-green-50/50 hover:border-green-400" 
                      : "border-slate-200 hover:border-blue-400 hover:bg-blue-50/50"
                  }`}
                >
                  <div className="text-center">
                    <div className={`mx-auto w-12 h-12 flex items-center justify-center rounded-full transition-colors ${
                      imagePreviewUrl 
                        ? "bg-green-100" 
                        : "bg-slate-100 group-hover:bg-blue-100"
                    }`}>
                      {imagePreviewUrl ? (
                        <ImageIcon className="w-6 h-6 text-green-500" />
                      ) : (
                        <UploadCloud className="w-6 h-6 text-slate-400 group-hover:text-blue-500 transition-colors" />
                      )}
                    </div>
                    <div className="mt-4 text-sm text-slate-600">
                      {imagePreviewUrl ? (
                        <>
                          <span className="font-semibold text-green-600">Image uploaded!</span>
                          <span className="text-slate-400"> Click to change</span>
                        </>
                      ) : (
                        <>
                          <span className="font-semibold text-blue-600 hover:text-blue-500">Click to upload</span>
                          <span className="text-slate-400"> or drag and drop</span>
                        </>
                      )}
                    </div>
                    <p className="mt-1 text-xs text-slate-400">PNG, JPG up to 10MB</p>
                  </div>
                </div>
                <input type="file" ref={fileInputRef} onChange={handleImageUpload} className="hidden" accept="image/png, image/jpeg" />
              </div>

              {/* Floating Image Preview Box */}
              {imagePreviewUrl && (
                <div className="absolute right-2 -top-2 z-10 transform rotate-3 transition-all duration-300 hover:rotate-0 hover:scale-105">
                  <div className="relative w-24 h-24 rounded-xl border-2 border-white shadow-lg bg-white p-1 group">
                    <img 
                      src={imagePreviewUrl} 
                      alt="Preview" 
                      className="w-full h-full object-contain rounded-lg bg-slate-50" 
                    />
                    <button 
                      onClick={(e) => { e.stopPropagation(); clearUpload(); }}
                      className="absolute -top-2 -right-2 p-1.5 bg-red-500 rounded-full text-white shadow-lg opacity-0 group-hover:opacity-100 transition-all hover:bg-red-600 hover:scale-110"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* 2. Text Prompt Section */}
            <div className="space-y-3">
              <label htmlFor="prompt" className="flex items-center gap-2 text-sm font-medium text-slate-700">
                <span className="flex items-center justify-center w-5 h-5 text-xs font-semibold bg-slate-100 text-slate-500 rounded-full">2</span>
                Describe Your Vision (Just anything in particular if you want)
              </label>
              <div className="relative">
                <textarea
                  id="prompt"
                  rows={4}
                  className="block w-full rounded-xl border border-slate-200 py-3.5 px-4 bg-white text-slate-900 shadow-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-400 text-sm leading-relaxed resize-none transition-all"
                  placeholder="A premium sneaker floating in clouds with golden hour lighting, minimalist luxury aesthetic..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
                <div className="absolute bottom-3 right-3 text-xs text-slate-400 bg-white/80 px-1.5 py-0.5 rounded">
                  {prompt.length}
                </div>
              </div>
            </div>

            {/* 3. Aspect Ratio Section */}
            <div className="space-y-3">
              <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
                <span className="flex items-center justify-center w-5 h-5 text-xs font-semibold bg-slate-100 text-slate-500 rounded-full">3</span>
                Aspect Ratio
              </label>
              <div className="grid grid-cols-2 gap-3">
                {ASPECT_RATIOS.map((ratio) => (
                  <button
                    key={ratio.value}
                    onClick={() => setSelectedRatio(ratio.value)}
                    className={`group flex items-center justify-center gap-2.5 px-4 py-3.5 text-sm font-medium rounded-xl border-2 transition-all duration-200 ${
                      selectedRatio === ratio.value
                        ? "border-blue-500 bg-blue-50 text-blue-600 shadow-sm"
                        : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50"
                    }`}
                  >
                    <span className={`transition-colors ${selectedRatio === ratio.value ? 'text-blue-500' : 'text-slate-400 group-hover:text-slate-500'}`}>
                      {ratio.icon}
                    </span>
                    {ratio.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={isLoading || !prompt}
              className={`group w-full flex items-center justify-center gap-2.5 rounded-xl py-4 px-6 text-sm font-semibold transition-all duration-300 ${
                isLoading || !prompt 
                  ? "bg-slate-100 text-slate-400 cursor-not-allowed"
                  : "bg-gradient-to-r from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-500/25 hover:shadow-xl hover:shadow-blue-500/30 hover:-translate-y-0.5 active:translate-y-0"
              }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Wand2 className="w-5 h-5 transition-transform group-hover:rotate-12" />
                  Generate Creative
                </>
              )}
            </button>
          </div>

          {/* Tips Section */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-slate-50 to-slate-100/50 border border-slate-200/80">
            <p className="text-xs text-slate-500 leading-relaxed">
              <span className="font-semibold text-slate-600">Pro tip:</span> Be specific with your prompts. Include details about lighting, style, mood, and composition for better results.
            </p>
          </div>
        </div>

        {/* RIGHT COLUMN: Output (Canvas) */}
        <div className="lg:col-span-7 lg:h-[700px] sticky top-24">
          <div className="h-full w-full rounded-2xl border border-slate-200 bg-gradient-to-br from-slate-50 to-white overflow-hidden relative flex items-center justify-center p-6 shadow-soft-lg">
            
            {/* State 1: Idle / Placeholder */}
            {!isLoading && !generatedImageUrl && (
              <div className="text-center space-y-5 select-none">
                <div className="inline-flex p-5 rounded-2xl bg-gradient-to-br from-slate-100 to-slate-50 border border-slate-200 shadow-soft animate-float">
                  <ImageIcon className="w-10 h-10 text-slate-400" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-lg font-semibold text-slate-700">Your canvas awaits</h3>
                  <p className="text-slate-400 text-sm max-w-xs mx-auto leading-relaxed">
                    Fill in the details on the left and hit generate to create your ad.
                  </p>
                </div>
              </div>
            )}

            {/* State 2: Loading */}
            {isLoading && (
              <div className="absolute inset-0 z-20 bg-white/80 backdrop-blur-md flex flex-col items-center justify-center">
                <div className="relative mb-6">
                  <div className="w-16 h-16 rounded-full border-4 border-slate-200 border-t-blue-500 animate-spin" />
                  <div className="absolute inset-0 rounded-full border-4 border-violet-400/20 animate-spin-slow blur-sm" />
                </div>
                <div className="text-center space-y-2">
                  <h3 className="text-xl font-semibold text-slate-800">Creating your ad</h3>
                  <p className="text-slate-500 text-sm animate-pulse-soft">This usually takes a few seconds...</p>
                </div>
                <div className="mt-6 flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"
                      style={{ animationDelay: `${i * 0.15}s` }}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* State 3: Result Display */}
            {generatedImageUrl && !isLoading && (
              <div className="relative w-full h-full flex flex-col items-center justify-center gap-4">
                <div className="relative group">
                  <div className="absolute -inset-4 bg-gradient-to-r from-blue-500/10 via-violet-500/10 to-blue-500/10 rounded-2xl blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  <img 
                    src={generatedImageUrl} 
                    alt="Generated Ad" 
                    className={`relative rounded-xl shadow-soft-lg object-contain max-h-[550px] w-auto border border-slate-200 transition-transform duration-300 group-hover:scale-[1.01]
                      ${selectedRatio === '1024x1024' ? 'aspect-square' : 'aspect-[1536/1024]'}
                    `}
                  />
                </div>
                
                {/* Action buttons */}
                <div className="flex items-center gap-3 mt-2">
                  <button 
                    onClick={handleGenerate}
                    className="flex items-center gap-2 px-4 py-2.5 text-sm font-medium text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 hover:border-slate-300 transition-all shadow-sm"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Regenerate
                  </button>
                  <a 
                    href={generatedImageUrl}
                    download="adforge-creative.png"
                    className="flex items-center gap-2 px-4 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-violet-600 rounded-lg hover:shadow-lg hover:shadow-blue-500/25 transition-all"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </a>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="relative z-10 border-t border-slate-100 mt-16">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 py-6 flex items-center justify-between">
          <p className="text-sm text-slate-400">© 2026 AdForge AI. All rights reserved.</p>
         
        </div>
      </footer>
    </main>
  );
}