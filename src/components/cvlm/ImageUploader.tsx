"use client";

import { useCallback, useState } from "react";
import { Upload, Image as ImageIcon, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface UploadedImage {
  /** Image data URL (for <img> display). */
  dataUrl: string;
  /** Raw pixel data as a 3D array [H][W][C] where C is RGBA from canvas. */
  pixels: number[][][];
  width: number;
  height: number;
  /** Grayscale version [H][W]. */
  gray: number[][];
  /** Per-channel arrays [H][W], one per RGB channel. */
  channels: number[][][];
}

interface ImageUploaderProps {
  /** Called when image is loaded. */
  onImage: (img: UploadedImage) => void;
  /** Currently loaded image (for clearing). */
  image?: UploadedImage | null;
  /** Max dimension (default 64). Larger is slower for live math demos. */
  maxSize?: number;
  className?: string;
  label?: string;
}

/**
 * Image upload that reads pixels via <canvas> getImageData — fully client-side, no server.
 * Resizes to at most maxSize x maxSize preserving aspect ratio (educational demos use small images).
 */
export function ImageUploader({
  onImage,
  image,
  maxSize = 64,
  className,
  label = "Upload image",
}: ImageUploaderProps) {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const process = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        setError("Please upload a valid image file (PNG, JPG, etc.)");
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string;
        const imgEl = new window.Image();
        imgEl.onload = () => {
          const scale = Math.min(maxSize / imgEl.width, maxSize / imgEl.height, 1);
          const w = Math.max(1, Math.floor(imgEl.width * scale));
          const h = Math.max(1, Math.floor(imgEl.height * scale));
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            setError("Canvas not supported in this browser.");
            return;
          }
          ctx.drawImage(imgEl, 0, 0, w, h);
          const data = ctx.getImageData(0, 0, w, h).data;

          // Build [h][w][4] RGBA pixel array.
          const pixels: number[][][] = Array.from({ length: h }, () =>
            Array.from({ length: w }, () => [0, 0, 0, 0])
          );
          const channels: number[][][] = [
            Array.from({ length: h }, () => new Array(w).fill(0)),
            Array.from({ length: h }, () => new Array(w).fill(0)),
            Array.from({ length: h }, () => new Array(w).fill(0)),
          ];
          const gray: number[][] = Array.from({ length: h }, () => new Array(w).fill(0));

          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              const idx = (y * w + x) * 4;
              const r = data[idx];
              const g = data[idx + 1];
              const b = data[idx + 2];
              const a = data[idx + 3];
              pixels[y][x] = [r, g, b, a];
              channels[0][y][x] = r;
              channels[1][y][x] = g;
              channels[2][y][x] = b;
              gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
          }
          setError(null);
          onImage({ dataUrl, pixels, width: w, height: h, gray, channels });
        };
        imgEl.onerror = () => setError("Failed to load image.");
        imgEl.src = dataUrl;
      };
      reader.onerror = () => setError("Failed to read file.");
      reader.readAsDataURL(file);
    },
    [maxSize, onImage]
  );

  return (
    <div className={cn("space-y-2", className)}>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) process(f);
        }}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center text-sm transition-colors",
          dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-accent/30"
        )}
      >
        {image ? (
          <div className="flex items-center gap-3">
            <img src={image.dataUrl} alt="uploaded" className="h-16 w-16 rounded object-cover" />
            <div className="text-left text-xs">
              <p className="font-medium">{image.width} × {image.height} × 3</p>
              <p className="text-muted-foreground">RGB image</p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              type="button"
              onClick={(e) => {
                e.preventDefault();
                onImage(null as unknown as UploadedImage);
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <>
            <Upload className="h-6 w-6 text-muted-foreground" />
            <span className="font-medium">{label}</span>
            <span className="text-xs text-muted-foreground">Drag &amp; drop or click • resized to ≤ {maxSize}px</span>
          </>
        )}
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) process(f);
          }}
        />
      </label>
      {error && <p className="text-xs text-destructive">{error}</p>}
      {!image && (
        <Button
          variant="outline"
          size="sm"
          type="button"
          onClick={() => {
            // Synthesize a small sample image (gradient + circle) for demo without uploading.
            const w = 32, h = 32;
            const canvas = document.createElement("canvas");
            canvas.width = w; canvas.height = h;
            const ctx = canvas.getContext("2d")!;
            // Background gradient
            const grad = ctx.createLinearGradient(0, 0, w, h);
            grad.addColorStop(0, "#1e3a8a");
            grad.addColorStop(1, "#f59e0b");
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, w, h);
            // Draw a "circle"
            ctx.fillStyle = "#ef4444";
            ctx.beginPath();
            ctx.arc(w / 2, h / 2, Math.min(w, h) / 3, 0, Math.PI * 2);
            ctx.fill();
            const url = canvas.toDataURL();
            const imgEl = new window.Image();
            imgEl.onload = () => {
              const data = ctx.getImageData(0, 0, w, h).data;
              const pixels: number[][][] = Array.from({ length: h }, () =>
                Array.from({ length: w }, () => [0, 0, 0, 0])
              );
              const channels: number[][][] = [
                Array.from({ length: h }, () => new Array(w).fill(0)),
                Array.from({ length: h }, () => new Array(w).fill(0)),
                Array.from({ length: h }, () => new Array(w).fill(0)),
              ];
              const gray: number[][] = Array.from({ length: h }, () => new Array(w).fill(0));
              for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                pixels[y][x] = [r, g, b, 255];
                channels[0][y][x] = r;
                channels[1][y][x] = g;
                channels[2][y][x] = b;
                gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b;
              }
              onImage({ dataUrl: url, pixels, width: w, height: h, gray, channels });
            };
            imgEl.src = url;
          }}
        >
          <ImageIcon className="h-4 w-4 mr-1" /> Use sample image
        </Button>
      )}
    </div>
  );
}
