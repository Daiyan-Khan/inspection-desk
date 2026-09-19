import { useEffect, useRef, useState } from 'react';
import { Expand, ImageOff, Layers2, Minus, Plus, RotateCcw } from 'lucide-react';

export interface ImageHeatmap {
  width: number;
  height: number;
  values: ArrayLike<number>;
  /** Fraction of the model input occupied by the original image, excluding letterbox padding. */
  imageRegion?: { x: number; y: number; width: number; height: number };
}

interface InspectionImageProps {
  imageUrl: string;
  name: string;
  heatmap?: ImageHeatmap;
  processing?: boolean;
  methodLabel?: string;
  width?: number;
  height?: number;
}

export function InspectionImage({ imageUrl, name, heatmap, processing, methodLabel = 'Anomaly map', width = 1, height = 1 }: InspectionImageProps) {
  const viewport = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ x: number; y: number; startX: number; startY: number } | null>(null);
  const [size, setSize] = useState({ width: 600, height: 500 });
  const [dimensions, setDimensions] = useState({ width, height });
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const [showMap, setShowMap] = useState(true);
  const [opacity, setOpacity] = useState(55);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  useEffect(() => {
    setLoaded(false);
    setFailed(false);
    setDimensions({ width, height });
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [imageUrl, width, height]);

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setSize({ width: entry.contentRect.width, height: entry.contentRect.height }));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = overlay.current;
    if (!canvas || !heatmap || heatmap.width < 1 || heatmap.height < 1) return;
    const raw = document.createElement('canvas');
    raw.width = heatmap.width;
    raw.height = heatmap.height;
    const context = raw.getContext('2d');
    const target = canvas.getContext('2d');
    if (!context || !target) return;
    const image = context.createImageData(raw.width, raw.height);
    for (let i = 0; i < raw.width * raw.height; i += 1) {
      const v = Math.max(0, Math.min(1, Number(heatmap.values[i]) || 0));
      const t = Math.max(0, (v - 0.1) / 0.9);
      image.data[i * 4] = 245;
      image.data[i * 4 + 1] = Math.round(190 - 145 * t);
      image.data[i * 4 + 2] = Math.round(62 - 30 * t);
      image.data[i * 4 + 3] = Math.round(255 * t);
    }
    context.putImageData(image, 0, 0);
    const region = heatmap.imageRegion ?? { x: 0, y: 0, width: 1, height: 1 };
    canvas.width = Math.max(1, Math.round(raw.width * region.width * 8));
    canvas.height = Math.max(1, Math.round(raw.height * region.height * 8));
    target.clearRect(0, 0, canvas.width, canvas.height);
    target.imageSmoothingEnabled = true;
    target.drawImage(raw, region.x * raw.width, region.y * raw.height, region.width * raw.width, region.height * raw.height, 0, 0, canvas.width, canvas.height);
  }, [heatmap]);

  const fitScale = Math.min((size.width - 48) / Math.max(1, dimensions.width), (size.height - 48) / Math.max(1, dimensions.height));
  const imageWidth = Math.max(1, dimensions.width * fitScale);
  const imageHeight = Math.max(1, dimensions.height * fitScale);
  const reset = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  return <section className="image-inspector" aria-label="Inspection image viewer">
    <div className="image-toolbar">
      <div className="segmented image-mode-switch" aria-label="Image layer">
        <button type="button" className={!showMap || !heatmap ? 'selected' : ''} onClick={() => setShowMap(false)}>Original</button>
        <button type="button" className={showMap && heatmap ? 'selected' : ''} onClick={() => setShowMap(true)} disabled={!heatmap}><Layers2 size={13} /> Anomaly map</button>
      </div>
      <div className="zoom-controls">
        <button type="button" className="icon-button" aria-label="Zoom out" disabled={zoom <= 1} onClick={() => { setZoom(value => Math.max(1, value - 0.25)); setPan({ x: 0, y: 0 }); }}><Minus size={15} /></button>
        <span className="mono zoom-value">{Math.round(zoom * 100)}%</span>
        <button type="button" className="icon-button" aria-label="Zoom in" disabled={zoom >= 3} onClick={() => setZoom(value => Math.min(3, value + 0.25))}><Plus size={15} /></button>
        <button type="button" className="icon-button fit-button" aria-label="Fit image to viewer" onClick={reset}><Expand size={15} /></button>
      </div>
    </div>
    <div className={`image-viewport ${zoom > 1 ? 'can-pan' : ''}`} ref={viewport}
      onPointerDown={event => {
        if (zoom <= 1) return;
        drag.current = { x: pan.x, y: pan.y, startX: event.clientX, startY: event.clientY };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={event => {
        if (!drag.current) return;
        const maxX = Math.max(0, (imageWidth * zoom - size.width) / 2 + 24);
        const maxY = Math.max(0, (imageHeight * zoom - size.height) / 2 + 24);
        setPan({ x: Math.max(-maxX, Math.min(maxX, drag.current.x + event.clientX - drag.current.startX)), y: Math.max(-maxY, Math.min(maxY, drag.current.y + event.clientY - drag.current.startY)) });
      }}
      onPointerUp={() => { drag.current = null; }}
      onPointerCancel={() => { drag.current = null; }}>
      <div className="viewer-corner top-left" /><div className="viewer-corner bottom-right" />
      {!loaded && !failed && <div className="viewer-loading"><span className="loading-ring" /><span>Opening inspection image</span></div>}
      {failed && <div className="viewer-error"><ImageOff size={27} /><h3>Image unavailable</h3><p>The source image could not be loaded.</p><button type="button" className="button secondary small" onClick={() => { setFailed(false); setLoaded(false); const img = viewport.current?.querySelector('img'); if (img) img.src = imageUrl; }}><RotateCcw size={13} /> Retry image</button></div>}
      <div className="image-plane" style={{ width: imageWidth, height: imageHeight, transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, opacity: loaded && !failed ? 1 : 0 }}>
        <img src={imageUrl} alt={`Manufacturing inspection sample ${name}`} draggable={false} onLoad={event => { setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight }); setLoaded(true); setFailed(false); }} onError={() => { setFailed(true); setLoaded(false); }} />
        <canvas ref={overlay} className="heatmap-overlay" aria-hidden="true" style={{ opacity: showMap && heatmap ? opacity / 100 : 0 }} />
      </div>
      <div className="viewer-caption"><span className="mono">{name}</span><span>{loaded ? `${dimensions.width} × ${dimensions.height} px` : 'SOURCE IMAGE'}</span></div>
      {processing && <div className="processing-chip"><span className="status-dot pulse" /> Analysing image</div>}
    </div>
    <div className="image-bottom-toolbar">
      {heatmap && showMap ? <><div className="heatmap-key"><span className="heatmap-gradient" /><span>Lower</span><span>Higher anomaly</span></div><label className="opacity-control">Opacity <input aria-label="Anomaly map opacity" type="range" min="10" max="90" value={opacity} onChange={event => setOpacity(Number(event.target.value))} /><span className="mono">{opacity}%</span></label></> : <><span className="viewer-hint">{zoom > 1 ? 'Drag to inspect the image · Fit to reset' : 'Original image · Zoom in to inspect details'}</span><span className="viewer-method">{methodLabel}</span></>}
    </div>
    {showMap && heatmap && <p className="map-disclaimer">The map highlights unusual appearance. It does not identify defect cause or severity.</p>}
  </section>;
}
