import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir, rename, access } from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

/** Encode the actual timestamped capture; this script does not operate or recreate the app. */
const root = process.cwd();
const framesPath = path.join(root, '.cache/walkthrough/frames.json');
const captionOverride = path.join(root, '.cache/walkthrough/captions.json');
const draftPath = path.join(root, 'docs/walkthrough-script.md');
const executable = path.join(root, '.cache/playwright/ffmpeg-1011/ffmpeg-win64.exe');
const outputDirectory = path.join(root, 'docs');
const outputPath = path.join(outputDirectory, 'walkthrough.webm');
const temporaryPath = path.join(outputDirectory, 'walkthrough.encoding.webm');
const FPS = 2;
const FRAME_COUNT = 240;
const DURATION_MS = 120_000;
const TICK_MS = 1000 / FPS;
const BAND_HEIGHT = 200;

const escapeXml = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[character]);
const clock = milliseconds => `${Math.floor(milliseconds / 60_000)}:${String(Math.floor(milliseconds / 1000) % 60).padStart(2, '0')}`;
const vttClock = milliseconds => {
  const hours = Math.floor(milliseconds / 3_600_000);
  const minutes = Math.floor(milliseconds / 60_000) % 60;
  const seconds = Math.floor(milliseconds / 1000) % 60;
  const remainder = Math.round(milliseconds % 1000);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(remainder).padStart(3, '0')}`;
};
const parseClock = value => {
  const [minutes, seconds] = value.trim().split(':').map(Number);
  if (!Number.isFinite(minutes) || !Number.isFinite(seconds) || seconds >= 60) throw new Error(`Invalid caption time ${value}`);
  return (minutes * 60 + seconds) * 1000;
};

function wrapCaption(text, maximumCharacters) {
  const lines = [];
  let line = '';
  for (const word of text.trim().split(/\s+/)) {
    if (word.length > maximumCharacters) throw new Error('A caption word is too long to display without clipping.');
    if (line && line.length + word.length + 1 > maximumCharacters) { lines.push(line); line = word; }
    else line += `${line ? ' ' : ''}${word}`;
  }
  if (line) lines.push(line);
  if (lines.length > 4) throw new Error(`Caption needs ${lines.length} lines. Shorten it to four lines or fewer.`);
  return lines;
}

async function readCaptions() {
  let captions;
  let source;
  try {
    captions = JSON.parse(await readFile(captionOverride, 'utf8'));
    source = '.cache/walkthrough/captions.json';
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const markdown = await readFile(draftPath, 'utf8');
    captions = markdown.split(/\r?\n/).flatMap(line => {
      const cells = line.split('|').map(cell => cell.trim());
      const times = /^(\d+:\d{2})\s*[–—-]\s*(\d+:\d{2})$/.exec(cells[1] ?? '');
      return times ? [{ startMs: parseClock(times[1]), endMs: parseClock(times[2]), text: cells[3] }] : [];
    });
    source = 'docs/walkthrough-script.md';
  }
  if (!Array.isArray(captions) || captions.length === 0) throw new Error('No walkthrough captions were found.');
  captions.sort((a, b) => a.startMs - b.startMs);
  for (const [index, caption] of captions.entries()) {
    if (!Number.isInteger(caption.startMs) || !Number.isInteger(caption.endMs) || caption.startMs < 0 ||
        caption.endMs > DURATION_MS || caption.endMs <= caption.startMs || typeof caption.text !== 'string' || !caption.text.trim()) {
      throw new Error(`Invalid caption ${index + 1}. Expected {startMs,endMs,text} within the two-minute recording.`);
    }
    if (index > 0 && caption.startMs < captions[index - 1].endMs) throw new Error('Caption intervals must not overlap.');
  }
  return { captions, source };
}

const frameManifestBytes = await readFile(framesPath);
const frames = JSON.parse(frameManifestBytes.toString('utf8'));
if (!Array.isArray(frames) || frames.length === 0) throw new Error('frames.json must contain at least one captured frame.');
for (const [index, frame] of frames.entries()) {
  if (typeof frame.path !== 'string' || !path.isAbsolute(frame.path) || !Number.isFinite(frame.atMs) || frame.atMs < 0) {
    throw new Error(`Invalid source frame ${index + 1}. Expected an absolute capture path and nonnegative atMs timestamp.`);
  }
  if (index > 0 && frame.atMs < frames[index - 1].atMs) throw new Error('Source frames must be in chronological order.');
}
if (frames[0].atMs >= DURATION_MS) throw new Error('The recording has no source frame within its first two minutes.');
const { captions, source: captionSource } = await readCaptions();
await access(executable);
const firstImage = await sharp(frames[0].path).metadata();
if (!['png', 'jpeg'].includes(firstImage.format) || !firstImage.width || !firstImage.height) throw new Error('Captured input must be a valid PNG or JPEG image.');
const width = Math.max(640, Math.floor(Math.min(firstImage.width, 1440) / 2) * 2);
const captureHeight = Math.max(2, Math.round(firstImage.height * width / firstImage.width / 2) * 2);
const height = captureHeight + BAND_HEIGHT;
const fontSize = width >= 1200 ? 26 : width >= 900 ? 23 : 20;
const maximumCharacters = Math.floor((width - 88) / (fontSize * 0.56));
for (const caption of captions) wrapCaption(caption.text, maximumCharacters);
await mkdir(outputDirectory, { recursive: true });

const ffmpeg = spawn(executable, [
  '-hide_banner', '-loglevel', 'warning', '-y',
  '-f', 'image2pipe', '-c:v', 'mjpeg', '-framerate', String(FPS), '-i', 'pipe:0',
  '-an', '-c:v', 'libvpx', '-b:v', '1400k', '-crf', '10', '-deadline', 'good', '-cpu-used', '4',
  '-threads', '2', '-g', '20', '-auto-alt-ref', '0', '-lag-in-frames', '0', '-pix_fmt', 'yuv420p', '-r', String(FPS),
  '-frames:v', String(FRAME_COUNT), '-t', '120', '-f', 'webm', temporaryPath,
], { cwd: root, windowsHide: true, stdio: ['pipe', 'ignore', 'pipe'] });
let stderr = '';
ffmpeg.stderr.on('data', chunk => { stderr = (stderr + chunk.toString()).slice(-65_536); });
const completed = new Promise((resolve, reject) => {
  ffmpeg.once('error', reject);
  ffmpeg.once('close', code => code === 0 ? resolve() : reject(new Error(`FFmpeg exited ${code}: ${stderr.trim()}`)));
});
// Register an early handler while writes are in flight, so a failed child cannot cause an unhandled rejection.
completed.catch(() => {});
ffmpeg.stdin.on('error', () => {});
let frameIndex = -1;
let cachedSourcePath = '';
let cachedCapture;
let blankTicks = 0;

try {
  for (let tick = 0; tick < FRAME_COUNT; tick++) {
    const atMs = tick * TICK_MS;
    while (frameIndex + 1 < frames.length && frames[frameIndex + 1].atMs <= atMs) frameIndex++;
    if (frameIndex < 0) {
      blankTicks++;
      cachedCapture ??= await sharp({ create: { width, height: captureHeight, channels: 3, background: '#eef1eb' } }).png().toBuffer();
    } else if (cachedSourcePath !== frames[frameIndex].path) {
      const image = sharp(frames[frameIndex].path);
      const metadata = await image.metadata();
      if (!['png', 'jpeg'].includes(metadata.format)) throw new Error(`Source frame ${frameIndex + 1} is not a supported screenshot image.`);
      cachedCapture = await image.resize(width, captureHeight, { fit: 'contain', background: '#eef1eb' }).flatten({ background: '#eef1eb' }).png().toBuffer();
      cachedSourcePath = frames[frameIndex].path;
    }
    const currentCaption = captions.find(caption => caption.startMs <= atMs && atMs < caption.endMs);
    const text = frameIndex < 0 ? 'Recording starts. Waiting for the first captured frame.' : currentCaption?.text ?? '';
    const lines = wrapCaption(text, maximumCharacters);
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${BAND_HEIGHT}">
      <rect width="100%" height="100%" fill="#15372d"/>
      <text x="44" y="29" fill="#aecbbb" font-family="Arial, sans-serif" font-size="12" letter-spacing="1.4">SILENT WALKTHROUGH · ACTUAL APP CAPTURE</text>
      <text x="${width - 44}" y="29" fill="#aecbbb" text-anchor="end" font-family="Arial, sans-serif" font-size="13">${clock(atMs)} / 2:00</text>
      ${lines.map((line, index) => `<text x="44" y="${66 + index * 34}" fill="#f8faf4" font-family="Arial, sans-serif" font-size="${fontSize}">${escapeXml(line)}</text>`).join('')}
    </svg>`;
    const composed = sharp(cachedCapture).extend({ bottom: BAND_HEIGHT, background: '#15372d' })
      .composite([{ input: Buffer.from(svg), left: 0, top: captureHeight }]);
    if (tick === 190) await composed.clone().png().toFile(path.join(outputDirectory, 'walkthrough-preview.png'));
    const jpeg = await composed.jpeg({ quality: 91, chromaSubsampling: '4:4:4' }).toBuffer();
    await new Promise((resolve, reject) => ffmpeg.stdin.write(jpeg, error => error ? reject(error) : resolve()));
    if ((tick + 1) % 40 === 0) console.log(`Encoded ${tick + 1}/${FRAME_COUNT} timestamp-sampled frames.`);
  }
  ffmpeg.stdin.end();
  await completed;
} catch (error) {
  ffmpeg.stdin.destroy();
  ffmpeg.kill();
  await completed.catch(() => {});
  throw error;
}

await rename(temporaryPath, outputPath);
const vtt = 'WEBVTT\n\n' + captions.map((caption, index) => `${index + 1}\n${vttClock(caption.startMs)} --> ${vttClock(caption.endMs)}\n${caption.text.replace(/-->/g, '→')}\n`).join('\n');
await writeFile(path.join(outputDirectory, 'walkthrough.vtt'), vtt);
await writeFile(path.join(outputDirectory, 'walkthrough.html'), `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inspection Desk — recorded walkthrough</title>
<style>body{margin:0;background:#f4f5ee;color:#203d31;font:16px/1.6 system-ui,sans-serif}main{max-width:1100px;margin:48px auto;padding:0 24px}small{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#597363}h1{font-size:clamp(28px,4vw,42px);line-height:1.15;letter-spacing:-.04em;margin:12px 0 18px}p{max-width:850px;color:#597363}video{display:block;width:100%;max-height:80vh;margin:28px 0 18px;background:#e4e9df;border-radius:12px;border:1px solid #ccd5c6;box-shadow:0 10px 40px #173a2710}a{color:#246341;text-underline-offset:4px}nav{display:flex;flex-wrap:wrap;gap:24px}.note{font-size:14px;margin-top:24px}</style></head>
<body><main><small>Inspection Desk · project walkthrough</small><h1>Two minutes inside the review workflow.</h1>
<p>A silent, captioned sequence of actual app captures, sampled at two frames per second from their recorded timestamps. Captions occupy a separate area below the app image. When there is no new capture, the most recent image is held; the original timeline is preserved.</p>
<video controls playsinline preload="metadata"><source src="walkthrough.webm" type="video/webm"><track kind="captions" src="walkthrough.vtt" srclang="en" label="English captions">Your browser does not support this video. <a href="walkthrough.webm">Download the WebM recording</a>.</video>
<nav aria-label="Recording downloads"><a href="walkthrough.webm" download>Download the two-minute video</a><a href="walkthrough.vtt" download>Download captions</a></nav>
<p class="note">There is no audio. Use the player to pause, seek or change playback speed. The recording shows the captured session, including any visible loading; it is not a measurement of cold-start performance on other devices.</p>
</main></body></html>\n`);
console.log(JSON.stringify({ output: 'docs/walkthrough.webm', frames: FRAME_COUNT, fps: FPS, durationSeconds: DURATION_MS / 1000,
  width, height, capturedSourceFrames: frames.length, firstCaptureAtMs: frames[0].atMs, lastCaptureAtMs: frames.at(-1).atMs,
  firstFrameWaitTicks: blankTicks, captionSource, sourceManifestSha256: createHash('sha256').update(frameManifestBytes).digest('hex') }, null, 2));
