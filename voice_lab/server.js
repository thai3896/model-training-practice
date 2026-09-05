import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Setup directories
const DATASET_DIR = path.join(__dirname, 'dataset');
const WAVS_DIR = path.join(DATASET_DIR, 'wavs');
const METADATA_FILE = path.join(DATASET_DIR, 'metadata.csv');

if (!fs.existsSync(WAVS_DIR)) fs.mkdirSync(WAVS_DIR, { recursive: true });

// Configure Multer for audio uploads
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(express.static(path.join(__dirname, 'public')));
app.use('/dataset', express.static(DATASET_DIR));
app.use(express.json());

// API to get the script lines
app.get('/api/script', (req, res) => {
    try {
        const scriptData = fs.readFileSync(path.join(__dirname, 'recording_script.txt'), 'utf8');
        const lines = scriptData.split('\n')
            .map(line => line.trim())
            .filter(line => /^\d+\.\s+/.test(line));
        res.json({ lines });
    } catch (error) {
        res.status(500).json({ error: 'recording_script.txt not found' });
    }
});

// API to save the audio and update metadata
app.post('/api/save', upload.single('audio'), (req, res) => {
    const { index, text } = req.body;
    const file_name = `audio_${String(index).padStart(3, '0')}`;
    const wav_path = path.join(WAVS_DIR, `${file_name}.wav`);

    // 1. Save the audio file
    fs.writeFileSync(wav_path, req.file.buffer);

    // 2. Clean the text (remove leading numbers like "1. ")
    const cleanText = text.replace(/^\d+\.\s*/, '');

    // 3. Update metadata.csv
    let metadata = {};
    if (fs.existsSync(METADATA_FILE)) {
        const existingData = fs.readFileSync(METADATA_FILE, 'utf8');
        existingData.split('\n').forEach(line => {
            const parts = line.split('|');
            if (parts.length >= 3) metadata[parts[0]] = line.trim();
        });
    }

    metadata[file_name] = `${file_name}|${cleanText}|${cleanText}`;

    // Write back sorted alphabetically
    const sortedKeys = Object.keys(metadata).sort();
    const newCsvData = sortedKeys.map(k => metadata[k]).join('\n') + '\n';
    fs.writeFileSync(METADATA_FILE, newCsvData, 'utf8');

    res.json({ success: true, file: file_name });
});

const server = app.listen(0, () => {
    const freePort = server.address().port;
    console.log(`\n✅ Server is running!`);
    console.log(`🎙️  Open http://localhost:${freePort} in your browser to start recording.\n`);
});
