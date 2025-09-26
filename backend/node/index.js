// backend/node/index.js
const express = require('express');
const admin = require('firebase-admin');
const fetch = (...args) => import('node-fetch').then(({default: f}) => f(...args));
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Initialize Firebase Admin - prefers GOOGLE_APPLICATION_CREDENTIALS env var
if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
  admin.initializeApp({
    credential: admin.credential.applicationDefault()
  });
} else {
  // If you don't have service account locally, initialize without credential for verifying id tokens
  admin.initializeApp();
}

async function verifyToken(req, res, next) {
  const header = req.headers.authorization || '';
  if (!header.startsWith('Bearer ')) return res.status(401).json({error: 'No token'});
  const token = header.split('Bearer ')[1];
  try {
    const decoded = await admin.auth().verifyIdToken(token);
    req.uid = decoded.uid;
    next();
  } catch (e) {
    console.error('verifyToken error', e);
    res.status(401).json({error: 'Unauthorized'});
  }
}

app.post('/api/style', verifyToken, async (req, res) => {
  const { imageUrl, occasion } = req.body;
  try {
    const target = process.env.STYLING_API_URL || 'http://localhost:8001/recommend';
    const r = await fetch(target, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uid: req.uid, imageUrl, occasion })
    });
    const data = await r.json();
    return res.json(data);
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: 'ai-service-failure' });
  }
});

app.post('/api/fitness', verifyToken, async (req, res) => {
  try {
    const target = process.env.FITNESS_API_URL || 'http://localhost:8002/plan';
    const r = await fetch(target, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uid: req.uid, ...req.body })
    });
    const data = await r.json();
    return res.json({ plan: data });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: 'ai-service-failure' });
  }
});

const port = process.env.PORT || 5000;
app.listen(port, () => console.log(`Node gateway listening ${port}`));
