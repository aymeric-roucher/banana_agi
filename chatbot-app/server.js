const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const cors = require('cors');
require('dotenv').config();

const GeminiClient = require('./gemini-client');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads/';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${uuidv4()}-${file.originalname}`;
    cb(null, uniqueName);
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb('Error: Images only!');
    }
  }
});

// Initialize Gemini client
const geminiClient = new GeminiClient();

// In-memory conversation storage (in production, use a database)
const conversations = new Map();

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start a new conversation
app.post('/api/conversation', (req, res) => {
  const conversationId = uuidv4();
  conversations.set(conversationId, {
    id: conversationId,
    messages: [],
    createdAt: new Date()
  });
  
  res.json({ conversationId });
});

// Get conversation history
app.get('/api/conversation/:id', (req, res) => {
  const conversation = conversations.get(req.params.id);
  if (!conversation) {
    return res.status(404).json({ error: 'Conversation not found' });
  }
  
  res.json(conversation);
});

// Send message with streaming
app.post('/api/conversation/:id/message/stream', upload.single('image'), async (req, res) => {
  try {
    const conversationId = req.params.id;
    const { message } = req.body;
    const imageFile = req.file;
    
    if (!conversations.has(conversationId)) {
      return res.status(404).json({ error: 'Conversation not found' });
    }
    
    const conversation = conversations.get(conversationId);
    
    // Add user message to conversation
    const userMessage = {
      id: uuidv4(),
      role: 'user',
      content: message,
      timestamp: new Date(),
      ...(imageFile && { image: imageFile.filename })
    };
    
    conversation.messages.push(userMessage);
    
    // Set up Server-Sent Events
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Cache-Control'
    });
    
    // Send user message first
    res.write(`data: ${JSON.stringify({ type: 'user_message', data: userMessage })}\n\n`);
    
    // Create assistant message placeholder
    const assistantMessageId = uuidv4();
    let assistantMessageText = '';
    let assistantImages = [];
    
    const assistantMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date()
    };
    
    // Send initial assistant message
    res.write(`data: ${JSON.stringify({ type: 'assistant_message_start', data: assistantMessage })}\n\n`);
    
    // Stream response from Gemini
    await geminiClient.generateStreamingResponse(
      conversation.messages,
      imageFile ? path.join(__dirname, 'uploads', imageFile.filename) : null,
      (chunk) => {
        if (chunk.type === 'text') {
          assistantMessageText = chunk.fullText;
          res.write(`data: ${JSON.stringify({ type: 'text_chunk', data: { id: assistantMessageId, content: chunk.content, fullText: chunk.fullText } })}\n\n`);
        } else if (chunk.type === 'image') {
          assistantImages = chunk.allImages;
          res.write(`data: ${JSON.stringify({ type: 'image_chunk', data: { id: assistantMessageId, image: chunk.content, allImages: chunk.allImages } })}\n\n`);
        }
      }
    );
    
    // Finalize the assistant message
    assistantMessage.content = assistantMessageText;
    if (assistantImages.length > 0) {
      assistantMessage.images = assistantImages;
    }
    
    conversation.messages.push(assistantMessage);
    
    // Send completion
    res.write(`data: ${JSON.stringify({ type: 'complete', data: assistantMessage })}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
    
  } catch (error) {
    console.error('Error processing streaming message:', error);
    res.write(`data: ${JSON.stringify({ type: 'error', data: { error: 'Failed to process message' } })}\n\n`);
    res.end();
  }
});

// Send message (non-streaming fallback)
app.post('/api/conversation/:id/message', upload.single('image'), async (req, res) => {
  try {
    const conversationId = req.params.id;
    const { message } = req.body;
    const imageFile = req.file;
    
    if (!conversations.has(conversationId)) {
      return res.status(404).json({ error: 'Conversation not found' });
    }
    
    const conversation = conversations.get(conversationId);
    
    // Add user message to conversation
    const userMessage = {
      id: uuidv4(),
      role: 'user',
      content: message,
      timestamp: new Date(),
      ...(imageFile && { image: imageFile.filename })
    };
    
    conversation.messages.push(userMessage);
    
    // Get response from Gemini
    const geminiResponse = await geminiClient.generateResponse(
      conversation.messages,
      imageFile ? path.join(__dirname, 'uploads', imageFile.filename) : null
    );
    
    // Add assistant response to conversation
    const assistantMessage = {
      id: uuidv4(),
      role: 'assistant',
      content: geminiResponse.text,
      timestamp: new Date(),
      ...(geminiResponse.images && geminiResponse.images.length > 0 && { images: geminiResponse.images })
    };
    
    conversation.messages.push(assistantMessage);
    
    res.json({
      userMessage,
      assistantMessage
    });
    
  } catch (error) {
    console.error('Error processing message:', error);
    res.status(500).json({ error: 'Failed to process message' });
  }
});

// Serve uploaded images
app.get('/uploads/:filename', (req, res) => {
  const filename = req.params.filename;
  const filepath = path.join(__dirname, 'uploads', filename);
  
  if (fs.existsSync(filepath)) {
    res.sendFile(filepath);
  } else {
    res.status(404).send('Image not found');
  }
});

// Delete conversation
app.delete('/api/conversation/:id', (req, res) => {
  const conversationId = req.params.id;
  if (conversations.has(conversationId)) {
    conversations.delete(conversationId);
    res.json({ message: 'Conversation deleted' });
  } else {
    res.status(404).json({ error: 'Conversation not found' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

module.exports = app;