const request = require('supertest');
const path = require('path');
const fs = require('fs');

// Mock the GeminiClient before requiring the server
const mockGenerateResponse = jest.fn();
jest.mock('../gemini-client', () => {
  return jest.fn().mockImplementation(() => ({
    generateResponse: mockGenerateResponse
  }));
});

const app = require('../server');

describe('Server API', () => {
  let server;

  beforeAll(() => {
    // Start server for testing
    server = app.listen(0); // Use port 0 to get any available port
  });

  afterAll((done) => {
    server.close(done);
  });

  describe('GET /', () => {
    it('should serve the main HTML page', async () => {
      const response = await request(app).get('/');
      expect(response.status).toBe(200);
      expect(response.type).toBe('text/html');
    });
  });

  describe('POST /api/conversation', () => {
    it('should create a new conversation', async () => {
      const response = await request(app)
        .post('/api/conversation')
        .expect(200);

      expect(response.body).toHaveProperty('conversationId');
      expect(typeof response.body.conversationId).toBe('string');
    });
  });

  describe('GET /api/conversation/:id', () => {
    let conversationId;

    beforeEach(async () => {
      const response = await request(app)
        .post('/api/conversation');
      conversationId = response.body.conversationId;
    });

    it('should get conversation by ID', async () => {
      const response = await request(app)
        .get(`/api/conversation/${conversationId}`)
        .expect(200);

      expect(response.body).toHaveProperty('id', conversationId);
      expect(response.body).toHaveProperty('messages');
      expect(response.body).toHaveProperty('createdAt');
      expect(Array.isArray(response.body.messages)).toBe(true);
    });

    it('should return 404 for non-existent conversation', async () => {
      const response = await request(app)
        .get('/api/conversation/non-existent-id')
        .expect(404);

      expect(response.body).toHaveProperty('error', 'Conversation not found');
    });
  });

  describe('POST /api/conversation/:id/message', () => {
    let conversationId;
    const GeminiClient = require('../gemini-client');

    beforeEach(async () => {
      const response = await request(app)
        .post('/api/conversation');
      conversationId = response.body.conversationId;
      
      // Reset the mock
      jest.clearAllMocks();
    });

    it('should send a text message', async () => {
      mockGenerateResponse.mockResolvedValue({
        text: 'Hello! How can I help you?',
        image: null
      });

      const response = await request(app)
        .post(`/api/conversation/${conversationId}/message`)
        .send({ message: 'Hello' })
        .expect(200);

      expect(response.body).toHaveProperty('userMessage');
      expect(response.body).toHaveProperty('assistantMessage');
      expect(response.body.userMessage.content).toBe('Hello');
      expect(response.body.assistantMessage.content).toBe('Hello! How can I help you?');
      expect(mockGenerateResponse).toHaveBeenCalled();
    });

    it('should return 404 for non-existent conversation', async () => {
      const response = await request(app)
        .post('/api/conversation/non-existent-id/message')
        .send({ message: 'Hello' })
        .expect(404);

      expect(response.body).toHaveProperty('error', 'Conversation not found');
    });

    it('should handle Gemini client errors', async () => {
      mockGenerateResponse.mockRejectedValue(new Error('API Error'));

      const response = await request(app)
        .post(`/api/conversation/${conversationId}/message`)
        .send({ message: 'Hello' })
        .expect(500);

      expect(response.body).toHaveProperty('error', 'Failed to process message');
    });
  });

  describe('DELETE /api/conversation/:id', () => {
    let conversationId;

    beforeEach(async () => {
      const response = await request(app)
        .post('/api/conversation');
      conversationId = response.body.conversationId;
    });

    it('should delete conversation by ID', async () => {
      const response = await request(app)
        .delete(`/api/conversation/${conversationId}`)
        .expect(200);

      expect(response.body).toHaveProperty('message', 'Conversation deleted');

      // Verify conversation is deleted
      await request(app)
        .get(`/api/conversation/${conversationId}`)
        .expect(404);
    });

    it('should return 404 for non-existent conversation', async () => {
      const response = await request(app)
        .delete('/api/conversation/non-existent-id')
        .expect(404);

      expect(response.body).toHaveProperty('error', 'Conversation not found');
    });
  });

  describe('GET /uploads/:filename', () => {
    it('should return 404 for non-existent image', async () => {
      await request(app)
        .get('/uploads/non-existent-image.jpg')
        .expect(404);
    });
  });
});