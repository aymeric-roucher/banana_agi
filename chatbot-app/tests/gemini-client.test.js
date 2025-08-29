const GeminiClient = require('../gemini-client');
const fs = require('fs');
const path = require('path');

// Mock the GoogleGenerativeAI module
jest.mock('@google/generative-ai', () => ({
  GoogleGenerativeAI: jest.fn().mockImplementation(() => ({
    getGenerativeModel: jest.fn().mockReturnValue({
      generateContent: jest.fn()
    })
  }))
}));

describe('GeminiClient', () => {
  let client;
  let mockModel;

  beforeEach(() => {
    process.env.GEMINI_API_KEY = 'test-api-key';
    client = new GeminiClient();
    mockModel = client.model;
    
    // Mock fs operations
    jest.spyOn(fs, 'existsSync').mockReturnValue(true);
    jest.spyOn(fs, 'readFileSync').mockReturnValue(Buffer.from('fake-image-data'));
    jest.spyOn(fs, 'writeFileSync').mockImplementation(() => {});
    jest.spyOn(fs, 'mkdirSync').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.clearAllMocks();
    delete process.env.GEMINI_API_KEY;
  });

  describe('constructor', () => {
    it('should throw error if API key is not provided', () => {
      delete process.env.GEMINI_API_KEY;
      expect(() => new GeminiClient()).toThrow('GEMINI_API_KEY not found in environment variables');
    });

    it('should initialize with API key', () => {
      expect(client.genAI).toBeDefined();
      expect(client.model).toBeDefined();
    });
  });

  describe('getMimeType', () => {
    it('should return correct MIME types', () => {
      expect(client.getMimeType('test.jpg')).toBe('image/jpeg');
      expect(client.getMimeType('test.png')).toBe('image/png');
      expect(client.getMimeType('test.gif')).toBe('image/gif');
      expect(client.getMimeType('test.webp')).toBe('image/webp');
      expect(client.getMimeType('test.unknown')).toBe('image/jpeg');
    });
  });

  describe('buildConversationContext', () => {
    it('should build context for single message', () => {
      const messages = [{ role: 'user', content: 'Hello' }];
      const context = client.buildConversationContext(messages);
      expect(context).toContain('You are a helpful AI assistant');
      expect(context).not.toContain('Previous conversation:');
    });

    it('should build context for multiple messages', () => {
      const messages = [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there!' },
        { role: 'user', content: 'How are you?' }
      ];
      const context = client.buildConversationContext(messages);
      expect(context).toContain('Previous conversation:');
      expect(context).toContain('User: Hello');
      expect(context).toContain('Assistant: Hi there!');
    });
  });

  describe('generateResponse', () => {
    it('should generate response for text-only message', async () => {
      const mockResponse = {
        response: {
          text: () => 'Hello! How can I help you?',
          candidates: [{ content: { parts: [] } }]
        }
      };
      mockModel.generateContent.mockResolvedValue(mockResponse);

      const messages = [{ role: 'user', content: 'Hello' }];
      const result = await client.generateResponse(messages);

      expect(result.text).toBe('Hello! How can I help you?');
      expect(result.image).toBeNull();
    });

    it('should handle image input', async () => {
      const mockResponse = {
        response: {
          text: () => 'I can see the image you shared!',
          candidates: [{ content: { parts: [] } }]
        }
      };
      mockModel.generateContent.mockResolvedValue(mockResponse);

      const messages = [{ role: 'user', content: 'What do you see?' }];
      const result = await client.generateResponse(messages, '/fake/path/image.jpg');

      expect(result.text).toBe('I can see the image you shared!');
      expect(mockModel.generateContent).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.stringContaining('What do you see?'),
          expect.objectContaining({
            inlineData: expect.objectContaining({
              mimeType: 'image/jpeg'
            })
          })
        ])
      );
    });

    it('should handle API errors gracefully', async () => {
      const error = new Error('API key invalid');
      mockModel.generateContent.mockRejectedValue(error);

      const messages = [{ role: 'user', content: 'Hello' }];
      
      await expect(client.generateResponse(messages)).rejects.toThrow(
        'Invalid API key. Please check your GEMINI_API_KEY environment variable.'
      );
    });

    it('should handle quota exceeded error', async () => {
      const error = new Error('quota exceeded');
      mockModel.generateContent.mockRejectedValue(error);

      const messages = [{ role: 'user', content: 'Hello' }];
      
      await expect(client.generateResponse(messages)).rejects.toThrow(
        'API quota exceeded. Please try again later.'
      );
    });

    it('should handle safety filter error', async () => {
      const error = new Error('safety filter triggered');
      mockModel.generateContent.mockRejectedValue(error);

      const messages = [{ role: 'user', content: 'Hello' }];
      
      await expect(client.generateResponse(messages)).rejects.toThrow(
        'Content was blocked by safety filters. Please try rephrasing your message.'
      );
    });
  });

  describe('fileToGenerativePart', () => {
    it('should convert file to generative part', async () => {
      const result = await client.fileToGenerativePart('/fake/path.jpg', 'image/jpeg');
      
      expect(result).toEqual({
        inlineData: {
          data: expect.any(String),
          mimeType: 'image/jpeg'
        }
      });
    });
  });
});